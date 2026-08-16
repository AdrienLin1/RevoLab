"""Unit tests for the hierarchical master/follower valve policy.

These tests never start Isaac Sim. They exercise:

* the 128-D tactile-latent contract between master and follower,
* the strict 159-D follower observation and everything it must NOT contain,
* the synchronized single-``env.step`` master/follower call chain,
* the 21 + 2 action split and name-based DOF resolution,
* the latched global curriculum and its checkpoint round-trip,
* master warm-start / hierarchical resume / legacy-PPO checkpoint separation.
"""

from __future__ import annotations

import ast
import copy
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn
import yaml
from omegaconf import OmegaConf

from BrainCo_DexHand.algo.hora.models.models import (
    ActorCritic,
    FollowerActorCritic,
    build_actor_critic_kwargs,
)
from BrainCo_DexHand.algo.hora.ppo.hierarchical_experience import (
    HierarchicalExperienceBuffer,
)
from BrainCo_DexHand.algo.hora.ppo.hierarchical_obs import (
    FOLLOWER_OBS_DIM,
    FOLLOWER_OBS_SLICES,
    FOLLOWER_OBS_SPEC,
    TACTILE_LATENT_DIM,
    build_follower_obs,
    follower_obs_from_env,
    validate_tactile_latent,
)
from BrainCo_DexHand.algo.hora.ppo.hierarchical_ppo import (
    CHECKPOINT_FORMAT,
    STAGE_FOLLOWER,
    STAGE_JOINT_FINETUNE,
    STAGE_MASTER,
    HierarchicalPPO,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SCREW_DIR = REPO_ROOT / "source/BrainCo_DexHand/BrainCo_DexHand/tasks/direct/hora_screw"
AGENT_DIR = SCREW_DIR / "agents"
XY_YAML = AGENT_DIR / "valvedriver_tactile_frame813_xy.yaml"
XY_SMOKE_YAML = AGENT_DIR / "valvedriver_tactile_frame813_xy_smoke.yaml"
FRAME813_YAML = AGENT_DIR / "valvedriver_tactile_frame813.yaml"
XY_ENV_PATH = SCREW_DIR / "revo3_hand_screw_tactile_xy_env.py"
XY_ENV_CFG_PATH = SCREW_DIR / "revo3_hand_screw_tactile_xy_env_cfg.py"
TRAIN_PATH = REPO_ROOT / "scripts/hora/train.py"
PLAY_PATH = REPO_ROOT / "scripts/hora/play.py"


def _load_module(path: Path, name: str):
    """Import a leaf module by path so the Isaac-importing package is skipped."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register before executing so ``@dataclass`` can resolve its own module.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


XY_STAGE = _load_module(SCREW_DIR / "xy_stage.py", "hierarchical_test_xy_stage")
TACTILE_LAYOUT = _load_module(
    REPO_ROOT / "source/BrainCo_DexHand/BrainCo_DexHand/tasks/tactile_layout.py",
    "hierarchical_test_tactile_layout",
)

# A three-finger task keeps the master small while preserving every contract
# under test (141-dim obs, 21-dim action, 128-dim tactile latent).
FINGERS = ("thumb", "index", "middle")
COUNTS = (31, 21, 21)
BASE_PRIV_DIM = 11
FRAME_DIM = 742
PRIV_INFO_DIM = BASE_PRIV_DIM + FRAME_DIM
MASTER_OBS_DIM = 141
MASTER_ACTION_DIM = 21
XY_ACTION_DIM = 2
ENV_ACTION_DIM = MASTER_ACTION_DIM + XY_ACTION_DIM


# ---------------------------------------------------------------------------
# fixtures / fakes
# ---------------------------------------------------------------------------


def _env_cfg():
    return SimpleNamespace(
        task="pytest_hierarchical_xy",
        tactile_layout="estimated_official",
        tactile_priv_offset=BASE_PRIV_DIM,
        tactile_active_finger_names=FINGERS,
        tactile_graph_sensor_counts=COUNTS,
        tactile_graph_total_nodes=sum(COUNTS),
        tactile_graph_finger_positions=tuple(
            tuple(
                tuple(float(value) for value in row)
                for row in TACTILE_LAYOUT.normalized_estimated_official_centers_xy(name)
            )
            for name in FINGERS
        ),
        teacher_tactile_frame_dim=FRAME_DIM,
        teacher_tactile_history_len=10,
        tactile_graph_common_channels=5,
        tactile_graph_force_channels=5,
        tactile_graph_context_channels=4,
        priv_info_dim=PRIV_INFO_DIM,
        observation_space=MASTER_OBS_DIM,
        action_space=ENV_ACTION_DIM,
        finger_action_space=MASTER_ACTION_DIM,
        xy_curriculum_ramp_steps=512,
    )


class _Box:
    def __init__(self, dim):
        self.shape = (dim,)
        self.low = -np.ones(dim, dtype=np.float32)
        self.high = np.ones(dim, dtype=np.float32)


class FakeHierarchicalEnv:
    """Minimal stand-in for the XY valve env with strict call accounting."""

    def __init__(self, num_envs=4, angular_velocity=0.0, seed=0):
        self.num_envs = num_envs
        self.cfg = _env_cfg()
        self.observation_space = _Box(MASTER_OBS_DIM)
        self.action_space = _Box(ENV_ACTION_DIM)
        self.common_step_counter = 0
        self.step_count = 0
        self.received_actions = []
        self.curriculum_calls = []
        self.workspace = 0.01
        self.action_scale = 0.002
        self.angular_velocity = float(angular_velocity)
        self._generator = torch.Generator().manual_seed(seed)
        self._obs = self._make_obs()

    # -- observation plumbing ------------------------------------------
    def _make_obs(self):
        rand = lambda *shape: torch.rand(  # noqa: E731 - compact fixture helper
            *shape, generator=self._generator
        )
        finger_positions = rand(self.num_envs, MASTER_ACTION_DIM) + 100.0
        finger_targets = rand(self.num_envs, MASTER_ACTION_DIM) + 200.0
        contacts = rand(self.num_envs, 5) + 300.0
        frame = torch.cat([finger_positions, finger_targets, contacts], dim=-1)
        # Every channel the follower must NOT see carries a large sentinel
        # offset, so a numeric leak is impossible to miss.
        return {
            "obs": frame.repeat(1, 3),
            "priv_info": rand(self.num_envs, PRIV_INFO_DIM) + 500.0,
            "tactile_hist": rand(self.num_envs, 10, FRAME_DIM) + 400.0,
            "proprio_hist": rand(self.num_envs, 30, 47),
            "xy_position": rand(self.num_envs, XY_ACTION_DIM) * 0.2,
            "xy_velocity": rand(self.num_envs, XY_ACTION_DIM) * 0.1,
            "xy_target": rand(self.num_envs, XY_ACTION_DIM) * 0.2,
            "previous_xy_action": rand(self.num_envs, XY_ACTION_DIM),
            "xy_workspace_margin": rand(self.num_envs, XY_ACTION_DIM),
            "xy_state": rand(self.num_envs, 10),
            "finger_position_sentinel": finger_positions,
            "finger_target_sentinel": finger_targets,
        }

    def reset(self):
        self._obs = self._make_obs()
        return self._obs

    def step(self, actions):
        assert actions.shape == (self.num_envs, ENV_ACTION_DIM), actions.shape
        self.step_count += 1
        self.received_actions.append(actions.detach().clone())
        self.common_step_counter += 1
        self._obs = self._make_obs()
        rewards = torch.ones(self.num_envs)
        dones = torch.zeros(self.num_envs, dtype=torch.uint8)
        infos = {
            "time_outs": torch.zeros(self.num_envs, dtype=torch.bool),
            "metrics/angular_velocity_per_env": torch.full(
                (self.num_envs,), self.angular_velocity
            ),
            "screw/angular_velocity": torch.tensor(self.angular_velocity),
        }
        return self._obs, rewards, dones, infos

    def set_xy_curriculum_progress(self, progress):
        self.curriculum_calls.append(float(progress))
        self.workspace = XY_STAGE.curriculum_value(0.01, 0.05, progress)
        self.action_scale = XY_STAGE.curriculum_value(0.002, 0.005, progress)
        return self.workspace, self.action_scale


def _full_config(tmp_path, *, num_actors=4, horizon=2, minibatch=8, follower_minibatch=8):
    raw = yaml.safe_load(XY_YAML.read_text(encoding="utf-8"))
    raw["ppo"]["num_actors"] = num_actors
    raw["ppo"]["horizon_length"] = horizon
    raw["ppo"]["minibatch_size"] = minibatch
    raw["ppo"]["mini_epochs"] = 1
    raw["ppo"]["priv_info_dim"] = PRIV_INFO_DIM
    raw["ppo"]["max_agent_steps"] = num_actors * horizon * 2
    raw["ppo"]["save_frequency"] = 0
    raw["follower"]["minibatch_size"] = follower_minibatch
    raw["follower"]["mini_epochs"] = 1
    raw["follower"]["actor_units"] = [32, 32]
    raw["follower"]["critic_units"] = [32, 32]
    raw["hierarchical"]["xy_curriculum_ramp_steps"] = 512
    return OmegaConf.create(
        {"rl_device": "cpu", "test": False, "seed": 0, "train": raw}
    )


def _make_agent(tmp_path, env=None, full_config=None):
    env = env or FakeHierarchicalEnv()
    config = full_config or _full_config(tmp_path)
    return HierarchicalPPO(env, str(tmp_path), config), env


# ---------------------------------------------------------------------------
# 1. tactile latent shape
# ---------------------------------------------------------------------------


def _master_kwargs(actions_num=MASTER_ACTION_DIM, yaml_path=XY_YAML):
    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    ppo = dict(raw["ppo"])
    ppo["priv_info_dim"] = PRIV_INFO_DIM
    return build_actor_critic_kwargs(
        raw["network"],
        ppo,
        actions_num,
        (MASTER_OBS_DIM,),
        47,
        False,
        env_cfg=_env_cfg(),
    )


def test_finger_attention_gru_master_returns_a_128_dim_latent_in_one_forward():
    master = ActorCritic(copy.deepcopy(_master_kwargs()))
    assert master.tactile_latent_output_dim == TACTILE_LATENT_DIM

    obs = {
        "obs": torch.randn(3, MASTER_OBS_DIM),
        "priv_info": torch.randn(3, PRIV_INFO_DIM),
        "tactile_hist": torch.randn(3, 10, FRAME_DIM),
    }
    encoder_calls = []
    handle = master.tactile_encoder.register_forward_hook(
        lambda _m, _i, output: encoder_calls.append(output.detach().clone())
    )
    try:
        result = master.act(obs)
    finally:
        handle.remove()

    assert result["tactile_latent"].shape == (3, TACTILE_LATENT_DIM)
    # The tactile encoder ran exactly once for this decision.
    assert len(encoder_calls) == 1
    torch.testing.assert_close(result["tactile_latent"], encoder_calls[0])
    # The latent is the encoder output itself, not a re-derived quantity.
    torch.testing.assert_close(
        result["tactile_latent"],
        master.tactile_encoder(obs["tactile_hist"]),
    )


def test_master_forward_also_exposes_the_same_latent_for_ppo_minibatches():
    master = ActorCritic(copy.deepcopy(_master_kwargs()))
    batch = {
        "obs": torch.randn(2, MASTER_OBS_DIM),
        "priv_info": torch.randn(2, PRIV_INFO_DIM),
        "tactile_hist": torch.randn(2, 10, FRAME_DIM),
        "prev_actions": torch.randn(2, MASTER_ACTION_DIM),
    }
    result = master(batch)
    assert result["tactile_latent"].shape == (2, TACTILE_LATENT_DIM)


def test_legacy_mlp_master_reports_no_tactile_latent():
    raw = yaml.safe_load(
        (AGENT_DIR / "Revo3HandScrewTactile.yaml").read_text(encoding="utf-8")
    )
    ppo = dict(raw["ppo"])
    ppo["priv_info_dim"] = PRIV_INFO_DIM
    kwargs = build_actor_critic_kwargs(
        raw["network"], ppo, MASTER_ACTION_DIM, (MASTER_OBS_DIM,), 47, False
    )
    master = ActorCritic(kwargs)
    assert master.tactile_latent_output_dim == 0
    result = master.act(
        {
            "obs": torch.randn(2, MASTER_OBS_DIM),
            "priv_info": torch.randn(2, PRIV_INFO_DIM),
        }
    )
    assert result["tactile_latent"] is None
    with pytest.raises(RuntimeError, match="128"):
        validate_tactile_latent(result["tactile_latent"])


@pytest.mark.parametrize("bad_width", (64, 127, 129, 160))
def test_follower_rejects_a_tactile_latent_that_is_not_128_dim(bad_width):
    with pytest.raises(RuntimeError, match=r"\[B, 128\]"):
        validate_tactile_latent(torch.zeros(4, bad_width))
    with pytest.raises(RuntimeError):
        build_follower_obs(
            executed_hand_action=torch.zeros(4, MASTER_ACTION_DIM),
            tactile_latent=torch.zeros(4, bad_width),
            xy_position=torch.zeros(4, 2),
            xy_velocity=torch.zeros(4, 2),
            xy_target=torch.zeros(4, 2),
            previous_xy_action=torch.zeros(4, 2),
            xy_workspace_margin=torch.zeros(4, 2),
        )


def test_hierarchical_ppo_refuses_a_master_without_a_128_dim_latent(tmp_path):
    config = _full_config(tmp_path)
    config.train.network.tactile_encoder.type = "mlp"
    with pytest.raises(ValueError, match="finger_attention_gru"):
        HierarchicalPPO(FakeHierarchicalEnv(), str(tmp_path), config)


# ---------------------------------------------------------------------------
# 2. synchronized master/follower timing around one env.step
# ---------------------------------------------------------------------------


def test_one_control_cycle_calls_each_policy_once_around_a_single_env_step(tmp_path):
    env = FakeHierarchicalEnv(num_envs=4)
    agent, _ = _make_agent(tmp_path, env=env)
    agent.current_stage = STAGE_FOLLOWER
    agent._apply_stage_freeze()
    agent.obs = env.reset()

    trace = []
    original_master_act = agent.master_act
    original_follower_act = agent.follower_act

    def traced_master_act(obs_dict):
        trace.append(("master", env.step_count))
        return original_master_act(obs_dict)

    def traced_follower_act(follower_obs, critic_priv):
        trace.append(("follower", env.step_count, follower_obs.detach().clone()))
        return original_follower_act(follower_obs, critic_priv)

    agent.master_act = traced_master_act
    agent.follower_act = traced_follower_act
    agent.horizon_length = 1
    agent.storage.transitions_per_env = 1
    agent.play_steps()

    # One rollout step + one bootstrap forward at the end.
    master_calls = [item for item in trace if item[0] == "master"]
    follower_calls = [item for item in trace if item[0] == "follower"]
    assert len(master_calls) == 2
    assert len(follower_calls) == 2
    assert env.step_count == 1

    # Cycle 0: master, then follower, then exactly one env.step.
    assert master_calls[0][1] == 0
    assert follower_calls[0][1] == 0, "env.step() ran between master and follower"
    assert trace[0][0] == "master" and trace[1][0] == "follower"

    executed_action = env.received_actions[0]
    assert executed_action.shape == (env.num_envs, ENV_ACTION_DIM)
    hand_block = executed_action[:, :MASTER_ACTION_DIM]
    xy_block = executed_action[:, MASTER_ACTION_DIM:]
    assert torch.all(hand_block.abs() <= 1.0 + 1.0e-6)
    assert torch.all(xy_block.abs() <= 1.0 + 1.0e-6)

    # The follower conditioned on exactly the clipped hand action that was sent.
    follower_obs = follower_calls[0][2]
    assert follower_obs.shape == (env.num_envs, FOLLOWER_OBS_DIM)
    torch.testing.assert_close(
        follower_obs[:, FOLLOWER_OBS_SLICES["executed_hand_action"]], hand_block
    )


def test_follower_conditions_on_the_current_step_tactile_latent_not_the_next(tmp_path):
    env = FakeHierarchicalEnv(num_envs=4)
    agent, _ = _make_agent(tmp_path, env=env)
    agent.current_stage = STAGE_FOLLOWER
    agent._apply_stage_freeze()
    agent.obs = env.reset()
    obs_t = agent.obs
    tactile_t = obs_t["tactile_hist"].clone()

    captured = {}
    original = agent.follower_act

    def traced(follower_obs, critic_priv):
        captured.setdefault("obs", follower_obs.detach().clone())
        return original(follower_obs, critic_priv)

    agent.follower_act = traced
    agent.horizon_length = 1
    agent.storage.transitions_per_env = 1
    agent.play_steps()

    tactile_next = agent.obs["tactile_hist"]
    assert not torch.allclose(tactile_t, tactile_next)
    latent_block = captured["obs"][:, FOLLOWER_OBS_SLICES["tactile_latent"]]
    with torch.no_grad():
        latent_t = agent.master.get_tactile_latent(tactile_t)
        latent_next = agent.master.get_tactile_latent(tactile_next)
    torch.testing.assert_close(latent_block, latent_t)
    assert not torch.allclose(latent_block, latent_next)


def test_stage0_never_samples_the_follower_and_sends_zero_xy(tmp_path):
    env = FakeHierarchicalEnv(num_envs=4)
    agent, _ = _make_agent(tmp_path, env=env)
    assert agent.current_stage == STAGE_MASTER
    agent.obs = env.reset()

    calls = []
    original = agent.follower_act
    agent.follower_act = lambda *a, **k: (calls.append(1), original(*a, **k))[1]
    agent.horizon_length = 1
    agent.storage.transitions_per_env = 1
    agent.play_steps()

    # Only the post-rollout bootstrap value uses the follower in Stage 0.
    assert len(calls) == 1
    xy_block = env.received_actions[0][:, MASTER_ACTION_DIM:]
    torch.testing.assert_close(xy_block, torch.zeros_like(xy_block))


# ---------------------------------------------------------------------------
# 3. action split and name-based DOF resolution
# ---------------------------------------------------------------------------


def test_action_split_separates_finger_and_xy_blocks():
    action = torch.arange(ENV_ACTION_DIM, dtype=torch.float32).repeat(5, 1)
    finger, xy = XY_STAGE.split_hierarchical_action(action, MASTER_ACTION_DIM)
    assert finger.shape == (5, MASTER_ACTION_DIM)
    assert xy.shape == (5, XY_ACTION_DIM)
    torch.testing.assert_close(finger, action[:, :MASTER_ACTION_DIM])
    torch.testing.assert_close(xy, action[:, MASTER_ACTION_DIM:])
    with pytest.raises(ValueError, match="23 channels"):
        XY_STAGE.split_hierarchical_action(torch.zeros(5, 22), MASTER_ACTION_DIM)


@pytest.mark.parametrize(
    "joint_names",
    (
        ("stage_x_joint", "stage_y_joint", "right_thumb_CMP_joint", "right_index_MCP_joint"),
        ("right_thumb_CMP_joint", "stage_y_joint", "right_index_MCP_joint", "stage_x_joint"),
        ("right_index_MCP_joint", "right_thumb_CMP_joint", "stage_x_joint", "stage_y_joint"),
    ),
)
def test_stage_dof_indices_come_from_joint_names_not_articulation_order(joint_names):
    indices = XY_STAGE.resolve_xy_dof_indices(joint_names)
    assert joint_names[indices[0]] == "stage_x_joint"
    assert joint_names[indices[1]] == "stage_y_joint"


def test_stage_dof_resolution_fails_loudly_when_a_joint_is_missing():
    with pytest.raises(ValueError, match="stage_y_joint"):
        XY_STAGE.resolve_xy_dof_indices(("stage_x_joint", "right_thumb_CMP_joint"))


def test_stage_joint_names_cannot_be_captured_by_the_finger_actuator_pattern():
    import re

    finger_pattern = re.compile("right_.*")
    for name in XY_STAGE.XY_STAGE_JOINT_NAMES:
        assert finger_pattern.fullmatch(name) is None
        assert re.compile(XY_STAGE.XY_STAGE_ACTUATOR_EXPR).fullmatch(name) is not None


def test_stage_pd_controller_is_effort_limited():
    target = torch.tensor([[0.05, -0.05]])
    position = torch.zeros(1, 2)
    velocity = torch.zeros(1, 2)
    effort = XY_STAGE.xy_pd_effort(
        target, position, velocity, pgain=2000.0, dgain=60.0, effort_limit=50.0
    )
    torch.testing.assert_close(effort, torch.tensor([[50.0, -50.0]]))

    # Inside the limit the controller is the ordinary PD law.
    effort = XY_STAGE.xy_pd_effort(
        torch.tensor([[0.001, 0.0]]),
        torch.zeros(1, 2),
        torch.tensor([[0.0, 0.1]]),
        pgain=2000.0,
        dgain=60.0,
        effort_limit=50.0,
    )
    torch.testing.assert_close(effort, torch.tensor([[2.0, -6.0]]))


def test_xy_target_respects_workspace_velocity_and_acceleration_limits():
    zeros = torch.zeros(1, 2)
    target, delta, smoothed = XY_STAGE.update_xy_target(
        torch.ones(1, 2),
        zeros,
        zeros,
        zeros,
        action_scale=0.005,
        workspace=0.05,
        velocity_limit=0.15,
        acceleration_limit=8.0,
        dt=0.05,
        smoothing=0.0,
    )
    torch.testing.assert_close(target, torch.full((1, 2), 0.005))
    torch.testing.assert_close(delta, torch.full((1, 2), 0.005))
    torch.testing.assert_close(smoothed, torch.ones(1, 2))

    # Velocity limit: 0.15 m/s * 0.05 s = 0.0075 m per control step.
    target, _delta, _smoothed = XY_STAGE.update_xy_target(
        torch.ones(1, 2),
        zeros,
        torch.full((1, 2), 0.02),
        zeros,
        action_scale=0.5,
        workspace=0.05,
        velocity_limit=0.15,
        acceleration_limit=1000.0,
        dt=0.05,
        smoothing=0.0,
    )
    torch.testing.assert_close(target, torch.full((1, 2), 0.0075))

    # Acceleration limit: |delta - prev_delta| <= a * dt^2 = 8 * 0.0025 = 0.02.
    target, _delta, _smoothed = XY_STAGE.update_xy_target(
        torch.ones(1, 2),
        zeros,
        torch.full((1, 2), -0.02),
        zeros,
        action_scale=0.5,
        workspace=0.5,
        velocity_limit=100.0,
        acceleration_limit=8.0,
        dt=0.05,
        smoothing=0.0,
    )
    torch.testing.assert_close(target, torch.zeros(1, 2))

    # Workspace clamp.
    target, delta, _smoothed = XY_STAGE.update_xy_target(
        torch.ones(1, 2),
        torch.full((1, 2), 0.0099),
        zeros,
        zeros,
        action_scale=0.005,
        workspace=0.01,
        velocity_limit=1.0,
        acceleration_limit=1000.0,
        dt=0.05,
        smoothing=0.0,
    )
    torch.testing.assert_close(target, torch.full((1, 2), 0.01))
    torch.testing.assert_close(delta, torch.full((1, 2), 0.0001))


def test_workspace_margin_is_bounded_continuous_and_per_axis():
    position = torch.tensor([[0.0, 0.01], [0.02, -0.02], [0.05, -0.09]])
    margin = XY_STAGE.xy_workspace_margin(position, 0.02)
    assert margin.shape == position.shape
    assert torch.all(margin >= 0.0) and torch.all(margin <= 1.0)
    torch.testing.assert_close(margin[0], torch.tensor([1.0, 0.5]))
    torch.testing.assert_close(margin[1], torch.tensor([0.0, 0.0]))
    torch.testing.assert_close(margin[2], torch.tensor([0.0, 0.0]))
    # Continuity: a small position change causes a small margin change.
    delta = (
        XY_STAGE.xy_workspace_margin(position + 1.0e-6, 0.02) - margin
    ).abs().max()
    assert float(delta) < 1.0e-3


def test_high_speed_reward_is_continuous_and_never_gated_at_0_8():
    omega = torch.linspace(-2.0, 10.0, 601)
    bonus = XY_STAGE.high_speed_reward(omega, clip_max=4.0, target=6.0, scale=3.0)
    assert float(bonus[omega <= 4.0].abs().max()) == 0.0
    assert float(bonus.max()) == pytest.approx(3.0)
    # No jump anywhere, in particular not at 0.8 rad/s.
    jumps = bonus.diff().abs()
    assert float(jumps.max()) < 0.1
    near_08 = (omega - 0.8).abs() < 0.05
    assert float(bonus[near_08].abs().max()) == 0.0


def test_xy_env_control_path_uses_named_indices_and_the_shared_helpers():
    """The env must route actions through the audited helpers, not raw slices."""
    tree = ast.parse(XY_ENV_PATH.read_text(encoding="utf-8"))

    def _body_source(node: ast.FunctionDef) -> str:
        """Unparse a function body without its docstring."""
        body = list(node.body)
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            body = body[1:]
        return "\n".join(ast.unparse(statement) for statement in body)

    functions = {
        node.name: _body_source(node)
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
    }
    pre_physics = functions["_pre_physics_step"]
    assert "split_hierarchical_action" in pre_physics
    assert "super()._pre_physics_step(finger_actions)" in pre_physics
    apply_action = functions["_apply_action"]
    assert "xy_pd_effort" in apply_action
    assert "joint_ids=self.xy_dof_indices" in apply_action
    assert "self.xy_dof_index_tensor" in apply_action
    # Finger torque/work penalties must stay finger-only: the XY env never
    # rewrites the base actuated-index reward slices.
    assert "actuated_dof_indices" not in functions["_compute_xy_stage_reward"]
    source = XY_ENV_PATH.read_text(encoding="utf-8")
    assert "write_root_pose_to_sim" not in source
    assert "write_root_state_to_sim" not in source


# ---------------------------------------------------------------------------
# 4. curriculum
# ---------------------------------------------------------------------------


def test_follower_stays_inactive_until_the_speed_ema_latches(tmp_path):
    agent, _env = _make_agent(tmp_path)
    agent.activation_speed_ema_beta = 0.0  # EMA == last rollout speed
    assert agent.activation_speed_threshold == pytest.approx(0.8)
    assert agent.activation_patience == 5

    for _ in range(20):
        agent._update_curriculum(0.79)
        assert agent.current_stage == STAGE_MASTER
        assert agent.follower_active is False
        assert agent.activation_patience_counter == 0

    for step in range(1, agent.activation_patience):
        agent._update_curriculum(0.81)
        assert agent.current_stage == STAGE_MASTER
        assert agent.activation_patience_counter == step

    # A single dip resets the consecutive-epoch counter.
    agent._update_curriculum(0.5)
    assert agent.activation_patience_counter == 0
    assert agent.current_stage == STAGE_MASTER

    for _ in range(agent.activation_patience):
        agent._update_curriculum(1.2)
    assert agent.current_stage == STAGE_FOLLOWER
    assert agent.follower_active is True
    assert agent.master_frozen is True
    assert agent.activation_agent_step == agent.agent_steps


def test_activation_happens_once_and_never_reverts(tmp_path):
    agent, _env = _make_agent(tmp_path)
    agent.activation_speed_ema_beta = 0.0
    for _ in range(agent.activation_patience):
        agent._update_curriculum(2.0)
    assert agent.current_stage == STAGE_FOLLOWER
    activation_step = agent.activation_agent_step
    stage_start = agent.stage_start_agent_step

    for speed in (-5.0, 0.0, 0.1, -1.0):
        agent.agent_steps += 1000
        agent._update_curriculum(speed)
        assert agent.current_stage == STAGE_FOLLOWER
        assert agent.activation_agent_step == activation_step
        assert agent.stage_start_agent_step == stage_start


def test_master_parameters_are_frozen_and_unchanged_in_stage1(tmp_path):
    env = FakeHierarchicalEnv(num_envs=4)
    agent, _ = _make_agent(tmp_path, env=env)
    agent.activation_speed_ema_beta = 0.0
    for _ in range(agent.activation_patience):
        agent._update_curriculum(2.0)
    assert agent.current_stage == STAGE_FOLLOWER
    assert all(not p.requires_grad for p in agent.master.parameters())

    before_master = copy.deepcopy(agent.master.state_dict())
    before_rms = copy.deepcopy(agent.master_running_mean_std.state_dict())
    before_follower = copy.deepcopy(agent.follower.state_dict())
    agent.obs = env.reset()
    stats = agent.train_epoch()

    assert stats["master_actor"] == []
    assert stats["follower_actor"] != []
    for key, value in agent.master.state_dict().items():
        torch.testing.assert_close(value, before_master[key])
    for key, value in agent.master_running_mean_std.state_dict().items():
        torch.testing.assert_close(value, before_rms[key])
    assert any(
        not torch.allclose(value, before_follower[key])
        for key, value in agent.follower.state_dict().items()
    )


def test_stage0_updates_the_master_and_never_the_follower(tmp_path):
    env = FakeHierarchicalEnv(num_envs=4)
    agent, _ = _make_agent(tmp_path, env=env)
    before_follower = copy.deepcopy(agent.follower.state_dict())
    before_master = copy.deepcopy(agent.master.state_dict())
    agent.obs = env.reset()
    stats = agent.train_epoch()

    assert stats["follower_actor"] == []
    assert stats["master_actor"] != []
    for key, value in agent.follower.state_dict().items():
        torch.testing.assert_close(value, before_follower[key])
    assert any(
        not torch.allclose(value, before_master[key])
        for key, value in agent.master.state_dict().items()
    )


def test_optional_joint_finetuning_unfreezes_the_master_but_not_the_encoder(tmp_path):
    config = _full_config(tmp_path)
    config.train.hierarchical.joint_finetune_enable = True
    config.train.hierarchical.follower_only_steps = 0
    env = FakeHierarchicalEnv(num_envs=4)
    agent, _ = _make_agent(tmp_path, env=env, full_config=config)
    agent.activation_speed_ema_beta = 0.0
    for _ in range(agent.activation_patience):
        agent._update_curriculum(2.0)
    assert agent.current_stage == STAGE_FOLLOWER
    assert agent.master_reference is not None

    agent.agent_steps += 1
    agent._update_curriculum(2.0)
    assert agent.current_stage == STAGE_JOINT_FINETUNE
    assert agent.master_frozen is False
    assert all(
        not p.requires_grad for p in agent.master.tactile_encoder.parameters()
    )
    assert any(p.requires_grad for p in agent.master.actor_mlp.parameters())
    assert agent.master.mu.weight.requires_grad

    agent.obs = env.reset()
    stats = agent.train_epoch()
    assert stats["master_actor"] != []
    assert stats["follower_actor"] != []
    assert stats["master_kl_reg"] != []
    # In Stage 2 the master learning rate is a small fraction of the follower's.
    assert agent.master_lr == pytest.approx(
        agent.follower_lr * agent.master_finetune_lr_scale
    )


def test_disabled_joint_finetuning_keeps_the_run_in_follower_only(tmp_path):
    agent, _env = _make_agent(tmp_path)
    assert agent.joint_finetune_enable is False
    agent.activation_speed_ema_beta = 0.0
    for _ in range(agent.activation_patience):
        agent._update_curriculum(2.0)
    for _ in range(10):
        agent.agent_steps += 10_000_000
        agent._update_curriculum(3.0)
    assert agent.current_stage == STAGE_FOLLOWER


def test_workspace_ramp_starts_at_activation_and_is_pushed_to_the_env(tmp_path):
    env = FakeHierarchicalEnv(num_envs=4)
    agent, _ = _make_agent(tmp_path, env=env)
    agent.agent_steps = 4_000
    assert agent.xy_curriculum_progress() == 0.0
    workspace, action_scale = agent._push_xy_curriculum()
    assert workspace == pytest.approx(0.01)
    assert action_scale == pytest.approx(0.002)

    agent.activation_speed_ema_beta = 0.0
    for _ in range(agent.activation_patience):
        agent._update_curriculum(2.0)
    assert agent.stage_start_agent_step == 4_000
    assert agent.xy_curriculum_progress() == 0.0

    agent.agent_steps = 4_000 + agent.xy_curriculum_ramp_steps // 2
    assert agent.xy_curriculum_progress() == pytest.approx(0.5)
    workspace, action_scale = agent._push_xy_curriculum()
    assert workspace == pytest.approx(0.03)
    assert action_scale == pytest.approx(0.0035)

    agent.agent_steps = 4_000 + 10 * agent.xy_curriculum_ramp_steps
    assert agent.xy_curriculum_progress() == pytest.approx(1.0)
    workspace, action_scale = agent._push_xy_curriculum()
    assert workspace == pytest.approx(0.05)
    assert action_scale == pytest.approx(0.005)


# ---------------------------------------------------------------------------
# 5. checkpoints
# ---------------------------------------------------------------------------


def _stage1_checkpoint(tmp_path, master):
    from BrainCo_DexHand.algo.hora.models.running_mean_std import RunningMeanStd

    running = RunningMeanStd((MASTER_OBS_DIM,))
    running.running_mean.fill_(0.25)
    value = RunningMeanStd((1,))
    value.running_mean.fill_(3.5)
    path = tmp_path / "stage1_best.pth"
    torch.save(
        {
            "model": master.state_dict(),
            "running_mean_std": running.state_dict(),
            "value_mean_std": value.state_dict(),
            "optimizer": {},
            "agent_steps": 12345,
            "epoch_num": 7,
            "best_rewards": 1.5,
            "last_lr": 5.0e-4,
        },
        path,
    )
    return path, running, value


def test_existing_21_dim_master_checkpoint_loads_strictly(tmp_path):
    reference_master = ActorCritic(copy.deepcopy(_master_kwargs()))
    with torch.no_grad():
        for parameter in reference_master.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.05)
    path, running, value = _stage1_checkpoint(tmp_path, reference_master)

    agent, _env = _make_agent(tmp_path)
    assert agent.master.mu.out_features == MASTER_ACTION_DIM
    agent.restore_master_checkpoint(str(path))

    for key, tensor in reference_master.state_dict().items():
        torch.testing.assert_close(agent.master.state_dict()[key], tensor)
    torch.testing.assert_close(
        agent.master_running_mean_std.running_mean, running.running_mean
    )
    torch.testing.assert_close(
        agent.master_value_mean_std.running_mean, value.running_mean
    )
    # A warm start is NOT a resume: counters and curriculum stay at zero.
    assert agent.agent_steps == 0
    assert agent.epoch_num == 0
    assert agent.current_stage == STAGE_MASTER


def test_master_state_dict_matches_the_stage1_ppo_teacher_contract():
    hierarchical_master = ActorCritic(copy.deepcopy(_master_kwargs()))
    stage1_master = ActorCritic(copy.deepcopy(_master_kwargs(yaml_path=FRAME813_YAML)))
    left = {k: tuple(v.shape) for k, v in hierarchical_master.state_dict().items()}
    right = {k: tuple(v.shape) for k, v in stage1_master.state_dict().items()}
    assert left == right
    hierarchical_master.load_state_dict(stage1_master.state_dict(), strict=True)


def test_stage1_checkpoint_is_rejected_by_the_hierarchical_resume_path(tmp_path):
    path, _running, _value = _stage1_checkpoint(
        tmp_path, ActorCritic(copy.deepcopy(_master_kwargs()))
    )
    agent, _env = _make_agent(tmp_path)
    with pytest.raises(RuntimeError, match="--master_checkpoint"):
        agent.restore_train(str(path))


def test_hierarchical_checkpoint_restores_the_complete_curriculum(tmp_path):
    env = FakeHierarchicalEnv(num_envs=4)
    agent, _ = _make_agent(tmp_path, env=env)
    agent.activation_speed_ema_beta = 0.0
    agent.obs = env.reset()
    agent.train_epoch()
    for _ in range(agent.activation_patience):
        agent._update_curriculum(2.5)
    assert agent.current_stage == STAGE_FOLLOWER
    agent.agent_steps += agent.xy_curriculum_ramp_steps // 4
    agent.obs = env.reset()
    agent.train_epoch()
    agent._update_curriculum(1.75)
    agent.best_rewards = 42.0
    agent.best_angular_velocity = 2.25
    agent.save(str(tmp_path / "hier"))

    restored, _restored_env = _make_agent(tmp_path / "restored")
    restored.restore_train(str(tmp_path / "hier.pth"))

    assert restored.current_stage == agent.current_stage
    assert restored.activation_speed_ema == pytest.approx(agent.activation_speed_ema)
    assert restored.activation_patience_counter == agent.activation_patience_counter
    assert restored.activation_agent_step == agent.activation_agent_step
    assert restored.stage_start_agent_step == agent.stage_start_agent_step
    assert restored.agent_steps == agent.agent_steps
    assert restored.epoch_num == agent.epoch_num
    assert restored.best_rewards == pytest.approx(agent.best_rewards)
    assert restored.best_angular_velocity == pytest.approx(agent.best_angular_velocity)
    assert restored.xy_curriculum_progress() == pytest.approx(
        agent.xy_curriculum_progress()
    )
    assert restored.master_frozen == agent.master_frozen
    for key, value in agent.master.state_dict().items():
        torch.testing.assert_close(restored.master.state_dict()[key], value)
    for key, value in agent.follower.state_dict().items():
        torch.testing.assert_close(restored.follower.state_dict()[key], value)
    for key, value in agent.follower_running_mean_std.state_dict().items():
        torch.testing.assert_close(
            restored.follower_running_mean_std.state_dict()[key], value
        )


def test_hierarchical_checkpoint_carries_its_own_format_marker(tmp_path):
    agent, _env = _make_agent(tmp_path)
    agent.save(str(tmp_path / "marker"))
    payload = torch.load(tmp_path / "marker.pth", map_location="cpu", weights_only=False)
    assert payload["format"] == CHECKPOINT_FORMAT
    for key in (
        "master",
        "follower",
        "master_optimizer",
        "follower_optimizer",
        "master_running_mean_std",
        "follower_running_mean_std",
        "master_value_mean_std",
        "follower_value_mean_std",
        "current_stage",
        "activation_speed_ema",
        "activation_patience_counter",
        "activation_agent_step",
        "stage_start_agent_step",
        "xy_curriculum_progress",
        "agent_steps",
        "epoch_num",
        "best_rewards",
        "best_angular_velocity",
    ):
        assert key in payload


def test_hierarchical_checkpoint_is_rejected_by_master_warm_start(tmp_path):
    agent, _env = _make_agent(tmp_path)
    agent.save(str(tmp_path / "hier"))
    with pytest.raises(RuntimeError, match="--checkpoint"):
        agent.restore_master_checkpoint(str(tmp_path / "hier.pth"))


def test_legacy_stage1_ppo_restore_path_is_unchanged():
    """The unmodified PPO resume contract must keep its exact key requirements."""
    ppo_source = (
        REPO_ROOT
        / "source/BrainCo_DexHand/BrainCo_DexHand/algo/hora/ppo/ppo.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(ppo_source)
    restore = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "restore_train"
    )
    body = ast.unparse(restore)
    for key in (
        "'model'",
        "'running_mean_std'",
        "'value_mean_std'",
        "'optimizer'",
        "'agent_steps'",
        "'epoch_num'",
        "'best_rewards'",
        "'last_lr'",
    ):
        assert key in body
    assert "hierarchical" not in ppo_source.lower()


# ---------------------------------------------------------------------------
# 6. observations
# ---------------------------------------------------------------------------


def test_follower_observation_layout_is_exactly_159_dims():
    assert FOLLOWER_OBS_DIM == 159
    assert FOLLOWER_OBS_SPEC == (
        ("executed_hand_action", 21),
        ("tactile_latent", 128),
        ("xy_position", 2),
        ("xy_velocity", 2),
        ("xy_target", 2),
        ("previous_xy_action", 2),
        ("xy_workspace_margin", 2),
    )
    assert sum(width for _name, width in FOLLOWER_OBS_SPEC) == 159

    blocks = {
        "executed_hand_action": torch.full((3, 21), 1.0),
        "tactile_latent": torch.full((3, 128), 2.0),
        "xy_position": torch.full((3, 2), 3.0),
        "xy_velocity": torch.full((3, 2), 4.0),
        "xy_target": torch.full((3, 2), 5.0),
        "previous_xy_action": torch.full((3, 2), 6.0),
        "xy_workspace_margin": torch.full((3, 2), 7.0),
    }
    follower_obs = build_follower_obs(**blocks)
    assert follower_obs.shape == (3, 159)
    for name, block in blocks.items():
        torch.testing.assert_close(follower_obs[:, FOLLOWER_OBS_SLICES[name]], block)


def test_follower_observation_excludes_every_forbidden_channel(tmp_path):
    env = FakeHierarchicalEnv(num_envs=4)
    agent, _ = _make_agent(tmp_path, env=env)
    obs = env.reset()
    master_result = agent.master_act(obs)
    executed_hand_action = torch.clamp(master_result["actions"], -1.0, 1.0)
    follower_obs = follower_obs_from_env(
        obs,
        executed_hand_action=executed_hand_action,
        tactile_latent=master_result["tactile_latent"].detach(),
    )
    assert follower_obs.shape == (4, FOLLOWER_OBS_DIM)

    # 1) Structural: the builder reads ONLY the five XY channels from the env.
    #    Feeding a dict that contains nothing else must still succeed, which is
    #    only possible if the master obs / priv_info / tactile history are never
    #    consulted.
    xy_only = {
        key: obs[key]
        for key in (
            "xy_position",
            "xy_velocity",
            "xy_target",
            "previous_xy_action",
            "xy_workspace_margin",
        )
    }
    torch.testing.assert_close(
        follower_obs_from_env(
            xy_only,
            executed_hand_action=executed_hand_action,
            tactile_latent=master_result["tactile_latent"].detach(),
        ),
        follower_obs,
    )

    # 2) Numeric: every forbidden channel carries a sentinel magnitude >= 100
    #    (finger joint positions +100, finger targets +200, contacts +300,
    #    tactile history +400, priv_info +500). The follower observation is a
    #    bounded quantity, so none of them can be hiding inside it.
    for forbidden in (
        obs["finger_position_sentinel"],
        obs["finger_target_sentinel"],
        obs["obs"],
        obs["priv_info"],
        obs["tactile_hist"],
    ):
        assert float(forbidden.abs().min()) >= 100.0
    assert float(follower_obs.abs().max()) < 10.0

    values = follower_obs.flatten()

    def _contains(tensor):
        return bool(
            (values.unsqueeze(0) - tensor.flatten().unsqueeze(1)).abs().min() < 1.0e-6
        )

    assert not _contains(obs["finger_position_sentinel"])
    assert not _contains(obs["finger_target_sentinel"])
    assert not _contains(obs["obs"])
    assert not _contains(obs["priv_info"])
    assert not _contains(obs["tactile_hist"])

    # 3) The 160-dim attended finger tokens and the master actor features are
    #    never referenced by the follower observation contract.
    node_inputs = agent.master.tactile_encoder.extract_node_inputs(obs["tactile_hist"])
    tokens = agent.master.tactile_encoder.apply_finger_attention(
        agent.master.tactile_encoder.encode_fingers(node_inputs)
    )
    assert tokens.flatten(start_dim=2).shape[-1] == len(FINGERS) * 32
    obs_module_tree = ast.parse(
        (
            REPO_ROOT
            / "source/BrainCo_DexHand/BrainCo_DexHand/algo/hora/ppo/hierarchical_obs.py"
        ).read_text(encoding="utf-8")
    )
    subscript_keys = {
        node.slice.value
        for node in ast.walk(obs_module_tree)
        if isinstance(node, ast.Subscript)
        and isinstance(node.slice, ast.Constant)
        and isinstance(node.slice.value, str)
    }
    # The follower observation builder indexes NOTHING but the five XY channels.
    assert subscript_keys == {
        "xy_position",
        "xy_velocity",
        "xy_target",
        "previous_xy_action",
        "xy_workspace_margin",
    }
    ppo_source = (
        REPO_ROOT
        / "source/BrainCo_DexHand/BrainCo_DexHand/algo/hora/ppo/hierarchical_ppo.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(ppo_source)
    play_steps = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "play_steps"
    )
    play_steps_source = ast.unparse(play_steps)
    assert "features" not in play_steps_source
    assert "follower_obs_from_env" in play_steps_source


def test_follower_actor_never_receives_the_privileged_critic_slice(tmp_path):
    agent, env = _make_agent(tmp_path)
    assert agent.follower_critic_priv_dim == 11
    assert agent.follower.actor_mlp.mlp[0].in_features == FOLLOWER_OBS_DIM
    assert agent.follower.critic_mlp.mlp[0].in_features == FOLLOWER_OBS_DIM + 11
    actor_parameters = {id(p) for p in agent.follower.actor_mlp.parameters()}
    actor_parameters |= {id(p) for p in agent.follower.mu.parameters()}
    critic_parameters = {id(p) for p in agent.follower.critic_mlp.parameters()}
    critic_parameters |= {id(p) for p in agent.follower.value.parameters()}
    assert actor_parameters.isdisjoint(critic_parameters)

    obs = env.reset()
    priv = obs["priv_info"][:, :11]
    follower_obs = torch.randn(env.num_envs, FOLLOWER_OBS_DIM)
    with torch.no_grad():
        baseline = agent.follower.act_inference(follower_obs)
        agent.follower.act(follower_obs, priv)
        changed = agent.follower.act_inference(follower_obs)
    torch.testing.assert_close(baseline, changed)
    # The deterministic actor output cannot depend on a privileged input at all.
    with pytest.raises(TypeError):
        agent.follower.act_inference(follower_obs, priv)


def test_master_observation_and_action_widths_are_unchanged(tmp_path):
    agent, env = _make_agent(tmp_path)
    assert agent.obs_shape == (141,)
    assert agent.master_action_dim == 21
    assert agent.master.mu.out_features == 21
    assert agent.env_action_dim == 23
    obs = env.reset()
    assert obs["obs"].shape[-1] == 141
    result = agent.master_act(obs)
    assert result["actions"].shape == (env.num_envs, 21)


def test_student_proprio_frame_stays_42_with_a_23_dim_action_space():
    """A 23-DOF robot must not widen the 42-dim student proprio frame."""
    source = (SCREW_DIR / "revo3_hand_screw_tactile_env_cfg.py").read_text(
        encoding="utf-8"
    )
    assert "finger_action_space" in source
    assert "2 * finger_action_space" in source

    cfg = SimpleNamespace(
        action_space=23,
        finger_action_space=21,
        student_proprio_command_dim=0,
    )
    finger_action_space = int(getattr(cfg, "finger_action_space", cfg.action_space))
    assert 2 * finger_action_space + cfg.student_proprio_command_dim == 42

    base_cfg = SimpleNamespace(action_space=21, student_proprio_command_dim=0)
    finger_action_space = int(
        getattr(base_cfg, "finger_action_space", base_cfg.action_space)
    )
    assert 2 * finger_action_space + base_cfg.student_proprio_command_dim == 42


# ---------------------------------------------------------------------------
# rollout storage / PPO mechanics
# ---------------------------------------------------------------------------


def test_rollout_stores_both_policies_and_the_joint_env_action(tmp_path):
    env = FakeHierarchicalEnv(num_envs=4)
    agent, _ = _make_agent(tmp_path, env=env)
    agent.current_stage = STAGE_FOLLOWER
    agent._apply_stage_freeze()
    agent.obs = env.reset()
    agent.play_steps()

    storage = agent.storage
    for key, expected in (
        ("obses", (agent.horizon_length, 4, 141)),
        ("priv_info", (agent.horizon_length, 4, PRIV_INFO_DIM)),
        ("tactile_hist", (agent.horizon_length, 4, 10, FRAME_DIM)),
        ("master_actions", (agent.horizon_length, 4, 21)),
        ("master_executed_actions", (agent.horizon_length, 4, 21)),
        ("follower_obses", (agent.horizon_length, 4, 159)),
        ("follower_actions", (agent.horizon_length, 4, 2)),
        ("follower_executed_actions", (agent.horizon_length, 4, 2)),
        ("env_actions", (agent.horizon_length, 4, 23)),
        ("rewards", (agent.horizon_length, 4, 1)),
        ("dones", (agent.horizon_length, 4)),
        ("timeouts", (agent.horizon_length, 4)),
    ):
        assert tuple(storage.storage_dict[key].shape) == expected, key

    executed = torch.cat(
        [
            storage.storage_dict["master_executed_actions"],
            storage.storage_dict["follower_executed_actions"],
        ],
        dim=-1,
    )
    torch.testing.assert_close(storage.storage_dict["env_actions"], executed)
    assert torch.all(storage.storage_dict["master_executed_actions"].abs() <= 1.0)
    assert torch.all(storage.storage_dict["follower_executed_actions"].abs() <= 1.0)


def test_master_and_follower_use_independent_ppo_ratios_and_advantages():
    buffer = HierarchicalExperienceBuffer(
        num_envs=2,
        horizon_length=2,
        device="cpu",
        master_obs_dim=141,
        master_action_dim=21,
        priv_info_dim=PRIV_INFO_DIM,
        tactile_hist_shape=(10, FRAME_DIM),
        follower_obs_dim=159,
        follower_action_dim=2,
        follower_critic_priv_dim=11,
        master_minibatch_size=4,
        follower_minibatch_size=2,
    )
    for step in range(2):
        buffer.update_data("master_rewards", step, torch.ones(2, 1))
        buffer.update_data("follower_rewards", step, torch.ones(2, 1))
        buffer.update_data("rewards", step, torch.ones(2, 1))
        buffer.update_data("master_values", step, torch.full((2, 1), 0.5))
        buffer.update_data("follower_values", step, torch.full((2, 1), -3.0))
    buffer.compute_returns(torch.zeros(2, 1), torch.zeros(2, 1), 0.99, 0.95)
    data = buffer.prepare_training(normalize_advantage=False)

    assert not torch.allclose(data["master_advantages"], data["follower_advantages"])
    assert data["master_advantages"].shape == (4,)
    assert data["follower_advantages"].shape == (4,)
    assert buffer.num_master_minibatches == 1
    assert buffer.num_follower_minibatches == 2
    master_batch = buffer.master_minibatch(0)
    follower_batch = buffer.follower_minibatch(0)
    assert "follower_obses" not in master_batch
    assert "obses" not in follower_batch
    assert "tactile_hist" not in follower_batch
    assert "priv_info" not in follower_batch


def test_follower_loss_never_backpropagates_into_the_master(tmp_path):
    env = FakeHierarchicalEnv(num_envs=4)
    agent, _ = _make_agent(tmp_path, env=env)
    agent.current_stage = STAGE_FOLLOWER
    agent._apply_stage_freeze()
    agent.obs = env.reset()
    agent.set_eval()
    agent.play_steps()
    agent.set_train()
    for parameter in agent.master.parameters():
        parameter.requires_grad_(True)
        parameter.grad = None
    agent.update_follower()
    assert all(parameter.grad is None for parameter in agent.master.parameters())


def test_each_policy_bootstraps_truncated_episodes_with_its_own_value():
    buffer = HierarchicalExperienceBuffer(
        num_envs=1,
        horizon_length=1,
        device="cpu",
        master_obs_dim=141,
        master_action_dim=21,
        priv_info_dim=PRIV_INFO_DIM,
        tactile_hist_shape=(10, FRAME_DIM),
        follower_obs_dim=159,
        follower_action_dim=2,
        follower_critic_priv_dim=11,
        master_minibatch_size=1,
        follower_minibatch_size=1,
    )
    assert "master_rewards" in buffer.storage_dict
    assert "follower_rewards" in buffer.storage_dict
    assert buffer.storage_dict["master_rewards"] is not buffer.storage_dict["rewards"]


# ---------------------------------------------------------------------------
# 7. backward compatibility and configuration
# ---------------------------------------------------------------------------


def test_original_valvedriver_tactile_teacher_still_builds_unchanged():
    master = ActorCritic(copy.deepcopy(_master_kwargs(yaml_path=FRAME813_YAML)))
    assert master.mu.out_features == 21
    assert master.actor_input_dim == 141 + BASE_PRIV_DIM + 128
    result = master.act(
        {
            "obs": torch.randn(2, 141),
            "priv_info": torch.randn(2, PRIV_INFO_DIM),
            "tactile_hist": torch.randn(2, 10, FRAME_DIM),
        }
    )
    for key in ("neglogpacs", "values", "actions", "mus", "sigmas", "features"):
        assert key in result
    assert result["actions"].shape == (2, 21)


def test_xy_yaml_matches_the_stage1_teacher_network_section():
    xy = yaml.safe_load(XY_YAML.read_text(encoding="utf-8"))
    base = yaml.safe_load(FRAME813_YAML.read_text(encoding="utf-8"))
    assert xy["network"] == base["network"]
    assert xy["tactile_layout"] == base["tactile_layout"]
    assert xy["algo"] == "HierarchicalPPO"
    assert xy["network"]["tactile_encoder"]["gru_hidden_dim"] == 128
    for key, value in base["ppo"].items():
        assert xy["ppo"][key] == value, key
    assert xy["hierarchical"]["activation_speed_threshold"] == 0.8
    assert xy["hierarchical"]["activation_patience"] == 5
    assert xy["hierarchical"]["joint_finetune_enable"] is False
    assert xy["follower"]["critic_priv_dim"] == 11


def test_smoke_yaml_exercises_every_curriculum_stage():
    smoke = yaml.safe_load(XY_SMOKE_YAML.read_text(encoding="utf-8"))
    assert smoke["algo"] == "HierarchicalPPO"
    assert smoke["hierarchical"]["joint_finetune_enable"] is True
    assert smoke["hierarchical"]["activation_patience"] == 1
    assert smoke["hierarchical"]["activation_speed_threshold"] < 0.0
    assert smoke["ppo"]["max_agent_steps"] <= 100_000
    assert smoke["network"]["tactile_encoder"]["gru_hidden_dim"] == 128


def test_train_entry_point_exposes_the_hierarchical_task_algo_and_flag():
    source = TRAIN_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    task_choices = None
    algo_choices = None
    flags = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
            continue
        name = ast.literal_eval(node.args[0])
        flags.add(name)
        choices = next(
            (kw.value for kw in node.keywords if kw.arg == "choices"), None
        )
        if name == "--task" and choices is not None:
            task_choices = tuple(ast.literal_eval(choices))
        if name == "--algo" and choices is not None:
            algo_choices = tuple(ast.literal_eval(choices))
    assert "valvedriver_tactile_xy" in task_choices
    assert "HierarchicalPPO" in algo_choices
    assert "--master_checkpoint" in flags
    # The original entry points stay available.
    assert {"PPO", "ProprioAdapt"}.issubset(set(algo_choices))
    assert {"valvedriver_tactile", "valvedriver"}.issubset(set(task_choices))
    assert "'HierarchicalPPO': HierarchicalPPO" in source


def test_play_entry_point_exposes_the_hierarchical_xy_task():
    source = PLAY_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    task_choices = None
    algo_choices = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
            continue
        name = ast.literal_eval(node.args[0])
        choices = next((kw.value for kw in node.keywords if kw.arg == "choices"), None)
        if name == "--task" and choices is not None:
            task_choices = tuple(ast.literal_eval(choices))
        if name == "--algo" and choices is not None:
            algo_choices = tuple(ast.literal_eval(choices))
    assert "valvedriver_tactile_xy" in task_choices
    assert {"auto", "HierarchicalPPO"}.issubset(set(algo_choices))
    # The XY task must reach its own env class and config, not the flat one.
    assert "'valvedriver_tactile_xy': Revo3HandVavleDriverTactileXYEnvCfg" in source
    assert "'valvedriver_tactile_xy': Revo3HandScrewTactileXYEnv" in source
    # The legacy actor frame holds finger joints only, never the two XY channels.
    assert "getattr(env_cfg, 'finger_action_space', env_cfg.action_space)" in source


def test_play_resolves_the_hierarchical_algo_from_the_task():
    args = SimpleNamespace(task="valvedriver_tactile_xy", algo="auto", checkpoint="hier_nn/best.pth")
    resolve = _compile_function(
        PLAY_PATH,
        "_resolve_algo",
        {
            "args": args,
            # The task tuple was renamed when the yaw variants were added; the
            # XY task's own resolution behaviour is unchanged.
            "_HIERARCHICAL_SCREW_TASKS": ("valvedriver_tactile_xy",),
            "_is_stage2_checkpoint": lambda path: path.endswith(".ckpt"),
        },
    )
    assert resolve() == "HierarchicalPPO"
    args.algo = "HierarchicalPPO"
    assert resolve() == "HierarchicalPPO"

    # A hierarchical checkpoint cannot be replayed as a flat Stage1 teacher.
    args.algo = "PPO"
    with pytest.raises(ValueError, match="requires"):
        resolve()

    # ... and the flat tasks keep their original auto-detection.
    args.task, args.algo = "valvedriver_tactile", "auto"
    assert resolve() == "PPO"
    args.checkpoint = "stage2_nn/model_best.ckpt"
    assert resolve() == "ProprioAdapt"
    args.algo = "HierarchicalPPO"
    with pytest.raises(ValueError, match="end-effector stage task"):
        resolve()


def test_play_hierarchical_action_matches_the_trainer_call_chain(tmp_path):
    """The play helper must reproduce ``HierarchicalPPO.test`` exactly."""
    agent, env = _make_agent(tmp_path)
    agent.set_eval()
    hierarchical_action = _compile_function(
        PLAY_PATH,
        "_hierarchical_action",
        {
            "torch": torch,
            "validate_tactile_latent": validate_tactile_latent,
            "follower_obs_from_env": follower_obs_from_env,
        },
    )

    obs_dict = env.reset()
    # Stage 0 parks the stage: the follower contributes exactly nothing.
    assert agent.current_stage == STAGE_MASTER
    action, executed_xy = hierarchical_action(agent, obs_dict)
    assert action.shape == (env.num_envs, ENV_ACTION_DIM)
    assert torch.equal(executed_xy, torch.zeros_like(executed_xy))
    assert torch.equal(action[:, MASTER_ACTION_DIM:], executed_xy)

    expected_hand, _latent = agent.master.act_inference_with_latent(
        {
            "obs": agent.master_running_mean_std(obs_dict["obs"]),
            "priv_info": obs_dict["priv_info"],
            "tactile_hist": obs_dict["tactile_hist"],
        }
    )
    assert torch.allclose(
        action[:, :MASTER_ACTION_DIM], torch.clamp(expected_hand, -1.0, 1.0)
    )

    # From Stage 1 the follower drives the stage inside the action bounds.
    agent.current_stage = STAGE_FOLLOWER
    action, executed_xy = hierarchical_action(agent, obs_dict)
    assert action.shape == (env.num_envs, ENV_ACTION_DIM)
    assert executed_xy.abs().max().item() <= 1.0
    assert not torch.equal(executed_xy, torch.zeros_like(executed_xy))
    # The env never sees a second step for the two policy halves.
    assert env.step_count == 0


def test_play_xy_statistics_report_physical_units():
    samples = _compile_function(PLAY_PATH, "_xy_stage_samples", {"torch": torch})
    env_cfg = SimpleNamespace(xy_position_obs_scale=0.05, xy_velocity_obs_scale=0.15)
    obs_dict = {
        # Normalized by the fixed 50 mm asset limit -> 20 mm of travel on X.
        "xy_position": torch.tensor([[0.4, 0.0]]),
        "xy_velocity": torch.tensor([[0.0, 1.0]]),
        "xy_target": torch.tensor([[0.5, 0.0]]),
        "xy_workspace_margin": torch.tensor([[0.8, 0.3]]),
    }
    result = samples(obs_dict, torch.tensor([[1.0, 0.2]]), env_cfg)
    assert result["offset_mm"] == pytest.approx(20.0)
    assert result["speed_mm_s"] == pytest.approx(150.0)
    assert result["tracking_error_mm"] == pytest.approx(5.0)
    # The reported margin is the worst axis, not the average of both.
    assert result["workspace_margin"] == pytest.approx(0.3)
    assert result["action_abs"] == pytest.approx(0.6)
    assert result["action_saturation"] == pytest.approx(0.5)


def test_xy_env_cfg_validates_the_stage_contract():
    cfg = SimpleNamespace(
        xy_joint_limit=0.05,
        xy_workspace_initial=0.01,
        xy_workspace_final=0.05,
        xy_action_scale_initial=0.002,
        xy_action_scale_final=0.005,
        xy_velocity_limit=0.15,
        xy_acceleration_limit=8.0,
        xy_effort_limit=50.0,
        xy_pgain=2000.0,
        xy_dgain=60.0,
        xy_action_smoothing=0.5,
        xy_curriculum_ramp_steps=20_000_000,
    )
    limits = XY_STAGE.validate_xy_stage_config(cfg)
    assert limits.workspace_final == pytest.approx(0.05)

    too_wide = copy.copy(cfg)
    too_wide.xy_workspace_final = 0.08
    with pytest.raises(ValueError, match="joint hard limit"):
        XY_STAGE.validate_xy_stage_config(too_wide)

    inverted = copy.copy(cfg)
    inverted.xy_workspace_initial = 0.06
    with pytest.raises(ValueError, match="xy_workspace_initial"):
        XY_STAGE.validate_xy_stage_config(inverted)


def test_high_speed_reward_config_validation():
    cfg = SimpleNamespace(
        high_speed_reward_enable=False,
        angvel_clip_max=4.0,
        high_speed_target=2.0,
        high_speed_reward_scale=3.0,
        high_speed_penalty_threshold=8.0,
    )
    XY_STAGE.validate_high_speed_reward_config(cfg)  # disabled: no validation
    cfg.high_speed_reward_enable = True
    with pytest.raises(ValueError, match="angvel_clip_max"):
        XY_STAGE.validate_high_speed_reward_config(cfg)
    cfg.high_speed_target = 6.0
    XY_STAGE.validate_high_speed_reward_config(cfg)
    cfg.high_speed_penalty_threshold = 5.0
    with pytest.raises(ValueError, match="high_speed_penalty_threshold"):
        XY_STAGE.validate_high_speed_reward_config(cfg)


def _xy_mixin_defaults() -> SimpleNamespace:
    """Read the shipped class-level XY defaults without importing Isaac Lab."""
    tree = ast.parse(XY_ENV_CFG_PATH.read_text(encoding="utf-8"))
    mixin = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "Revo3HandScrewTactileXYMixinCfg"
    )
    namespace = {
        "NUM_XY_DOFS": XY_STAGE.NUM_XY_DOFS,
        "XY_STAGE_JOINT_NAMES": XY_STAGE.XY_STAGE_JOINT_NAMES,
        "XY_STAGE_WORLD_AXES": XY_STAGE.XY_STAGE_WORLD_AXES,
    }
    values = {}
    for statement in mixin.body:
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            continue
        target = statement.targets[0]
        if not isinstance(target, ast.Name):
            continue
        values[target.id] = eval(  # noqa: S307 - fixed literals from our own source
            ast.unparse(statement.value), {"__builtins__": {}}, namespace
        )
    return SimpleNamespace(**values)


def test_shipped_xy_defaults_satisfy_the_stage_contract():
    defaults = _xy_mixin_defaults()
    limits = XY_STAGE.validate_xy_stage_config(defaults)
    assert limits.joint_limit == pytest.approx(0.05)
    assert limits.workspace_initial == pytest.approx(0.01)
    assert limits.workspace_final == pytest.approx(0.05)
    assert defaults.action_space == 23
    assert defaults.finger_action_space == 21
    assert defaults.xy_position_obs_scale == pytest.approx(limits.joint_limit)
    assert defaults.xy_velocity_obs_scale == pytest.approx(limits.velocity_limit)
    # Costs are non-zero but small, and are all costs (never bonuses).
    for name in (
        "xy_velocity_penalty_scale",
        "xy_acceleration_penalty_scale",
        "xy_jerk_penalty_scale",
        "xy_effort_penalty_scale",
        "xy_power_penalty_scale",
        "xy_boundary_penalty_scale",
    ):
        value = float(getattr(defaults, name))
        assert value < 0.0, name
        assert abs(value) <= 0.1, name
    # The fair-comparison run keeps the original reward exactly.
    assert defaults.high_speed_reward_enable is False
    setattr(defaults, "angvel_clip_max", 4.0)
    XY_STAGE.validate_high_speed_reward_config(defaults)
    defaults.high_speed_reward_enable = True
    XY_STAGE.validate_high_speed_reward_config(defaults)

    # One control step at 20 Hz cannot move further than the velocity limit.
    dt = 1.0 / 20.0
    assert limits.action_scale_final <= limits.velocity_limit * dt
    # A full command reversal stays inside the acceleration limit.
    assert 2.0 * limits.action_scale_final <= limits.acceleration_limit * dt * dt


def test_xy_env_cfg_source_declares_every_required_knob():
    source = XY_ENV_CFG_PATH.read_text(encoding="utf-8")
    for knob in (
        "xy_workspace_initial",
        "xy_workspace_final",
        "xy_action_scale_initial",
        "xy_action_scale_final",
        "xy_velocity_limit",
        "xy_acceleration_limit",
        "xy_effort_limit",
        "xy_pgain",
        "xy_dgain",
        "xy_action_smoothing",
        "xy_curriculum_ramp_steps",
        "xy_velocity_penalty_scale",
        "xy_acceleration_penalty_scale",
        "xy_jerk_penalty_scale",
        "xy_effort_penalty_scale",
        "xy_power_penalty_scale",
        "xy_boundary_penalty_scale",
        "high_speed_reward_enable",
        "high_speed_target",
        "high_speed_reward_scale",
        "high_speed_penalty_threshold",
    ):
        assert f"{knob} =" in source, knob
    assert "xy_workspace_final = 0.05" in source
    assert "xy_joint_limit = 0.05" in source


def test_hierarchical_ppo_writes_the_documented_tensorboard_tags(tmp_path):
    env = FakeHierarchicalEnv(num_envs=4)
    agent, _ = _make_agent(tmp_path, env=env)
    written = {}
    agent.writer.add_scalar = lambda tag, value, step: written.__setitem__(tag, value)
    agent.obs = env.reset()
    stats = agent.train_epoch()
    agent.write_stats(stats, 0.01, 0.002)
    for tag in (
        "curriculum/hierarchical_stage",
        "curriculum/activation_speed_ema",
        "curriculum/activation_patience_counter",
        "curriculum/xy_workspace",
        "curriculum/xy_action_scale",
        "hierarchical/master_frozen",
        "hierarchical/joint_finetune_enabled",
        "losses/master_actor",
        "losses/master_critic",
        "info/master_kl",
        "info/master_lr",
        "info/follower_lr",
    ):
        assert tag in written, tag


def _compile_function(path: Path, function_name: str, namespace):
    """Compile one top-level/method function without importing Isaac Sim."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    selected = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    ]
    assert len(selected) == 1, function_name
    module = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    compiled = dict(namespace)
    exec(compile(module, str(path), "exec"), compiled)
    return compiled[function_name]


def test_stage_usd_authoring_creates_two_world_aligned_prismatic_joints():
    """Run the real USD authoring code and inspect the resulting stage.

    Requires the USD python bindings (available inside Isaac Sim); skipped
    elsewhere. No physics is stepped - only the authored topology is checked.
    """
    pxr = pytest.importorskip("pxr", reason="USD python bindings are not available")
    from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics

    hand_usd = (
        REPO_ROOT / "assets/usd/tactile_dexscrew/revo3_right_tactile.usda"
    )
    if not hand_usd.exists():  # pragma: no cover - asset is part of the repo
        pytest.skip("tactile hand USD asset is missing")

    # Palm-down grasp orientation used by the valve task.
    root_rot = (0.37992820, 0.59636781, 0.59636781, -0.37992820)
    stage = Usd.Stage.CreateInMemory()
    hand_path = "/World/envs/env_0/hand"
    hand = UsdGeom.Xform.Define(stage, hand_path)
    hand.GetPrim().GetReferences().AddReference(str(hand_usd))
    # The referenced hand already carries translate/orient/scale ops.
    ops = {op.GetOpName(): op for op in hand.GetOrderedXformOps()}
    ops["xformOp:translate"].Set(Gf.Vec3d(0.0, 0.078, 0.195))
    ops["xformOp:orient"].Set(Gf.Quatd(root_rot[0], Gf.Vec3d(*root_rot[1:])))

    cfg = SimpleNamespace(
        robot_cfg=SimpleNamespace(init_state=SimpleNamespace(rot=root_rot)),
        xy_carriage_mass=0.5,
        xy_carriage_inertia=1.0e-3,
        xy_joint_limit=0.05,
        xy_effort_limit=50.0,
        xy_joint_velocity_limit_sim=1.0,
    )
    fake_env = SimpleNamespace(
        scene=SimpleNamespace(stage=stage, env_prim_paths=["/World/envs/env_0"]),
        cfg=cfg,
    )
    author = _compile_function(
        XY_ENV_PATH,
        "_author_robot_stage_overrides",
        {
            "Gf": Gf,
            "UsdGeom": UsdGeom,
            "UsdPhysics": UsdPhysics,
            "PhysxSchema": PhysxSchema,
            "XY_STAGE_CARRIAGE_BODY_NAME": XY_STAGE.XY_STAGE_CARRIAGE_BODY_NAME,
            "XY_STAGE_JOINT_NAMES": XY_STAGE.XY_STAGE_JOINT_NAMES,
            "XY_STAGE_WORLD_AXES": XY_STAGE.XY_STAGE_WORLD_AXES,
            "_BASE_LINK_NAME": "right_hand_base_link",
            "_WORLD_LINK_NAME": "world",
            "_BASE_FIXED_JOINT_NAME": "right_hand_base_joint",
            "_JOINTS_SCOPE_NAME": "joints",
        },
    )
    author(fake_env)

    # The rigid world -> hand weld is gone.
    base_joint = stage.GetPrimAtPath(f"{hand_path}/joints/right_hand_base_joint")
    assert not base_joint.IsValid() or not base_joint.IsActive()

    # The carriage is a gravity-free rigid body with explicit mass/inertia.
    carriage = stage.GetPrimAtPath(f"{hand_path}/stage_x_carriage")
    assert carriage.IsValid()
    assert carriage.HasAPI(UsdPhysics.RigidBodyAPI)
    mass_api = UsdPhysics.MassAPI(carriage)
    assert mass_api.GetMassAttr().Get() == pytest.approx(0.5)
    assert tuple(mass_api.GetDiagonalInertiaAttr().Get()) == pytest.approx(
        (1.0e-3, 1.0e-3, 1.0e-3)
    )
    assert PhysxSchema.PhysxRigidBodyAPI(carriage).GetDisableGravityAttr().Get() is True

    expected_chain = (
        ("stage_x_joint", "X", "world", "stage_x_carriage", Gf.Vec3d(1.0, 0.0, 0.0)),
        (
            "stage_y_joint",
            "Y",
            "stage_x_carriage",
            "right_hand_base_link",
            Gf.Vec3d(0.0, 1.0, 0.0),
        ),
    )
    for joint_name, axis, body0, body1, world_axis in expected_chain:
        joint_prim = stage.GetPrimAtPath(f"{hand_path}/joints/{joint_name}")
        assert joint_prim.IsValid(), joint_name
        joint = UsdPhysics.PrismaticJoint(joint_prim)
        assert joint.GetAxisAttr().Get() == axis
        assert joint.GetLowerLimitAttr().Get() == pytest.approx(-0.05)
        assert joint.GetUpperLimitAttr().Get() == pytest.approx(0.05)
        assert [str(p) for p in joint.GetBody0Rel().GetTargets()] == [
            f"{hand_path}/{body0}"
        ]
        assert [str(p) for p in joint.GetBody1Rel().GetTargets()] == [
            f"{hand_path}/{body1}"
        ]
        # A finite drive force is authored alongside the actuator effort limit.
        drive = UsdPhysics.DriveAPI(joint_prim, "linear")
        assert drive.GetMaxForceAttr().Get() == pytest.approx(50.0)
        assert drive.GetStiffnessAttr().Get() == pytest.approx(0.0)
        assert drive.GetDampingAttr().Get() == pytest.approx(0.0)

        # The joint frame cancels the palm-down root rotation, so the prismatic
        # axis points along the WORLD axis, not a hand-local one.
        body_rotation = Gf.Quatd(
            root_rot[0], Gf.Vec3d(root_rot[1], root_rot[2], root_rot[3])
        )
        local_rot = joint.GetLocalRot0Attr().Get()
        joint_rotation = body_rotation * Gf.Quatd(
            float(local_rot.GetReal()),
            Gf.Vec3d(*[float(v) for v in local_rot.GetImaginary()]),
        )
        local_axis = {"X": Gf.Vec3d(1, 0, 0), "Y": Gf.Vec3d(0, 1, 0)}[axis]
        rotated = Gf.Rotation(joint_rotation).TransformDir(local_axis)
        for actual, expected in zip(rotated, world_axis):
            assert float(actual) == pytest.approx(float(expected), abs=1.0e-5)
        # Both frames coincide at rest.
        assert joint.GetLocalRot0Attr().Get() == joint.GetLocalRot1Attr().Get()
        assert tuple(joint.GetLocalPos0Attr().Get()) == (0.0, 0.0, 0.0)
        assert tuple(joint.GetLocalPos1Attr().Get()) == (0.0, 0.0, 0.0)


def test_finger_effort_penalties_can_never_include_the_stage_joints():
    """Finger torque/work penalties index actuated_dof_indices, which is the
    21 finger joints resolved by name. The stage joints are not in that list, so
    XY effort is priced only by the separate xy_* cost terms."""
    cfg_source = (SCREW_DIR / "revo3_hand_screw_env_cfg.py").read_text(encoding="utf-8")
    tree = ast.parse(cfg_source)
    base = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Revo3HandScrewEnvCfg"
    )
    actuated = next(
        ast.literal_eval(statement.value)
        for statement in base.body
        if isinstance(statement, ast.Assign)
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == "actuated_joint_names"
    )
    assert len(actuated) == 21
    assert all(name.startswith("right_") for name in actuated)
    for stage_joint in XY_STAGE.XY_STAGE_JOINT_NAMES:
        assert stage_joint not in actuated

    env_source = (SCREW_DIR / "revo3_hand_screw_env.py").read_text(encoding="utf-8")
    env_tree = ast.parse(env_source)
    rewards = next(
        node
        for node in ast.walk(env_tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_get_rewards"
    )
    rewards_source = ast.unparse(rewards)
    assert "torque_penalty = (self.torques[:, self.actuated_dof_indices] ** 2)" in (
        rewards_source
    )
    assert "self.finger_dof_indices" not in rewards_source


def test_base_screw_env_logs_the_speed_distribution_metrics():
    source = (SCREW_DIR / "revo3_hand_screw_env.py").read_text(encoding="utf-8")
    assert 'self.extras["screw/angular_velocity_positive_mean"]' in source
    assert 'f"screw/fraction_above_{label}"' in source
    assert "(0.8, 1.0, 2.0, 4.0)" in source


def test_follower_model_shapes_and_gaussian_head():
    follower = FollowerActorCritic(
        obs_dim=FOLLOWER_OBS_DIM,
        actions_num=2,
        actor_units=(256, 128, 64),
        critic_units=(256, 128, 64),
        critic_priv_dim=11,
    )
    linears = [m for m in follower.actor_mlp.modules() if isinstance(m, nn.Linear)]
    assert [(m.in_features, m.out_features) for m in linears] == [
        (159, 256),
        (256, 128),
        (128, 64),
    ]
    assert follower.mu.out_features == 2
    assert follower.sigma.shape == (2,)
    result = follower.act(torch.randn(5, 159), torch.randn(5, 11))
    assert result["actions"].shape == (5, 2)
    assert result["values"].shape == (5, 1)
    assert result["neglogpacs"].shape == (5,)
    with pytest.raises(RuntimeError, match=r"\[B, 159\]"):
        follower.act(torch.randn(5, 141), torch.randn(5, 11))
