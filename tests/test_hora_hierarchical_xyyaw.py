"""Unit tests for the hierarchical XY+yaw and yaw-only valve policies.

These tests never start Isaac Sim (the USD authoring test runs only when the
``pxr`` bindings are importable). They exercise:

* the 24-channel ``21 + 2 + 1`` and 22-channel ``21 + 1`` action contracts,
* name-based stage DOF resolution in the fixed ``[x, y, yaw]`` order,
* the strict 164-D / 154-D follower observations and the untouched 159-D one,
* the radian yaw controller: smoothing, angular velocity / acceleration limits,
  workspace clamp, effort-limited PD torque, margin and boundary saturation,
* the single latched curriculum that unlocks translation AND yaw at one step
  and interpolates both from one shared dimensionless progress,
* Stage 0/1/2 freeze and update behaviour with a 3-D follower,
* checkpoint round-trips and the explicit refusal of a 2-D follower checkpoint,
* the train/play entry points, runtime metadata and the authored USD topology.
"""

from __future__ import annotations

import ast
import copy
import importlib.util
import math
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml
from omegaconf import OmegaConf

from BrainCo_DexHand.algo.hora.models.models import (
    ActorCritic,
    FollowerActorCritic,
    build_actor_critic_kwargs,
)
from BrainCo_DexHand.algo.hora.ppo.hierarchical_obs import (
    FOLLOWER_OBS_DIM,
    STAGE_OBS_BLOCKS,
    build_stage_follower_obs,
    follower_obs_dim,
    follower_obs_slices,
    follower_obs_spec,
    stage_follower_obs_from_env,
    validate_stage_dofs,
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
XYYAW_YAML = AGENT_DIR / "valvedriver_tactile_frame813_xyyaw.yaml"
YAW_YAML = AGENT_DIR / "valvedriver_tactile_frame813_yaw.yaml"
XYYAW_SMOKE_YAML = AGENT_DIR / "valvedriver_tactile_frame813_xyyaw_smoke.yaml"
FRAME813_YAML = AGENT_DIR / "valvedriver_tactile_frame813.yaml"
YAW_ENV_PATH = SCREW_DIR / "revo3_hand_screw_tactile_yaw_env.py"
YAW_ENV_CFG_PATH = SCREW_DIR / "revo3_hand_screw_tactile_yaw_env_cfg.py"
XYYAW_ENV_PATH = SCREW_DIR / "revo3_hand_screw_tactile_xyyaw_env.py"
XYYAW_ENV_CFG_PATH = SCREW_DIR / "revo3_hand_screw_tactile_xyyaw_env_cfg.py"
XY_ENV_PATH = SCREW_DIR / "revo3_hand_screw_tactile_xy_env.py"
TRAIN_PATH = REPO_ROOT / "scripts/hora/train.py"
PLAY_PATH = REPO_ROOT / "scripts/hora/play.py"

_PACKAGE_NAME = "hierarchical_xyyaw_test_stage_pkg"


def _load_stage_modules():
    """Import ``xy_stage`` and ``yaw_stage`` without the Isaac-importing package.

    ``yaw_stage`` imports ``xy_stage`` relatively, so both are loaded into a
    synthetic package whose ``__path__`` points at the task directory.
    """
    if _PACKAGE_NAME not in sys.modules:
        package = types.ModuleType(_PACKAGE_NAME)
        package.__path__ = [str(SCREW_DIR)]
        sys.modules[_PACKAGE_NAME] = package
    loaded = []
    for name in ("xy_stage", "yaw_stage"):
        full_name = f"{_PACKAGE_NAME}.{name}"
        if full_name not in sys.modules:
            spec = importlib.util.spec_from_file_location(full_name, SCREW_DIR / f"{name}.py")
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            sys.modules[full_name] = module
            spec.loader.exec_module(module)
        loaded.append(sys.modules[full_name])
    return loaded


XY_STAGE, YAW_STAGE = _load_stage_modules()
TACTILE_LAYOUT_SPEC = importlib.util.spec_from_file_location(
    "hierarchical_xyyaw_test_tactile_layout",
    REPO_ROOT / "source/BrainCo_DexHand/BrainCo_DexHand/tasks/tactile_layout.py",
)
TACTILE_LAYOUT = importlib.util.module_from_spec(TACTILE_LAYOUT_SPEC)
sys.modules[TACTILE_LAYOUT_SPEC.name] = TACTILE_LAYOUT
TACTILE_LAYOUT_SPEC.loader.exec_module(TACTILE_LAYOUT)

# Same compact three-finger master as the XY suite: it preserves every contract
# under test (141-dim obs, 21-dim action, 128-dim tactile latent).
FINGERS = ("thumb", "index", "middle")
COUNTS = (31, 21, 21)
BASE_PRIV_DIM = 11
FRAME_DIM = 742
PRIV_INFO_DIM = BASE_PRIV_DIM + FRAME_DIM
MASTER_OBS_DIM = 141
MASTER_ACTION_DIM = 21
XYYAW_ACTION_DIM = 3
YAW_ACTION_DIM = 1
XYYAW_ENV_ACTION_DIM = MASTER_ACTION_DIM + XYYAW_ACTION_DIM  # 24
YAW_ENV_ACTION_DIM = MASTER_ACTION_DIM + YAW_ACTION_DIM  # 22
XYYAW_FOLLOWER_OBS_DIM = 164
YAW_FOLLOWER_OBS_DIM = 154

STAGE_JOINT_ORDER = ("stage_x_joint", "stage_y_joint", "stage_yaw_joint")


# ---------------------------------------------------------------------------
# fixtures / fakes
# ---------------------------------------------------------------------------


def _env_cfg(stage_dofs: int):
    joint_names = YAW_STAGE.stage_joint_names(stage_dofs)
    return SimpleNamespace(
        task="pytest_hierarchical_xyyaw",
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
        action_space=MASTER_ACTION_DIM + stage_dofs,
        finger_action_space=MASTER_ACTION_DIM,
        stage_joint_names=joint_names,
        stage_world_axes=YAW_STAGE.STAGE_WORLD_AXES_BY_DOF[stage_dofs],
        xy_curriculum_ramp_steps=512,
    )


class _Box:
    def __init__(self, dim):
        self.shape = (dim,)
        self.low = -np.ones(dim, dtype=np.float32)
        self.high = np.ones(dim, dtype=np.float32)


class FakeStageEnv:
    """Minimal stand-in for a 1-/3-DOF stage valve env with call accounting."""

    def __init__(self, stage_dofs=XYYAW_ACTION_DIM, num_envs=4, angular_velocity=0.0, seed=0):
        self.stage_dofs = int(stage_dofs)
        self.num_envs = num_envs
        self.cfg = _env_cfg(self.stage_dofs)
        self.observation_space = _Box(MASTER_OBS_DIM)
        self.action_space = _Box(MASTER_ACTION_DIM + self.stage_dofs)
        self.common_step_counter = 0
        self.step_count = 0
        self.received_actions = []
        self.curriculum_calls = []
        self.lock_calls = []
        self.stage_follower_active = False
        self.angular_velocity = float(angular_velocity)
        self._generator = torch.Generator().manual_seed(seed)
        # Metres for XY, radians for yaw: kept in separate entries, never in one
        # shared scale.
        self.curriculum = {
            "xy_workspace": 0.01,
            "xy_action_scale": 0.002,
            "yaw_workspace": 0.05,
            "yaw_action_scale": 0.005,
        }
        self._obs = self._make_obs()

    def _make_obs(self):
        rand = lambda *shape: torch.rand(  # noqa: E731 - compact fixture helper
            *shape, generator=self._generator
        )
        finger_positions = rand(self.num_envs, MASTER_ACTION_DIM) + 100.0
        finger_targets = rand(self.num_envs, MASTER_ACTION_DIM) + 200.0
        contacts = rand(self.num_envs, 5) + 300.0
        frame = torch.cat([finger_positions, finger_targets, contacts], dim=-1)
        obs = {
            "obs": frame.repeat(1, 3),
            "priv_info": rand(self.num_envs, PRIV_INFO_DIM) + 500.0,
            "tactile_hist": rand(self.num_envs, 10, FRAME_DIM) + 400.0,
            "finger_position_sentinel": finger_positions,
            "finger_target_sentinel": finger_targets,
        }
        for name in STAGE_OBS_BLOCKS:
            obs[name] = rand(self.num_envs, self.stage_dofs)
        if self.stage_dofs == XYYAW_ACTION_DIM:
            for name in (
                "xy_position",
                "xy_velocity",
                "xy_target",
                "previous_xy_action",
                "xy_workspace_margin",
            ):
                obs[name] = rand(self.num_envs, 2)
        return obs

    def reset(self):
        self._obs = self._make_obs()
        return self._obs

    def step(self, actions):
        expected = (self.num_envs, MASTER_ACTION_DIM + self.stage_dofs)
        assert actions.shape == expected, actions.shape
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

    def set_stage_follower_active(self, active):
        self.lock_calls.append(bool(active))
        self.stage_follower_active = bool(active)
        return self.stage_follower_active

    def set_stage_curriculum_progress(self, progress):
        self.curriculum_calls.append(float(progress))
        values = {
            "yaw_workspace": YAW_STAGE.yaw_curriculum_value(0.05, 0.25, progress),
            "yaw_action_scale": YAW_STAGE.yaw_curriculum_value(0.005, 0.020, progress),
        }
        if self.stage_dofs == XYYAW_ACTION_DIM:
            values["xy_workspace"] = XY_STAGE.curriculum_value(0.01, 0.05, progress)
            values["xy_action_scale"] = XY_STAGE.curriculum_value(0.002, 0.005, progress)
        self.curriculum = values
        return values


def _full_config(yaml_path=XYYAW_YAML, *, num_actors=4, horizon=2, minibatch=8):
    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    raw["ppo"]["num_actors"] = num_actors
    raw["ppo"]["horizon_length"] = horizon
    raw["ppo"]["minibatch_size"] = minibatch
    raw["ppo"]["mini_epochs"] = 1
    raw["ppo"]["priv_info_dim"] = PRIV_INFO_DIM
    raw["ppo"]["max_agent_steps"] = num_actors * horizon * 2
    raw["ppo"]["save_frequency"] = 0
    raw["follower"]["minibatch_size"] = minibatch
    raw["follower"]["mini_epochs"] = 1
    raw["follower"]["actor_units"] = [32, 32]
    raw["follower"]["critic_units"] = [32, 32]
    raw["hierarchical"]["xy_curriculum_ramp_steps"] = 512
    # Keep the shipped Stage-1 length unless a test overrides it.
    return OmegaConf.create({"rl_device": "cpu", "test": False, "seed": 0, "train": raw})


def _make_agent(tmp_path, env=None, full_config=None, stage_dofs=XYYAW_ACTION_DIM):
    env = env or FakeStageEnv(stage_dofs=stage_dofs)
    config = full_config or _full_config(
        XYYAW_YAML if stage_dofs == XYYAW_ACTION_DIM else YAW_YAML
    )
    return HierarchicalPPO(env, str(tmp_path), config), env


def _master_kwargs(yaml_path=XYYAW_YAML):
    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    ppo = dict(raw["ppo"])
    ppo["priv_info_dim"] = PRIV_INFO_DIM
    return build_actor_critic_kwargs(
        raw["network"],
        ppo,
        MASTER_ACTION_DIM,
        (MASTER_OBS_DIM,),
        47,
        False,
        env_cfg=_env_cfg(XYYAW_ACTION_DIM),
    )


def _compile_functions(path: Path, names, namespace):
    """Compile the named top-level/method functions without importing Isaac."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    wanted = list(names)
    selected = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    found = {node.name for node in selected}
    assert found == set(wanted), f"missing {set(wanted) - found} in {path}"
    module = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    compiled = dict(namespace)
    exec(compile(module, str(path), "exec"), compiled)
    # Let the compiled functions call each other.
    for name in wanted:
        compiled[name] = compiled[name]
    return {name: compiled[name] for name in wanted}


def _compile_function(path: Path, name: str, namespace):
    return _compile_functions(path, [name], namespace)[name]


def _class_defaults(path: Path, class_name: str, namespace) -> SimpleNamespace:
    """Read a config class's shipped literal defaults without Isaac Lab."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.ClassDef) and item.name == class_name
    )
    values = {}
    for statement in node.body:
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            continue
        target = statement.targets[0]
        if not isinstance(target, ast.Name):
            continue
        values[target.id] = eval(  # noqa: S307 - fixed literals from our own source
            ast.unparse(statement.value), {"__builtins__": {}}, dict(namespace)
        )
    return SimpleNamespace(**values)


def _yaw_cfg_defaults() -> SimpleNamespace:
    return _class_defaults(
        YAW_ENV_CFG_PATH,
        "Revo3HandScrewTactileYawMixinCfg",
        {
            "NUM_YAW_DOFS": YAW_STAGE.NUM_YAW_DOFS,
            "YAW_ONLY_STAGE_JOINT_NAMES": YAW_STAGE.YAW_ONLY_STAGE_JOINT_NAMES,
            "YAW_ONLY_STAGE_WORLD_AXES": YAW_STAGE.YAW_ONLY_STAGE_WORLD_AXES,
        },
    )


def _xyyaw_cfg_defaults() -> SimpleNamespace:
    return _class_defaults(
        XYYAW_ENV_CFG_PATH,
        "Revo3HandScrewTactileXYYawMixinCfg",
        {
            "NUM_XY_DOFS": XY_STAGE.NUM_XY_DOFS,
            "NUM_XYYAW_DOFS": YAW_STAGE.NUM_XYYAW_DOFS,
            "XYYAW_STAGE_JOINT_NAMES": YAW_STAGE.XYYAW_STAGE_JOINT_NAMES,
            "XYYAW_STAGE_WORLD_AXES": YAW_STAGE.XYYAW_STAGE_WORLD_AXES,
        },
    )


# ---------------------------------------------------------------------------
# 1. action-space and stage-joint contracts
# ---------------------------------------------------------------------------


def test_xyyaw_config_declares_a_24_channel_action_space():
    defaults = _xyyaw_cfg_defaults()
    assert defaults.action_space == 24
    assert defaults.finger_action_space == 21
    assert defaults.action_space - defaults.finger_action_space == 3
    assert tuple(defaults.stage_joint_names) == STAGE_JOINT_ORDER
    assert tuple(defaults.stage_world_axes) == ("X", "Y", "Z")


def test_yaw_only_config_declares_a_22_channel_action_space():
    defaults = _yaw_cfg_defaults()
    assert defaults.action_space == 22
    assert defaults.finger_action_space == 21
    assert tuple(defaults.stage_joint_names) == ("stage_yaw_joint",)
    assert tuple(defaults.stage_world_axes) == ("Z",)


@pytest.mark.parametrize(
    "joint_names",
    (
        ("stage_x_joint", "stage_y_joint", "stage_yaw_joint", "right_thumb_CMP_joint"),
        ("stage_yaw_joint", "right_thumb_CMP_joint", "stage_y_joint", "stage_x_joint"),
        ("right_index_MCP_joint", "stage_y_joint", "stage_yaw_joint", "stage_x_joint"),
    ),
)
def test_stage_dof_indices_come_from_joint_names_not_articulation_order(joint_names):
    indices = YAW_STAGE.resolve_xyyaw_dof_indices(joint_names)
    assert [joint_names[index] for index in indices] == list(STAGE_JOINT_ORDER)
    assert joint_names[YAW_STAGE.resolve_yaw_dof_index(joint_names)] == "stage_yaw_joint"


def test_stage_dof_resolution_fails_loudly_when_the_yaw_joint_is_missing():
    with pytest.raises(ValueError, match="stage_yaw_joint"):
        YAW_STAGE.resolve_xyyaw_dof_indices(
            ("stage_x_joint", "stage_y_joint", "right_thumb_CMP_joint")
        )


def test_yaw_joint_cannot_be_captured_by_the_finger_or_xy_actuator_patterns():
    import re

    finger_pattern = re.compile("right_.*")
    xy_pattern = re.compile(XY_STAGE.XY_STAGE_ACTUATOR_EXPR)
    yaw_pattern = re.compile(YAW_STAGE.YAW_STAGE_ACTUATOR_EXPR)
    assert finger_pattern.fullmatch(YAW_STAGE.YAW_STAGE_JOINT_NAME) is None
    # The critical regression: the XY group must NOT swallow the yaw joint,
    # or yaw would inherit the 120 N linear effort limit.
    assert xy_pattern.fullmatch(YAW_STAGE.YAW_STAGE_JOINT_NAME) is None
    assert yaw_pattern.fullmatch(YAW_STAGE.YAW_STAGE_JOINT_NAME) is not None
    for name in XY_STAGE.XY_STAGE_JOINT_NAMES:
        assert xy_pattern.fullmatch(name) is not None
        assert yaw_pattern.fullmatch(name) is None
    assert YAW_STAGE.YAW_STAGE_ACTUATOR_GROUP != XY_STAGE.XY_STAGE_ACTUATOR_GROUP


def test_action_split_separates_finger_xy_and_yaw_blocks():
    action = torch.arange(XYYAW_ENV_ACTION_DIM, dtype=torch.float32).repeat(5, 1)
    finger, xy, yaw = YAW_STAGE.split_xyyaw_action(action, MASTER_ACTION_DIM)
    assert finger.shape == (5, 21)
    assert xy.shape == (5, 2)
    assert yaw.shape == (5, 1)
    torch.testing.assert_close(finger, action[:, :21])
    torch.testing.assert_close(xy, action[:, 21:23])
    torch.testing.assert_close(yaw, action[:, 23:24])


def test_yaw_only_action_split_separates_finger_and_yaw_blocks():
    action = torch.arange(YAW_ENV_ACTION_DIM, dtype=torch.float32).repeat(3, 1)
    finger, yaw = YAW_STAGE.split_yaw_action(action, MASTER_ACTION_DIM)
    assert finger.shape == (3, 21)
    assert yaw.shape == (3, 1)
    torch.testing.assert_close(yaw, action[:, 21:22])


@pytest.mark.parametrize("bad_width", (21, 22, 23, 25))
def test_wrong_action_width_is_rejected_with_an_explicit_message(bad_width):
    with pytest.raises(ValueError, match="24 channels"):
        YAW_STAGE.split_xyyaw_action(torch.zeros(4, bad_width), MASTER_ACTION_DIM)


# ---------------------------------------------------------------------------
# 2. follower observation widths
# ---------------------------------------------------------------------------


def test_three_dof_follower_observation_is_exactly_164_dims():
    assert follower_obs_dim(3) == XYYAW_FOLLOWER_OBS_DIM
    assert follower_obs_spec(3) == (
        ("executed_hand_action", 21),
        ("tactile_latent", 128),
        ("stage_position", 3),
        ("stage_velocity", 3),
        ("stage_target", 3),
        ("previous_stage_action", 3),
        ("stage_workspace_margin", 3),
    )
    blocks = {
        "stage_position": torch.full((3, 3), 3.0),
        "stage_velocity": torch.full((3, 3), 4.0),
        "stage_target": torch.full((3, 3), 5.0),
        "previous_stage_action": torch.full((3, 3), 6.0),
        "stage_workspace_margin": torch.full((3, 3), 7.0),
    }
    follower_obs = build_stage_follower_obs(
        executed_hand_action=torch.full((3, 21), 1.0),
        tactile_latent=torch.full((3, 128), 2.0),
        stage_blocks=blocks,
        num_stage_dofs=3,
    )
    assert follower_obs.shape == (3, 164)
    slices = follower_obs_slices(3)
    torch.testing.assert_close(
        follower_obs[:, slices["executed_hand_action"]], torch.full((3, 21), 1.0)
    )
    torch.testing.assert_close(
        follower_obs[:, slices["tactile_latent"]], torch.full((3, 128), 2.0)
    )
    for name, block in blocks.items():
        torch.testing.assert_close(follower_obs[:, slices[name]], block)
    # Field boundaries are exactly where the contract says they are.
    assert slices["stage_position"] == slice(149, 152)
    assert slices["stage_velocity"] == slice(152, 155)
    assert slices["stage_target"] == slice(155, 158)
    assert slices["previous_stage_action"] == slice(158, 161)
    assert slices["stage_workspace_margin"] == slice(161, 164)


def test_one_dof_follower_observation_is_exactly_154_dims():
    assert follower_obs_dim(1) == YAW_FOLLOWER_OBS_DIM
    slices = follower_obs_slices(1)
    assert slices["stage_position"] == slice(149, 150)
    assert slices["stage_workspace_margin"] == slice(153, 154)


def test_two_dof_follower_observation_is_still_exactly_159_dims():
    """The shipped XY layout and its ``xy_*`` block names must not move."""
    assert FOLLOWER_OBS_DIM == 159
    assert follower_obs_dim(2) == 159
    assert follower_obs_spec(2) == (
        ("executed_hand_action", 21),
        ("tactile_latent", 128),
        ("xy_position", 2),
        ("xy_velocity", 2),
        ("xy_target", 2),
        ("previous_xy_action", 2),
        ("xy_workspace_margin", 2),
    )


@pytest.mark.parametrize("bad_width", (0, 4, 6, 21))
def test_unsupported_stage_widths_are_rejected(bad_width):
    with pytest.raises(ValueError, match="Unsupported hierarchical stage width"):
        validate_stage_dofs(bad_width)


def test_three_dof_follower_reader_touches_only_the_five_stage_channels(tmp_path):
    env = FakeStageEnv(num_envs=4)
    agent, _ = _make_agent(tmp_path, env=env)
    obs = env.reset()
    master_result = agent.master_act(obs)
    executed_hand_action = torch.clamp(master_result["actions"], -1.0, 1.0)
    follower_obs = agent._follower_obs_from_env(
        obs,
        executed_hand_action=executed_hand_action,
        tactile_latent=master_result["tactile_latent"].detach(),
    )
    assert follower_obs.shape == (4, 164)

    # Structural: a dict holding ONLY the five stage channels reproduces it.
    stage_only = {name: obs[name] for name in STAGE_OBS_BLOCKS}
    torch.testing.assert_close(
        stage_follower_obs_from_env(
            stage_only,
            executed_hand_action=executed_hand_action,
            tactile_latent=master_result["tactile_latent"].detach(),
            num_stage_dofs=3,
        ),
        follower_obs,
    )

    # Numeric: every forbidden channel carries a sentinel magnitude >= 100.
    for forbidden in (
        obs["finger_position_sentinel"],
        obs["finger_target_sentinel"],
        obs["obs"],
        obs["priv_info"],
        obs["tactile_hist"],
    ):
        assert float(forbidden.abs().min()) >= 100.0
    assert float(follower_obs.abs().max()) < 10.0


def test_three_dof_follower_actor_never_receives_the_privileged_critic_slice(tmp_path):
    agent, env = _make_agent(tmp_path)
    assert agent.follower_critic_priv_dim == 11
    assert agent.follower.actor_mlp.mlp[0].in_features == 164
    assert agent.follower.critic_mlp.mlp[0].in_features == 164 + 11
    obs = env.reset()
    follower_obs = torch.randn(env.num_envs, 164)
    with torch.no_grad():
        baseline = agent.follower.act_inference(follower_obs)
        agent.follower.act(follower_obs, obs["priv_info"][:, :11])
        changed = agent.follower.act_inference(follower_obs)
    torch.testing.assert_close(baseline, changed)
    with pytest.raises(TypeError):
        agent.follower.act_inference(follower_obs, obs["priv_info"][:, :11])


def test_missing_stage_channels_raise_a_helpful_key_error():
    with pytest.raises(KeyError, match="stage_position"):
        stage_follower_obs_from_env(
            {"xy_position": torch.zeros(2, 2)},
            executed_hand_action=torch.zeros(2, 21),
            tactile_latent=torch.zeros(2, 128),
            num_stage_dofs=3,
        )


# ---------------------------------------------------------------------------
# 3. yaw controller (radians)
# ---------------------------------------------------------------------------


def test_yaw_target_respects_workspace_velocity_and_acceleration_limits():
    zeros = torch.zeros(1, 1)
    # Unconstrained by the rate limits: delta == action_scale * a_s.
    target, delta, smoothed = YAW_STAGE.update_yaw_target(
        torch.ones(1, 1),
        zeros,
        zeros,
        zeros,
        action_scale=0.040,
        workspace=0.60,
        velocity_limit=1.2,
        acceleration_limit=10_000.0,
        dt=0.05,
        smoothing=0.0,
    )
    torch.testing.assert_close(target, torch.full((1, 1), 0.040))
    torch.testing.assert_close(delta, torch.full((1, 1), 0.040))
    torch.testing.assert_close(smoothed, torch.ones(1, 1))

    # Smoothing: a_s = 0.5 * a + 0.5 * a_prev.
    _target, _delta, smoothed = YAW_STAGE.update_yaw_target(
        torch.ones(1, 1),
        zeros,
        zeros,
        torch.full((1, 1), -1.0),
        action_scale=0.040,
        workspace=0.60,
        velocity_limit=1.2,
        acceleration_limit=10_000.0,
        dt=0.05,
        smoothing=0.5,
    )
    torch.testing.assert_close(smoothed, torch.zeros(1, 1))

    # Angular velocity limit: 1.2 rad/s * 0.05 s = 0.06 rad per control step.
    target, _delta, _smoothed = YAW_STAGE.update_yaw_target(
        torch.ones(1, 1),
        zeros,
        torch.full((1, 1), 0.5),
        zeros,
        action_scale=5.0,
        workspace=1.0,
        velocity_limit=1.2,
        acceleration_limit=10_000.0,
        dt=0.05,
        smoothing=0.0,
    )
    torch.testing.assert_close(target, torch.full((1, 1), 0.06))

    # Angular acceleration limit: |delta - prev| <= a * dt^2 = 12 * 0.0025 = 0.03.
    target, _delta, _smoothed = YAW_STAGE.update_yaw_target(
        torch.ones(1, 1),
        zeros,
        torch.full((1, 1), -0.03),
        zeros,
        action_scale=5.0,
        workspace=1.0,
        velocity_limit=1_000.0,
        acceleration_limit=12.0,
        dt=0.05,
        smoothing=0.0,
    )
    torch.testing.assert_close(target, torch.zeros(1, 1))

    # Workspace clamp at the CURRENT curriculum half-range.
    target, delta, _smoothed = YAW_STAGE.update_yaw_target(
        torch.ones(1, 1),
        torch.full((1, 1), 0.149),
        zeros,
        zeros,
        action_scale=0.015,
        workspace=0.15,
        velocity_limit=10.0,
        acceleration_limit=10_000.0,
        dt=0.05,
        smoothing=0.0,
    )
    torch.testing.assert_close(target, torch.full((1, 1), 0.15))
    torch.testing.assert_close(delta, torch.full((1, 1), 0.001))


def test_shipped_yaw_defaults_reach_full_command_in_one_step_and_reverse_in_two():
    """Pin the shipped rate limits against the SHIPPED defaults, not literals.

    With the narrowed curriculum the acceleration ceiling
    ``a * dt^2 = 12 * 0.0025 = 0.03 rad`` now exceeds the final action scale
    (0.020 rad), so a full-scale command takes effect immediately and only a
    full reversal (a 0.040 rad swing) is rate limited, over two control steps.
    """
    cfg = _yaw_cfg_defaults()
    scale = cfg.yaw_action_scale_final
    workspace = cfg.yaw_workspace_final
    accel = cfg.yaw_acceleration_limit
    dt = 1.0 / 20.0
    max_delta_change = accel * dt * dt
    assert max_delta_change == pytest.approx(0.03)
    # A single full-scale command is no longer acceleration limited.
    assert max_delta_change > scale
    # A full reversal still is.
    assert 2.0 * scale > max_delta_change

    def advance(action, target, delta, smoothed):
        return YAW_STAGE.update_yaw_target(
            action,
            target,
            delta,
            smoothed,
            action_scale=scale,
            workspace=workspace,
            velocity_limit=cfg.yaw_velocity_limit,
            acceleration_limit=accel,
            dt=dt,
            smoothing=0.0,
        )

    target = torch.zeros(1, 1)
    delta = torch.zeros(1, 1)
    smoothed = torch.zeros(1, 1)
    target, delta, smoothed = advance(torch.ones(1, 1), target, delta, smoothed)
    assert float(delta) == pytest.approx(scale)

    reverse = []
    for _ in range(2):
        target, delta, smoothed = advance(-torch.ones(1, 1), target, delta, smoothed)
        reverse.append(float(delta))
    assert reverse == pytest.approx([scale - max_delta_change, -scale])


def test_yaw_pd_controller_is_torque_limited():
    effort = YAW_STAGE.yaw_pd_effort(
        torch.tensor([[0.60, -0.60]]),
        torch.zeros(1, 2),
        torch.zeros(1, 2),
        pgain=8.0,
        dgain=0.5,
        effort_limit=0.30,
    )
    torch.testing.assert_close(effort, torch.tensor([[0.30, -0.30]]))

    # Inside the ceiling it is the ordinary PD law: 8 * 0.01 - 0.5 * 0.2 = -0.02.
    effort = YAW_STAGE.yaw_pd_effort(
        torch.tensor([[0.01]]),
        torch.zeros(1, 1),
        torch.tensor([[0.2]]),
        pgain=8.0,
        dgain=0.5,
        effort_limit=0.30,
    )
    torch.testing.assert_close(effort, torch.tensor([[-0.02]]))

    # Saturation begins at |error| = tau_limit / kp.
    boundary = YAW_STAGE.yaw_pd_effort(
        torch.tensor([[0.0375]]),
        torch.zeros(1, 1),
        torch.zeros(1, 1),
        pgain=8.0,
        dgain=0.5,
        effort_limit=0.30,
    )
    assert float(boundary) == pytest.approx(0.30)


def test_shipped_yaw_actuator_can_track_and_is_not_damping_starved():
    """The torque budget must leave room for the proportional term.

    The first shipped ceiling (0.30 N*m) failed both checks below: the damping
    term alone reached kd * yaw_velocity_limit = 0.6 N*m = 200% of the budget,
    so above half the velocity limit the controller was a pure saturated brake,
    and the linear tracking band was only 6% of the final workspace.
    """
    cfg = _yaw_cfg_defaults()
    kp, kd, tau = cfg.yaw_pgain, cfg.yaw_dgain, cfg.yaw_effort_limit

    # 1) Damping at full commanded speed must not eat the whole budget.
    damping_at_max_speed = kd * cfg.yaw_velocity_limit
    assert damping_at_max_speed / tau < 0.5, (
        f"damping alone uses {damping_at_max_speed / tau:.0%} of the torque budget"
    )

    # 2) The linear (non-saturated) band must cover most of the workspace.
    linear_band = tau / kp
    assert linear_band / cfg.yaw_workspace_final >= 0.5, (
        f"linear band {linear_band:.3f} rad is only "
        f"{linear_band / cfg.yaw_workspace_final:.0%} of the final workspace"
    )

    # 3) The gains themselves are unchanged and still well damped against the
    #    measured hand inertia about the world-Z yaw axis (9.95e-3 kg*m^2),
    #    matching the validated XY stage's zeta ~ 0.91.
    inertia = 9.95e-3
    zeta = kd / (2.0 * math.sqrt(kp * inertia))
    assert 0.7 <= zeta <= 1.1, f"yaw damping ratio {zeta:.2f} is outside [0.7, 1.1]"

    # 4) It must track inside the early curriculum without saturating, while
    #    still binding on large excursions - the XY stage is sized the same way
    #    (its ceiling binds 10-25% of the time under load), so the joint never
    #    becomes a free position source.
    def hold(error):
        return abs(float(YAW_STAGE.yaw_pd_effort(
            torch.zeros(1, 1), torch.tensor([[error]]), torch.zeros(1, 1),
            pgain=kp, dgain=kd, effort_limit=tau,
        )))

    assert hold(cfg.yaw_workspace_initial) < tau, "saturates inside the initial workspace"
    assert hold(0.9 * linear_band) < tau
    assert hold(cfg.yaw_workspace_final) == pytest.approx(tau), (
        "the ceiling must still bind on a full-workspace excursion"
    )


def test_yaw_workspace_margin_and_boundary_saturation():
    position = torch.tensor([[0.0], [0.075], [0.15], [-0.30]])
    margin = YAW_STAGE.yaw_workspace_margin(position, 0.15)
    torch.testing.assert_close(margin, torch.tensor([[1.0], [0.5], [0.0], [0.0]]))
    assert torch.all(margin >= 0.0) and torch.all(margin <= 1.0)

    saturation = YAW_STAGE.yaw_boundary_saturation(position, 0.15, 0.10)
    # 0 inside 0.9 * W = 0.135 rad, then linear to 1 at the boundary.
    torch.testing.assert_close(
        saturation, torch.tensor([[0.0], [0.0], [1.0], [1.0]])
    )
    mid = YAW_STAGE.yaw_boundary_saturation(torch.tensor([[0.1425]]), 0.15, 0.10)
    assert float(mid) == pytest.approx(0.5, abs=1.0e-6)


def test_yaw_config_contract_validation():
    cfg = _yaw_cfg_defaults()
    limits = YAW_STAGE.validate_yaw_stage_config(cfg)
    assert limits.joint_limit == pytest.approx(0.70)
    assert limits.workspace_initial == pytest.approx(0.05)
    assert limits.workspace_final == pytest.approx(0.25)
    assert limits.action_scale_initial == pytest.approx(0.005)
    assert limits.action_scale_final == pytest.approx(0.020)
    assert limits.velocity_limit == pytest.approx(1.2)
    assert limits.acceleration_limit == pytest.approx(12.0)
    assert limits.effort_limit == pytest.approx(1.5)
    assert limits.pgain == pytest.approx(8.0)
    assert limits.dgain == pytest.approx(0.5)
    assert limits.action_smoothing == pytest.approx(0.8)
    assert limits.curriculum_ramp_steps == 20_000_000
    assert cfg.yaw_use_action_delay is True
    assert cfg.yaw_joint_velocity_limit_sim == pytest.approx(3.0)
    assert cfg.yaw_position_obs_scale == pytest.approx(limits.joint_limit)
    assert cfg.yaw_velocity_obs_scale == pytest.approx(limits.velocity_limit)

    too_wide = copy.copy(cfg)
    too_wide.yaw_workspace_final = 0.9
    with pytest.raises(ValueError, match="joint hard limit"):
        YAW_STAGE.validate_yaw_stage_config(too_wide)

    continuous = copy.copy(cfg)
    continuous.yaw_joint_limit = 2.0 * math.pi
    with pytest.raises(ValueError, match="below pi"):
        YAW_STAGE.validate_yaw_stage_config(continuous)

    inverted = copy.copy(cfg)
    inverted.yaw_workspace_initial = 0.8
    with pytest.raises(ValueError, match="yaw_workspace_initial"):
        YAW_STAGE.validate_yaw_stage_config(inverted)


def test_yaw_costs_are_all_non_positive_and_small():
    cfg = _yaw_cfg_defaults()
    for name in (
        "yaw_velocity_penalty_scale",
        "yaw_acceleration_penalty_scale",
        "yaw_jerk_penalty_scale",
        "yaw_effort_penalty_scale",
        "yaw_power_penalty_scale",
        "yaw_boundary_penalty_scale",
    ):
        value = float(getattr(cfg, name))
        assert value < 0.0, name
        assert abs(value) <= 0.1, name
    assert cfg.high_speed_reward_enable is False
    # One control step at 20 Hz cannot command more than the angular velocity
    # limit allows.
    assert cfg.yaw_action_scale_final <= cfg.yaw_velocity_limit * (1.0 / 20.0)
    # The jerk reference must leave the measured jerk BELOW 1.0 in cost units.
    # Jerk is a double finite difference at 20 Hz (amplified by 1/dt^2 = 400),
    # so a reference of only a few times the acceleration limit turns this term
    # into the dominant cost purely from differentiation noise.
    assert cfg.yaw_jerk_reference >= 20.0 * cfg.yaw_acceleration_limit


def test_yaw_runtime_limits_are_validated_in_radians_not_degrees():
    limit = 0.70
    YAW_STAGE.assert_runtime_yaw_limits(
        torch.full((4, 1), -limit), torch.full((4, 1), limit), limit
    )
    degrees = YAW_STAGE.radians_to_degrees(limit)
    assert degrees == pytest.approx(40.1070456, abs=1.0e-5)
    # A missing conversion leaves the runtime limits in degrees: caught loudly.
    with pytest.raises(RuntimeError, match="DEGREES"):
        YAW_STAGE.assert_runtime_yaw_limits(
            torch.full((4, 1), -degrees), torch.full((4, 1), degrees), limit
        )
    assert YAW_STAGE.degrees_to_radians(degrees) == pytest.approx(limit)


# ---------------------------------------------------------------------------
# 4. curriculum: one latch, one shared progress
# ---------------------------------------------------------------------------


def test_stage0_sends_exactly_three_zero_stage_channels(tmp_path):
    env = FakeStageEnv(num_envs=4)
    agent, _ = _make_agent(tmp_path, env=env)
    assert agent.current_stage == STAGE_MASTER
    assert agent.follower_action_dim == 3
    agent.obs = env.reset()
    agent.horizon_length = 1
    agent.storage.transitions_per_env = 1
    agent.play_steps()

    stage_block = env.received_actions[0][:, MASTER_ACTION_DIM:]
    assert stage_block.shape == (4, 3)
    torch.testing.assert_close(stage_block, torch.zeros_like(stage_block))
    assert torch.count_nonzero(stage_block) == 0


def test_stage0_yaw_only_sends_exactly_one_zero_channel(tmp_path):
    env = FakeStageEnv(stage_dofs=1, num_envs=4)
    agent, _ = _make_agent(tmp_path, env=env, stage_dofs=1)
    assert agent.follower_action_dim == 1
    assert agent.follower_obs_dim == 154
    agent.obs = env.reset()
    agent.horizon_length = 1
    agent.storage.transitions_per_env = 1
    agent.play_steps()
    stage_block = env.received_actions[0][:, MASTER_ACTION_DIM:]
    assert stage_block.shape == (4, 1)
    torch.testing.assert_close(stage_block, torch.zeros_like(stage_block))


def test_xy_and_yaw_activate_at_exactly_the_same_agent_step(tmp_path):
    env = FakeStageEnv(num_envs=4)
    agent, _ = _make_agent(tmp_path, env=env)
    agent.activation_speed_ema_beta = 0.0  # EMA == last rollout speed
    assert agent.activation_speed_threshold == pytest.approx(0.8)
    assert agent.activation_patience == 5
    agent.agent_steps = 4_000

    for _ in range(10):
        agent._update_curriculum(0.79)
        assert agent.current_stage == STAGE_MASTER
        assert agent.activation_patience_counter == 0

    for step in range(1, agent.activation_patience):
        agent._update_curriculum(0.81)
        assert agent.current_stage == STAGE_MASTER
        assert agent.activation_patience_counter == step
    # A single dip resets the consecutive-epoch counter.
    agent._update_curriculum(0.5)
    assert agent.activation_patience_counter == 0

    for _ in range(agent.activation_patience):
        agent._update_curriculum(1.2)
    assert agent.current_stage == STAGE_FOLLOWER
    activation_step = agent.activation_agent_step
    assert activation_step == 4_000

    # One latch => translation and yaw share the same activation step, and the
    # first published curriculum is each DOF's own INITIAL value.
    curriculum = agent._push_stage_curriculum()
    assert curriculum["stage_progress"] == pytest.approx(0.0)
    assert curriculum["xy_workspace"] == pytest.approx(0.01)
    assert curriculum["xy_action_scale"] == pytest.approx(0.002)
    assert curriculum["yaw_workspace"] == pytest.approx(0.05)
    assert curriculum["yaw_action_scale"] == pytest.approx(0.005)

    # Latched: a later speed collapse never reverts or re-times the stage.
    for speed in (-5.0, 0.0, 0.1):
        agent.agent_steps += 1_000
        agent._update_curriculum(speed)
        assert agent.current_stage == STAGE_FOLLOWER
        assert agent.activation_agent_step == activation_step


def test_one_shared_progress_interpolates_metres_and_radians_together(tmp_path):
    env = FakeStageEnv(num_envs=4)
    agent, _ = _make_agent(tmp_path, env=env)
    agent.activation_speed_ema_beta = 0.0
    agent.agent_steps = 4_000
    for _ in range(agent.activation_patience):
        agent._update_curriculum(2.0)
    assert agent.current_stage == STAGE_FOLLOWER
    ramp = agent.xy_curriculum_ramp_steps

    for fraction, xy_workspace, xy_scale, yaw_workspace, yaw_scale in (
        (0.0, 0.01, 0.002, 0.05, 0.005),
        (0.25, 0.02, 0.00275, 0.10, 0.00875),
        (0.5, 0.03, 0.0035, 0.15, 0.0125),
        (1.0, 0.05, 0.005, 0.25, 0.020),
        (3.0, 0.05, 0.005, 0.25, 0.020),  # clamped past the end of the ramp
    ):
        agent.agent_steps = 4_000 + int(fraction * ramp)
        curriculum = agent._push_stage_curriculum()
        expected_progress = min(fraction, 1.0)
        # The SAME dimensionless progress drives both DOFs...
        assert curriculum["stage_progress"] == pytest.approx(expected_progress)
        assert agent.xy_curriculum_progress() == pytest.approx(expected_progress)
        assert agent.stage_curriculum_progress() == pytest.approx(expected_progress)
        # ...but each is interpolated between its OWN units.
        assert curriculum["xy_workspace"] == pytest.approx(xy_workspace)
        assert curriculum["xy_action_scale"] == pytest.approx(xy_scale)
        assert curriculum["yaw_workspace"] == pytest.approx(yaw_workspace)
        assert curriculum["yaw_action_scale"] == pytest.approx(yaw_scale)
        # Metres and radians are never the same number.
        assert curriculum["xy_workspace"] != pytest.approx(curriculum["yaw_workspace"])

    # The environment received exactly one progress push per call, never two
    # separately timed pushes.
    assert env.curriculum_calls == pytest.approx(
        [0.0, 0.25, 0.5, 1.0, 1.0], abs=1.0e-9
    )


def test_stage_joints_stay_mechanically_locked_until_the_follower_activates(tmp_path):
    """A zero target is not a lock: Stage 0 must hold the joints physically."""
    env = FakeStageEnv(num_envs=4)
    agent, _ = _make_agent(tmp_path, env=env)
    agent.activation_speed_ema_beta = 0.0

    # Stage 0: every push tells the environment to keep the stage locked.
    for _ in range(3):
        curriculum = agent._push_stage_curriculum()
        assert curriculum["stage_locked"] == 1.0
    assert env.lock_calls == [False, False, False]
    assert env.stage_follower_active is False

    for _ in range(agent.activation_patience):
        agent._update_curriculum(2.0)
    assert agent.current_stage == STAGE_FOLLOWER

    # Activation releases the joints in the SAME push that first samples them.
    curriculum = agent._push_stage_curriculum()
    assert curriculum["stage_locked"] == 0.0
    assert env.lock_calls[-1] is True
    assert env.stage_follower_active is True

    # Latched: it never re-locks, even if the speed collapses afterwards.
    for speed in (-5.0, 0.0):
        agent.agent_steps += 1_000
        agent._update_curriculum(speed)
        assert agent._push_stage_curriculum()["stage_locked"] == 0.0
    assert all(env.lock_calls[3:])


def test_env_latches_the_lock_and_writes_limits_only_on_transitions():
    """The real env writes PhysX limits once per transition, not every epoch."""
    writes = []

    class _FakeHand:
        def write_joint_position_limit_to_sim(self, limits, joint_ids=None, **kwargs):
            writes.append((limits.clone(), list(joint_ids)))

    unlocked = torch.tensor([[[-0.05, 0.05], [-0.05, 0.05], [-0.70, 0.70]]]).repeat(
        2, 1, 1
    )
    env = SimpleNamespace(
        cfg=SimpleNamespace(
            stage_lock_when_follower_inactive=True,
            stage_lock_tolerance_m=1.0e-4,
            stage_lock_tolerance_rad=1.0e-4,
            stage_joint_names=STAGE_JOINT_ORDER,
        ),
        hand=_FakeHand(),
        stage_follower_active=None,
        stage_lock_dof_indices=[0, 1, 2],
        _stage_unlocked_limits=unlocked,
        _num_preceding_stage_dofs=lambda: 2,
        _reset_stage_controller_buffers=lambda env_ids: None,
    )
    functions = _compile_functions(
        YAW_ENV_PATH,
        ["set_stage_follower_active"],
        {"torch": torch, "stage_lock_limits": YAW_STAGE.stage_lock_limits},
    )
    set_active = functions["set_stage_follower_active"]

    set_active(env, False)
    assert len(writes) == 1
    locked_limits, joint_ids = writes[0]
    assert joint_ids == [0, 1, 2]
    # Every stage DOF is pinned to its residual play, in its own unit.
    assert float(locked_limits[0, 0, 1]) == pytest.approx(1.0e-4)  # X, metres
    assert float(locked_limits[0, 2, 1]) == pytest.approx(1.0e-4)  # yaw, radians

    # Repeated Stage-0 pushes are a no-op: no extra PhysX writes.
    for _ in range(5):
        set_active(env, False)
    assert len(writes) == 1

    # Activation restores EXACTLY the authored hard limits.
    set_active(env, True)
    assert len(writes) == 2
    torch.testing.assert_close(writes[1][0], unlocked)
    for _ in range(5):
        set_active(env, True)
    assert len(writes) == 2

    # The lock can be disabled outright for an ablation.
    env.cfg.stage_lock_when_follower_inactive = False
    env.stage_follower_active = None
    set_active(env, False)
    assert len(writes) == 2


def test_stage_lock_limits_hold_every_dof_at_zero_in_its_own_unit():
    # Authored hard limits: +/-0.05 m on X and Y, +/-0.70 rad on yaw.
    unlocked = torch.tensor(
        [[[-0.05, 0.05], [-0.05, 0.05], [-0.70, 0.70]]]
    ).repeat(4, 1, 1)
    locked = YAW_STAGE.stage_lock_limits(
        unlocked,
        num_linear_dofs=2,
        linear_tolerance=1.0e-4,
        angular_tolerance=1.0e-4,
    )
    assert locked.shape == unlocked.shape
    torch.testing.assert_close(locked[:, :2, 0], torch.full((4, 2), -1.0e-4))
    torch.testing.assert_close(locked[:, :2, 1], torch.full((4, 2), 1.0e-4))
    torch.testing.assert_close(locked[:, 2, 0], torch.full((4,), -1.0e-4))
    torch.testing.assert_close(locked[:, 2, 1], torch.full((4,), 1.0e-4))
    # Metres and radians keep independent tolerances.
    mixed = YAW_STAGE.stage_lock_limits(
        unlocked, num_linear_dofs=2, linear_tolerance=1.0e-3, angular_tolerance=2.0e-4
    )
    assert float(mixed[0, 0, 1]) == pytest.approx(1.0e-3)
    assert float(mixed[0, 2, 1]) == pytest.approx(2.0e-4)
    # The yaw-only chain has no linear DOF at all.
    yaw_only = YAW_STAGE.stage_lock_limits(
        torch.tensor([[[-0.70, 0.70]]]), num_linear_dofs=0, angular_tolerance=1.0e-4
    )
    torch.testing.assert_close(yaw_only, torch.tensor([[[-1.0e-4, 1.0e-4]]]))
    with pytest.raises(ValueError, match="num_linear_dofs"):
        YAW_STAGE.stage_lock_limits(unlocked, num_linear_dofs=5)


def test_the_lock_uses_joint_limits_and_never_raises_the_effort_ceiling():
    """The lock must be a limit constraint, not a stronger or teleporting drive."""
    source = YAW_ENV_PATH.read_text(encoding="utf-8")
    assert "write_joint_position_limit_to_sim" in source
    # It never rewrites the effort/velocity budget or teleports the joint.
    for forbidden in (
        "write_joint_effort_limit_to_sim",
        "write_joint_position_to_sim",
        "write_joint_state_to_sim",
        "write_root_pose_to_sim",
    ):
        assert forbidden not in source, forbidden
    # The authored limits are snapshotted so the release restores them exactly.
    assert "self._stage_unlocked_limits" in source
    ppo_source = (
        REPO_ROOT
        / "source/BrainCo_DexHand/BrainCo_DexHand/algo/hora/ppo/hierarchical_ppo.py"
    ).read_text(encoding="utf-8")
    assert "set_stage_follower_active" in ppo_source


def test_yaw_cfg_declares_the_stage_lock_knobs():
    defaults = _yaw_cfg_defaults()
    assert defaults.stage_lock_when_follower_inactive is True
    assert 0.0 < defaults.stage_lock_tolerance_m <= 1.0e-2
    assert 0.0 < defaults.stage_lock_tolerance_rad <= 1.0e-2


def test_stage0_keeps_the_curriculum_at_progress_zero(tmp_path):
    env = FakeStageEnv(num_envs=4)
    agent, _ = _make_agent(tmp_path, env=env)
    agent.agent_steps = 10 * agent.xy_curriculum_ramp_steps
    curriculum = agent._push_stage_curriculum()
    assert curriculum["stage_progress"] == 0.0
    assert curriculum["yaw_workspace"] == pytest.approx(0.05)
    assert curriculum["xy_workspace"] == pytest.approx(0.01)


# ---------------------------------------------------------------------------
# 5. stage 1 / stage 2 training behaviour
# ---------------------------------------------------------------------------


def test_stage1_freezes_the_master_and_trains_the_three_dof_follower(tmp_path):
    env = FakeStageEnv(num_envs=4)
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
    # All three follower channels are live, including yaw.
    assert agent.follower.mu.out_features == 3
    assert agent.follower.sigma.shape == (3,)


def test_stage2_starts_after_the_configured_follower_only_steps(tmp_path):
    config = _full_config()
    env = FakeStageEnv(num_envs=4)
    agent, _ = _make_agent(tmp_path, env=env, full_config=config)
    assert agent.joint_finetune_enable is True
    assert agent.follower_only_steps == 5_000_000
    agent.activation_speed_ema_beta = 0.0
    for _ in range(agent.activation_patience):
        agent._update_curriculum(2.0)
    assert agent.current_stage == STAGE_FOLLOWER

    # One step short of the boundary: still Stage 1.
    agent.agent_steps = agent.stage_start_agent_step + 5_000_000 - 1
    agent._update_curriculum(2.0)
    assert agent.current_stage == STAGE_FOLLOWER

    agent.agent_steps = agent.stage_start_agent_step + 5_000_000
    agent._update_curriculum(2.0)
    assert agent.current_stage == STAGE_JOINT_FINETUNE


def test_stage2_updates_the_master_and_the_three_dof_follower_every_epoch(tmp_path):
    config = _full_config()
    config.train.hierarchical.follower_only_steps = 0
    env = FakeStageEnv(num_envs=4)
    agent, _ = _make_agent(tmp_path, env=env, full_config=config)
    agent.activation_speed_ema_beta = 0.0
    for _ in range(agent.activation_patience):
        agent._update_curriculum(2.0)
    assert agent.master_reference is not None
    agent.agent_steps += 1
    agent._update_curriculum(2.0)
    assert agent.current_stage == STAGE_JOINT_FINETUNE
    assert agent.master_frozen is False

    # The tactile encoder stays frozen; the trunk, head and critic do not.
    assert all(not p.requires_grad for p in agent.master.tactile_encoder.parameters())
    assert any(p.requires_grad for p in agent.master.actor_mlp.parameters())
    assert agent.master.mu.weight.requires_grad

    before_master = copy.deepcopy(agent.master.state_dict())
    before_encoder = copy.deepcopy(agent.master.tactile_encoder.state_dict())
    before_follower = copy.deepcopy(agent.follower.state_dict())
    agent.obs = env.reset()
    stats = agent.train_epoch()

    # BOTH optimizers really stepped in this one rollout.
    assert stats["master_actor"] != []
    assert stats["follower_actor"] != []
    assert stats["master_kl_reg"] != []
    assert any(
        not torch.allclose(value, before_master[key])
        for key, value in agent.master.state_dict().items()
    )
    assert any(
        not torch.allclose(value, before_follower[key])
        for key, value in agent.follower.state_dict().items()
    )
    for key, value in agent.master.tactile_encoder.state_dict().items():
        torch.testing.assert_close(value, before_encoder[key])
    assert agent.master_lr == pytest.approx(
        agent.follower_lr * agent.master_finetune_lr_scale
    )
    assert agent.master_finetune_lr_scale == pytest.approx(0.07)
    assert agent.master_kl_coef == pytest.approx(1.0)


def test_follower_loss_never_backpropagates_into_the_master(tmp_path):
    env = FakeStageEnv(num_envs=4)
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


def test_rollout_stores_the_full_24_channel_joint_action(tmp_path):
    env = FakeStageEnv(num_envs=4)
    agent, _ = _make_agent(tmp_path, env=env)
    agent.current_stage = STAGE_FOLLOWER
    agent._apply_stage_freeze()
    agent.obs = env.reset()
    agent.play_steps()

    storage = agent.storage
    assert tuple(storage.storage_dict["env_actions"].shape) == (
        agent.horizon_length,
        4,
        24,
    )
    assert tuple(storage.storage_dict["follower_obses"].shape) == (
        agent.horizon_length,
        4,
        164,
    )
    assert tuple(storage.storage_dict["follower_actions"].shape) == (
        agent.horizon_length,
        4,
        3,
    )
    executed = torch.cat(
        [
            storage.storage_dict["master_executed_actions"],
            storage.storage_dict["follower_executed_actions"],
        ],
        dim=-1,
    )
    torch.testing.assert_close(storage.storage_dict["env_actions"], executed)
    assert torch.all(storage.storage_dict["follower_executed_actions"].abs() <= 1.0)


# ---------------------------------------------------------------------------
# 6. checkpoints
# ---------------------------------------------------------------------------


def test_checkpoint_records_every_action_dimension_and_the_stage_order(tmp_path):
    agent, _env = _make_agent(tmp_path)
    agent.save(str(tmp_path / "hier"))
    payload = torch.load(tmp_path / "hier.pth", map_location="cpu", weights_only=False)
    assert payload["format"] == CHECKPOINT_FORMAT
    assert payload["master_action_dim"] == 21
    assert payload["follower_action_dim"] == 3
    assert payload["follower_obs_dim"] == 164
    assert payload["env_action_dim"] == 24
    assert list(payload["stage_dof_names"]) == list(STAGE_JOINT_ORDER)
    for key in (
        "current_stage",
        "activation_agent_step",
        "stage_start_agent_step",
        "stage_curriculum_progress",
        "stage_curriculum_ramp_steps",
        "xy_curriculum_progress",
        "agent_steps",
        "epoch_num",
    ):
        assert key in payload


def test_checkpoint_round_trip_restores_stage_activation_and_curriculum(tmp_path):
    env = FakeStageEnv(num_envs=4)
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
    assert restored.activation_agent_step == agent.activation_agent_step
    assert restored.stage_start_agent_step == agent.stage_start_agent_step
    assert restored.agent_steps == agent.agent_steps
    assert restored.epoch_num == agent.epoch_num
    assert restored.follower_action_dim == 3
    assert restored.follower_obs_dim == 164
    assert restored.xy_curriculum_progress() == pytest.approx(
        agent.xy_curriculum_progress()
    )
    restored_curriculum = restored._push_stage_curriculum()
    agent_curriculum = agent._push_stage_curriculum()
    for key, value in agent_curriculum.items():
        assert restored_curriculum[key] == pytest.approx(value)
    for key, value in agent.follower.state_dict().items():
        torch.testing.assert_close(restored.follower.state_dict()[key], value)
    for key, value in agent.master.state_dict().items():
        torch.testing.assert_close(restored.master.state_dict()[key], value)


def _two_dof_checkpoint(tmp_path):
    """Write a genuine 2-D XY hierarchical checkpoint."""
    from BrainCo_DexHand.algo.hora.ppo.hierarchical_ppo import (
        HierarchicalPPO as _HierarchicalPPO,
    )

    class _XYFakeEnv(FakeStageEnv):
        def __init__(self):
            super().__init__(stage_dofs=2)
            self.cfg.stage_joint_names = ("stage_x_joint", "stage_y_joint")

        def _make_obs(self):
            obs = super()._make_obs()
            for name in (
                "xy_position",
                "xy_velocity",
                "xy_target",
                "previous_xy_action",
                "xy_workspace_margin",
            ):
                obs[name] = torch.rand(self.num_envs, 2, generator=self._generator)
            return obs

    raw = yaml.safe_load(XY_YAML.read_text(encoding="utf-8"))
    raw["ppo"].update(
        num_actors=4,
        horizon_length=2,
        minibatch_size=8,
        mini_epochs=1,
        priv_info_dim=PRIV_INFO_DIM,
        max_agent_steps=16,
        save_frequency=0,
    )
    raw["follower"].update(minibatch_size=8, mini_epochs=1, actor_units=[32, 32], critic_units=[32, 32])
    config = OmegaConf.create({"rl_device": "cpu", "test": False, "seed": 0, "train": raw})
    xy_agent = _HierarchicalPPO(_XYFakeEnv(), str(tmp_path / "xy"), config)
    assert xy_agent.follower_action_dim == 2
    assert xy_agent.follower_obs_dim == 159
    path = tmp_path / "xy_hier"
    xy_agent.save(str(path))
    return f"{path}.pth"


def test_two_dof_follower_checkpoint_is_refused_by_the_three_dof_task(tmp_path):
    checkpoint_path = _two_dof_checkpoint(tmp_path)
    agent, _env = _make_agent(tmp_path / "xyyaw")
    with pytest.raises(RuntimeError) as error:
        agent.restore_train(checkpoint_path)
    message = str(error.value)
    assert "follower_action_dim 2 != 3" in message
    assert "follower_obs_dim 159 != 164" in message
    assert "no weight migration" in message
    # It is never partially loaded.
    assert "--master_checkpoint" in message


def test_two_dof_follower_checkpoint_is_refused_even_without_dimension_fields(tmp_path):
    """Old checkpoints predate the dimension fields; the weights still say 2-D."""
    checkpoint_path = _two_dof_checkpoint(tmp_path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    for key in ("follower_action_dim", "follower_obs_dim", "stage_dof_names", "master_action_dim"):
        payload.pop(key, None)
    legacy_path = tmp_path / "legacy_xy.pth"
    torch.save(payload, legacy_path)

    agent, _env = _make_agent(tmp_path / "legacy")
    with pytest.raises(RuntimeError, match="follower_action_dim 2 != 3"):
        agent.restore_train(str(legacy_path))


def test_a_plain_21_dim_master_checkpoint_still_warm_starts_strictly(tmp_path):
    from BrainCo_DexHand.algo.hora.models.running_mean_std import RunningMeanStd

    reference_master = ActorCritic(copy.deepcopy(_master_kwargs()))
    with torch.no_grad():
        for parameter in reference_master.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.05)
    running = RunningMeanStd((MASTER_OBS_DIM,))
    running.running_mean.fill_(0.25)
    value = RunningMeanStd((1,))
    value.running_mean.fill_(3.5)
    path = tmp_path / "stage1_best.pth"
    torch.save(
        {
            "model": reference_master.state_dict(),
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

    agent, _env = _make_agent(tmp_path / "agent")
    agent.restore_master_checkpoint(str(path))
    for key, tensor in reference_master.state_dict().items():
        torch.testing.assert_close(agent.master.state_dict()[key], tensor)
    # A warm start is NOT a resume: counters and curriculum stay at zero.
    assert agent.agent_steps == 0
    assert agent.current_stage == STAGE_MASTER
    assert agent.follower_action_dim == 3


def test_master_state_dict_matches_the_stage1_ppo_teacher_contract():
    hierarchical_master = ActorCritic(copy.deepcopy(_master_kwargs()))
    stage1_master = ActorCritic(copy.deepcopy(_master_kwargs(yaml_path=FRAME813_YAML)))
    left = {k: tuple(v.shape) for k, v in hierarchical_master.state_dict().items()}
    right = {k: tuple(v.shape) for k, v in stage1_master.state_dict().items()}
    assert left == right
    hierarchical_master.load_state_dict(stage1_master.state_dict(), strict=True)


# ---------------------------------------------------------------------------
# 7. configuration files and entry points
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", (XYYAW_YAML, YAW_YAML))
def test_new_yamls_reuse_the_stage1_teacher_network_and_ppo_sections(path):
    new = yaml.safe_load(path.read_text(encoding="utf-8"))
    base = yaml.safe_load(FRAME813_YAML.read_text(encoding="utf-8"))
    xy = yaml.safe_load(XY_YAML.read_text(encoding="utf-8"))
    assert new["algo"] == "HierarchicalPPO"
    assert new["network"] == base["network"] == xy["network"]
    assert new["tactile_layout"] == base["tactile_layout"]
    assert new["network"]["tactile_encoder"]["gru_hidden_dim"] == 128
    for key, value in base["ppo"].items():
        assert new["ppo"][key] == value, key
    assert new["follower"]["critic_priv_dim"] == 11
    hierarchical = new["hierarchical"]
    assert hierarchical["joint_finetune_enable"] is True
    assert hierarchical["follower_only_steps"] == 5_000_000
    assert hierarchical["master_finetune_lr_scale"] == 0.07
    assert hierarchical["master_kl_coef"] == 1.0
    assert hierarchical["freeze_tactile_encoder_in_joint_finetune"] is True
    assert hierarchical["xy_curriculum_ramp_steps"] == 20_000_000
    assert hierarchical["activation_speed_threshold"] == 0.8
    assert hierarchical["activation_patience"] == 5


def test_the_shipped_xy_yaml_still_runs_follower_only_by_default():
    """The 2-DOF baseline keeps its clean follower-only ablation."""
    xy = yaml.safe_load(XY_YAML.read_text(encoding="utf-8"))
    assert xy["hierarchical"]["joint_finetune_enable"] is False


def test_xyyaw_smoke_yaml_exercises_every_curriculum_stage():
    smoke = yaml.safe_load(XYYAW_SMOKE_YAML.read_text(encoding="utf-8"))
    assert smoke["algo"] == "HierarchicalPPO"
    assert smoke["hierarchical"]["joint_finetune_enable"] is True
    assert smoke["hierarchical"]["activation_patience"] == 1
    assert smoke["hierarchical"]["activation_speed_threshold"] < 0.0
    assert smoke["ppo"]["max_agent_steps"] <= 100_000


def test_train_entry_point_registers_both_new_tasks():
    source = TRAIN_PATH.read_text(encoding="utf-8")
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
    assert {"valvedriver_tactile_xyyaw", "valvedriver_tactile_yaw"}.issubset(
        set(task_choices)
    )
    # The XY baseline and the flat tasks stay available.
    assert {"valvedriver_tactile_xy", "valvedriver_tactile", "valvedriver"}.issubset(
        set(task_choices)
    )
    assert {"PPO", "ProprioAdapt", "HierarchicalPPO"}.issubset(set(algo_choices))
    for line in (
        "'valvedriver_tactile_xyyaw': Revo3HandVavleDriverTactileXYYawEnvCfg",
        "'valvedriver_tactile_yaw': Revo3HandVavleDriverTactileYawEnvCfg",
        "'valvedriver_tactile_xyyaw': Revo3HandScrewTactileXYYawEnv",
        "'valvedriver_tactile_yaw': Revo3HandScrewTactileYawEnv",
        "'valvedriver_tactile_xyyaw': 'valvedriver_tactile_frame813_xyyaw'",
        "'valvedriver_tactile_yaw': 'valvedriver_tactile_frame813_yaw'",
    ):
        assert line in source, line


def test_play_entry_point_registers_both_new_tasks():
    source = PLAY_PATH.read_text(encoding="utf-8")
    for line in (
        "'valvedriver_tactile_xyyaw': Revo3HandVavleDriverTactileXYYawEnvCfg",
        "'valvedriver_tactile_yaw': Revo3HandVavleDriverTactileYawEnvCfg",
        "'valvedriver_tactile_xyyaw': Revo3HandScrewTactileXYYawEnv",
        "'valvedriver_tactile_yaw': Revo3HandScrewTactileYawEnv",
    ):
        assert line in source, line


@pytest.mark.parametrize(
    "task", ("valvedriver_tactile_xy", "valvedriver_tactile_xyyaw", "valvedriver_tactile_yaw")
)
def test_play_resolves_hierarchical_ppo_for_every_stage_task(task):
    hierarchical = (
        "valvedriver_tactile_xy",
        "valvedriver_tactile_xyyaw",
        "valvedriver_tactile_yaw",
    )
    args = SimpleNamespace(task=task, algo="auto", checkpoint="hier_nn/best.pth")
    resolve = _compile_function(
        PLAY_PATH,
        "_resolve_algo",
        {
            "args": args,
            "_HIERARCHICAL_SCREW_TASKS": hierarchical,
            "_is_stage2_checkpoint": lambda path: path.endswith(".ckpt"),
        },
    )
    assert resolve() == "HierarchicalPPO"
    args.algo = "HierarchicalPPO"
    assert resolve() == "HierarchicalPPO"
    args.algo = "PPO"
    with pytest.raises(ValueError, match="requires"):
        resolve()


@pytest.mark.parametrize("stage_dofs", (1, 3))
def test_play_stage0_executes_exactly_zero_stage_channels(tmp_path, stage_dofs):
    """Deterministic playback must reproduce ``HierarchicalPPO.test`` exactly."""
    env = FakeStageEnv(stage_dofs=stage_dofs)
    agent, _ = _make_agent(tmp_path, env=env, stage_dofs=stage_dofs)
    agent.set_eval()
    hierarchical_action = _compile_function(
        PLAY_PATH,
        "_hierarchical_action",
        {"torch": torch, "validate_tactile_latent": lambda value: value},
    )

    obs_dict = env.reset()
    assert agent.current_stage == STAGE_MASTER
    action, executed_stage = hierarchical_action(agent, obs_dict)
    assert action.shape == (env.num_envs, MASTER_ACTION_DIM + stage_dofs)
    assert executed_stage.shape == (env.num_envs, stage_dofs)
    assert torch.equal(executed_stage, torch.zeros_like(executed_stage))
    assert torch.equal(action[:, MASTER_ACTION_DIM:], executed_stage)

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

    # From Stage 1 the follower drives every stage DOF inside the bounds.
    agent.current_stage = STAGE_FOLLOWER
    action, executed_stage = hierarchical_action(agent, obs_dict)
    assert executed_stage.abs().max().item() <= 1.0
    assert not torch.equal(executed_stage, torch.zeros_like(executed_stage))
    # The env never sees a second step for the two policy halves.
    assert env.step_count == 0


def test_play_reports_yaw_in_radians_never_in_millimetres():
    samples = _compile_function(PLAY_PATH, "_yaw_stage_samples", {"torch": torch})
    env_cfg = SimpleNamespace(yaw_position_obs_scale=0.70, yaw_velocity_obs_scale=1.2)
    obs_dict = {
        # Normalized by the fixed 0.70 rad asset limit -> 0.35 rad of rotation.
        "yaw_position": torch.tensor([[0.5]]),
        "yaw_velocity": torch.tensor([[-1.0]]),
        "yaw_target": torch.tensor([[0.6]]),
        "yaw_workspace_margin": torch.tensor([[0.4]]),
    }
    result = samples(obs_dict, torch.tensor([[1.0]]), env_cfg)
    assert result["angle_rad"] == pytest.approx(0.35)
    assert result["angle_deg"] == pytest.approx(math.degrees(0.35), abs=1.0e-4)
    assert result["rate_rad_s"] == pytest.approx(1.2)
    assert result["tracking_error_rad"] == pytest.approx(0.07)
    assert result["workspace_margin"] == pytest.approx(0.4)
    assert result["action_abs"] == pytest.approx(1.0)
    assert result["action_saturation"] == pytest.approx(1.0)
    assert not any("mm" in key for key in result)


def _runtime_metadata(env_cfg):
    attach = _compile_function(
        TRAIN_PATH, "_attach_env_runtime_to_config", {"OmegaConf": OmegaConf}
    )
    full_config = OmegaConf.create({})
    attach(full_config, env_cfg, "pytest_task")
    return OmegaConf.to_container(full_config.env_runtime)


def _metadata_env_cfg(**overrides):
    base = SimpleNamespace(
        sim=SimpleNamespace(dt=1.0 / 120.0),
        decimation=6,
        grasp_cache_path="assets/grasp_cache/hora/pytest",
        tactile_layout="estimated_official",
        action_scale=0.04167,
        observation_space=MASTER_OBS_DIM,
        priv_info_dim=PRIV_INFO_DIM,
        # Empty tuple: skips the tactile metadata block, which is unchanged.
        tactile_active_finger_names=(),
        high_speed_reward_enable=False,
    )
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


def _xy_metadata_fields():
    return dict(
        xy_stage_joint_names=("stage_x_joint", "stage_y_joint"),
        xy_stage_world_axes=("X", "Y"),
        xy_joint_limit=0.05,
        xy_workspace_initial=0.01,
        xy_workspace_final=0.05,
        xy_action_scale_initial=0.002,
        xy_action_scale_final=0.005,
        xy_effort_limit=120.0,
        xy_velocity_limit=0.15,
        xy_acceleration_limit=8.0,
        xy_pgain=8000.0,
        xy_dgain=200.0,
        xy_curriculum_ramp_steps=20_000_000,
    )


def _yaw_metadata_fields():
    defaults = _yaw_cfg_defaults()
    return {
        name: getattr(defaults, name)
        for name in (
            "yaw_joint_limit",
            "yaw_workspace_initial",
            "yaw_workspace_final",
            "yaw_action_scale_initial",
            "yaw_action_scale_final",
            "yaw_effort_limit",
            "yaw_velocity_limit",
            "yaw_acceleration_limit",
            "yaw_pgain",
            "yaw_dgain",
            "yaw_action_smoothing",
            "yaw_curriculum_ramp_steps",
        )
    }


def test_runtime_metadata_separates_xy_and_yaw_action_dimensions():
    env_cfg = _metadata_env_cfg(
        action_space=24,
        finger_action_space=21,
        stage_joint_names=STAGE_JOINT_ORDER,
        **_xy_metadata_fields(),
        **_yaw_metadata_fields(),
    )
    runtime = _runtime_metadata(env_cfg)
    # The critical regression: the three trailing channels are NOT all "xy".
    assert runtime["finger_action_dim"] == 21
    assert runtime["xy_action_dim"] == 2
    assert runtime["yaw_action_dim"] == 1
    assert runtime["follower_action_dim"] == 3
    assert runtime["follower_obs_dim"] == 164
    assert runtime["stage_joint_names"] == list(STAGE_JOINT_ORDER)
    assert runtime["yaw_joint_limit_rad"] == pytest.approx(0.70)
    assert runtime["yaw_workspace_rad"] == [0.05, 0.25]
    assert runtime["yaw_action_scale_rad"] == [0.005, 0.020]
    assert runtime["yaw_effort_limit_nm"] == pytest.approx(1.5)
    assert runtime["yaw_velocity_limit_rad_s"] == pytest.approx(1.2)
    assert runtime["yaw_pgain_nm_per_rad"] == pytest.approx(8.0)
    assert runtime["yaw_curriculum_ramp_steps"] == 20_000_000
    # The XY block is unchanged and still in metres.
    assert runtime["xy_joint_limit_m"] == pytest.approx(0.05)
    assert runtime["xy_workspace_m"] == [0.01, 0.05]


def test_runtime_metadata_for_the_yaw_only_ablation():
    env_cfg = _metadata_env_cfg(
        action_space=22,
        finger_action_space=21,
        stage_joint_names=("stage_yaw_joint",),
        **_yaw_metadata_fields(),
    )
    runtime = _runtime_metadata(env_cfg)
    assert runtime["xy_action_dim"] == 0
    assert runtime["yaw_action_dim"] == 1
    assert runtime["follower_action_dim"] == 1
    assert runtime["follower_obs_dim"] == 154
    assert "xy_joint_limit_m" not in runtime


def test_runtime_metadata_for_the_unchanged_xy_baseline():
    env_cfg = _metadata_env_cfg(
        action_space=23,
        finger_action_space=21,
        xy_stage_joint_names=("stage_x_joint", "stage_y_joint"),
        **{k: v for k, v in _xy_metadata_fields().items() if k != "xy_stage_joint_names"},
    )
    runtime = _runtime_metadata(env_cfg)
    assert runtime["xy_action_dim"] == 2
    assert runtime["yaw_action_dim"] == 0
    assert runtime["follower_action_dim"] == 2
    assert runtime["follower_obs_dim"] == 159


# ---------------------------------------------------------------------------
# 8. environment source contracts
# ---------------------------------------------------------------------------


def test_yaw_env_control_path_uses_named_indices_and_the_shared_helpers():
    """The env must route actions through the audited helpers, not raw slices."""
    tree = ast.parse(YAW_ENV_PATH.read_text(encoding="utf-8"))

    def _body_source(node: ast.FunctionDef) -> str:
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
    assert "super()._pre_physics_step(leading)" in pre_physics
    apply_action = functions["_apply_action"]
    assert "yaw_pd_effort" in apply_action
    assert "joint_ids=self.yaw_dof_indices" in apply_action
    assert "self.yaw_dof_index_tensor" in apply_action
    # Finger torque/work penalties must stay finger-only.
    assert "actuated_dof_indices" not in functions["_compute_yaw_stage_reward"]
    # No root teleport anywhere in the yaw control path.
    for path in (YAW_ENV_PATH, XYYAW_ENV_PATH):
        source = path.read_text(encoding="utf-8")
        assert "write_root_pose_to_sim" not in source
        assert "write_root_state_to_sim" not in source
        assert "set_world_poses" not in source


def test_yaw_is_excluded_from_every_finger_specific_mechanism():
    source = YAW_ENV_PATH.read_text(encoding="utf-8")
    for line in (
        "self.action_mask[self.yaw_dof_index_tensor] = 0.0",
        "self.pose_diff_mask[self.yaw_dof_index_tensor] = 0.0",
        "self.init_joint_pos[:, self.yaw_dof_index_tensor] = 0.0",
    ):
        assert line in source, line


def test_yaw_reset_clears_every_controller_buffer():
    tree = ast.parse(YAW_ENV_PATH.read_text(encoding="utf-8"))
    reset = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_reset_idx"
    )
    source = ast.unparse(reset)
    for buffer in (
        "yaw_target",
        "yaw_delayed_target",
        "yaw_prev_delta",
        "yaw_smoothed_action",
        "yaw_executed_action",
        "yaw_effort",
        "yaw_prev_velocity",
        "yaw_prev_acceleration",
    ):
        assert f"self.{buffer}[env_ids] = 0.0" in source, buffer


def test_yaw_diagnostics_cover_the_documented_tensorboard_tags():
    source = YAW_ENV_PATH.read_text(encoding="utf-8")
    for tag in (
        "yaw/position",
        "yaw/velocity",
        "yaw/target",
        "yaw/tracking_error",
        "yaw/effort",
        "yaw/power",
        "yaw/action_abs",
        "yaw/action_saturation_ratio",
        "yaw/boundary_saturation_ratio",
        "yaw/workspace_utilization",
        "yaw/at_positive_limit_ratio",
        "yaw/at_negative_limit_ratio",
        "curriculum/yaw_workspace",
        "curriculum/yaw_action_scale",
        "curriculum/stage_progress",
    ):
        assert f'"{tag}"' in source, tag
    for cost in ("velocity", "acceleration", "jerk", "effort", "power", "boundary"):
        assert f'"{cost}"' in source, cost
    assert 'f"yaw_cost/{name}"' in source
    assert 'f"yaw_penalty/{name}"' in source


def test_hierarchical_ppo_logs_the_stage_curriculum_tags(tmp_path):
    env = FakeStageEnv(num_envs=4)
    agent, _ = _make_agent(tmp_path, env=env)
    written = {}
    agent.writer.add_scalar = lambda tag, value, step: written.__setitem__(tag, value)
    agent.obs = env.reset()
    curriculum = agent._push_stage_curriculum()
    stats = agent.train_epoch()
    agent.write_stats(stats, curriculum=curriculum)
    for tag in (
        "curriculum/xy_workspace",
        "curriculum/xy_action_scale",
        "curriculum/yaw_workspace",
        "curriculum/yaw_action_scale",
        "curriculum/stage_progress",
        "curriculum/xy_curriculum_progress",
        "curriculum/hierarchical_stage",
        "curriculum/activation_speed_ema",
        "hierarchical/follower_action_dim",
        "hierarchical/master_frozen",
    ):
        assert tag in written, tag
    assert written["hierarchical/follower_action_dim"] == 3


def test_activation_log_message_is_not_xy_specific():
    source = (
        REPO_ROOT
        / "source/BrainCo_DexHand/BrainCo_DexHand/algo/hora/ppo/hierarchical_ppo.py"
    ).read_text(encoding="utf-8")
    assert "activating the XY follower" not in source
    assert "DOF stage follower" in source


def test_follower_model_supports_one_two_and_three_dof_stages():
    for stage_dofs, obs_dim in ((1, 154), (2, 159), (3, 164)):
        follower = FollowerActorCritic(
            obs_dim=obs_dim,
            actions_num=stage_dofs,
            actor_units=(64, 32),
            critic_units=(64, 32),
            critic_priv_dim=11,
        )
        assert follower.mu.out_features == stage_dofs
        assert follower.sigma.shape == (stage_dofs,)
        result = follower.act(torch.randn(5, obs_dim), torch.randn(5, 11))
        assert result["actions"].shape == (5, stage_dofs)


# ---------------------------------------------------------------------------
# 9. USD authoring (requires the USD python bindings)
# ---------------------------------------------------------------------------


def _author_stage(root_rot, num_preceding, yaw_joint_limit=0.70):
    """Run the real authoring code on an in-memory stage and return it."""
    from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics

    hand_usd = REPO_ROOT / "assets/usd/tactile_dexscrew/revo3_right_tactile.usda"
    if not hand_usd.exists():  # pragma: no cover - asset is part of the repo
        pytest.skip("tactile hand USD asset is missing")

    stage = Usd.Stage.CreateInMemory()
    hand_path = "/World/envs/env_0/hand"
    hand = UsdGeom.Xform.Define(stage, hand_path)
    hand.GetPrim().GetReferences().AddReference(str(hand_usd))
    ops = {op.GetOpName(): op for op in hand.GetOrderedXformOps()}
    ops["xformOp:translate"].Set(Gf.Vec3d(0.0, 0.078, 0.195))
    ops["xformOp:orient"].Set(Gf.Quatd(root_rot[0], Gf.Vec3d(*root_rot[1:])))

    cfg = SimpleNamespace(
        robot_cfg=SimpleNamespace(init_state=SimpleNamespace(rot=root_rot)),
        xy_carriage_mass=0.5,
        xy_carriage_inertia=1.0e-3,
        xy_joint_limit=0.05,
        xy_effort_limit=120.0,
        xy_joint_velocity_limit_sim=1.0,
        yaw_joint_limit=yaw_joint_limit,
        yaw_effort_limit=0.30,
        yaw_joint_velocity_limit_sim=3.0,
    )
    fake_env = SimpleNamespace(
        scene=SimpleNamespace(stage=stage, env_prim_paths=["/World/envs/env_0"]),
        cfg=cfg,
        _num_preceding_stage_dofs=lambda: num_preceding,
    )
    functions = _compile_functions(
        YAW_ENV_PATH,
        [
            "_author_robot_stage_overrides",
            "author_stage_carriage",
            "author_stage_prismatic_joint",
            "author_stage_revolute_joint",
        ],
        {
            "Gf": Gf,
            "UsdGeom": UsdGeom,
            "UsdPhysics": UsdPhysics,
            "PhysxSchema": PhysxSchema,
            "radians_to_degrees": YAW_STAGE.radians_to_degrees,
            "NUM_XY_DOFS": XY_STAGE.NUM_XY_DOFS,
            "XY_STAGE_CARRIAGE_BODY_NAME": XY_STAGE.XY_STAGE_CARRIAGE_BODY_NAME,
            "XY_STAGE_JOINT_NAMES": XY_STAGE.XY_STAGE_JOINT_NAMES,
            "XY_STAGE_WORLD_AXES": XY_STAGE.XY_STAGE_WORLD_AXES,
            "YAW_STAGE_CARRIAGE_BODY_NAME": YAW_STAGE.YAW_STAGE_CARRIAGE_BODY_NAME,
            "YAW_STAGE_JOINT_NAME": YAW_STAGE.YAW_STAGE_JOINT_NAME,
            "YAW_STAGE_WORLD_AXIS": YAW_STAGE.YAW_STAGE_WORLD_AXIS,
            "_BASE_LINK_NAME": "right_hand_base_link",
            "_WORLD_LINK_NAME": "world",
            "_BASE_FIXED_JOINT_NAME": "right_hand_base_joint",
            "_JOINTS_SCOPE_NAME": "joints",
        },
    )
    # Let the authoring routine reach the module-level helpers.
    for name, function in functions.items():
        function.__globals__.update(functions)
    functions["_author_robot_stage_overrides"](fake_env)
    return stage, hand_path


def test_xyyaw_usd_chain_is_world_aligned_with_degree_authored_yaw_limits():
    """Author the real 3-DOF chain and inspect the resulting topology."""
    pytest.importorskip("pxr", reason="USD python bindings are not available")
    from pxr import Gf, PhysxSchema, UsdPhysics

    root_rot = (0.37992820, 0.59636781, 0.59636781, -0.37992820)
    stage, hand_path = _author_stage(root_rot, num_preceding=2)

    # The rigid world -> hand weld is gone: no teleport, a real joint chain.
    base_joint = stage.GetPrimAtPath(f"{hand_path}/joints/right_hand_base_joint")
    assert not base_joint.IsValid() or not base_joint.IsActive()

    # Both carriages exist as gravity-free rigid bodies.
    for name in ("stage_x_carriage", "stage_y_carriage"):
        carriage = stage.GetPrimAtPath(f"{hand_path}/{name}")
        assert carriage.IsValid(), name
        assert carriage.HasAPI(UsdPhysics.RigidBodyAPI)
        assert UsdPhysics.MassAPI(carriage).GetMassAttr().Get() == pytest.approx(0.5)
        assert (
            PhysxSchema.PhysxRigidBodyAPI(carriage).GetDisableGravityAttr().Get() is True
        )

    expected_chain = (
        ("stage_x_joint", "X", "world", "stage_x_carriage"),
        ("stage_y_joint", "Y", "stage_x_carriage", "stage_y_carriage"),
        ("stage_yaw_joint", "Z", "stage_y_carriage", "right_hand_base_link"),
    )
    for joint_name, axis, body0, body1 in expected_chain:
        joint_prim = stage.GetPrimAtPath(f"{hand_path}/joints/{joint_name}")
        assert joint_prim.IsValid(), joint_name
        assert [str(p) for p in joint_prim.GetRelationship("physics:body0").GetTargets()] == [
            f"{hand_path}/{body0}"
        ]
        assert [str(p) for p in joint_prim.GetRelationship("physics:body1").GetTargets()] == [
            f"{hand_path}/{body1}"
        ]
        assert joint_prim.GetAttribute("physics:axis").Get() == axis

    # The yaw joint is revolute, limited, and authored in DEGREES.
    yaw_prim = stage.GetPrimAtPath(f"{hand_path}/joints/stage_yaw_joint")
    yaw = UsdPhysics.RevoluteJoint(yaw_prim)
    assert yaw_prim.IsA(UsdPhysics.RevoluteJoint)
    expected_deg = math.degrees(0.70)
    assert yaw.GetLowerLimitAttr().Get() == pytest.approx(-expected_deg, abs=1.0e-4)
    assert yaw.GetUpperLimitAttr().Get() == pytest.approx(expected_deg, abs=1.0e-4)
    # ...and it is a LIMITED joint, never a free continuous one.
    assert expected_deg < 180.0
    # Angular force drive with a finite torque ceiling and zero implicit PD.
    drive = UsdPhysics.DriveAPI(yaw_prim, "angular")
    assert drive.GetTypeAttr().Get() == "force"
    assert drive.GetMaxForceAttr().Get() == pytest.approx(0.30)
    assert drive.GetStiffnessAttr().Get() == pytest.approx(0.0)
    assert drive.GetDampingAttr().Get() == pytest.approx(0.0)
    # Degrees per second for an angular DOF.
    max_velocity = PhysxSchema.PhysxJointAPI(yaw_prim).GetMaxJointVelocityAttr().Get()
    assert max_velocity == pytest.approx(math.degrees(3.0), abs=1.0e-3)

    # The XY joints keep their 120 N linear drive, unaffected by the yaw torque.
    for joint_name in ("stage_x_joint", "stage_y_joint"):
        linear = UsdPhysics.DriveAPI(
            stage.GetPrimAtPath(f"{hand_path}/joints/{joint_name}"), "linear"
        )
        assert linear.GetMaxForceAttr().Get() == pytest.approx(120.0)

    # A positive yaw command rotates the palm about WORLD Z, and the joint
    # anchor is the hand base mount (both local positions are the frame origin).
    body_rotation = Gf.Quatd(root_rot[0], Gf.Vec3d(root_rot[1], root_rot[2], root_rot[3]))
    local_rot = yaw.GetLocalRot0Attr().Get()
    joint_rotation = body_rotation * Gf.Quatd(
        float(local_rot.GetReal()),
        Gf.Vec3d(*[float(v) for v in local_rot.GetImaginary()]),
    )
    rotated = Gf.Rotation(joint_rotation).TransformDir(Gf.Vec3d(0, 0, 1))
    for actual, expected in zip(rotated, (0.0, 0.0, 1.0)):
        assert float(actual) == pytest.approx(float(expected), abs=1.0e-5)
    assert yaw.GetLocalRot0Attr().Get() == yaw.GetLocalRot1Attr().Get()
    assert tuple(yaw.GetLocalPos0Attr().Get()) == (0.0, 0.0, 0.0)
    assert tuple(yaw.GetLocalPos1Attr().Get()) == (0.0, 0.0, 0.0)


def test_yaw_only_usd_chain_connects_world_directly_to_the_hand_mount():
    pytest.importorskip("pxr", reason="USD python bindings are not available")
    from pxr import UsdPhysics

    root_rot = (0.37992820, 0.59636781, 0.59636781, -0.37992820)
    stage, hand_path = _author_stage(root_rot, num_preceding=0)

    for absent in ("stage_x_joint", "stage_y_joint"):
        assert not stage.GetPrimAtPath(f"{hand_path}/joints/{absent}").IsValid()
    for absent in ("stage_x_carriage", "stage_y_carriage"):
        assert not stage.GetPrimAtPath(f"{hand_path}/{absent}").IsValid()

    yaw_prim = stage.GetPrimAtPath(f"{hand_path}/joints/stage_yaw_joint")
    assert yaw_prim.IsA(UsdPhysics.RevoluteJoint)
    assert [str(p) for p in yaw_prim.GetRelationship("physics:body0").GetTargets()] == [
        f"{hand_path}/world"
    ]
    assert [str(p) for p in yaw_prim.GetRelationship("physics:body1").GetTargets()] == [
        f"{hand_path}/right_hand_base_link"
    ]
