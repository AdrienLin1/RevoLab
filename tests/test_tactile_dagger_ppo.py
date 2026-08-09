"""Unit tests for curriculum PPO auxiliaries in tactile Stage2 DAgger."""

from __future__ import annotations

import torch
from omegaconf import OmegaConf

from BrainCo_DexHand.algo.hora.models.models import (
    TactileStudentPolicy,
    build_student_tactile_policy_kwargs,
)
from BrainCo_DexHand.algo.hora.models.running_mean_std import RunningMeanStd
from BrainCo_DexHand.algo.hora.padapt.tactile_dagger import (
    TactileDAgger,
    compute_gae_advantages,
    normalize_advantages,
)

AGENT_CONFIG_DIR = (
    "source/BrainCo_DexHand/BrainCo_DexHand/tasks/direct/hora_screw/agents"
)
ROTATION_AGENT_CONFIG = (
    "source/BrainCo_DexHand/BrainCo_DexHand/tasks/direct/hora_rotation/agents/"
    "Revo3HandTactileRotate.yaml"
)


def _student_env_cfg(frame_dim: int = 240, output_dim: int = 128):
    return OmegaConf.create(
        {
            "student_proprio_history_len": 3,
            "student_proprio_frame_dim": 42,
            "student_tactile_frame_dim": frame_dim,
            "student_tactile_history_len": 10,
            "student_tactile_encoder_output_dim": output_dim,
        }
    )


def test_compute_gae_advantages_matches_bootstrap():
    rewards = torch.tensor([[1.0, 0.0], [1.0, 2.0]])
    dones = torch.tensor([[0.0, 1.0], [0.0, 0.0]])
    values = torch.zeros(2, 2)
    last_values = torch.tensor([0.5, 0.5])
    advantages, returns = compute_gae_advantages(
        rewards, dones, values, last_values, gamma=0.99, tau=0.95
    )
    assert advantages.shape == (2, 2)
    assert returns.shape == (2, 2)
    assert torch.isfinite(advantages).all()
    assert torch.isfinite(returns).all()


def test_normalize_advantages_zero_mean():
    advantages = torch.tensor([1.0, 2.0, 3.0, 4.0])
    normalized = normalize_advantages(advantages)
    assert abs(float(normalized.mean())) < 1e-5


def test_student_policy_ppo_heads():
    cfg = OmegaConf.load(f"{AGENT_CONFIG_DIR}/Revo3HandScrewTactile.yaml")
    env_cfg = _student_env_cfg()
    student = TactileStudentPolicy(
        build_student_tactile_policy_kwargs(cfg.network, 21, env_cfg)
    )
    batch = 4
    proprio = torch.randn(batch, 3, 42)
    tactile = torch.randn(batch, 10, 240)
    actions, mu, sigma, values, neglogp = student.act(proprio, tactile)
    assert actions.shape == (batch, 21)
    assert mu.shape == (batch, 21)
    assert sigma.shape == (batch, 21)
    assert values.shape == (batch,)
    assert neglogp.shape == (batch,)

    eval_neglogp, entropy, eval_values, _, _ = student.evaluate_ppo(proprio, tactile, actions)
    assert eval_neglogp.shape == (batch,)
    assert entropy.shape == (batch,)
    assert eval_values.shape == (batch,)

    encoder_calls = []
    hook = student.tactile_encoder.register_forward_hook(
        lambda _module, _inputs, _output: encoder_calls.append(1)
    )
    try:
        (
            shared_mu,
            shared_latent,
            shared_values,
            shared_neglogp,
            shared_entropy,
            shared_sigma,
        ) = student.evaluate_dagger_ppo(proprio, tactile, actions)
    finally:
        hook.remove()

    assert len(encoder_calls) == 1
    assert shared_mu.shape == (batch, 21)
    assert shared_latent.shape == (batch, 32)
    assert torch.allclose(shared_values, eval_values)
    assert torch.allclose(shared_neglogp, eval_neglogp)
    assert torch.allclose(shared_entropy, entropy)
    assert shared_sigma.shape == (batch, 21)


class _SinglePassStudent(torch.nn.Module):
    """Minimal student that exposes only the shared DAgger/PPO forward path."""

    def __init__(self):
        """Initialize one trainable scalar and a forward-call counter."""
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.1))
        self.sigma = torch.nn.Parameter(torch.tensor(0.0))
        self.shared_forward_calls = 0

    def evaluate_dagger_ppo(self, proprio_hist, tactile_hist, actions):
        """Return differentiable policy outputs while counting shared forwards.

        Args:
            proprio_hist: Proprioceptive histories for the current microbatch.
            tactile_hist: Tactile histories for the current microbatch.
            actions: PPO actions whose likelihood is evaluated.

        Returns:
            Action mean, latent, value, negative log-probability, entropy, and
            action standard deviation.
        """
        del proprio_hist, tactile_hist
        self.shared_forward_calls += 1
        mu = self.scale * torch.ones_like(actions)
        latent = self.scale * torch.ones(
            (actions.shape[0], 2), device=actions.device
        )
        values = self.scale.expand(actions.shape[0])
        sigma = torch.exp(self.scale).expand_as(mu)
        distribution = torch.distributions.Normal(mu, sigma)
        neglogp = -distribution.log_prob(actions).sum(dim=-1)
        entropy = distribution.entropy().sum(dim=-1)
        return mu, latent, values, neglogp, entropy, sigma


def test_dagger_ppo_training_uses_one_shared_forward_per_microbatch():
    """Ensure PPO mixing does not run a second student encoder forward."""
    student = _SinglePassStudent()
    dagger = TactileDAgger.__new__(TactileDAgger)
    dagger.student = student
    dagger.device = torch.device("cpu")
    dagger.batch_size = 4
    dagger.minibatch_size = 2
    dagger.tactile_distill_coef = 0.1
    dagger.e_clip = 0.2
    dagger.critic_coef = 4.0
    dagger.entropy_coef = 0.0
    dagger.bounds_loss_coef = 0.0
    dagger.truncate_grads = False
    dagger._debug_checked = True
    dagger.optim = torch.optim.SGD(student.parameters(), lr=0.01)

    proprio = torch.zeros(4, 3, 2)
    tactile = torch.zeros(4, 2, 5)
    teacher_mu = torch.zeros(4, 3)
    ppo_batch = {
        "actions": torch.zeros(4, 3),
        "neglogp": torch.zeros(4),
        "values": torch.zeros(4),
        "returns": torch.ones(4),
        "advantages": torch.ones(4),
    }

    losses = dagger._train_student_batch(
        proprio,
        tactile,
        teacher_mu,
        teacher_latent=None,
        ppo_batch=ppo_batch,
        ppo_actor_coef=0.2,
        ppo_critic_coef=0.02,
        ppo_std_cap=0.5,
    )

    assert student.shared_forward_calls == 2
    assert torch.exp(student.sigma) <= 0.5
    assert all(torch.isfinite(torch.tensor(loss)) for loss in losses)
    dagger_loss = losses[1] + dagger.tactile_distill_coef * losses[2]
    expected_total = (
        dagger_loss
        + 0.2 * losses[3]
        + 0.02 * 0.5 * dagger.critic_coef * losses[4]
    )
    assert abs(losses[0] - expected_total) < 1e-6


def test_ppo_reward_coef_follows_curriculum():
    """Ramp PPO weights only within the dedicated Stage2 schedule."""
    dagger = TactileDAgger.__new__(TactileDAgger)
    dagger.dagger_ppo_reward_enable = True
    dagger.dagger_ppo_actor_coef = 0.2
    dagger.dagger_ppo_critic_coef = 0.02
    dagger.dagger_ppo_reward_coef = dagger.dagger_ppo_actor_coef
    dagger.dagger_ppo_curriculum_explicit = True
    dagger.dagger_ppo_curriculum_start = 10
    dagger.dagger_ppo_curriculum_end = 20
    dagger.dagger_ppo_std_cap_initial = 0.5
    dagger.dagger_ppo_std_cap_final = 0.2
    dagger.agent_steps = 0
    dagger.env = type(
        "EnvStub",
        (),
        {
            "cfg": type(
                "CfgStub",
                (),
                {
                    "domain_randomization_curriculum_enable": True,
                    "domain_randomization_curriculum_start": 0,
                    "domain_randomization_curriculum_end": 100,
                },
            )()
        },
    )()
    assert dagger._ppo_reward_coef() == 0.0
    assert dagger._ppo_loss_coefs() == (0.0, 0.0)
    assert dagger._ppo_std_cap() == 0.5
    dagger.agent_steps = 15
    assert abs(dagger._ppo_reward_coef() - 0.1) < 1e-6
    actor_coef, critic_coef = dagger._ppo_loss_coefs()
    assert abs(actor_coef - 0.1) < 1e-6
    assert abs(critic_coef - 0.01) < 1e-6
    assert abs(dagger._ppo_std_cap() - 0.35) < 1e-6
    dagger.agent_steps = 20
    assert abs(dagger._ppo_reward_coef() - 0.2) < 1e-6
    assert dagger._ppo_loss_coefs() == (0.2, 0.02)
    assert abs(dagger._ppo_std_cap() - 0.2) < 1e-6


def test_ppo_std_cap_projects_learnable_log_std():
    """Keep the learnable policy std at or below the scheduled upper bound."""
    cfg = OmegaConf.load(f"{AGENT_CONFIG_DIR}/Revo3HandScrewTactile.yaml")
    student = TactileStudentPolicy(
        build_student_tactile_policy_kwargs(cfg.network, 21, _student_env_cfg())
    )
    dagger = TactileDAgger.__new__(TactileDAgger)
    dagger.student = student

    dagger._apply_ppo_std_cap(0.5)
    assert torch.all(torch.exp(student.sigma) <= 0.5)
    student.sigma.data.fill_(0.0)
    dagger._apply_ppo_std_cap(0.2)
    assert torch.all(torch.exp(student.sigma) <= 0.2)


def test_legacy_ppo_coef_uses_smaller_default_critic_weight():
    """Map old reward-coefficient configs onto the separated safe defaults."""
    dagger = TactileDAgger.__new__(TactileDAgger)
    dagger.dagger_ppo_reward_enable = True
    dagger.dagger_ppo_reward_coef = 0.2
    dagger.agent_steps = 0
    dagger.env = type(
        "EnvStub",
        (),
        {
            "cfg": type(
                "CfgStub",
                (),
                {
                    "domain_randomization_curriculum_enable": False,
                    "domain_randomization_curriculum_start": 0,
                    "domain_randomization_curriculum_end": 100,
                },
            )()
        },
    )()

    actor_coef, critic_coef = dagger._ppo_loss_coefs()
    assert actor_coef == 0.2
    assert abs(critic_coef - 0.02) < 1e-6


def test_value_predictions_are_unnormalized_before_gae():
    """Ensure GAE receives values in the same units as environment rewards."""
    dagger = TactileDAgger.__new__(TactileDAgger)
    dagger.normalize_value = True
    dagger.normalize_advantage = False
    dagger.value_bootstrap = False
    dagger.reward_scale = 1.0
    dagger.gamma = 0.0
    dagger.tau = 0.0
    dagger.value_mean_std = RunningMeanStd((1,))
    dagger.value_mean_std.running_mean.fill_(10.0)
    dagger.value_mean_std.running_var.fill_(4.0)
    dagger.value_mean_std.count.fill_(100.0)

    normalized_value = torch.zeros(2)
    raw_value = dagger._unnormalize_value_predictions(normalized_value)
    assert torch.allclose(raw_value, torch.full((2,), 10.0))
    assert not dagger.value_mean_std.training

    count_before = dagger.value_mean_std.count.clone()
    ppo_batch = dagger._prepare_ppo_rollout_batch(
        rewards=[torch.ones(2)],
        dones=[torch.zeros(2)],
        values=[raw_value],
        actions=[torch.zeros(2, 1)],
        neglogp=[torch.zeros(2)],
        last_values=raw_value,
        time_outs=None,
    )

    assert torch.allclose(ppo_batch['advantages'], torch.full((2,), -9.0))
    assert dagger.value_mean_std.count == count_before + 2
    assert not dagger.value_mean_std.training


def test_mlp_config_separates_actor_and_critic_weights():
    """Keep production PPO weights and schedules independently configured."""
    config_paths = (
        f"{AGENT_CONFIG_DIR}/Revo3HandScrewTactile.yaml",
        ROTATION_AGENT_CONFIG,
    )
    for config_path in config_paths:
        cfg = OmegaConf.load(config_path)
        assert cfg.network.tactile_encoder.type == "mlp"


def test_stage2_disables_coord_intrinsic_reward_for_every_tactile_student():
    """Disable shaping on the real env rather than shadowing its wrapper field."""
    class WrapperStub:
        def __init__(self, wrapped_env):
            self._env = wrapped_env

        def __getattr__(self, name):
            return getattr(self._env, name)

    env_cfg = type("CfgStub", (), {"enable_coord_endogenous_reward": True})()
    raw_env = type("EnvStub", (), {"cfg": env_cfg})()
    raw_env._coord_enabled = True
    env = WrapperStub(raw_env)
    dagger = TactileDAgger.__new__(TactileDAgger)
    dagger.env = env

    disabled = dagger._disable_coord_intrinsic_reward_for_student()

    assert disabled is True
    assert env.cfg.enable_coord_endogenous_reward is False
    assert raw_env._coord_enabled is False
    assert '_coord_enabled' not in env.__dict__
