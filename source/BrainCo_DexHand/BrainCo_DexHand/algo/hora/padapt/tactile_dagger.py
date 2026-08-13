"""Stage2 DAgger distillation for the tactile screw/valve tasks.

Unlike ProprioAdapt (latent MSE onto the shared Stage1 trunk), the student here is a
separate policy restricted to real-robot sensing:

  student obs = proprio history 3 x (21 joint_pos + 21 targets)
              + layout-aware tactile history (grid CNN or physical-node GNN)

Teacher = frozen Stage1 ActorCritic (obs 141 incl. contacts + task privilege and
TacSL state). It may use the legacy flat ``priv_info`` MLP or the explicit
finger-attention-GRU tactile-history encoder, and distills action means:

  L = MSE(student μ, clamp(teacher μ))

Optional curriculum PPO actor and critic auxiliary terms ramp independently
with a Stage2-specific schedule:

  w_actor = progress(agent_steps) * dagger_ppo_actor_coef
  w_critic = progress(agent_steps) * dagger_ppo_critic_coef
  std_cap = lerp(dagger_ppo_std_cap_initial, dagger_ppo_std_cap_final, progress)
  L = L_dagger + w_actor * L_actor + w_critic * L_critic

Keeping the DAgger term at full weight prevents the reward objective from
weakening teacher alignment. The separately smaller critic weight keeps its
high-magnitude target controlled. The legacy ``dagger_ppo_reward_coef`` key
remains an alias for ``dagger_ppo_actor_coef``.

Coordination intrinsic shaping is Stage1-only. TactileDAgger disables it on the
Stage2 environment instance, so student PPO uses the ordinary task/visible
reward while DAgger transfers the teacher's shaped behavior.

The State/Shift queries remain ordered learned summary slots, but no per-slot
semantic claim or loss is imposed. Distillation targets the history encoder's
joint output instead.

Structural tactile channels intentionally omit force magnitude for sim2real.

Stage2 sampling decouples parallel environment count from optimizer batch size.
Each complete rollout is offloaded to CPU before teacher-aligned shuffled epochs;
only one learner microbatch is transferred back to the GPU at a time.
"""
import os
import time
import math
import torch
import torch.nn.functional as F
from termcolor import cprint

from BrainCo_DexHand.algo.hora.utils.misc import AverageScalarMeter, normalize_tensorboard_tag, tprint
from BrainCo_DexHand.algo.hora.models.models import (
    ActorCritic,
    TactileStudentPolicy,
    build_actor_critic_kwargs,
    build_student_tactile_policy_kwargs,
    infer_teacher_tactile_encoder_type_from_state_dict,
    resolve_tactile_encoder_type,
    validate_teacher_tactile_checkpoint_compatibility,
)
from BrainCo_DexHand.algo.hora.models.running_mean_std import RunningMeanStd
from tensorboardX import SummaryWriter


def compute_gae_advantages(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    last_values: torch.Tensor,
    gamma: float,
    tau: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute GAE advantages and returns for a step-major rollout.

    Args:
        rewards: Per-step rewards with shape ``(T, N)``.
        dones: Episode termination flags with shape ``(T, N)``.
        values: Per-step value predictions with shape ``(T, N)``.
        last_values: Bootstrap values after the final step, shape ``(N,)``.
        gamma: Discount factor.
        tau: GAE lambda.

    Returns:
        Advantages and returns, both with shape ``(T, N)``.
    """
    if rewards.ndim != 2 or dones.shape != rewards.shape or values.shape != rewards.shape:
        raise ValueError(
            "GAE inputs must share shape (T, N); got "
            f"rewards={tuple(rewards.shape)}, dones={tuple(dones.shape)}, "
            f"values={tuple(values.shape)}"
        )
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(rewards.shape[1], dtype=rewards.dtype, device=rewards.device)
    horizon = rewards.shape[0]
    for step in reversed(range(horizon)):
        next_values = last_values if step == horizon - 1 else values[step + 1]
        nonterminal = 1.0 - dones[step].float()
        delta = rewards[step] + gamma * next_values * nonterminal - values[step]
        last_gae = delta + gamma * tau * nonterminal * last_gae
        advantages[step] = last_gae
    returns = advantages + values
    return advantages, returns


def normalize_advantages(advantages: torch.Tensor) -> torch.Tensor:
    """Standardize flattened advantages for PPO updates."""
    return (advantages - advantages.mean()) / (advantages.std() + 1e-8)


def resolve_dagger_batching(
    num_actors: int,
    target_rollout_batch_size: int,
    train_batch_size: int,
    minibatch_size: int,
) -> tuple[int, int, int, int]:
    """Resolve rollout, optimizer batch, and microbatch sizes for tactile DAgger.

    The effective batch contains complete parallel environment steps. When the
    target is not divisible by the environment count, it rounds up rather than
    dropping samples.

    Args:
        num_actors: Number of parallel simulation environments.
        target_rollout_batch_size: Preferred new samples per outer iteration.
        train_batch_size: Preferred samples per optimizer update.
        minibatch_size: Maximum samples processed by one forward/backward pass.

    Returns:
        Rollout steps, effective rollout size, optimizer batch size, and
        effective microbatch size.
    """
    if num_actors <= 0:
        raise ValueError(f"num_actors must be positive, got {num_actors}")
    if target_rollout_batch_size <= 0:
        raise ValueError(
            "dagger_rollout_batch_size must be positive, got "
            f"{target_rollout_batch_size}"
        )
    if train_batch_size <= 0:
        raise ValueError(f"dagger_batch_size must be positive, got {train_batch_size}")
    if minibatch_size <= 0:
        raise ValueError(f"dagger_minibatch_size must be positive, got {minibatch_size}")
    requested_train_batch_size = min(train_batch_size, target_rollout_batch_size)
    train_rollout_steps = max(1, math.ceil(requested_train_batch_size / num_actors))
    effective_train_batch_size = num_actors * train_rollout_steps
    updates_per_iteration = max(
        1,
        math.ceil(target_rollout_batch_size / effective_train_batch_size),
    )
    rollout_steps = train_rollout_steps * updates_per_iteration
    effective_rollout_batch_size = effective_train_batch_size * updates_per_iteration
    effective_minibatch_size = min(minibatch_size, effective_train_batch_size)
    return (
        rollout_steps,
        effective_rollout_batch_size,
        effective_train_batch_size,
        effective_minibatch_size,
    )


class TactileDAgger(object):
    def __init__(self, env, output_dir, full_config):
        self.device = full_config['rl_device']
        self.network_config = full_config.train.network
        self.ppo_config = full_config.train.ppo
        # ---- build environment ----
        self.env = env
        self.student_coord_intrinsic_disabled = (
            self._disable_coord_intrinsic_reward_for_student()
        )
        self.num_actors = self.ppo_config['num_actors']
        self.observation_space = self.env.observation_space
        self.obs_shape = self.observation_space.shape
        self.action_space = self.env.action_space
        self.actions_num = self.action_space.shape[0]
        self.priv_info_dim = self.ppo_config['priv_info_dim']
        # ---- student obs dims (from the tactile env cfg, no magic numbers) ----
        env_cfg = self.env.cfg
        self.proprio_hist_len = int(env_cfg.student_proprio_history_len)
        self.proprio_frame_dim = int(env_cfg.student_proprio_frame_dim)
        self.proprio_hist_dim = int(env_cfg.student_proprio_history_dim)
        self.tactile_hist_len = int(env_cfg.student_tactile_history_len)
        self.tactile_frame_dim = int(env_cfg.student_tactile_frame_dim)
        self.tactile_layout = str(getattr(env_cfg, 'tactile_layout', 'regular_grid'))
        self.tactile_graph_total_nodes = int(
            getattr(env_cfg, 'tactile_graph_total_nodes', 0)
        )
        self.tactile_emb_dim = int(env_cfg.student_tactile_encoder_output_dim)
        self.student_obs_dim = int(env_cfg.student_obs_dim)
        self.tactile_distill_coef = float(self.ppo_config.get('tactile_distill_coef', 0.1))
        self.initialize_student_tactile_from_teacher = bool(
            self.ppo_config.get('initialize_student_tactile_from_teacher', False)
        )
        self.dagger_ppo_reward_enable = bool(
            self.ppo_config.get('dagger_ppo_reward_enable', False)
        )
        legacy_ppo_coef = float(self.ppo_config.get('dagger_ppo_reward_coef', 0.2))
        self.dagger_ppo_actor_coef = float(
            self.ppo_config.get('dagger_ppo_actor_coef', legacy_ppo_coef)
        )
        self.dagger_ppo_critic_coef = float(
            self.ppo_config.get(
                'dagger_ppo_critic_coef',
                self.dagger_ppo_actor_coef * 0.1,
            )
        )
        self.dagger_ppo_curriculum_explicit = any(
            key in self.ppo_config
            for key in ('dagger_ppo_curriculum_start', 'dagger_ppo_curriculum_end')
        )
        self.dagger_ppo_curriculum_start = int(
            self.ppo_config.get(
                'dagger_ppo_curriculum_start',
                getattr(env_cfg, 'domain_randomization_curriculum_start', 0),
            )
        )
        self.dagger_ppo_curriculum_end = int(
            self.ppo_config.get(
                'dagger_ppo_curriculum_end',
                getattr(env_cfg, 'domain_randomization_curriculum_end', 0),
            )
        )
        self.dagger_ppo_std_cap_initial = float(
            self.ppo_config.get('dagger_ppo_std_cap_initial', 0.5)
        )
        self.dagger_ppo_std_cap_final = float(
            self.ppo_config.get('dagger_ppo_std_cap_final', 0.2)
        )
        # Backward-compatible alias for old configs and external diagnostics.
        self.dagger_ppo_reward_coef = self.dagger_ppo_actor_coef
        for name, coef in (
            ('dagger_ppo_actor_coef', self.dagger_ppo_actor_coef),
            ('dagger_ppo_critic_coef', self.dagger_ppo_critic_coef),
        ):
            if not 0.0 <= coef <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {coef}")
        if self.dagger_ppo_std_cap_initial <= 0.0 or self.dagger_ppo_std_cap_final <= 0.0:
            raise ValueError(
                "dagger PPO std caps must be positive, got "
                f"{self.dagger_ppo_std_cap_initial} and {self.dagger_ppo_std_cap_final}"
            )
        self.reward_scale = float(self.ppo_config.get('reward_scale', 1.0))
        self.gamma = float(self.ppo_config.get('gamma', 0.99))
        self.tau = float(self.ppo_config.get('tau', 0.95))
        self.e_clip = float(self.ppo_config.get('e_clip', 0.2))
        self.critic_coef = float(self.ppo_config.get('critic_coef', 4.0))
        self.entropy_coef = float(self.ppo_config.get('entropy_coef', 0.0))
        self.bounds_loss_coef = float(self.ppo_config.get('bounds_loss_coef', 0.0))
        self.value_bootstrap = bool(self.ppo_config.get('value_bootstrap', True))
        self.normalize_value = bool(self.ppo_config.get('normalize_value', True))
        self.normalize_advantage = bool(self.ppo_config.get('normalize_advantage', True))
        self.truncate_grads = bool(self.ppo_config.get('truncate_grads', True))
        self.grad_norm = float(self.ppo_config.get('grad_norm', 1.0))
        # ---- teacher: frozen Stage1 ActorCritic, original inputs ----
        teacher_config = build_actor_critic_kwargs(
            self.network_config,
            self.ppo_config,
            self.actions_num,
            self.obs_shape,
            self.obs_shape[0] // 3,
            proprio_adapt=False,
            env_cfg=env_cfg,
        )
        self.teacher = ActorCritic(teacher_config)
        self.teacher.to(self.device)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher_obs_rms = RunningMeanStd(self.obs_shape).to(self.device)
        self.teacher_obs_rms.eval()
        # ---- student ----
        student_config = build_student_tactile_policy_kwargs(
            self.network_config,
            self.actions_num,
            env_cfg,
        )
        self.student = TactileStudentPolicy(student_config)
        self.student.to(self.device)
        # Allow network yaml to override encoder output dim (student_obs_dim follows encoder).
        self.tactile_emb_dim = int(self.student.tactile_emb_dim)
        expected_obs_dim = self.proprio_hist_dim + self.tactile_emb_dim
        assert self.student.obs_dim == expected_obs_dim, \
            f'student net input {self.student.obs_dim} != proprio+tactile {expected_obs_dim}'
        self.student_obs_dim = expected_obs_dim
        self._teacher_has_tactile_latent = False
        self._validate_tactile_mask_alignment(env_cfg)
        # proprio normalizer (online stats); structural tactile stays unnormalized
        self.proprio_mean_std = RunningMeanStd((self.proprio_hist_len, self.proprio_frame_dim)).to(self.device)
        self.proprio_mean_std.train()
        self.value_mean_std = RunningMeanStd((1,)).to(self.device)
        if self.normalize_value:
            self.value_mean_std.train()
        else:
            self.value_mean_std.eval()
        # ---- Output Dir ----
        self.output_dir = output_dir
        self.nn_dir = os.path.join(self.output_dir, 'stage2_nn')
        self.tb_dir = os.path.join(self.output_dir, 'stage2_tb')
        os.makedirs(self.nn_dir, exist_ok=True)
        os.makedirs(self.tb_dir, exist_ok=True)
        self.writer = SummaryWriter(self.tb_dir)
        self.direct_info = {}
        # ---- Misc ----
        target_rollout_batch_size = int(
            self.ppo_config.get('dagger_rollout_batch_size', self.num_actors)
        )
        target_batch_size = int(
            self.ppo_config.get('dagger_batch_size', target_rollout_batch_size)
        )
        requested_minibatch_size = int(
            self.ppo_config.get('dagger_minibatch_size', target_batch_size)
        )
        (
            self.rollout_steps,
            self.rollout_batch_size,
            self.batch_size,
            self.minibatch_size,
        ) = resolve_dagger_batching(
            self.num_actors,
            target_rollout_batch_size,
            target_batch_size,
            requested_minibatch_size,
        )
        self.mini_epochs = int(self.ppo_config.get('dagger_mini_epochs', 1))
        if self.mini_epochs <= 0:
            raise ValueError(
                f"dagger_mini_epochs must be positive, got {self.mini_epochs}"
            )
        self.update_batches_per_iteration = self.rollout_batch_size // self.batch_size
        self.updates_per_iteration = self.update_batches_per_iteration * self.mini_epochs
        self.rollout_steps_per_update = self.rollout_steps // self.update_batches_per_iteration
        self.mean_eps_reward = AverageScalarMeter(window_size=20000)
        self.mean_eps_length = AverageScalarMeter(window_size=20000)
        self.best_rewards = -10000
        self.agent_steps = 0
        self.max_agent_steps = self.ppo_config['max_agent_steps']
        self.save_frequency = int(self.ppo_config.get('save_frequency', 0))
        # ---- Optim: student only (actor MLP + mu head + tactile encoder) ----
        self.optim = torch.optim.Adam(self.student.parameters(), lr=3e-4)
        self._debug_checked = False
        self.step_reward = torch.zeros(self.num_actors, dtype=torch.float32, device=self.device)
        self.step_length = torch.zeros(self.num_actors, dtype=torch.float32, device=self.device)
        cprint(
            '[INFO] TactileDAgger batching: '
            f'{self.num_actors} envs x {self.rollout_steps} rollout steps = '
            f'{self.rollout_batch_size} samples/iteration; '
            f'batch={self.batch_size}, microbatch={self.minibatch_size}, '
            f'epochs={self.mini_epochs}, Adam updates/iteration={self.updates_per_iteration}.',
            'cyan',
        )
        if self.student_coord_intrinsic_disabled:
            cprint(
                '[INFO] Stage2 student coord intrinsic reward disabled; '
                'the frozen teacher retains the behavior learned with Stage1 shaping.',
                'cyan',
            )
        if self.dagger_ppo_reward_enable:
            cprint(
                '[INFO] TactileDAgger curriculum PPO reward enabled: '
                f'max_actor_coef={self.dagger_ppo_actor_coef:.3f}, '
                f'max_critic_coef={self.dagger_ppo_critic_coef:.3f}, '
                f'steps={self.dagger_ppo_curriculum_start}->{self.dagger_ppo_curriculum_end}, '
                f'std_cap={self.dagger_ppo_std_cap_initial:.3f}'
                f'->{self.dagger_ppo_std_cap_final:.3f}.',
                'cyan',
            )

    def _disable_coord_intrinsic_reward_for_student(self) -> bool:
        """Disable Stage1 coordination shaping in this Stage2 environment.

        Returns:
            Whether the environment had coordination intrinsic reward enabled.
        """
        env_layers = []
        env_layer = self.env
        seen_layers = set()
        while env_layer is not None and id(env_layer) not in seen_layers:
            seen_layers.add(id(env_layer))
            env_layers.append(env_layer)
            layer_vars = getattr(env_layer, '__dict__', {})
            env_layer = layer_vars.get('_env', layer_vars.get('env'))

        env_cfg = self.env.cfg
        was_enabled = bool(getattr(env_cfg, 'enable_coord_endogenous_reward', False))
        was_enabled = was_enabled or any(
            bool(getattr(layer, '__dict__', {}).get('_coord_enabled', False))
            for layer in env_layers
        )
        if hasattr(env_cfg, 'enable_coord_endogenous_reward'):
            env_cfg.enable_coord_endogenous_reward = False
        for layer in env_layers:
            if '_coord_enabled' in getattr(layer, '__dict__', {}):
                layer._coord_enabled = False
        return was_enabled

    def _ppo_curriculum_progress(self) -> float:
        """Return the Stage2 PPO curriculum progress in ``[0, 1]``.

        Explicit PPO bounds take precedence over the environment randomization
        bounds. The environment values remain a backward-compatible fallback.

        Returns:
            Interpolation progress from configured curriculum bounds.
        """
        env_cfg = self.env.cfg
        if not getattr(self, 'dagger_ppo_curriculum_explicit', False) and not bool(
            getattr(env_cfg, 'domain_randomization_curriculum_enable', False)
        ):
            return 1.0
        start = int(
            getattr(
                self,
                'dagger_ppo_curriculum_start',
                getattr(env_cfg, 'domain_randomization_curriculum_start', 0),
            )
        )
        end = int(
            getattr(
                self,
                'dagger_ppo_curriculum_end',
                getattr(env_cfg, 'domain_randomization_curriculum_end', 0),
            )
        )
        if end <= start:
            return 1.0
        progress = (self.agent_steps - start) / float(end - start)
        return min(max(progress, 0.0), 1.0)

    def _ppo_std_cap(self) -> float:
        """Return the current upper bound for the student's action std.

        Returns:
            Linearly annealed standard-deviation cap for PPO sampling.
        """
        progress = self._ppo_curriculum_progress()
        initial = getattr(self, 'dagger_ppo_std_cap_initial', 0.5)
        final = getattr(self, 'dagger_ppo_std_cap_final', 0.2)
        return initial + progress * (final - initial)

    def _apply_ppo_std_cap(self, std_cap: float | None) -> None:
        """Project the learnable log standard deviation below its current cap.

        Args:
            std_cap: Positive action standard-deviation cap, or ``None`` when PPO
                is inactive.
        """
        if std_cap is None:
            return
        if std_cap <= 0.0:
            raise ValueError(f"PPO std cap must be positive, got {std_cap}")
        with torch.no_grad():
            self.student.sigma.clamp_(max=math.log(std_cap))

    def _ppo_reward_coef(self) -> float:
        """Return the current actor weight for legacy callers."""
        return self._ppo_loss_coefs()[0]

    def _ppo_loss_coefs(self) -> tuple[float, float]:
        """Return independently scheduled PPO actor and critic weights.

        Returns:
            Current actor and critic weights for the mixed Stage2 objective.
        """
        if not self.dagger_ppo_reward_enable:
            return 0.0, 0.0
        progress = self._ppo_curriculum_progress()
        actor_max = getattr(
            self,
            'dagger_ppo_actor_coef',
            getattr(self, 'dagger_ppo_reward_coef', 0.2),
        )
        critic_max = getattr(self, 'dagger_ppo_critic_coef', actor_max * 0.1)
        return progress * actor_max, progress * critic_max

    def _encoder_num_fingers(self, encoder):
        spatial = getattr(encoder, "spatial", None)
        if spatial is not None and hasattr(spatial, "num_fingers"):
            return int(spatial.num_fingers)
        spatial_encoder = getattr(encoder, "spatial_encoder", None)
        if spatial_encoder is not None and hasattr(spatial_encoder, "num_fingers"):
            return int(spatial_encoder.num_fingers)
        if hasattr(encoder, "num_fingers"):
            return int(encoder.num_fingers)
        return None

    def _validate_tactile_mask_alignment(self, env_cfg) -> None:
        active_names = tuple(getattr(env_cfg, "tactile_active_finger_names", ()))
        expected_fingers = int(len(getattr(env_cfg, "tactile_tip_body_names", ())))
        expected_teacher_frame = int(getattr(env_cfg, "teacher_tactile_frame_dim", 0))
        expected_student_frame = int(getattr(env_cfg, "student_tactile_frame_dim", 0))
        errors = []

        teacher_encoder = getattr(self.teacher, "tactile_encoder", None)
        if teacher_encoder is not None:
            teacher_fingers = self._encoder_num_fingers(teacher_encoder)
            if teacher_fingers is not None and teacher_fingers != expected_fingers:
                errors.append(
                    f"teacher num_fingers={teacher_fingers}, env active fingers={expected_fingers}"
                )
            teacher_frame = int(getattr(self.teacher, "tactile_frame_dim", 0))
            if teacher_frame and teacher_frame != expected_teacher_frame:
                errors.append(
                    f"teacher tactile_frame_dim={teacher_frame}, env teacher frame={expected_teacher_frame}"
                )

        student_encoder = getattr(self.student, "tactile_encoder", None)
        student_frame = int(getattr(student_encoder, "frame_dim", 0))
        if student_frame and student_frame != expected_student_frame:
            errors.append(
                f"student tactile_frame_dim={student_frame}, env student frame={expected_student_frame}"
            )
        student_fingers = self._encoder_num_fingers(student_encoder)
        if student_fingers is not None and student_fingers != expected_fingers:
            errors.append(
                f"student num_fingers={student_fingers}, env active fingers={expected_fingers}"
            )

        if errors:
            detail = "\n  - ".join(errors)
            raise ValueError(
                "Tactile teacher/student active-finger alignment failed.\n"
                f"  active tactile fingers: {', '.join(active_names)}\n"
                f"  - {detail}"
            )

    @torch.no_grad()
    def _initialize_student_encoder_from_teacher(self) -> None:
        """Keep the independently specified student tactile encoder unchanged."""
        if self.initialize_student_tactile_from_teacher:
            cprint(
                '[INFO] Skipped tactile encoder warm start: the Stage1 Teacher and '
                'Stage2 Student use independent architectures.',
                'yellow',
            )

    def set_eval(self):
        self.student.eval()
        self.proprio_mean_std.eval()

    @torch.no_grad()
    def _teacher_actions(self, obs_dict):
        teacher_input = {
            'obs': self.teacher_obs_rms(obs_dict['obs']),
            'priv_info': obs_dict['priv_info'],
        }
        if 'tactile_hist' in obs_dict:
            teacher_input['tactile_hist'] = obs_dict['tactile_hist']
        mu = self.teacher.act_inference(teacher_input)
        return torch.clamp(mu, -1.0, 1.0)

    def test(self):
        self.set_eval()
        obs_dict = self.env.reset()
        while True:
            proprio_hist = self.proprio_mean_std(obs_dict['student_proprio_hist'])
            mu = self.student(proprio_hist, obs_dict['student_tactile_hist'])
            mu = torch.clamp(mu, -1.0, 1.0)
            obs_dict, r, done, info = self.env.step(mu)

    def _update_episode_statistics(self, reward, done) -> None:
        """Update episodic reward and length meters after one environment step."""
        self.step_reward += reward
        self.step_length += 1
        done_indices = done.nonzero(as_tuple=False)
        self.mean_eps_reward.update(self.step_reward[done_indices])
        self.mean_eps_length.update(self.step_length[done_indices])

        not_dones = 1.0 - done.float()
        self.step_reward *= not_dones
        self.step_length *= not_dones

    def _train_student_batch(
        self,
        proprio_hist: torch.Tensor,
        tactile_hist: torch.Tensor,
        teacher_mu: torch.Tensor,
        teacher_latent: torch.Tensor | None,
        ppo_batch: dict[str, torch.Tensor] | None = None,
        ppo_actor_coef: float = 0.0,
        ppo_critic_coef: float = 0.0,
        ppo_std_cap: float | None = None,
    ) -> tuple[float, float, float, float, float, float]:
        """Update the student once using gradient-accumulated microbatches.

        Each microbatch loss is weighted by its sample fraction, producing the
        same mean gradient as one full-batch update while lowering peak learner
        activation memory.

        Args:
            proprio_hist: Normalized student proprioceptive histories.
            tactile_hist: Structural tactile histories.
            teacher_mu: Frozen teacher action targets.
            teacher_latent: Optional frozen teacher tactile latent targets.
            ppo_batch: Optional rollout PPO tensors aligned with ``proprio_hist``.
            ppo_actor_coef: Current curriculum weight on the PPO actor objective.
            ppo_critic_coef: Current curriculum weight on the PPO critic objective.
            ppo_std_cap: Current upper bound for the learnable action std.

        Returns:
            Total loss, action loss, tactile latent loss, PPO actor loss,
            PPO critic loss, and PPO entropy.
        """
        sample_count = int(proprio_hist.shape[0])
        if sample_count <= 0 or sample_count > self.batch_size:
            raise RuntimeError(
                f"DAgger batch has {sample_count} samples, expected 1..{self.batch_size}"
            )
        if tactile_hist.shape[0] != sample_count or teacher_mu.shape[0] != sample_count:
            raise RuntimeError("DAgger batch tensors must have the same leading dimension")
        if teacher_latent is not None and teacher_latent.shape[0] != sample_count:
            raise RuntimeError("DAgger teacher latent batch size does not match observations")
        use_ppo = ppo_actor_coef > 0.0 or ppo_critic_coef > 0.0
        if use_ppo:
            if ppo_batch is None:
                raise RuntimeError("A PPO coefficient is positive but ppo_batch is missing")
            for key, tensor in ppo_batch.items():
                if int(tensor.shape[0]) != sample_count:
                    raise RuntimeError(
                        f"PPO batch key {key!r} has leading dim {tensor.shape[0]}, "
                        f"expected {sample_count}"
                    )

        self._apply_ppo_std_cap(ppo_std_cap)
        self.optim.zero_grad(set_to_none=True)
        loss_mu_value = 0.0
        loss_z_value = 0.0
        loss_ppo_actor_value = 0.0
        loss_ppo_critic_value = 0.0
        loss_ppo_entropy_value = 0.0
        loss_total_value = 0.0
        for start in range(0, sample_count, self.minibatch_size):
            stop = min(sample_count, start + self.minibatch_size)
            sample_weight = (stop - start) / sample_count
            proprio_microbatch = proprio_hist[start:stop].to(
                self.device, non_blocking=True
            )
            tactile_microbatch = tactile_hist[start:stop].to(
                self.device, non_blocking=True
            )
            teacher_mu_microbatch = teacher_mu[start:stop].to(
                self.device, non_blocking=True
            )
            if use_ppo:
                actions = ppo_batch['actions'][start:stop].to(
                    self.device, non_blocking=True
                )
                old_neglogp = ppo_batch['neglogp'][start:stop].to(
                    self.device, non_blocking=True
                )
                old_values = ppo_batch['values'][start:stop].to(
                    self.device, non_blocking=True
                )
                returns = ppo_batch['returns'][start:stop].to(
                    self.device, non_blocking=True
                )
                advantages = ppo_batch['advantages'][start:stop].to(
                    self.device, non_blocking=True
                )
                (
                    mu,
                    z_student,
                    values,
                    neglogp,
                    entropy,
                    _,
                ) = self.student.evaluate_dagger_ppo(
                    proprio_microbatch,
                    tactile_microbatch,
                    actions,
                )
            else:
                mu, z_student = self.student.forward_with_latent(
                    proprio_microbatch,
                    tactile_microbatch,
                )
            loss_mu = F.mse_loss(mu, teacher_mu_microbatch)
            loss_z = torch.zeros((), device=self.device)
            if teacher_latent is not None:
                z_teacher = teacher_latent[start:stop].to(
                    self.device, non_blocking=True
                )
                if z_student.shape != z_teacher.shape or z_student.ndim != 2:
                    raise RuntimeError(
                        "Tactile latent alignment requires matching (B,D) tensors, got "
                        f"student={tuple(z_student.shape)}, teacher={tuple(z_teacher.shape)}"
                    )
                z_student = F.layer_norm(z_student, (z_student.shape[-1],))
                z_teacher = F.layer_norm(z_teacher, (z_teacher.shape[-1],))
                loss_z = F.mse_loss(z_student, z_teacher)
            dagger_loss = loss_mu + self.tactile_distill_coef * loss_z

            if use_ppo:
                ratio = torch.exp(old_neglogp - neglogp)
                surr1 = advantages * ratio
                surr2 = advantages * torch.clamp(
                    ratio, 1.0 - self.e_clip, 1.0 + self.e_clip
                )
                actor_loss = torch.max(-surr1, -surr2).mean()
                value_pred_clipped = old_values + (values - old_values).clamp(
                    -self.e_clip, self.e_clip
                )
                value_losses = (values - returns) ** 2
                value_losses_clipped = (value_pred_clipped - returns) ** 2
                critic_loss = torch.max(value_losses, value_losses_clipped).mean()
                actor_objective = (
                    actor_loss
                    - entropy.mean() * self.entropy_coef
                )
                if self.bounds_loss_coef > 0.0:
                    soft_bound = 1.1
                    mu_loss_high = torch.clamp_max(mu - soft_bound, 0.0) ** 2
                    mu_loss_low = torch.clamp_max(-mu + soft_bound, 0.0) ** 2
                    actor_objective = actor_objective + (
                        (mu_loss_low + mu_loss_high).sum(dim=-1).mean() * self.bounds_loss_coef
                    )
                critic_objective = 0.5 * critic_loss * self.critic_coef
                micro_loss = (
                    dagger_loss
                    + ppo_actor_coef * actor_objective
                    + ppo_critic_coef * critic_objective
                )
                loss_ppo_actor_value += float(actor_loss.detach()) * sample_weight
                loss_ppo_critic_value += float(critic_loss.detach()) * sample_weight
                loss_ppo_entropy_value += float(entropy.mean().detach()) * sample_weight
            else:
                micro_loss = dagger_loss

            (micro_loss * sample_weight).backward()
            loss_total_value += float(micro_loss.detach()) * sample_weight
            loss_mu_value += float(loss_mu.detach()) * sample_weight
            loss_z_value += float(loss_z.detach()) * sample_weight

        if not self._debug_checked:
            self._debug_check_grads()
            self._debug_checked = True
        if self.truncate_grads:
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), self.grad_norm)
        self.optim.step()
        self._apply_ppo_std_cap(ppo_std_cap)
        return (
            loss_total_value,
            loss_mu_value,
            loss_z_value,
            loss_ppo_actor_value,
            loss_ppo_critic_value,
            loss_ppo_entropy_value,
        )

    def _train_student_rollout(
        self,
        proprio_hist: torch.Tensor,
        tactile_hist: torch.Tensor,
        teacher_mu: torch.Tensor,
        teacher_latent: torch.Tensor | None,
        ppo_batch: dict[str, torch.Tensor] | None = None,
        ppo_actor_coef: float = 0.0,
        ppo_critic_coef: float = 0.0,
        ppo_std_cap: float | None = None,
    ) -> tuple[float, float, float, float, float, float]:
        """Train over one rollout using shuffled logical batches and epochs.

        Args:
            proprio_hist: Normalized student proprioceptive rollout histories.
            tactile_hist: Structural tactile rollout histories.
            teacher_mu: Frozen teacher action targets for the rollout.
            teacher_latent: Optional frozen teacher latent targets.
            ppo_batch: Optional flattened PPO rollout tensors.
            ppo_actor_coef: Current curriculum PPO actor weight.
            ppo_critic_coef: Current curriculum PPO critic weight.
            ppo_std_cap: Current upper bound for the learnable action std.

        Returns:
            Sample-weighted mean total, DAgger, and PPO component losses.
        """
        sample_count = int(proprio_hist.shape[0])
        if sample_count <= 0 or sample_count % self.batch_size != 0:
            raise RuntimeError(
                f"DAgger rollout has {sample_count} samples; expected a positive multiple "
                f"of batch_size={self.batch_size}"
            )

        total_loss = 0.0
        total_loss_mu = 0.0
        total_loss_z = 0.0
        total_loss_ppo_actor = 0.0
        total_loss_ppo_critic = 0.0
        total_loss_ppo_entropy = 0.0
        total_weight = 0
        for _ in range(self.mini_epochs):
            permutation = torch.randperm(sample_count, device=proprio_hist.device)
            for start in range(0, sample_count, self.batch_size):
                indices = permutation[start:min(sample_count, start + self.batch_size)]
                latent_batch = (
                    teacher_latent.index_select(0, indices)
                    if teacher_latent is not None
                    else None
                )
                ppo_slice = None
                if ppo_batch is not None:
                    ppo_slice = {
                        key: tensor.index_select(0, indices)
                        for key, tensor in ppo_batch.items()
                    }
                (
                    loss,
                    loss_mu,
                    loss_z,
                    loss_ppo_actor,
                    loss_ppo_critic,
                    loss_ppo_entropy,
                ) = self._train_student_batch(
                    proprio_hist.index_select(0, indices),
                    tactile_hist.index_select(0, indices),
                    teacher_mu.index_select(0, indices),
                    latent_batch,
                    ppo_slice,
                    ppo_actor_coef,
                    ppo_critic_coef,
                    ppo_std_cap,
                )
                batch_count = int(indices.numel())
                total_loss += loss * batch_count
                total_loss_mu += loss_mu * batch_count
                total_loss_z += loss_z * batch_count
                total_loss_ppo_actor += loss_ppo_actor * batch_count
                total_loss_ppo_critic += loss_ppo_critic * batch_count
                total_loss_ppo_entropy += loss_ppo_entropy * batch_count
                total_weight += batch_count
        return (
            total_loss / total_weight,
            total_loss_mu / total_weight,
            total_loss_z / total_weight,
            total_loss_ppo_actor / total_weight,
            total_loss_ppo_critic / total_weight,
            total_loss_ppo_entropy / total_weight,
        )

    def _prepare_ppo_rollout_batch(
        self,
        rewards: list[torch.Tensor],
        dones: list[torch.Tensor],
        values: list[torch.Tensor],
        actions: list[torch.Tensor],
        neglogp: list[torch.Tensor],
        last_values: torch.Tensor,
        time_outs: list[torch.Tensor] | None,
    ) -> dict[str, torch.Tensor]:
        """Flatten a raw-value-scale rollout and compute GAE targets for PPO.

        ``values`` and ``last_values`` must be unnormalized so reward, timeout
        bootstrap, value predictions, and GAE deltas share the same units. The
        running statistics are updated once from return targets, then one fixed
        snapshot normalizes both stored values and returns.
        """
        reward_steps = torch.stack(rewards, dim=0)
        done_steps = torch.stack(dones, dim=0).float()
        value_steps = torch.stack(values, dim=0)
        shaped_rewards = reward_steps * self.reward_scale
        if self.value_bootstrap and time_outs is not None:
            timeout_steps = torch.stack(time_outs, dim=0).float()
            shaped_rewards = shaped_rewards + self.gamma * value_steps * timeout_steps
        advantages, returns = compute_gae_advantages(
            shaped_rewards,
            done_steps,
            value_steps,
            last_values,
            self.gamma,
            self.tau,
        )
        if self.normalize_value:
            self.value_mean_std.train()
            self.value_mean_std(returns.reshape(-1, 1))
            self.value_mean_std.eval()
            value_steps = self.value_mean_std(value_steps.reshape(-1, 1)).reshape(
                value_steps.shape
            )
            returns = self.value_mean_std(returns.reshape(-1, 1)).reshape(
                returns.shape
            )
        advantages_flat = advantages.reshape(-1)
        if self.normalize_advantage:
            advantages_flat = normalize_advantages(advantages_flat)
        return {
            'actions': torch.cat(actions, dim=0),
            'neglogp': torch.cat(neglogp, dim=0),
            'values': value_steps.reshape(-1),
            'returns': returns.reshape(-1),
            'advantages': advantages_flat,
        }

    def _unnormalize_value_predictions(self, values: torch.Tensor) -> torch.Tensor:
        """Convert value-head outputs back to the reward scale without updates.

        Args:
            values: Normalized value predictions from the student value head.

        Returns:
            Value predictions in the same units as environment rewards.
        """
        if not self.normalize_value:
            return values
        self.value_mean_std.eval()
        return self.value_mean_std(
            values.reshape(-1, 1),
            unnorm=True,
        ).reshape(values.shape)

    def train(self):
        _t = time.time()
        _last_t = time.time()
        total_iters = max(1, math.ceil(self.max_agent_steps / self.rollout_batch_size))
        iter_num = 0
        self.student.train()
        self.proprio_mean_std.train()

        obs_dict = self.env.reset()
        while self.agent_steps < self.max_agent_steps:
            iter_num += 1
            iter_start_t = time.time()
            ppo_actor_coef, ppo_critic_coef = self._ppo_loss_coefs()
            use_ppo = ppo_actor_coef > 0.0 or ppo_critic_coef > 0.0
            ppo_std_cap = self._ppo_std_cap() if self.dagger_ppo_reward_enable else None
            self._apply_ppo_std_cap(ppo_std_cap)

            rollout_proprio = []
            rollout_tactile = []
            rollout_teacher_mu = []
            rollout_teacher_latent = []
            rollout_rewards = []
            rollout_dones = []
            rollout_values = []
            rollout_actions = []
            rollout_neglogp = []
            rollout_time_outs = []
            info_totals = {}
            collect_start_t = time.time()
            for rollout_index in range(self.rollout_steps):
                with torch.no_grad():
                    proprio_hist = self.proprio_mean_std(
                        obs_dict['student_proprio_hist'].detach()
                    )
                    tactile_hist = obs_dict['student_tactile_hist'].detach()
                    if rollout_index == 0 and not self._debug_checked:
                        self._debug_check_once(obs_dict, proprio_hist, tactile_hist)
                    if use_ppo:
                        behavior_mu, _, _, values, neglogp = self.student.act(
                            proprio_hist,
                            tactile_hist,
                            deterministic=False,
                        )
                        values = self._unnormalize_value_predictions(values)
                    else:
                        behavior_mu = self.student(proprio_hist, tactile_hist)
                        values = neglogp = None
                    teacher_mu = self._teacher_actions(obs_dict)
                    teacher_latent = None
                    if self.tactile_distill_coef > 0.0 and self._teacher_has_tactile_latent:
                        teacher_latent = self.teacher.get_tactile_latent(
                            obs_dict['priv_info'],
                            tactile_hist=obs_dict.get('tactile_hist'),
                        ).detach()

                rollout_proprio.append(proprio_hist.cpu())
                rollout_tactile.append(tactile_hist.cpu())
                rollout_teacher_mu.append(teacher_mu.cpu())
                if teacher_latent is not None:
                    rollout_teacher_latent.append(teacher_latent.cpu())

                obs_dict, r, done, info = self.env.step(
                    torch.clamp(behavior_mu, -1.0, 1.0)
                )
                self._update_episode_statistics(r, done)
                if use_ppo:
                    rollout_rewards.append(r.detach().cpu())
                    rollout_dones.append(done.detach().cpu())
                    rollout_values.append(values.detach().cpu())
                    rollout_actions.append(behavior_mu.detach().cpu())
                    rollout_neglogp.append(neglogp.detach().cpu())
                    rollout_time_outs.append(
                        info['time_outs'].detach().cpu()
                        if isinstance(info.get('time_outs'), torch.Tensor)
                        else torch.zeros_like(done, dtype=torch.bool).cpu()
                    )
                for key, value in info.items():
                    if isinstance(value, (int, float)) or (
                        isinstance(value, torch.Tensor) and value.numel() == 1
                    ):
                        value_sum, value_count = info_totals.get(key, (0.0, 0))
                        info_totals[key] = (value_sum + float(value), value_count + 1)
            collect_t = time.time() - collect_start_t

            for key, (value_sum, value_count) in info_totals.items():
                self.direct_info[key] = value_sum / value_count

            learn_start_t = time.time()
            proprio_batch = torch.cat(rollout_proprio, dim=0)
            tactile_batch = torch.cat(rollout_tactile, dim=0)
            teacher_mu_batch = torch.cat(rollout_teacher_mu, dim=0)
            teacher_latent_batch = (
                torch.cat(rollout_teacher_latent, dim=0)
                if rollout_teacher_latent
                else None
            )
            ppo_batch = None
            if use_ppo:
                with torch.no_grad():
                    last_proprio = self.proprio_mean_std(
                        obs_dict['student_proprio_hist'].detach()
                    )
                    last_tactile = obs_dict['student_tactile_hist'].detach()
                    _, _, _, last_values, _ = self.student.act(
                        last_proprio,
                        last_tactile,
                        deterministic=True,
                    )
                    last_values = self._unnormalize_value_predictions(last_values)
                ppo_batch = self._prepare_ppo_rollout_batch(
                    rollout_rewards,
                    rollout_dones,
                    rollout_values,
                    rollout_actions,
                    rollout_neglogp,
                    last_values.detach().cpu(),
                    rollout_time_outs,
                )
            del proprio_hist, tactile_hist, teacher_mu, teacher_latent, behavior_mu
            del (
                rollout_proprio,
                rollout_tactile,
                rollout_teacher_mu,
                rollout_teacher_latent,
                rollout_rewards,
                rollout_dones,
                rollout_values,
                rollout_actions,
                rollout_neglogp,
                rollout_time_outs,
            )
            (
                loss_val,
                loss_mu_val,
                loss_z_val,
                loss_ppo_actor_val,
                loss_ppo_critic_val,
                loss_ppo_entropy_val,
            ) = self._train_student_rollout(
                proprio_batch,
                tactile_batch,
                teacher_mu_batch,
                teacher_latent_batch,
                ppo_batch,
                ppo_actor_coef,
                ppo_critic_coef,
                ppo_std_cap,
            )
            del proprio_batch, tactile_batch, teacher_mu_batch, teacher_latent_batch, ppo_batch
            learn_t = time.time() - learn_start_t
            assert math.isfinite(loss_val), f'DAgger loss is not finite: {loss_val}'
            self.direct_info['dagger_loss/mu'] = loss_mu_val
            self.direct_info['dagger_loss/distill_z'] = loss_z_val
            self.direct_info['dagger_loss/distill_coef'] = self.tactile_distill_coef
            # Keep the old tag as an actor-weight alias for existing dashboards.
            self.direct_info['dagger_loss/ppo_coef'] = ppo_actor_coef
            self.direct_info['dagger_loss/ppo_actor_coef'] = ppo_actor_coef
            self.direct_info['dagger_loss/ppo_critic_coef'] = ppo_critic_coef
            self.direct_info['dagger_loss/ppo_std_cap'] = (
                ppo_std_cap if ppo_std_cap is not None else 0.0
            )
            self.direct_info['dagger_loss/ppo_std_mean'] = float(
                torch.exp(self.student.sigma.detach()).mean()
            )
            self.direct_info['dagger_loss/ppo_actor'] = loss_ppo_actor_val
            self.direct_info['dagger_loss/ppo_critic'] = loss_ppo_critic_val
            self.direct_info['dagger_loss/ppo_entropy'] = loss_ppo_entropy_val
            dagger_objective = loss_mu_val + self.tactile_distill_coef * loss_z_val
            self.direct_info['dagger_loss/weighted_dagger'] = dagger_objective
            self.direct_info['dagger_loss/weighted_ppo_actor'] = (
                ppo_actor_coef
                * (
                    loss_ppo_actor_val
                    - self.entropy_coef * loss_ppo_entropy_val
                )
            )
            self.direct_info['dagger_loss/weighted_ppo_critic'] = (
                ppo_critic_coef
                * 0.5
                * self.critic_coef
                * loss_ppo_critic_val
            )
            self.direct_info['curriculum/dagger_ppo_progress'] = (
                self._ppo_curriculum_progress()
            )
            self.agent_steps += self.rollout_batch_size

            self.writer.add_scalar('dagger_loss/step', loss_val, self.agent_steps)
            self.writer.add_scalar('episode_rewards/step', self.mean_eps_reward.get_mean(), self.agent_steps)
            self.writer.add_scalar('episode_lengths/step', self.mean_eps_length.get_mean(), self.agent_steps)
            for k, v in self.direct_info.items():
                self.writer.add_scalar(normalize_tensorboard_tag(k), v, self.agent_steps)

            if self.save_frequency > 0 and iter_num % self.save_frequency == 0:
                step_m = int(self.agent_steps // 1e6)
                self.save(os.path.join(self.nn_dir, f'{step_m:04d}M'))
                self.save(os.path.join(self.nn_dir, 'model_last'))

            mean_rewards = self.mean_eps_reward.get_mean()
            if mean_rewards > self.best_rewards:
                self.save(os.path.join(self.nn_dir, 'model_best'))
                self.best_rewards = mean_rewards

            all_fps = self.agent_steps / (time.time() - _t)
            last_fps = self.rollout_batch_size / (time.time() - _last_t)
            _last_t = time.time()
            info_string = f'Agent Steps: {int(self.agent_steps // 1e6):04}M | FPS: {all_fps:.1f} | ' \
                          f'Last FPS: {last_fps:.1f} | ' \
                          f'DAgger Loss: {loss_val:.6f} | ' \
                          f'Current Best: {self.best_rewards:.2f}'
            tprint(info_string)
            print("", flush=True)
            self._print_epoch_log(
                iter_num=iter_num,
                total_iters=total_iters,
                collect_t=collect_t,
                learn_t=learn_t,
                iter_t=time.time() - iter_start_t,
                elapsed=time.time() - _t,
                mean_rewards=mean_rewards,
                mean_lengths=self.mean_eps_length.get_mean(),
                loss_val=loss_val,
            )

    def _debug_check_once(self, obs_dict, proprio_hist_norm, tactile_hist):
        """One-time shape / value / wiring checks on the first training step."""
        raw_proprio = obs_dict['student_proprio_hist']
        current_struct = tactile_hist[:, -1]
        assert current_struct.shape[-1] == self.tactile_frame_dim
        assert tactile_hist.shape[-2:] == (self.tactile_hist_len, self.tactile_frame_dim)
        assert raw_proprio.shape[-2:] == (self.proprio_hist_len, self.proprio_frame_dim)

        student_obs, tactile_emb = self.student.build_obs(proprio_hist_norm, tactile_hist)
        proprio_flat = proprio_hist_norm.flatten(1)
        assert tactile_emb.shape[-1] == self.tactile_emb_dim
        assert proprio_flat.shape[-1] == self.proprio_hist_dim
        assert torch.isfinite(tactile_emb).all()
        assert torch.isfinite(student_obs).all()

        if self.tactile_layout == 'estimated_official':
            graph_width = self.tactile_graph_total_nodes * 5
            graph_nodes = current_struct[:, :graph_width].view(
                current_struct.shape[0], self.tactile_graph_total_nodes, 5
            )
            contact_b = graph_nodes[..., 0]
            contact_on = graph_nodes[..., 1]
            contact_off = graph_nodes[..., 2]
            duration = graph_nodes[..., 3]
            eta = graph_nodes[..., 4]
            assert contact_on.min() >= 0.0 and contact_on.max() <= 1.0
            assert contact_off.min() >= 0.0 and contact_off.max() <= 1.0
            assert eta.min() >= -1.0 and eta.max() <= 1.0
        else:
            contact_b = current_struct[:, 0::3]
            delta_b = current_struct[:, 1::3]
            duration = current_struct[:, 2::3]
            assert delta_b.min() >= -1.0 and delta_b.max() <= 1.0
        assert contact_b.min() >= 0.0 and contact_b.max() <= 1.0
        assert duration.min() >= 0.0 and duration.max() <= 1.0

        opt_params = {id(p) for group in self.optim.param_groups for p in group['params']}
        enc_params = {id(p) for p in self.student.tactile_encoder.parameters()}
        assert enc_params <= opt_params, 'tactile encoder parameters missing from optimizer'
        assert not any(id(p) in opt_params for p in self.teacher.parameters()), \
            'teacher parameters leaked into the student optimizer'
        assert not any(p.requires_grad for p in self.teacher.parameters())

        print(
            '[DEBUG] TactileDAgger first-step check:\n'
            f'  teacher obs {tuple(obs_dict["obs"].shape)} | priv_info {tuple(obs_dict["priv_info"].shape)}\n'
            f'  student proprio hist {tuple(raw_proprio.shape)} -> flat {tuple(proprio_flat.shape)}\n'
            f'  structural tactile hist {tuple(tactile_hist.shape)} -> embedding {tuple(tactile_emb.shape)}\n'
            f'  fused/actor input {tuple(student_obs.shape)} | tactile_emb_dim {self.tactile_emb_dim}\n'
            f'  tactile layout {self.tactile_layout} | distill_coef {self.tactile_distill_coef} '
            f'| teacher_tactile_latent '
            f'{self._teacher_has_tactile_latent}\n'
            f'  optimizer params {len(opt_params)} (encoder {len(enc_params)}), teacher frozen',
            flush=True,
        )

    def _debug_check_grads(self):
        enc_grads = [p.grad for p in self.student.tactile_encoder.parameters()]
        assert any(g is not None for g in enc_grads), 'tactile encoder received no gradient'
        assert all(torch.isfinite(g).all() for g in enc_grads if g is not None), \
            'tactile encoder gradient is not finite'
        print('[DEBUG] TactileDAgger first-step check: tactile encoder gradients OK', flush=True)

    def restore_train(self, fn):
        if not fn:
            raise ValueError('TactileDAgger requires --checkpoint (Stage1 teacher .pth or Stage2 .ckpt)')
        checkpoint = torch.load(fn, map_location=self.device)
        is_stage2_ckpt = str(fn).endswith('.ckpt') or ('stage2_nn' in str(fn))
        if is_stage2_ckpt:
            required_keys = ['student', 'teacher_model', 'teacher_running_mean_std',
                             'proprio_mean_std', 'optimizer', 'agent_steps', 'best_rewards']
            missing = [k for k in required_keys if k not in checkpoint]
            if missing:
                raise RuntimeError(f'Stage2 resume failed: missing keys {missing} in checkpoint: {fn}')
            self.student.load_state_dict(checkpoint['student'], strict=False)
            validate_teacher_tactile_checkpoint_compatibility(
                checkpoint['teacher_model'],
                self.teacher.state_dict(),
                checkpoint_path=str(fn),
            )
            self.teacher.load_state_dict(checkpoint['teacher_model'], strict=True)
            self.teacher_obs_rms.load_state_dict(checkpoint['teacher_running_mean_std'])
            self.proprio_mean_std.load_state_dict(checkpoint['proprio_mean_std'])
            if 'value_mean_std' in checkpoint:
                self.value_mean_std.load_state_dict(checkpoint['value_mean_std'])
            self.optim.load_state_dict(checkpoint['optimizer'])
            self.agent_steps = int(checkpoint['agent_steps'])
            self.best_rewards = float(checkpoint['best_rewards'])
            print(
                f'[INFO] Resumed tactile DAgger: agent_steps={self.agent_steps}, '
                f'best_rewards={self.best_rewards:.4f}',
                flush=True,
            )
            return

        # Stage1 teacher checkpoint: strict load — the teacher network/input dims are unchanged
        try:
            validate_teacher_tactile_checkpoint_compatibility(
                checkpoint['model'],
                self.teacher.state_dict(),
                checkpoint_path=str(fn),
            )
            self.teacher.load_state_dict(checkpoint['model'], strict=True)
        except (RuntimeError, ValueError) as exc:
            ckpt_teacher = infer_teacher_tactile_encoder_type_from_state_dict(checkpoint['model'])
            cfg_teacher, _ = resolve_tactile_encoder_type(self.network_config)
            raise RuntimeError(
                f"Failed to load Stage1 teacher from {fn}: {exc}\n"
                f"[HINT] Checkpoint teacher tactile_encoder.type={ckpt_teacher!r}, "
                f"but --train_cfg expects {cfg_teacher!r}. "
                f"Teacher yaml must match Stage1 (see run config_*.yaml). "
                f"Examples: Revo3HandScrewTactile / Revo3HandScrewTactileGRU "
                f"(MLP teacher + conv1d/GRU student)."
            ) from exc
        self.teacher_obs_rms.load_state_dict(checkpoint['running_mean_std'])
        self._initialize_student_encoder_from_teacher()
        student_init = (
            'aligned encoder warm-start enabled'
            if self.initialize_student_tactile_from_teacher
            else 'student trains from scratch'
        )
        cprint(f'[INFO] Loaded frozen Stage1 teacher from {fn} ({student_init}).',
               'green', attrs=['bold'])

    def restore_test(self, fn):
        checkpoint = torch.load(fn, map_location=self.device)
        self.student.load_state_dict(checkpoint['student'])
        validate_teacher_tactile_checkpoint_compatibility(
            checkpoint['teacher_model'],
            self.teacher.state_dict(),
            checkpoint_path=str(fn),
        )
        self.teacher.load_state_dict(checkpoint['teacher_model'], strict=True)
        self.teacher_obs_rms.load_state_dict(checkpoint['teacher_running_mean_std'])
        self.proprio_mean_std.load_state_dict(checkpoint['proprio_mean_std'])

    def save(self, name):
        ckpt_path = f'{name}.ckpt'
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        weights = {
            'student': self.student.state_dict(),
            'teacher_model': self.teacher.state_dict(),
            'teacher_running_mean_std': self.teacher_obs_rms.state_dict(),
            'proprio_mean_std': self.proprio_mean_std.state_dict(),
            'optimizer': self.optim.state_dict(),
            'agent_steps': int(self.agent_steps),
            'best_rewards': float(self.best_rewards),
        }
        if self.dagger_ppo_reward_enable:
            weights['value_mean_std'] = self.value_mean_std.state_dict()
        torch.save(weights, ckpt_path)

    def _print_epoch_log(self, iter_num, total_iters, collect_t, learn_t, iter_t, elapsed,
                         mean_rewards, mean_lengths, loss_val):
        width = 100
        pad = 30
        fps = int(self.rollout_batch_size / max(1e-6, collect_t + learn_t))
        eta_sec = max(0.0, (total_iters - iter_num) * (elapsed / max(1, iter_num)))

        rew_items = self._console_direct_items()

        header = f' Learning iteration {iter_num}/{total_iters} '
        lines = [
            '#' * width,
            header.center(width, ' '),
            '',
            f"{'Computation:':>{pad}} {fps} steps/s (collection: {collect_t:.3f}s, learning: {learn_t:.3f}s)",
            f"{'DAgger loss:':>{pad}} {loss_val:.6f}",
            f"{'Mean reward:':>{pad}} {mean_rewards:.4f}",
            f"{'Mean episode length:':>{pad}} {mean_lengths:.4f}",
        ]
        for k, v in rew_items:
            lines.append(f"{k + ':':>{pad}} {v:.6f}")
        lines.extend([
            '-' * width,
            f"{'Total timesteps:':>{pad}} {self.agent_steps}",
            f"{'Iteration time:':>{pad}} {iter_t:.2f}s",
            f"{'Time elapsed:':>{pad}} {time.strftime('%H:%M:%S', time.gmtime(elapsed))}",
            f"{'ETA:':>{pad}} {time.strftime('%H:%M:%S', time.gmtime(eta_sec))}",
        ])
        print('\n'.join(lines))

    def _console_direct_items(self):
        """Return the compact subset of Stage2 diagnostics shown in the terminal."""
        console_hidden_keys = {
            "dagger_loss/distill_coef",
            "randomization/action_delay_mean",
            "randomization/object_xy_offset_norm",
            "screw/index_nut_dist",
            "screw/proximity_finger_dist",
            "screw/thumb_nut_dist",
            "tactile/priv_abs_max",
            "tactile/priv_abs_mean",
            "tactile/dip_physical_contact_ratio",
            "tactile/student_contact_mean",
            "tactile/student_duration_mean",
            "tactile/target_finger_active_ratio",
            "tactile/target_finger_contact_count",
            "tactile/target_finger_visible_progress",
            "tactile/teacher_duration_mean",
            "tactile/visible_contact_ratio",
            "tactile/task_aligned_visible_progress",
            "tactile/visible_contribution_gate",
            "tactile/visible_reward_retention",
            "visible_contact_progress_abs_ema",
            "visible_contact_reward_scale",
            "visible_contact_task_reward_abs_ema",
        }
        console_coord_keys = {
            "tactile/coord_reward",
            "tactile/coord_reward_abs_ratio",
            "tactile/coord_intrinsic_weight",
            "tactile/coord_effective_fingers",
            "tactile/coord_axis_efficiency",
            "tactile/coord_q_floor",
            "tactile/coord_nonzero_ratio",
        }
        console_hidden_suffixes = (
            "_negative_torque_contribution",
            "_positive_torque_contribution",
            "_signed_axis_torque",
        )
        console_dagger_keys = {
            "dagger_loss/mu",
            "dagger_loss/distill_z",
            "dagger_loss/ppo_actor_coef",
            "dagger_loss/ppo_critic_coef",
            "dagger_loss/ppo_std_cap",
            "dagger_loss/ppo_std_mean",
            "dagger_loss/ppo_actor",
            "dagger_loss/ppo_critic",
        }

        items = []
        for key in sorted(self.direct_info.keys()):
            if (
                key.startswith("randomization/")
                or key in console_hidden_keys
                or key.endswith(console_hidden_suffixes)
            ):
                continue
            if key.startswith("tactile/coord_") and key not in console_coord_keys:
                continue
            if key.startswith("dagger_loss/") and key not in console_dagger_keys:
                continue
            value = self.direct_info[key]
            if isinstance(value, (int, float)):
                items.append((key, float(value)))
        return items
