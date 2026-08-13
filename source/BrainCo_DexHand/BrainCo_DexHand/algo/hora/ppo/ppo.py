"""PPO trainer for Stage1 teacher policy.

Training loop: collect horizon_length (8) steps × num_envs → GAE returns →
  PPO clipped loss with KL-adaptive learning rate, 5 mini-epochs.

Value bootstrap: when episode truncates (timeout, not termination), the last
  value estimate bootstraps the return to avoid penalizing unfinished episodes.

Minibatch size must divide batch_size (num_envs × horizon) exactly. train.py
  resolves the configured preferred maximum to an exact divisor at runtime.

Tactile runs use the standard single-return PPO path by default, with intrinsic
coordination bonuses already included in the environment reward. The optional
separate coordination GAE path remains available for legacy experiments. Env
extras (scalar means) are logged to TensorBoard via extra_info dict.
"""
import copy
import os
import time
import math
import torch

from BrainCo_DexHand.algo.hora.ppo.experience import ExperienceBuffer
from BrainCo_DexHand.algo.hora.models.models import (
    ActorCritic,
    build_actor_critic_kwargs,
    validate_teacher_tactile_checkpoint_compatibility,
)
from BrainCo_DexHand.algo.hora.models.running_mean_std import RunningMeanStd

from BrainCo_DexHand.algo.hora.utils.misc import AverageScalarMeter, normalize_tensorboard_tag, tprint

from tensorboardX import SummaryWriter


class PPO(object):
    def __init__(self, env, output_dif, full_config):
        self.device = full_config['rl_device']
        self.network_config = full_config.train.network
        self.ppo_config = full_config.train.ppo
        self.separate_coord_advantage = bool(
            self.ppo_config.get('separate_coord_advantage', False)
        )
        self.coord_advantage_coef_initial = float(
            self.ppo_config.get('coord_advantage_coef_initial', 0.2)
        )
        self.coord_advantage_coef_final = float(
            self.ppo_config.get('coord_advantage_coef_final', 0.05)
        )
        self.coord_advantage_curriculum_start = int(
            self.ppo_config.get('coord_advantage_curriculum_start', 0)
        )
        self.coord_advantage_curriculum_end = int(
            self.ppo_config.get('coord_advantage_curriculum_end', 15_000_000)
        )
        self.coord_value_loss_coef = float(
            self.ppo_config.get('coord_value_loss_coef', 1.0)
        )
        if self.coord_advantage_curriculum_end < self.coord_advantage_curriculum_start:
            raise ValueError(
                'coord_advantage_curriculum_end must be >= '
                'coord_advantage_curriculum_start'
            )
        if min(
            self.coord_advantage_coef_initial,
            self.coord_advantage_coef_final,
            self.coord_value_loss_coef,
        ) < 0.0:
            raise ValueError('coordination PPO coefficients must be non-negative')
        # ---- build environment ----
        self.env = env
        self.num_actors = self.ppo_config['num_actors']
        action_space = self.env.action_space
        self.actions_num = action_space.shape[0]
        self.actions_low = torch.from_numpy(action_space.low.copy()).float().to(self.device)
        self.actions_high = torch.from_numpy(action_space.high.copy()).float().to(self.device)
        self.observation_space = self.env.observation_space
        self.obs_shape = self.observation_space.shape
        # ---- Priv Info ----
        self.priv_info_dim = self.ppo_config['priv_info_dim']
        self.priv_info = self.ppo_config['priv_info']
        self.proprio_adapt = self.ppo_config['proprio_adapt']
        # ---- Model ----
        net_config = build_actor_critic_kwargs(
            self.network_config,
            self.ppo_config,
            self.actions_num,
            self.obs_shape,
            self.obs_shape[0] // 3,
            self.proprio_adapt,
            env_cfg=self.env.cfg,
        )
        self.model = ActorCritic(net_config)
        self.model.to(self.device)
        self.coord_value_head = None
        if self.separate_coord_advantage:
            self.coord_value_head = torch.nn.Linear(self.model.units[-1], 1).to(
                self.device
            )
            torch.nn.init.zeros_(self.coord_value_head.weight)
            torch.nn.init.zeros_(self.coord_value_head.bias)
        self.running_mean_std = RunningMeanStd(self.obs_shape).to(self.device)
        self.value_mean_std = RunningMeanStd((1,)).to(self.device)
        self.coord_value_mean_std = (
            RunningMeanStd((1,)).to(self.device)
            if self.separate_coord_advantage
            else None
        )
        # ---- Output Dir ----
        self.output_dir = output_dif
        self.nn_dir = os.path.join(self.output_dir, 'stage1_nn')
        self.tb_dif = os.path.join(self.output_dir, 'stage1_tb')
        os.makedirs(self.nn_dir, exist_ok=True)
        os.makedirs(self.tb_dif, exist_ok=True)
        # ---- Optim ----
        self.last_lr = float(self.ppo_config['learning_rate'])
        self.weight_decay = self.ppo_config.get('weight_decay', 0.0)
        self.tactile_encoder_lr_scale = float(self.ppo_config.get('tactile_encoder_lr_scale', 0.3))
        self.optimizer = self._build_optimizer()
        # ---- PPO Train Param ----
        self.e_clip = self.ppo_config['e_clip']
        self.clip_value = self.ppo_config['clip_value']
        self.entropy_coef = self.ppo_config['entropy_coef']
        self.critic_coef = self.ppo_config['critic_coef']
        self.bounds_loss_coef = self.ppo_config['bounds_loss_coef']
        self.gamma = self.ppo_config['gamma']
        self.tau = self.ppo_config['tau']
        self.truncate_grads = self.ppo_config['truncate_grads']
        self.grad_norm = self.ppo_config['grad_norm']
        self.value_bootstrap = self.ppo_config['value_bootstrap']
        self.normalize_advantage = self.ppo_config['normalize_advantage']
        self.normalize_input = self.ppo_config['normalize_input']
        self.normalize_value = self.ppo_config['normalize_value']
        self.reward_scale = float(self.ppo_config.get('reward_scale', 0.01))
        # ---- PPO Collect Param ----
        self.horizon_length = self.ppo_config['horizon_length']
        self.batch_size = self.horizon_length * self.num_actors
        self.minibatch_size = self.ppo_config['minibatch_size']
        self.mini_epochs_num = self.ppo_config['mini_epochs']
        if not full_config.test and self.batch_size % self.minibatch_size != 0:
            raise ValueError(
                f'PPO batch_size ({self.batch_size}) must be divisible by minibatch_size '
                f'({self.minibatch_size}).'
            )
        # ---- scheduler ----
        self.kl_threshold = self.ppo_config['kl_threshold']
        self.scheduler = AdaptiveScheduler(self.kl_threshold)
        # ---- Snapshot
        self.save_freq = self.ppo_config['save_frequency']
        self.save_best_after = self.ppo_config['save_best_after']
        # ---- Tensorboard Logger ----
        self.extra_info = {}
        writer = SummaryWriter(self.tb_dif)
        self.writer = writer

        self.episode_rewards = AverageScalarMeter(100)
        self.episode_raw_rewards = AverageScalarMeter(100)
        self.episode_lengths = AverageScalarMeter(100)
        self.obs = None
        self.epoch_num = 0
        tactile_hist_shape = None
        if getattr(self.model, "use_tactile_history", False):
            tactile_hist_shape = (
                int(self.model.tactile_history_len),
                int(self.model.tactile_frame_dim),
            )
        self.storage = ExperienceBuffer(
            self.num_actors, self.horizon_length, self.batch_size, self.minibatch_size, self.obs_shape[0],
            self.actions_num, self.priv_info_dim, self.device,
            tactile_hist_shape=tactile_hist_shape,
            separate_coord_advantage=self.separate_coord_advantage,
        )

        batch_size = self.num_actors
        current_rewards_shape = (batch_size, 1)
        self.current_rewards = torch.zeros(current_rewards_shape, dtype=torch.float32, device=self.device)
        self.current_raw_rewards = torch.zeros(current_rewards_shape, dtype=torch.float32, device=self.device)
        self.current_lengths = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
        self.dones = torch.ones((batch_size,), dtype=torch.uint8, device=self.device)
        self.agent_steps = 0
        self.max_agent_steps = self.ppo_config['max_agent_steps']
        self.best_rewards = -10000
        # ---- Timing
        self.data_collect_time = 0
        self.rl_train_time = 0
        self.all_time = 0
        self.coord_gradient_stats = {}

    def _build_optimizer(self):
        """Build Adam with optional lower LR for a separate tactile encoder submodule."""
        tactile_encoder = getattr(self.model, "tactile_encoder", None)
        coord_value_params = (
            list(self.coord_value_head.parameters())
            if self.coord_value_head is not None
            else []
        )
        if tactile_encoder is None or abs(self.tactile_encoder_lr_scale - 1.0) < 1.0e-8:
            return torch.optim.Adam(
                list(self.model.parameters()) + coord_value_params,
                self.last_lr,
                weight_decay=self.weight_decay,
            )
        enc_params = list(tactile_encoder.parameters())
        enc_ids = {id(p) for p in enc_params}
        other_params = [p for p in self.model.parameters() if id(p) not in enc_ids]
        other_params.extend(coord_value_params)
        return torch.optim.Adam(
            [
                {"params": other_params, "lr": self.last_lr},
                {"params": enc_params, "lr": self.last_lr * self.tactile_encoder_lr_scale},
            ],
            weight_decay=self.weight_decay,
        )

    def _set_optimizer_lr(self, base_lr: float):
        """Set param-group learning rates, preserving tactile encoder LR scale."""
        self.last_lr = float(base_lr)
        if len(self.optimizer.param_groups) == 1:
            self.optimizer.param_groups[0]["lr"] = self.last_lr
            return
        self.optimizer.param_groups[0]["lr"] = self.last_lr
        self.optimizer.param_groups[1]["lr"] = self.last_lr * self.tactile_encoder_lr_scale

    def _coord_advantage_coef(self) -> float:
        """Return the stationary-schedule coordination actor coefficient."""
        start = self.coord_advantage_curriculum_start
        end = self.coord_advantage_curriculum_end
        if end <= start:
            progress = 1.0
        else:
            progress = (self.agent_steps - start) / float(end - start)
            progress = min(max(progress, 0.0), 1.0)
        initial = self.coord_advantage_coef_initial
        final = self.coord_advantage_coef_final
        return initial + (final - initial) * progress

    def _coord_values_from_features(self, features: torch.Tensor) -> torch.Tensor:
        """Predict coordination values from the shared actor features.

        Args:
            features: Final shared actor-trunk features.

        Returns:
            Coordination value prediction for each sample.
        """
        if self.coord_value_head is None:
            raise RuntimeError('coordination value head is disabled')
        return self.coord_value_head(features)

    def _record_policy_gradient_stats(
        self,
        task_actor_loss: torch.Tensor,
        coord_actor_loss: torch.Tensor,
        coord_coef: float,
    ) -> None:
        """Measure task and coordination actor-gradient magnitude and alignment.

        Args:
            task_actor_loss: Clipped PPO loss for the task advantage stream.
            coord_actor_loss: Clipped PPO loss for the coordination stream.
            coord_coef: Current coordination actor coefficient.
        """
        policy_params = [
            parameter
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad and not name.startswith('value.')
        ]
        task_grads = torch.autograd.grad(
            task_actor_loss,
            policy_params,
            retain_graph=True,
            allow_unused=True,
        )
        coord_grads = torch.autograd.grad(
            coord_actor_loss * float(coord_coef),
            policy_params,
            retain_graph=False,
            allow_unused=True,
        )
        task_norm_sq = torch.zeros((), device=self.device)
        coord_norm_sq = torch.zeros((), device=self.device)
        dot = torch.zeros((), device=self.device)
        for task_grad, coord_grad in zip(task_grads, coord_grads):
            if task_grad is not None:
                task_norm_sq += task_grad.detach().square().sum()
            if coord_grad is not None:
                coord_norm_sq += coord_grad.detach().square().sum()
            if task_grad is not None and coord_grad is not None:
                dot += (task_grad.detach() * coord_grad.detach()).sum()
        task_norm = torch.sqrt(task_norm_sq)
        coord_norm = torch.sqrt(coord_norm_sq)
        cosine = dot / (task_norm * coord_norm).clamp_min(1.0e-12)
        self.coord_gradient_stats = {
            'task_policy_grad_norm': task_norm.item(),
            'coord_policy_grad_norm': coord_norm.item(),
            'task_coord_grad_cosine': cosine.item(),
        }

    def write_stats(
        self,
        a_losses,
        c_losses,
        b_losses,
        entropies,
        kls,
        coord_a_losses=None,
        coord_c_losses=None,
    ):
        """Write PPO losses and coordination-gradient diagnostics."""
        def _mean_or_none(items):
            if not items:
                return None
            return torch.mean(torch.stack(items)).item()

        self.writer.add_scalar('performance/RLTrainFPS', self.agent_steps / self.rl_train_time, self.agent_steps)
        self.writer.add_scalar('performance/EnvStepFPS', self.agent_steps / self.data_collect_time, self.agent_steps)

        actor_loss = _mean_or_none(a_losses)
        bounds_loss = _mean_or_none(b_losses)
        critic_loss = _mean_or_none(c_losses)
        entropy = _mean_or_none(entropies)
        coord_actor_loss = _mean_or_none(coord_a_losses)
        coord_critic_loss = _mean_or_none(coord_c_losses)
        if actor_loss is not None:
            self.writer.add_scalar('losses/actor_loss', actor_loss, self.agent_steps)
        if bounds_loss is not None:
            self.writer.add_scalar('losses/bounds_loss', bounds_loss, self.agent_steps)
        if critic_loss is not None:
            self.writer.add_scalar('losses/critic_loss', critic_loss, self.agent_steps)
        if entropy is not None:
            self.writer.add_scalar('losses/entropy', entropy, self.agent_steps)
        if coord_actor_loss is not None:
            self.writer.add_scalar(
                'losses/coord_actor_loss', coord_actor_loss, self.agent_steps
            )
        if coord_critic_loss is not None:
            self.writer.add_scalar(
                'losses/coord_critic_loss', coord_critic_loss, self.agent_steps
            )

        self.writer.add_scalar('info/last_lr', self.last_lr, self.agent_steps)
        self.writer.add_scalar('info/e_clip', self.e_clip, self.agent_steps)
        kl_mean = _mean_or_none(kls)
        if kl_mean is not None:
            self.writer.add_scalar('info/kl', kl_mean, self.agent_steps)
        if self.separate_coord_advantage:
            self.writer.add_scalar(
                'coord/advantage_coef',
                self._coord_advantage_coef(),
                self.agent_steps,
            )
            for key, value in self.storage.rollout_stats.items():
                self.writer.add_scalar(f'coord/{key}', value, self.agent_steps)
            for key, value in self.coord_gradient_stats.items():
                self.writer.add_scalar(f'coord/{key}', value, self.agent_steps)
        for k, v in self.extra_info.items():
            self.writer.add_scalar(normalize_tensorboard_tag(k), v, self.agent_steps)

    def set_eval(self):
        self.model.eval()
        if self.coord_value_head is not None:
            self.coord_value_head.eval()
        if self.normalize_input:
            self.running_mean_std.eval()
        if self.normalize_value:
            self.value_mean_std.eval()
            if self.coord_value_mean_std is not None:
                self.coord_value_mean_std.eval()

    def set_train(self):
        self.model.train()
        if self.coord_value_head is not None:
            self.coord_value_head.train()
        if self.normalize_input:
            self.running_mean_std.train()
        if self.normalize_value:
            self.value_mean_std.train()
            if self.coord_value_mean_std is not None:
                self.coord_value_mean_std.train()

    @torch.no_grad()
    def model_act(self, obs_dict):
        processed_obs = self.running_mean_std(obs_dict['obs'])
        input_dict = {
            'obs': processed_obs,
            'priv_info': obs_dict['priv_info'],
        }
        if 'tactile_hist' in obs_dict:
            input_dict['tactile_hist'] = obs_dict['tactile_hist']
        res_dict = self.model.act(input_dict)
        if self.normalize_value:
            res_dict['values'] = self.value_mean_std(res_dict['values'], True)
        if self.separate_coord_advantage:
            coord_values = self._coord_values_from_features(res_dict['features'])
            res_dict['coord_values'] = (
                self.coord_value_mean_std(coord_values, True)
                if self.normalize_value
                else coord_values
            )
        return res_dict

    def _as_per_env_column(
        self,
        value,
        name: str,
        expected_envs: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Convert a rollout signal to ``(num_envs, 1)`` without broadcasting.

        Args:
            value: Tensor or scalar-like per-environment signal.
            name: Signal name used in shape errors.
            expected_envs: Number of parallel environments in this step.
            dtype: Target floating-point dtype.

        Returns:
            A tensor with exactly one value per environment.

        Raises:
            RuntimeError: If the signal is aggregated or has the wrong size.
        """
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value, device=self.device)
        value = value.to(device=self.device, dtype=dtype)
        if value.numel() != expected_envs:
            raise RuntimeError(
                f'{name} must contain one value per environment; '
                f'got shape {tuple(value.shape)} for {expected_envs} environments'
            )
        return value.reshape(expected_envs, 1)

    def train(self):
        _t = time.time()
        _last_t = time.time()
        self.obs = self.env.reset()
        if self.agent_steps == 0:
            self.agent_steps = self.batch_size
        total_iters = max(1, math.ceil(self.max_agent_steps / self.batch_size))

        while self.agent_steps < self.max_agent_steps:
            self.epoch_num += 1
            iter_start_t = time.time()
            (
                a_losses,
                c_losses,
                b_losses,
                entropies,
                kls,
                coord_a_losses,
                coord_c_losses,
                collect_t,
                learn_t,
            ) = self.train_epoch()
            self.storage.data_dict = None

            all_fps = self.agent_steps / (time.time() - _t)
            last_fps = self.batch_size / (time.time() - _last_t)
            _last_t = time.time()
            self.write_stats(
                a_losses,
                c_losses,
                b_losses,
                entropies,
                kls,
                coord_a_losses,
                coord_c_losses,
            )

            mean_rewards = self.episode_rewards.get_mean()
            mean_raw_rewards = self.episode_raw_rewards.get_mean()
            mean_lengths = self.episode_lengths.get_mean()
            self.writer.add_scalar('episode_rewards/step', mean_rewards, self.agent_steps)
            self.writer.add_scalar('episode_rewards_raw/step', mean_raw_rewards, self.agent_steps)
            self.writer.add_scalar('episode_lengths/step', mean_lengths, self.agent_steps)
            checkpoint_name = f'ep_{self.epoch_num}_step_{int(self.agent_steps // 1e6):04}M_reward_{mean_rewards:.2f}'
            info_string = f'Agent Steps: {int(self.agent_steps // 1e6):04}M | FPS: {all_fps:.1f} | ' \
                          f'Last FPS: {last_fps:.1f} | ' \
                          f'Current Best: {self.best_rewards:.2f}'
            tprint(info_string)
            print("", flush=True)
            self._print_epoch_log(
                total_iters=total_iters,
                collect_t=collect_t,
                learn_t=learn_t,
                iter_t=time.time() - iter_start_t,
                elapsed=time.time() - _t,
                mean_rewards=mean_rewards,
                mean_lengths=mean_lengths,
            )
            if self.save_freq > 0 and self.epoch_num % self.save_freq == 0:
                self.save(os.path.join(self.nn_dir, checkpoint_name))
                self.save(os.path.join(self.nn_dir, 'last'))

            if mean_rewards > self.best_rewards and self.epoch_num >= self.save_best_after:
                print(f'save current best reward: {mean_rewards:.2f}')
                self.best_rewards = mean_rewards
                self.save(os.path.join(self.nn_dir, 'best'))

        print('max steps achieved')

    def save(self, name):
        weights = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'agent_steps': int(self.agent_steps),
            'epoch_num': int(self.epoch_num),
            'best_rewards': float(self.best_rewards),
            'last_lr': float(self.last_lr),
        }
        if self.running_mean_std:
            weights['running_mean_std'] = self.running_mean_std.state_dict()
        if self.value_mean_std:
            weights['value_mean_std'] = self.value_mean_std.state_dict()
        if self.coord_value_head is not None:
            weights['coord_value_head'] = self.coord_value_head.state_dict()
            weights['coord_value_mean_std'] = self.coord_value_mean_std.state_dict()
        torch.save(weights, f'{name}.pth')

    def _load_optimizer_state(self, saved_state, has_coord_state):
        """Load optimizer state, dropping a legacy coordination head if needed.

        Tactile configs now use the ordinary single-return PPO objective. Older
        checkpoints appended the two coordination-value parameters to the first
        optimizer group; removing exactly those entries preserves all actor and
        task-critic Adam moments during resume.

        Args:
            saved_state: Serialized optimizer state dictionary.
            has_coord_state: Whether the checkpoint contains a coordination head.
        """
        try:
            self.optimizer.load_state_dict(saved_state)
            return
        except ValueError:
            if self.separate_coord_advantage or not has_coord_state:
                raise

        migrated = copy.deepcopy(saved_state)
        current = self.optimizer.state_dict()
        saved_groups = migrated.get('param_groups', [])
        current_groups = current.get('param_groups', [])
        if len(saved_groups) != len(current_groups):
            raise ValueError(
                'Cannot migrate legacy coordination optimizer state: parameter '
                'group count changed.'
            )

        removed_param_ids = []
        for saved_group, current_group in zip(saved_groups, current_groups):
            excess = len(saved_group['params']) - len(current_group['params'])
            if excess < 0:
                raise ValueError(
                    'Cannot migrate legacy coordination optimizer state: current '
                    'parameter group is larger than the checkpoint group.'
                )
            if excess:
                removed_param_ids.extend(saved_group['params'][-excess:])
                saved_group['params'] = saved_group['params'][:-excess]
        if len(removed_param_ids) != 2:
            raise ValueError(
                'Cannot migrate legacy coordination optimizer state: expected '
                f'exactly 2 obsolete parameters, found {len(removed_param_ids)}.'
            )
        for param_id in removed_param_ids:
            migrated.get('state', {}).pop(param_id, None)
        self.optimizer.load_state_dict(migrated)
        print(
            '[INFO] Migrated optimizer state from the legacy coordination critic '
            'to standard single-return PPO.',
            flush=True,
        )

    def restore_train(self, fn):
        if not fn:
            return
        checkpoint = torch.load(fn, map_location=self.device)
        required_keys = [
            'model',
            'running_mean_std',
            'value_mean_std',
            'optimizer',
            'agent_steps',
            'epoch_num',
            'best_rewards',
            'last_lr',
        ]
        missing = [k for k in required_keys if k not in checkpoint]
        if missing:
            raise RuntimeError(
                f"Strict Stage1 resume failed: missing keys {missing} in checkpoint: {fn}"
            )

        validate_teacher_tactile_checkpoint_compatibility(
            checkpoint['model'],
            self.model.state_dict(),
            checkpoint_path=str(fn),
        )
        self.model.load_state_dict(checkpoint['model'], strict=True)
        self.running_mean_std.load_state_dict(checkpoint['running_mean_std'])
        self.value_mean_std.load_state_dict(checkpoint['value_mean_std'])
        has_coord_state = 'coord_value_head' in checkpoint
        if self.coord_value_head is not None and has_coord_state:
            self.coord_value_head.load_state_dict(checkpoint['coord_value_head'])
            if 'coord_value_mean_std' in checkpoint:
                self.coord_value_mean_std.load_state_dict(
                    checkpoint['coord_value_mean_std']
                )
        if self.separate_coord_advantage and not has_coord_state:
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            except ValueError:
                print(
                    '[WARN] Legacy checkpoint has no coordination critic optimizer '
                    'state; resumed model weights with a fresh optimizer.',
                    flush=True,
                )
        else:
            self._load_optimizer_state(checkpoint['optimizer'], has_coord_state)
        self.agent_steps = int(checkpoint['agent_steps'])
        self.epoch_num = int(checkpoint['epoch_num'])
        self.best_rewards = float(checkpoint['best_rewards'])
        self.last_lr = float(checkpoint['last_lr'])
        if hasattr(self.env, 'common_step_counter'):
            self.env.common_step_counter = self.agent_steps // self.num_actors
        self._set_optimizer_lr(self.last_lr)
        print(
            f"[INFO] Restored train state: agent_steps={self.agent_steps}, "
            f"epoch_num={self.epoch_num}, best_rewards={self.best_rewards:.4f}, lr={self.last_lr:.6g}",
            flush=True,
        )

    def restore_test(self, fn):
        checkpoint = torch.load(fn, map_location=self.device)
        validate_teacher_tactile_checkpoint_compatibility(
            checkpoint['model'],
            self.model.state_dict(),
            checkpoint_path=str(fn),
        )
        self.model.load_state_dict(checkpoint['model'], strict=True)
        if self.normalize_input:
            self.running_mean_std.load_state_dict(checkpoint['running_mean_std'])

    def test(self):
        self.set_eval()
        obs_dict = self.env.reset()
        while True:
            input_dict = {
                'obs': self.running_mean_std(obs_dict['obs']),
                'priv_info': obs_dict['priv_info'],
            }
            if 'tactile_hist' in obs_dict:
                input_dict['tactile_hist'] = obs_dict['tactile_hist']
            mu = self.model.act_inference(input_dict)
            mu = torch.clamp(mu, -1.0, 1.0)
            obs_dict, r, done, info = self.env.step(mu)

    def train_epoch(self):
        _t = time.time()
        self.set_eval()
        self.play_steps()
        collect_t = time.time() - _t
        self.data_collect_time += collect_t
        _t = time.time()
        self.set_train()
        a_losses, b_losses, c_losses = [], [], []
        coord_a_losses, coord_c_losses = [], []
        entropies, kls = [], []
        coord_coef = self._coord_advantage_coef()
        self.coord_gradient_stats = {}

        for mini_epoch in range(0, self.mini_epochs_num):
            ep_kls = []
            for i in range(len(self.storage)):
                batch = self.storage[i]
                value_preds = batch['values']
                old_action_log_probs = batch['neglogpacs']
                task_advantage = batch['task_advantages']
                old_mu = batch['mus']
                old_sigma = batch['sigmas']
                returns = batch['returns']
                actions = batch['actions']
                obs = batch['obses']
                priv_info = batch['priv_info']
                tactile_hist = batch.get('tactile_hist')

                obs = self.running_mean_std(obs)
                batch_dict = {
                    'prev_actions': actions,
                    'obs': obs,
                    'priv_info': priv_info,
                }
                if tactile_hist is not None:
                    batch_dict['tactile_hist'] = tactile_hist
                res_dict = self.model(batch_dict)
                action_log_probs = res_dict['prev_neglogp']
                values = res_dict['values']
                entropy = res_dict['entropy']
                mu = res_dict['mus']
                sigma = res_dict['sigmas']

                # Actor losses remain separate through clipping so beta directly
                # weights the coordination policy-gradient contribution.
                ratio = torch.exp(old_action_log_probs - action_log_probs)
                clipped_ratio = torch.clamp(
                    ratio, 1.0 - self.e_clip, 1.0 + self.e_clip
                )
                task_surr1 = task_advantage * ratio
                task_surr2 = task_advantage * clipped_ratio
                task_a_loss = torch.max(-task_surr1, -task_surr2).mean()
                coord_a_loss = None
                if self.separate_coord_advantage:
                    coord_advantage = batch['coord_advantages']
                    coord_surr1 = coord_advantage * ratio
                    coord_surr2 = coord_advantage * clipped_ratio
                    coord_a_loss = torch.max(-coord_surr1, -coord_surr2).mean()
                    a_loss = task_a_loss + coord_coef * coord_a_loss
                else:
                    a_loss = task_a_loss

                # critic loss
                value_pred_clipped = value_preds + (values - value_preds).clamp(-self.e_clip, self.e_clip)
                value_losses = (values - returns) ** 2
                value_losses_clipped = (value_pred_clipped - returns) ** 2
                task_c_loss = torch.max(value_losses, value_losses_clipped).mean()
                coord_c_loss = None
                if self.separate_coord_advantage:
                    coord_values = self._coord_values_from_features(
                        res_dict['features']
                    )
                    coord_value_preds = batch['coord_values']
                    coord_returns = batch['coord_returns']
                    coord_value_pred_clipped = coord_value_preds + (
                        coord_values - coord_value_preds
                    ).clamp(-self.e_clip, self.e_clip)
                    coord_value_losses = (coord_values - coord_returns) ** 2
                    coord_value_losses_clipped = (
                        coord_value_pred_clipped - coord_returns
                    ) ** 2
                    coord_c_loss = torch.max(
                        coord_value_losses, coord_value_losses_clipped
                    ).mean()
                    c_loss = (
                        task_c_loss + self.coord_value_loss_coef * coord_c_loss
                    )
                else:
                    c_loss = task_c_loss
                # bounded loss
                if self.bounds_loss_coef > 0:
                    soft_bound = 1.1
                    mu_loss_high = torch.clamp_max(mu - soft_bound, 0.0) ** 2
                    mu_loss_low = torch.clamp_max(-mu + soft_bound, 0.0) ** 2
                    b_loss = (mu_loss_low + mu_loss_high).sum(axis=-1)
                else:
                    b_loss = torch.zeros((), device=self.device)
                entropy = entropy.mean()
                b_loss = torch.mean(b_loss)

                loss = a_loss + 0.5 * c_loss * self.critic_coef - entropy * self.entropy_coef + b_loss * self.bounds_loss_coef

                self.optimizer.zero_grad()
                record_gradient_stats = (
                    self.separate_coord_advantage
                    and mini_epoch == 0
                    and i == 0
                )
                # Complete the real PPO backward first.  Some compiled/checkpointed
                # tactile encoders do not support probing a live graph with
                # ``autograd.grad`` before that graph's main backward pass.
                loss.backward(retain_graph=record_gradient_stats)
                if record_gradient_stats:
                    self._record_policy_gradient_stats(
                        task_a_loss,
                        coord_a_loss,
                        coord_coef,
                    )
                if self.truncate_grads:
                    grad_params = list(self.model.parameters())
                    if self.coord_value_head is not None:
                        grad_params.extend(self.coord_value_head.parameters())
                    torch.nn.utils.clip_grad_norm_(grad_params, self.grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    kl_dist = policy_kl(mu.detach(), sigma.detach(), old_mu, old_sigma)

                kl = kl_dist
                a_losses.append(a_loss)
                c_losses.append(c_loss)
                ep_kls.append(kl)
                entropies.append(entropy)
                if coord_a_loss is not None:
                    coord_a_losses.append(coord_a_loss)
                if coord_c_loss is not None:
                    coord_c_losses.append(coord_c_loss)
                if self.bounds_loss_coef is not None:
                    b_losses.append(b_loss)

                self.storage.update_mu_sigma(mu.detach(), sigma.detach())

            if len(ep_kls) == 0:
                av_kls = torch.tensor(0.0, device=self.device)
            else:
                av_kls = torch.mean(torch.stack(ep_kls))
            self.last_lr = self.scheduler.update(self.last_lr, av_kls.item())
            self._set_optimizer_lr(self.last_lr)
            kls.append(av_kls)

        learn_t = time.time() - _t
        self.rl_train_time += learn_t
        return (
            a_losses,
            c_losses,
            b_losses,
            entropies,
            kls,
            coord_a_losses,
            coord_c_losses,
            collect_t,
            learn_t,
        )

    def _print_epoch_log(self, total_iters, collect_t, learn_t, iter_t, elapsed, mean_rewards, mean_lengths):
        width = 100
        pad = 30
        fps = int(self.batch_size / max(1e-6, collect_t + learn_t))
        eta_sec = max(0.0, (total_iters - self.epoch_num) * (elapsed / max(1, self.epoch_num)))

        console_hidden_keys = {
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
            "tactile/coord_positive_reward_budget",
            "tactile/coord_reward_budget_abs_ratio",
            "tactile/coord_intrinsic_weight",
            "tactile/coord_effective_fingers",
            "tactile/coord_axis_efficiency",
            "tactile/coord_load_utility",
            "tactile/coord_effective_torque_normalized",
            "tactile/coord_effective_torque_guide",
            "tactile/coord_effective_torque_reward_weight",
            "tactile/coord_effective_torque_guide_bonus",
            "tactile/coord_quality_weight",
            "tactile/coord_presence_floor",
            "tactile/coord_presence_gate_raw",
            "tactile/coord_presence_gate",
            "tactile/coord_overload",
            "tactile/coord_waste",
            "tactile/coord_penalty_reference",
            "tactile/coord_weight_penalty_ratio",
            "tactile/coord_q_floor",
            "tactile/coord_nonzero_ratio",
        }
        console_hidden_suffixes = (
            "_negative_torque_contribution",
            "_positive_torque_contribution",
            "_signed_axis_torque",
        )

        rew_items = []
        for k in sorted(self.extra_info.keys()):
            if (
                k.startswith("randomization/")
                or k in console_hidden_keys
                or k.endswith(console_hidden_suffixes)
            ):
                continue
            if k.startswith("tactile/coord_") and k not in console_coord_keys:
                continue
            v = self.extra_info[k]
            if isinstance(v, torch.Tensor):
                v = v.item()
            if isinstance(v, (int, float)):
                rew_items.append((k, float(v)))

        header = f" Learning iteration {self.epoch_num}/{total_iters} "
        lines = [
            "#" * width,
            header.center(width, " "),
            "",
            f"{'Computation:':>{pad}} {fps} steps/s (collection: {collect_t:.3f}s, learning: {learn_t:.3f}s)",
            f"{'Mean reward:':>{pad}} {mean_rewards:.4f}",
            f"{'Mean episode length:':>{pad}} {mean_lengths:.4f}",
        ]
        for k, v in rew_items:
            lines.append(f"{k + ':':>{pad}} {v:.6f}")
        lines.extend([
            "-" * width,
            f"{'Total timesteps:':>{pad}} {self.agent_steps}",
            f"{'Iteration time:':>{pad}} {iter_t:.2f}s",
            f"{'Time elapsed:':>{pad}} {time.strftime('%H:%M:%S', time.gmtime(elapsed))}",
            f"{'ETA:':>{pad}} {time.strftime('%H:%M:%S', time.gmtime(eta_sec))}",
        ])
        print("\n".join(lines))

    def play_steps(self):
        for n in range(self.horizon_length):
            res_dict = self.model_act(self.obs)
            # collect o_t
            self.storage.update_data('obses', n, self.obs['obs'])
            self.storage.update_data('priv_info', n, self.obs['priv_info'])
            if 'tactile_hist' in self.storage.storage_dict and 'tactile_hist' in self.obs:
                self.storage.update_data('tactile_hist', n, self.obs['tactile_hist'])
            for k in ['actions', 'neglogpacs', 'values', 'mus', 'sigmas']:
                self.storage.update_data(k, n, res_dict[k].detach())
            if self.separate_coord_advantage:
                self.storage.update_data(
                    'coord_values', n, res_dict['coord_values'].detach()
                )
            # do env step
            actions = torch.clamp(res_dict['actions'], -1.0, 1.0)
            self.obs, rewards, self.dones, infos = self.env.step(actions)
            rewards = rewards.unsqueeze(1)
            assert isinstance(infos, dict), 'Info Should be a Dict'
            # update dones and rewards after env step
            self.storage.update_data('dones', n, self.dones)
            task_rewards = rewards
            coord_rewards = None
            if self.separate_coord_advantage:
                coord_env_enabled = bool(
                    getattr(
                        getattr(self.env, 'cfg', None),
                        'enable_coord_endogenous_reward',
                        False,
                    )
                )
                reward_keys = (
                    'tactile/task_reward_per_env',
                    'tactile/coord_reward_per_env',
                )
                missing_reward_keys = [key for key in reward_keys if key not in infos]
                if coord_env_enabled and missing_reward_keys:
                    raise RuntimeError(
                        'Separate coordination advantage requires per-environment '
                        f'reward info, missing {missing_reward_keys}'
                    )
                expected_envs = rewards.shape[0]
                task_rewards = self._as_per_env_column(
                    infos.get('tactile/task_reward_per_env', rewards.squeeze(1)),
                    'tactile/task_reward_per_env',
                    expected_envs,
                    rewards.dtype,
                )
                coord_rewards = self._as_per_env_column(
                    infos.get(
                        'tactile/coord_reward_per_env',
                        torch.zeros_like(rewards.squeeze(1)),
                    ),
                    'tactile/coord_reward_per_env',
                    expected_envs,
                    rewards.dtype,
                )
            shaped_rewards = self.reward_scale * task_rewards.clone()
            shaped_coord_rewards = (
                self.reward_scale * coord_rewards.clone()
                if coord_rewards is not None
                else None
            )
            if self.value_bootstrap and 'time_outs' in infos:
                time_outs = self._as_per_env_column(
                    infos['time_outs'],
                    'time_outs',
                    shaped_rewards.shape[0],
                    shaped_rewards.dtype,
                )
                shaped_rewards += (
                    self.gamma * res_dict['values'].detach() * time_outs
                )
                if shaped_coord_rewards is not None:
                    shaped_coord_rewards += (
                        self.gamma
                        * res_dict['coord_values'].detach()
                        * time_outs
                    )
            self.storage.update_data('rewards', n, shaped_rewards)
            if shaped_coord_rewards is not None:
                self.storage.update_data(
                    'coord_rewards', n, shaped_coord_rewards
                )

            self.current_rewards += self.reward_scale * rewards
            self.current_raw_rewards += rewards
            self.current_lengths += 1
            done_indices = self.dones.nonzero(as_tuple=False)
            self.episode_rewards.update(self.current_rewards[done_indices])
            self.episode_raw_rewards.update(self.current_raw_rewards[done_indices])
            self.episode_lengths.update(self.current_lengths[done_indices])

            self.extra_info = {}
            for k, v in infos.items():
                # only log scalars
                if isinstance(v, float) or isinstance(v, int) or (isinstance(v, torch.Tensor) and len(v.shape) == 0):
                    if isinstance(v, torch.Tensor):
                        v = v.item()
                    if isinstance(k, str) and k.startswith("rew/"):
                        self.extra_info[k] = float(v) * self.reward_scale
                    else:
                        self.extra_info[k] = v

            not_dones = 1.0 - self.dones.float()

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_raw_rewards = self.current_raw_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones

        res_dict = self.model_act(self.obs)
        last_values = res_dict['values']
        coord_last_values = (
            res_dict['coord_values'] if self.separate_coord_advantage else None
        )

        self.agent_steps += self.batch_size
        self.storage.computer_return(
            last_values,
            self.gamma,
            self.tau,
            coord_last_values=coord_last_values,
        )
        self.storage.prepare_training(
            normalize_advantage=self.normalize_advantage,
            coord_advantage_coef=self._coord_advantage_coef(),
        )

        returns = self.storage.data_dict['returns']
        values = self.storage.data_dict['values']
        if self.normalize_value:
            self.value_mean_std.train()
            values = self.value_mean_std(values)
            returns = self.value_mean_std(returns)
            self.value_mean_std.eval()
        self.storage.data_dict['values'] = values
        self.storage.data_dict['returns'] = returns
        if self.separate_coord_advantage:
            coord_returns = self.storage.data_dict['coord_returns']
            coord_values = self.storage.data_dict['coord_values']
            if self.normalize_value:
                self.coord_value_mean_std.train()
                coord_values = self.coord_value_mean_std(coord_values)
                coord_returns = self.coord_value_mean_std(coord_returns)
                self.coord_value_mean_std.eval()
            self.storage.data_dict['coord_values'] = coord_values
            self.storage.data_dict['coord_returns'] = coord_returns


def policy_kl(p0_mu, p0_sigma, p1_mu, p1_sigma):
    c1 = torch.log(p1_sigma/p0_sigma + 1e-5)
    c2 = (p0_sigma ** 2 + (p1_mu - p0_mu) ** 2) / (2.0 * (p1_sigma ** 2 + 1e-5))
    c3 = -1.0 / 2.0
    kl = c1 + c2 + c3
    kl = kl.sum(dim=-1)
    return kl.mean()
class AdaptiveScheduler(object):
    def __init__(self, kl_threshold=0.008):
        super().__init__()
        self.min_lr = 1e-6
        self.max_lr = 1e-2
        self.kl_threshold = kl_threshold

    def update(self, current_lr, kl_dist):
        lr = current_lr
        if kl_dist > (2.0 * self.kl_threshold):
            lr = max(current_lr / 1.5, self.min_lr)
        if kl_dist < (0.5 * self.kl_threshold):
            lr = min(current_lr * 1.5, self.max_lr)
        return lr
