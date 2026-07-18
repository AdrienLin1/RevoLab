"""Stage2 DAgger distillation for the tactile screw/valve tasks.

Unlike ProprioAdapt (latent MSE onto the shared Stage1 trunk), the student here is a
separate policy restricted to real-robot sensing:

  student obs (366) = proprio history 3 x (21 joint_pos + 21 targets) = 126
                    + TactileHistoryEncoder(10 x 240 binary tactile)  = 240

Teacher = frozen Stage1 ActorCritic (obs 141 incl. contacts + priv_info incl. TacSL
tactile). Online DAgger: the student acts, the teacher labels every visited state,
loss = MSE(student mu, clamp(teacher mu)). Only student parameters are optimized.

Normalization: proprio history uses an online RunningMeanStd (sa_mean_std style);
binary tactile enters the encoder unnormalized to preserve {0,1} semantics.

Checkpoint: .ckpt with student + teacher + both normalizers, so stage2 resume/test
does not need the Stage1 .pth.
"""
import os
import time
import math
import torch
from termcolor import cprint

from BrainCo_DexHand.algo.hora.utils.misc import AverageScalarMeter, tprint
from BrainCo_DexHand.algo.hora.models.models import ActorCritic, TactileStudentPolicy
from BrainCo_DexHand.algo.hora.models.running_mean_std import RunningMeanStd
from tensorboardX import SummaryWriter


class TactileDAgger(object):
    def __init__(self, env, output_dir, full_config):
        self.device = full_config['rl_device']
        self.network_config = full_config.train.network
        self.ppo_config = full_config.train.ppo
        # ---- build environment ----
        self.env = env
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
        self.tactile_emb_dim = int(env_cfg.student_tactile_encoder_output_dim)
        self.student_obs_dim = int(env_cfg.student_obs_dim)
        # ---- teacher: frozen Stage1 ActorCritic, original inputs ----
        teacher_config = {
            'actor_units': self.network_config.mlp.units,
            'priv_mlp_units': self.network_config.priv_mlp.units,
            'actions_num': self.actions_num,
            'input_shape': self.obs_shape,
            'priv_info': True,
            'proprio_adapt': False,
            'priv_info_dim': self.priv_info_dim,
        }
        self.teacher = ActorCritic(teacher_config)
        self.teacher.to(self.device)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher_obs_rms = RunningMeanStd(self.obs_shape).to(self.device)
        self.teacher_obs_rms.eval()
        # ---- student ----
        student_config = {
            'actor_units': self.network_config.mlp.units,
            'actions_num': self.actions_num,
            'proprio_hist_len': self.proprio_hist_len,
            'proprio_frame_dim': self.proprio_frame_dim,
            'tactile_frame_dim': self.tactile_frame_dim,
            'tactile_hist_len': self.tactile_hist_len,
            'tactile_emb_dim': self.tactile_emb_dim,
        }
        self.student = TactileStudentPolicy(student_config)
        self.student.to(self.device)
        assert self.student.obs_dim == self.student_obs_dim, \
            f'student net input {self.student.obs_dim} != cfg student_obs_dim {self.student_obs_dim}'
        # proprio normalizer (online stats, sa_mean_std style); tactile stays binary
        self.proprio_mean_std = RunningMeanStd((self.proprio_hist_len, self.proprio_frame_dim)).to(self.device)
        self.proprio_mean_std.train()
        # ---- Output Dir ----
        self.output_dir = output_dir
        self.nn_dir = os.path.join(self.output_dir, 'stage2_nn')
        self.tb_dir = os.path.join(self.output_dir, 'stage2_tb')
        os.makedirs(self.nn_dir, exist_ok=True)
        os.makedirs(self.tb_dir, exist_ok=True)
        self.writer = SummaryWriter(self.tb_dir)
        self.direct_info = {}
        # ---- Misc ----
        self.batch_size = self.num_actors
        self.mean_eps_reward = AverageScalarMeter(window_size=20000)
        self.mean_eps_length = AverageScalarMeter(window_size=20000)
        self.best_rewards = -10000
        self.agent_steps = 0
        self.max_agent_steps = self.ppo_config['max_agent_steps']
        self.save_frequency = int(self.ppo_config.get('save_frequency', 0))
        # ---- Optim: student only (actor MLP + mu head + tactile encoder) ----
        self.optim = torch.optim.Adam(self.student.parameters(), lr=3e-4)
        self._debug_checked = False
        self.step_reward = torch.zeros(self.batch_size, dtype=torch.float32, device=self.device)
        self.step_length = torch.zeros(self.batch_size, dtype=torch.float32, device=self.device)

    def set_eval(self):
        self.student.eval()
        self.proprio_mean_std.eval()

    @torch.no_grad()
    def _teacher_actions(self, obs_dict):
        teacher_input = {
            'obs': self.teacher_obs_rms(obs_dict['obs']),
            'priv_info': obs_dict['priv_info'],
        }
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

    def train(self):
        _t = time.time()
        _last_t = time.time()
        total_iters = max(1, math.ceil(self.max_agent_steps / self.batch_size))
        iter_num = 0
        self.student.train()
        self.proprio_mean_std.train()

        obs_dict = self.env.reset()
        while self.agent_steps < self.max_agent_steps:
            iter_num += 1
            iter_start_t = time.time()

            learn_start_t = time.time()
            proprio_hist = self.proprio_mean_std(obs_dict['student_proprio_hist'].detach())
            tactile_hist = obs_dict['student_tactile_hist'].detach()
            if not self._debug_checked:
                self._debug_check_once(obs_dict, proprio_hist, tactile_hist)
            mu = self.student(proprio_hist, tactile_hist)
            teacher_mu = self._teacher_actions(obs_dict)
            loss = ((mu - teacher_mu) ** 2).mean()
            self.optim.zero_grad()
            loss.backward()
            if not self._debug_checked:
                self._debug_check_grads()
                self._debug_checked = True
            self.optim.step()
            learn_t = time.time() - learn_start_t

            loss_val = loss.item()
            assert math.isfinite(loss_val), f'DAgger loss is not finite: {loss_val}'

            mu = torch.clamp(mu.detach(), -1.0, 1.0)
            collect_start_t = time.time()
            obs_dict, r, done, info = self.env.step(mu)
            for k, v in info.items():
                if isinstance(v, (int, float)) or (isinstance(v, torch.Tensor) and v.numel() == 1):
                    self.direct_info[k] = float(v)
            collect_t = time.time() - collect_start_t
            self.agent_steps += self.batch_size

            # ---- statistics
            self.step_reward += r
            self.step_length += 1
            done_indices = done.nonzero(as_tuple=False)
            self.mean_eps_reward.update(self.step_reward[done_indices])
            self.mean_eps_length.update(self.step_length[done_indices])

            not_dones = 1.0 - done.float()
            self.step_reward = self.step_reward * not_dones
            self.step_length = self.step_length * not_dones

            self.writer.add_scalar('dagger_loss/step', loss_val, self.agent_steps)
            self.writer.add_scalar('episode_rewards/step', self.mean_eps_reward.get_mean(), self.agent_steps)
            self.writer.add_scalar('episode_lengths/step', self.mean_eps_length.get_mean(), self.agent_steps)
            for k, v in self.direct_info.items():
                self.writer.add_scalar(f'{k}/frame', v, self.agent_steps)

            if self.save_frequency > 0 and iter_num % self.save_frequency == 0:
                step_m = int(self.agent_steps // 1e6)
                self.save(os.path.join(self.nn_dir, f'{step_m:04d}M'))
                self.save(os.path.join(self.nn_dir, 'model_last'))

            mean_rewards = self.mean_eps_reward.get_mean()
            if mean_rewards > self.best_rewards:
                self.save(os.path.join(self.nn_dir, 'model_best'))
                self.best_rewards = mean_rewards

            all_fps = self.agent_steps / (time.time() - _t)
            last_fps = self.batch_size / (time.time() - _last_t)
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
        """One-time shape / binary-value / wiring checks on the first training step."""
        num_envs = tactile_hist.shape[0]
        raw_proprio = obs_dict['student_proprio_hist']
        current_binary_tactile = tactile_hist[:, -1]
        assert current_binary_tactile.shape[-1] == self.tactile_frame_dim
        assert tactile_hist.shape[-2:] == (self.tactile_hist_len, self.tactile_frame_dim)
        assert tactile_hist.reshape(num_envs, -1).shape[-1] == self.tactile_hist_len * self.tactile_frame_dim
        assert raw_proprio.shape[-2:] == (self.proprio_hist_len, self.proprio_frame_dim)

        student_obs, tactile_emb = self.student.build_obs(proprio_hist_norm, tactile_hist)
        proprio_flat = proprio_hist_norm.flatten(1)
        assert tactile_emb.shape[-1] == self.tactile_emb_dim
        assert proprio_flat.shape[-1] == self.proprio_hist_dim
        assert student_obs.shape[-1] == self.student_obs_dim
        assert student_obs.shape[-1] == proprio_flat.shape[-1] + tactile_emb.shape[-1]
        assert torch.isfinite(tactile_emb).all()
        assert torch.isfinite(student_obs).all()

        assert tactile_hist.min() >= 0 and tactile_hist.max() <= 1
        sample = tactile_hist[: min(num_envs, 64)]
        sampled_unique = torch.unique(sample)
        assert all(v in (0.0, 1.0) for v in sampled_unique.tolist()), \
            f'binary tactile contains non-binary values: {sampled_unique.tolist()}'

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
            f'  raw tactile hist {tuple(tactile_hist.shape)} -> embedding {tuple(tactile_emb.shape)}\n'
            f'  student obs {tuple(student_obs.shape)} | student net input dim {self.student.obs_dim}\n'
            f'  tactile min/max {tactile_hist.min().item():.1f}/{tactile_hist.max().item():.1f} '
            f'| sampled unique {sampled_unique.tolist()}\n'
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
            self.student.load_state_dict(checkpoint['student'], strict=True)
            self.teacher.load_state_dict(checkpoint['teacher_model'], strict=True)
            self.teacher_obs_rms.load_state_dict(checkpoint['teacher_running_mean_std'])
            self.proprio_mean_std.load_state_dict(checkpoint['proprio_mean_std'])
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
        self.teacher.load_state_dict(checkpoint['model'], strict=True)
        self.teacher_obs_rms.load_state_dict(checkpoint['running_mean_std'])
        cprint(f'[INFO] Loaded frozen Stage1 teacher from {fn} (student trains from scratch).',
               'green', attrs=['bold'])

    def restore_test(self, fn):
        checkpoint = torch.load(fn, map_location=self.device)
        self.student.load_state_dict(checkpoint['student'])
        self.teacher.load_state_dict(checkpoint['teacher_model'])
        self.teacher_obs_rms.load_state_dict(checkpoint['teacher_running_mean_std'])
        self.proprio_mean_std.load_state_dict(checkpoint['proprio_mean_std'])

    def save(self, name):
        weights = {
            'student': self.student.state_dict(),
            'teacher_model': self.teacher.state_dict(),
            'teacher_running_mean_std': self.teacher_obs_rms.state_dict(),
            'proprio_mean_std': self.proprio_mean_std.state_dict(),
            'optimizer': self.optim.state_dict(),
            'agent_steps': int(self.agent_steps),
            'best_rewards': float(self.best_rewards),
        }
        torch.save(weights, f'{name}.ckpt')

    def _print_epoch_log(self, iter_num, total_iters, collect_t, learn_t, iter_t, elapsed,
                         mean_rewards, mean_lengths, loss_val):
        width = 100
        pad = 30
        fps = int(self.batch_size / max(1e-6, collect_t + learn_t))
        eta_sec = max(0.0, (total_iters - iter_num) * (elapsed / max(1, iter_num)))

        rew_items = []
        for k in sorted(self.direct_info.keys()):
            v = self.direct_info[k]
            if isinstance(v, (int, float)):
                rew_items.append((k, float(v)))

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
