#!/usr/bin/env python3
"""Visualize a trained HORA Stage1 (PPO) policy in the Isaac Sim viewer.

Runs the deterministic policy (mu, no sampling) in a small number of envs with
the GUI open, and prints rolling episode statistics (reward, length, and for
screw tasks the nut rotation achieved per episode).

Example:
    python scripts/hora/play.py \
        --task nutbolt \
        --checkpoint outputs/hora/revo3_right/nutbolt_teacher_20260712_171600/stage1_nn/best.pth \
        --num_envs 16

Add --headless for a stats-only evaluation without the viewer.
"""

import argparse
import os
import sys
import tempfile
import traceback
from pathlib import Path

os.environ.setdefault("HORA_SKIP_SIM_CLOSE", "1")

REPO_ROOT = Path(__file__).resolve().parents[2]
EXTENSION_PATH = REPO_ROOT / "source" / "BrainCo_DexHand"
if str(EXTENSION_PATH) not in sys.path:
    sys.path.insert(0, str(EXTENSION_PATH))

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='nutbolt',
                    choices=['ball', 'cylinder', 'nutbolt', 'screwdriver',
                             'nutbolt_tactile', 'screwdriver_tactile'])
parser.add_argument('--checkpoint', type=str, required=True,
                    help='Stage1 checkpoint (.pth) from stage1_nn/, e.g. .../stage1_nn/best.pth')
parser.add_argument('--train_cfg', type=str, default='',
                    help='Train yaml name (default: Revo3HandHora for ball/cylinder, '
                         'Revo3HandScrew for nutbolt/screwdriver).')
parser.add_argument('--num_envs', type=int, default=16)
parser.add_argument('--steps', type=int, default=0,
                    help='Number of control steps to run (0 = run until the window is closed).')
parser.add_argument('--log_every', type=int, default=100,
                    help='Print rolling stats every N control steps.')
parser.add_argument('--seed', type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

_TACTILE_SCREW_TASKS = ('nutbolt_tactile', 'screwdriver_tactile')
_SCREW_TASKS = ('nutbolt', 'screwdriver') + _TACTILE_SCREW_TASKS
if not args.train_cfg:
    if args.task in _TACTILE_SCREW_TASKS:
        args.train_cfg = 'Revo3HandScrewTactile'
    else:
        args.train_cfg = 'Revo3HandScrew' if args.task in _SCREW_TASKS else 'Revo3HandHora'
if args.checkpoint.endswith('.ckpt'):
    raise ValueError(
        'play.py loads Stage1 PPO checkpoints (.pth from stage1_nn/). '
        'Stage2 ProprioAdapt checkpoints (.ckpt) are not supported here.'
    )
if not os.path.exists(args.checkpoint):
    raise FileNotFoundError(f'Checkpoint not found: {args.checkpoint}')

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
from omegaconf import OmegaConf

from BrainCo_DexHand.algo.hora.ppo.ppo import PPO
from BrainCo_DexHand.algo.hora.utils.misc import set_np_formatting, set_seed
from BrainCo_DexHand.tasks.direct.hora_rotation.assets import (
    BALL_OBJECT_CFG, CYLINDER_OBJECT_CFG,
    REVO3_HAND_BALL_CFG, REVO3_HAND_CYLINDER_CFG,
)
from BrainCo_DexHand.tasks.direct.hora_rotation.hora_compat_wrapper import HoraCompatWrapper
from BrainCo_DexHand.tasks.direct.hora_rotation.revo3_hand_hora_env import Revo3HandHoraEnv
from BrainCo_DexHand.tasks.direct.hora_rotation.revo3_hand_hora_env_cfg import Revo3HandHoraEnvCfg
from BrainCo_DexHand.tasks.direct.hora_screw.revo3_hand_screw_env import Revo3HandScrewEnv
from BrainCo_DexHand.tasks.direct.hora_screw.revo3_hand_screw_env_cfg import (
    Revo3HandScrewDriverEnvCfg,
    Revo3HandScrewNutBoltEnvCfg,
)
from BrainCo_DexHand.tasks.direct.hora_screw.revo3_hand_screw_tactile_env import Revo3HandScrewTactileEnv
from BrainCo_DexHand.tasks.direct.hora_screw.revo3_hand_screw_tactile_env_cfg import (
    Revo3HandScrewDriverTactileEnvCfg,
    Revo3HandScrewNutBoltTactileEnvCfg,
)

_SCREW_ENV_CFG = {
    'nutbolt': Revo3HandScrewNutBoltEnvCfg,
    'screwdriver': Revo3HandScrewDriverEnvCfg,
    'nutbolt_tactile': Revo3HandScrewNutBoltTactileEnvCfg,
    'screwdriver_tactile': Revo3HandScrewDriverTactileEnvCfg,
}


def _build_env_cfg():
    if args.task in _SCREW_TASKS:
        env_cfg = _SCREW_ENV_CFG[args.task]()
        # camera on env 0: screw scene sits near the ground
        env_cfg.viewer.origin_type = 'env'
        env_cfg.viewer.env_index = 0
        env_cfg.viewer.eye = (0.45, -0.45, 0.40)
        env_cfg.viewer.lookat = (0.0, 0.0, 0.10)
    else:
        env_cfg = Revo3HandHoraEnvCfg()
        env_cfg.robot_cfg = {'ball': REVO3_HAND_BALL_CFG, 'cylinder': REVO3_HAND_CYLINDER_CFG}[args.task]
        env_cfg.object_cfg = {'ball': BALL_OBJECT_CFG, 'cylinder': CYLINDER_OBJECT_CFG}[args.task]
        env_cfg.grasp_cache_path = f'assets/grasp_cache/hora/revo3_right_grasp_{args.task}'
        # play with full gravity, no curriculum (same as train.py --test)
        env_cfg.gravity_curriculum = False
        env_cfg.sim.gravity = (0.0, 0.0, -9.81)
        # camera on env 0: rotation scene sits around z ~ 1.6
        env_cfg.viewer.origin_type = 'env'
        env_cfg.viewer.env_index = 0
        env_cfg.viewer.eye = (0.5, -0.6, 1.95)
        env_cfg.viewer.lookat = (0.0, -0.08, 1.60)
    env_cfg.scene.num_envs = args.num_envs
    if hasattr(env_cfg, 'seed'):
        env_cfg.seed = args.seed
    if hasattr(env_cfg.sim, 'device') and getattr(args, 'device', None):
        env_cfg.sim.device = args.device
    return env_cfg


def _build_full_config(seed: int):
    _tasks_dir = REPO_ROOT / "source" / "BrainCo_DexHand" / "BrainCo_DexHand" / "tasks" / "direct"
    _candidates = [
        _tasks_dir / "hora_rotation" / "agents" / f"{args.train_cfg}.yaml",
        _tasks_dir / "hora_screw" / "agents" / f"{args.train_cfg}.yaml",
    ]
    cfg_path = next((p for p in _candidates if p.exists()), None)
    if cfg_path is None:
        raise FileNotFoundError(
            f"Train config '{args.train_cfg}.yaml' not found in: {[str(p.parent) for p in _candidates]}")
    train_cfg = OmegaConf.load(str(cfg_path))
    train_cfg.algo = 'PPO'
    train_cfg.load_path = os.path.abspath(args.checkpoint)
    train_cfg.ppo.num_actors = args.num_envs
    train_cfg.ppo.priv_info = True
    train_cfg.ppo.proprio_adapt = False
    rl_device = getattr(args, 'device', None) or 'cuda:0'
    return OmegaConf.create({
        'rl_device': rl_device,
        'test': True,
        'seed': seed,
        'train': train_cfg,
    })


def main():
    set_np_formatting()
    seed = set_seed(args.seed)
    full_config = _build_full_config(seed)

    env_cfg = _build_env_cfg()
    if args.task in _TACTILE_SCREW_TASKS:
        if int(full_config.train.ppo.priv_info_dim) != int(env_cfg.priv_info_dim):
            full_config.train.ppo.priv_info_dim = int(env_cfg.priv_info_dim)
        env_class = Revo3HandScrewTactileEnv
    else:
        env_class = Revo3HandScrewEnv if args.task in _SCREW_TASKS else Revo3HandHoraEnv
    env = env_class(
        cfg=env_cfg,
        render_mode=None if getattr(args, 'headless', False) else 'human',
    )
    env = HoraCompatWrapper(env)

    # PPO only serves as policy container here; its outputs go to a throwaway dir
    agent = PPO(env, tempfile.mkdtemp(prefix='hora_play_'), full_config=full_config)
    agent.restore_test(full_config.train.load_path)
    agent.set_eval()
    print(f'[INFO] Loaded checkpoint: {full_config.train.load_path}', flush=True)

    is_screw = args.task in _SCREW_TASKS
    device = agent.device
    num_envs = env.num_envs

    # per-env accumulators
    ep_reward = torch.zeros(num_envs, device=device)
    ep_len = torch.zeros(num_envs, device=device)
    # completed-episode statistics
    done_count = 0
    sum_reward = 0.0
    sum_len = 0.0
    sum_rot = 0.0   # nut rotation at episode end (rad), screw tasks only
    best_rot = 0.0

    obs_dict = env.reset()
    step = 0
    try:
        while args.steps <= 0 or step < args.steps:
            # snapshot nut position before stepping: done envs are reset inside step()
            if is_screw:
                nut_pos_before = env.nut_dof_pos.clone()

            input_dict = {
                'obs': agent.running_mean_std(obs_dict['obs']),
                'priv_info': obs_dict['priv_info'],
            }
            mu = agent.model.act_inference(input_dict)
            mu = torch.clamp(mu, -1.0, 1.0)
            obs_dict, rewards, dones, infos = env.step(mu)

            ep_reward += rewards.to(device)
            ep_len += 1
            done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
            if len(done_ids) > 0:
                done_count += len(done_ids)
                sum_reward += ep_reward[done_ids].sum().item()
                sum_len += ep_len[done_ids].sum().item()
                if is_screw:
                    final_rot = nut_pos_before[done_ids]
                    sum_rot += final_rot.sum().item()
                    best_rot = max(best_rot, final_rot.max().item())
                ep_reward[done_ids] = 0
                ep_len[done_ids] = 0

            step += 1
            if step % args.log_every == 0:
                line = f'[step {step:6d}] episodes done: {done_count}'
                if done_count > 0:
                    line += f' | ep_reward {sum_reward / done_count:8.2f} | ep_len {sum_len / done_count:6.1f}'
                    if is_screw:
                        mean_rot = sum_rot / done_count
                        line += (f' | nut rot/ep {mean_rot:6.3f} rad ({mean_rot / 6.2832:5.2f} rev)'
                                 f' | best {best_rot:6.3f} rad')
                if is_screw:
                    line += f' | live nut vel {env.nut_dof_vel_cf.mean().item():+.3f} rad/s'
                print(line, flush=True)
    except KeyboardInterrupt:
        print('\n[INFO] Interrupted by user.', flush=True)

    if done_count > 0:
        print(f'[SUMMARY] {done_count} episodes | mean reward {sum_reward / done_count:.2f} '
              f'| mean length {sum_len / done_count:.1f}'
              + (f' | mean nut rotation {sum_rot / done_count:.3f} rad '
                 f'({sum_rot / done_count / 6.2832:.2f} rev) | best {best_rot:.3f} rad' if is_screw else ''),
              flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        print('\n[ERROR] play.py terminated with an exception. Full traceback:', flush=True)
        traceback.print_exc()
        raise
    finally:
        if os.getenv('HORA_SKIP_SIM_CLOSE', '0') == '1':
            print('[INFO] Skip simulation_app.close() due to HORA_SKIP_SIM_CLOSE=1', flush=True)
        else:
            simulation_app.close()
