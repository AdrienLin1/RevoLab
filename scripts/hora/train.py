#!/usr/bin/env python3
"""Training entry point for Stage1 (PPO) and Stage2 (ProprioAdapt).

Task selection: --task ball|cylinder selects robot_cfg, object_cfg, and grasp cache.
  robot_cfg and object_cfg are chosen from assets.py (not env_cfg.py class defaults).

Cache path: {grasp_cache_path}.npy under assets/grasp_cache/hora/.
Override with --cache_file.

PPO minibatches: the YAML minibatch_size is treated as a preferred maximum.
  At runtime it is reduced, when necessary, to the largest value that exactly
  divides num_envs × horizon_length. No rollout samples are dropped.

Gotcha — Stage2 enable_contact_in_obs=False: actor obs contacts zeroed, but
  proprio_hist retains real contact history for adapt_tconv distillation.
"""

import argparse
import copy
import datetime
import os
import sys
import traceback
from pathlib import Path

os.environ.setdefault("HORA_SKIP_SIM_CLOSE", "1")

REPO_ROOT = Path(__file__).resolve().parents[2]
EXTENSION_PATH = REPO_ROOT / "source" / "BrainCo_DexHand"
if str(EXTENSION_PATH) not in sys.path:
    sys.path.insert(0, str(EXTENSION_PATH))

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='cylinder',
                    choices=['ball', 'cylinder', 'nutbolt', 'screwdriver',
                             'nutbolt_tactile', 'screwdriver_tactile',
                             'valvedriver_tactile', 'valvedriver_tactile_40',
                             'vavledriver_tactile'])
parser.add_argument('--algo', type=str, default='PPO', choices=['PPO', 'ProprioAdapt'])
parser.add_argument('--train_cfg', type=str, default='',
                    help='Train yaml name (default: Revo3HandHora for ball/cylinder, '
                         'Revo3HandScrew for screw/valve tasks).')
parser.add_argument('--output_name', type=str, default='debug')
parser.add_argument('--checkpoint', type=str, default='')
parser.add_argument('--cache_file', type=str, default='', help='Override grasp cache filename under assets/grasp_cache/hora/.')
parser.add_argument('--usd', type=str, default='', help='Override hand USD path.')
parser.add_argument('--num_envs', type=int, default=4096)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--test', action='store_true')
parser.add_argument('--force_overwrite', action='store_true')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

_TACTILE_SCREW_TASKS = (
    'nutbolt_tactile', 'screwdriver_tactile',
    'valvedriver_tactile', 'valvedriver_tactile_40',
    'vavledriver_tactile',  # backward-compatible alias for the original typo
)
_SCREW_TASKS = ('nutbolt', 'screwdriver') + _TACTILE_SCREW_TASKS
if not args.train_cfg:
    if args.task in _TACTILE_SCREW_TASKS:
        args.train_cfg = 'Revo3HandScrewTactile'
    else:
        args.train_cfg = 'Revo3HandScrew' if args.task in _SCREW_TASKS else 'Revo3HandHora'

if not getattr(args, 'headless', False) and args.num_envs > 512:
    viewer_hint = (
        "--train_cfg Revo3HandScrewViewer --num_envs 128"
        if args.task in _SCREW_TASKS
        else "a smaller --num_envs value"
    )
    print(
        f"[WARNING] GUI training with {args.num_envs} environments renders every cloned asset and may exhaust "
        f"GPU memory. Use --headless for full training, or use {viewer_hint} for visualization.",
        flush=True,
    )


def _is_stage2_checkpoint(path: str) -> bool:
    if not path:
        return False
    return path.endswith('.ckpt') or 'stage2_nn' in path


def _default_output_name() -> str:
    if args.algo == 'PPO':
        return 'run1_continue' if args.checkpoint else f'run_{args.task}'
    # Stage2: output to Stage1's run dir
    if not _is_stage2_checkpoint(args.checkpoint):
        return f'run_{args.task}'
    # Stage2 resume: output to same directory as checkpoint
    return 'run2_continue'


def _resolve_minibatch_size(rollout_batch_size: int, preferred_size: int) -> int:
    """Return the largest minibatch no greater than preferred_size that divides the rollout exactly."""
    if rollout_batch_size <= 0:
        raise ValueError(f'rollout_batch_size must be positive, got {rollout_batch_size}')
    if preferred_size <= 0:
        raise ValueError(f'minibatch_size must be positive, got {preferred_size}')
    for size in range(min(rollout_batch_size, preferred_size), 0, -1):
        if rollout_batch_size % size == 0:
            return size
    raise RuntimeError('Failed to resolve PPO minibatch size')  # unreachable for positive integers


def _timestamped_path(base_path: str, label: str | None = None) -> str:
    """Return a non-existing timestamped path derived from base_path."""
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    suffix = f'_{label}_{timestamp}' if label else f'_{timestamp}'
    candidate = f'{base_path}{suffix}'
    counter = 2
    while os.path.exists(candidate):
        candidate = f'{base_path}{suffix}_{counter}'
        counter += 1
    return candidate


if not args.test and args.output_name == 'debug':
    args.output_name = _default_output_name()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from omegaconf import OmegaConf
from termcolor import cprint

from BrainCo_DexHand.algo.hora.padapt.padapt import ProprioAdapt
from BrainCo_DexHand.algo.hora.padapt.tactile_dagger import TactileDAgger
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
    Revo3HandValveDriver40TactileEnvCfg,
    Revo3HandVavleDriverTactileEnvCfg,
)


_ALGO_MAP = {
    'PPO': PPO,
    'ProprioAdapt': ProprioAdapt,
}

_TASK_ROBOT_CFG = {'ball': REVO3_HAND_BALL_CFG, 'cylinder': REVO3_HAND_CYLINDER_CFG}
_TASK_OBJECT_CFG = {'ball': BALL_OBJECT_CFG, 'cylinder': CYLINDER_OBJECT_CFG}
_TASK_CACHE = {
    'ball': 'assets/grasp_cache/hora/revo3_right_grasp_ball',
    'cylinder': 'assets/grasp_cache/hora/revo3_right_grasp_cylinder',
}

def _build_full_config(seed: int):
    _tasks_dir = REPO_ROOT / "source" / "BrainCo_DexHand" / "BrainCo_DexHand" / "tasks" / "direct"
    _candidates = [
        _tasks_dir / "hora_rotation" / "agents" / f"{args.train_cfg}.yaml",
        _tasks_dir / "hora_screw" / "agents" / f"{args.train_cfg}.yaml",
    ]
    cfg_path = next((p for p in _candidates if p.exists()), None)
    if cfg_path is None:
        raise FileNotFoundError(f"Train config '{args.train_cfg}.yaml' not found in: {[str(p.parent) for p in _candidates]}")
    train_cfg = OmegaConf.load(str(cfg_path))
    train_cfg.algo = args.algo
    train_cfg.load_path = os.path.abspath(args.checkpoint) if args.checkpoint else ''
    train_cfg.ppo.output_name = args.output_name
    if args.num_envs <= 0:
        raise ValueError(f'num_envs must be positive, got {args.num_envs}')
    if args.algo == 'PPO' and not args.test:
        horizon = int(train_cfg.ppo.horizon_length)
        rollout_batch = args.num_envs * horizon
        preferred_minibatch = int(train_cfg.ppo.minibatch_size)
        effective_minibatch = _resolve_minibatch_size(rollout_batch, preferred_minibatch)
        train_cfg.ppo.minibatch_size = effective_minibatch
        if effective_minibatch != preferred_minibatch:
            print(
                f'[INFO] Adjusted PPO minibatch_size: {preferred_minibatch} -> {effective_minibatch} '
                f'(num_envs={args.num_envs}, horizon={horizon}, rollout_batch={rollout_batch}).',
                flush=True,
            )
        if rollout_batch < 1024:
            print(
                f'[WARNING] PPO rollout batch is only {rollout_batch}; training is valid but gradient estimates '
                'may be noisy. Consider increasing --num_envs for policy training.',
                flush=True,
            )
    train_cfg.ppo.num_actors = args.num_envs
    train_cfg.ppo.priv_info = True
    train_cfg.ppo.proprio_adapt = args.algo == 'ProprioAdapt'

    rl_device = getattr(args, 'device', None) or 'cuda:0'
    return OmegaConf.create({
        'rl_device': rl_device,
        'test': args.test,
        'seed': seed,
        'train': train_cfg,
    })


_SCREW_ENV_CFG = {
    'nutbolt': Revo3HandScrewNutBoltEnvCfg,
    'screwdriver': Revo3HandScrewDriverEnvCfg,
    'nutbolt_tactile': Revo3HandScrewNutBoltTactileEnvCfg,
    'screwdriver_tactile': Revo3HandScrewDriverTactileEnvCfg,
    'valvedriver_tactile': Revo3HandVavleDriverTactileEnvCfg,
    'valvedriver_tactile_40': Revo3HandValveDriver40TactileEnvCfg,
    'vavledriver_tactile': Revo3HandVavleDriverTactileEnvCfg,
}


def _build_env_cfg(seed: int):
    if args.task in _SCREW_TASKS:
        env_cfg = _SCREW_ENV_CFG[args.task]()
    else:
        env_cfg = Revo3HandHoraEnvCfg()
        env_cfg.robot_cfg = _TASK_ROBOT_CFG.get(args.task, REVO3_HAND_CYLINDER_CFG)
        env_cfg.object_cfg = _TASK_OBJECT_CFG.get(args.task, CYLINDER_OBJECT_CFG)
        env_cfg.grasp_cache_path = _TASK_CACHE.get(args.task, 'assets/grasp_cache/hora/revo3_right_grasp_cylinder')
        if args.cache_file:
            env_cfg.grasp_cache_path = f"assets/grasp_cache/hora/{args.cache_file.replace('.npy', '')}"
    if args.usd:
        usd_path = os.path.abspath(args.usd)
        if not os.path.exists(usd_path):
            raise FileNotFoundError(f"--usd path not found: {usd_path}")
        env_cfg.robot_cfg = copy.deepcopy(env_cfg.robot_cfg)
        if env_cfg.robot_cfg.spawn is None or not hasattr(env_cfg.robot_cfg.spawn, "usd_path"):
            raise RuntimeError("env_cfg.robot_cfg.spawn has no usd_path to override.")
        env_cfg.robot_cfg.spawn.usd_path = usd_path

    env_cfg.scene.num_envs = args.num_envs


    if hasattr(env_cfg, 'seed'):
        env_cfg.seed = seed
    if hasattr(env_cfg.sim, 'device') and getattr(args, 'device', None):
        env_cfg.sim.device = args.device
    return env_cfg


def _save_run_metadata(output_dif: str, full_config) -> None:
    date = str(datetime.datetime.now().strftime('%m%d%H'))
    with open(os.path.join(output_dif, 'gitdiff.patch'), 'w', encoding='utf-8') as f:
        f.write('')
    config_name = f'config_{date}.yaml'

    with open(os.path.join(output_dif, config_name), 'w', encoding='utf-8') as f:
        f.write(OmegaConf.to_yaml(full_config))


def _attach_env_runtime_to_config(full_config, env_cfg) -> None:
    full_config.env_runtime = OmegaConf.create(
        {
            'grasp_cache_path': str(env_cfg.grasp_cache_path),
        }
    )


def main():
    if args.test and not args.checkpoint:
        raise ValueError('--test requires --checkpoint')
    if args.algo == 'ProprioAdapt' and not args.checkpoint:
        raise ValueError('ProprioAdapt training requires --checkpoint')

    set_np_formatting()
    seed = set_seed(args.seed)
    full_config = _build_full_config(seed)

    cprint('Start Building the Environment', 'green', attrs=['bold'])
    env_cfg = _build_env_cfg(seed)
    if args.task in _TACTILE_SCREW_TASKS:
        yaml_priv_dim = int(full_config.train.ppo.priv_info_dim)
        env_priv_dim = int(env_cfg.priv_info_dim)
        if yaml_priv_dim != env_priv_dim:
            print(
                f'[INFO] Syncing ppo.priv_info_dim: {yaml_priv_dim} (yaml) -> {env_priv_dim} (env cfg, '
                f'11 base + {env_cfg.tactile_priv_dim} tactile).',
                flush=True,
            )
            full_config.train.ppo.priv_info_dim = env_priv_dim
    if args.algo == 'ProprioAdapt' and args.task not in _TACTILE_SCREW_TASKS:
        env_cfg.enable_contact_in_obs = False  # Stage2: actor sees zero contact, adapt_tconv still sees contact history
    # Tactile tasks keep enable_contact_in_obs=True in Stage2: the DAgger teacher
    # labels actions with its original Stage1 observation (incl. real contacts);
    # the student only reads the 366-dim student_proprio_hist/student_tactile_hist.
    if args.test:
        env_cfg.gravity_curriculum = False
        env_cfg.sim.gravity = (0.0, 0.0, -9.81)  # full gravity for test/play
    if args.task in _TACTILE_SCREW_TASKS:
        env_class = Revo3HandScrewTactileEnv
    else:
        env_class = Revo3HandScrewEnv if args.task in _SCREW_TASKS else Revo3HandHoraEnv
    env = env_class(
        cfg=env_cfg,
        render_mode=None if getattr(args, 'headless', False) else 'human',
    )
    env = HoraCompatWrapper(env)

    # Never overwrite a previous experiment by default. Stage2 runs are grouped
    # under the checkpoint's run directory; repeated distillation runs therefore
    # also retain all older student models.
    if args.algo == 'ProprioAdapt':
        checkpoint_run_dir = os.path.dirname(os.path.dirname(os.path.abspath(args.checkpoint)))
        if args.force_overwrite:
            output_dif = checkpoint_run_dir
        else:
            stage2_root = os.path.join(checkpoint_run_dir, 'stage2_runs')
            output_dif = os.path.join(stage2_root, datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
            counter = 2
            candidate = output_dif
            while os.path.exists(candidate):
                candidate = f'{output_dif}_{counter}'
                counter += 1
            output_dif = candidate
    else:
        base_output_dif = os.path.join('outputs', 'hora', 'revo3_right', args.output_name)
        if args.force_overwrite or not os.path.exists(base_output_dif):
            output_dif = base_output_dif
        else:
            output_dif = _timestamped_path(base_output_dif)
    os.makedirs(output_dif, exist_ok=True)
    print(f'[INFO] Output directory: {os.path.abspath(output_dif)}', flush=True)
    algo_name = str(full_config.train.algo)
    if algo_name not in _ALGO_MAP:
        raise ValueError(f"Unsupported algo: {algo_name}. Available: {list(_ALGO_MAP.keys())}")
    agent_cls = _ALGO_MAP[algo_name]
    if algo_name == 'ProprioAdapt' and args.task in _TACTILE_SCREW_TASKS:
        # tactile screw tasks distill into a real-robot-obs student (366 dims) via DAgger
        agent_cls = TactileDAgger
    agent = agent_cls(env, output_dif, full_config=full_config)

    if args.test:
        agent.restore_test(full_config.train.load_path)
        agent.test()
    else:
        best_ckpt_path = os.path.join(
            output_dif,
            'stage1_nn' if full_config.train.algo == 'PPO' else 'stage2_nn',
            'best.pth' if full_config.train.algo == 'PPO' else 'model_best.ckpt',
        )
        if os.path.exists(best_ckpt_path) and args.force_overwrite:
            print(f"[WARNING] --force_overwrite enabled; existing checkpoints in {output_dif} may be replaced.", flush=True)

        _attach_env_runtime_to_config(full_config, env_cfg)
        _save_run_metadata(output_dif, full_config)
        agent.restore_train(full_config.train.load_path)
        agent.train()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        print("\n[ERROR] Training terminated with an exception. Full traceback:", flush=True)
        traceback.print_exc()
        raise
    finally:
        if os.getenv("HORA_SKIP_SIM_CLOSE", "0") == "1":
            print("[INFO] Skip simulation_app.close() due to HORA_SKIP_SIM_CLOSE=1", flush=True)
        else:
            simulation_app.close()
