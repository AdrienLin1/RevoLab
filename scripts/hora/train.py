#!/usr/bin/env python3
"""Train HORA Stage1 PPO teachers or Stage2 adaptation students.

Overview:
The entry point selects the task assets and environment, synchronizes tactile
layout dimensions into the network config, and runs PPO or ProprioAdapt/DAgger.

Quick Start:
    python scripts/hora/train.py --task valvedriver --headless

Full Command:
    python scripts/hora/train.py --task valvedriver_tactile --train_cfg Revo3HandScrewTactile --num_envs 1024 --headless

Options:
    --tactile_layout: Override the TacSL point layout (default: use tactile_layout from YAML).
    --checkpoint: Resume Stage1 or provide the frozen Stage1 teacher for Stage2.
    --visualize_tactile: Write tactile visualization frames during training.

Notes:
Task assets come from the task asset map rather than environment class defaults.
The YAML PPO minibatch is a preferred maximum resolved against rollout divisibility.
Stage2 non-tactile adaptation hides actor contacts while retaining contact history
for distillation.
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
                    choices=['ball', 'cylinder',
                             'rotate_ball_tactile', 'rotate_cylinder_tactile',
                             'nutbolt', 'screwdriver',
                             'valvedriver', 'valvedriver_25', 'valvedriver_40', 'vavledriver',
                             'nutbolt_tactile', 'screwdriver_tactile',
                             'valvedriver_tactile', 'valvedriver_tactile25', 'valvedriver_tactile_40',
                             'vavledriver_tactile',
                             'valvedriver_tactile_xy'])
parser.add_argument('--algo', type=str, default='PPO',
                    choices=['PPO', 'ProprioAdapt', 'HierarchicalPPO'])
parser.add_argument('--train_cfg', type=str, default='',
                    help='Train yaml name (default: Revo3HandHora for ball/cylinder, '
                         'Revo3HandScrew for screw/valve tasks).')
parser.add_argument('--output_name', type=str, default='debug')
parser.add_argument('--checkpoint', type=str, default='',
                    help='Full training resume. For --algo HierarchicalPPO this must be a '
                         'complete hierarchical checkpoint (models + optimizers + curriculum).')
parser.add_argument('--master_checkpoint', type=str, default='',
                    help='HierarchicalPPO only: strictly warm-start the 21-D master policy '
                         '(weights + normalization) from an existing Stage-1 teacher '
                         'checkpoint. This is NOT a hierarchical resume.')
parser.add_argument('--cache_file', type=str, default='', help='Override grasp cache filename under assets/grasp_cache/hora/.')
parser.add_argument('--usd', type=str, default='', help='Override hand USD path.')
parser.add_argument('--num_envs', type=int, default=4096)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--test', action='store_true')
parser.add_argument('--force_overwrite', action='store_true')
parser.add_argument('--keep_all_tactile_fingers', action='store_true',
                    help='Use the legacy 5-finger tactile observation/token layout for tactile tasks.')
parser.add_argument('--tactile_layout', type=str, default=None,
                    choices=['regular_grid', 'estimated_official'],
                    help='Override the TacSL point layout (default: use tactile_layout from YAML).')
parser.add_argument('--visualize_tactile', action='store_true',
                    help='Save HORA tactile array/contact-force frames during training.')
parser.add_argument('--tactile_vis_every', type=int, default=2,
                    help='Render tactile visualization every N environment steps.')
parser.add_argument('--tactile_vis_show', action='store_true',
                    help='Show a live OpenCV tactile window in addition to writing frames.')
parser.add_argument('--tactile_vis_dir', type=str, default='',
                    help='Output dir for tactile frames (default: /tmp/revo3_tactile_vis_train_<task>).')
parser.add_argument('--tactile_gui_vis', action='store_true',
                    help='Show TacSL taxel positions in the Isaac GUI.')
parser.add_argument('--tactile_gui_contact_forces', action='store_true',
                    help='Also show ContactSensor force-tip markers in the Isaac GUI.')
parser.add_argument('--tactile_gui_env_index', type=int, default=0,
                    help='Environment index used for tactile GUI and frame visualization.')
parser.add_argument('--tactile_marker_radius', type=float, default=0.0,
                    help='Isaac GUI taxel marker radius in metres (0 = env cfg default).')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

if args.tactile_vis_show and not args.visualize_tactile:
    args.visualize_tactile = True

# Tasks whose hand additionally rides on a physical two-axis translation stage.
_XY_SCREW_TASKS = ('valvedriver_tactile_xy',)
_TACTILE_SCREW_TASKS = (
    'nutbolt_tactile', 'screwdriver_tactile',
    'valvedriver_tactile', 'valvedriver_tactile25', 'valvedriver_tactile_40',
    'vavledriver_tactile',  # backward-compatible alias for the original typo
) + _XY_SCREW_TASKS
_TACTILE_ROTATE_TASKS = ('rotate_ball_tactile', 'rotate_cylinder_tactile')
_TACTILE_TASKS = _TACTILE_SCREW_TASKS + _TACTILE_ROTATE_TASKS
_NON_TACTILE_SCREW_TASKS = (
    'nutbolt', 'screwdriver',
    'valvedriver', 'valvedriver_25', 'valvedriver_40',
    'vavledriver',  # backward-compatible alias for the original typo
)
_SCREW_TASKS = _NON_TACTILE_SCREW_TASKS + _TACTILE_SCREW_TASKS
if not args.train_cfg:
    if args.task in _XY_SCREW_TASKS:
        args.train_cfg = 'valvedriver_tactile_frame813_xy'
    elif args.task in _TACTILE_ROTATE_TASKS:
        args.train_cfg = 'Revo3HandTactileRotate'
    elif args.task in _TACTILE_SCREW_TASKS:
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
    if args.algo == 'HierarchicalPPO':
        return 'run_hier_continue' if args.checkpoint else f'run_hier_{args.task}'
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


def _cap_layout_minibatch_size(train_cfg, preferred_size: int) -> tuple[int, int | None]:
    """Return the preferred PPO minibatch size for public MLP training."""
    del train_cfg
    return preferred_size, None


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


def _sync_tactile_network_num_fingers(train_cfg, env_cfg) -> None:
    """Synchronize tactile encoder metadata from the environment when present."""
    del train_cfg, env_cfg


if not args.test and args.output_name == 'debug':
    args.output_name = _default_output_name()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from omegaconf import OmegaConf
from termcolor import cprint

from BrainCo_DexHand.tasks.tactile_layout import (
    resolve_agent_tactile_layout,
    validate_tactile_layout_name,
)
from BrainCo_DexHand.algo.hora.models.models import (
    infer_teacher_tactile_encoder_type_from_state_dict,
    resolve_tactile_encoder_type,
)
from BrainCo_DexHand.algo.hora.padapt.padapt import ProprioAdapt
from BrainCo_DexHand.algo.hora.padapt.tactile_dagger import TactileDAgger
from BrainCo_DexHand.algo.hora.ppo.hierarchical_ppo import HierarchicalPPO
from BrainCo_DexHand.algo.hora.ppo.ppo import PPO
from BrainCo_DexHand.algo.hora.utils.misc import set_np_formatting, set_seed
from BrainCo_DexHand.tasks.direct.hora_rotation.assets import (
    BALL_OBJECT_CFG, CYLINDER_OBJECT_CFG,
    REVO3_HAND_BALL_CFG, REVO3_HAND_CYLINDER_CFG,
)
from BrainCo_DexHand.tasks.direct.hora_rotation.hora_compat_wrapper import HoraCompatWrapper
from BrainCo_DexHand.tasks.direct.hora_rotation.revo3_hand_hora_env import Revo3HandHoraEnv
from BrainCo_DexHand.tasks.direct.hora_rotation.revo3_hand_hora_env_cfg import Revo3HandHoraEnvCfg
from BrainCo_DexHand.tasks.direct.hora_rotation.revo3_hand_tactile_rotate_env import (
    Revo3HandTactileRotateEnv,
)
from BrainCo_DexHand.tasks.direct.hora_rotation.revo3_hand_tactile_rotate_env_cfg import (
    Revo3HandTactileRotateBallEnvCfg,
    Revo3HandTactileRotateCylinderEnvCfg,
)
from BrainCo_DexHand.tasks.direct.hora_screw.revo3_hand_screw_env import Revo3HandScrewEnv
from BrainCo_DexHand.tasks.direct.hora_screw.revo3_hand_screw_env_cfg import (
    Revo3HandScrewDriverEnvCfg,
    Revo3HandScrewNutBoltEnvCfg,
    Revo3HandValveDriver25EnvCfg,
    Revo3HandValveDriver40EnvCfg,
    Revo3HandVavleDriverEnvCfg,
)
from BrainCo_DexHand.tasks.direct.hora_screw.revo3_hand_screw_tactile_env import Revo3HandScrewTactileEnv
from BrainCo_DexHand.tasks.direct.hora_screw.revo3_hand_screw_tactile_env_cfg import (
    Revo3HandScrewDriverTactileEnvCfg,
    Revo3HandScrewNutBoltTactileEnvCfg,
    Revo3HandValveDriver25TactileEnvCfg,
    Revo3HandValveDriver40TactileEnvCfg,
    Revo3HandVavleDriverTactileEnvCfg,
)
from BrainCo_DexHand.tasks.direct.hora_screw.revo3_hand_screw_tactile_xy_env import (
    Revo3HandScrewTactileXYEnv,
)
from BrainCo_DexHand.tasks.direct.hora_screw.revo3_hand_screw_tactile_xy_env_cfg import (
    Revo3HandVavleDriverTactileXYEnvCfg,
)


_ALGO_MAP = {
    'PPO': PPO,
    'ProprioAdapt': ProprioAdapt,
    'HierarchicalPPO': HierarchicalPPO,
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
    yaml_layout = resolve_agent_tactile_layout(train_cfg)
    layout_source = 'CLI' if args.tactile_layout is not None else 'YAML'
    if args.tactile_layout is None:
        args.tactile_layout = yaml_layout
    else:
        args.tactile_layout = validate_tactile_layout_name(args.tactile_layout)
    print(f'[INFO] Tactile layout: {args.tactile_layout} (source: {layout_source})', flush=True)
    train_cfg.algo = args.algo
    train_cfg.load_path = os.path.abspath(args.checkpoint) if args.checkpoint else ''
    # HierarchicalPPO only: warm-start path for the 21-D master policy. It is
    # deliberately separate from ``load_path`` so an old Stage-1 checkpoint can
    # never be mistaken for a complete hierarchical resume.
    master_checkpoint = getattr(args, 'master_checkpoint', '')
    train_cfg.master_load_path = (
        os.path.abspath(master_checkpoint) if master_checkpoint else ''
    )
    train_cfg.ppo.output_name = args.output_name
    if args.num_envs <= 0:
        raise ValueError(f'num_envs must be positive, got {args.num_envs}')
    if args.algo == 'PPO' and not args.test:
        horizon = int(train_cfg.ppo.horizon_length)
        rollout_batch = args.num_envs * horizon
        yaml_minibatch = int(train_cfg.ppo.minibatch_size)
        preferred_minibatch, graph_cap = _cap_layout_minibatch_size(train_cfg, yaml_minibatch)
        effective_minibatch = _resolve_minibatch_size(rollout_batch, preferred_minibatch)
        train_cfg.ppo.minibatch_size = effective_minibatch
        if graph_cap is not None and preferred_minibatch != yaml_minibatch:
            print(
                f'[INFO] Capped estimated_official GNN PPO minibatch_size: '
                f'{yaml_minibatch} -> {preferred_minibatch}.',
                flush=True,
            )
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
    'valvedriver': Revo3HandVavleDriverEnvCfg,
    'valvedriver_25': Revo3HandValveDriver25EnvCfg,
    'valvedriver_40': Revo3HandValveDriver40EnvCfg,
    'vavledriver': Revo3HandVavleDriverEnvCfg,
    'nutbolt_tactile': Revo3HandScrewNutBoltTactileEnvCfg,
    'screwdriver_tactile': Revo3HandScrewDriverTactileEnvCfg,
    'valvedriver_tactile': Revo3HandVavleDriverTactileEnvCfg,
    'valvedriver_tactile25': Revo3HandValveDriver25TactileEnvCfg,
    'valvedriver_tactile_40': Revo3HandValveDriver40TactileEnvCfg,
    'vavledriver_tactile': Revo3HandVavleDriverTactileEnvCfg,
    'valvedriver_tactile_xy': Revo3HandVavleDriverTactileXYEnvCfg,
}

_TACTILE_ROTATE_ENV_CFG = {
    'rotate_ball_tactile': Revo3HandTactileRotateBallEnvCfg,
    'rotate_cylinder_tactile': Revo3HandTactileRotateCylinderEnvCfg,
}


def _build_env_cfg(seed: int):
    if args.task in _TACTILE_ROTATE_TASKS:
        os.environ['REVO3_TACTILE_LAYOUT'] = args.tactile_layout
        if args.keep_all_tactile_fingers:
            os.environ['HORA_PACK_ACTIVE_TACTILE_ONLY'] = '0'
        env_cfg = _TACTILE_ROTATE_ENV_CFG[args.task]()
    elif args.task in _SCREW_TASKS:
        if args.task in _TACTILE_SCREW_TASKS:
            os.environ['REVO3_TACTILE_LAYOUT'] = args.tactile_layout
        if args.task in _TACTILE_SCREW_TASKS and args.keep_all_tactile_fingers:
            os.environ['HORA_PACK_ACTIVE_TACTILE_ONLY'] = '0'
        env_cfg = _SCREW_ENV_CFG[args.task]()
    else:
        env_cfg = Revo3HandHoraEnvCfg()
        env_cfg.robot_cfg = _TASK_ROBOT_CFG.get(args.task, REVO3_HAND_CYLINDER_CFG)
        env_cfg.object_cfg = _TASK_OBJECT_CFG.get(args.task, CYLINDER_OBJECT_CFG)
        env_cfg.grasp_cache_path = _TASK_CACHE.get(args.task, 'assets/grasp_cache/hora/revo3_right_grasp_cylinder')
        if args.cache_file:
            env_cfg.grasp_cache_path = f"assets/grasp_cache/hora/{args.cache_file.replace('.npy', '')}"
    if args.task in _TACTILE_TASKS:
        env_cfg.tactile_debug_vis = bool(
            args.tactile_gui_vis or args.tactile_gui_contact_forces
        )
        env_cfg.tactile_visualize_taxel_points = bool(args.tactile_gui_vis)
        env_cfg.tactile_visualize_contact_forces = bool(args.tactile_gui_contact_forces)
        env_cfg.tactile_vis_env_index = int(args.tactile_gui_env_index)
        if args.tactile_marker_radius > 0.0:
            env_cfg.tactile_taxel_marker_radius = float(args.tactile_marker_radius)
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


def _attach_env_runtime_to_config(full_config, env_cfg, task_name: str) -> None:
    """Persist task-resolved observation facts beside each checkpoint.

    Args:
        full_config: Mutable complete training configuration.
        env_cfg: Constructed environment configuration after task overrides.
        task_name: Requested HORA task name.
    """

    sim_dt = float(env_cfg.sim.dt)
    decimation = int(env_cfg.decimation)
    runtime = {
        'task': str(task_name),
        'grasp_cache_path': str(env_cfg.grasp_cache_path),
        'tactile_layout': str(getattr(env_cfg, 'tactile_layout', 'not_applicable')),
        'policy_rate_hz': 1.0 / (sim_dt * decimation),
        'action_scale': float(env_cfg.action_scale),
        'action_dim': int(env_cfg.action_space),
        'observation_dim': int(env_cfg.observation_space),
        'priv_info_dim': int(env_cfg.priv_info_dim),
        'student_tactile_duration_tau': float(
            getattr(env_cfg, 'student_tactile_duration_tau', 20.0)
        ),
        'student_tactile_duration_max': float(
            getattr(env_cfg, 'student_tactile_duration_max', 100.0)
        ),
        'tactile_shift_ema_beta': float(
            getattr(env_cfg, 'tactile_shift_ema_beta', 0.7)
        ),
        'tactile_shift_max': float(getattr(env_cfg, 'tactile_shift_max', 0.2)),
    }
    observation_fingers = tuple(
        str(value) for value in getattr(env_cfg, 'tactile_active_finger_names', ())
    )
    if observation_fingers:
        all_fingers = ('thumb', 'index', 'middle', 'ring', 'little')
        masked_joints = tuple(
            str(value) for value in getattr(env_cfg, 'masked_action_joint_names', ())
        )
        action_fingers = tuple(
            finger
            for finger in all_fingers
            if not any(f'_{finger}_' in joint_name for joint_name in masked_joints)
        )
        public_command_dim = int(
            getattr(env_cfg, 'student_proprio_command_dim', 0)
        )
        public_command_channels = (
            ['target_angular_velocity_rad_s'] if public_command_dim == 1 else []
        )
        if public_command_dim not in (0, 1):
            raise ValueError(
                'Unsupported student_proprio_command_dim: '
                f'{public_command_dim}; expected 0 or 1.'
            )
        runtime.update(
            {
                'tactile_active_finger_names': list(observation_fingers),
                'tactile_graph_sensor_counts': [
                    int(value)
                    for value in getattr(env_cfg, 'tactile_graph_sensor_counts', ())
                ],
                'tactile_graph_total_nodes': int(
                    getattr(env_cfg, 'tactile_graph_total_nodes', 0)
                ),
                'tactile_priv_offset': int(env_cfg.tactile_priv_offset),
                'tactile_priv_dim': int(env_cfg.tactile_priv_dim),
                'teacher_tactile_history_len': int(
                    env_cfg.teacher_tactile_history_len
                ),
                'teacher_tactile_frame_dim': int(
                    env_cfg.teacher_tactile_frame_dim
                ),
                'tactile_observation_fingers': list(observation_fingers),
                'action_fingers': list(action_fingers),
                'student_proprio_history_len': int(
                    env_cfg.student_proprio_history_len
                ),
                'student_proprio_frame_dim': int(env_cfg.student_proprio_frame_dim),
                'student_observation_schema': (
                    'hora_tactile_student_physical_graph_command_v2'
                    if public_command_channels
                    else 'hora_tactile_student_physical_graph_v1'
                ),
                'student_public_command_channels': public_command_channels,
                'student_proprio_raw_force_dim': 0,
                'student_tactile_history_len': int(
                    env_cfg.student_tactile_history_len
                ),
                'student_tactile_frame_dim': int(env_cfg.student_tactile_frame_dim),
                'student_tactile_total_nodes': int(
                    getattr(env_cfg, 'tactile_graph_total_nodes', 0)
                ),
            }
        )
        if public_command_channels:
            runtime['target_angvel_range_rad_s'] = [
                float(env_cfg.target_angvel_min),
                float(env_cfg.target_angvel_max),
            ]
    # The two-axis translation stage is identified by its own config contract,
    # not by a task-name list, so this stays valid for any future XY variant.
    if hasattr(env_cfg, 'xy_joint_limit'):
        runtime.update(
            {
                'finger_action_dim': int(env_cfg.finger_action_space),
                'xy_action_dim': int(env_cfg.action_space) - int(env_cfg.finger_action_space),
                'xy_stage_joint_names': [
                    str(value) for value in env_cfg.xy_stage_joint_names
                ],
                'xy_stage_world_axes': [
                    str(value) for value in env_cfg.xy_stage_world_axes
                ],
                'xy_joint_limit_m': float(env_cfg.xy_joint_limit),
                'xy_workspace_m': [
                    float(env_cfg.xy_workspace_initial),
                    float(env_cfg.xy_workspace_final),
                ],
                'xy_action_scale_m': [
                    float(env_cfg.xy_action_scale_initial),
                    float(env_cfg.xy_action_scale_final),
                ],
                'xy_effort_limit_n': float(env_cfg.xy_effort_limit),
                'xy_velocity_limit_mps': float(env_cfg.xy_velocity_limit),
                'xy_acceleration_limit_mps2': float(env_cfg.xy_acceleration_limit),
                'xy_pgain': float(env_cfg.xy_pgain),
                'xy_dgain': float(env_cfg.xy_dgain),
                'xy_curriculum_ramp_steps': int(env_cfg.xy_curriculum_ramp_steps),
                'follower_obs_dim': 159,
                'high_speed_reward_enable': bool(env_cfg.high_speed_reward_enable),
            }
        )
    full_config.env_runtime = OmegaConf.create(runtime)


def _checkpoint_teacher_num_fingers(model_state) -> int | None:
    """Infer tactile teacher finger slots from Stage1 checkpoint weights."""
    for key in (
        'tactile_encoder.spatial.queries',
        'tactile_encoder.queries',
    ):
        value = model_state.get(key)
        if value is not None and getattr(value, 'ndim', 0) >= 2:
            return int(value.shape[1])
    for key in (
        'tactile_encoder.spatial.finger_embedding.weight',
        'tactile_encoder.finger_embedding.weight',
        'tactile_encoder.graph_encoder.finger_embedding.weight',
        'tactile_encoder.common_branch.finger_embedding.weight',
    ):
        value = model_state.get(key)
        if value is not None and getattr(value, 'ndim', 0) >= 1:
            return int(value.shape[0])
    return None


def _checkpoint_env_mlp_input_dim(model_state) -> int | None:
    """Infer flat teacher privileged-input width from Stage1 checkpoint weights."""
    value = model_state.get('env_mlp.mlp.0.weight')
    if value is not None and getattr(value, 'ndim', 0) == 2:
        return int(value.shape[1])
    return None


def _validate_stage1_teacher_for_tactile_stage2(checkpoint_path: str, full_config, env_cfg) -> None:
    """Fail fast when Stage2 teacher/student tactile masks are inconsistent."""
    if _is_stage2_checkpoint(checkpoint_path):
        return

    import torch

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model_state = checkpoint.get('model')
    if model_state is None:
        return

    ckpt_teacher = infer_teacher_tactile_encoder_type_from_state_dict(model_state)
    run_dir = os.path.dirname(os.path.dirname(os.path.abspath(checkpoint_path)))
    cfg_teacher, _ = resolve_tactile_encoder_type(full_config.train.network)
    errors = []
    if ckpt_teacher != cfg_teacher:
        errors.append(
            f"teacher tactile_encoder.type mismatch: checkpoint={ckpt_teacher!r}, "
            f"train_cfg={cfg_teacher!r}"
        )
    env_priv_dim = int(env_cfg.priv_info_dim)
    cfg_priv_dim = int(full_config.train.ppo.priv_info_dim)
    if cfg_priv_dim != env_priv_dim:
        errors.append(f"ppo.priv_info_dim mismatch: train_cfg={cfg_priv_dim}, env_cfg={env_priv_dim}")

    active_fingers = tuple(getattr(env_cfg, 'tactile_active_finger_names', ()))
    env_num_fingers = len(active_fingers)
    ckpt_priv_dim = _checkpoint_env_mlp_input_dim(model_state)
    if ckpt_priv_dim is not None and ckpt_priv_dim != env_priv_dim:
        errors.append(
            f"flat teacher priv_info_dim mismatch: checkpoint={ckpt_priv_dim}, "
            f"env active priv={env_priv_dim} ({', '.join(active_fingers)})"
        )

    if str(getattr(env_cfg, 'tactile_layout', 'regular_grid')) == 'estimated_official':
        expected_student_frame = (
            int(env_cfg.tactile_graph_total_nodes) * int(env_cfg.tactile_graph_common_channels)
            + env_num_fingers * int(env_cfg.tactile_graph_context_channels)
        )
    else:
        expected_student_frame = (
            env_num_fingers
            * int(env_cfg.tactile_array_pool[0])
            * int(env_cfg.tactile_array_pool[1])
            * 3
        )
    if int(env_cfg.student_tactile_frame_dim) != expected_student_frame:
        errors.append(
            f"student_tactile_frame_dim mismatch: env_cfg={env_cfg.student_tactile_frame_dim}, "
            f"expected active-only={expected_student_frame}"
        )

    if errors:
        detail = "\n  - ".join(errors)
        raise ValueError(
            "Stage2 tactile teacher/student mask mismatch.\n"
            f"  checkpoint: {checkpoint_path}\n"
            f"  Stage1 run dir (check config_*.yaml): {run_dir}\n"
            f"  active tactile fingers: {', '.join(active_fingers)}\n"
            f"  - {detail}\n"
            "Fix: use a Stage1 checkpoint trained with the same task-active tactile fingers, "
            "or retrain Stage1 after the active-only tactile change."
        )


def main():
    if args.test and not args.checkpoint:
        raise ValueError('--test requires --checkpoint')
    if args.algo == 'ProprioAdapt' and not args.checkpoint:
        raise ValueError('ProprioAdapt training requires --checkpoint')
    if args.master_checkpoint and args.algo != 'HierarchicalPPO':
        raise ValueError('--master_checkpoint is only valid with --algo HierarchicalPPO')
    if args.master_checkpoint and args.checkpoint:
        raise ValueError(
            '--master_checkpoint (warm-start the master only) and --checkpoint '
            '(full hierarchical resume) are mutually exclusive.'
        )
    if args.algo == 'HierarchicalPPO' and args.task not in _XY_SCREW_TASKS:
        raise ValueError(
            f'--algo HierarchicalPPO requires a two-axis stage task '
            f'{_XY_SCREW_TASKS}, got {args.task!r}.'
        )
    if args.task in _XY_SCREW_TASKS and args.algo != 'HierarchicalPPO':
        raise ValueError(
            f'Task {args.task!r} has a 23-dim action space and requires '
            '--algo HierarchicalPPO.'
        )

    set_np_formatting()
    seed = set_seed(args.seed)
    full_config = _build_full_config(seed)

    cprint('Start Building the Environment', 'green', attrs=['bold'])
    env_cfg = _build_env_cfg(seed)
    if args.task in _TACTILE_TASKS:
        yaml_priv_dim = int(full_config.train.ppo.priv_info_dim)
        env_priv_dim = int(env_cfg.priv_info_dim)
        if yaml_priv_dim != env_priv_dim:
            print(
                f'[INFO] Syncing ppo.priv_info_dim: {yaml_priv_dim} (yaml) -> {env_priv_dim} (env cfg, '
                f'{env_cfg.tactile_priv_offset} base + {env_cfg.tactile_priv_dim} tactile).',
                flush=True,
            )
            full_config.train.ppo.priv_info_dim = env_priv_dim
        _sync_tactile_network_num_fingers(full_config.train, env_cfg)
    if args.algo == 'ProprioAdapt' and args.task in _TACTILE_TASKS and args.checkpoint:
        _validate_stage1_teacher_for_tactile_stage2(
            os.path.abspath(args.checkpoint),
            full_config,
            env_cfg,
        )
    if args.algo == 'ProprioAdapt' and args.task not in _TACTILE_TASKS:
        env_cfg.enable_contact_in_obs = False  # Stage2: actor sees zero contact, adapt_tconv still sees contact history
    # Tactile tasks keep enable_contact_in_obs=True in Stage2: the DAgger teacher
    # labels actions with its original Stage1 observation (incl. real contacts);
    # The student reads public proprio history plus derived tactile history only.
    if args.test:
        env_cfg.gravity_curriculum = False
        env_cfg.sim.gravity = (0.0, 0.0, -9.81)  # full gravity for test/play
    if args.task in _TACTILE_ROTATE_TASKS:
        env_class = Revo3HandTactileRotateEnv
    elif args.task in _XY_SCREW_TASKS:
        env_class = Revo3HandScrewTactileXYEnv
    elif args.task in _TACTILE_SCREW_TASKS:
        env_class = Revo3HandScrewTactileEnv
    else:
        env_class = Revo3HandScrewEnv if args.task in _SCREW_TASKS else Revo3HandHoraEnv
    env = env_class(
        cfg=env_cfg,
        render_mode=None if getattr(args, 'headless', False) else 'human',
    )
    # This wrapper must see the DirectRLEnv step before HoraCompatWrapper adapts its API.
    if args.visualize_tactile:
        if args.task not in _TACTILE_TASKS:
            print('[WARN] --visualize_tactile is only meaningful for tactile tasks.', flush=True)
        tactile_vis_path = REPO_ROOT / 'scripts' / 'rsl_rl'
        if str(tactile_vis_path) not in sys.path:
            sys.path.insert(0, str(tactile_vis_path))
        import tactile_vis

        out_dir = args.tactile_vis_dir or os.path.join(
            '/tmp', f'revo3_tactile_vis_train_{args.task}'
        )
        env = tactile_vis.TactileVizWrapper(
            env,
            out_dir=out_dir,
            every=args.tactile_vis_every,
            show=args.tactile_vis_show,
            env_index=args.tactile_gui_env_index,
            show_depth=False,
            show_arrays=True,
            show_contact_forces=True,
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
    if algo_name == 'ProprioAdapt' and args.task in _TACTILE_TASKS:
        # Tactile tasks distill into a real-robot-observation student via DAgger.
        agent_cls = TactileDAgger
    agent = agent_cls(env, output_dif, full_config=full_config)

    if args.test:
        agent.restore_test(full_config.train.load_path)
        agent.test()
    else:
        _BEST_CKPT_BY_ALGO = {
            'PPO': ('stage1_nn', 'best.pth'),
            'HierarchicalPPO': ('hier_nn', 'best_reward.pth'),
            'ProprioAdapt': ('stage2_nn', 'model_best.ckpt'),
        }
        best_dir, best_name = _BEST_CKPT_BY_ALGO[str(full_config.train.algo)]
        best_ckpt_path = os.path.join(output_dif, best_dir, best_name)
        if os.path.exists(best_ckpt_path) and args.force_overwrite:
            print(f"[WARNING] --force_overwrite enabled; existing checkpoints in {output_dif} may be replaced.", flush=True)

        _attach_env_runtime_to_config(full_config, env_cfg, args.task)
        _save_run_metadata(output_dif, full_config)
        if full_config.train.master_load_path:
            agent.restore_master_checkpoint(full_config.train.master_load_path)
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
