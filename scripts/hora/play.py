#!/usr/bin/env python3
"""Play a trained HORA Stage1 teacher or Stage2 student policy.

Overview:
The script runs deterministic actions in a small environment batch, opens the
Isaac viewer when requested, and reports rolling reward, episode length, and
task rotation statistics.

Quick Start:
    python scripts/hora/play.py --task nutbolt --checkpoint CHECKPOINT

Full Command:
    python scripts/hora/play.py --task valvedriver_tactile --checkpoint CHECKPOINT --train_cfg Revo3HandScrewTactile --num_envs 16

Options:
    --tactile_layout: Match the layout used to train the checkpoint.
    --visualize_tactile: Write and optionally display tactile frames.
    --tactile_gui_vis: Display tactile sensor markers in the Isaac viewer.
    --tactile_gui_env_index: Focus the viewer and tactile displays on one parallel env.
    --tactile_force_scale: Multiply force-bearing tactile channels by k.
    --tactile_spatial_dropout: Permanently disable a fraction of taxels per env.
    --tactile_noise_std: Add Gaussian noise to force-bearing encoder channels.
    --tactile_binary_flip_prob: Flip Stage2 structural contact bits.
    --domain_randomization_final: Evaluate curriculum-controlled randomization at its final ranges.

Notes:
Use ``--headless`` for statistics-only evaluation without the viewer.
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
                    choices=['ball', 'cylinder',
                             'rotate_ball_tactile', 'rotate_cylinder_tactile',
                             'nutbolt', 'screwdriver',
                             'valvedriver', 'valvedriver_40', 'vavledriver',
                             'nutbolt_tactile', 'screwdriver_tactile',
                             'valvedriver_tactile', 'valvedriver_tactile_40',
                             'vavledriver_tactile'])
parser.add_argument('--checkpoint', type=str, required=True,
                    help='Stage1 .pth from stage1_nn/ or Stage2 .ckpt from stage2_nn/.')
parser.add_argument('--train_cfg', type=str, default='',
                    help='Train yaml name (default: Revo3HandHora for ball/cylinder, '
                         'Revo3HandScrew for screw/valve tasks).')
parser.add_argument('--num_envs', type=int, default=16)
parser.add_argument('--steps', type=int, default=0,
                    help='Number of control steps to run (0 = run until the window is closed).')
parser.add_argument('--log_every', type=int, default=100,
                    help='Print rolling stats every N control steps.')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--visualize_tactile', action='store_true',
                    help='Save HORA tactile array/contact-force frames to disk during play.')
parser.add_argument('--tactile_vis_every', type=int, default=2,
                    help='Render tactile visualization every N control steps.')
parser.add_argument('--tactile_vis_show', action='store_true',
                    help='Show a live OpenCV tactile window in addition to writing frames.')
parser.add_argument('--tactile_vis_dir', type=str, default='',
                    help='Output dir for tactile frames (default: /tmp/revo3_tactile_vis_<task>).')
parser.add_argument('--tactile_gui_vis', action='store_true',
                    help='Show TacSL taxel points (blue/orange) in the Isaac GUI.')
parser.add_argument('--tactile_gui_contact_forces', action='store_true',
                    help='Also show ContactSensor force-tip red spheres in Isaac GUI (off by default).')
parser.add_argument('--tactile_gui_env_index', type=int, default=0,
                    help='Environment index used for the Isaac viewer, tactile GUI, and tactile frames.')
parser.add_argument('--tactile_marker_radius', type=float, default=0.0,
                    help='Isaac GUI taxel sphere radius in metres (0 = env cfg default).')
parser.add_argument('--keep_all_tactile_fingers', action='store_true',
                    help='Use the legacy 5-finger tactile observation/token layout for tactile tasks.')
parser.add_argument('--tactile_layout', type=str, default=None,
                    choices=['regular_grid', 'estimated_official'],
                    help='Override the TacSL point layout (default: use tactile_layout from YAML).')
parser.add_argument('--tactile_force_scale', type=float, default=1.0,
                    help='Play-only force multiplier k applied to force-bearing encoder inputs.')
parser.add_argument('--tactile_spatial_dropout', type=float, default=0.0,
                    help='Play-only taxel failure ratio in [0,1]; exact-count masks are fixed per env.')
parser.add_argument('--tactile_noise_std', type=float, default=0.0,
                    help='Play-only Gaussian noise std in scaled TacSL force units.')
parser.add_argument('--tactile_binary_flip_prob', type=float, default=0.0,
                    help='Play-only Stage2 contact-bit flip probability in [0,1].')
parser.add_argument('--domain_randomization_final', action=argparse.BooleanOptionalAction, default=False,
                    help='Use final domain-randomization curriculum ranges instead of starting at progress 0.')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

if args.tactile_vis_show and not args.visualize_tactile:
    args.visualize_tactile = True
if args.tactile_force_scale <= 0.0:
    parser.error('--tactile_force_scale must be positive')
if not 0.0 <= args.tactile_spatial_dropout <= 1.0:
    parser.error('--tactile_spatial_dropout must be in [0, 1]')
if args.tactile_noise_std < 0.0:
    parser.error('--tactile_noise_std must be non-negative')
if not 0.0 <= args.tactile_binary_flip_prob <= 1.0:
    parser.error('--tactile_binary_flip_prob must be in [0, 1]')

_TACTILE_SCREW_TASKS = (
    'nutbolt_tactile', 'screwdriver_tactile',
    'valvedriver_tactile', 'valvedriver_tactile_40',
    'vavledriver_tactile',  # backward-compatible alias for the original typo
)
_TACTILE_ROTATE_TASKS = ('rotate_ball_tactile', 'rotate_cylinder_tactile')
_TACTILE_TASKS = _TACTILE_SCREW_TASKS + _TACTILE_ROTATE_TASKS
_NON_TACTILE_SCREW_TASKS = (
    'nutbolt', 'screwdriver',
    'valvedriver', 'valvedriver_40',
    'vavledriver',  # backward-compatible alias for the original typo
)
_SCREW_TASKS = _NON_TACTILE_SCREW_TASKS + _TACTILE_SCREW_TASKS
if not args.train_cfg:
    if args.task in _TACTILE_ROTATE_TASKS:
        args.train_cfg = 'Revo3HandTactileRotate'
    elif args.task in _TACTILE_SCREW_TASKS:
        args.train_cfg = 'Revo3HandScrewTactile'
    else:
        args.train_cfg = 'Revo3HandScrew' if args.task in _SCREW_TASKS else 'Revo3HandHora'
if not os.path.exists(args.checkpoint):
    raise FileNotFoundError(f'Checkpoint not found: {args.checkpoint}')


def _is_stage2_checkpoint(path: str) -> bool:
    """Return whether a checkpoint path follows the Stage2 naming contract.

    Args:
        path: Checkpoint path supplied on the command line.

    Returns:
        True for any Stage2 student checkpoint.
    """
    return path.endswith('.ckpt') or 'stage2_nn' in path


app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
from omegaconf import OmegaConf

from BrainCo_DexHand.algo.hora.ppo.ppo import PPO
from BrainCo_DexHand.algo.hora.padapt.padapt import ProprioAdapt
from BrainCo_DexHand.algo.hora.padapt.tactile_dagger import TactileDAgger
from BrainCo_DexHand.tasks.tactile_layout import (
    resolve_agent_tactile_layout,
    validate_tactile_layout_name,
)
from BrainCo_DexHand.algo.hora.utils.misc import set_np_formatting, set_seed
from BrainCo_DexHand.algo.hora.utils.tactile_robustness import TactileObservationPerturber
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
    Revo3HandValveDriver40EnvCfg,
    Revo3HandVavleDriverEnvCfg,
)
from BrainCo_DexHand.tasks.direct.hora_screw.revo3_hand_screw_tactile_env import Revo3HandScrewTactileEnv
from BrainCo_DexHand.tasks.direct.hora_screw.revo3_hand_screw_tactile_env_cfg import (
    Revo3HandScrewDriverTactileEnvCfg,
    Revo3HandScrewNutBoltTactileEnvCfg,
    Revo3HandValveDriver40TactileEnvCfg,
    Revo3HandVavleDriverTactileEnvCfg,
)

_SCREW_ENV_CFG = {
    'nutbolt': Revo3HandScrewNutBoltEnvCfg,
    'screwdriver': Revo3HandScrewDriverEnvCfg,
    'valvedriver': Revo3HandVavleDriverEnvCfg,
    'valvedriver_40': Revo3HandValveDriver40EnvCfg,
    'vavledriver': Revo3HandVavleDriverEnvCfg,
    'nutbolt_tactile': Revo3HandScrewNutBoltTactileEnvCfg,
    'screwdriver_tactile': Revo3HandScrewDriverTactileEnvCfg,
    'valvedriver_tactile': Revo3HandVavleDriverTactileEnvCfg,
    'valvedriver_tactile_40': Revo3HandValveDriver40TactileEnvCfg,
    'vavledriver_tactile': Revo3HandVavleDriverTactileEnvCfg,
}

_TACTILE_ROTATE_ENV_CFG = {
    'rotate_ball_tactile': Revo3HandTactileRotateBallEnvCfg,
    'rotate_cylinder_tactile': Revo3HandTactileRotateCylinderEnvCfg,
}


def _sync_tactile_network_num_fingers(train_cfg, env_cfg) -> None:
    """No-op for public MLP teacher configs."""
    del train_cfg, env_cfg


def _build_env_cfg():
    """Build the task-specific environment configuration for playback.

    Returns:
        A configured HORA rotation, screw, or tactile environment config.
    """
    if args.task in _TACTILE_ROTATE_TASKS:
        os.environ['REVO3_TACTILE_LAYOUT'] = args.tactile_layout
        if args.keep_all_tactile_fingers:
            os.environ['HORA_PACK_ACTIVE_TACTILE_ONLY'] = '0'
        env_cfg = _TACTILE_ROTATE_ENV_CFG[args.task]()
        env_cfg.gravity_curriculum = False
        env_cfg.sim.gravity = (0.0, 0.0, -9.81)
        env_cfg.viewer.origin_type = 'env'
        env_cfg.viewer.env_index = 0
        env_cfg.viewer.eye = (0.5, -0.6, 1.95)
        env_cfg.viewer.lookat = (0.0, -0.08, 1.60)
    elif args.task in _SCREW_TASKS:
        if args.task in _TACTILE_SCREW_TASKS:
            os.environ['REVO3_TACTILE_LAYOUT'] = args.tactile_layout
        if args.task in _TACTILE_SCREW_TASKS and args.keep_all_tactile_fingers:
            os.environ['HORA_PACK_ACTIVE_TACTILE_ONLY'] = '0'
        env_cfg = _SCREW_ENV_CFG[args.task]()
        # Screw-scene camera framing; the target env index is selected below.
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
        # Rotation-scene camera framing; tactile tasks select the target env below.
        env_cfg.viewer.origin_type = 'env'
        env_cfg.viewer.env_index = 0
        env_cfg.viewer.eye = (0.5, -0.6, 1.95)
        env_cfg.viewer.lookat = (0.0, -0.08, 1.60)
    if args.domain_randomization_final and hasattr(env_cfg, 'domain_randomization_curriculum_enable'):
        # Disabling interpolation makes the environment helpers return progress=1.0
        # while preserving randomized physics and their final bounds.
        env_cfg.domain_randomization_curriculum_enable = False
        print(
            '[INFO] Domain randomization: using final curriculum ranges '
            '(curriculum progress fixed at 1.0).',
            flush=True,
        )
    env_cfg.scene.num_envs = args.num_envs
    if args.task in _TACTILE_TASKS:
        tactile_env_index = int(args.tactile_gui_env_index)
        if tactile_env_index < 0 or tactile_env_index >= int(args.num_envs):
            raise ValueError(
                '--tactile_gui_env_index must be within the created environment batch: '
                f'got {tactile_env_index}, expected 0 <= index < {args.num_envs}'
            )
        # Keep the Isaac viewport and all tactile visualizers focused on the
        # same parallel environment. The radius-scale table printed at startup
        # can be used to choose an env with the desired object size.
        env_cfg.viewer.env_index = tactile_env_index
        env_cfg.tactile_debug_vis = bool(args.tactile_gui_vis or args.tactile_gui_contact_forces)
        env_cfg.tactile_visualize_taxel_points = bool(args.tactile_gui_vis)
        env_cfg.tactile_visualize_contact_forces = bool(args.tactile_gui_contact_forces)
        env_cfg.tactile_vis_env_index = tactile_env_index
        if args.tactile_marker_radius > 0.0:
            env_cfg.tactile_taxel_marker_radius = float(args.tactile_marker_radius)
    if hasattr(env_cfg, 'seed'):
        env_cfg.seed = args.seed
    if hasattr(env_cfg.sim, 'device') and getattr(args, 'device', None):
        env_cfg.sim.device = args.device
    return env_cfg


def _build_full_config(seed: int):
    """Load the selected training YAML and adapt it for deterministic play.

    Args:
        seed: Resolved random seed shared by the environment and policy.

    Returns:
        The complete OmegaConf configuration consumed by the HORA agent.
    """
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
    yaml_layout = resolve_agent_tactile_layout(train_cfg)
    layout_source = 'CLI' if args.tactile_layout is not None else 'YAML'
    if args.tactile_layout is None:
        args.tactile_layout = yaml_layout
    else:
        args.tactile_layout = validate_tactile_layout_name(args.tactile_layout)
    print(f'[INFO] Tactile layout: {args.tactile_layout} (source: {layout_source})', flush=True)
    train_cfg.algo = 'PPO'
    train_cfg.load_path = os.path.abspath(args.checkpoint)
    train_cfg.ppo.num_actors = args.num_envs
    train_cfg.ppo.priv_info = True
    train_cfg.ppo.proprio_adapt = (
        _is_stage2_checkpoint(args.checkpoint) and args.task not in _TACTILE_TASKS
    )
    rl_device = getattr(args, 'device', None) or 'cuda:0'
    return OmegaConf.create({
        'rl_device': rl_device,
        'test': True,
        'seed': seed,
        'train': train_cfg,
    })


def _tactile_robustness_enabled() -> bool:
    """Return whether any play-time tactile perturbation differs from baseline."""
    return (
        args.tactile_force_scale != 1.0
        or args.tactile_spatial_dropout > 0.0
        or args.tactile_noise_std > 0.0
        or args.tactile_binary_flip_prob > 0.0
    )


def main():
    """Run deterministic HORA playback and report robustness metrics."""
    set_np_formatting()
    seed = set_seed(args.seed)
    full_config = _build_full_config(seed)

    if _tactile_robustness_enabled() and args.task not in _TACTILE_TASKS:
        raise ValueError('Tactile robustness switches require a tactile task.')

    env_cfg = _build_env_cfg()
    if args.task in _TACTILE_TASKS:
        if int(full_config.train.ppo.priv_info_dim) != int(env_cfg.priv_info_dim):
            full_config.train.ppo.priv_info_dim = int(env_cfg.priv_info_dim)
        _sync_tactile_network_num_fingers(full_config.train, env_cfg)
    elif _is_stage2_checkpoint(full_config.train.load_path):
        # Match training: the actor sees zero contact while the adaptation
        # history retains the real contact signal used to infer extrinsics.
        env_cfg.enable_contact_in_obs = False
    if args.task in _TACTILE_ROTATE_TASKS:
        env_class = Revo3HandTactileRotateEnv
    elif args.task in _TACTILE_SCREW_TASKS:
        env_class = Revo3HandScrewTactileEnv
    else:
        env_class = Revo3HandScrewEnv if args.task in _SCREW_TASKS else Revo3HandHoraEnv
    env = env_class(
        cfg=env_cfg,
        render_mode=None if getattr(args, 'headless', False) else 'human',
    )
    # TactileVizWrapper must wrap the DirectRLEnv (gymnasium.Env), not HoraCompatWrapper.
    if args.visualize_tactile:
        if args.task not in _TACTILE_TASKS:
            print('[WARN] --visualize_tactile is only meaningful for tactile tasks.', flush=True)
        tactile_vis_path = REPO_ROOT / "scripts" / "rsl_rl"
        if str(tactile_vis_path) not in sys.path:
            sys.path.insert(0, str(tactile_vis_path))
        import tactile_vis

        out_dir = args.tactile_vis_dir or os.path.join('/tmp', f'revo3_tactile_vis_{args.task}')
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

    # Policy containers write only to a throwaway directory during playback.
    if _is_stage2_checkpoint(full_config.train.load_path):
        if args.task in _TACTILE_TASKS:
            agent = TactileDAgger(
                env,
                tempfile.mkdtemp(prefix='hora_play_stage2_'),
                full_config=full_config,
            )
            policy_mode = 'tactile_student'
        else:
            agent = ProprioAdapt(
                env,
                tempfile.mkdtemp(prefix='hora_play_stage2_'),
                full_config=full_config,
            )
            policy_mode = 'proprio_student'
    else:
        agent = PPO(env, tempfile.mkdtemp(prefix='hora_play_'), full_config=full_config)
        policy_mode = 'stage1_teacher'
    agent.restore_test(full_config.train.load_path)
    agent.set_eval()
    print(f'[INFO] Loaded {policy_mode} checkpoint: {full_config.train.load_path}', flush=True)

    is_screw = args.task in _SCREW_TASKS
    device = agent.device
    num_envs = env.num_envs
    tactile_perturber = None
    if args.task in _TACTILE_TASKS:
        tactile_perturber = TactileObservationPerturber(
            layout=str(env_cfg.tactile_layout),
            policy_mode=policy_mode,
            num_envs=num_envs,
            teacher_frame_dim=int(env_cfg.teacher_tactile_frame_dim),
            student_frame_dim=int(env_cfg.student_tactile_frame_dim),
            graph_total_nodes=int(env_cfg.tactile_graph_total_nodes),
            graph_sensor_counts=tuple(env_cfg.tactile_graph_sensor_counts),
            tactile_priv_offset=int(env_cfg.tactile_priv_offset),
            tactile_priv_dim=int(env_cfg.tactile_priv_dim),
            force_scale=float(args.tactile_force_scale),
            spatial_dropout=float(args.tactile_spatial_dropout),
            noise_std=float(args.tactile_noise_std),
            binary_flip_prob=float(args.tactile_binary_flip_prob),
            graph_force_limit=float(env_cfg.tactile_force_clip),
            legacy_action_dim=int(env_cfg.action_space),
            legacy_frame_dim=int(env_cfg.observation_space // 3),
            active_finger_indices=tuple(env_cfg.tactile_active_finger_indices),
            device=device,
        )
        keep_fraction = tactile_perturber.keep_mask.float().mean().item()
        print(
            '[INFO] Tactile robustness: '
            f'force_scale={args.tactile_force_scale:g}, '
            f'spatial_dropout={args.tactile_spatial_dropout:g} '
            f'(realized_keep={keep_fraction:.3f}), '
            f'noise_std={args.tactile_noise_std:g}, '
            f'binary_flip_prob={args.tactile_binary_flip_prob:g}',
            flush=True,
        )
        if policy_mode == 'tactile_student' and (
            args.tactile_force_scale != 1.0 or args.tactile_noise_std > 0.0
        ):
            print(
                '[INFO] Stage2 structural tactile input contains no force amplitude; '
                'force scaling/Gaussian force noise only affect unused teacher/legacy channels.',
                flush=True,
            )

    # per-env accumulators
    ep_reward = torch.zeros(num_envs, device=device)
    ep_len = torch.zeros(num_envs, device=device)
    # completed-episode statistics
    done_count = 0
    sum_reward = 0.0
    sum_len = 0.0
    sum_rot = 0.0   # nut rotation at episode end (rad), screw tasks only
    best_rot = 0.0
    abnormal_reset_count = 0
    timeout_success_count = 0
    angular_velocity_sum = 0.0
    angular_speed_sum = 0.0
    angular_velocity_samples = 0

    obs_dict = env.reset()
    if tactile_perturber is not None:
        obs_dict = tactile_perturber(obs_dict)
    step = 0
    try:
        while args.steps <= 0 or step < args.steps:
            # snapshot nut position before stepping: done envs are reset inside step()
            if is_screw:
                nut_pos_before = env.nut_dof_pos.clone()

            if policy_mode == 'tactile_student':
                proprio_hist = agent.proprio_mean_std(obs_dict['student_proprio_hist'])
                mu = agent.student(proprio_hist, obs_dict['student_tactile_hist'])
            elif policy_mode == 'proprio_student':
                input_dict = {
                    'obs': agent.running_mean_std(obs_dict['obs']),
                    'proprio_hist': agent.sa_mean_std(obs_dict['proprio_hist'].detach()),
                }
                mu = agent.model.act_inference(input_dict)
            else:
                input_dict = {
                    'obs': agent.running_mean_std(obs_dict['obs']),
                    'priv_info': obs_dict['priv_info'],
                }
                if 'tactile_hist' in obs_dict:
                    input_dict['tactile_hist'] = obs_dict['tactile_hist']
                mu = agent.model.act_inference(input_dict)
            mu = torch.clamp(mu, -1.0, 1.0)
            obs_dict, rewards, dones, infos = env.step(mu)
            angular_velocity = infos.get('metrics/angular_velocity_per_env')
            if not isinstance(angular_velocity, torch.Tensor):
                if is_screw:
                    angular_velocity = env.nut_dof_vel_cf
                else:
                    angular_velocity = (env.object_angvel * env.rot_axis).sum(dim=-1)
            angular_velocity_sum += angular_velocity.sum().item()
            angular_speed_sum += angular_velocity.abs().sum().item()
            angular_velocity_samples += int(angular_velocity.numel())

            ep_reward += rewards.to(device)
            ep_len += 1
            done_mask = dones.bool()
            terminated = infos.get('terminated', torch.zeros_like(done_mask)).bool()
            time_outs = infos.get('time_outs', torch.zeros_like(done_mask)).bool()
            timeout_success_count += int((done_mask & time_outs).sum().item())
            # Timeout is an accepted task outcome. Only non-timeout terminations
            # are counted as abnormal resets, even if both flags are present.
            abnormal_reset_count += int((done_mask & terminated & ~time_outs).sum().item())
            done_ids = done_mask.nonzero(as_tuple=False).squeeze(-1)
            if tactile_perturber is not None:
                obs_dict = tactile_perturber(obs_dict, reset_mask=done_mask)
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
                    success_rate = 1.0 - abnormal_reset_count / done_count
                    mean_survival_s = sum_len / done_count * float(env.step_dt)
                    line += f' | ep_reward {sum_reward / done_count:8.2f} | ep_len {sum_len / done_count:6.1f}'
                    line += (
                        f' | success {success_rate:6.2%}'
                        f' (timeout {timeout_success_count})'
                        f' | survival {mean_survival_s:6.2f} s'
                    )
                    if is_screw:
                        mean_rot = sum_rot / done_count
                        line += (f' | nut rot/ep {mean_rot:6.3f} rad ({mean_rot / 6.2832:5.2f} rev)'
                                 f' | best {best_rot:6.3f} rad')
                if is_screw:
                    line += f' | live nut vel {env.nut_dof_vel_cf.mean().item():+.3f} rad/s'
                if angular_velocity_samples > 0:
                    line += f' | mean ang vel {angular_velocity_sum / angular_velocity_samples:+.3f} rad/s'
                print(line, flush=True)
    except KeyboardInterrupt:
        print('\n[INFO] Interrupted by user.', flush=True)

    if done_count > 0:
        success_rate = 1.0 - abnormal_reset_count / done_count
        mean_survival_s = sum_len / done_count * float(env.step_dt)
        print(f'[SUMMARY] {done_count} episodes | success rate {success_rate:.2%} '
              f'({timeout_success_count} timeout successes, {abnormal_reset_count} abnormal resets) '
              f'| mean survival {mean_survival_s:.2f} s '
              f'| mean angular velocity {angular_velocity_sum / angular_velocity_samples:+.3f} rad/s '
              f'| mean angular speed {angular_speed_sum / angular_velocity_samples:.3f} rad/s '
              f'| mean reward {sum_reward / done_count:.2f} | mean length {sum_len / done_count:.1f}'
              + (f' | mean nut rotation {sum_rot / done_count:.3f} rad '
                 f'({sum_rot / done_count / 6.2832:.2f} rev) | best {best_rot:.3f} rad' if is_screw else ''),
              flush=True)
    elif angular_velocity_samples > 0:
        print(
            '[SUMMARY] no completed episodes; success rate and survival are unavailable '
            f'| mean angular velocity {angular_velocity_sum / angular_velocity_samples:+.3f} rad/s '
            f'| mean angular speed {angular_speed_sum / angular_velocity_samples:.3f} rad/s',
            flush=True,
        )


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
