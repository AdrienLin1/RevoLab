# Copyright (c) 2026, BrainCo.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Probe the separability of tactile event features w.r.t. impulse windows.

Rolls out the frozen base policy in the phase-2 env with forced impulses and reports, per event
feature, the class-conditional statistics and ROC-AUC of (a) the configured ``tacres_event``
features and (b) raw (unsmoothed) force-difference candidates computed directly from the sensors.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Probe tactile event-feature separability.")
parser.add_argument("--task", type=str, default="BrainCo-Dexsuite-Revo3-Right-Lift-TacRes-Phase2-v0")
parser.add_argument("--checkpoint", type=str, required=True, help="Phase-1 (base) checkpoint.")
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--steps", type=int, default=720)
parser.add_argument("--impulse", type=float, default=5.0)
parser.add_argument("--dump", type=str, default=None, help="Optional .pt path to dump features/labels for offline joint-AUC analysis.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import importlib.metadata as metadata

import gymnasium as gym
import torch

from isaaclab.managers import CurriculumTermCfg as CurrTerm

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

from rsl_rl.utils import resolve_callable

import BrainCo_DexHand  # noqa: F401

from isaaclab.managers import ManagerTermBase

PHASE1_TASK = "BrainCo-Dexsuite-Revo3-Right-Lift-TacRes-Phase1-v0"


class FixedDifficulty(ManagerTermBase):
    """Pins difficulty_frac to a constant (same as in eval_tacres.py, which cannot be imported
    because it runs argparse at module level)."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.difficulty_frac = torch.tensor(float(cfg.params.get("frac", 1.0)), device=env.device)

    def __call__(self, env, env_ids, frac: float = 1.0):
        return self.difficulty_frac


def auc(pos: torch.Tensor, neg: torch.Tensor) -> float:
    """Rank-based ROC-AUC of scalar feature separating pos from neg."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    scores = torch.cat([pos, neg])
    labels = torch.cat([torch.ones(len(pos)), torch.zeros(len(neg))])
    order = scores.argsort()
    ranks = torch.empty_like(order, dtype=torch.float)
    ranks[order] = torch.arange(1, len(scores) + 1, dtype=torch.float)
    r_pos = ranks[labels.bool()].sum()
    n_p, n_n = float(len(pos)), float(len(neg))
    return float((r_pos - n_p * (n_p + 1) / 2) / (n_p * n_n))


def main():
    device = args_cli.device if args_cli.device is not None else "cuda:0"
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = 123
    env_cfg.sim.device = device
    env_cfg.curriculum.adr = CurrTerm(func=FixedDifficulty, params={"frac": 1.0})
    env_cfg.curriculum.tacres_impulse_magnitude_adr = None
    env_cfg.events.tacres_perturbation.params["probability"] = 1.0
    env_cfg.events.tacres_perturbation.params["magnitude_range"] = (args_cli.impulse, args_cli.impulse)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=None)
    unwrapped = env.unwrapped
    pert = unwrapped.event_manager.cfg.tacres_perturbation.func
    sensor_names = list(env_cfg.observations.tacres_event.event.params["contact_sensor_names"])

    # frozen base policy
    agent_cfg = load_cfg_from_registry(PHASE1_TASK, "rsl_rl_cfg_entry_point")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    train_cfg = agent_cfg.to_dict()
    actor_cfg = dict(train_cfg["actor"])
    actor_class = resolve_callable(actor_cfg.pop("class_name"))
    obs = env.get_observations().to(device)
    actor = actor_class(obs, {"actor": train_cfg["obs_groups"]["actor"]}, "actor", env.num_actions, **actor_cfg).to(device)
    loaded = torch.load(args_cli.checkpoint, weights_only=False, map_location=device)
    actor.load_state_dict(loaded["actor_state_dict"], strict=True)
    actor.eval()

    ema_feats, raw_feats, labels, actives = [], [], [], []
    prev_force = None
    n_envs = unwrapped.num_envs
    ema_fast = torch.zeros(n_envs, len(sensor_names), 3, device=device)
    ema_slow = torch.zeros_like(ema_fast)
    prev_contact = torch.zeros(n_envs, len(sensor_names), device=device)
    change_window = []
    prev_joint_vel = None
    prev_obj_vel_b = None
    robot = unwrapped.scene["robot"]
    obj = unwrapped.scene["object"]
    from isaaclab.utils.math import quat_apply_inverse

    with torch.inference_mode():
        for _ in range(args_cli.steps):
            actions = actor(obs)
            obs, _, _, _ = env.step(actions.to(env.device))
            obs = obs.to(device)
            # configured event features (EMA-based, 6 dims)
            ema_feats.append(obs["tacres_event"].cpu().clone())
            # raw candidates computed straight from the sensors
            force = torch.stack(
                [unwrapped.scene.sensors[n].data.force_matrix_w.view(n_envs, 3) for n in sensor_names],
                dim=1,
            )
            if prev_force is None:
                prev_force = force.clone()
            diff = force - prev_force
            prev_force = force.clone()
            # band-pass tactile: fast EMA minus slow EMA
            ema_fast = 0.5 * ema_fast + 0.5 * force
            ema_slow = 0.9 * ema_slow + 0.1 * force
            band = ema_fast - ema_slow
            # windowed contact-change count (last 6 steps)
            contact = (force.norm(dim=-1) > 1.0).float()
            change_window.append((contact - prev_contact).abs().sum(dim=1))
            prev_contact = contact
            if len(change_window) > 6:
                change_window.pop(0)
            change6 = torch.stack(change_window, dim=0).sum(dim=0)
            # proprio jerk: joint velocity change norm
            jv = robot.data.joint_vel
            if prev_joint_vel is None:
                prev_joint_vel = jv.clone()
            jerk = (jv - prev_joint_vel).norm(dim=-1)
            prev_joint_vel = jv.clone()
            # object centroid velocity/acceleration in robot frame (perception-derivable proxy)
            obj_vel_b = quat_apply_inverse(robot.data.root_quat_w, obj.data.root_lin_vel_w)
            if prev_obj_vel_b is None:
                prev_obj_vel_b = obj_vel_b.clone()
            obj_acc = (obj_vel_b - prev_obj_vel_b).norm(dim=-1)
            prev_obj_vel_b = obj_vel_b.clone()
            raw = torch.stack(
                (
                    diff.norm(dim=-1).amax(dim=1),                     # max per-finger raw force diff
                    diff.sum(dim=1).norm(dim=-1),                      # net raw force diff
                    force.norm(dim=-1).amax(dim=1),                    # max raw force magnitude
                    band.norm(dim=-1).amax(dim=1),                     # band-pass tactile max
                    band.sum(dim=1).norm(dim=-1),                      # band-pass net
                    change6,                                           # contact changes in 6-step window
                    jerk,                                              # joint velocity jerk
                    obj_vel_b.norm(dim=-1),                            # object speed (robot frame)
                    obj_acc,                                           # object accel proxy
                ),
                dim=-1,
            )
            raw_feats.append(raw.cpu().clone())
            labels.append(pert.gate_label(unwrapped).cpu().clone())
            actives.append(pert.active_flag(unwrapped).cpu().clone())

    ema = torch.cat(ema_feats).view(-1, ema_feats[0].shape[-1])
    raw = torch.cat(raw_feats).view(-1, raw_feats[0].shape[-1])
    lab = torch.cat(labels).view(-1).bool()
    act = torch.cat(actives).view(-1).bool()
    print(f"[probe] samples={len(lab)}  label_rate={lab.float().mean():.4f}  active_rate={act.float().mean():.4f}",
          flush=True)

    ema_names = [f"event_feat[{i}]" for i in range(ema.shape[1])]
    raw_names = ["max_force_diff(RAW)", "net_force_diff(RAW)", "max_force_mag(RAW)",
                 "bandpass_max", "bandpass_net", "contact_change_w6", "joint_vel_jerk",
                 "obj_speed_b", "obj_accel_b"]
    for names, tensor in ((ema_names, ema), (raw_names, raw)):
        for i, name in enumerate(names):
            f = tensor[:, i]
            a = auc(f[lab], f[~lab])
            print(f"[probe] {name:24s} pos={f[lab].mean():8.4f}±{f[lab].std():7.4f} "
                  f"neg={f[~lab].mean():8.4f}±{f[~lab].std():7.4f} AUC={a:.3f}", flush=True)

    if args_cli.dump:
        torch.save({"event": ema, "raw": raw, "label": lab, "active": act}, args_cli.dump)
        print(f"[probe] dumped features to {args_cli.dump}", flush=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
