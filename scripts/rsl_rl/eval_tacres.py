# Copyright (c) 2026, BrainCo.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate TacRes / base-policy success rates on the Revo3 lift task under force impulses.

Runs a deterministic policy in the TacRes phase-2 environment with a frozen difficulty level and a
fixed impulse magnitude, and reports the per-episode success rate (object within ``pos_tol`` of the
commanded position at episode end, same criterion as the training ADR scheduler).

Examples:
  # base policy (B0), no perturbation
  python scripts/rsl_rl/eval_tacres.py --mode base --checkpoint <phase1>/model_N.pt --impulse 0 --headless
  # TacRes under 5N impulses
  python scripts/rsl_rl/eval_tacres.py --mode tacres --checkpoint <phase2>/model_N.pt --impulse 5 --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate TacRes success rates under impulses.")
parser.add_argument("--task", type=str, default="BrainCo-Dexsuite-Revo3-Right-Lift-TacRes-Phase2-v0")
parser.add_argument("--mode", type=str, choices=["base", "tacres"], required=True,
                    help="'base' evaluates a phase-1 proprio-only actor (B0); 'tacres' a phase-2 TacRes actor.")
parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint .pt file to evaluate.")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--episodes", type=int, default=512, help="Minimum number of finished episodes to count.")
parser.add_argument("--impulse", type=float, default=0.0, help="Impulse magnitude in N (0 disables impulses).")
parser.add_argument("--difficulty", type=float, default=10.0, help="Frozen ADR difficulty in [0, 10].")
parser.add_argument("--pos_tol", type=float, default=0.05, help="Success position tolerance (m).")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--out", type=str, default=None, help="Optional JSON file to append the result to.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import importlib.metadata as metadata
import json
import math
import os

import gymnasium as gym
import torch

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ManagerTermBase

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

from rsl_rl.utils import resolve_callable

import BrainCo_DexHand  # noqa: F401

PHASE1_TASK = "BrainCo-Dexsuite-Revo3-Right-Lift-TacRes-Phase1-v0"


class FixedDifficulty(ManagerTermBase):
    """Replacement for the ADR DifficultyScheduler that pins ``difficulty_frac`` to a constant.

    The curriculum interpolation terms only read ``.difficulty_frac`` from this term's instance, so
    replacing the scheduler freezes gravity/noise (and the impulse magnitude term is removed anyway).
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.difficulty_frac = torch.tensor(float(cfg.params.get("frac", 1.0)), device=env.device)

    def __call__(self, env, env_ids, frac: float = 1.0):
        return self.difficulty_frac


class SuccessRecorder(ManagerTermBase):
    """Counts episode successes at reset time (same terminal-state criterion as DifficultyScheduler)."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.num_success = 0
        self.num_episodes = 0
        self.pos_dist_sum = 0.0

    def __call__(self, env, env_ids, pos_tol: float = 0.05, min_steps: int = 5):
        from isaaclab.utils.math import combine_frame_transforms, compute_pose_error

        robot = env.scene["robot"]
        obj = env.scene["object"]
        command = env.command_manager.get_command("object_pose")
        des_pos_w, des_quat_w = combine_frame_transforms(
            robot.data.root_pos_w[env_ids], robot.data.root_quat_w[env_ids],
            command[env_ids, :3], command[env_ids, 3:7],
        )
        pos_err, _ = compute_pose_error(
            des_pos_w, des_quat_w, obj.data.root_pos_w[env_ids], obj.data.root_quat_w[env_ids]
        )
        pos_dist = torch.norm(pos_err, dim=1)
        # skip startup resets (episode_length_buf is still the terminal length here)
        valid = env.episode_length_buf[env_ids] > min_steps
        success = (pos_dist < pos_tol) & valid
        self.num_success += int(success.sum().item())
        self.num_episodes += int(valid.sum().item())
        self.pos_dist_sum += float(pos_dist[valid].sum().item())
        return torch.tensor(self.num_success / max(self.num_episodes, 1))


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def build_policy(mode: str, checkpoint: str, obs, num_actions: int, device: str):
    """Instantiate the actor from the task registry's agent cfg and load the checkpoint."""
    installed_version = metadata.version("rsl-rl-lib")
    if mode == "tacres":
        agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    else:
        agent_cfg = load_cfg_from_registry(PHASE1_TASK, "rsl_rl_cfg_entry_point")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    train_cfg = agent_cfg.to_dict()
    actor_cfg = dict(train_cfg["actor"])
    actor_class = resolve_callable(actor_cfg.pop("class_name"))
    obs_groups = {"actor": train_cfg["obs_groups"]["actor"]}
    actor = actor_class(obs, obs_groups, "actor", num_actions, **actor_cfg).to(device)
    loaded = torch.load(checkpoint, weights_only=False, map_location=device)
    actor.load_state_dict(loaded["actor_state_dict"], strict=True)
    actor.eval()
    return actor


def main():
    device = args_cli.device if args_cli.device is not None else "cuda:0"

    # -- environment configuration
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = device

    # freeze difficulty (gravity/noise) at the requested level
    env_cfg.curriculum.adr = CurrTerm(func=FixedDifficulty, params={"frac": args_cli.difficulty / 10.0})
    # fixed impulse magnitude instead of the training curriculum
    env_cfg.curriculum.tacres_impulse_magnitude_adr = None
    pert_params = env_cfg.events.tacres_perturbation.params
    if args_cli.impulse > 0.0:
        pert_params["probability"] = 1.0
        pert_params["magnitude_range"] = (args_cli.impulse, args_cli.impulse)
    else:
        pert_params["probability"] = 0.0
    # success bookkeeping
    env_cfg.curriculum.tacres_success_recorder = CurrTerm(
        func=SuccessRecorder, params={"pos_tol": args_cli.pos_tol, "min_steps": 5}
    )

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=None)

    unwrapped = env.unwrapped
    recorder: SuccessRecorder = unwrapped.curriculum_manager.cfg.tacres_success_recorder.func
    pert_term = unwrapped.event_manager.cfg.tacres_perturbation.func

    obs = env.get_observations().to(device)
    policy = build_policy(args_cli.mode, args_cli.checkpoint, obs, env.num_actions, device)
    print(f"[eval] mode={args_cli.mode} impulse={args_cli.impulse}N difficulty={args_cli.difficulty}"
          f" ckpt={args_cli.checkpoint}")

    # -- rollout
    gate_stats = {"in_sum": 0.0, "in_n": 0, "out_sum": 0.0, "out_n": 0, "corr_in": 0.0, "corr_out": 0.0}
    max_steps = int(args_cli.episodes / args_cli.num_envs + 3) * int(unwrapped.max_episode_length) + 100
    step = 0
    with torch.inference_mode():
        while recorder.num_episodes < args_cli.episodes and step < max_steps:
            actions = policy(obs)
            if args_cli.mode == "tacres":
                active = pert_term.active_flag(unwrapped).bool()
                gate = policy.last_gate.squeeze(-1)
                corr = (policy.alpha * policy.last_gate * policy.last_residual).norm(dim=-1)
                gate_stats["in_sum"] += gate[active].sum().item()
                gate_stats["in_n"] += int(active.sum().item())
                gate_stats["out_sum"] += gate[~active].sum().item()
                gate_stats["out_n"] += int((~active).sum().item())
                gate_stats["corr_in"] += corr[active].sum().item()
                gate_stats["corr_out"] += corr[~active].sum().item()
            obs, _, _, _ = env.step(actions.to(env.device))
            obs = obs.to(device)
            step += 1

    # -- report
    n, s = recorder.num_episodes, recorder.num_success
    rate = s / max(n, 1)
    lo, hi = wilson_ci(s, n)
    result = {
        "mode": args_cli.mode,
        "checkpoint": args_cli.checkpoint,
        "impulse_N": args_cli.impulse,
        "difficulty": args_cli.difficulty,
        "episodes": n,
        "successes": s,
        "success_rate": round(rate, 4),
        "ci95": [round(lo, 4), round(hi, 4)],
        "mean_final_pos_dist": round(recorder.pos_dist_sum / max(n, 1), 4),
        "seed": args_cli.seed,
    }
    if args_cli.mode == "tacres":
        result["gate_mean_inside_window"] = round(gate_stats["in_sum"] / max(gate_stats["in_n"], 1), 4)
        result["gate_mean_outside_window"] = round(gate_stats["out_sum"] / max(gate_stats["out_n"], 1), 4)
        result["corr_norm_inside_window"] = round(gate_stats["corr_in"] / max(gate_stats["in_n"], 1), 5)
        result["corr_norm_outside_window"] = round(gate_stats["corr_out"] / max(gate_stats["out_n"], 1), 5)
    print("[eval-result] " + json.dumps(result), flush=True)
    if args_cli.out:
        os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)), exist_ok=True)
        with open(args_cli.out, "a") as f:
            f.write(json.dumps(result) + "\n")
            f.flush()

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
