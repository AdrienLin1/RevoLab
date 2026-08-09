"""Export supported Revo3 policies and deployment metadata.

Overview:
The tactile path resolves task-specific observation and action contracts from a
saved HORA checkpoint/config, while the Stage-2 path uses the Isaac Lab registry.

Quick Start:
    python scripts/export_policy.py --stage tactile_student --task nutbolt_tactile \
        --checkpoint CHECKPOINT --output-dir OUTPUT_DIR

Full Command:
    python scripts/export_policy.py --stage tactile_student --task TASK \
        --checkpoint CHECKPOINT --config CONFIG --profile PROFILE \
        --sensor-map SENSOR_MAP --output-dir OUTPUT_DIR

Options:
    --task: Supported tactile task name/alias, or an Isaac Lab task for Stage-2.
    --config: Optional tactile config override; otherwise discovered near checkpoint.

Notes:
    Physical tactile export supports estimated_official transformer/GNN students.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def build_parser() -> argparse.ArgumentParser:
    """Build the policy export command-line parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Export an Isaac Lab policy or a HORA tactile student checkpoint to ONNX "
            "plus deploy policy.yaml."
        )
    )
    parser.add_argument(
        "--stage",
        choices=("stage2", "tactile_student"),
        default="stage2",
        help="Export path. tactile_student uses the saved HORA config without Isaac Lab.",
    )
    parser.add_argument(
        "--task",
        default="",
        help="Tactile task name/alias, or Isaac Lab task ID for --stage stage2.",
    )
    parser.add_argument("--checkpoint", required=True, help="Path to .ckpt or .pt checkpoint.")
    parser.add_argument(
        "--config",
        default="",
        help="Saved HORA config override; auto-discovered beside a tactile checkpoint.",
    )
    parser.add_argument(
        "--layout",
        default="",
        help="Optional estimated-official tactile layout JSON override.",
    )
    parser.add_argument(
        "--sensor-map",
        default="",
        help="Optional verified diagram sensor-ID to SDK-channel mapping YAML.",
    )
    parser.add_argument("--profile", default="config/revo3_right.yaml", help="Robot profile with policy joint order.")
    parser.add_argument("--output-dir", required=True, help="Directory for policy.onnx and policy.yaml.")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--no-dynamic-batch", action="store_true")
    parser.add_argument("--agent", default="rsl_rl_cfg_entry_point")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--headless", action="store_true", default=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Dispatch to the task-aware tactile or Isaac Lab export path."""

    args = build_parser().parse_args(argv)
    if args.stage == "tactile_student":
        if not args.task:
            raise ValueError("--task is required for --stage tactile_student.")
        from revo3_deploy.tactile_exporter import export_tactile_student

        onnx_path, policy_path = export_tactile_student(
            checkpoint_path=args.checkpoint,
            config_path=args.config or None,
            profile_path=args.profile,
            output_dir=args.output_dir,
            task=args.task,
            layout_path=args.layout or None,
            sensor_map_path=args.sensor_map or None,
            opset=args.opset,
            dynamic_batch=not args.no_dynamic_batch,
        )
        print(f"[OK] Exported tactile student ONNX: {onnx_path}")
        print(f"[OK] Exported tactile deploy contract: {policy_path}")
    else:
        if not args.task:
            raise ValueError("--task is required for --stage stage2.")
        export_with_isaaclab(args)
    return 0


def export_with_isaaclab(args: argparse.Namespace) -> None:
    """Run the Isaac Lab export path without entering the play loop."""

    from isaaclab.app import AppLauncher

    app_args = argparse.Namespace(
        headless=args.headless,
        enable_cameras=False,
        device=args.device or "cuda:0",
        livestream=0,
        offscreen_render=False,
        render_mode=None,
        experience="",
        kit_args="",
    )
    app_launcher = AppLauncher(app_args)
    simulation_app = app_launcher.app

    try:
        import gymnasium as gym
        from rsl_rl.runners import DistillationRunner, OnPolicyRunner

        from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
        from isaaclab_tasks.utils.hydra import hydra_task_config
        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

        import BrainCo_DexHand  # noqa: F401

        env_cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
        agent_cfg = load_cfg_from_registry(args.task, args.agent)
        env_cfg.scene.num_envs = args.num_envs
        if args.device is not None:
            env_cfg.sim.device = args.device
            agent_cfg.device = args.device

        env_cfg.log_dir = str(Path(args.output_dir).resolve())
        env = gym.make(args.task, cfg=env_cfg)
        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        if agent_cfg.class_name == "OnPolicyRunner":
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        elif agent_cfg.class_name == "DistillationRunner":
            runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        else:
            raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")

        runner.load(str(Path(args.checkpoint).resolve()))
        try:
            policy_nn = runner.alg.policy
        except AttributeError:
            policy_nn = runner.alg.actor_critic

        normalizer = getattr(policy_nn, "actor_obs_normalizer", None)
        if normalizer is None:
            normalizer = getattr(policy_nn, "student_obs_normalizer", None)

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        export_policy_as_jit(policy_nn, normalizer=normalizer, path=str(output_dir), filename="policy.pt")
        export_policy_as_onnx(policy_nn, normalizer=normalizer, path=str(output_dir), filename="policy.onnx")
        write_policy_yaml(args, output_dir / "policy.yaml")
        env.close()
    finally:
        simulation_app.close()


def write_policy_yaml(args: argparse.Namespace, output_path: Path) -> None:
    """Write the legacy non-tactile Stage-2 deployment contract."""

    profile_path = Path(args.profile)
    with profile_path.open("r", encoding="utf-8") as f:
        profile = yaml.safe_load(f) or {}

    policy_order = list(profile.get("policy_joint_order") or [])
    cfg = {
        "export": {
            "stage": "stage2",
            "source_checkpoint": str(Path(args.checkpoint).resolve()),
            "task": args.task,
        },
        "artifacts": {"onnx": "policy.onnx"},
        "io_contract": {
            "inputs": [
                {"name": "obs", "shape": ["B", 126], "dtype": "float32"},
                {"name": "proprio_hist", "shape": ["B", 30, 42], "dtype": "float32"},
            ],
            "outputs": [{"name": "action", "shape": ["B", 21], "dtype": "float32"}],
            "action_semantics": "delta",
            "action_formula": "cur_targets = prev_targets + action_scale * action, then clamp to joint limits",
            "action_clip": [-1.0, 1.0],
            "policy_rate_hz": float(profile.get("default_rate_hz", 20.0)),
            "joint_order_right_hand": policy_order,
        },
        "normalization": {"baked_in_onnx": True},
    }
    with output_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


if __name__ == "__main__":
    raise SystemExit(main())
