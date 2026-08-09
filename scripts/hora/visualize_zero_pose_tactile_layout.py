#!/usr/bin/env python3
"""Inspect the estimated Revo3 fingertip sensor layout at zero joint pose.

Overview:
The script creates the same five-finger TacSL sensors used by HORA, forces all
21 hand joints to zero, overlays the physical 31/21-node centers in Isaac Sim,
and exports the exact local and world coordinates used by the simulation.

Quick Start:
    python scripts/hora/visualize_zero_pose_tactile_layout.py

Full Command:
    python scripts/hora/visualize_zero_pose_tactile_layout.py --finger thumb --marker_radius_mm 1.25 --output_csv OUTPUT.csv

Options:
    --finger: Show all markers or isolate one fingertip.
    --marker_radius_mm: Set the viewport marker radius in millimeters.
    --output_csv: Select the coordinate export path.
    --steps: Exit after a fixed number of render frames; zero waits for viewer close.
    --show_task_geometry: Keep the screw object visible.

Notes:
    Run without ``--headless`` for interactive inspection. The sensor centers
    come directly from the same TacSL projection used to construct GNN inputs.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
EXTENSION_PATH = REPO_ROOT / "source" / "BrainCo_DexHand"
if str(EXTENSION_PATH) not in sys.path:
    sys.path.insert(0, str(EXTENSION_PATH))

os.environ["REVO3_TACTILE_LAYOUT"] = "estimated_official"
os.environ["HORA_PACK_ACTIVE_TACTILE_ONLY"] = "0"

from isaaclab.app import AppLauncher


def _build_parser() -> argparse.ArgumentParser:
    """Build command-line arguments before launching Isaac Sim.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--finger",
        choices=("all", "thumb", "index", "middle", "ring", "little"),
        default="all",
        help="Keep all markers visible or isolate one fingertip.",
    )
    parser.add_argument(
        "--marker_radius_mm",
        type=float,
        default=1.25,
        help="Viewport sphere radius in millimeters; 1.25 matches the sensor patch model.",
    )
    parser.add_argument(
        "--output_csv",
        default="outputs/hora/tactile_layout/zero_pose_sensor_positions.csv",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=0,
        help="Exit after this many render frames; 0 keeps the viewer open.",
    )
    parser.add_argument(
        "--show_task_geometry",
        action="store_true",
        help="Keep the screw object visible; the dark reference ground is always shown.",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser


parser = _build_parser()
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np
import omni.usd
import torch
from pxr import Gf, UsdGeom, Vt

import isaaclab.sim as sim_utils

from BrainCo_DexHand.tasks.direct.hora_screw.revo3_hand_screw_tactile_env import (
    Revo3HandScrewTactileEnv,
)
from BrainCo_DexHand.tasks.direct.hora_screw.revo3_hand_screw_tactile_env_cfg import (
    Revo3HandVavleDriverTactileEnvCfg,
)
from BrainCo_DexHand.tasks.tactile_layout import estimated_official_centers_xy


FINGER_COLORS = {
    "thumb": (0.90, 0.28, 0.16),
    "index": (0.10, 0.45, 0.85),
    "middle": (0.10, 0.65, 0.36),
    "ring": (0.72, 0.35, 0.75),
    "little": (0.95, 0.65, 0.12),
}
GROUND_MATERIAL_PATH = "/World/VisualMaterials/ZeroPoseGround"
GROUND_COLOR = (0.075, 0.095, 0.12)
# Original HORA palm-up root orientation, before the task-specific world-Y flip.
VIEWER_HAND_ROOT_ROT = (0.59636781, 0.37992820, -0.37992820, 0.59636781)


def _set_stage_visibility(stage, prim_path: str, visible: bool) -> None:
    """Set USD visibility when a stage prim exists.

    Args:
        stage: Active USD stage.
        prim_path: Absolute path of the prim to update.
        visible: Whether the prim should be visible.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if prim.IsValid():
        imageable = UsdGeom.Imageable(prim)
        imageable.MakeVisible() if visible else imageable.MakeInvisible()


def _style_ground(stage) -> None:
    """Give the reference ground a dark matte material.

    Args:
        stage: Active USD stage containing the task ground.
    """
    ground_prim = stage.GetPrimAtPath("/World/ground")
    if not ground_prim.IsValid():
        return
    if not stage.GetPrimAtPath(GROUND_MATERIAL_PATH).IsValid():
        material_cfg = sim_utils.PreviewSurfaceCfg(
            diffuse_color=GROUND_COLOR,
            roughness=0.9,
            metallic=0.0,
        )
        material_cfg.func(GROUND_MATERIAL_PATH, material_cfg)
    sim_utils.bind_visual_material(
        "/World/ground",
        GROUND_MATERIAL_PATH,
        stage=stage,
        stronger_than_descendants=True,
    )
    _set_stage_visibility(stage, "/World/ground", True)


def _freeze_zero_pose(env: Revo3HandScrewTactileEnv) -> None:
    """Write zero joint positions, velocities, and controller targets.

    Args:
        env: Tactile environment containing the Revo3 hand.
    """
    joint_pos = torch.zeros_like(env.hand.data.joint_pos)
    joint_vel = torch.zeros_like(env.hand.data.joint_vel)
    env.hand.set_joint_position_target(joint_pos)
    env.hand.write_joint_state_to_sim(joint_pos, joint_vel)
    for target_name in ("prev_targets", "cur_targets", "delayed_targets"):
        target = getattr(env, target_name, None)
        if target is not None:
            target.zero_()


def _sensor_records(env: Revo3HandScrewTactileEnv) -> list[dict[str, object]]:
    """Collect aligned local and world coordinates for every physical sensor.

    Args:
        env: Initialized tactile environment at the inspected pose.

    Returns:
        CSV-ready sensor records in canonical finger and official-ID order.
    """
    positions_w = env._tactile_taxel_positions_w(0)
    if positions_w is None:
        raise RuntimeError("The tactile environment did not expose physical sensor centers.")
    positions_w = positions_w.detach().cpu().numpy()
    records = []
    offset = 0
    for finger_name, sensor in zip(env.cfg.tactile_active_finger_names, env._tactile_sensor):
        local = sensor._tactile_physical_center_pos_local.detach().cpu().numpy()
        layout_xy = estimated_official_centers_xy(finger_name)
        finger_world = positions_w[offset : offset + len(local)]
        for sensor_index, (xy, local_xyz, world_xyz) in enumerate(
            zip(layout_xy, local, finger_world),
            start=1,
        ):
            records.append(
                {
                    "finger": str(finger_name),
                    "official_sensor_id": sensor_index,
                    "finger_longitudinal_m": float(xy[0]),
                    "finger_lateral_m": float(xy[1]),
                    "tip_local_x_m": float(local_xyz[0]),
                    "tip_local_y_m": float(local_xyz[1]),
                    "tip_local_z_m": float(local_xyz[2]),
                    "world_x_m": float(world_xyz[0]),
                    "world_y_m": float(world_xyz[1]),
                    "world_z_m": float(world_xyz[2]),
                }
            )
        offset += len(local)
    return records


def _style_markers(env: Revo3HandScrewTactileEnv, selected_finger: str) -> None:
    """Color markers by finger and optionally hide non-selected fingers.

    Args:
        env: Tactile environment that owns the marker prims.
        selected_finger: Canonical finger name or ``all``.
    """
    stage = omni.usd.get_context().get_stage()
    marker_index = 0
    for finger_name, sensor in zip(env.cfg.tactile_active_finger_names, env._tactile_sensor):
        count = int(sensor._tactile_physical_center_pos_local.shape[0])
        color = Vt.Vec3fArray([Gf.Vec3f(*FINGER_COLORS[str(finger_name)])])
        visible = selected_finger == "all" or selected_finger == str(finger_name)
        for _ in range(count):
            prim = stage.GetPrimAtPath(f"/Visuals/TactileTaxelSpheres/Taxel_{marker_index:04d}")
            if prim.IsValid():
                UsdGeom.Gprim(prim).CreateDisplayColorAttr().Set(color)
                imageable = UsdGeom.Imageable(prim)
                imageable.MakeVisible() if visible else imageable.MakeInvisible()
            marker_index += 1


def _set_focus_camera(env: Revo3HandScrewTactileEnv, records: list[dict[str, object]]) -> None:
    """Frame the hand from the front side of its tactile sensor plane.

    The fitted plane follows the current physical sensor coordinates, so the
    startup view remains face-on if the standalone hand pose changes later.

    Args:
        env: Tactile environment providing the viewport camera.
        records: Sensor coordinate records used to compute the target bounds.
    """
    selected = [
        row
        for row in records
        if args.finger == "all" or row["finger"] == args.finger
    ]
    positions = np.asarray(
        [[row["world_x_m"], row["world_y_m"], row["world_z_m"]] for row in selected],
        dtype=np.float64,
    )
    center = 0.5 * (positions.min(axis=0) + positions.max(axis=0))
    span = max(float(np.linalg.norm(np.ptp(positions, axis=0))), 0.025)
    centered = positions - positions.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / max(len(positions) - 1, 1)
    _, eigenvectors = np.linalg.eigh(covariance)
    view_direction = eigenvectors[:, 0]
    preferred_side = np.asarray((0.0, -1.0, 1.0), dtype=np.float64)
    if float(np.dot(view_direction, preferred_side)) < 0.0:
        view_direction = -view_direction
    view_direction /= max(float(np.linalg.norm(view_direction)), 1.0e-8)
    distance = max(0.09, 1.25 * span)
    eye = center + view_direction * distance
    env.sim.set_camera_view(eye=eye.tolist(), target=center.tolist())


def _write_csv(records: list[dict[str, object]], output_path: Path) -> None:
    """Write zero-pose sensor coordinates for calibration and review.

    Args:
        records: Sensor coordinate rows.
        output_path: Destination CSV path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    """Create the zero-pose scene and keep the viewport alive."""
    if float(args.marker_radius_mm) <= 0.0:
        raise ValueError("--marker_radius_mm must be positive.")
    cfg = Revo3HandVavleDriverTactileEnvCfg()
    cfg.seed = 0
    cfg.scene.num_envs = 1
    cfg.tactile_layout = "estimated_official"
    cfg.pack_active_tactile_only = False
    cfg.tactile_visualize_taxel_points = True
    cfg.tactile_visualize_contact_forces = False
    cfg.tactile_taxel_marker_radius = float(args.marker_radius_mm) * 1.0e-3
    cfg.tactile_vis_env_index = 0
    cfg.debug_show_axes = False
    cfg.reset_joint_noise_frac = 0.0
    cfg.randomize_object_xy_position = False
    cfg.robot_cfg.init_state.rot = VIEWER_HAND_ROOT_ROT
    cfg.robot_cfg.init_state.joint_pos = {
        joint_name: 0.0 for joint_name in cfg.robot_cfg.init_state.joint_pos
    }

    env = Revo3HandScrewTactileEnv(cfg=cfg)
    try:
        _freeze_zero_pose(env)
        env.sim.forward()
        env.scene.update(float(cfg.sim.dt))
        env._update_tactile_debug_visualization()

        stage = omni.usd.get_context().get_stage()
        _style_ground(stage)
        if not args.show_task_geometry:
            _set_stage_visibility(stage, "/World/envs/env_0/object", False)

        records = _sensor_records(env)
        _style_markers(env, args.finger)
        _set_focus_camera(env, records)
        output_path = Path(args.output_csv).expanduser().resolve()
        _write_csv(records, output_path)

        counts = {
            finger_name: len(estimated_official_centers_xy(finger_name))
            for finger_name in env.cfg.tactile_active_finger_names
        }
        print(f"[INFO] Zero-pose physical sensors: {counts}", flush=True)
        print(f"[INFO] Sensor coordinates: {output_path}", flush=True)
        print("[INFO] Close the Isaac viewport to exit.", flush=True)

        frame = 0
        while simulation_app.is_running():
            with torch.inference_mode():
                _freeze_zero_pose(env)
                env.sim.step(render=True)
                env.scene.update(float(cfg.sim.dt))
                env._update_tactile_debug_visualization()
            frame += 1
            if int(args.steps) > 0 and frame >= int(args.steps):
                break
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
