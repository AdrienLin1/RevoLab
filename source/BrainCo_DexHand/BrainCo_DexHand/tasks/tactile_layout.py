# Copyright (c) 2026, BrainCo.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Versioned Revo3 fingertip force-sensor layouts."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np


REGULAR_GRID_LAYOUT = "regular_grid"
ESTIMATED_OFFICIAL_LAYOUT = "estimated_official"
TACTILE_LAYOUT_NAMES = (REGULAR_GRID_LAYOUT, ESTIMATED_OFFICIAL_LAYOUT)

_ESTIMATED_LAYOUT_PATH = (
    Path(__file__).resolve().parent
    / "tactile_layouts"
    / "revo3_right_official_diagram_estimate_v5.json"
)
_EXPECTED_SENSOR_COUNTS = {
    "little": 21,
    "ring": 21,
    "middle": 21,
    "index": 21,
    "thumb": 31,
}


def validate_tactile_layout_name(layout_name: str) -> str:
    """Validate and return a tactile layout name."""

    layout_name = str(layout_name).strip().lower()
    if layout_name not in TACTILE_LAYOUT_NAMES:
        raise ValueError(
            f"Unknown tactile layout {layout_name!r}; expected one of {TACTILE_LAYOUT_NAMES}."
        )
    return layout_name


def resolve_tactile_layout_name(default: str = REGULAR_GRID_LAYOUT) -> str:
    """Resolve the optional process-wide layout override."""

    return validate_tactile_layout_name(os.getenv("REVO3_TACTILE_LAYOUT", default))


def resolve_agent_tactile_layout(agent_cfg, default: str = REGULAR_GRID_LAYOUT) -> str:
    """Resolve TacSL layout from an agent YAML configuration.

    Args:
        agent_cfg: Loaded OmegaConf/dict for one HORA agent YAML.
        default: Layout used when neither ``tactile_layout`` nor legacy
            ``network.tactile_layout_encoder.layout`` is present.

    Returns:
        Validated layout name in ``regular_grid`` or ``estimated_official``.
    """
    layout = getattr(agent_cfg, "tactile_layout", None)
    if layout is not None:
        return validate_tactile_layout_name(str(layout))
    network = getattr(agent_cfg, "network", None)
    layout_encoder = getattr(network, "tactile_layout_encoder", None) if network is not None else None
    if layout_encoder is not None:
        return validate_tactile_layout_name(
            str(getattr(layout_encoder, "layout", default))
        )
    return validate_tactile_layout_name(default)


@lru_cache(maxsize=1)
def load_estimated_official_layout() -> dict:
    """Load and validate the versioned official-diagram estimate."""

    with _ESTIMATED_LAYOUT_PATH.open("r", encoding="utf-8") as stream:
        layout = json.load(stream)

    if layout.get("schema_version") != 2:
        raise ValueError(f"Unexpected schema_version in {_ESTIMATED_LAYOUT_PATH}.")
    if layout.get("layout_version") != "revo3_right_official_diagram_estimate_v5":
        raise ValueError(f"Unexpected layout_version in {_ESTIMATED_LAYOUT_PATH}.")
    if layout.get("layout_name") != ESTIMATED_OFFICIAL_LAYOUT:
        raise ValueError(f"Unexpected layout_name in {_ESTIMATED_LAYOUT_PATH}.")
    adjustments = layout.get("manual_adjustments", {})
    patch_model = layout.get("sensor_patch_model", {})
    patch_radius = float(patch_model.get("radius_m", np.nan))
    patch_samples = int(patch_model.get("quadrature_samples", 0))
    patch_weights = np.asarray(patch_model.get("quadrature_weights", []), dtype=np.float32)
    if (
        patch_model.get("shape") != "circle"
        or not np.isfinite(patch_radius)
        or patch_radius <= 0.0
        or patch_samples != 7
        or patch_weights.shape != (patch_samples,)
        or np.any(patch_weights <= 0.0)
        or not np.isclose(float(patch_weights.sum()), 1.0, atol=1.0e-6)
    ):
        raise ValueError(f"Invalid circular sensor patch model in {_ESTIMATED_LAYOUT_PATH}.")
    fingers = layout.get("fingers", {})
    for finger_name, expected_count in _EXPECTED_SENSOR_COUNTS.items():
        finger = fingers.get(finger_name, {})
        fingertip_offset = float(finger.get("applied_fingertip_offset_m", np.nan))
        calibrated_longitudinal_half_span = float(
            finger.get("calibrated_longitudinal_half_span_m", np.nan)
        )
        calibrated_lateral_half_span = float(
            finger.get("calibrated_lateral_half_span_m", np.nan)
        )
        if (
            not np.isfinite(fingertip_offset)
            or not np.isfinite(calibrated_longitudinal_half_span)
            or not np.isfinite(calibrated_lateral_half_span)
            or calibrated_longitudinal_half_span <= 0.0
            or calibrated_lateral_half_span <= 0.0
        ):
            raise ValueError(f"Invalid image-to-finger calibration for {finger_name}.")
        try:
            adjustment_values = np.asarray(
                (
                    adjustments.get("finger_longitudinal_offset_m_by_finger", {}).get(
                        finger_name
                    ),
                    adjustments.get("longitudinal_half_span_m_by_finger", {}).get(finger_name),
                    adjustments.get("lateral_half_span_m_by_finger", {}).get(finger_name),
                ),
                dtype=np.float64,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Missing manual-adjustment summary for {finger_name}.") from exc
        if not np.allclose(
            adjustment_values,
            (fingertip_offset, calibrated_longitudinal_half_span, calibrated_lateral_half_span),
            rtol=0.0,
            atol=1.0e-9,
        ):
            raise ValueError(f"Inconsistent manual-adjustment summary for {finger_name}.")
        sensors = finger.get("sensors", [])
        sensor_ids = [int(sensor["official_sensor_id"]) for sensor in sensors]
        if sensor_ids != list(range(1, expected_count + 1)):
            raise ValueError(
                f"{finger_name} must contain official sensor ids 1..{expected_count}, got {sensor_ids}."
            )
        positions = np.asarray(
            [sensor["position_finger_longitudinal_lateral_m"] for sensor in sensors],
            dtype=np.float32,
        )
        if positions.shape != (expected_count, 2) or not np.isfinite(positions).all():
            raise ValueError(f"Invalid estimated xy positions for {finger_name}: {positions.shape}.")

        image_width, image_height = map(int, finger["source_image_size_px"])
        u_min, u_max, v_min, v_max = map(float, finger["calibration_roi_px"])
        source_centers = np.asarray(
            [sensor["source_center_px"] for sensor in sensors],
            dtype=np.float32,
        )
        if (
            np.any(source_centers[:, 0] < 0)
            or np.any(source_centers[:, 0] >= image_width)
            or np.any(source_centers[:, 1] < 0)
            or np.any(source_centers[:, 1] >= image_height)
        ):
            raise ValueError(f"Source circle center outside the {finger_name} screenshot.")

        distal_uv = np.asarray(finger["source_finger_direction_image_uv"], dtype=np.float32)
        lateral_uv = np.asarray(finger["source_lateral_direction_image_uv"], dtype=np.float32)
        if (
            distal_uv.shape != (2,)
            or lateral_uv.shape != (2,)
            or not np.isclose(np.linalg.norm(distal_uv), 1.0, atol=1.0e-5)
            or not np.isclose(np.linalg.norm(lateral_uv), 1.0, atol=1.0e-5)
            or not np.isclose(float(distal_uv @ lateral_uv), 0.0, atol=1.0e-5)
        ):
            raise ValueError(f"Invalid screenshot direction basis for {finger_name}.")
        roi_center = np.asarray(((u_min + u_max) * 0.5, (v_min + v_max) * 0.5))
        roi_half_size = np.asarray(((u_max - u_min) * 0.5, (v_max - v_min) * 0.5))
        source_offsets = source_centers - roi_center
        longitudinal_pixel_half_span = float(np.abs(distal_uv) @ roi_half_size)
        lateral_pixel_half_span = float(np.abs(lateral_uv) @ roi_half_size)
        expected_positions = np.stack(
            (
                source_offsets
                @ distal_uv
                / longitudinal_pixel_half_span
                * calibrated_longitudinal_half_span,
                source_offsets
                @ lateral_uv
                / lateral_pixel_half_span
                * calibrated_lateral_half_span,
            ),
            axis=-1,
        )
        expected_positions[:, 0] += fingertip_offset
        if not np.allclose(positions, expected_positions, rtol=0.0, atol=1.0e-6):
            raise ValueError(f"Stored xy positions do not match the {finger_name} pixel conversion.")
        if not np.isclose(
            float(finger.get("applied_fingertip_offset_m", np.nan)),
            fingertip_offset,
            rtol=0.0,
            atol=1.0e-9,
        ):
            raise ValueError(f"Unexpected applied fingertip offset for {finger_name}.")

        sensor_plane_matrix = np.asarray(
            finger["sensor_plane_from_finger_xy"],
            dtype=np.float32,
        )
        if sensor_plane_matrix.shape != (2, 2) or not np.allclose(
            sensor_plane_matrix @ sensor_plane_matrix.T,
            np.eye(2),
            rtol=0.0,
            atol=1.0e-6,
        ):
            raise ValueError(f"Invalid finger-to-sensor-plane matrix for {finger_name}.")
        sensor_plane_positions = np.asarray(
            [sensor["position_sensor_plane_uv_m"] for sensor in sensors],
            dtype=np.float32,
        )
        expected_sensor_plane_positions = positions @ sensor_plane_matrix.T
        if not np.allclose(
            sensor_plane_positions,
            expected_sensor_plane_positions,
            rtol=0.0,
            atol=1.0e-6,
        ):
            raise ValueError(f"Stored sensor-plane positions do not match {finger_name} finger xy.")
    return layout


def estimated_official_fingertip_offset_m(finger_name: str) -> float:
    """Return one finger's positive longitudinal offset toward its fingertip."""

    finger_name = str(finger_name).strip().lower()
    try:
        return float(
            load_estimated_official_layout()["fingers"][finger_name][
                "applied_fingertip_offset_m"
            ]
        )
    except KeyError as exc:
        raise ValueError(f"No estimated tactile layout for finger {finger_name!r}.") from exc


def estimated_official_calibration_bounds_xy(
    finger_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return longitudinal and lateral bounds used for screenshot calibration."""

    finger_name = str(finger_name).strip().lower()
    try:
        finger = load_estimated_official_layout()["fingers"][finger_name]
    except KeyError as exc:
        raise ValueError(f"No estimated tactile layout for finger {finger_name!r}.") from exc
    offset = float(finger["applied_fingertip_offset_m"])
    longitudinal_half_span = float(finger["calibrated_longitudinal_half_span_m"])
    lateral_half_span = float(finger["calibrated_lateral_half_span_m"])
    return (
        np.asarray(
            (offset - longitudinal_half_span, offset + longitudinal_half_span),
            dtype=np.float32,
        ),
        np.asarray((-lateral_half_span, lateral_half_span), dtype=np.float32),
    )


def estimated_official_sensor_patch_radius_m() -> float:
    """Return the estimated physical radius of one circular sensor patch."""

    return float(load_estimated_official_layout()["sensor_patch_model"]["radius_m"])


def estimated_official_sensor_patch_quadrature() -> tuple[np.ndarray, np.ndarray]:
    """Return seven-point disk offsets and normalized area-integration weights."""

    patch_model = load_estimated_official_layout()["sensor_patch_model"]
    radius = float(patch_model["radius_m"])
    angles = np.arange(6, dtype=np.float32) * (np.pi / 3.0)
    ring = np.stack((np.cos(angles), np.sin(angles)), axis=-1) * radius
    offsets = np.concatenate((np.zeros((1, 2), dtype=np.float32), ring), axis=0)
    weights = np.asarray(patch_model["quadrature_weights"], dtype=np.float32)
    return offsets, weights


def estimated_official_centers_xy(finger_name: str) -> np.ndarray:
    """Return centers in the finger longitudinal/lateral coordinate frame."""

    finger_name = str(finger_name).strip().lower()
    try:
        sensors = load_estimated_official_layout()["fingers"][finger_name]["sensors"]
    except KeyError as exc:
        raise ValueError(
            f"No estimated tactile layout for finger {finger_name!r}; "
            f"expected one of {tuple(_EXPECTED_SENSOR_COUNTS)}."
        ) from exc
    return np.asarray(
        [sensor["position_finger_longitudinal_lateral_m"] for sensor in sensors],
        dtype=np.float32,
    )


def estimated_official_sensor_plane_xy(
    finger_name: str,
    finger_xy: np.ndarray,
) -> np.ndarray:
    """Map longitudinal/lateral finger coordinates into the gel mesh plane."""

    finger_name = str(finger_name).strip().lower()
    try:
        matrix = np.asarray(
            load_estimated_official_layout()["fingers"][finger_name][
                "sensor_plane_from_finger_xy"
            ],
            dtype=np.float32,
        )
    except KeyError as exc:
        raise ValueError(f"No sensor-plane mapping for finger {finger_name!r}.") from exc
    finger_xy = np.asarray(finger_xy, dtype=np.float32)
    if finger_xy.ndim != 2 or finger_xy.shape[1] != 2:
        raise ValueError(f"finger_xy must have shape (N, 2), got {finger_xy.shape}.")
    return finger_xy @ matrix.T


def rasterize_estimated_official_layout(
    finger_name: str,
    grid_axis_x: np.ndarray,
    grid_axis_y: np.ndarray,
) -> np.ndarray:
    """Snap a finger-coordinate query grid to the nearest estimated sensor centers.

    TacSL and the existing policy consume a fixed rectangular tensor. Repeating each
    physical center across its nearest grid cells preserves that tensor contract while
    ensuring every force query lies on one of the estimated real sensor centers.
    """

    centers = estimated_official_centers_xy(finger_name)
    nearest_center_indices = estimated_official_slot_center_indices(
        finger_name,
        grid_axis_x,
        grid_axis_y,
    )
    return centers[nearest_center_indices].copy()


def estimated_official_slot_center_indices(
    finger_name: str,
    grid_axis_x: np.ndarray,
    grid_axis_y: np.ndarray,
) -> np.ndarray:
    """Map each fixed-grid query slot to its nearest physical sensor index."""

    centers = estimated_official_centers_xy(finger_name)
    grid_xy = np.asarray(
        [(x, y) for x in grid_axis_x for y in grid_axis_y],
        dtype=np.float32,
    )
    if grid_xy.size == 0:
        raise ValueError("TacSL grid axes must not be empty.")
    distances_sq = np.sum((grid_xy[:, None, :] - centers[None, :, :]) ** 2, axis=-1)
    nearest_center_indices = np.argmin(distances_sq, axis=1)
    missing_sensor_indices = sorted(set(range(len(centers))) - set(nearest_center_indices.tolist()))
    if missing_sensor_indices:
        missing_ids = [index + 1 for index in missing_sensor_indices]
        raise ValueError(
            f"TacSL grid is too coarse to represent {finger_name} estimated sensors {missing_ids}."
        )
    return nearest_center_indices.astype(np.int64, copy=False)


def finger_name_from_tip_body_path(path: str) -> str:
    """Extract the canonical finger name from a Revo3 fingertip prim path."""

    path = str(path).lower()
    for finger_name in _EXPECTED_SENSOR_COUNTS:
        if f"_{finger_name}_tip_link" in path:
            return finger_name
    raise ValueError(f"Cannot determine Revo3 finger from fingertip path: {path!r}.")
