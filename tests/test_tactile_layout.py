"""Tests for the versioned Revo3 tactile layout data."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "source"
    / "BrainCo_DexHand"
    / "BrainCo_DexHand"
    / "tasks"
    / "tactile_layout.py"
)
_SPEC = importlib.util.spec_from_file_location("revo3_tactile_layout", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
tactile_layout = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(tactile_layout)
_LAYOUT_DIR = _MODULE_PATH.parent / "tactile_layouts"


@pytest.mark.parametrize(
    ("finger_name", "expected_count"),
    [("little", 21), ("ring", 21), ("middle", 21), ("index", 21), ("thumb", 31)],
)
def test_estimated_centers_are_complete_and_bounded(finger_name: str, expected_count: int):
    centers = tactile_layout.estimated_official_centers_xy(finger_name)

    assert centers.shape == (expected_count, 2)
    assert np.unique(centers, axis=0).shape[0] == expected_count
    assert np.min(centers[:, 0]) >= -0.006 - 1.0e-8
    assert np.max(centers[:, 0]) <= 0.012 + 1.0e-8
    assert np.max(np.abs(centers[:, 1])) <= 0.0085 + 1.0e-8


@pytest.mark.parametrize("finger_name", ["little", "ring", "middle", "index", "thumb"])
def test_estimated_layout_preserves_fixed_tacsl_tensor_shape(finger_name: str):
    snapped = tactile_layout.rasterize_estimated_official_layout(
        finger_name,
        np.linspace(-0.006, 0.012, 16, dtype=np.float32),
        np.linspace(-0.0085, 0.0085, 16, dtype=np.float32),
    )
    centers = tactile_layout.estimated_official_centers_xy(finger_name)

    assert snapped.shape == (256, 2)
    assert {tuple(position) for position in snapped.tolist()} == {
        tuple(position) for position in centers.tolist()
    }


@pytest.mark.parametrize(
    ("finger_name", "expected_angle_deg"),
    [
        ("little", 99.1897),
        ("ring", 92.6094),
        ("middle", 85.9192),
        ("index", 81.2185),
        ("thumb", 0.9442),
    ],
)
def test_v5_preserves_estimated_source_finger_directions(
    finger_name: str,
    expected_angle_deg: float,
):
    layout = tactile_layout.load_estimated_official_layout()
    finger = layout["fingers"][finger_name]

    assert layout["schema_version"] == 2
    assert layout["layout_version"] == "revo3_right_official_diagram_estimate_v5"
    assert finger["source_finger_direction_deg_ccw_from_image_right"] == pytest.approx(
        expected_angle_deg,
        abs=1.0e-4,
    )


def test_v3_only_shifts_v2_positions_toward_the_fingertips():
    v3 = json.loads(
        (_LAYOUT_DIR / "revo3_right_official_diagram_estimate_v3.json").read_text(
            encoding="utf-8"
        )
    )
    v2 = json.loads(
        (_LAYOUT_DIR / "revo3_right_official_diagram_estimate_v2.json").read_text(
            encoding="utf-8"
        )
    )

    assert v3["manual_adjustments"]["finger_longitudinal_offset_m"] == pytest.approx(0.001)
    for finger_name in ("little", "ring", "middle", "index", "thumb"):
        v2_finger = v2["fingers"][finger_name]
        v3_finger = v3["fingers"][finger_name]
        assert {
            key: value
            for key, value in v3_finger.items()
            if key not in ("sensors", "applied_fingertip_offset_m")
        } == {key: value for key, value in v2_finger.items() if key != "sensors"}

        for v2_sensor, v3_sensor in zip(v2_finger["sensors"], v3_finger["sensors"]):
            position_keys = {
                "position_finger_longitudinal_lateral_m",
                "position_sensor_plane_uv_m",
            }
            assert {
                key: value for key, value in v3_sensor.items() if key not in position_keys
            } == {key: value for key, value in v2_sensor.items() if key not in position_keys}
            np.testing.assert_allclose(
                np.asarray(v3_sensor["position_finger_longitudinal_lateral_m"])
                - np.asarray(v2_sensor["position_finger_longitudinal_lateral_m"]),
                (0.001, 0.0),
                rtol=0.0,
                atol=1.0e-12,
            )
            np.testing.assert_allclose(
                np.asarray(v3_sensor["position_sensor_plane_uv_m"])
                - np.asarray(v2_sensor["position_sensor_plane_uv_m"]),
                (0.0, 0.001),
                rtol=0.0,
                atol=1.0e-12,
            )


def test_v4_adds_patch_model_without_changing_v3_sensor_coordinates():
    v4 = json.loads(
        (_LAYOUT_DIR / "revo3_right_official_diagram_estimate_v4.json").read_text(
            encoding="utf-8"
        )
    )
    v3 = json.loads(
        (_LAYOUT_DIR / "revo3_right_official_diagram_estimate_v3.json").read_text(
            encoding="utf-8"
        )
    )

    assert tactile_layout.estimated_official_sensor_patch_radius_m() == pytest.approx(0.00125)
    for finger_name in ("little", "ring", "middle", "index", "thumb"):
        assert v4["fingers"][finger_name] == v3["fingers"][finger_name]


@pytest.mark.parametrize(
    ("finger_name", "offset_m", "longitudinal_half_span_m", "lateral_half_span_m"),
    [
        ("little", 0.0035, 0.007, 0.007),
        ("ring", 0.0035, 0.007, 0.007),
        ("middle", 0.0035, 0.007, 0.007),
        ("index", 0.0035, 0.007, 0.007),
        ("thumb", 0.003, 0.009, 0.0085),
    ],
)
def test_v5_recalibrates_v4_coordinates_from_the_real_sensor_images(
    finger_name: str,
    offset_m: float,
    longitudinal_half_span_m: float,
    lateral_half_span_m: float,
):
    v5 = tactile_layout.load_estimated_official_layout()
    v4 = json.loads(
        (_LAYOUT_DIR / "revo3_right_official_diagram_estimate_v4.json").read_text(
            encoding="utf-8"
        )
    )
    v5_finger = v5["fingers"][finger_name]
    v4_finger = v4["fingers"][finger_name]

    assert tactile_layout.estimated_official_fingertip_offset_m(finger_name) == pytest.approx(
        offset_m
    )
    assert v5_finger["calibrated_longitudinal_half_span_m"] == pytest.approx(
        longitudinal_half_span_m
    )
    assert v5_finger["calibrated_lateral_half_span_m"] == pytest.approx(
        lateral_half_span_m
    )
    for v4_sensor, v5_sensor in zip(v4_finger["sensors"], v5_finger["sensors"]):
        assert v5_sensor["official_sensor_id"] == v4_sensor["official_sensor_id"]
        assert v5_sensor["source_center_px"] == v4_sensor["source_center_px"]
        v4_xy = np.asarray(v4_sensor["position_finger_longitudinal_lateral_m"])
        expected_xy = np.asarray(
            (
                (v4_xy[0] - 0.001) * longitudinal_half_span_m / 0.009 + offset_m,
                v4_xy[1] * lateral_half_span_m / 0.007,
            )
        )
        np.testing.assert_allclose(
            v5_sensor["position_finger_longitudinal_lateral_m"],
            expected_xy,
            rtol=0.0,
            atol=1.1e-6,
        )
        np.testing.assert_allclose(
            v5_sensor["position_sensor_plane_uv_m"],
            expected_xy[::-1],
            rtol=0.0,
            atol=1.1e-6,
        )


@pytest.mark.parametrize(
    ("finger_name", "expected_longitudinal", "expected_lateral"),
    [
        ("index", (-0.0035, 0.0105), (-0.007, 0.007)),
        ("thumb", (-0.006, 0.012), (-0.0085, 0.0085)),
    ],
)
def test_v5_exposes_per_finger_calibration_bounds(
    finger_name: str,
    expected_longitudinal: tuple[float, float],
    expected_lateral: tuple[float, float],
):
    longitudinal, lateral = tactile_layout.estimated_official_calibration_bounds_xy(finger_name)

    np.testing.assert_allclose(longitudinal, expected_longitudinal, atol=1.0e-9)
    np.testing.assert_allclose(lateral, expected_lateral, atol=1.0e-9)


def test_seven_point_patch_quadrature_matches_disk_area_moments():
    offsets, weights = tactile_layout.estimated_official_sensor_patch_quadrature()
    radius = tactile_layout.estimated_official_sensor_patch_radius_m()

    assert offsets.shape == (7, 2)
    assert weights.shape == (7,)
    np.testing.assert_allclose(offsets[0], (0.0, 0.0), atol=1.0e-12)
    np.testing.assert_allclose(np.linalg.norm(offsets[1:], axis=-1), radius, rtol=1.0e-6)
    np.testing.assert_allclose(weights.sum(), 1.0, atol=1.0e-7)
    np.testing.assert_allclose((offsets * weights[:, None]).sum(axis=0), 0.0, atol=1.0e-10)
    np.testing.assert_allclose(
        ((offsets**2) * weights[:, None]).sum(axis=0),
        (radius**2 / 4.0, radius**2 / 4.0),
        rtol=1.0e-6,
    )


@pytest.mark.parametrize("finger_name", ["little", "ring", "middle", "index", "thumb"])
def test_slot_indices_cover_every_physical_sensor(finger_name: str):
    indices = tactile_layout.estimated_official_slot_center_indices(
        finger_name,
        np.linspace(-0.006, 0.012, 16, dtype=np.float32),
        np.linspace(-0.0085, 0.0085, 16, dtype=np.float32),
    )
    sensor_count = len(tactile_layout.estimated_official_centers_xy(finger_name))

    assert indices.shape == (256,)
    np.testing.assert_array_equal(np.unique(indices), np.arange(sensor_count))


@pytest.mark.parametrize("finger_name", ["little", "ring", "middle", "index", "thumb"])
def test_finger_coordinates_are_swapped_into_real_gel_axes(finger_name: str):
    finger_xy = tactile_layout.estimated_official_centers_xy(finger_name)
    sensor_plane_xy = tactile_layout.estimated_official_sensor_plane_xy(
        finger_name,
        finger_xy,
    )

    np.testing.assert_allclose(sensor_plane_xy[:, 0], finger_xy[:, 1])
    np.testing.assert_allclose(sensor_plane_xy[:, 1], finger_xy[:, 0])


def test_regular_grid_is_the_declared_default_layout():
    assert tactile_layout.REGULAR_GRID_LAYOUT == "regular_grid"
    assert tactile_layout.validate_tactile_layout_name("regular_grid") == "regular_grid"
    with pytest.raises(ValueError, match="Unknown tactile layout"):
        tactile_layout.validate_tactile_layout_name("unknown")


def test_resolve_agent_tactile_layout_prefers_top_level_field():
    class _Cfg:
        tactile_layout = "estimated_official"

    assert (
        tactile_layout.resolve_agent_tactile_layout(_Cfg())
        == tactile_layout.ESTIMATED_OFFICIAL_LAYOUT
    )


def test_resolve_agent_tactile_layout_falls_back_to_legacy_encoder_block():
    class _LayoutEncoder:
        layout = "estimated_official"

    class _Network:
        tactile_layout_encoder = _LayoutEncoder()

    class _Cfg:
        network = _Network()

    assert (
        tactile_layout.resolve_agent_tactile_layout(_Cfg())
        == tactile_layout.ESTIMATED_OFFICIAL_LAYOUT
    )
