"""Tests for physical-layout tactile visualization helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


_MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "rsl_rl" / "tactile_vis.py"
_SPEC = importlib.util.spec_from_file_location("revo3_tactile_vis", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
tactile_vis = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(tactile_vis)


def test_repeated_query_slots_are_averaged_per_physical_center():
    plane_xy = np.array(
        [[-0.001, 0.002], [-0.001, 0.002], [0.003, -0.004]],
        dtype=np.float32,
    )
    force = np.array(
        [[1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [7.0, 8.0, 9.0]],
        dtype=np.float32,
    )

    centers, aggregated = tactile_vis.TactileVizWrapper._aggregate_layout_slots(
        force,
        plane_xy,
    )

    np.testing.assert_allclose(centers, [[-0.001, 0.002], [0.003, -0.004]])
    np.testing.assert_allclose(aggregated, [[2.0, 3.0, 4.0], [7.0, 8.0, 9.0]])


def test_estimated_layout_orients_every_finger_distal_direction_up():
    centers = np.array([[0.004, -0.002]], dtype=np.float32)

    ordinary = tactile_vis.TactileVizWrapper._layout_screen_coordinates(centers, "index")
    thumb = tactile_vis.TactileVizWrapper._layout_screen_coordinates(centers, "thumb")

    np.testing.assert_allclose(ordinary, [[-0.002, 0.004]])
    np.testing.assert_allclose(thumb, [[-0.002, 0.004]])
