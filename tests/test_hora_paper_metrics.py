"""Test physical-layout-aware HORA paper metrics.

Overview:
The tests validate node-count normalization without launching Isaac Sim.

Quick Start:
    python -m unittest tests.test_hora_paper_metrics
"""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "scripts" / "hora" / "plot_attention_mechanism.py"
SPEC = importlib.util.spec_from_file_location("plot_attention_mechanism", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Could not load {MODULE_PATH}.")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PhysicalLayoutMetricTest(unittest.TestCase):
    """Verify node-count normalization for physical finger interventions."""

    def test_policy_kl_is_normalized_by_physical_sensor_count(self) -> None:
        """Divide whole-finger KL by aligned 31/21-node counts."""
        policy_kl = np.asarray([[31.0, 42.0, 63.0], [15.5, 21.0, 42.0]])
        data = {"tactile_sensor_counts": np.asarray([31, 21, 21])}

        actual = MODULE._policy_kl_per_sensor(data, policy_kl)

        np.testing.assert_allclose(
            actual,
            np.asarray([[1.0, 2.0, 3.0], [0.5, 1.0, 2.0]]),
        )

    def test_stored_normalized_metric_must_match_primary_shape(self) -> None:
        """Reject stale normalized arrays that do not align with policy KL."""
        data = {"policy_kl_per_sensor": np.ones((3, 2))}

        with self.assertRaisesRegex(ValueError, "does not match"):
            MODULE._policy_kl_per_sensor(data, np.ones((4, 2)))


if __name__ == "__main__":
    unittest.main()
