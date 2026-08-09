"""Tests for play-time HORA tactile observation perturbations."""

from __future__ import annotations

import sys
import math
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
EXTENSION_PATH = REPO_ROOT / "source" / "BrainCo_DexHand"
if str(EXTENSION_PATH) not in sys.path:
    sys.path.insert(0, str(EXTENSION_PATH))

from BrainCo_DexHand.algo.hora.utils.tactile_robustness import (  # noqa: E402
    TactileObservationPerturber,
)


def _regular_perturber(**kwargs) -> TactileObservationPerturber:
    """Build a two-taxel regular-grid perturbation fixture."""
    defaults = {
        "layout": "regular_grid",
        "policy_mode": "stage1_teacher",
        "num_envs": 1,
        "teacher_frame_dim": 8,
        "student_frame_dim": 6,
        "tactile_priv_offset": 2,
        "tactile_priv_dim": 8,
        "legacy_action_dim": 1,
        "legacy_frame_dim": 7,
        "active_finger_indices": (0, 1),
    }
    defaults.update(kwargs)
    return TactileObservationPerturber(**defaults)


def test_teacher_force_scale_preserves_duration_in_history_and_priv_info() -> None:
    """Scale only force channels in regular teacher frames."""
    perturber = _regular_perturber(force_scale=2.0)
    frame = torch.tensor([[1.0, 2.0, 3.0, 0.25, -1.0, -2.0, -3.0, 0.75]])
    obs = {
        "tactile_hist": frame.unsqueeze(1).repeat(1, 2, 1),
        "priv_info": torch.cat([torch.tensor([[9.0, 8.0]]), frame], dim=-1),
    }

    result = perturber(obs)

    expected = torch.tensor([[2.0, 4.0, 6.0, 0.25, -2.0, -4.0, -6.0, 0.75]])
    torch.testing.assert_close(result["tactile_hist"][:, 0], expected)
    torch.testing.assert_close(result["priv_info"][:, :2], obs["priv_info"][:, :2])
    torch.testing.assert_close(result["priv_info"][:, 2:], expected)
    torch.testing.assert_close(obs["tactile_hist"][:, 0], frame)


def test_spatial_dropout_is_persistent_and_zeros_all_student_channels() -> None:
    """Reuse one full-dropout mask across repeated student observations."""
    perturber = _regular_perturber(
        policy_mode="tactile_student",
        spatial_dropout=1.0,
    )
    tactile = torch.ones((1, 3, 6))

    first = perturber({"student_tactile_hist": tactile})["student_tactile_hist"]
    second = perturber({"student_tactile_hist": tactile})["student_tactile_hist"]

    assert not perturber.keep_mask.any()
    assert torch.count_nonzero(first) == 0
    torch.testing.assert_close(first, second)


def test_binary_flip_recomputes_regular_grid_contact_delta() -> None:
    """Keep ``delta_b`` consistent with contact bits after deterministic flips."""
    perturber = TactileObservationPerturber(
        layout="regular_grid",
        policy_mode="tactile_student",
        num_envs=1,
        teacher_frame_dim=4,
        student_frame_dim=3,
        binary_flip_prob=1.0,
        active_finger_indices=(0,),
    )
    tactile = torch.tensor(
        [[
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 0.2],
            [1.0, 0.0, 0.4],
        ]]
    )

    result = perturber({"student_tactile_hist": tactile})["student_tactile_hist"]

    torch.testing.assert_close(result[0, :, 0], torch.tensor([1.0, 0.0, 0.0]))
    torch.testing.assert_close(result[0, :, 1], torch.tensor([0.0, -1.0, 0.0]))
    torch.testing.assert_close(result[0, :, 2], tactile[0, :, 2])


def test_graph_teacher_scaling_does_not_change_common_or_context_channels() -> None:
    """Scale only the five force channels of graph teacher nodes."""
    perturber = TactileObservationPerturber(
        layout="estimated_official",
        policy_mode="stage1_teacher",
        num_envs=1,
        teacher_frame_dim=14,
        student_frame_dim=9,
        graph_total_nodes=1,
        graph_sensor_counts=(1,),
        force_scale=0.5,
        active_finger_indices=(0,),
    )
    common = torch.tensor([1.0, 0.0, 1.0, 0.4, -0.2])
    force = torch.tensor([0.8, -0.4, 0.2, 0.1, -0.1])
    context = torch.tensor([0.3, -0.3, 1.0, 0.5])
    frame = torch.cat([common, force, context]).view(1, 1, -1)

    result = perturber({"tactile_hist": frame})["tactile_hist"]

    torch.testing.assert_close(result[0, 0, :5], common)
    denominator = math.log1p(5.0 / 0.05)
    raw_normal = 0.05 * torch.expm1(force[0] * denominator)
    raw_shear = torch.sign(force[1:3]) * 0.05 * torch.expm1(
        force[1:3].abs() * denominator
    )
    expected_normal = torch.log1p(0.5 * raw_normal / 0.05) / denominator
    expected_shear = (
        torch.sign(raw_shear)
        * torch.log1p(0.5 * raw_shear.abs() / 0.05)
        / denominator
    )
    torch.testing.assert_close(result[0, 0, 5], expected_normal)
    torch.testing.assert_close(result[0, 0, 6:8], expected_shear)
    torch.testing.assert_close(result[0, 0, 8:10], force[3:5])
    torch.testing.assert_close(result[0, 0, 10:], context)


def test_legacy_contact_channels_follow_force_and_spatial_faults() -> None:
    """Apply pressure faults to the teacher actor's legacy contact magnitudes."""
    perturber = _regular_perturber(force_scale=2.0, spatial_dropout=1.0)
    frame = torch.tensor([0.1, 0.2, 1.0, 2.0, 3.0, 4.0, 5.0])
    obs = frame.repeat(3).view(1, -1)

    result = perturber({"obs": obs})["obs"].view(1, 3, 7)

    torch.testing.assert_close(result[..., :2], obs.view(1, 3, 7)[..., :2])
    assert torch.count_nonzero(result[..., 2:4]) == 0
    torch.testing.assert_close(
        result[..., 4:7],
        obs.view(1, 3, 7)[..., 4:7] * 2.0,
    )


def test_zero_dropout_does_not_advance_torch_rng() -> None:
    """Keep baseline play reproducible when spatial dropout is disabled."""
    torch.manual_seed(123)
    expected = torch.rand(4)
    torch.manual_seed(123)

    _regular_perturber(spatial_dropout=0.0)
    actual = torch.rand(4)

    torch.testing.assert_close(actual, expected)


def test_graph_dropout_scales_contact_ratio_and_zeros_nodes() -> None:
    """Remove failed graph nodes and their contribution to context contact ratio."""
    perturber = TactileObservationPerturber(
        layout="estimated_official",
        policy_mode="tactile_student",
        num_envs=1,
        teacher_frame_dim=24,
        student_frame_dim=14,
        graph_total_nodes=2,
        graph_sensor_counts=(2,),
        active_finger_indices=(0,),
        spatial_dropout=0.5,
    )
    perturber.keep_mask[:] = torch.tensor([[True, False]])
    perturber.finger_keep_ratio = perturber._build_finger_keep_ratio()
    nodes = torch.ones(10)
    context = torch.tensor([0.2, -0.1, 1.0, 0.8])
    frame = torch.cat([nodes, context]).view(1, 1, -1)

    result = perturber({"student_tactile_hist": frame})["student_tactile_hist"]

    assert torch.count_nonzero(result[0, 0, :5]) == 5
    assert torch.count_nonzero(result[0, 0, 5:10]) == 0
    torch.testing.assert_close(result[0, 0, 10:13], context[:3])
    torch.testing.assert_close(result[0, 0, 13], torch.tensor(0.4))


def test_gaussian_noise_is_not_resampled_for_cached_history() -> None:
    """Retain prior noisy samples as the temporal observation window shifts."""
    torch.manual_seed(7)
    perturber = _regular_perturber(noise_std=0.2)
    first_clean = torch.zeros((1, 3, 8))
    first = perturber({"tactile_hist": first_clean})["tactile_hist"]
    second_clean = torch.zeros((1, 3, 8))

    second = perturber({"tactile_hist": second_clean})["tactile_hist"]

    torch.testing.assert_close(second[:, :2], first[:, 1:])
    assert not torch.equal(second[:, -1], first[:, -1])


def test_partial_batch_reset_rebuilds_only_selected_history() -> None:
    """Rebuild reset rows without mismatching full-batch persistent masks."""
    torch.manual_seed(11)
    perturber = TactileObservationPerturber(
        layout="regular_grid",
        policy_mode="stage1_teacher",
        num_envs=2,
        teacher_frame_dim=4,
        student_frame_dim=3,
        noise_std=0.1,
        active_finger_indices=(0,),
    )
    clean = torch.zeros((2, 3, 4))
    first = perturber({"tactile_hist": clean})["tactile_hist"]

    second = perturber(
        {"tactile_hist": clean},
        reset_mask=torch.tensor([True, False]),
    )["tactile_hist"]

    torch.testing.assert_close(second[1, :2], first[1, 1:])
    assert not torch.equal(second[0], first[0])
