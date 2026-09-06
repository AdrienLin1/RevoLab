# Copyright (c) 2026, BrainCo.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Numeric contract of ``valvedriver_tactile_xy96``.

The task exists to make the end effector *trim* the hand instead of being able
to turn the valve by itself, and to stop the hand mount being a rigid weld.
These tests pin the two properties that encode that intent -- the travel/rate
envelope and the compliance topology -- without needing Isaac Lab.
"""

from __future__ import annotations

import ast
import importlib.util
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
_TASK_DIR = (
    _REPO_ROOT / "source" / "BrainCo_DexHand" / "BrainCo_DexHand" / "tasks" / "direct" / "hora_screw"
)
XY96_CFG_PATH = _TASK_DIR / "revo3_hand_screw_tactile_xy96_env_cfg.py"
XY96_ENV_PATH = _TASK_DIR / "revo3_hand_screw_tactile_xy96_env.py"
XY_CFG_PATH = _TASK_DIR / "revo3_hand_screw_tactile_xy_env_cfg.py"
XY96_YAML = _TASK_DIR / "agents" / "valvedriver_tactile_frame813_xy96.yaml"
XY_YAML = _TASK_DIR / "agents" / "valvedriver_tactile_frame813_xy.yaml"

# 35 mm valve handle circumradius (coord_obj_radius of the tactile valve task).
VALVE_RADIUS = 0.035
CONTROL_DT = 0.05  # 20 Hz control step
HANDOVER_STEPS = 8  # coord_delta_h, the release window the stage must beat


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


xy_stage = _load("xy_stage_under_test", _TASK_DIR / "xy_stage.py")
xy_compliance = _load("xy_compliance_under_test", _TASK_DIR / "xy_compliance.py")


def _class_attributes(path: Path, class_name: str) -> dict:
    """Return literal class-level assignments of one config class."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    class_node = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    attributes: dict = {}
    for statement in class_node.body:
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            continue
        target = statement.targets[0]
        if not isinstance(target, ast.Name):
            continue
        try:
            attributes[target.id] = ast.literal_eval(statement.value)
        except ValueError:
            # Non-literal (e.g. COMPLIANCE_AXIS_KEYS); resolved separately.
            attributes[target.id] = ast.unparse(statement.value)
    return attributes


@pytest.fixture(scope="module")
def xy96_attrs() -> dict:
    return _class_attributes(XY96_CFG_PATH, "Revo3HandScrewTactileXY96MixinCfg")


@pytest.fixture(scope="module")
def xy_attrs() -> dict:
    return _class_attributes(XY_CFG_PATH, "Revo3HandScrewTactileXYMixinCfg")


def _stage_cfg(attrs: dict, base: dict) -> SimpleNamespace:
    merged = dict(base)
    merged.update({key: value for key, value in attrs.items() if key.startswith("xy_")})
    return SimpleNamespace(**merged)


# ----------------------------------------------------------------------
# travel envelope
# ----------------------------------------------------------------------


def test_the_stage_config_satisfies_the_shared_numeric_contract(xy96_attrs, xy_attrs):
    cfg = _stage_cfg(xy96_attrs, xy_attrs)
    limits = xy_stage.validate_xy_stage_config(cfg)
    assert limits.workspace_final == pytest.approx(0.015)
    assert limits.action_scale_final == pytest.approx(0.001)


def test_the_workspace_cannot_cover_the_valve_radius(xy96_attrs, xy_attrs):
    """W/R >= 1 lets the stage orbit the handle instead of trimming the hand."""
    ratio_96 = float(xy96_attrs["xy_workspace_final"]) / VALVE_RADIUS
    ratio_xy = float(xy_attrs["xy_workspace_final"]) / VALVE_RADIUS
    assert ratio_96 < 0.6, f"xy96 W/R = {ratio_96:.2f} is large enough to substitute for the hand"
    assert ratio_xy > 1.0, "the baseline task is expected to be the permissive one"


def test_position_and_velocity_observation_scales_track_their_limits(xy96_attrs):
    assert xy96_attrs["xy_position_obs_scale"] == xy96_attrs["xy_joint_limit"]
    assert xy96_attrs["xy_velocity_obs_scale"] == xy96_attrs["xy_velocity_limit"]


# ----------------------------------------------------------------------
# rate envelope
# ----------------------------------------------------------------------


def _roll_targets(actions, attrs, workspace=None):
    action_scale = float(attrs["xy_action_scale_final"])
    target = torch.zeros(1, 2)
    delta = torch.zeros(1, 2)
    smoothed = torch.zeros(1, 2)
    trace = []
    for action in actions:
        target, delta, smoothed = xy_stage.update_xy_target(
            torch.full((1, 2), float(action)),
            target,
            delta,
            smoothed,
            action_scale=action_scale,
            workspace=float(workspace if workspace is not None else attrs["xy_workspace_final"]),
            velocity_limit=float(attrs["xy_velocity_limit"]),
            acceleration_limit=float(attrs["xy_acceleration_limit"]),
            dt=CONTROL_DT,
            smoothing=float(attrs["xy_action_smoothing"]),
        )
        trace.append((target[0, 0].item(), delta[0, 0].item()))
    return trace


def test_no_clamp_in_the_rate_chain_is_dead(xy96_attrs):
    """Every clamp must be reachable; an unreachable one also kills its cost term."""
    action_scale = float(xy96_attrs["xy_action_scale_final"])
    velocity_step = float(xy96_attrs["xy_velocity_limit"]) * CONTROL_DT
    accel_step = float(xy96_attrs["xy_acceleration_limit"]) * CONTROL_DT**2
    # The velocity clamp coincides with the action scale rather than sitting
    # above it and never triggering, as it did in the baseline task.
    assert velocity_step == pytest.approx(action_scale)
    # The acceleration clamp binds: it allows less than a full-scale increment
    # change per step, so a reversal costs at least one extra control step.
    assert accel_step < action_scale


def test_a_sustained_oscillation_is_far_smaller_than_in_the_baseline(xy96_attrs, xy_attrs):
    """The observed failure was a ~1.7 Hz full-workspace sweep, not fine trim."""
    square = ([1.0] * 6 + [-1.0] * 6) * 20  # 0.6 s period ~ 1.7 Hz

    def peak_to_peak(attrs):
        trace = _roll_targets(square, attrs)
        tail = [target for target, _ in trace[-24:]]
        return max(tail) - min(tail)

    assert peak_to_peak(xy96_attrs) < 0.2 * peak_to_peak(xy_attrs)
    assert peak_to_peak(xy96_attrs) < 0.005  # metres


def test_the_stage_keeps_enough_authority_to_trim_inside_one_handover(xy96_attrs):
    """Suppressing chatter must not suppress the compensation it exists for."""
    trace = _roll_targets([1.0] * HANDOVER_STEPS, xy96_attrs)
    displacement = trace[-1][0]
    assert displacement > 0.003, "cannot trim 3 mm within one handover window"
    assert displacement < float(xy96_attrs["xy_workspace_final"])


def test_crossing_the_workspace_takes_about_one_gait_cycle(xy96_attrs):
    workspace = float(xy96_attrs["xy_workspace_final"])
    trace = _roll_targets([1.0] * 200, xy96_attrs)
    steps = next(index for index, (target, _) in enumerate(trace) if target >= workspace - 1e-9) + 1
    assert 12 <= steps <= 30, f"centre->boundary took {steps} steps ({steps * CONTROL_DT:.2f} s)"


# ----------------------------------------------------------------------
# costs
# ----------------------------------------------------------------------


def _effective(xy_attrs: dict, xy96_attrs: dict) -> dict:
    """Return xy96's effective config: baseline values overridden by its own."""
    merged = dict(xy_attrs)
    merged.update(xy96_attrs)
    return merged


def _reward_parameters(attrs: dict) -> dict:
    """Return exactly the quantities that enter ``_compute_xy_stage_reward``.

    Mirrors the parent env: the three normalizers fall back to their clamp
    limits unless the task pins an explicit cost reference.
    """
    velocity = float(attrs.get("xy_velocity_cost_reference", attrs["xy_velocity_limit"]))
    acceleration = float(
        attrs.get("xy_acceleration_cost_reference", attrs["xy_acceleration_limit"])
    )
    effort = float(attrs.get("xy_effort_cost_reference", attrs["xy_effort_limit"]))
    return {
        "velocity_reference": velocity,
        "acceleration_reference": acceleration,
        "effort_reference": effort,
        "jerk_reference": float(attrs["xy_jerk_reference"]),
        "power_reference": effort * velocity,
        "boundary_margin": float(attrs["xy_boundary_margin"]),
        "scale_velocity": float(attrs["xy_velocity_penalty_scale"]),
        "scale_acceleration": float(attrs["xy_acceleration_penalty_scale"]),
        "scale_jerk": float(attrs["xy_jerk_penalty_scale"]),
        "scale_effort": float(attrs["xy_effort_penalty_scale"]),
        "scale_power": float(attrs["xy_power_penalty_scale"]),
        "scale_boundary": float(attrs["xy_boundary_penalty_scale"]),
        "high_speed_enable": bool(attrs["high_speed_reward_enable"]),
        "high_speed_target": float(attrs["high_speed_target"]),
        "high_speed_scale": float(attrs["high_speed_reward_scale"]),
    }


def test_the_reward_is_identical_to_the_baseline_task(xy96_attrs, xy_attrs):
    """xy96 is a controlled A/B of the stage MECHANICS: the reward must not move.

    Every term, weight and normalizer of the stage cost has to match
    ``valvedriver_tactile_xy``. Three normalizers are read off clamp limits that
    this task deliberately tightens, so they are pinned through explicit cost
    references; this test is what stops a mechanics change from silently
    re-weighting the reward.
    """
    baseline = _reward_parameters(xy_attrs)
    variant = _reward_parameters(_effective(xy_attrs, xy96_attrs))
    differing = {k: (baseline[k], variant[k]) for k in baseline if baseline[k] != variant[k]}
    assert not differing, f"reward parameters drifted from valvedriver_tactile_xy: {differing}"


def test_tightening_a_clamp_cannot_re_weight_a_cost(xy96_attrs, xy_attrs):
    """The pinned references must survive the mechanics changes around them."""
    effective = _effective(xy_attrs, xy96_attrs)
    # The clamps really did tighten ...
    for name in ("xy_velocity_limit", "xy_acceleration_limit", "xy_effort_limit"):
        assert float(effective[name]) < float(xy_attrs[name]), name
    # ... while the cost normalizers stayed at the baseline's numbers.
    assert float(effective["xy_velocity_cost_reference"]) == float(xy_attrs["xy_velocity_limit"])
    assert float(effective["xy_acceleration_cost_reference"]) == float(
        xy_attrs["xy_acceleration_limit"]
    )
    assert float(effective["xy_effort_cost_reference"]) == float(xy_attrs["xy_effort_limit"])


def test_no_extra_cost_term_was_added(xy96_attrs):
    """No penalty scale may exist here that the baseline task does not have."""
    for name in ("xy_action_rate_penalty_scale", "xy_displacement_penalty_scale"):
        assert name not in xy96_attrs, f"{name} would make the reward differ from xy"


def test_every_stage_cost_scale_is_non_positive(xy96_attrs, xy_attrs):
    for name, value in _effective(xy_attrs, xy96_attrs).items():
        if name.startswith("xy_") and name.endswith("_penalty_scale"):
            assert float(value) <= 0.0, f"{name} = {value} is a reward, not a cost"


def test_the_env_override_leaves_the_stage_reward_untouched():
    """The env may log diagnostics, but must return the parent cost verbatim."""
    tree = ast.parse(XY96_ENV_PATH.read_text(encoding="utf-8"))
    reward = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_compute_xy_stage_reward"
    )
    body = ast.unparse(reward)
    assert "stage_reward = super()._compute_xy_stage_reward()" in body
    assert "return stage_reward" in body
    # No arithmetic on the returned value, and no rewriting of the parent's
    # reward bookkeeping.
    for forbidden in (
        "stage_reward +",
        "stage_reward -",
        "stage_reward *",
        "xy_cost/",
        "xy_penalty/",
        "xy/stage_reward",
    ):
        assert forbidden not in body, f"reward override must not contain {forbidden!r}"
    # Finger costs stay finger-only, exactly as in the parent task.
    assert "actuated_dof_indices" not in body


def test_the_env_still_reports_the_oscillation_diagnostics():
    """Removing the penalties must not remove the ability to SEE the behaviour."""
    body = XY96_ENV_PATH.read_text(encoding="utf-8")
    for tag in (
        "xy/action_reversal_rate",
        "xy/action_rate_sq",
        "xy/normalized_displacement_sq",
    ):
        assert tag in body, tag


# ----------------------------------------------------------------------
# mount compliance
# ----------------------------------------------------------------------


def test_compliance_joints_can_never_be_captured_by_the_actuated_pattern():
    pattern = xy_stage.XY_STAGE_ACTUATOR_EXPR
    for axis in xy_compliance.COMPLIANCE_AXES:
        assert re.fullmatch(pattern, axis.joint_name) is None, axis.joint_name


def test_the_compliance_chain_ends_at_the_hand_base_link():
    axes = xy_compliance.resolve_compliance_axes(xy_compliance.COMPLIANCE_AXIS_KEYS)
    nodes = xy_compliance.compliance_body_chain(axes, "BASE")
    assert len(nodes) == len(axes) + 1
    assert nodes[0] == xy_compliance.COMPLIANCE_Y_CARRIAGE_BODY_NAME
    assert nodes[-1] == "BASE"
    assert len(set(nodes)) == len(nodes)


def test_disabling_compliance_reproduces_the_rigid_baseline_topology():
    axes = xy_compliance.resolve_compliance_axes(())
    assert axes == ()
    assert xy_compliance.compliance_body_chain(axes, "BASE") == ["BASE"]


def test_axis_order_is_canonical_regardless_of_config_order():
    forward = xy_compliance.resolve_compliance_axes(("z", "roll", "pitch", "yaw"))
    shuffled = xy_compliance.resolve_compliance_axes(("yaw", "pitch", "z", "roll"))
    assert [axis.key for axis in forward] == [axis.key for axis in shuffled]


def test_the_valve_axis_is_the_softest_released_direction():
    """The valve turns about world Z, so yaw carries the task's reaction moment."""
    rotational = {
        axis.key: axis.stiffness
        for axis in xy_compliance.COMPLIANCE_AXES
        if not axis.is_prismatic
    }
    assert rotational["yaw"] == min(rotational.values())


@pytest.mark.parametrize(
    "field, value",
    [
        ("xy_compliance_stiffness_scale_range", (0.0, 1.0)),
        ("xy_compliance_stiffness_scale_range", (1.5, 0.5)),
        ("xy_compliance_body_mass", 0.0),
        ("xy_compliance_body_inertia", -1.0),
    ],
)
def test_bad_compliance_configs_are_rejected(field, value):
    cfg = SimpleNamespace(
        xy_compliance_axes=xy_compliance.COMPLIANCE_AXIS_KEYS,
        xy_compliance_randomize=True,
        xy_compliance_stiffness_scale_range=(0.5, 1.5),
        xy_compliance_body_mass=0.3,
        xy_compliance_body_inertia=1.0e-3,
    )
    setattr(cfg, field, value)
    with pytest.raises(ValueError):
        xy_compliance.validate_compliance_config(cfg)


def test_unknown_compliance_axis_is_rejected():
    with pytest.raises(ValueError, match="Unknown compliance axis"):
        xy_compliance.resolve_compliance_axes(("z", "surge"))


def test_the_env_never_commands_the_passive_joints():
    source = XY96_ENV_PATH.read_text(encoding="utf-8")
    assert "set_joint_effort_target" not in source
    assert "compliance_dof_indices" in source
    # The passive joints must be excluded from every finger mechanism.
    for mechanism in ("action_mask", "pose_diff_mask", "init_joint_pos"):
        assert f"self.{mechanism}[" in source or f"self.{mechanism}[:," in source


# ----------------------------------------------------------------------
# training config parity
# ----------------------------------------------------------------------


def test_xy96_yaml_differs_from_xy_only_in_documented_ways():
    """An xy96 vs xy run must be an A/B of the environment, not of the learner.

    Three hierarchical keys are allowed to move, and each is a schedule
    decision rather than a learner tuning: whether Stage 2 runs at all, how
    fast the hand must already turn the valve before the stage is unfrozen,
    and whether the Stage-1 envelope is ramped or handed over whole.
    """
    xy96 = yaml.safe_load(XY96_YAML.read_text(encoding="utf-8"))
    xy = yaml.safe_load(XY_YAML.read_text(encoding="utf-8"))
    assert xy96["algo"] == xy["algo"] == "HierarchicalPPO"
    assert xy96["network"] == xy["network"]
    assert xy96["ppo"] == xy["ppo"]
    assert xy96["follower"] == xy["follower"]
    allowed_hierarchical_diffs = {
        "joint_finetune_enable",
        "activation_speed_threshold",
        "xy_curriculum_ramp_steps",
    }
    differing = {
        key
        for key in set(xy96["hierarchical"]) | set(xy["hierarchical"])
        if xy96["hierarchical"].get(key) != xy["hierarchical"].get(key)
    }
    assert differing <= allowed_hierarchical_diffs, differing


def test_stage_one_activates_only_above_the_raised_speed_gate():
    """The Stage 0 -> Stage 1 gate is pinned: 1.1 rad/s, not the baseline 0.8."""
    xy96 = yaml.safe_load(XY96_YAML.read_text(encoding="utf-8"))
    threshold = float(xy96["hierarchical"]["activation_speed_threshold"])
    assert threshold == pytest.approx(1.1)
    assert int(xy96["hierarchical"]["activation_patience"]) >= 1


def test_stage_one_hands_over_the_full_envelope_with_no_curriculum(xy96_attrs):
    """No workspace / action-scale ramp: two independent switches enforce it."""
    xy96 = yaml.safe_load(XY96_YAML.read_text(encoding="utf-8"))
    assert int(xy96["hierarchical"]["xy_curriculum_ramp_steps"]) == 0
    assert int(xy96_attrs["xy_curriculum_ramp_steps"]) == 0
    # A zero-length ramp latches progress to its final value immediately.
    assert xy_stage.curriculum_progress(0, 0, 0) == 1.0
    # ... and the interpolation is constant anyway, at every progress value.
    for progress in (0.0, 0.25, 1.0):
        assert xy_stage.curriculum_value(
            float(xy96_attrs["xy_workspace_initial"]),
            float(xy96_attrs["xy_workspace_final"]),
            progress,
        ) == pytest.approx(float(xy96_attrs["xy_workspace_final"]))
        assert xy_stage.curriculum_value(
            float(xy96_attrs["xy_action_scale_initial"]),
            float(xy96_attrs["xy_action_scale_final"]),
            progress,
        ) == pytest.approx(float(xy96_attrs["xy_action_scale_final"]))


def test_the_joint_finetune_switch_is_explicit():
    """Stage 2 on/off is a live experimental knob; only require it be declared.

    Its value is deliberately NOT pinned here: whether a given run stays in
    follower-only mode or goes on to joint fine-tuning is an experiment
    decision, and the parity test above already allows it to be the single
    hierarchical difference against the baseline yaml.
    """
    xy96 = yaml.safe_load(XY96_YAML.read_text(encoding="utf-8"))
    assert isinstance(xy96["hierarchical"]["joint_finetune_enable"], bool)
