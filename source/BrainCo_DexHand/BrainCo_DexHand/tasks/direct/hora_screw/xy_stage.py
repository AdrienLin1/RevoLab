# Copyright (c) 2026, BrainCo.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Two-axis prismatic stage helpers for the hierarchical valve task.

This module is deliberately free of Isaac Lab / USD imports so the same code
paths are exercised by simulator-free unit tests and by the environment at
runtime.  It owns three things:

* the canonical stage joint/body names and name-based DOF resolution
  (never rely on the articulation's internal joint ordering),
* the XY position-target controller (increment + smoothing + velocity /
  acceleration / workspace limiting),
* the curriculum interpolation and the follower's XY state channels.

Physical contract
-----------------
``stage_x_joint`` translates the X carriage along **world X** and
``stage_y_joint`` translates the hand mount along **world Y**.  Both are real
prismatic joints with finite effort limits; nothing here teleports the hand.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# Canonical names authored into the articulation.  They intentionally do not
# start with ``right_`` so the hand's ``right_.*`` finger actuator group can
# never match them.
XY_STAGE_JOINT_NAMES: tuple[str, str] = ("stage_x_joint", "stage_y_joint")
XY_STAGE_CARRIAGE_BODY_NAME: str = "stage_x_carriage"
XY_STAGE_ACTUATOR_GROUP: str = "xy_stage"
# Exact two-joint pattern. It must NOT be a wildcard such as ``stage_.*_joint``:
# the XY+yaw task adds ``stage_yaw_joint``, which would otherwise silently
# inherit the XY group's 120 N linear effort limit instead of its own torque
# limit. The two groups stay completely independent.
XY_STAGE_ACTUATOR_EXPR: str = "stage_[xy]_joint"
# World axes driven by the two joints, in the same order as the joint names.
XY_STAGE_WORLD_AXES: tuple[str, str] = ("X", "Y")

NUM_XY_DOFS: int = 2


def resolve_dof_indices(joint_names, wanted_names) -> list[int]:
    """Return DOF indices for ``wanted_names`` resolved purely by joint name.

    Args:
        joint_names: Ordered articulation joint names as reported by physics.
        wanted_names: Joint names to locate, in the desired output order.

    Returns:
        One DOF index per requested joint name, in request order.

    Raises:
        ValueError: If a requested joint name is missing or duplicated.
    """
    ordered = list(joint_names)
    indices: list[int] = []
    for name in wanted_names:
        matches = [index for index, value in enumerate(ordered) if value == name]
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one articulation joint named {name!r}, found "
                f"{len(matches)}; available joints: {ordered}"
            )
        indices.append(matches[0])
    if len(set(indices)) != len(indices):
        raise ValueError(f"Duplicate DOF indices resolved for {tuple(wanted_names)}")
    return indices


def resolve_xy_dof_indices(joint_names, stage_joint_names=XY_STAGE_JOINT_NAMES) -> list[int]:
    """Return the ``[x, y]`` stage DOF indices by joint name."""
    return resolve_dof_indices(joint_names, stage_joint_names)


def split_hierarchical_action(
    action: torch.Tensor,
    num_finger_dofs: int,
    num_xy_dofs: int = NUM_XY_DOFS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split the joint ``[hand | xy]`` action into its two policy blocks.

    Args:
        action: Joint action of shape ``(..., num_finger_dofs + num_xy_dofs)``.
        num_finger_dofs: Number of leading finger action channels.
        num_xy_dofs: Number of trailing horizontal-translation channels.

    Returns:
        The finger action block and the XY action block.

    Raises:
        ValueError: If the trailing dimension does not match the contract.
    """
    expected = int(num_finger_dofs) + int(num_xy_dofs)
    if action.shape[-1] != expected:
        raise ValueError(
            f"Hierarchical action must have {expected} channels "
            f"({num_finger_dofs} finger + {num_xy_dofs} xy), got {action.shape[-1]}"
        )
    return action[..., :num_finger_dofs], action[..., num_finger_dofs:]


def curriculum_value(initial: float, final: float, progress: float) -> float:
    """Linearly interpolate a curriculum knob and clamp progress to ``[0, 1]``."""
    clamped = min(max(float(progress), 0.0), 1.0)
    return float(initial) + (float(final) - float(initial)) * clamped


def curriculum_progress(agent_steps: int, start_step: int, ramp_steps: int) -> float:
    """Return ramp progress in ``[0, 1]`` from global agent steps.

    Args:
        agent_steps: Current global agent-step counter.
        start_step: Agent step at which the ramp started (Stage-1 activation).
        ramp_steps: Number of agent steps the ramp spans.

    Returns:
        ``0.0`` before the ramp, ``1.0`` after it, linear in between.  A
        non-positive ``ramp_steps`` collapses the ramp to its final value.
    """
    if int(ramp_steps) <= 0:
        return 1.0
    progress = (int(agent_steps) - int(start_step)) / float(int(ramp_steps))
    return min(max(progress, 0.0), 1.0)


def update_xy_target(
    xy_action: torch.Tensor,
    prev_target: torch.Tensor,
    prev_delta: torch.Tensor,
    prev_smoothed_action: torch.Tensor,
    *,
    action_scale: float,
    workspace: float,
    velocity_limit: float,
    acceleration_limit: float,
    dt: float,
    smoothing: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Advance the XY position-target trajectory by one control step.

    The normalized action is interpreted as a **position-target increment**::

        a_s     = (1 - smoothing) * clamp(a, -1, 1) + smoothing * a_s_prev
        delta   = clamp(action_scale * a_s,
                        prev_delta - accel_limit * dt^2,
                        prev_delta + accel_limit * dt^2)
        delta   = clamp(delta, -velocity_limit * dt, +velocity_limit * dt)
        target  = clamp(prev_target + delta, -workspace, +workspace)

    The acceleration clamp bounds how fast the commanded velocity may change,
    the velocity clamp bounds the commanded speed, and the workspace clamp is
    the software boundary of the current curriculum stage.  The physical
    actuator additionally enforces a finite effort limit.

    Args:
        xy_action: Normalized action of shape ``(..., 2)``.
        prev_target: Previous control-step target, metres.
        prev_delta: Previous applied target increment, metres.
        prev_smoothed_action: Previous smoothed normalized action.
        action_scale: Metres of target increment per unit action.
        workspace: Current software half-range per axis, metres.
        velocity_limit: Maximum commanded target speed, m/s.
        acceleration_limit: Maximum commanded target acceleration, m/s^2.
        dt: Control-step duration, seconds.
        smoothing: Weight on the previous smoothed action in ``[0, 1)``.

    Returns:
        ``(target, applied_delta, smoothed_action)`` where ``applied_delta`` is
        measured after the workspace clamp.
    """
    action = xy_action.clamp(-1.0, 1.0)
    smoothing = float(smoothing)
    smoothed = (1.0 - smoothing) * action + smoothing * prev_smoothed_action
    raw_delta = float(action_scale) * smoothed

    max_delta_change = float(acceleration_limit) * float(dt) * float(dt)
    delta = torch.clamp(
        raw_delta,
        prev_delta - max_delta_change,
        prev_delta + max_delta_change,
    )
    max_step = float(velocity_limit) * float(dt)
    delta = delta.clamp(-max_step, max_step)

    workspace = float(workspace)
    target = (prev_target + delta).clamp(-workspace, workspace)
    return target, target - prev_target, smoothed


def xy_pd_effort(
    target: torch.Tensor,
    position: torch.Tensor,
    velocity: torch.Tensor,
    *,
    pgain: float,
    dgain: float,
    effort_limit: float,
) -> torch.Tensor:
    """Return the effort-limited PD force applied to the stage joints.

    Identical in form to the finger torque controller::

        F = clamp(kp * (target - q) - kd * qdot, -F_limit, +F_limit)

    The same bound is configured as the actuator's ``effort_limit_sim``, so the
    stage can never deliver unbounded force. This is a real actuator model, not
    a kinematic teleport.

    Args:
        target: Commanded joint positions, metres.
        position: Measured joint positions, metres.
        velocity: Measured joint velocities, m/s.
        pgain: Proportional gain, N/m.
        dgain: Derivative gain, N*s/m.
        effort_limit: Symmetric force bound, newtons.

    Returns:
        Applied joint force with the same shape as ``position``.
    """
    limit = abs(float(effort_limit))
    return (float(pgain) * (target - position) - float(dgain) * velocity).clamp(
        -limit, limit
    )


def xy_workspace_margin(position: torch.Tensor, workspace: float) -> torch.Tensor:
    """Return the normalized per-axis distance to the software boundary.

    ``margin_i = clamp((W - |p_i|) / W, 0, 1)`` with ``W`` the **current**
    curriculum workspace.  It equals ``1`` at the workspace centre, decreases
    continuously and monotonically as the axis approaches the boundary, and is
    exactly ``0`` at or beyond it.  The value is therefore continuous in the
    joint position, bounded in ``[0, 1]`` for every curriculum stage, and
    always defined for both X and Y.

    Args:
        position: XY joint positions of shape ``(..., 2)``, metres.
        workspace: Current curriculum workspace half-range, metres.

    Returns:
        Per-axis margin of the same shape as ``position``.
    """
    workspace = max(float(workspace), 1.0e-6)
    return (1.0 - position.abs() / workspace).clamp(0.0, 1.0)


def xy_boundary_saturation(position: torch.Tensor, workspace: float, margin: float = 0.05) -> torch.Tensor:
    """Return a per-axis boundary-saturation measure in ``[0, 1]``.

    ``0`` while the axis stays inside ``(1 - margin) * W``; it then rises
    linearly to ``1`` at the boundary.  Used both for logging and for the
    optional boundary penalty.

    Args:
        position: XY joint positions of shape ``(..., 2)``, metres.
        workspace: Current curriculum workspace half-range, metres.
        margin: Fraction of the workspace treated as the boundary band.

    Returns:
        Per-axis saturation of the same shape as ``position``.
    """
    margin = min(max(float(margin), 1.0e-6), 1.0)
    usage = 1.0 - xy_workspace_margin(position, workspace)
    return ((usage - (1.0 - margin)) / margin).clamp(0.0, 1.0)


@dataclass(frozen=True)
class XYStageLimits:
    """Validated numeric contract shared by the config, asset and controller."""

    joint_limit: float
    workspace_initial: float
    workspace_final: float
    action_scale_initial: float
    action_scale_final: float
    velocity_limit: float
    acceleration_limit: float
    effort_limit: float
    pgain: float
    dgain: float
    action_smoothing: float
    curriculum_ramp_steps: int


def validate_xy_stage_config(cfg) -> XYStageLimits:
    """Validate every XY stage knob and return the resolved numeric contract.

    Args:
        cfg: Any object exposing the ``xy_*`` configuration attributes.

    Returns:
        The validated limits.

    Raises:
        ValueError: If any knob is out of range or mutually inconsistent.
    """

    def _get(name: str) -> float:
        if not hasattr(cfg, name):
            raise ValueError(f"Missing required XY stage config field: {name}")
        return float(getattr(cfg, name))

    limits = XYStageLimits(
        joint_limit=_get("xy_joint_limit"),
        workspace_initial=_get("xy_workspace_initial"),
        workspace_final=_get("xy_workspace_final"),
        action_scale_initial=_get("xy_action_scale_initial"),
        action_scale_final=_get("xy_action_scale_final"),
        velocity_limit=_get("xy_velocity_limit"),
        acceleration_limit=_get("xy_acceleration_limit"),
        effort_limit=_get("xy_effort_limit"),
        pgain=_get("xy_pgain"),
        dgain=_get("xy_dgain"),
        action_smoothing=_get("xy_action_smoothing"),
        curriculum_ramp_steps=int(getattr(cfg, "xy_curriculum_ramp_steps")),
    )

    if limits.joint_limit <= 0.0:
        raise ValueError(f"xy_joint_limit must be positive, got {limits.joint_limit}")
    if not 0.0 < limits.workspace_initial <= limits.workspace_final:
        raise ValueError(
            "XY workspace curriculum must satisfy 0 < xy_workspace_initial <= "
            f"xy_workspace_final, got {limits.workspace_initial} / {limits.workspace_final}"
        )
    if limits.workspace_final > limits.joint_limit + 1.0e-12:
        raise ValueError(
            f"xy_workspace_final ({limits.workspace_final}) must not exceed the asset "
            f"joint hard limit xy_joint_limit ({limits.joint_limit})"
        )
    if not 0.0 < limits.action_scale_initial <= limits.action_scale_final:
        raise ValueError(
            "XY action-scale curriculum must satisfy 0 < xy_action_scale_initial <= "
            f"xy_action_scale_final, got {limits.action_scale_initial} / "
            f"{limits.action_scale_final}"
        )
    if limits.action_scale_final > limits.workspace_final:
        raise ValueError(
            f"xy_action_scale_final ({limits.action_scale_final}) must not exceed "
            f"xy_workspace_final ({limits.workspace_final})"
        )
    if limits.velocity_limit <= 0.0:
        raise ValueError(f"xy_velocity_limit must be positive, got {limits.velocity_limit}")
    if limits.acceleration_limit <= 0.0:
        raise ValueError(
            f"xy_acceleration_limit must be positive, got {limits.acceleration_limit}"
        )
    if limits.effort_limit <= 0.0:
        raise ValueError(f"xy_effort_limit must be positive, got {limits.effort_limit}")
    if limits.pgain <= 0.0 or limits.dgain < 0.0:
        raise ValueError(
            f"xy_pgain must be positive and xy_dgain non-negative, got "
            f"{limits.pgain} / {limits.dgain}"
        )
    if not 0.0 <= limits.action_smoothing < 1.0:
        raise ValueError(
            f"xy_action_smoothing must be in [0, 1), got {limits.action_smoothing}"
        )
    if limits.curriculum_ramp_steps < 0:
        raise ValueError(
            f"xy_curriculum_ramp_steps must be non-negative, got {limits.curriculum_ramp_steps}"
        )
    return limits


def validate_high_speed_reward_config(cfg) -> None:
    """Validate the optional high-speed rotation reward knobs.

    Raises:
        ValueError: If the shaping is enabled with an inconsistent schedule.
    """
    if not bool(getattr(cfg, "high_speed_reward_enable", False)):
        return
    clip_max = float(getattr(cfg, "angvel_clip_max"))
    target = float(getattr(cfg, "high_speed_target"))
    scale = float(getattr(cfg, "high_speed_reward_scale"))
    threshold = float(getattr(cfg, "high_speed_penalty_threshold"))
    if target <= clip_max:
        raise ValueError(
            f"high_speed_target ({target}) must exceed angvel_clip_max ({clip_max}) so "
            "the extra reward starts exactly where the base rotate reward saturates"
        )
    if scale < 0.0:
        raise ValueError(f"high_speed_reward_scale must be non-negative, got {scale}")
    if threshold < target:
        raise ValueError(
            f"high_speed_penalty_threshold ({threshold}) must be >= high_speed_target ({target})"
        )


def high_speed_reward(
    angular_velocity: torch.Tensor,
    *,
    clip_max: float,
    target: float,
    scale: float,
) -> torch.Tensor:
    """Return the continuous high-speed rotation bonus.

    The bonus is exactly zero at and below ``clip_max`` (where the base
    ``rotate_reward`` saturates), rises linearly to ``scale`` at ``target`` and
    stays flat above it.  There is no gate at 0.8 rad/s: that threshold is only
    used by the global hierarchical curriculum, never by the reward.

    Args:
        angular_velocity: Signed valve angular velocity, rad/s.
        clip_max: Angular velocity at which the base rotate reward saturates.
        target: Angular velocity at which the bonus saturates.
        scale: Maximum bonus.

    Returns:
        Per-environment bonus with the same shape as ``angular_velocity``.
    """
    span = max(float(target) - float(clip_max), 1.0e-6)
    ramp = ((angular_velocity - float(clip_max)) / span).clamp(0.0, 1.0)
    return float(scale) * ramp
