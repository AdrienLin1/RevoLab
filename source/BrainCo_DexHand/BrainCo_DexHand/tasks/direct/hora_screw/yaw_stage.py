# Copyright (c) 2026, BrainCo.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""World-Z yaw stage helpers for the hierarchical valve tasks.

Like :mod:`xy_stage` this module is deliberately free of Isaac Lab / USD
imports so the same code paths are exercised by simulator-free unit tests and
by the environment at runtime.  It owns

* the canonical yaw joint / actuator names and name-based DOF resolution,
* the yaw position-target controller (increment + smoothing + angular
  velocity / acceleration / workspace limiting), all in **radians**,
* the yaw workspace margin / boundary saturation measures,
* the numeric contract validation for every ``yaw_*`` configuration knob.

Units
-----
Everything here is SI-with-radians: positions and targets in ``rad``,
velocities in ``rad/s``, accelerations in ``rad/s^2``, jerk in ``rad/s^3`` and
efforts in ``N*m``.  Metres and radians are never mixed into one scale or one
limit: the XY controller keeps its own state and its own limits in
:mod:`xy_stage`, and the two only ever share the dimensionless curriculum
*progress*.

Physical contract
-----------------
``stage_yaw_joint`` rotates the hand mount about **world Z**.  It is a real,
limited revolute joint driven by an effort-limited PD law; nothing here
teleports or re-parents the hand, and the rotation axis passes through the
hand base mount, never through the valve centre.

USD authoring note
------------------
``UsdPhysics.RevoluteJoint`` authors its ``lowerLimit`` / ``upperLimit`` and
``PhysxJointAPI.maxJointVelocity`` in **degrees**, while Isaac Lab / PhysX
articulation joint state is reported in **radians**.  Use
:func:`radians_to_degrees` at authoring time and validate the runtime limits
against the configured radian value with :func:`assert_runtime_yaw_limits`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .xy_stage import (
    NUM_XY_DOFS,
    XY_STAGE_JOINT_NAMES,
    XY_STAGE_WORLD_AXES,
    curriculum_value,
    resolve_dof_indices,
    split_hierarchical_action,
    update_xy_target,
    xy_boundary_saturation,
    xy_pd_effort,
    xy_workspace_margin,
)

# Canonical names authored into the articulation.  Like the prismatic stage
# joints they deliberately do not start with ``right_`` so the hand's
# ``right_.*`` finger actuator group can never match them, and the expression
# below is exact so the ``stage_[xy]_joint`` group can never match yaw either.
YAW_STAGE_JOINT_NAME: str = "stage_yaw_joint"
YAW_STAGE_ACTUATOR_GROUP: str = "yaw_stage"
YAW_STAGE_ACTUATOR_EXPR: str = "stage_yaw_joint"
YAW_STAGE_WORLD_AXIS: str = "Z"
# Rigid body inserted between ``stage_y_joint`` and ``stage_yaw_joint`` in the
# three-DOF chain.  The yaw-only chain does not need it.
YAW_STAGE_CARRIAGE_BODY_NAME: str = "stage_y_carriage"

NUM_YAW_DOFS: int = 1
NUM_XYYAW_DOFS: int = NUM_XY_DOFS + NUM_YAW_DOFS

# Fixed stage joint order for each supported follower width.  Every DOF lookup
# goes through these names; the articulation's internal ordering is never used.
YAW_ONLY_STAGE_JOINT_NAMES: tuple[str, ...] = (YAW_STAGE_JOINT_NAME,)
YAW_ONLY_STAGE_WORLD_AXES: tuple[str, ...] = (YAW_STAGE_WORLD_AXIS,)
XYYAW_STAGE_JOINT_NAMES: tuple[str, ...] = XY_STAGE_JOINT_NAMES + (YAW_STAGE_JOINT_NAME,)
XYYAW_STAGE_WORLD_AXES: tuple[str, ...] = XY_STAGE_WORLD_AXES + (YAW_STAGE_WORLD_AXIS,)

STAGE_JOINT_NAMES_BY_DOF: dict[int, tuple[str, ...]] = {
    NUM_YAW_DOFS: YAW_ONLY_STAGE_JOINT_NAMES,
    NUM_XY_DOFS: XY_STAGE_JOINT_NAMES,
    NUM_XYYAW_DOFS: XYYAW_STAGE_JOINT_NAMES,
}
STAGE_WORLD_AXES_BY_DOF: dict[int, tuple[str, ...]] = {
    NUM_YAW_DOFS: YAW_ONLY_STAGE_WORLD_AXES,
    NUM_XY_DOFS: XY_STAGE_WORLD_AXES,
    NUM_XYYAW_DOFS: XYYAW_STAGE_WORLD_AXES,
}
# Angular stage DOFs, per follower width. Used to report yaw in radians rather
# than in metres and to keep the linear/angular cost terms apart.
STAGE_ANGULAR_DOF_INDICES_BY_DOF: dict[int, tuple[int, ...]] = {
    NUM_YAW_DOFS: (0,),
    NUM_XY_DOFS: (),
    NUM_XYYAW_DOFS: (2,),
}


# Residual play left in a locked stage joint. Small enough to be mechanically
# irrelevant (0.1 mm / 0.1 mrad), large enough that PhysX never receives a
# zero-width limit range.
STAGE_LOCK_TOLERANCE_M: float = 1.0e-4
STAGE_LOCK_TOLERANCE_RAD: float = 1.0e-4


def stage_lock_limits(
    reference_limits: torch.Tensor,
    *,
    num_linear_dofs: int,
    linear_tolerance: float = STAGE_LOCK_TOLERANCE_M,
    angular_tolerance: float = STAGE_LOCK_TOLERANCE_RAD,
) -> torch.Tensor:
    """Return near-zero joint position limits shaped like ``reference_limits``.

    Stage 0 commands a strictly zero stage action, but a zero *target* is not a
    lock: the yaw PD saturates at ``yaw_effort_limit / yaw_pgain`` rad of
    tracking error, so the grasp reaction torque can drag the wrist all the way
    to its hard stop while the follower is still inactive.  Clamping the joint
    **position limits** instead holds it with a PhysX limit constraint, which is
    a hard constraint and is *not* bounded by the drive's ``maxForce``.  The
    configured effort ceiling therefore stays exactly the same in every stage;
    nothing is ever teleported and no drive is ever strengthened.

    The leading ``num_linear_dofs`` entries are prismatic (metres) and the rest
    are revolute (radians), so the two units keep their own tolerances.

    Args:
        reference_limits: Unlocked limits of shape ``(num_envs, num_stage_dofs, 2)``.
        num_linear_dofs: Number of leading prismatic stage DOFs.
        linear_tolerance: Residual play of a locked prismatic joint, metres.
        angular_tolerance: Residual play of a locked revolute joint, radians.

    Returns:
        A tensor shaped like ``reference_limits`` holding every stage DOF at zero.
    """
    num_linear_dofs = int(num_linear_dofs)
    if not 0 <= num_linear_dofs <= reference_limits.shape[-2]:
        raise ValueError(
            f"num_linear_dofs ({num_linear_dofs}) must be within "
            f"[0, {reference_limits.shape[-2]}]"
        )
    linear_tolerance = abs(float(linear_tolerance))
    angular_tolerance = abs(float(angular_tolerance))
    locked = torch.empty_like(reference_limits)
    locked[:, :num_linear_dofs, 0] = -linear_tolerance
    locked[:, :num_linear_dofs, 1] = linear_tolerance
    locked[:, num_linear_dofs:, 0] = -angular_tolerance
    locked[:, num_linear_dofs:, 1] = angular_tolerance
    return locked


def radians_to_degrees(value: float) -> float:
    """Convert a radian quantity to degrees for USD authoring."""
    return math.degrees(float(value))


def degrees_to_radians(value: float) -> float:
    """Convert an authored degree quantity back to radians."""
    return math.radians(float(value))


def stage_joint_names(num_stage_dofs: int) -> tuple[str, ...]:
    """Return the fixed stage joint order for a supported follower width.

    Args:
        num_stage_dofs: Number of stage DOFs (1 = yaw, 2 = XY, 3 = XY+yaw).

    Returns:
        The canonical joint names, in action order.

    Raises:
        ValueError: If the width is not one of the supported stage layouts.
    """
    try:
        return STAGE_JOINT_NAMES_BY_DOF[int(num_stage_dofs)]
    except KeyError:
        raise ValueError(
            f"Unsupported stage width {num_stage_dofs}; supported layouts are "
            f"{sorted(STAGE_JOINT_NAMES_BY_DOF)} "
            "(1 = yaw only, 2 = XY, 3 = XY + yaw)"
        ) from None


def resolve_yaw_dof_index(joint_names) -> int:
    """Return the yaw DOF index resolved purely by joint name."""
    return resolve_dof_indices(joint_names, YAW_ONLY_STAGE_JOINT_NAMES)[0]


def resolve_xyyaw_dof_indices(joint_names) -> list[int]:
    """Return the ``[x, y, yaw]`` stage DOF indices by joint name."""
    return resolve_dof_indices(joint_names, XYYAW_STAGE_JOINT_NAMES)


def resolve_stage_dof_indices(joint_names, num_stage_dofs: int) -> list[int]:
    """Return the stage DOF indices for a supported follower width, by name."""
    return resolve_dof_indices(joint_names, stage_joint_names(num_stage_dofs))


def split_xyyaw_action(
    action: torch.Tensor, num_finger_dofs: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split the joint ``[hand | xy | yaw]`` action into its three blocks.

    Args:
        action: Joint action of shape ``(..., num_finger_dofs + 3)``.
        num_finger_dofs: Number of leading finger action channels.

    Returns:
        ``(finger_action, xy_action, yaw_action)`` with widths
        ``num_finger_dofs``, 2 and 1.

    Raises:
        ValueError: If the trailing dimension does not match the contract.
    """
    finger_action, stage_action = split_hierarchical_action(
        action, num_finger_dofs, NUM_XYYAW_DOFS
    )
    return (
        finger_action,
        stage_action[..., :NUM_XY_DOFS],
        stage_action[..., NUM_XY_DOFS:],
    )


def split_yaw_action(
    action: torch.Tensor, num_finger_dofs: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split the joint ``[hand | yaw]`` action into its two blocks.

    Args:
        action: Joint action of shape ``(..., num_finger_dofs + 1)``.
        num_finger_dofs: Number of leading finger action channels.

    Returns:
        ``(finger_action, yaw_action)`` with widths ``num_finger_dofs`` and 1.

    Raises:
        ValueError: If the trailing dimension does not match the contract.
    """
    return split_hierarchical_action(action, num_finger_dofs, NUM_YAW_DOFS)


def update_yaw_target(
    yaw_action: torch.Tensor,
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
    """Advance the yaw position-target trajectory by one control step.

    Identical in *form* to :func:`xy_stage.update_xy_target` but in angular
    units throughout; the two controllers never share state, scales or limits::

        a_s     = (1 - smoothing) * clamp(a, -1, 1) + smoothing * a_s_prev
        delta   = clamp(action_scale * a_s,
                        prev_delta - accel_limit * dt^2,
                        prev_delta + accel_limit * dt^2)
        delta   = clamp(delta, -velocity_limit * dt, +velocity_limit * dt)
        target  = clamp(prev_target + delta, -workspace, +workspace)

    Args:
        yaw_action: Normalized action of shape ``(..., 1)``.
        prev_target: Previous control-step target, radians.
        prev_delta: Previous applied target increment, radians.
        prev_smoothed_action: Previous smoothed normalized action.
        action_scale: Radians of target increment per unit action.
        workspace: Current software half-range, radians.
        velocity_limit: Maximum commanded target speed, rad/s.
        acceleration_limit: Maximum commanded target acceleration, rad/s^2.
        dt: Control-step duration, seconds.
        smoothing: Weight on the previous smoothed action in ``[0, 1)``.

    Returns:
        ``(target, applied_delta, smoothed_action)`` where ``applied_delta`` is
        measured after the workspace clamp.
    """
    return update_xy_target(
        yaw_action,
        prev_target,
        prev_delta,
        prev_smoothed_action,
        action_scale=action_scale,
        workspace=workspace,
        velocity_limit=velocity_limit,
        acceleration_limit=acceleration_limit,
        dt=dt,
        smoothing=smoothing,
    )


def yaw_pd_effort(
    target: torch.Tensor,
    position: torch.Tensor,
    velocity: torch.Tensor,
    *,
    pgain: float,
    dgain: float,
    effort_limit: float,
) -> torch.Tensor:
    """Return the effort-limited PD torque applied to the yaw joint.

    ``tau = clamp(kp * (target - q) - kd * qdot, -tau_limit, +tau_limit)`` with
    ``kp`` in ``N*m/rad``, ``kd`` in ``N*m*s/rad`` and ``tau_limit`` in
    ``N*m``.  The same bound is configured as the yaw actuator's
    ``effort_limit_sim``, so the joint can never deliver unbounded torque.

    Args:
        target: Commanded joint position, radians.
        position: Measured joint position, radians.
        velocity: Measured joint velocity, rad/s.
        pgain: Proportional gain, N*m/rad.
        dgain: Derivative gain, N*m*s/rad.
        effort_limit: Symmetric torque bound, N*m.

    Returns:
        Applied joint torque with the same shape as ``position``.
    """
    return xy_pd_effort(
        target,
        position,
        velocity,
        pgain=pgain,
        dgain=dgain,
        effort_limit=effort_limit,
    )


def yaw_workspace_margin(position: torch.Tensor, workspace: float) -> torch.Tensor:
    """Return the normalized angular distance to the software boundary.

    ``margin = clamp((W - |q|) / W, 0, 1)`` with ``W`` the **current**
    curriculum workspace in radians: exactly the XY definition, applied to the
    angular DOF.

    Args:
        position: Yaw joint position of shape ``(..., 1)``, radians.
        workspace: Current curriculum workspace half-range, radians.

    Returns:
        Margin with the same shape as ``position``.
    """
    return xy_workspace_margin(position, workspace)


def yaw_boundary_saturation(
    position: torch.Tensor, workspace: float, margin: float = 0.05
) -> torch.Tensor:
    """Return the yaw boundary-saturation measure in ``[0, 1]``.

    ``0`` while the joint stays inside ``(1 - margin) * W``; it then rises
    linearly to ``1`` at the boundary.

    Args:
        position: Yaw joint position of shape ``(..., 1)``, radians.
        workspace: Current curriculum workspace half-range, radians.
        margin: Fraction of the workspace treated as the boundary band.

    Returns:
        Saturation with the same shape as ``position``.
    """
    return xy_boundary_saturation(position, workspace, margin)


def yaw_curriculum_value(initial: float, final: float, progress: float) -> float:
    """Interpolate a yaw curriculum knob at the SHARED stage progress.

    The progress argument is dimensionless and identical to the one used for
    the XY knobs; only the interpolated endpoints carry angular units.
    """
    return curriculum_value(initial, final, progress)


@dataclass(frozen=True)
class YawStageLimits:
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


def validate_yaw_stage_config(cfg) -> YawStageLimits:
    """Validate every yaw stage knob and return the resolved numeric contract.

    Args:
        cfg: Any object exposing the ``yaw_*`` configuration attributes.

    Returns:
        The validated limits, in radians / rad-per-second / N*m.

    Raises:
        ValueError: If any knob is out of range or mutually inconsistent.
    """

    def _get(name: str) -> float:
        if not hasattr(cfg, name):
            raise ValueError(f"Missing required yaw stage config field: {name}")
        return float(getattr(cfg, name))

    limits = YawStageLimits(
        joint_limit=_get("yaw_joint_limit"),
        workspace_initial=_get("yaw_workspace_initial"),
        workspace_final=_get("yaw_workspace_final"),
        action_scale_initial=_get("yaw_action_scale_initial"),
        action_scale_final=_get("yaw_action_scale_final"),
        velocity_limit=_get("yaw_velocity_limit"),
        acceleration_limit=_get("yaw_acceleration_limit"),
        effort_limit=_get("yaw_effort_limit"),
        pgain=_get("yaw_pgain"),
        dgain=_get("yaw_dgain"),
        action_smoothing=_get("yaw_action_smoothing"),
        curriculum_ramp_steps=int(getattr(cfg, "yaw_curriculum_ramp_steps")),
    )

    if limits.joint_limit <= 0.0:
        raise ValueError(f"yaw_joint_limit must be positive, got {limits.joint_limit}")
    if limits.joint_limit >= math.pi:
        raise ValueError(
            f"yaw_joint_limit ({limits.joint_limit} rad) must stay below pi: the yaw "
            "joint is a limited revolute joint, never a continuous one"
        )
    if not 0.0 < limits.workspace_initial <= limits.workspace_final:
        raise ValueError(
            "yaw workspace curriculum must satisfy 0 < yaw_workspace_initial <= "
            f"yaw_workspace_final, got {limits.workspace_initial} / {limits.workspace_final}"
        )
    if limits.workspace_final > limits.joint_limit + 1.0e-12:
        raise ValueError(
            f"yaw_workspace_final ({limits.workspace_final}) must not exceed the asset "
            f"joint hard limit yaw_joint_limit ({limits.joint_limit})"
        )
    if not 0.0 < limits.action_scale_initial <= limits.action_scale_final:
        raise ValueError(
            "yaw action-scale curriculum must satisfy 0 < yaw_action_scale_initial <= "
            f"yaw_action_scale_final, got {limits.action_scale_initial} / "
            f"{limits.action_scale_final}"
        )
    if limits.action_scale_final > limits.workspace_final:
        raise ValueError(
            f"yaw_action_scale_final ({limits.action_scale_final}) must not exceed "
            f"yaw_workspace_final ({limits.workspace_final})"
        )
    if limits.velocity_limit <= 0.0:
        raise ValueError(f"yaw_velocity_limit must be positive, got {limits.velocity_limit}")
    if limits.acceleration_limit <= 0.0:
        raise ValueError(
            f"yaw_acceleration_limit must be positive, got {limits.acceleration_limit}"
        )
    if limits.effort_limit <= 0.0:
        raise ValueError(f"yaw_effort_limit must be positive, got {limits.effort_limit}")
    if limits.pgain <= 0.0 or limits.dgain < 0.0:
        raise ValueError(
            f"yaw_pgain must be positive and yaw_dgain non-negative, got "
            f"{limits.pgain} / {limits.dgain}"
        )
    if not 0.0 <= limits.action_smoothing < 1.0:
        raise ValueError(
            f"yaw_action_smoothing must be in [0, 1), got {limits.action_smoothing}"
        )
    if limits.curriculum_ramp_steps < 0:
        raise ValueError(
            f"yaw_curriculum_ramp_steps must be non-negative, got "
            f"{limits.curriculum_ramp_steps}"
        )
    return limits


def assert_runtime_yaw_limits(
    lower: torch.Tensor, upper: torch.Tensor, joint_limit: float, *, atol: float = 1.0e-4
) -> None:
    """Fail fast unless the runtime yaw DOF limits are the configured RADIANS.

    ``UsdPhysics.RevoluteJoint`` authors its limits in degrees while the
    articulation reports joint state in radians, so this catches a missing or
    doubled unit conversion immediately instead of at training time.

    Args:
        lower: Runtime lower DOF limit(s) reported by the articulation, radians.
        upper: Runtime upper DOF limit(s) reported by the articulation, radians.
        joint_limit: Configured symmetric hard limit, radians.
        atol: Absolute tolerance, radians.

    Raises:
        RuntimeError: If the runtime limits are not ``+/- joint_limit`` rad.
    """
    expected = abs(float(joint_limit))
    if not torch.allclose(lower, torch.full_like(lower, -expected), atol=atol):
        raise RuntimeError(
            f"Yaw lower limit {lower.flatten()[:4].tolist()} rad != -yaw_joint_limit "
            f"({-expected} rad). Authored USD revolute limits are DEGREES; the "
            f"runtime articulation reports RADIANS "
            f"(expected {radians_to_degrees(expected):.4f} deg in USD)."
        )
    if not torch.allclose(upper, torch.full_like(upper, expected), atol=atol):
        raise RuntimeError(
            f"Yaw upper limit {upper.flatten()[:4].tolist()} rad != yaw_joint_limit "
            f"({expected} rad). Authored USD revolute limits are DEGREES; the "
            f"runtime articulation reports RADIANS "
            f"(expected {radians_to_degrees(expected):.4f} deg in USD)."
        )
