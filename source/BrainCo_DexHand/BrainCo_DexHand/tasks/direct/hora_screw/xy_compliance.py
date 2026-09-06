# Copyright (c) 2026, BrainCo.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Passive mount-compliance helpers for the `xy96` arm-hand stage.

The plain XY stage welds every direction it does not actuate: the hand mount is
rigid in world Z, roll, pitch and yaw.  A real arm is not.  When the hand drives
a valve against resistance, the reaction wrench deflects the wrist, and the
**dominant** component is the moment about the valve axis (world Z, i.e. yaw),
followed by the vertical push (world Z translation) that the fingers generate
while squeezing.  A rigid mount absorbs all of it for free, which both flatters
the policy and hides a real sim-to-real gap.

This module describes the passive spring-damper joints that replace those welds.
They carry **no policy action**: their position target stays at zero forever and
an implicit PD drive pulls them back, i.e. each one is a linear spring in
parallel with a damper.  The stiffnesses are per-axis and randomizable, so the
policy cannot rely on one exact mount.

The module deliberately avoids Isaac Lab / USD imports so the numeric contract
is unit-testable without a simulator, exactly like :mod:`xy_stage`.
"""

from __future__ import annotations

from dataclasses import dataclass

# Actuator groups.  Prismatic and revolute compliance are separated because
# their stiffness/damping/armature carry different units (N/m vs N*m/rad).
COMPLIANCE_LINEAR_ACTUATOR_GROUP: str = "xy_compliance_lin"
COMPLIANCE_ANGULAR_ACTUATOR_GROUP: str = "xy_compliance_rot"

# Body inserted between ``stage_y_joint`` and the first compliance joint.  With
# compliance disabled the Y joint attaches straight to the hand base link, which
# is exactly the plain XY topology.
COMPLIANCE_Y_CARRIAGE_BODY_NAME: str = "stage_y_carriage"

# The ``stage_c*`` prefix keeps every compliance joint outside the actuated
# ``stage_[xy]_joint`` pattern; the two groups can never capture each other.
_JOINT_PREFIX: str = "stage_c"


@dataclass(frozen=True)
class ComplianceAxisSpec:
    """One passive mount degree of freedom.

    Attributes:
        key: Short config key (``"z"``, ``"roll"``, ``"pitch"``, ``"yaw"``).
        joint_name: Authored articulation joint name.
        body_name: Rigid body authored *below* this joint.  The last enabled
            axis attaches directly to the hand base link instead.
        is_prismatic: ``True`` for the translational axis, ``False`` for the
            three rotational ones.
        world_axis: Axis of motion in the world-aligned joint frame.
        limit: Symmetric hard stop, metres for prismatic and radians for
            revolute.  It is a mechanical end stop, not the operating range.
        stiffness: Nominal spring constant, N/m or N*m/rad.
        damping: Nominal damper constant, N*s/m or N*m*s/rad.
        armature: Reflected inertia added to the joint, kg or kg*m^2.  Small but
            non-zero values keep a stiff serial chain numerically well behaved.
    """

    key: str
    joint_name: str
    body_name: str
    is_prismatic: bool
    world_axis: str
    limit: float
    stiffness: float
    damping: float
    armature: float


# Nominal values for a UR5e-class arm holding a ~1.5 kg hand at a typical
# valve-turning configuration.  Translational Cartesian stiffness of a
# position-controlled 6-DOF industrial arm sits in the 1e4-1e5 N/m band and its
# rotational stiffness in the 1e2-1e3 N*m/rad band; these sit deliberately at
# the compliant end of that range so the abstraction does not understate the
# gap.  Every value is scaled per environment by the randomization range in the
# task config, so no result should depend on the exact numbers below.
COMPLIANCE_AXES: tuple[ComplianceAxisSpec, ...] = (
    ComplianceAxisSpec(
        key="z",
        joint_name=f"{_JOINT_PREFIX}z_joint",
        body_name=f"{_JOINT_PREFIX}z_carriage",
        is_prismatic=True,
        world_axis="Z",
        limit=0.010,  # m
        stiffness=2.0e4,  # N/m
        damping=4.0e2,  # N*s/m  -> critically damped near 2 kg of moving mass
        armature=0.05,  # kg
    ),
    ComplianceAxisSpec(
        key="roll",
        joint_name=f"{_JOINT_PREFIX}roll_joint",
        body_name=f"{_JOINT_PREFIX}roll_carriage",
        is_prismatic=False,
        world_axis="X",
        limit=0.05,  # rad
        stiffness=4.0e2,  # N*m/rad
        damping=6.0,  # N*m*s/rad
        armature=0.01,  # kg*m^2
    ),
    ComplianceAxisSpec(
        key="pitch",
        joint_name=f"{_JOINT_PREFIX}pitch_joint",
        body_name=f"{_JOINT_PREFIX}pitch_carriage",
        is_prismatic=False,
        world_axis="Y",
        limit=0.05,  # rad
        stiffness=4.0e2,
        damping=6.0,
        armature=0.01,
    ),
    ComplianceAxisSpec(
        # The valve turns about world Z, so this axis carries the reaction
        # moment of the task itself.  It is the single most important weld to
        # release and is intentionally the softest.
        key="yaw",
        joint_name=f"{_JOINT_PREFIX}yaw_joint",
        body_name=f"{_JOINT_PREFIX}yaw_carriage",
        is_prismatic=False,
        world_axis="Z",
        limit=0.08,  # rad
        stiffness=2.5e2,
        damping=5.0,
        armature=0.01,
    ),
)

COMPLIANCE_AXIS_KEYS: tuple[str, ...] = tuple(axis.key for axis in COMPLIANCE_AXES)
_AXES_BY_KEY = {axis.key: axis for axis in COMPLIANCE_AXES}


def resolve_compliance_axes(keys) -> tuple[ComplianceAxisSpec, ...]:
    """Return the enabled compliance axes in canonical chain order.

    The chain order (``z -> roll -> pitch -> yaw``) is fixed regardless of the
    order the keys are given in, so the authored kinematic chain is identical
    for any permutation of the same set.

    Args:
        keys: Iterable of axis keys to enable; may be empty.

    Returns:
        The matching specs, ordered as in :data:`COMPLIANCE_AXES`.

    Raises:
        ValueError: If a key is unknown or repeated.
    """
    requested = list(keys)
    unknown = [key for key in requested if key not in _AXES_BY_KEY]
    if unknown:
        raise ValueError(
            f"Unknown compliance axis keys {unknown}; valid keys are {list(COMPLIANCE_AXIS_KEYS)}"
        )
    if len(set(requested)) != len(requested):
        raise ValueError(f"Duplicate compliance axis keys in {requested}")
    enabled = set(requested)
    return tuple(axis for axis in COMPLIANCE_AXES if axis.key in enabled)


def compliance_body_chain(axes: tuple[ComplianceAxisSpec, ...], base_link_path: str) -> list[str]:
    """Return the body path suffixes the compliance joints connect, in order.

    The returned list has ``len(axes) + 1`` entries: the Y carriage, one body per
    axis except the last, and finally the hand base link.  Joint ``i`` connects
    entry ``i`` to entry ``i + 1``.

    Args:
        axes: Enabled axes in chain order.
        base_link_path: Absolute prim path of the hand base link.

    Returns:
        Body names/paths forming the chain.  Intermediate entries are plain
        names to be authored; the final entry is ``base_link_path``.
    """
    if not axes:
        return [base_link_path]
    nodes: list[str] = [COMPLIANCE_Y_CARRIAGE_BODY_NAME]
    nodes.extend(axis.body_name for axis in axes[:-1])
    nodes.append(base_link_path)
    return nodes


def validate_compliance_config(cfg) -> tuple[ComplianceAxisSpec, ...]:
    """Validate the compliance knobs and return the enabled axes.

    Args:
        cfg: Object exposing ``xy_compliance_axes``,
            ``xy_compliance_stiffness_scale_range``,
            ``xy_compliance_randomize`` and ``xy_compliance_body_mass``.

    Returns:
        The enabled axes in chain order (possibly empty).

    Raises:
        ValueError: If any knob is missing or out of range.
    """
    for name in (
        "xy_compliance_axes",
        "xy_compliance_stiffness_scale_range",
        "xy_compliance_randomize",
        "xy_compliance_body_mass",
        "xy_compliance_body_inertia",
    ):
        if not hasattr(cfg, name):
            raise ValueError(f"Missing required compliance config field: {name}")

    axes = resolve_compliance_axes(cfg.xy_compliance_axes)

    low, high = (float(value) for value in cfg.xy_compliance_stiffness_scale_range)
    if not 0.0 < low <= high:
        raise ValueError(
            "xy_compliance_stiffness_scale_range must satisfy 0 < low <= high, "
            f"got ({low}, {high})"
        )
    if float(cfg.xy_compliance_body_mass) <= 0.0:
        raise ValueError(
            f"xy_compliance_body_mass must be positive, got {cfg.xy_compliance_body_mass}"
        )
    if float(cfg.xy_compliance_body_inertia) <= 0.0:
        raise ValueError(
            f"xy_compliance_body_inertia must be positive, got {cfg.xy_compliance_body_inertia}"
        )
    return axes


def sampled_gain_scales(scale: "object", low: float, high: float):
    """Return ``(stiffness_scale, damping_scale)`` for a uniform sample.

    Damping is scaled by the square root of the stiffness scale so the damping
    ratio ``zeta = D / (2 sqrt(K m))`` stays put while the stiffness varies.  A
    randomization that changed the damping ratio would confound "the mount is
    softer" with "the mount rings", and only the first is intended here.

    Args:
        scale: Tensor of uniform samples in ``[low, high]``.
        low: Lower bound of the sampling range (kept for call-site clarity).
        high: Upper bound of the sampling range.

    Returns:
        The stiffness scale and the matching damping scale.
    """
    del low, high  # Bounds are applied by the caller that draws ``scale``.
    return scale, scale.sqrt()
