# Copyright (c) 2026, BrainCo.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Environment config for the tactile valve task with a 2-D translation stage.

This variant keeps ``valvedriver_tactile`` byte-for-byte intact and adds a real
two-axis prismatic stage between the fixed world base and the hand mount:

    world (fixed)
      -> stage_x_joint (prismatic, world X)
      -> stage_x_carriage
      -> stage_y_joint (prismatic, world Y)
      -> right_hand_base_link (y carriage / hand mount)
      -> Revo3 palm + 21 finger joints

Only the action space widens (21 -> 23). The 141-dim teacher observation, the
privileged info layout, the 1170-dim teacher tactile frame and the 42-dim
student proprio frame are all unchanged, so an existing 21-D
``valvedriver_tactile_frame813`` master checkpoint loads strictly.
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .assets import make_xy_stage_hand_cfg
from .revo3_hand_screw_tactile_env_cfg import Revo3HandVavleDriverTactileEnvCfg
from .xy_stage import (
    NUM_XY_DOFS,
    XY_STAGE_JOINT_NAMES,
    XY_STAGE_WORLD_AXES,
    validate_high_speed_reward_config,
    validate_xy_stage_config,
)


@configclass
class Revo3HandScrewTactileXYMixinCfg:
    """Add a physically actuated horizontal translation stage to a tactile task."""

    # ---- action layout -------------------------------------------------
    # action[:, :21] -> finger joints, action[:, 21:23] -> XY stage.
    action_space = 21 + NUM_XY_DOFS
    finger_action_space = 21

    # ---- stage asset (hard physical limits) ----------------------------
    # Joint names and the world axes they drive. Every DOF lookup goes through
    # these names, never through the articulation's internal joint ordering.
    xy_stage_joint_names = XY_STAGE_JOINT_NAMES
    xy_stage_world_axes = XY_STAGE_WORLD_AXES
    # Prismatic hard limit authored into the articulation. The software
    # workspace curriculum below never exceeds it.
    xy_joint_limit = 0.05  # m
    xy_carriage_mass = 0.5  # kg; X carriage between the fixed base and the hand
    xy_carriage_inertia = 1.0e-3  # kg*m^2 diagonal (no collider on the carriage)
    xy_joint_armature = 0.0  # kg
    xy_joint_friction = 0.0  # N
    # PhysX-side joint velocity ceiling; the controller limits the *target*
    # velocity far below this, so it only guards against solver blow-ups.
    xy_joint_velocity_limit_sim = 1.0  # m/s

    # ---- stage controller ----------------------------------------------
    # Explicit, effort-limited PD in the environment (identical scheme to the
    # finger torque controller). ``xy_effort_limit`` is enforced both here and
    # by the actuator's ``effort_limit_sim``.
    #
    # Sizing was measured in simulation: dragging the *grasping* hand sideways
    # against the five-finger valve wrap needs 45-80 N, and the grasp behaves
    # like a ~2 kN/m spring. A soft stage (2 kN/m, 50 N) reached only ~3 mm of
    # the 50 mm workspace toward the thumb side, i.e. the follower would have
    # had almost no authority in half the plane. 8 kN/m with a 120 N ceiling
    # reaches the full workspace in every direction while the ceiling still
    # binds 10-25% of the time under load, so the stage never becomes a free
    # position source. Stability: omega_n ~ 73 rad/s, zeta ~ 0.91 at the
    # 240 Hz physics rate.
    xy_pgain = 8000.0  # N/m
    xy_dgain = 200.0  # N*s/m
    xy_effort_limit = 120.0  # N
    # Commanded-target rate limits (never a teleport, only a smoother command).
    xy_velocity_limit = 0.15  # m/s
    xy_acceleration_limit = 8.0  # m/s^2
    # EMA weight on the previous normalized action (0 disables smoothing).
    xy_action_smoothing = 0.5
    # Reuse the finger action-delay draw for the stage command so both the hand
    # and the arm see the same per-env command latency.
    xy_use_action_delay = True

    # ---- workspace / action-scale curriculum ---------------------------
    # Target increment per unit action, and the software workspace half-range.
    xy_action_scale_initial = 0.002  # m per unit action
    xy_action_scale_final = 0.005  # m per unit action
    xy_workspace_initial = 0.01  # m
    xy_workspace_final = 0.05  # m
    # Agent steps over which both ramp from initial to final once the
    # hierarchical curriculum activates the follower.
    xy_curriculum_ramp_steps = 20_000_000

    # ---- observation normalization for the follower --------------------
    # Positions/targets are normalized by the FIXED asset limit so the follower
    # observation stays stationary while the curriculum widens the workspace.
    # Only ``xy_workspace_margin`` uses the current curriculum workspace.
    xy_position_obs_scale = 0.05  # m; kept equal to xy_joint_limit
    xy_velocity_obs_scale = 0.15  # m/s; kept equal to xy_velocity_limit

    # ---- XY physical cost terms (fair-comparison switches) -------------
    # All terms are normalized to O(1) at their respective limits so the scales
    # below read directly as "reward units at full stage usage". Defaults are
    # small but non-zero: the stage must never be a free energy source.
    xy_velocity_penalty_scale = -0.05  # per sum((v / v_limit)^2)
    xy_acceleration_penalty_scale = -0.02  # per sum((a / a_limit)^2)
    xy_jerk_penalty_scale = -0.01  # per sum((j / j_ref)^2)
    xy_effort_penalty_scale = -0.05  # per sum((F / F_limit)^2)
    xy_power_penalty_scale = -0.02  # per sum(|F * v|) / P_ref
    xy_boundary_penalty_scale = -0.05  # per mean axis boundary saturation
    xy_jerk_reference = 40.0  # m/s^3
    # Fraction of the workspace treated as the boundary band for saturation.
    xy_boundary_margin = 0.10

    # ---- optional high-speed rotation shaping --------------------------
    # Disabled by default so run A is an exactly fair comparison against
    # valvedriver_tactile. When enabled, the bonus starts where the base
    # rotate reward saturates (angvel_clip_max) and rises continuously; the
    # 0.8 rad/s hierarchical activation threshold is never a reward gate.
    high_speed_reward_enable = False
    high_speed_target = 6.0  # rad/s
    high_speed_reward_scale = 3.0
    high_speed_penalty_threshold = 8.0  # rad/s

    def __post_init__(self):
        super().__post_init__()
        self._configure_xy_stage()

    def _configure_xy_stage(self):
        """Validate the stage contract and attach the stage actuator group."""
        expected_action_space = int(self.finger_action_space) + NUM_XY_DOFS
        if int(self.action_space) != expected_action_space:
            raise ValueError(
                f"action_space ({self.action_space}) must equal finger_action_space "
                f"({self.finger_action_space}) + {NUM_XY_DOFS} XY channels"
            )
        limits = validate_xy_stage_config(self)
        validate_high_speed_reward_config(self)

        if abs(float(self.xy_position_obs_scale) - limits.joint_limit) > 1.0e-12:
            raise ValueError(
                f"xy_position_obs_scale ({self.xy_position_obs_scale}) must equal "
                f"xy_joint_limit ({limits.joint_limit}) so the follower observation "
                "stays in [-1, 1] and stationary across the curriculum"
            )
        if abs(float(self.xy_velocity_obs_scale) - limits.velocity_limit) > 1.0e-12:
            raise ValueError(
                f"xy_velocity_obs_scale ({self.xy_velocity_obs_scale}) must equal "
                f"xy_velocity_limit ({limits.velocity_limit})"
            )
        if float(self.xy_carriage_mass) <= 0.0 or float(self.xy_carriage_inertia) <= 0.0:
            raise ValueError("xy_carriage_mass and xy_carriage_inertia must be positive")
        if float(self.xy_joint_armature) < 0.0 or float(self.xy_joint_friction) < 0.0:
            raise ValueError("xy_joint_armature and xy_joint_friction must be non-negative")
        if float(self.xy_joint_velocity_limit_sim) < limits.velocity_limit:
            raise ValueError(
                f"xy_joint_velocity_limit_sim ({self.xy_joint_velocity_limit_sim}) must be "
                f">= xy_velocity_limit ({limits.velocity_limit})"
            )
        if float(self.xy_jerk_reference) <= 0.0:
            raise ValueError("xy_jerk_reference must be positive")
        if not 0.0 < float(self.xy_boundary_margin) <= 1.0:
            raise ValueError("xy_boundary_margin must be in (0, 1]")
        penalty_scales = (
            ("xy_velocity_penalty_scale", float(self.xy_velocity_penalty_scale)),
            ("xy_acceleration_penalty_scale", float(self.xy_acceleration_penalty_scale)),
            ("xy_jerk_penalty_scale", float(self.xy_jerk_penalty_scale)),
            ("xy_effort_penalty_scale", float(self.xy_effort_penalty_scale)),
            ("xy_power_penalty_scale", float(self.xy_power_penalty_scale)),
            ("xy_boundary_penalty_scale", float(self.xy_boundary_penalty_scale)),
        )
        for name, value in penalty_scales:
            if value > 0.0:
                raise ValueError(f"{name} ({value}) must be <= 0 (it is a cost)")

        self.robot_cfg = make_xy_stage_hand_cfg(
            self.robot_cfg,
            joint_limit=limits.joint_limit,
            effort_limit=limits.effort_limit,
            velocity_limit=float(self.xy_joint_velocity_limit_sim),
            armature=float(self.xy_joint_armature),
            friction=float(self.xy_joint_friction),
        )


@configclass
class Revo3HandVavleDriverTactileXYEnvCfg(
    Revo3HandScrewTactileXYMixinCfg,
    Revo3HandVavleDriverTactileEnvCfg,
):
    """Five-finger tactile valve task on a two-axis physical translation stage."""
