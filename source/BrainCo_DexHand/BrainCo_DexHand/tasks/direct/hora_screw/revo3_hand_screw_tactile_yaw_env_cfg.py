# Copyright (c) 2026, BrainCo.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Environment config for the world-Z yaw stage of the hierarchical valve tasks.

:class:`Revo3HandScrewTactileYawMixinCfg` owns every ``yaw_*`` knob and the yaw
actuator group. It is used twice:

* ``valvedriver_tactile_yaw``   - yaw only, 21 + 1 = 22 action channels;
* ``valvedriver_tactile_xyyaw`` - XY + yaw, 21 + 2 + 1 = 24 action channels
  (see :mod:`revo3_hand_screw_tactile_xyyaw_env_cfg`).

Units are radians throughout. The XY knobs in
:mod:`revo3_hand_screw_tactile_xy_env_cfg` stay in metres and keep their own
limits: no scale, workspace or rate limit is ever shared between the two.
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .assets import make_yaw_stage_hand_cfg
from .revo3_hand_screw_tactile_env_cfg import Revo3HandVavleDriverTactileEnvCfg
from .xy_stage import validate_high_speed_reward_config
from .yaw_stage import (
    NUM_YAW_DOFS,
    YAW_ONLY_STAGE_JOINT_NAMES,
    YAW_ONLY_STAGE_WORLD_AXES,
    validate_yaw_stage_config,
)


@configclass
class Revo3HandScrewTactileYawMixinCfg:
    """Add a physically actuated world-Z yaw joint at the hand mount."""

    # ---- action layout -------------------------------------------------
    # Yaw-only task: action[:, :21] -> fingers, action[:, 21:22] -> yaw.
    action_space = 21 + NUM_YAW_DOFS
    finger_action_space = 21

    # ---- stage identity (also read by HierarchicalPPO) ------------------
    stage_joint_names = YAW_ONLY_STAGE_JOINT_NAMES
    stage_world_axes = YAW_ONLY_STAGE_WORLD_AXES

    # ---- yaw asset (hard physical limits) ------------------------------
    # Revolute hard limit authored into the articulation, RADIANS. The USD
    # RevoluteJoint limit attributes are authored in degrees (converted at
    # authoring time); the runtime articulation reports radians and is checked
    # against this value at startup.
    yaw_joint_limit = 0.70  # rad (~40.1 deg)
    yaw_joint_armature = 0.0  # kg*m^2
    yaw_joint_friction = 0.0  # N*m
    # PhysX-side joint velocity ceiling; the controller limits the *target*
    # angular velocity far below this, so it only guards against solver blow-ups.
    yaw_joint_velocity_limit_sim = 3.0  # rad/s

    # ---- yaw controller -------------------------------------------------
    # Explicit, effort-limited PD in the environment (identical scheme to the
    # finger torque controller and to the XY stage). ``yaw_effort_limit`` is
    # enforced both here and by the yaw actuator's ``effort_limit_sim``.
    #
    # Sizing (measured, not guessed). The hand's inertia about the WORLD-Z yaw
    # axis is J = 9.95e-3 kg*m^2 (1.105 kg, 95 mm radius of gyration: the
    # fingers stick out perpendicular to the axis). With kp = 8 and kd = 0.5
    # that gives omega_n = 28.4 rad/s and zeta = 0.89 - essentially the XY
    # stage's validated 0.91 - so the gains themselves are already right and
    # are deliberately left alone.
    #
    # The first ceiling tried here (0.30 N*m) was the actual defect: the
    # damping term alone reaches kd * yaw_velocity_limit = 0.6 N*m, i.e. 200%
    # of that budget, so above half the velocity limit the proportional term
    # had nothing left and the controller degenerated into a saturated brake.
    # Measured consequence: |tau| pinned at 79% of the ceiling, the follower
    # slammed its action to +/-1 for 74% of all steps, and the joint still
    # barely moved. 1.5 N*m keeps omega_n and zeta unchanged (they depend only
    # on kp/kd/J) while widening the linear tracking band from
    # yaw_effort_limit / yaw_pgain = 38 mrad to 188 mrad, i.e. 75% of the
    # final workspace below. It stays a finite, priced torque source.
    yaw_pgain = 8.0  # N*m/rad
    yaw_dgain = 0.5  # N*m*s/rad
    yaw_effort_limit = 1.5  # N*m
    # Commanded-target rate limits (never a teleport, only a smoother command).
    yaw_velocity_limit = 1.2  # rad/s
    yaw_acceleration_limit = 12.0  # rad/s^2
    # EMA weight on the previous normalized action (0 disables smoothing).
    # Raised from 0.5 after measuring a 74% action-saturation ratio: the extra
    # low-pass keeps a bang-bang follower from injecting high-frequency command
    # noise into a wrist the frozen master cannot compensate for.
    yaw_action_smoothing = 0.8
    # Reuse the finger action-delay draw for the yaw command so the hand and the
    # wrist see the same per-env command latency. The yaw controller keeps its
    # own delayed target buffer; only the per-env delay *sample* is shared.
    yaw_use_action_delay = True

    # ---- workspace / action-scale curriculum ---------------------------
    # Target increment per unit action, and the software workspace half-range.
    #
    # Narrowed from 0.15/0.60 rad and 0.015/0.040 rad after the first long run:
    # the follower commanded full-scale actions 74% of the time yet only used
    # 20% of its travel, and 0.60 rad (34 deg) rotates the contact frame far
    # more than a five-finger wrap tuned against a rigid wrist can absorb -
    # rotation speed decayed from 0.79 to 0.63 rad/s after activation. At the
    # valve's 35 mm circumradius the 0.25 rad final workspace still moves the
    # contact points ~9 mm tangentially, which is the same order as what the XY
    # stage contributes.
    yaw_action_scale_initial = 0.005  # rad per unit action
    yaw_action_scale_final = 0.020  # rad per unit action
    yaw_workspace_initial = 0.05  # rad
    yaw_workspace_final = 0.25  # rad
    # Agent steps over which both ramp. This MUST equal the XY ramp on the
    # combined task: translation and yaw share one dimensionless progress and
    # activate at the same agent step.
    yaw_curriculum_ramp_steps = 20_000_000

    # ---- Stage-0 mechanical lock ---------------------------------------
    # While the hierarchical follower is inactive the policy commands a strictly
    # zero stage action, but a zero *target* does not hold the joint: the yaw PD
    # saturates at yaw_effort_limit / yaw_pgain = 37.5 mrad of error, so the
    # grasp reaction torque drags the wrist to its hard stop (measured: -0.47 rad
    # mean and 35% of envs pinned at -0.70 rad after 34 Stage-0 epochs). Locking
    # the joint POSITION LIMITS to +/- the tolerances below holds every stage DOF
    # with a PhysX limit constraint, which no drive-force ceiling can overrun, so
    # Stage 0 reproduces the rigidly-mounted baseline exactly. The limits are
    # restored the moment the curriculum activates the follower; the effort
    # ceiling is never raised.
    stage_lock_when_follower_inactive = True
    stage_lock_tolerance_m = 1.0e-4  # residual play of a locked prismatic joint
    stage_lock_tolerance_rad = 1.0e-4  # residual play of a locked revolute joint

    # ---- observation normalization for the follower --------------------
    # Position/target are normalized by the FIXED asset limit so the follower
    # observation stays stationary while the curriculum widens the workspace.
    # Only ``yaw_workspace_margin`` uses the current curriculum workspace.
    yaw_position_obs_scale = 0.70  # rad; kept equal to yaw_joint_limit
    yaw_velocity_obs_scale = 1.2  # rad/s; kept equal to yaw_velocity_limit

    # ---- yaw physical cost terms (fair-comparison switches) ------------
    # Same style and magnitude as the XY costs: each term is normalized to O(1)
    # at its own limit, so these scales read directly as "reward units at full
    # yaw usage". Yaw must never be a free energy source, and its torque is
    # deliberately NOT added to the finger torque/work penalties.
    yaw_velocity_penalty_scale = -0.05  # per (w / w_limit)^2
    yaw_acceleration_penalty_scale = -0.02  # per (a / a_limit)^2
    yaw_jerk_penalty_scale = -0.01  # per (j / j_ref)^2
    yaw_effort_penalty_scale = -0.05  # per (tau / tau_limit)^2
    yaw_power_penalty_scale = -0.02  # per |tau * w| / P_ref
    yaw_boundary_penalty_scale = -0.05  # per boundary saturation
    # Jerk is a DOUBLE finite difference of the measured joint velocity at the
    # 20 Hz control rate, so it is amplified by 1/dt^2 = 400 and is dominated by
    # differentiation noise on a light, torque-limited revolute joint. The first
    # value tried here (60 = 5x the acceleration limit, by analogy with the XY
    # stage) was badly mis-scaled: measured yaw jerk ran at 4.7x that reference,
    # so this single term produced 80% of the whole yaw cost budget
    # (-0.222 of -0.278 per step, i.e. -178 reward per 800-step episode) while
    # the XY stage sits at 0.25x its own reference. 600 rad/s^3 puts the
    # measured jerk at ~0.5x, which is the same normalization regime as XY.
    yaw_jerk_reference = 600.0  # rad/s^3
    # Fraction of the workspace treated as the boundary band for saturation.
    yaw_boundary_margin = 0.10

    # ---- optional high-speed rotation shaping --------------------------
    # Mirrors the XY task so the yaw ablation is reward-identical to it.
    high_speed_reward_enable = False
    high_speed_target = 6.0  # rad/s
    high_speed_reward_scale = 3.0
    high_speed_penalty_threshold = 8.0  # rad/s

    def __post_init__(self):
        super().__post_init__()
        self._configure_yaw_stage()

    def _stage_action_space(self) -> int:
        """Return the number of trailing stage action channels of this task."""
        return NUM_YAW_DOFS

    def _configure_yaw_stage(self):
        """Validate the yaw contract and attach the yaw actuator group."""
        expected_action_space = int(self.finger_action_space) + self._stage_action_space()
        if int(self.action_space) != expected_action_space:
            raise ValueError(
                f"action_space ({self.action_space}) must equal finger_action_space "
                f"({self.finger_action_space}) + {self._stage_action_space()} stage channels"
            )
        limits = validate_yaw_stage_config(self)
        validate_high_speed_reward_config(self)

        if abs(float(self.yaw_position_obs_scale) - limits.joint_limit) > 1.0e-12:
            raise ValueError(
                f"yaw_position_obs_scale ({self.yaw_position_obs_scale}) must equal "
                f"yaw_joint_limit ({limits.joint_limit}) so the follower observation "
                "stays in [-1, 1] and stationary across the curriculum"
            )
        if abs(float(self.yaw_velocity_obs_scale) - limits.velocity_limit) > 1.0e-12:
            raise ValueError(
                f"yaw_velocity_obs_scale ({self.yaw_velocity_obs_scale}) must equal "
                f"yaw_velocity_limit ({limits.velocity_limit})"
            )
        if float(self.yaw_joint_armature) < 0.0 or float(self.yaw_joint_friction) < 0.0:
            raise ValueError("yaw_joint_armature and yaw_joint_friction must be non-negative")
        if float(self.yaw_joint_velocity_limit_sim) < limits.velocity_limit:
            raise ValueError(
                f"yaw_joint_velocity_limit_sim ({self.yaw_joint_velocity_limit_sim}) must be "
                f">= yaw_velocity_limit ({limits.velocity_limit})"
            )
        if float(self.yaw_jerk_reference) <= 0.0:
            raise ValueError("yaw_jerk_reference must be positive")
        if not 0.0 < float(self.yaw_boundary_margin) <= 1.0:
            raise ValueError("yaw_boundary_margin must be in (0, 1]")
        for name in ("stage_lock_tolerance_m", "stage_lock_tolerance_rad"):
            value = float(getattr(self, name))
            if not 0.0 < value <= 1.0e-2:
                raise ValueError(
                    f"{name} ({value}) must be a small positive residual play in "
                    "(0, 1e-2]; it is a mechanical lock, not a workspace"
                )
        penalty_scales = (
            ("yaw_velocity_penalty_scale", float(self.yaw_velocity_penalty_scale)),
            ("yaw_acceleration_penalty_scale", float(self.yaw_acceleration_penalty_scale)),
            ("yaw_jerk_penalty_scale", float(self.yaw_jerk_penalty_scale)),
            ("yaw_effort_penalty_scale", float(self.yaw_effort_penalty_scale)),
            ("yaw_power_penalty_scale", float(self.yaw_power_penalty_scale)),
            ("yaw_boundary_penalty_scale", float(self.yaw_boundary_penalty_scale)),
        )
        for name, value in penalty_scales:
            if value > 0.0:
                raise ValueError(f"{name} ({value}) must be <= 0 (it is a cost)")
        # Translation and yaw must ramp on ONE shared progress. Different ramp
        # lengths would desynchronize them even though they activate together.
        xy_ramp = getattr(self, "xy_curriculum_ramp_steps", None)
        if xy_ramp is not None and int(xy_ramp) != limits.curriculum_ramp_steps:
            raise ValueError(
                f"yaw_curriculum_ramp_steps ({limits.curriculum_ramp_steps}) must equal "
                f"xy_curriculum_ramp_steps ({int(xy_ramp)}): both stage DOFs share one "
                "dimensionless curriculum progress"
            )

        self.robot_cfg = make_yaw_stage_hand_cfg(
            self.robot_cfg,
            joint_limit=limits.joint_limit,
            effort_limit=limits.effort_limit,
            velocity_limit=float(self.yaw_joint_velocity_limit_sim),
            armature=float(self.yaw_joint_armature),
            friction=float(self.yaw_joint_friction),
        )


@configclass
class Revo3HandVavleDriverTactileYawEnvCfg(
    Revo3HandScrewTactileYawMixinCfg,
    Revo3HandVavleDriverTactileEnvCfg,
):
    """Five-finger tactile valve task on a single world-Z yaw joint.

    This is the yaw-only ablation of ``valvedriver_tactile_xyyaw``: the hand
    mount can rotate about world Z but cannot translate, so the policy emits
    ``21 + 1 = 22`` channels and the follower is 1-D.
    """
