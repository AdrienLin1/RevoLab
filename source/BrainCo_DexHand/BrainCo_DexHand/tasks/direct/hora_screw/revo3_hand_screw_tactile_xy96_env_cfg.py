# Copyright (c) 2026, BrainCo.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Config for ``valvedriver_tactile_xy96``: fine-compensation XY stage.

This variant keeps ``valvedriver_tactile_xy`` byte-for-byte intact and retunes
the end-effector stage around one requirement: **the hand turns the valve, the
end effector only trims**.

Two problems with the original stage motivate every number below.

1. *The stage could substitute for the hand.*  Its workspace was +/-50 mm while
   the valve handle circumradius is 35 mm, so ``W/R ~ 1.4``.  A stage that can
   translate further than the object's own radius can simply row the handle
   around, turning the fingers into a one-way ratchet.  Here ``W/R ~ 0.43``.

2. *Oscillation was nearly free.*  Only ``xy_action_scale`` ever bound: the
   velocity clamp (0.0075 m/step) and the acceleration clamp (0.020 m/step of
   increment change) both sat above the 0.005 m/step the action could request,
   so a full reversal was permitted every single control step.  Here the
   commanded peak-to-peak travel of a sustained oscillation at the observed
   ~1.7 Hz drops from 20.3 mm to 2.8 mm.

**The reward is deliberately left alone.**  Every cost term, weight and
normalizer is identical to ``valvedriver_tactile_xy``, so an xy96-vs-xy run is a
controlled comparison of the stage mechanics and of nothing else.  Because three
of the baseline's normalizers were read straight off clamp limits that this task
changes, they are pinned to the baseline's numeric values through explicit cost
references; see section 4 of the config below.

The stage also stops being a rigid weld in the four directions it does not
actuate; see :mod:`xy_compliance`.

The follower observation stays exactly 159-dim and the master interface is
untouched, so an existing 21-D master checkpoint still loads strictly and the
follower architecture is unchanged.
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .assets import make_xy_compliance_hand_cfg
from .revo3_hand_screw_tactile_env_cfg import Revo3HandVavleDriverTactileEnvCfg
from .revo3_hand_screw_tactile_xy_env_cfg import Revo3HandScrewTactileXYMixinCfg
from .xy_compliance import COMPLIANCE_AXIS_KEYS, validate_compliance_config


@configclass
class Revo3HandScrewTactileXY96MixinCfg(Revo3HandScrewTactileXYMixinCfg):
    """Fine-compensation stage with a compliant, non-rigid mount."""

    # ------------------------------------------------------------------
    # 1. Travel envelope: the stage trims, it does not carry the task
    # ------------------------------------------------------------------
    # Hard stop and software workspace, both referred to the 35 mm valve handle
    # circumradius R:  W_final / R = 0.015 / 0.035 = 0.43.  Anything near or
    # above 1.0 lets the stage orbit the handle on its own.
    xy_joint_limit = 0.02  # m   (was 0.05)
    # Stage-1 curriculum is OFF for this task: initial == final, so the
    # workspace is never narrowed below its design value.  See section 6.
    xy_workspace_initial = 0.015  # m   (was 0.004, ramped)
    xy_workspace_final = 0.015  # m   (was 0.05)
    # Observation normalizer must equal the asset hard limit (validator).
    xy_position_obs_scale = 0.02  # m   (was 0.05)

    # ------------------------------------------------------------------
    # 2. Rate envelope: reversal costs time
    # ------------------------------------------------------------------
    # Per control step (dt = 0.05 s) the target may move at most
    # ``action_scale`` = 1 mm, i.e. 0.02 m/s -- one workspace crossing takes
    # 0.75 s, the timescale of a single finger handover (coord_delta_h = 8
    # steps), instead of the previous 0.5 s for a five-times-larger workspace.
    # As with the workspace, initial == final: no Stage-1 ramp on this knob.
    xy_action_scale_initial = 0.002  # m per unit action (was 0.0004, ramped)
    xy_action_scale_final = 0.002 # m per unit action (was 0.005)
    # Velocity clamp now coincides exactly with the action scale
    # (0.02 m/s * 0.05 s = 0.001 m = one full-action increment) instead of
    # sitting 50% above it and never triggering.  Coincident rather than
    # binding is deliberate: it keeps a single speed number that is true both
    # as a clamp and as the cost normalizer below.
    xy_velocity_limit = 0.04  # m/s  (was 0.15)
    xy_velocity_obs_scale = 0.04  # m/s (validator: == xy_velocity_limit)
    # Acceleration clamp: max increment change = a * dt^2 = 0.5 mm per step.
    # Measured effect: a commanded reversal takes 2 control steps instead of 1.
    # It is a mild, physically motivated rate limit -- a real wrist carrying a
    # ~2 kg hand does not reverse in 50 ms -- and NOT the main chatter fix; the
    # action scale and the smoothing filter do that work (see below).  At 0.4
    # and above the clamp stops binding at all, which is where the old 8.0 sat.
    xy_acceleration_limit = 0.4  # m/s^2 (was 8.0)
    # First-order action filter.  For an alternating +/-1 command the steady
    # state amplitude is (1 - s) / (1 + s): 0.33 at s = 0.5, 0.176 at s = 0.7.
    # DC gain stays 1, so slow compensation is untouched.  It is deliberately
    # NOT pushed higher: the filter time constant -1 / ln(s) is 2.8 control
    # steps (0.14 s) here, already a third of one handover window
    # (coord_delta_h = 8 steps), and s = 0.85 would stretch it to 0.31 s --
    # slower than the event the stage is supposed to compensate for.
    #
    # Measured envelope of the whole rate chain, for a sustained +/-1 square
    # wave at the ~1.7 Hz the old policy actually oscillated at, peak-to-peak
    # commanded travel:
    #     old (scale 5 mm, s = 0.5)          20.3 mm   <- larger than 2R/3
    #     scale 1 mm, s = 0.5                 4.1 mm
    #     scale 1 mm, s = 0.7 (this task)     2.8 mm
    # while a one-directional correction still reaches 5.8 mm within a single
    # 8-step handover window (old: 35.0 mm).  The stage keeps the authority it
    # needs to trim and loses the authority to row.
    xy_action_smoothing = 0.7  # (was 0.5)
    xy_joint_velocity_limit_sim = 0.2  # m/s (was 1.0)

    # ------------------------------------------------------------------
    # 3. Mechanics: a real arm end, not a massless slider
    # ------------------------------------------------------------------
    # A UR5e wrist carrying the Revo3 reflects several kg of inertia at the tool
    # frame.  The 0.5 kg massless-slider model made high-frequency motion almost
    # free (F = m A omega^2), which is exactly the behaviour to price out.
    # ``armature`` is the numerically well-behaved way to add reflected inertia.
    xy_carriage_mass = 1.5  # kg   (was 0.5)
    xy_joint_armature = 2.0  # kg   (was 0.0)
    xy_joint_friction = 3.0  # N    (was 0.0) -- dissipates on every reversal
    # Halved so the stage cannot muscle the valve; it still exceeds the contact
    # wrench by a wide margin.  Deflection at full force is 60 / 20000 = 3 mm.
    xy_effort_limit = 60.0  # N    (was 120.0)
    # Re-tuned for the heavier effective mass: with m_eff ~ 4.5 kg these give a
    # damping ratio near 1.  The previous 8000/200 pair would ring at ~7 Hz.
    xy_pgain = 20000.0  # N/m      (was 8000.0)
    xy_dgain = 600.0  # N*s/m      (was 200.0)

    # ------------------------------------------------------------------
    # 4. Costs: BIT-IDENTICAL to valvedriver_tactile_xy
    # ------------------------------------------------------------------
    # This task is a controlled A/B of the stage MECHANICS (travel envelope,
    # rate envelope, mount compliance).  The reward must therefore be the same
    # function of the measured stage state as in ``valvedriver_tactile_xy``:
    # same terms, same weights, same normalizers.  Nothing here may be tuned
    # without breaking that comparison.
    #
    # Three of the normalizers used to be read straight off the clamp limits
    # (``xy_velocity_limit``, ``xy_acceleration_limit``, ``xy_effort_limit``),
    # which are MECHANICS knobs that this task deliberately changes.  They are
    # therefore pinned here to the baseline's numeric values through explicit
    # cost references, so tightening a clamp cannot silently re-weight a cost.
    # The parent env falls back to the clamp limits when a reference is absent,
    # which keeps every other task byte-for-byte unchanged.
    xy_velocity_cost_reference = 0.15  # m/s   == valvedriver_tactile_xy's xy_velocity_limit
    xy_acceleration_cost_reference = 8.0  # m/s^2 == its xy_acceleration_limit
    xy_effort_cost_reference = 120.0  # N     == its xy_effort_limit
    # power_reference = effort_cost_reference * velocity_cost_reference
    #                 = 120.0 * 0.15 = 18.0, the baseline value.
    xy_jerk_reference = 40.0  # m/s^3

    xy_velocity_penalty_scale = -0.05
    xy_acceleration_penalty_scale = -0.02
    xy_jerk_penalty_scale = -0.01
    xy_effort_penalty_scale = -0.05
    xy_power_penalty_scale = -0.02
    xy_boundary_penalty_scale = -0.05
    xy_boundary_margin = 0.10

    # ------------------------------------------------------------------
    # 5. Passive mount compliance (see xy_compliance.py)
    # ------------------------------------------------------------------
    # World Z, roll, pitch and yaw stop being rigid welds.  Yaw matters most:
    # the valve turns about world Z, so that axis carries the task's own
    # reaction moment, which a rigid mount absorbed for free.
    xy_compliance_axes = COMPLIANCE_AXIS_KEYS
    xy_compliance_randomize = True
    # Per-environment stiffness multiplier; damping follows sqrt(scale) so the
    # damping ratio is held while the stiffness varies.
    xy_compliance_stiffness_scale_range = (0.5, 1.5)
    xy_compliance_body_mass = 0.3  # kg per intermediate compliance body
    xy_compliance_body_inertia = 1.0e-3  # kg*m^2 diagonal

    # ------------------------------------------------------------------
    # 6. Stage-1 curriculum: none
    # ------------------------------------------------------------------
    # The follower enters Stage 1 with the whole travel/rate envelope already
    # available.  Two independent switches enforce that, so neither a stale
    # agent yaml nor a resumed checkpoint can re-introduce a ramp:
    #   * ``xy_curriculum_ramp_steps = 0`` makes the trainer's progress latch
    #     to 1.0 at the activation step (see ``curriculum_progress``), and
    #   * ``*_initial == *_final`` above makes the interpolation constant, so
    #     any progress value in [0, 1] resolves to the same numbers.
    # The rationale is that the xy96 envelope IS the restriction: 15 mm of
    # travel is 0.43 R, already far below what would let the stage orbit the
    # handle, so easing into it only delays the behaviour being measured.
    xy_curriculum_ramp_steps = 0

    def __post_init__(self):
        super().__post_init__()
        self._configure_xy_compliance()

    def _configure_xy_compliance(self):
        """Validate the compliance contract and attach its actuator groups."""
        for name in (
            "xy_velocity_cost_reference",
            "xy_acceleration_cost_reference",
            "xy_effort_cost_reference",
        ):
            value = float(getattr(self, name))
            if value <= 0.0:
                raise ValueError(f"{name} ({value}) must be positive")
        axes = validate_compliance_config(self)
        self.robot_cfg = make_xy_compliance_hand_cfg(self.robot_cfg, axes)


@configclass
class Revo3HandVavleDriverTactileXY96EnvCfg(
    Revo3HandScrewTactileXY96MixinCfg,
    Revo3HandVavleDriverTactileEnvCfg,
):
    """Five-finger tactile valve task on a fine-compensation compliant stage."""
