# Copyright (c) 2026, BrainCo.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tactile valve task with a physically actuated two-axis translation stage.

The environment extends :class:`Revo3HandScrewTactileEnv` with a real prismatic
X/Y stage between the fixed world base and the hand mount::

    world (fixed by the asset's global root joint)
      -> stage_x_joint  (prismatic, world X, limits +/- xy_joint_limit)
      -> stage_x_carriage
      -> stage_y_joint  (prismatic, world Y, limits +/- xy_joint_limit)
      -> right_hand_base_link  (y carriage / hand mount)
      -> Revo3 palm and its 21 finger joints

There is no root teleport anywhere in the control loop: the hand root pose is
only written at reset (unchanged base behaviour), and horizontal motion is
produced exclusively by effort-limited PD torques on the two prismatic joints.

Action layout (fixed, resolved by joint NAME, never by articulation ordering)::

    action[:, :21]   -> finger joints
    action[:, 21:23] -> XY stage
"""

from __future__ import annotations

import torch
from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

from .revo3_hand_screw_tactile_env import Revo3HandScrewTactileEnv
from .revo3_hand_screw_tactile_xy_env_cfg import Revo3HandScrewTactileXYMixinCfg
from .xy_stage import (
    NUM_XY_DOFS,
    XY_STAGE_CARRIAGE_BODY_NAME,
    XY_STAGE_JOINT_NAMES,
    XY_STAGE_WORLD_AXES,
    curriculum_value,
    high_speed_reward,
    resolve_xy_dof_indices,
    split_hierarchical_action,
    update_xy_target,
    xy_boundary_saturation,
    xy_pd_effort,
    xy_workspace_margin,
)

_BASE_LINK_NAME = "right_hand_base_link"
_WORLD_LINK_NAME = "world"
_BASE_FIXED_JOINT_NAME = "right_hand_base_joint"
_JOINTS_SCOPE_NAME = "joints"


class Revo3HandScrewTactileXYEnv(Revo3HandScrewTactileEnv):
    """Tactile valve environment whose hand rides on two prismatic joints."""

    cfg: Revo3HandScrewTactileXYMixinCfg

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # ---- name-based DOF resolution (never rely on joint ordering) ----
        self.xy_dof_indices = resolve_xy_dof_indices(self.hand.joint_names)
        self.xy_dof_index_tensor = torch.tensor(
            self.xy_dof_indices, dtype=torch.long, device=self.device
        )
        if self.num_robot_dofs != self.num_finger_dofs + NUM_XY_DOFS:
            raise RuntimeError(
                f"Expected {self.num_finger_dofs} finger DOFs + {NUM_XY_DOFS} stage DOFs "
                f"= {self.num_finger_dofs + NUM_XY_DOFS} robot DOFs, got {self.num_robot_dofs}. "
                f"Articulation joints: {self.hand.joint_names}"
            )
        overlap = set(self.xy_dof_indices) & set(self.finger_dof_indices)
        if overlap:
            raise RuntimeError(f"Stage DOFs overlap finger DOFs at indices {sorted(overlap)}")

        # Keep the stage out of every finger-specific mechanism: the finger
        # action mask, the reset joint noise and the pose-diff reference.
        self.action_mask[self.xy_dof_index_tensor] = 0.0
        self.pose_diff_mask[self.xy_dof_index_tensor] = 0.0
        self.init_joint_pos[:, self.xy_dof_index_tensor] = 0.0

        # ``dof_limits_scale`` is a finger-safety margin. The stage must keep
        # its authored hard limit so the final curriculum workspace is reachable.
        raw_limits = self.hand.root_physx_view.get_dof_limits().to(self.device)
        self.hand_dof_lower_limits[:, self.xy_dof_index_tensor] = raw_limits[
            ..., 0
        ][:, self.xy_dof_index_tensor]
        self.hand_dof_upper_limits[:, self.xy_dof_index_tensor] = raw_limits[
            ..., 1
        ][:, self.xy_dof_index_tensor]
        self._validate_stage_joint_limits()

        # ---- stage controller buffers (all at control-step rate) ----
        zeros = torch.zeros((self.num_envs, NUM_XY_DOFS), device=self.device, dtype=torch.float)
        self.xy_target = zeros.clone()
        self.xy_delayed_target = zeros.clone()
        self.xy_prev_delta = zeros.clone()
        self.xy_smoothed_action = zeros.clone()
        self.xy_executed_action = zeros.clone()
        self.xy_effort = zeros.clone()
        self.xy_prev_velocity = zeros.clone()
        self.xy_prev_acceleration = zeros.clone()

        # ---- curriculum state (owned by the trainer, mirrored here) ----
        self.xy_curriculum_progress = 0.0
        self.xy_workspace_current = float(self.cfg.xy_workspace_initial)
        self.xy_action_scale_current = float(self.cfg.xy_action_scale_initial)

        print(
            "[INFO] XY stage ready: joints "
            f"{tuple(XY_STAGE_JOINT_NAMES)} -> DOF indices {self.xy_dof_indices} "
            f"(world axes {tuple(XY_STAGE_WORLD_AXES)}), hard limit "
            f"+/-{float(self.cfg.xy_joint_limit):.3f} m, effort limit "
            f"{float(self.cfg.xy_effort_limit):.1f} N.",
            flush=True,
        )

    # ------------------------------------------------------------------
    # action/DOF bookkeeping
    # ------------------------------------------------------------------

    def _num_extra_action_dofs(self) -> int:
        return NUM_XY_DOFS

    def _validate_stage_joint_limits(self) -> None:
        """Fail fast if the authored prismatic limits do not match the config."""
        expected = float(self.cfg.xy_joint_limit)
        lower = self.hand_dof_lower_limits[:, self.xy_dof_index_tensor]
        upper = self.hand_dof_upper_limits[:, self.xy_dof_index_tensor]
        if not torch.allclose(lower, torch.full_like(lower, -expected), atol=1.0e-6):
            raise RuntimeError(
                f"Stage lower limits {lower[0].tolist()} != -xy_joint_limit ({-expected})"
            )
        if not torch.allclose(upper, torch.full_like(upper, expected), atol=1.0e-6):
            raise RuntimeError(
                f"Stage upper limits {upper[0].tolist()} != xy_joint_limit ({expected})"
            )

    # ------------------------------------------------------------------
    # scene authoring
    # ------------------------------------------------------------------

    def _author_robot_stage_overrides(self) -> None:
        """Author the two prismatic stage joints on the env_0 hand prototype.

        Runs after the hand is spawned and before ``clone_environments`` so all
        cloned environments inherit the same articulation topology. The joint
        frames are rotated by the inverse of the hand root orientation, which
        makes ``stage_x_joint`` translate along **world X** and
        ``stage_y_joint`` along **world Y** regardless of the palm-down grasp
        orientation configured in ``assets.py``.
        """
        stage = self.scene.stage
        hand_path = f"{self.scene.env_prim_paths[0]}/hand"
        hand_prim = stage.GetPrimAtPath(hand_path)
        if not hand_prim.IsValid():
            raise RuntimeError(f"Hand prim not found at '{hand_path}'.")

        world_path = f"{hand_path}/{_WORLD_LINK_NAME}"
        base_link_path = f"{hand_path}/{_BASE_LINK_NAME}"
        joints_scope = f"{hand_path}/{_JOINTS_SCOPE_NAME}"
        for path in (world_path, base_link_path, joints_scope):
            if not stage.GetPrimAtPath(path).IsValid():
                raise RuntimeError(
                    f"XY stage authoring requires '{path}' in the Revo3 hand USD."
                )

        # 1) Remove the rigid world -> hand base weld: the stage replaces it.
        base_joint_path = f"{joints_scope}/{_BASE_FIXED_JOINT_NAME}"
        base_joint_prim = stage.GetPrimAtPath(base_joint_path)
        if not base_joint_prim.IsValid():
            raise RuntimeError(
                f"Expected the fixed hand base joint at '{base_joint_path}'."
            )
        base_joint_prim.SetActive(False)

        # 2) X carriage rigid body (no collider; gravity disabled like the hand).
        carriage_path = f"{hand_path}/{XY_STAGE_CARRIAGE_BODY_NAME}"
        carriage = UsdGeom.Xform.Define(stage, carriage_path)
        carriage.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))
        carriage.AddOrientOp(UsdGeom.XformOp.PrecisionFloat).Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        carriage.AddScaleOp().Set(Gf.Vec3d(1.0, 1.0, 1.0))
        carriage_prim = carriage.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(carriage_prim).CreateRigidBodyEnabledAttr().Set(True)
        mass_api = UsdPhysics.MassAPI.Apply(carriage_prim)
        mass_api.CreateMassAttr().Set(float(self.cfg.xy_carriage_mass))
        inertia = float(self.cfg.xy_carriage_inertia)
        mass_api.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(inertia, inertia, inertia))
        mass_api.CreatePrincipalAxesAttr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        physx_body = PhysxSchema.PhysxRigidBodyAPI.Apply(carriage_prim)
        physx_body.CreateDisableGravityAttr().Set(True)
        physx_body.CreateRetainAccelerationsAttr().Set(False)

        # 3) Two prismatic joints with world-aligned joint frames.
        root_quat = tuple(float(value) for value in self.cfg.robot_cfg.init_state.rot)
        if len(root_quat) != 4:
            raise RuntimeError(
                f"Hand init_state.rot must be a (w, x, y, z) quaternion, got {root_quat}"
            )
        # Inverse of the unit root quaternion: it cancels the palm-down grasp
        # rotation so the joint frame coincides with the world frame.
        inverse_root = Gf.Quatf(
            root_quat[0], -root_quat[1], -root_quat[2], -root_quat[3]
        )
        limit = float(self.cfg.xy_joint_limit)
        effort_limit = float(self.cfg.xy_effort_limit)
        max_velocity = float(self.cfg.xy_joint_velocity_limit_sim)
        chain = (
            (XY_STAGE_JOINT_NAMES[0], XY_STAGE_WORLD_AXES[0], world_path, carriage_path),
            (XY_STAGE_JOINT_NAMES[1], XY_STAGE_WORLD_AXES[1], carriage_path, base_link_path),
        )
        for joint_name, axis, body0, body1 in chain:
            joint_path = f"{joints_scope}/{joint_name}"
            if stage.GetPrimAtPath(joint_path).IsValid():
                raise RuntimeError(f"Stage joint '{joint_path}' already exists.")
            joint = UsdPhysics.PrismaticJoint.Define(stage, joint_path)
            joint.CreateBody0Rel().SetTargets([body0])
            joint.CreateBody1Rel().SetTargets([body1])
            joint.CreateAxisAttr().Set(axis)
            joint.CreateLowerLimitAttr().Set(-limit)
            joint.CreateUpperLimitAttr().Set(limit)
            joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
            joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
            joint.CreateLocalRot0Attr().Set(inverse_root)
            joint.CreateLocalRot1Attr().Set(inverse_root)
            joint.CreateCollisionEnabledAttr().Set(False)
            joint.CreateExcludeFromArticulationAttr().Set(False)
            joint.CreateJointEnabledAttr().Set(True)
            # Force drive with zero implicit PD: the environment applies explicit
            # effort targets, exactly like the finger joints.
            drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
            drive.CreateTypeAttr().Set("force")
            drive.CreateStiffnessAttr().Set(0.0)
            drive.CreateDampingAttr().Set(0.0)
            drive.CreateMaxForceAttr().Set(effort_limit)
            drive.CreateTargetPositionAttr().Set(0.0)
            drive.CreateTargetVelocityAttr().Set(0.0)
            PhysxSchema.PhysxJointAPI.Apply(joint.GetPrim()).CreateMaxJointVelocityAttr().Set(
                max_velocity
            )

        print(
            "[INFO] XY stage authored: "
            f"{world_path} -[{XY_STAGE_JOINT_NAMES[0]}/{XY_STAGE_WORLD_AXES[0]}]-> "
            f"{carriage_path} -[{XY_STAGE_JOINT_NAMES[1]}/{XY_STAGE_WORLD_AXES[1]}]-> "
            f"{base_link_path}; disabled '{base_joint_path}'.",
            flush=True,
        )

    # ------------------------------------------------------------------
    # curriculum
    # ------------------------------------------------------------------

    def set_xy_curriculum_progress(self, progress: float) -> tuple[float, float]:
        """Set the global workspace / action-scale ramp progress.

        The hierarchical trainer owns the schedule so it can be checkpointed and
        restored exactly. This is a global, per-rollout setting: never per
        environment and never per timestep.

        Args:
            progress: Ramp progress in ``[0, 1]``; values outside are clamped.

        Returns:
            The resolved ``(workspace, action_scale)`` for this rollout.
        """
        self.xy_curriculum_progress = min(max(float(progress), 0.0), 1.0)
        self.xy_workspace_current = curriculum_value(
            self.cfg.xy_workspace_initial,
            self.cfg.xy_workspace_final,
            self.xy_curriculum_progress,
        )
        self.xy_action_scale_current = curriculum_value(
            self.cfg.xy_action_scale_initial,
            self.cfg.xy_action_scale_final,
            self.xy_curriculum_progress,
        )
        return self.xy_workspace_current, self.xy_action_scale_current

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        finger_actions, xy_actions = split_hierarchical_action(
            actions, self.num_finger_dofs, NUM_XY_DOFS
        )
        # Fingers keep the untouched base-task path (mask, delta target, delay).
        super()._pre_physics_step(finger_actions)
        self._update_xy_stage_targets(xy_actions)

    def _update_xy_stage_targets(self, xy_actions: torch.Tensor) -> None:
        """Advance the stage position target for one control step."""
        executed = torch.clamp(xy_actions, -1.0, 1.0)
        # The previous control-step target is what the delayed physics substeps
        # keep applying, mirroring the finger action-delay model.
        self.xy_delayed_target = self.xy_target.clone()
        target, delta, smoothed = update_xy_target(
            executed,
            self.xy_target,
            self.xy_prev_delta,
            self.xy_smoothed_action,
            action_scale=self.xy_action_scale_current,
            workspace=self.xy_workspace_current,
            velocity_limit=float(self.cfg.xy_velocity_limit),
            acceleration_limit=float(self.cfg.xy_acceleration_limit),
            dt=float(self.step_dt),
            smoothing=float(self.cfg.xy_action_smoothing),
        )
        self.xy_target = target
        self.xy_prev_delta = delta
        self.xy_smoothed_action = smoothed
        self.xy_executed_action = executed

    def _apply_action(self) -> None:
        substep_index = self._physics_substep_idx
        # Fingers first: this refreshes the articulation state, applies the
        # finger efforts and advances the substep/target bookkeeping.
        super()._apply_action()

        if bool(self.cfg.xy_use_action_delay):
            switch_at = self.action_delay * float(self.cfg.decimation)
            use_delayed = (substep_index < switch_at).unsqueeze(-1)
            applied_target = torch.where(use_delayed, self.xy_delayed_target, self.xy_target)
        else:
            applied_target = self.xy_target

        position = self.hand_dof_pos[:, self.xy_dof_index_tensor]
        velocity = self.hand_dof_vel[:, self.xy_dof_index_tensor]
        effort = xy_pd_effort(
            applied_target,
            position,
            velocity,
            pgain=float(self.cfg.xy_pgain),
            dgain=float(self.cfg.xy_dgain),
            effort_limit=float(self.cfg.xy_effort_limit),
        )
        self.hand.set_joint_effort_target(effort, joint_ids=self.xy_dof_indices)
        self.torques[:, self.xy_dof_index_tensor] = effort
        # Effort of the most recent physics substep; used for the effort/power
        # cost terms and the xy/* diagnostics of this control step.
        self.xy_effort = effort

    # ------------------------------------------------------------------
    # observations
    # ------------------------------------------------------------------

    def _xy_observation_channels(self) -> dict[str, torch.Tensor]:
        """Return the stage self-state channels for the follower policy.

        Positions and targets are normalized by the FIXED asset joint limit so
        the follower observation is stationary while the curriculum widens the
        software workspace; only ``xy_workspace_margin`` uses the current
        curriculum workspace. Velocities are normalized by the commanded
        velocity limit and clipped to +/-2 to stay bounded during contact
        transients.
        """
        position = self.hand_dof_pos[:, self.xy_dof_index_tensor]
        velocity = self.hand_dof_vel[:, self.xy_dof_index_tensor]
        position_scale = max(float(self.cfg.xy_position_obs_scale), 1.0e-6)
        velocity_scale = max(float(self.cfg.xy_velocity_obs_scale), 1.0e-6)
        channels = {
            "xy_position": (position / position_scale).clamp(-1.0, 1.0),
            "xy_velocity": (velocity / velocity_scale).clamp(-2.0, 2.0),
            "xy_target": (self.xy_target / position_scale).clamp(-1.0, 1.0),
            "previous_xy_action": self.xy_executed_action.clone(),
            "xy_workspace_margin": xy_workspace_margin(position, self.xy_workspace_current),
        }
        channels["xy_state"] = torch.cat(
            [
                channels["xy_position"],
                channels["xy_velocity"],
                channels["xy_target"],
                channels["previous_xy_action"],
                channels["xy_workspace_margin"],
            ],
            dim=-1,
        )
        return channels

    def _get_observations(self) -> dict:
        obs_dict = super()._get_observations()
        obs_dict.update(self._xy_observation_channels())
        return obs_dict

    # ------------------------------------------------------------------
    # rewards
    # ------------------------------------------------------------------

    def _get_rewards(self) -> torch.Tensor:
        total_reward = super()._get_rewards()
        total_reward = total_reward + self._compute_xy_stage_reward()
        if bool(self.cfg.high_speed_reward_enable):
            total_reward = total_reward + self._compute_high_speed_reward()
        self.extras["total_reward"] = total_reward.mean()
        return total_reward

    def _compute_xy_stage_reward(self) -> torch.Tensor:
        """Return the (non-positive) physical cost of using the stage.

        Every term is normalized so it is O(1) at its own limit; the configured
        scales therefore read directly as reward units at full stage usage.
        Finger torque/work penalties stay finger-only (they index
        ``actuated_dof_indices``), so the stage can never be a free energy
        source and finger effort is never double-counted.
        """
        dt = max(float(self.step_dt), 1.0e-6)
        position = self.hand_dof_pos[:, self.xy_dof_index_tensor]
        velocity = self.hand_dof_vel[:, self.xy_dof_index_tensor]
        acceleration = (velocity - self.xy_prev_velocity) / dt
        jerk = (acceleration - self.xy_prev_acceleration) / dt
        effort = self.xy_effort
        power = effort * velocity

        velocity_limit = max(float(self.cfg.xy_velocity_limit), 1.0e-6)
        acceleration_limit = max(float(self.cfg.xy_acceleration_limit), 1.0e-6)
        jerk_reference = max(float(self.cfg.xy_jerk_reference), 1.0e-6)
        effort_limit = max(float(self.cfg.xy_effort_limit), 1.0e-6)
        power_reference = max(effort_limit * velocity_limit, 1.0e-6)

        saturation = xy_boundary_saturation(
            position,
            self.xy_workspace_current,
            float(self.cfg.xy_boundary_margin),
        )
        costs = {
            "velocity": ((velocity / velocity_limit) ** 2).sum(dim=-1),
            "acceleration": ((acceleration / acceleration_limit) ** 2).sum(dim=-1),
            "jerk": ((jerk / jerk_reference) ** 2).sum(dim=-1),
            "effort": ((effort / effort_limit) ** 2).sum(dim=-1),
            "power": power.abs().sum(dim=-1) / power_reference,
            "boundary": saturation.mean(dim=-1),
        }
        scales = {
            "velocity": float(self.cfg.xy_velocity_penalty_scale),
            "acceleration": float(self.cfg.xy_acceleration_penalty_scale),
            "jerk": float(self.cfg.xy_jerk_penalty_scale),
            "effort": float(self.cfg.xy_effort_penalty_scale),
            "power": float(self.cfg.xy_power_penalty_scale),
            "boundary": float(self.cfg.xy_boundary_penalty_scale),
        }
        stage_reward = torch.zeros_like(costs["velocity"])
        for name, cost in costs.items():
            weighted = scales[name] * cost
            stage_reward = stage_reward + weighted
            # Unweighted magnitude and the weighted reward contribution.
            self.extras[f"xy_cost/{name}"] = cost.mean()
            self.extras[f"xy_penalty/{name}"] = weighted.mean()

        self.xy_prev_velocity = velocity.clone()
        self.xy_prev_acceleration = acceleration.clone()

        self.extras["xy/position_x"] = position[:, 0].mean()
        self.extras["xy/position_y"] = position[:, 1].mean()
        self.extras["xy/velocity_norm"] = velocity.norm(dim=-1).mean()
        self.extras["xy/acceleration_norm"] = acceleration.norm(dim=-1).mean()
        self.extras["xy/effort_norm"] = effort.norm(dim=-1).mean()
        self.extras["xy/power"] = power.abs().sum(dim=-1).mean()
        self.extras["xy/action_saturation_ratio"] = (
            self.xy_executed_action.abs() >= 1.0 - 1.0e-6
        ).float().mean()
        self.extras["xy/boundary_saturation_ratio"] = (saturation > 0.0).float().mean()
        self.extras["xy/workspace_utilization"] = (
            position.abs().amax(dim=-1) / max(self.xy_workspace_current, 1.0e-6)
        ).mean()
        self.extras["xy/stage_reward"] = stage_reward.mean()
        self.extras["curriculum/xy_workspace"] = torch.tensor(
            self.xy_workspace_current, device=self.device
        )
        self.extras["curriculum/xy_action_scale"] = torch.tensor(
            self.xy_action_scale_current, device=self.device
        )
        self.extras["curriculum/xy_progress"] = torch.tensor(
            self.xy_curriculum_progress, device=self.device
        )
        return stage_reward

    def _compute_high_speed_reward(self) -> torch.Tensor:
        """Return the optional continuous high-speed rotation shaping.

        The bonus is exactly zero up to ``angvel_clip_max`` where the base
        rotate reward saturates, then ramps continuously to
        ``high_speed_reward_scale`` at ``high_speed_target``. An over-speed
        penalty reuses ``rotate_penalty_scale`` above
        ``high_speed_penalty_threshold``. Nothing here switches at 0.8 rad/s:
        that value is only the global hierarchical activation threshold.
        """
        angular_velocity = self.nut_dof_vel_cf
        bonus = high_speed_reward(
            angular_velocity,
            clip_max=float(self.cfg.angvel_clip_max),
            target=float(self.cfg.high_speed_target),
            scale=float(self.cfg.high_speed_reward_scale),
        )
        overspeed = (
            angular_velocity - float(self.cfg.high_speed_penalty_threshold)
        ).clamp_min(0.0)
        penalty = float(self.cfg.rotate_penalty_scale) * overspeed
        self.extras["screw/high_speed_reward"] = bonus.mean()
        self.extras["screw/high_speed_penalty"] = penalty.mean()
        return bonus + penalty

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = self.hand._ALL_INDICES
        super()._reset_idx(env_ids)
        # The base reset already writes zero stage joint position/velocity
        # (init pose is zero and the reset noise is masked out). Clear every
        # derived controller buffer so no target, action or rate history leaks
        # across episodes.
        self.xy_target[env_ids] = 0.0
        self.xy_delayed_target[env_ids] = 0.0
        self.xy_prev_delta[env_ids] = 0.0
        self.xy_smoothed_action[env_ids] = 0.0
        self.xy_executed_action[env_ids] = 0.0
        self.xy_effort[env_ids] = 0.0
        self.xy_prev_velocity[env_ids] = 0.0
        self.xy_prev_acceleration[env_ids] = 0.0
