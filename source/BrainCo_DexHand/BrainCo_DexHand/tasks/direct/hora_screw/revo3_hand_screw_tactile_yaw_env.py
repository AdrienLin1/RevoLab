# Copyright (c) 2026, BrainCo.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""World-Z yaw stage for the hierarchical tactile valve tasks.

:class:`Revo3HandYawStageMixin` adds one **real, limited revolute joint** at the
hand mount to any tactile screw environment.  It is mixed into two tasks:

* :class:`Revo3HandScrewTactileYawEnv` - yaw only (21 + 1 = 22 actions)::

      world (fixed by the asset's global root joint)
        -> stage_yaw_joint (revolute, world Z, limits +/- yaw_joint_limit)
        -> right_hand_base_link (hand mount)
        -> Revo3 palm and its 21 finger joints

* ``Revo3HandScrewTactileXYYawEnv`` - XY + yaw (21 + 2 + 1 = 24 actions), see
  :mod:`revo3_hand_screw_tactile_xyyaw_env`.

There is no root teleport anywhere in the control loop: the hand root pose is
only written at reset (unchanged base behaviour), and the yaw rotation is
produced exclusively by an effort-limited PD torque on the revolute joint.  The
rotation axis passes through the **hand base mount**, not through the valve
centre, so nothing here can revolve the hand around the object.

Units: the yaw controller is radians / rad-s / N*m throughout and keeps its own
targets, smoothed action, rate history and effort buffer.  It never shares a
scale, limit or buffer with the metre-based XY controller; the two only share
the dimensionless curriculum progress and the per-env action-delay draw.
"""

from __future__ import annotations

import torch
from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

from .revo3_hand_screw_tactile_env import Revo3HandScrewTactileEnv
from .xy_stage import (
    NUM_XY_DOFS,
    XY_STAGE_CARRIAGE_BODY_NAME,
    XY_STAGE_JOINT_NAMES,
    XY_STAGE_WORLD_AXES,
    curriculum_value,
    split_hierarchical_action,
)
from .yaw_stage import (
    NUM_YAW_DOFS,
    YAW_STAGE_CARRIAGE_BODY_NAME,
    YAW_STAGE_JOINT_NAME,
    YAW_STAGE_WORLD_AXIS,
    assert_runtime_yaw_limits,
    radians_to_degrees,
    resolve_yaw_dof_index,
    stage_lock_limits,
    update_yaw_target,
    yaw_boundary_saturation,
    yaw_pd_effort,
    yaw_workspace_margin,
)

_BASE_LINK_NAME = "right_hand_base_link"
_WORLD_LINK_NAME = "world"
_BASE_FIXED_JOINT_NAME = "right_hand_base_joint"
_JOINTS_SCOPE_NAME = "joints"

# Yaw observation channels, in the same block order as the generic stage
# channels assembled for the follower.
YAW_OBS_BLOCKS: tuple[str, ...] = (
    "yaw_position",
    "yaw_velocity",
    "yaw_target",
    "previous_yaw_action",
    "yaw_workspace_margin",
)
# Generic stage blocks consumed by the hierarchical follower observation.
STAGE_OBS_BLOCKS: tuple[str, ...] = (
    "stage_position",
    "stage_velocity",
    "stage_target",
    "previous_stage_action",
    "stage_workspace_margin",
)
# Distance from the hard limit that still counts as "at the limit", radians.
_AT_LIMIT_TOLERANCE = 1.0e-3


def author_stage_carriage(stage, path: str, mass: float, inertia: float):
    """Author one massive, collision-free, gravity-free stage carriage body.

    Args:
        stage: USD stage owning the env_0 hand prototype.
        path: Absolute prim path of the carriage.
        mass: Carriage mass, kg.
        inertia: Diagonal inertia entry, kg*m^2.

    Returns:
        The created ``UsdGeom.Xform`` prim.
    """
    carriage = UsdGeom.Xform.Define(stage, path)
    carriage.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))
    carriage.AddOrientOp(UsdGeom.XformOp.PrecisionFloat).Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    carriage.AddScaleOp().Set(Gf.Vec3d(1.0, 1.0, 1.0))
    prim = carriage.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(prim).CreateRigidBodyEnabledAttr().Set(True)
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateMassAttr().Set(float(mass))
    mass_api.CreateDiagonalInertiaAttr().Set(
        Gf.Vec3f(float(inertia), float(inertia), float(inertia))
    )
    mass_api.CreatePrincipalAxesAttr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    physx_body = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    physx_body.CreateDisableGravityAttr().Set(True)
    physx_body.CreateRetainAccelerationsAttr().Set(False)
    return prim


def author_stage_prismatic_joint(
    stage,
    joint_path: str,
    *,
    axis: str,
    body0: str,
    body1: str,
    limit: float,
    effort_limit: float,
    max_velocity: float,
    local_rot,
):
    """Author one world-aligned prismatic stage joint with a force drive.

    Args:
        stage: USD stage owning the env_0 hand prototype.
        joint_path: Absolute prim path of the joint.
        axis: Joint axis in the (world-aligned) joint frame, ``"X"``/``"Y"``.
        body0: Parent body prim path.
        body1: Child body prim path.
        limit: Symmetric translation hard limit, metres.
        effort_limit: Drive force ceiling, newtons.
        max_velocity: PhysX joint velocity ceiling, m/s.
        local_rot: Joint-frame rotation cancelling the hand root orientation.

    Returns:
        The created ``UsdPhysics.PrismaticJoint``.
    """
    if stage.GetPrimAtPath(joint_path).IsValid():
        raise RuntimeError(f"Stage joint '{joint_path}' already exists.")
    joint = UsdPhysics.PrismaticJoint.Define(stage, joint_path)
    joint.CreateBody0Rel().SetTargets([body0])
    joint.CreateBody1Rel().SetTargets([body1])
    joint.CreateAxisAttr().Set(axis)
    # Prismatic limits are authored in the stage's linear unit (metres).
    joint.CreateLowerLimitAttr().Set(float(-limit))
    joint.CreateUpperLimitAttr().Set(float(limit))
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot0Attr().Set(local_rot)
    joint.CreateLocalRot1Attr().Set(local_rot)
    joint.CreateCollisionEnabledAttr().Set(False)
    joint.CreateExcludeFromArticulationAttr().Set(False)
    joint.CreateJointEnabledAttr().Set(True)
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
    drive.CreateTypeAttr().Set("force")
    drive.CreateStiffnessAttr().Set(0.0)
    drive.CreateDampingAttr().Set(0.0)
    drive.CreateMaxForceAttr().Set(float(effort_limit))
    drive.CreateTargetPositionAttr().Set(0.0)
    drive.CreateTargetVelocityAttr().Set(0.0)
    PhysxSchema.PhysxJointAPI.Apply(joint.GetPrim()).CreateMaxJointVelocityAttr().Set(
        float(max_velocity)
    )
    return joint


def author_stage_revolute_joint(
    stage,
    joint_path: str,
    *,
    axis: str,
    body0: str,
    body1: str,
    limit_rad: float,
    effort_limit: float,
    max_velocity_rad_s: float,
    local_rot,
):
    """Author the world-Z yaw revolute joint with an angular force drive.

    Unit contract: ``UsdPhysics.RevoluteJoint`` authors its ``lowerLimit`` /
    ``upperLimit`` in **degrees** and ``PhysxJointAPI.maxJointVelocity`` in
    **degrees per second**, while the Isaac Lab / PhysX articulation reports
    joint state in **radians**. Both are converted here explicitly; the runtime
    limits are re-checked in radians at environment startup.
    ``maxForce`` on an angular drive is a torque and needs no conversion.

    Args:
        stage: USD stage owning the env_0 hand prototype.
        joint_path: Absolute prim path of the joint.
        axis: Rotation axis in the (world-aligned) joint frame, ``"Z"``.
        body0: Parent body prim path.
        body1: Child body prim path.
        limit_rad: Symmetric rotation hard limit, radians.
        effort_limit: Drive torque ceiling, N*m.
        max_velocity_rad_s: PhysX joint velocity ceiling, rad/s.
        local_rot: Joint-frame rotation cancelling the hand root orientation.

    Returns:
        The created ``UsdPhysics.RevoluteJoint``.
    """
    if stage.GetPrimAtPath(joint_path).IsValid():
        raise RuntimeError(f"Stage joint '{joint_path}' already exists.")
    joint = UsdPhysics.RevoluteJoint.Define(stage, joint_path)
    joint.CreateBody0Rel().SetTargets([body0])
    joint.CreateBody1Rel().SetTargets([body1])
    joint.CreateAxisAttr().Set(axis)
    limit_deg = radians_to_degrees(limit_rad)
    joint.CreateLowerLimitAttr().Set(float(-limit_deg))
    joint.CreateUpperLimitAttr().Set(float(limit_deg))
    # The rotation anchor is the hand base mount (both local positions are the
    # frame origin), never the valve centre: yaw spins the wrist in place.
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot0Attr().Set(local_rot)
    joint.CreateLocalRot1Attr().Set(local_rot)
    joint.CreateCollisionEnabledAttr().Set(False)
    joint.CreateExcludeFromArticulationAttr().Set(False)
    joint.CreateJointEnabledAttr().Set(True)
    # Angular force drive with zero implicit PD: the environment applies an
    # explicit, effort-limited PD torque, exactly like the finger joints.
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
    drive.CreateTypeAttr().Set("force")
    drive.CreateStiffnessAttr().Set(0.0)
    drive.CreateDampingAttr().Set(0.0)
    drive.CreateMaxForceAttr().Set(float(effort_limit))
    drive.CreateTargetPositionAttr().Set(0.0)
    drive.CreateTargetVelocityAttr().Set(0.0)
    PhysxSchema.PhysxJointAPI.Apply(joint.GetPrim()).CreateMaxJointVelocityAttr().Set(
        float(radians_to_degrees(max_velocity_rad_s))
    )
    return joint


class Revo3HandYawStageMixin:
    """Add a real, limited world-Z yaw joint at the hand mount."""

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._setup_yaw_stage()

    def _num_extra_action_dofs(self) -> int:
        """Yaw plus whatever stage DOFs the wrapped environment already has."""
        return NUM_YAW_DOFS + super()._num_extra_action_dofs()

    def _num_preceding_stage_dofs(self) -> int:
        """Number of stage action channels that come before yaw."""
        return super()._num_extra_action_dofs()

    def _preceding_stage_obs_blocks(self) -> tuple[str, ...]:
        """Observation keys of the stage DOFs preceding yaw, in block order."""
        return ()

    def _setup_yaw_stage(self) -> None:
        """Resolve the yaw DOF by name and build its controller buffers."""
        self.yaw_dof_index = resolve_yaw_dof_index(self.hand.joint_names)
        self.yaw_dof_indices = [self.yaw_dof_index]
        self.yaw_dof_index_tensor = torch.tensor(
            self.yaw_dof_indices, dtype=torch.long, device=self.device
        )
        expected_robot_dofs = self.num_finger_dofs + self._num_extra_action_dofs()
        if self.num_robot_dofs != expected_robot_dofs:
            raise RuntimeError(
                f"Expected {self.num_finger_dofs} finger DOFs + "
                f"{self._num_extra_action_dofs()} stage DOFs = {expected_robot_dofs} robot "
                f"DOFs, got {self.num_robot_dofs}. Articulation joints: "
                f"{self.hand.joint_names}"
            )
        overlap = set(self.yaw_dof_indices) & set(self.finger_dof_indices)
        if overlap:
            raise RuntimeError(f"Yaw DOF overlaps finger DOFs at indices {sorted(overlap)}")
        preceding = getattr(self, "xy_dof_indices", ())
        if self.yaw_dof_index in set(preceding):
            raise RuntimeError(
                f"Yaw DOF index {self.yaw_dof_index} collides with the XY stage "
                f"indices {list(preceding)}"
            )

        # Keep yaw out of every finger-specific mechanism: the finger action
        # mask, the reset joint noise and the pose-diff reference.
        self.action_mask[self.yaw_dof_index_tensor] = 0.0
        self.pose_diff_mask[self.yaw_dof_index_tensor] = 0.0
        self.init_joint_pos[:, self.yaw_dof_index_tensor] = 0.0

        # ``dof_limits_scale`` is a finger-safety margin. Yaw must keep its
        # authored hard limit so the final curriculum workspace is reachable.
        raw_limits = self.hand.root_physx_view.get_dof_limits().to(self.device)
        self.hand_dof_lower_limits[:, self.yaw_dof_index_tensor] = raw_limits[..., 0][
            :, self.yaw_dof_index_tensor
        ]
        self.hand_dof_upper_limits[:, self.yaw_dof_index_tensor] = raw_limits[..., 1][
            :, self.yaw_dof_index_tensor
        ]
        # Radians, always: this is the unit check for the degree-authored USD.
        assert_runtime_yaw_limits(
            self.hand_dof_lower_limits[:, self.yaw_dof_index_tensor],
            self.hand_dof_upper_limits[:, self.yaw_dof_index_tensor],
            float(self.cfg.yaw_joint_limit),
        )

        # ---- yaw controller buffers (all at control-step rate, radians) ----
        zeros = torch.zeros(
            (self.num_envs, NUM_YAW_DOFS), device=self.device, dtype=torch.float
        )
        self.yaw_target = zeros.clone()
        self.yaw_delayed_target = zeros.clone()
        self.yaw_prev_delta = zeros.clone()
        self.yaw_smoothed_action = zeros.clone()
        self.yaw_executed_action = zeros.clone()
        self.yaw_effort = zeros.clone()
        self.yaw_prev_velocity = zeros.clone()
        self.yaw_prev_acceleration = zeros.clone()

        # ---- curriculum state (owned by the trainer, mirrored here) ----
        self.yaw_curriculum_progress = 0.0
        self.yaw_workspace_current = float(self.cfg.yaw_workspace_initial)
        self.yaw_action_scale_current = float(self.cfg.yaw_action_scale_initial)

        # ---- Stage-0 mechanical lock ----
        # Every stage DOF of this task, in action order: the preceding prismatic
        # joints (if any) followed by yaw.
        self.stage_lock_dof_indices = list(getattr(self, "xy_dof_indices", [])) + list(
            self.yaw_dof_indices
        )
        stage_index_tensor = torch.tensor(
            self.stage_lock_dof_indices, dtype=torch.long, device=self.device
        )
        # Snapshot the authored hard limits (validated just above) so the lock can
        # be released back to exactly them.
        self._stage_unlocked_limits = torch.stack(
            [
                self.hand_dof_lower_limits[:, stage_index_tensor],
                self.hand_dof_upper_limits[:, stage_index_tensor],
            ],
            dim=-1,
        ).clone()
        self.stage_follower_active: bool | None = None
        self.set_stage_follower_active(False)

        print(
            "[INFO] Yaw stage ready: joint "
            f"'{YAW_STAGE_JOINT_NAME}' -> DOF index {self.yaw_dof_index} "
            f"(world axis {YAW_STAGE_WORLD_AXIS}), hard limit "
            f"+/-{float(self.cfg.yaw_joint_limit):.4f} rad "
            f"(+/-{radians_to_degrees(self.cfg.yaw_joint_limit):.2f} deg), torque limit "
            f"{float(self.cfg.yaw_effort_limit):.3f} N*m.",
            flush=True,
        )

    # ------------------------------------------------------------------
    # scene authoring
    # ------------------------------------------------------------------

    def _author_robot_stage_overrides(self) -> None:
        """Author the complete stage chain on the env_0 hand prototype.

        Runs after the hand is spawned and before ``clone_environments`` so all
        cloned environments inherit the same articulation topology.  Every
        joint frame is rotated by the inverse of the hand root orientation,
        which makes the prismatic joints translate along **world X / Y** and
        the yaw joint rotate about **world Z**, regardless of the palm-down
        grasp orientation configured in ``assets.py``.

        The chain depends on how many translation DOFs precede yaw::

            0 -> world -[stage_yaw_joint/Z]-> right_hand_base_link
            2 -> world -[stage_x_joint/X]-> stage_x_carriage
                       -[stage_y_joint/Y]-> stage_y_carriage
                       -[stage_yaw_joint/Z]-> right_hand_base_link
        """
        stage = self.scene.stage
        hand_path = f"{self.scene.env_prim_paths[0]}/hand"
        if not stage.GetPrimAtPath(hand_path).IsValid():
            raise RuntimeError(f"Hand prim not found at '{hand_path}'.")

        world_path = f"{hand_path}/{_WORLD_LINK_NAME}"
        base_link_path = f"{hand_path}/{_BASE_LINK_NAME}"
        joints_scope = f"{hand_path}/{_JOINTS_SCOPE_NAME}"
        for path in (world_path, base_link_path, joints_scope):
            if not stage.GetPrimAtPath(path).IsValid():
                raise RuntimeError(
                    f"Stage authoring requires '{path}' in the Revo3 hand USD."
                )

        # 1) Remove the rigid world -> hand base weld: the stage replaces it.
        base_joint_path = f"{joints_scope}/{_BASE_FIXED_JOINT_NAME}"
        base_joint_prim = stage.GetPrimAtPath(base_joint_path)
        if not base_joint_prim.IsValid():
            raise RuntimeError(
                f"Expected the fixed hand base joint at '{base_joint_path}'."
            )
        base_joint_prim.SetActive(False)

        # 2) Joint frames aligned with the world axes.
        root_quat = tuple(float(value) for value in self.cfg.robot_cfg.init_state.rot)
        if len(root_quat) != 4:
            raise RuntimeError(
                f"Hand init_state.rot must be a (w, x, y, z) quaternion, got {root_quat}"
            )
        inverse_root = Gf.Quatf(root_quat[0], -root_quat[1], -root_quat[2], -root_quat[3])

        num_preceding = self._num_preceding_stage_dofs()
        if num_preceding not in (0, NUM_XY_DOFS):
            raise RuntimeError(
                f"The yaw stage supports 0 or {NUM_XY_DOFS} preceding translation DOFs, "
                f"got {num_preceding}"
            )

        # 3) Translation carriages and prismatic joints, when present.
        parent_path = world_path
        described = [f"{world_path}"]
        if num_preceding == NUM_XY_DOFS:
            carriage_mass = float(self.cfg.xy_carriage_mass)
            carriage_inertia = float(self.cfg.xy_carriage_inertia)
            xy_limit = float(self.cfg.xy_joint_limit)
            xy_effort = float(self.cfg.xy_effort_limit)
            xy_max_velocity = float(self.cfg.xy_joint_velocity_limit_sim)
            for joint_name, axis, carriage_name in (
                (XY_STAGE_JOINT_NAMES[0], XY_STAGE_WORLD_AXES[0], XY_STAGE_CARRIAGE_BODY_NAME),
                (XY_STAGE_JOINT_NAMES[1], XY_STAGE_WORLD_AXES[1], YAW_STAGE_CARRIAGE_BODY_NAME),
            ):
                carriage_path = f"{hand_path}/{carriage_name}"
                author_stage_carriage(stage, carriage_path, carriage_mass, carriage_inertia)
                author_stage_prismatic_joint(
                    stage,
                    f"{joints_scope}/{joint_name}",
                    axis=axis,
                    body0=parent_path,
                    body1=carriage_path,
                    limit=xy_limit,
                    effort_limit=xy_effort,
                    max_velocity=xy_max_velocity,
                    local_rot=inverse_root,
                )
                described.append(f"-[{joint_name}/{axis}]-> {carriage_path}")
                parent_path = carriage_path

        # 4) The yaw joint always closes the chain onto the hand base mount.
        author_stage_revolute_joint(
            stage,
            f"{joints_scope}/{YAW_STAGE_JOINT_NAME}",
            axis=YAW_STAGE_WORLD_AXIS,
            body0=parent_path,
            body1=base_link_path,
            limit_rad=float(self.cfg.yaw_joint_limit),
            effort_limit=float(self.cfg.yaw_effort_limit),
            max_velocity_rad_s=float(self.cfg.yaw_joint_velocity_limit_sim),
            local_rot=inverse_root,
        )
        described.append(
            f"-[{YAW_STAGE_JOINT_NAME}/{YAW_STAGE_WORLD_AXIS}]-> {base_link_path}"
        )
        print(
            "[INFO] Stage authored: " + " ".join(described) + f"; disabled '{base_joint_path}'.",
            flush=True,
        )

    # ------------------------------------------------------------------
    # curriculum
    # ------------------------------------------------------------------

    def set_yaw_curriculum_progress(self, progress: float) -> tuple[float, float]:
        """Set the yaw workspace / action-scale ramp progress.

        Args:
            progress: The SHARED stage ramp progress in ``[0, 1]``; values
                outside are clamped. It is dimensionless, so translation and
                yaw are always at the same point of their own curricula.

        Returns:
            The resolved ``(workspace, action_scale)`` in radians.
        """
        self.yaw_curriculum_progress = min(max(float(progress), 0.0), 1.0)
        self.yaw_workspace_current = curriculum_value(
            self.cfg.yaw_workspace_initial,
            self.cfg.yaw_workspace_final,
            self.yaw_curriculum_progress,
        )
        self.yaw_action_scale_current = curriculum_value(
            self.cfg.yaw_action_scale_initial,
            self.cfg.yaw_action_scale_final,
            self.yaw_curriculum_progress,
        )
        return self.yaw_workspace_current, self.yaw_action_scale_current

    def set_stage_follower_active(self, active: bool) -> bool:
        """Mechanically lock or release every stage joint of this task.

        Stage 0 commands a strictly zero stage action, but a zero *target* is
        not a lock.  The yaw PD saturates at ``yaw_effort_limit / yaw_pgain``
        (37.5 mrad at the shipped gains), so the grasp reaction torque simply
        overpowers the holding torque and walks the wrist to its hard stop while
        the follower is still inactive.  Locking the joint **position limits**
        to a residual play instead holds every stage DOF with a PhysX limit
        constraint, which is a hard constraint and is not bounded by the drive's
        ``maxForce``: Stage 0 then reproduces the rigidly-mounted baseline
        exactly.

        The effort ceiling is never raised and nothing is teleported; only the
        joint's own limits move, and they are restored to the authored hard
        limits the moment the hierarchical curriculum activates the follower.

        Once locked, the PD error is ~0, so the controller naturally applies ~0
        torque and every yaw/XY physical cost term reads zero in Stage 0.

        Args:
            active: Whether the hierarchical follower samples this rollout.

        Returns:
            The resolved active flag.
        """
        active = bool(active)
        if self.stage_follower_active == active:
            return active
        self.stage_follower_active = active
        if not bool(getattr(self.cfg, "stage_lock_when_follower_inactive", True)):
            return active

        if active:
            limits = self._stage_unlocked_limits
        else:
            limits = stage_lock_limits(
                self._stage_unlocked_limits,
                num_linear_dofs=self._num_preceding_stage_dofs(),
                linear_tolerance=float(self.cfg.stage_lock_tolerance_m),
                angular_tolerance=float(self.cfg.stage_lock_tolerance_rad),
            )
            # No target, action or rate history may survive the locked phase.
            self._reset_stage_controller_buffers(slice(None))
        self.hand.write_joint_position_limit_to_sim(
            limits, joint_ids=self.stage_lock_dof_indices, warn_limit_violation=False
        )
        print(
            f"[INFO] Stage joints {tuple(self.cfg.stage_joint_names)} "
            f"{'RELEASED to their authored hard limits' if active else 'LOCKED at zero'} "
            f"(follower_active={active}).",
            flush=True,
        )
        return active

    def _reset_stage_controller_buffers(self, env_ids) -> None:
        """Clear the yaw (and, when present, XY) controller history."""
        for name in (
            "yaw_target",
            "yaw_delayed_target",
            "yaw_prev_delta",
            "yaw_smoothed_action",
            "yaw_executed_action",
            "yaw_effort",
            "yaw_prev_velocity",
            "yaw_prev_acceleration",
            "xy_target",
            "xy_delayed_target",
            "xy_prev_delta",
            "xy_smoothed_action",
            "xy_executed_action",
            "xy_effort",
            "xy_prev_velocity",
            "xy_prev_acceleration",
        ):
            buffer = getattr(self, name, None)
            if buffer is not None:
                buffer[env_ids] = 0.0

    def set_stage_curriculum_progress(self, progress: float) -> dict[str, float]:
        """Publish one shared progress to every stage DOF of this task.

        Args:
            progress: Ramp progress in ``[0, 1]``.

        Returns:
            Resolved per-DOF curriculum values: ``xy_workspace`` /
            ``xy_action_scale`` in metres when the task translates, plus
            ``yaw_workspace`` / ``yaw_action_scale`` in radians.
        """
        values: dict[str, float] = {}
        # Bypass this class's own XY override (if any) so there is exactly one
        # push per DOF and no recursion.
        push_xy = getattr(super(), "set_xy_curriculum_progress", None)
        if push_xy is not None:
            workspace, action_scale = push_xy(progress)
            values["xy_workspace"] = float(workspace)
            values["xy_action_scale"] = float(action_scale)
        yaw_workspace, yaw_action_scale = self.set_yaw_curriculum_progress(progress)
        values["yaw_workspace"] = float(yaw_workspace)
        values["yaw_action_scale"] = float(yaw_action_scale)
        xy_progress = getattr(self, "xy_curriculum_progress", None)
        if xy_progress is not None and abs(
            float(xy_progress) - self.yaw_curriculum_progress
        ) > 1.0e-12:
            raise RuntimeError(
                f"Stage curricula desynchronized: xy progress {xy_progress} != yaw "
                f"progress {self.yaw_curriculum_progress}"
            )
        return values

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # Strip only the trailing yaw channel; everything before it takes the
        # untouched path of the wrapped environment (fingers, and XY when it
        # is present), so the XY control law is literally the same code.
        leading, yaw_actions = split_hierarchical_action(
            actions,
            self.num_finger_dofs + self._num_preceding_stage_dofs(),
            NUM_YAW_DOFS,
        )
        super()._pre_physics_step(leading)
        self._update_yaw_stage_targets(yaw_actions)

    def _update_yaw_stage_targets(self, yaw_actions: torch.Tensor) -> None:
        """Advance the yaw position target for one control step."""
        executed = torch.clamp(yaw_actions, -1.0, 1.0)
        # The previous control-step target is what the delayed physics substeps
        # keep applying, mirroring the finger action-delay model.
        self.yaw_delayed_target = self.yaw_target.clone()
        target, delta, smoothed = update_yaw_target(
            executed,
            self.yaw_target,
            self.yaw_prev_delta,
            self.yaw_smoothed_action,
            action_scale=self.yaw_action_scale_current,
            workspace=self.yaw_workspace_current,
            velocity_limit=float(self.cfg.yaw_velocity_limit),
            acceleration_limit=float(self.cfg.yaw_acceleration_limit),
            dt=float(self.step_dt),
            smoothing=float(self.cfg.yaw_action_smoothing),
        )
        self.yaw_target = target
        self.yaw_prev_delta = delta
        self.yaw_smoothed_action = smoothed
        self.yaw_executed_action = executed

    def _apply_action(self) -> None:
        substep_index = self._physics_substep_idx
        # Fingers (and XY when present) first: this refreshes the articulation
        # state, applies their efforts and advances the substep bookkeeping.
        super()._apply_action()

        if bool(self.cfg.yaw_use_action_delay):
            switch_at = self.action_delay * float(self.cfg.decimation)
            use_delayed = (substep_index < switch_at).unsqueeze(-1)
            applied_target = torch.where(
                use_delayed, self.yaw_delayed_target, self.yaw_target
            )
        else:
            applied_target = self.yaw_target

        position = self.hand_dof_pos[:, self.yaw_dof_index_tensor]
        velocity = self.hand_dof_vel[:, self.yaw_dof_index_tensor]
        effort = yaw_pd_effort(
            applied_target,
            position,
            velocity,
            pgain=float(self.cfg.yaw_pgain),
            dgain=float(self.cfg.yaw_dgain),
            effort_limit=float(self.cfg.yaw_effort_limit),
        )
        self.hand.set_joint_effort_target(effort, joint_ids=self.yaw_dof_indices)
        self.torques[:, self.yaw_dof_index_tensor] = effort
        # Torque of the most recent physics substep; used for the effort/power
        # cost terms and the yaw/* diagnostics of this control step.
        self.yaw_effort = effort

    # ------------------------------------------------------------------
    # observations
    # ------------------------------------------------------------------

    def _yaw_observation_channels(self) -> dict[str, torch.Tensor]:
        """Return the yaw self-state channels for the follower policy.

        Position and target are normalized by the FIXED asset joint limit so
        the follower observation is stationary while the curriculum widens the
        software workspace; only ``yaw_workspace_margin`` uses the current
        curriculum workspace. Velocity is normalized by the commanded angular
        velocity limit and clipped to +/-2 to stay bounded during contact
        transients.
        """
        position = self.hand_dof_pos[:, self.yaw_dof_index_tensor]
        velocity = self.hand_dof_vel[:, self.yaw_dof_index_tensor]
        position_scale = max(float(self.cfg.yaw_position_obs_scale), 1.0e-6)
        velocity_scale = max(float(self.cfg.yaw_velocity_obs_scale), 1.0e-6)
        return {
            "yaw_position": (position / position_scale).clamp(-1.0, 1.0),
            "yaw_velocity": (velocity / velocity_scale).clamp(-2.0, 2.0),
            "yaw_target": (self.yaw_target / position_scale).clamp(-1.0, 1.0),
            "previous_yaw_action": self.yaw_executed_action.clone(),
            "yaw_workspace_margin": yaw_workspace_margin(
                position, self.yaw_workspace_current
            ),
        }

    def _stage_observation_channels(self, obs_dict) -> dict[str, torch.Tensor]:
        """Assemble the generic stage channels the follower actually reads.

        Each block concatenates the preceding translation DOFs (if any) with
        yaw, in exactly the task's stage joint order, so
        ``stage_position[:, -1]`` is always the yaw channel.
        """
        yaw_channels = self._yaw_observation_channels()
        preceding = self._preceding_stage_obs_blocks()
        if preceding and len(preceding) != len(STAGE_OBS_BLOCKS):
            raise RuntimeError(
                f"Preceding stage blocks {preceding} must have one entry per generic "
                f"block {STAGE_OBS_BLOCKS}"
            )
        channels: dict[str, torch.Tensor] = {}
        for index, name in enumerate(STAGE_OBS_BLOCKS):
            parts = []
            if preceding:
                parts.append(obs_dict[preceding[index]])
            parts.append(yaw_channels[YAW_OBS_BLOCKS[index]])
            channels[name] = torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]
        channels.update(yaw_channels)
        return channels

    def _get_observations(self) -> dict:
        obs_dict = super()._get_observations()
        obs_dict.update(self._stage_observation_channels(obs_dict))
        return obs_dict

    # ------------------------------------------------------------------
    # rewards
    # ------------------------------------------------------------------

    def _get_rewards(self) -> torch.Tensor:
        total_reward = super()._get_rewards()
        total_reward = total_reward + self._compute_yaw_stage_reward()
        self.extras["total_reward"] = total_reward.mean()
        return total_reward

    def _compute_yaw_stage_reward(self) -> torch.Tensor:
        """Return the (non-positive) physical cost of using the yaw joint.

        Every term is normalized so it is O(1) at its own limit; the configured
        scales therefore read directly as reward units at full yaw usage. The
        finger torque/work penalties stay finger-only (they index
        ``actuated_dof_indices``) and the XY costs stay XY-only, so yaw can
        never be a free energy source and no effort is double-counted.
        """
        dt = max(float(self.step_dt), 1.0e-6)
        position = self.hand_dof_pos[:, self.yaw_dof_index_tensor]
        velocity = self.hand_dof_vel[:, self.yaw_dof_index_tensor]
        acceleration = (velocity - self.yaw_prev_velocity) / dt
        jerk = (acceleration - self.yaw_prev_acceleration) / dt
        effort = self.yaw_effort
        power = effort * velocity

        velocity_limit = max(float(self.cfg.yaw_velocity_limit), 1.0e-6)
        acceleration_limit = max(float(self.cfg.yaw_acceleration_limit), 1.0e-6)
        jerk_reference = max(float(self.cfg.yaw_jerk_reference), 1.0e-6)
        effort_limit = max(float(self.cfg.yaw_effort_limit), 1.0e-6)
        power_reference = max(effort_limit * velocity_limit, 1.0e-6)

        saturation = yaw_boundary_saturation(
            position,
            self.yaw_workspace_current,
            float(self.cfg.yaw_boundary_margin),
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
            "velocity": float(self.cfg.yaw_velocity_penalty_scale),
            "acceleration": float(self.cfg.yaw_acceleration_penalty_scale),
            "jerk": float(self.cfg.yaw_jerk_penalty_scale),
            "effort": float(self.cfg.yaw_effort_penalty_scale),
            "power": float(self.cfg.yaw_power_penalty_scale),
            "boundary": float(self.cfg.yaw_boundary_penalty_scale),
        }
        stage_reward = torch.zeros_like(costs["velocity"])
        for name, cost in costs.items():
            weighted = scales[name] * cost
            stage_reward = stage_reward + weighted
            # Unweighted magnitude and the weighted reward contribution.
            self.extras[f"yaw_cost/{name}"] = cost.mean()
            self.extras[f"yaw_penalty/{name}"] = weighted.mean()

        self.yaw_prev_velocity = velocity.clone()
        self.yaw_prev_acceleration = acceleration.clone()

        joint_limit = float(self.cfg.yaw_joint_limit)
        self.extras["yaw/position"] = position.mean()
        self.extras["yaw/velocity"] = velocity.mean()
        self.extras["yaw/target"] = self.yaw_target.mean()
        self.extras["yaw/tracking_error"] = (self.yaw_target - position).abs().mean()
        self.extras["yaw/effort"] = effort.abs().mean()
        self.extras["yaw/power"] = power.abs().mean()
        self.extras["yaw/action_abs"] = self.yaw_executed_action.abs().mean()
        self.extras["yaw/action_saturation_ratio"] = (
            self.yaw_executed_action.abs() >= 1.0 - 1.0e-6
        ).float().mean()
        self.extras["yaw/boundary_saturation_ratio"] = (saturation > 0.0).float().mean()
        self.extras["yaw/workspace_utilization"] = (
            position.abs() / max(self.yaw_workspace_current, 1.0e-6)
        ).mean()
        self.extras["yaw/at_positive_limit_ratio"] = (
            position >= joint_limit - _AT_LIMIT_TOLERANCE
        ).float().mean()
        self.extras["yaw/at_negative_limit_ratio"] = (
            position <= -joint_limit + _AT_LIMIT_TOLERANCE
        ).float().mean()
        self.extras["yaw/stage_reward"] = stage_reward.mean()
        # 1 while the stage joints are mechanically held at zero (Stage 0).
        self.extras["stage/locked"] = torch.tensor(
            float(
                bool(self.cfg.stage_lock_when_follower_inactive)
                and not bool(self.stage_follower_active)
            ),
            device=self.device,
        )
        self.extras["curriculum/yaw_workspace"] = torch.tensor(
            self.yaw_workspace_current, device=self.device
        )
        self.extras["curriculum/yaw_action_scale"] = torch.tensor(
            self.yaw_action_scale_current, device=self.device
        )
        self.extras["curriculum/stage_progress"] = torch.tensor(
            self.yaw_curriculum_progress, device=self.device
        )
        return stage_reward

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = self.hand._ALL_INDICES
        super()._reset_idx(env_ids)
        # The base reset already writes zero yaw joint position/velocity (init
        # pose is zero and the reset noise is masked out). Clear every derived
        # controller buffer so no target, action or rate history leaks across
        # episodes.
        self.yaw_target[env_ids] = 0.0
        self.yaw_delayed_target[env_ids] = 0.0
        self.yaw_prev_delta[env_ids] = 0.0
        self.yaw_smoothed_action[env_ids] = 0.0
        self.yaw_executed_action[env_ids] = 0.0
        self.yaw_effort[env_ids] = 0.0
        self.yaw_prev_velocity[env_ids] = 0.0
        self.yaw_prev_acceleration[env_ids] = 0.0


class Revo3HandScrewTactileYawEnv(Revo3HandYawStageMixin, Revo3HandScrewTactileEnv):
    """Tactile valve environment whose hand rides on a single yaw joint.

    Action layout (fixed, resolved by joint NAME, never by articulation
    ordering)::

        action[:, :21]   -> finger joints
        action[:, 21:22] -> yaw about world Z

    This is the yaw-only ablation of the XY+yaw task: identical yaw physics,
    controller, costs and curriculum, without any translation authority.
    """

    def _preceding_stage_obs_blocks(self) -> tuple[str, ...]:
        """Yaw is the only stage DOF, so no channels precede it."""
        return ()
