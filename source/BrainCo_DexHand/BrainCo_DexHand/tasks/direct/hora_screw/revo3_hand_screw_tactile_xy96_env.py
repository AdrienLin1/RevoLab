# Copyright (c) 2026, BrainCo.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""``valvedriver_tactile_xy96``: fine-compensation XY stage on a compliant mount.

Kinematic chain (compliance axes are passive; no policy action reaches them)::

    world
      -> stage_x_joint      prismatic, world X, ACTUATED by the follower
      -> stage_x_carriage
      -> stage_y_joint      prismatic, world Y, ACTUATED by the follower
      -> stage_y_carriage
      -> stage_cz_joint     prismatic, world Z, PASSIVE spring-damper
      -> stage_cz_carriage
      -> stage_croll_joint  revolute,  world X, PASSIVE spring-damper
      -> stage_croll_carriage
      -> stage_cpitch_joint revolute,  world Y, PASSIVE spring-damper
      -> stage_cpitch_carriage
      -> stage_cyaw_joint   revolute,  world Z, PASSIVE spring-damper
      -> right_hand_base_link
      -> Revo3 palm and its 21 finger joints

The action layout is unchanged from ``valvedriver_tactile_xy``::

    action[:, :21]   -> finger joints
    action[:, 21:23] -> XY stage

and so is the 159-dim follower observation: the compliance deflections are
deliberately NOT observable.  They act as unmodelled mount dynamics, which is
what a real arm's finite stiffness is from the policy's point of view.  A
variant that feeds them to the follower would answer a different question and
would break checkpoint compatibility with every existing follower.
"""

from __future__ import annotations

import torch
from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

from .revo3_hand_screw_tactile_xy96_env_cfg import Revo3HandScrewTactileXY96MixinCfg
from .revo3_hand_screw_tactile_xy_env import (
    _BASE_FIXED_JOINT_NAME,
    _BASE_LINK_NAME,
    _JOINTS_SCOPE_NAME,
    _WORLD_LINK_NAME,
    Revo3HandScrewTactileXYEnv,
)
from .xy_compliance import (
    COMPLIANCE_Y_CARRIAGE_BODY_NAME,
    compliance_body_chain,
    resolve_compliance_axes,
)
from .xy_stage import (
    XY_STAGE_CARRIAGE_BODY_NAME,
    XY_STAGE_JOINT_NAMES,
    XY_STAGE_WORLD_AXES,
)
from .yaw_stage import radians_to_degrees


class Revo3HandScrewTactileXY96Env(Revo3HandScrewTactileXYEnv):
    """Fine-compensation XY stage whose mount is a spring, not a weld."""

    cfg: Revo3HandScrewTactileXY96MixinCfg

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        # Resolved before ``super().__init__`` so the parent's DOF-count check
        # can account for the passive joints through ``_num_passive_stage_dofs``.
        self._compliance_axes = resolve_compliance_axes(cfg.xy_compliance_axes)
        super().__init__(cfg, render_mode, **kwargs)

        self.compliance_dof_indices: list[int] = []
        if self._compliance_axes:
            joint_names = list(self.hand.joint_names)
            for axis in self._compliance_axes:
                if axis.joint_name not in joint_names:
                    raise RuntimeError(
                        f"Compliance joint '{axis.joint_name}' is missing from the "
                        f"articulation. Joints: {joint_names}"
                    )
                self.compliance_dof_indices.append(joint_names.index(axis.joint_name))
            overlap = set(self.compliance_dof_indices) & (
                set(self.finger_dof_indices) | set(self.xy_dof_indices)
            )
            if overlap:
                raise RuntimeError(
                    f"Compliance DOFs overlap actuated DOFs at indices {sorted(overlap)}"
                )

            self.compliance_dof_index_tensor = torch.tensor(
                self.compliance_dof_indices, dtype=torch.long, device=self.device
            )
            # Passive joints stay out of every finger mechanism: no action, no
            # reset noise (``joint_noise`` is multiplied by ``action_mask``), no
            # pose-diff penalty and a zero neutral pose.
            self.action_mask[self.compliance_dof_index_tensor] = 0.0
            self.pose_diff_mask[self.compliance_dof_index_tensor] = 0.0
            self.init_joint_pos[:, self.compliance_dof_index_tensor] = 0.0
            # Keep the authored mechanical end stops instead of the finger
            # safety margin ``dof_limits_scale``.
            raw_limits = self.hand.root_physx_view.get_dof_limits().to(self.device)
            self.hand_dof_lower_limits[:, self.compliance_dof_index_tensor] = raw_limits[
                ..., 0
            ][:, self.compliance_dof_index_tensor]
            self.hand_dof_upper_limits[:, self.compliance_dof_index_tensor] = raw_limits[
                ..., 1
            ][:, self.compliance_dof_index_tensor]

            self._compliance_nominal_stiffness = torch.tensor(
                [axis.stiffness for axis in self._compliance_axes],
                dtype=torch.float,
                device=self.device,
            )
            self._compliance_nominal_damping = torch.tensor(
                [axis.damping for axis in self._compliance_axes],
                dtype=torch.float,
                device=self.device,
            )
            # Per-axis nominal gains override the group-average authored by the
            # actuator config, for every environment.
            self._write_compliance_gains(
                self.hand._ALL_INDICES,
                torch.ones(
                    (self.num_envs, len(self.compliance_dof_indices)),
                    dtype=torch.float,
                    device=self.device,
                ),
            )
            print(
                "[INFO] XY96 mount compliance: "
                + ", ".join(
                    f"{axis.joint_name}(K={axis.stiffness:g}, D={axis.damping:g}, "
                    f"lim=+/-{axis.limit:g})"
                    for axis in self._compliance_axes
                )
                + f" -> DOF indices {self.compliance_dof_indices}; randomize="
                + f"{bool(self.cfg.xy_compliance_randomize)} "
                + f"scale={tuple(self.cfg.xy_compliance_stiffness_scale_range)}",
                flush=True,
            )
        else:
            self.compliance_dof_index_tensor = torch.zeros(
                0, dtype=torch.long, device=self.device
            )
            print("[INFO] XY96 mount compliance disabled (rigid weld).", flush=True)

        # Chatter/drift cost history.
        self.xy_prev_executed_action = torch.zeros(
            (self.num_envs, self.cfg.action_space - self.cfg.finger_action_space),
            dtype=torch.float,
            device=self.device,
        )

    def _num_passive_stage_dofs(self) -> int:
        return len(self._compliance_axes)

    # ------------------------------------------------------------------
    # scene authoring
    # ------------------------------------------------------------------

    def _define_stage_body(self, stage, path: str, mass: float, inertia: float) -> None:
        """Author one gravity-free, collider-free rigid body of the stage chain."""
        body = UsdGeom.Xform.Define(stage, path)
        body.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))
        body.AddOrientOp(UsdGeom.XformOp.PrecisionFloat).Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        body.AddScaleOp().Set(Gf.Vec3d(1.0, 1.0, 1.0))
        prim = body.GetPrim()
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

    def _author_robot_stage_overrides(self) -> None:
        """Author the actuated XY joints plus the passive compliance chain.

        Overrides the parent wholesale because the chain itself changes: the Y
        joint no longer attaches to the hand base link but to the head of the
        compliance chain.  With ``xy_compliance_axes = ()`` the authored topology
        is identical to the parent's.
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
                    f"XY96 stage authoring requires '{path}' in the Revo3 hand USD."
                )

        # 1) The world -> hand weld is replaced by the whole stage chain.
        base_joint_path = f"{joints_scope}/{_BASE_FIXED_JOINT_NAME}"
        base_joint_prim = stage.GetPrimAtPath(base_joint_path)
        if not base_joint_prim.IsValid():
            raise RuntimeError(f"Expected the fixed hand base joint at '{base_joint_path}'.")
        base_joint_prim.SetActive(False)

        # 2) Bodies. The X carriage keeps the reflected-inertia mass; every
        #    compliance body is light so the springs stay well conditioned.
        carriage_path = f"{hand_path}/{XY_STAGE_CARRIAGE_BODY_NAME}"
        self._define_stage_body(
            stage,
            carriage_path,
            float(self.cfg.xy_carriage_mass),
            float(self.cfg.xy_carriage_inertia),
        )
        nodes = compliance_body_chain(self._compliance_axes, base_link_path)
        node_paths: list[str] = []
        for node in nodes:
            if node == base_link_path:
                node_paths.append(base_link_path)
                continue
            path = f"{hand_path}/{node}"
            self._define_stage_body(
                stage,
                path,
                float(self.cfg.xy_compliance_body_mass),
                float(self.cfg.xy_compliance_body_inertia),
            )
            node_paths.append(path)

        # 3) Joint frames are rotated by the inverse hand root quaternion so the
        #    authored axes coincide with world X/Y/Z despite the palm-down grasp.
        root_quat = tuple(float(value) for value in self.cfg.robot_cfg.init_state.rot)
        if len(root_quat) != 4:
            raise RuntimeError(
                f"Hand init_state.rot must be a (w, x, y, z) quaternion, got {root_quat}"
            )
        inverse_root = Gf.Quatf(root_quat[0], -root_quat[1], -root_quat[2], -root_quat[3])

        # 4) The two ACTUATED prismatic joints. Zero implicit PD: the
        #    environment applies explicit, effort-limited PD forces.
        limit = float(self.cfg.xy_joint_limit)
        effort_limit = float(self.cfg.xy_effort_limit)
        max_velocity = float(self.cfg.xy_joint_velocity_limit_sim)
        y_child_path = node_paths[0]
        actuated_chain = (
            (XY_STAGE_JOINT_NAMES[0], XY_STAGE_WORLD_AXES[0], world_path, carriage_path),
            (XY_STAGE_JOINT_NAMES[1], XY_STAGE_WORLD_AXES[1], carriage_path, y_child_path),
        )
        for joint_name, axis, body0, body1 in actuated_chain:
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

        # 5) The PASSIVE compliance joints. USD drive gains stay at zero: the
        #    real spring constants are written through the Isaac Lab implicit
        #    actuator and the runtime randomization, both of which use SI
        #    radian units, whereas USD angular drive gains are per-degree.
        #    Authoring zero here keeps a single source of truth for the units.
        for index, axis in enumerate(self._compliance_axes):
            joint_path = f"{joints_scope}/{axis.joint_name}"
            if stage.GetPrimAtPath(joint_path).IsValid():
                raise RuntimeError(f"Compliance joint '{joint_path}' already exists.")
            body0 = node_paths[index]
            body1 = node_paths[index + 1]
            if axis.is_prismatic:
                joint = UsdPhysics.PrismaticJoint.Define(stage, joint_path)
                joint.CreateLowerLimitAttr().Set(float(-axis.limit))
                joint.CreateUpperLimitAttr().Set(float(axis.limit))
                drive_kind = "linear"
            else:
                joint = UsdPhysics.RevoluteJoint.Define(stage, joint_path)
                # UsdPhysics revolute limits are authored in DEGREES.
                limit_deg = radians_to_degrees(float(axis.limit))
                joint.CreateLowerLimitAttr().Set(float(-limit_deg))
                joint.CreateUpperLimitAttr().Set(float(limit_deg))
                drive_kind = "angular"
            joint.CreateBody0Rel().SetTargets([body0])
            joint.CreateBody1Rel().SetTargets([body1])
            joint.CreateAxisAttr().Set(axis.world_axis)
            joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
            joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
            joint.CreateLocalRot0Attr().Set(inverse_root)
            joint.CreateLocalRot1Attr().Set(inverse_root)
            joint.CreateCollisionEnabledAttr().Set(False)
            joint.CreateExcludeFromArticulationAttr().Set(False)
            joint.CreateJointEnabledAttr().Set(True)
            drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), drive_kind)
            drive.CreateTypeAttr().Set("force")
            drive.CreateStiffnessAttr().Set(0.0)
            drive.CreateDampingAttr().Set(0.0)
            drive.CreateMaxForceAttr().Set(1.0e6)
            drive.CreateTargetPositionAttr().Set(0.0)
            drive.CreateTargetVelocityAttr().Set(0.0)

        chain_text = " -> ".join(
            [world_path.rsplit("/", 1)[-1], XY_STAGE_JOINT_NAMES[0], XY_STAGE_CARRIAGE_BODY_NAME,
             XY_STAGE_JOINT_NAMES[1]]
            + [
                part
                for index, axis in enumerate(self._compliance_axes)
                for part in (node_paths[index].rsplit("/", 1)[-1], axis.joint_name)
            ]
            + [_BASE_LINK_NAME]
        )
        print(f"[INFO] XY96 stage authored: {chain_text}; disabled '{base_joint_path}'.", flush=True)

    # ------------------------------------------------------------------
    # compliance gains
    # ------------------------------------------------------------------

    def _write_compliance_gains(self, env_ids, stiffness_scale: torch.Tensor) -> None:
        """Write per-environment spring gains for the passive mount joints.

        Damping is scaled by ``sqrt(stiffness_scale)`` so the damping ratio is
        preserved: the randomization varies how soft the mount is, not whether
        it rings.

        Args:
            env_ids: Environment indices to write.
            stiffness_scale: ``(len(env_ids), n_compliance)`` multipliers.
        """
        if not self.compliance_dof_indices:
            return
        stiffness = self._compliance_nominal_stiffness.unsqueeze(0) * stiffness_scale
        damping = self._compliance_nominal_damping.unsqueeze(0) * stiffness_scale.sqrt()
        self.hand.write_joint_stiffness_to_sim(
            stiffness, joint_ids=self.compliance_dof_indices, env_ids=env_ids
        )
        self.hand.write_joint_damping_to_sim(
            damping, joint_ids=self.compliance_dof_indices, env_ids=env_ids
        )

    # ------------------------------------------------------------------
    # rewards
    # ------------------------------------------------------------------

    def _compute_xy_stage_reward(self) -> torch.Tensor:
        """Parent stage costs plus the chatter and drift terms.

        The parent prices *how fast* the stage moves.  These two terms price
        *how often it reverses* and *how far it wanders from home*, which are
        the two behaviours that turn a compensating end effector into one that
        rows the valve around by itself.
        """
        stage_reward = super()._compute_xy_stage_reward()

        action_rate = ((self.xy_executed_action - self.xy_prev_executed_action) ** 2).sum(dim=-1)
        position = self.hand_dof_pos[:, self.xy_dof_index_tensor]
        workspace = max(float(self.xy_workspace_current), 1.0e-6)
        displacement = ((position / workspace) ** 2).mean(dim=-1)

        rate_scale = float(self.cfg.xy_action_rate_penalty_scale)
        displacement_scale = float(self.cfg.xy_displacement_penalty_scale)
        extra = rate_scale * action_rate + displacement_scale * displacement

        self.extras["xy_cost/action_rate"] = action_rate.mean()
        self.extras["xy_penalty/action_rate"] = (rate_scale * action_rate).mean()
        self.extras["xy_cost/displacement"] = displacement.mean()
        self.extras["xy_penalty/displacement"] = (displacement_scale * displacement).mean()
        # Sign-change rate of the commanded action: the direct read-out of the
        # oscillation this task was created to remove. 1.0 means every axis
        # reverses on every control step (Nyquist chatter).
        reversal = (
            (self.xy_executed_action * self.xy_prev_executed_action) < 0.0
        ).float().mean()
        self.extras["xy/action_reversal_rate"] = reversal
        self.xy_prev_executed_action = self.xy_executed_action.clone()

        if self.compliance_dof_indices:
            deflection = self.hand_dof_pos[:, self.compliance_dof_index_tensor]
            for index, axis in enumerate(self._compliance_axes):
                self.extras[f"compliance/{axis.key}_deflection"] = (
                    deflection[:, index].abs().mean()
                )

        stage_reward = stage_reward + extra
        self.extras["xy/stage_reward"] = stage_reward.mean()
        return stage_reward

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = self.hand._ALL_INDICES
        super()._reset_idx(env_ids)
        self.xy_prev_executed_action[env_ids] = 0.0
        if self.compliance_dof_indices and bool(self.cfg.xy_compliance_randomize):
            low, high = (
                float(value) for value in self.cfg.xy_compliance_stiffness_scale_range
            )
            scale = torch.empty(
                (len(env_ids), len(self.compliance_dof_indices)),
                dtype=torch.float,
                device=self.device,
            ).uniform_(low, high)
            self._write_compliance_gains(env_ids, scale)
