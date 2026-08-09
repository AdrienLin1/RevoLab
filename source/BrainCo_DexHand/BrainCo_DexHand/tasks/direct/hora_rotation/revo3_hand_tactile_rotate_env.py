# Copyright (c) 2026, BrainCo.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tactile HORA environment for continuous ball/cylinder rotation.

The environment keeps the HORA rigid-object dynamics, grasp-cache resets, and
domain randomization while adding TacSL fingertip arrays compatible with the
Stage1 teacher and Stage2 student pipeline.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.sensors import ContactSensor
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import saturate
from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, Vt

from .revo3_hand_hora_env import _LOCAL_GROUND_USD, Revo3HandHoraEnv
from .revo3_hand_tactile_rotate_env_cfg import Revo3HandTactileRotateEnvCfg
from ..hora_screw.revo3_hand_screw_tactile_env import Revo3HandScrewTactileEnv
from ...tactile_layout import ESTIMATED_OFFICIAL_LAYOUT


class Revo3HandTactileRotateEnv(Revo3HandHoraEnv):
    """Rotate a free object continuously with tactile sensing."""

    cfg: Revo3HandTactileRotateEnvCfg

    # These helpers are independent of the screw articulation.  Reusing the
    # canonical implementation keeps teacher/student tactile frames identical
    # across all tasks in the T-S pipeline.
    _build_graph_finger_positions = Revo3HandScrewTactileEnv._build_graph_finger_positions
    _compute_physical_tactile_features = (
        Revo3HandScrewTactileEnv._compute_physical_tactile_features
    )
    _compute_graph_tactile_frames = Revo3HandScrewTactileEnv._compute_graph_tactile_frames
    _compute_teacher_student_tactile_frames = (
        Revo3HandScrewTactileEnv._compute_teacher_student_tactile_frames
    )
    _compute_tactile_array_features = Revo3HandScrewTactileEnv._compute_tactile_array_features
    _build_teacher_tactile_priv = Revo3HandScrewTactileEnv._build_teacher_tactile_priv
    _select_active_finger_slots = Revo3HandScrewTactileEnv._select_active_finger_slots
    _setup_tactile_visualizers = Revo3HandScrewTactileEnv._setup_tactile_visualizers
    _setup_tactile_taxel_spheres = Revo3HandScrewTactileEnv._setup_tactile_taxel_spheres
    _contact_forces_w = Revo3HandScrewTactileEnv._contact_forces_w
    _tactile_taxel_positions_w = Revo3HandScrewTactileEnv._tactile_taxel_positions_w
    _update_tactile_debug_visualization = (
        Revo3HandScrewTactileEnv._update_tactile_debug_visualization
    )

    def __init__(
        self,
        cfg: Revo3HandTactileRotateEnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ):
        """Initialize action-delay and T-S tactile history buffers.

        Args:
            cfg: Resolved tactile rotation environment configuration.
            render_mode: Optional Isaac Lab render mode.
            **kwargs: Additional arguments forwarded to ``DirectRLEnv``.
        """
        super().__init__(cfg, render_mode, **kwargs)

        self.object_size_scale_ids = self.object_size_scale_ids.to(self.device)
        self.object_size_scales = self.object_size_scales.to(self.device)
        self.priv_info_buf[:, 8] = self.object_size_scales
        self.delayed_targets = self.cur_targets.clone()
        self.action_delay = torch.zeros(
            self.num_envs,
            device=self.device,
            dtype=torch.float,
        )
        self._physics_substep_idx = 0

        self.student_tactile_hist_buf = torch.zeros(
            (
                self.num_envs,
                self.cfg.student_tactile_history_len,
                self.cfg.student_tactile_frame_dim,
            ),
            device=self.device,
            dtype=torch.float,
        )
        self.teacher_tactile_hist_buf = torch.zeros(
            (
                self.num_envs,
                self.cfg.teacher_tactile_history_len,
                self.cfg.teacher_tactile_frame_dim,
            ),
            device=self.device,
            dtype=torch.float,
        )
        self.prev_teacher_tactile_buf = torch.zeros(
            (self.num_envs, self.cfg.tactile_current_dim),
            device=self.device,
            dtype=torch.float,
        )

        num_taxels = int(self.cfg.student_tactile_num_taxels)
        self.student_prev_contact_b = torch.zeros(
            (self.num_envs, num_taxels), device=self.device, dtype=torch.float
        )
        self.student_contact_duration = torch.zeros_like(self.student_prev_contact_b)
        self.student_hysteresis_contact_b = torch.zeros_like(self.student_prev_contact_b)
        self.prev_graph_force = torch.zeros(
            (self.num_envs, num_taxels, 3), device=self.device, dtype=torch.float
        )
        num_tactile_fingers = len(self.cfg.tactile_tip_body_names)
        self.graph_prev_centroid = torch.zeros(
            (self.num_envs, num_tactile_fingers, 2),
            device=self.device,
            dtype=torch.float,
        )
        self.graph_prev_contact_valid = torch.zeros(
            (self.num_envs, num_tactile_fingers),
            device=self.device,
            dtype=torch.bool,
        )
        self.graph_shift_ema = torch.zeros_like(self.graph_prev_centroid)
        self._graph_finger_positions = self._build_graph_finger_positions()
        self.tactile_tip_body_ids = [
            self.hand.body_names.index(body_name)
            for body_name in self.cfg.tactile_tip_body_names
        ]

        self._tactile_taxel_visualizer = None
        self._tactile_force_tip_visualizer = None
        self._tactile_debug_draw = None
        self._tactile_vis_warned = False
        self._setup_tactile_visualizers()

    def _setup_scene(self) -> None:
        """Create HORA assets plus SDF-backed TacSL fingertip sensors."""
        self.hand = Articulation(self.cfg.robot_cfg)
        self.object = RigidObject(self.cfg.object_cfg)
        self._configure_object_sdf_collision(0)
        spawn_ground_plane(
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(usd_path=_LOCAL_GROUND_USD),
        )
        self.scene.clone_environments(copy_from_source=False)
        self._initialize_object_size_randomization()
        self.scene.filter_collisions(global_prim_paths=["/World/ground"])
        self.scene.articulations["hand"] = self.hand
        self.scene.rigid_objects["object"] = self.object

        self._contact_sensor = []
        for sensor_index, sensor_cfg in enumerate(self.cfg.contact_sensor):
            sensor = ContactSensor(sensor_cfg)
            self._contact_sensor.append(sensor)
            self.scene.sensors[f"contact_sensor_{sensor_index}"] = sensor

        self._tactile_sensor = []
        for sensor_name, sensor_cfg in zip(
            self.cfg.tactile_vis_sensor_names,
            self.cfg.tactile_sensor,
        ):
            sensor = sensor_cfg.class_type(sensor_cfg)
            self._tactile_sensor.append(sensor)
            self.scene.sensors[sensor_name] = sensor

        light_cfg = sim_utils.DomeLightCfg(
            intensity=2000.0,
            color=(0.75, 0.75, 0.75),
        )
        light_cfg.func("/World/Light", light_cfg)

    def _configure_object_sdf_collision(self, env_index: int) -> UsdGeom.Mesh:
        """Switch one generated object mesh to PhysX SDF collision.

        Args:
            env_index: Index of the cloned environment whose mesh is configured.

        Returns:
            The configured USD mesh schema.
        """
        mesh_path = f"{self.scene.env_prim_paths[env_index]}/object/geometry/mesh"
        mesh_prim = self.scene.stage.GetPrimAtPath(mesh_path)
        if not mesh_prim.IsValid():
            raise RuntimeError(f"Tactile rotation object mesh not found at '{mesh_path}'.")
        UsdPhysics.CollisionAPI.Apply(mesh_prim).CreateCollisionEnabledAttr().Set(True)
        UsdPhysics.MeshCollisionAPI.Apply(mesh_prim).CreateApproximationAttr().Set("sdf")
        sdf_api = PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(mesh_prim)
        sdf_api.CreateSdfResolutionAttr().Set(int(self.cfg.tactile_sdf_resolution))
        return UsdGeom.Mesh(mesh_prim)

    def _initialize_object_size_randomization(self) -> None:
        """Assign persistent object-size buckets and author every cloned SDF mesh.

        The per-environment point override is required even for nominal scale.
        With ``replicate_physics=False``, relying on the env-0 SDF API alone can
        leave cloned meshes without a cooked PhysX SDF shape.
        """
        prototype_mesh = self._configure_object_sdf_collision(0)
        nominal_points = tuple(prototype_mesh.GetPointsAttr().Get())
        scale_levels = torch.tensor(
            self.cfg.object_size_scale_levels,
            dtype=torch.float32,
            device="cpu",
        )
        num_envs = len(self.scene.env_prim_paths)
        if self.cfg.randomize_object_size:
            repeats = (num_envs + len(scale_levels) - 1) // len(scale_levels)
            scale_ids = torch.arange(len(scale_levels), dtype=torch.long).repeat(repeats)
            scale_ids = scale_ids[:num_envs][torch.randperm(num_envs)]
        else:
            nominal_id = int(torch.argmin(torch.abs(scale_levels - 1.0)).item())
            scale_ids = torch.full((num_envs,), nominal_id, dtype=torch.long)

        self.object_size_scale_ids = scale_ids
        self.object_size_scales = scale_levels[scale_ids]
        for env_index, scale in enumerate(self.object_size_scales.tolist()):
            self._apply_object_size_scale_to_env(
                env_index,
                float(scale),
                nominal_points,
            )

    def _apply_object_size_scale_to_env(
        self,
        env_index: int,
        scale: float,
        nominal_points: tuple,
    ) -> None:
        """Scale one object's mesh points and explicitly author its SDF schema.

        Args:
            env_index: Index of the cloned parallel environment.
            scale: Multiplicative object-size scale.
            nominal_points: Unscaled prototype mesh points.
        """
        mesh = self._configure_object_sdf_collision(env_index)
        axes = tuple(bool(value) for value in self.cfg.object_size_scale_axes)
        factors = tuple(scale if enabled else 1.0 for enabled in axes)
        scaled_points = Vt.Vec3fArray(
            [
                Gf.Vec3f(
                    float(point[0]) * factors[0],
                    float(point[1]) * factors[1],
                    float(point[2]) * factors[2],
                )
                for point in nominal_points
            ]
        )
        mesh.GetPointsAttr().Set(scaled_points)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Prepare one control step while retaining the prior delayed target."""
        self._physics_substep_idx = 0
        super()._pre_physics_step(actions)

    def _apply_action(self) -> None:
        """Apply per-environment action delay across physics substeps."""
        self._refresh_lab()
        switch_at = self.action_delay * float(self.cfg.decimation)
        use_delayed = self._physics_substep_idx < switch_at
        applied_targets = torch.where(
            use_delayed.unsqueeze(-1),
            self.delayed_targets,
            self.cur_targets,
        )
        if self.cfg.torque_control:
            self.torques = (
                self.p_gain * (applied_targets - self.hand_dof_pos)
                - self.d_gain * self.hand_dof_vel
            )
            self.hand.set_joint_effort_target(
                self.torques[:, self.actuated_dof_indices],
                joint_ids=self.actuated_dof_indices,
            )
        else:
            self.hand.set_joint_position_target(
                applied_targets[:, self.actuated_dof_indices],
                joint_ids=self.actuated_dof_indices,
            )

        self._physics_substep_idx += 1
        if self._physics_substep_idx >= self.cfg.decimation:
            self.prev_targets[:, self.actuated_dof_indices] = self.cur_targets[
                :, self.actuated_dof_indices
            ]
            self.delayed_targets[:, self.actuated_dof_indices] = self.cur_targets[
                :, self.actuated_dof_indices
            ]

    def compute_observations(self) -> torch.Tensor:
        """Build teacher and student observations with aligned target commands."""
        at_reset_env_ids = self.at_reset_buf.nonzero(as_tuple=False).squeeze(-1)
        obs_buf = super().compute_observations()
        tactile = (
            self._compute_physical_tactile_features()
            if self.cfg.tactile_layout == ESTIMATED_OFFICIAL_LAYOUT
            else self._compute_tactile_array_features()
        )
        teacher_force = tactile
        if self.cfg.enable_visible_contact_noise:
            teacher_force = _apply_force_vector_noise(
                tactile.view(self.num_envs, -1, 3),
                float(self.cfg.visible_contact_force_noise_frac),
            ).reshape_as(tactile)
            teacher_force = torch.clamp(
                teacher_force,
                -float(self.cfg.tactile_force_clip),
                float(self.cfg.tactile_force_clip),
            )

        if self.cfg.tactile_layout == ESTIMATED_OFFICIAL_LAYOUT:
            teacher_frame, student_frame = self._compute_graph_tactile_frames(
                force_clean=tactile,
                force_teacher=teacher_force,
            )
        else:
            teacher_frame, student_frame = self._compute_teacher_student_tactile_frames(
                force_clean=tactile,
                force_teacher=teacher_force,
            )

        teacher_tactile = self._build_teacher_tactile_priv(
            teacher_frame,
            at_reset_env_ids,
        )
        self.priv_info_buf[:, self.cfg.tactile_priv_offset:] = teacher_tactile
        self.teacher_tactile_hist_buf[:] = torch.cat(
            [self.teacher_tactile_hist_buf[:, 1:], teacher_frame.unsqueeze(1)],
            dim=1,
        )
        self.student_tactile_hist_buf[:] = torch.cat(
            [self.student_tactile_hist_buf[:, 1:], student_frame.unsqueeze(1)],
            dim=1,
        )
        if len(at_reset_env_ids) > 0:
            self.teacher_tactile_hist_buf[at_reset_env_ids] = teacher_frame[
                at_reset_env_ids
            ].unsqueeze(1).repeat(1, self.cfg.teacher_tactile_history_len, 1)
            self.student_tactile_hist_buf[at_reset_env_ids] = student_frame[
                at_reset_env_ids
            ].unsqueeze(1).repeat(1, self.cfg.student_tactile_history_len, 1)

        self.extras["tactile/priv_abs_mean"] = teacher_force.abs().mean()
        self.extras["tactile/priv_abs_max"] = teacher_force.abs().max()
        if self.cfg.tactile_layout == ESTIMATED_OFFICIAL_LAYOUT:
            graph_nodes = int(self.cfg.tactile_graph_total_nodes)
            student_nodes = student_frame[:, : graph_nodes * 5].view(
                self.num_envs, graph_nodes, 5
            )
            self.extras["tactile/student_contact_mean"] = student_nodes[..., 0].mean()
            self.extras["tactile/student_duration_mean"] = student_nodes[..., 3].mean()
        else:
            self.extras["tactile/student_contact_mean"] = student_frame[:, 0::3].mean()
            self.extras["tactile/student_duration_mean"] = student_frame[:, 2::3].mean()
        self._update_tactile_debug_visualization()
        return obs_buf

    def _get_observations(self) -> dict[str, torch.Tensor]:
        """Return Stage1 teacher and Stage2 student tactile observations."""
        obs_dict = super()._get_observations()
        history = self.obs_buf_lag_history[:, -self.cfg.student_proprio_history_len:]
        obs_dict["student_proprio_hist"] = history[
            :, :, : 2 * self.num_hand_dofs
        ].clone()
        obs_dict["student_tactile_hist"] = self.student_tactile_hist_buf.clone()
        obs_dict["tactile_hist"] = self.teacher_tactile_hist_buf.clone()
        return obs_dict

    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        """Reset the grasp, randomization, and tactile histories."""
        if env_ids is None:
            env_ids = self.hand._ALL_INDICES
        env_ids_t = torch.as_tensor(
            env_ids,
            dtype=torch.long,
            device=self.device,
        ).reshape(-1)
        super()._reset_idx(env_ids)
        self._apply_reset_domain_randomization(env_ids_t)
        self.student_tactile_hist_buf[env_ids] = 0.0
        self.teacher_tactile_hist_buf[env_ids] = 0.0
        self.prev_teacher_tactile_buf[env_ids] = 0.0
        self.student_prev_contact_b[env_ids] = 0.0
        self.student_contact_duration[env_ids] = 0.0
        self.student_hysteresis_contact_b[env_ids] = 0.0
        self.prev_graph_force[env_ids] = 0.0
        self.graph_prev_centroid[env_ids] = 0.0
        self.graph_prev_contact_valid[env_ids] = False
        self.graph_shift_ema[env_ids] = 0.0

    def _apply_reset_domain_randomization(self, env_ids: torch.Tensor) -> None:
        """Apply shared hora_screw reset randomizations to free objects.

        Args:
            env_ids: One-dimensional device tensor of environment indices.
        """
        if len(env_ids) == 0:
            return
        if self.cfg.randomize_friction:
            self._randomize_reset_friction(env_ids)

        if self.cfg.randomize_object_xy_position:
            xy_noise = float(self.cfg.object_xy_position_noise)
            xy_offset = torch.empty(
                (len(env_ids), 2),
                device=self.device,
            ).uniform_(-xy_noise, xy_noise)
            object_state = self.object.data.root_state_w[env_ids].clone()
            object_state[:, :2] += xy_offset
            self.object.write_root_pose_to_sim(object_state[:, :7], env_ids)
            self.object_default_pose[env_ids, :2] += xy_offset
            self.extras["randomization/object_xy_offset_norm"] = (
                xy_offset.norm(dim=-1).mean()
            )

        reset_joint_noise_frac = float(self.cfg.reset_joint_noise_frac)
        if reset_joint_noise_frac > 0.0:
            half_range = (
                self.hand_dof_upper_limits[env_ids]
                - self.hand_dof_lower_limits[env_ids]
            ) / 2.0
            joint_noise = (
                torch.rand(
                    (len(env_ids), self.num_hand_dofs),
                    device=self.device,
                )
                * 2.0
                - 1.0
            ) * reset_joint_noise_frac * half_range
            dof_pos = saturate(
                self.hand.data.joint_pos[env_ids].clone() + joint_noise,
                self.hand_dof_lower_limits[env_ids],
                self.hand_dof_upper_limits[env_ids],
            )
            dof_vel = torch.zeros_like(self.hand.data.joint_vel[env_ids])
            self.hand.set_joint_position_target(dof_pos, env_ids=env_ids)
            self.hand.write_joint_state_to_sim(dof_pos, dof_vel, env_ids=env_ids)

        self.delayed_targets[env_ids] = self.cur_targets[env_ids]
        delay_low, delay_high = self.cfg.action_delay
        self.action_delay[env_ids] = torch.empty(
            len(env_ids),
            device=self.device,
        ).uniform_(float(delay_low), float(delay_high))
        self._refresh_lab()
        self.object_pos_prev[env_ids] = self.object_pos[env_ids]
        self.object_rot_prev[env_ids] = self.object_rot[env_ids]
        self.extras["randomization/action_delay_mean"] = self.action_delay.mean()

    def _randomize_reset_friction(self, env_ids: torch.Tensor) -> None:
        """Resample the shared hand/object friction scale at reset.

        Args:
            env_ids: One-dimensional device tensor of environment indices.
        """
        friction_scale = torch.empty(
            len(env_ids),
            device=self.device,
        ).uniform_(
            float(self.cfg.randomize_friction_scale_lower),
            float(self.cfg.randomize_friction_scale_upper),
        )
        for asset, base_friction in (
            (self.object, float(self.cfg.object_base_friction)),
            (self.hand, float(self.cfg.metal_base_friction)),
        ):
            materials = asset.root_physx_view.get_material_properties()
            values = friction_scale.to(materials.device) * base_friction
            material_env_ids = env_ids.to(materials.device)
            materials[material_env_ids, :, 0] = values.unsqueeze(1)
            materials[material_env_ids, :, 1] = values.unsqueeze(1)
            asset.root_physx_view.set_material_properties(materials, env_ids.cpu())
        self.priv_info_buf[env_ids, 3] = friction_scale


def _apply_force_vector_noise(
    force: torch.Tensor,
    fraction: float,
) -> torch.Tensor:
    """Apply magnitude-relative random-direction noise to force vectors.

    Args:
        force: Force vectors with final dimension three.
        fraction: Noise magnitude as a fraction of the force norm.

    Returns:
        Noisy force vectors with the original shape.
    """
    if fraction <= 0.0:
        return force
    direction = F.normalize(torch.randn_like(force), dim=-1)
    magnitude = torch.linalg.vector_norm(force, dim=-1, keepdim=True)
    return force + fraction * magnitude * direction
