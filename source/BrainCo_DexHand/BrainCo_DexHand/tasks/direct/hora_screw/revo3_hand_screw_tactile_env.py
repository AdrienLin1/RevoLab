# Copyright (c) 2026, BrainCo.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""DirectRLEnv for Revo3 HORA screw/valve tasks with TacSL fingertip sensing.

Extends Revo3HandScrewEnv with either five regular 16x16 TacSL force fields or
the estimated 21/31-node physical sensor layout. Layout-specific frames are
written into the tail of ``priv_info`` and into tactile history. The actor
observation (141 dims) and all
terminations/randomizations of the base task are unchanged. Optional reward terms:

* ``multi_contact_reward`` (valve tasks): multi-finger coordination bonus from TacSL.
* ``visible_contact_reward`` (nutbolt/screwdriver tactile): reward TacSL-visible
  contacts only when the same finger contributes positive object-axis torque.
* ``coord_endogenous_reward`` (opt-in via ``enable_coord_endogenous_reward``):
  an annealed intrinsic utility for stable, capacity-aware positive-torque load
  sharing (+ optional handover and inefficient-force penalties).

The TacSL force field computes per-taxel penetration against the rotating
nut/handle via PhysX SDF queries, which requires that body's collision mesh to
use the "sdf" approximation. The approximation override is applied to the
env_0 prototype before cloning, so every cloned env inherits it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sensors import ContactSensor
from isaaclab.sim.views import XformPrimView
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_apply
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics, Vt

from .revo3_hand_screw_env import _LOCAL_GROUND_USD, Revo3HandScrewEnv, apply_force_vector_noise
from .revo3_hand_screw_tactile_env_cfg import Revo3HandScrewTactileMixinCfg
from ...tactile_layout import (
    ESTIMATED_OFFICIAL_LAYOUT,
    normalized_estimated_official_centers_xy,
)


class Revo3HandScrewTactileEnv(Revo3HandScrewEnv):
    cfg: Revo3HandScrewTactileMixinCfg

    def __init__(self, cfg: Revo3HandScrewTactileMixinCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._contact_buffer_peak_utilization = torch.tensor(0.0, device=self.device)
        # Stage2 student: structural tactile history [b, Δb, d] per taxel (no force mag).
        self.student_tactile_hist_buf = torch.zeros(
            (self.num_envs, self.cfg.student_tactile_history_len, self.cfg.student_tactile_frame_dim),
            device=self.device, dtype=torch.float,
        )
        # Stage1 teacher: [Fx, Fy, Fz, d_norm] history (same T / taxel grid as student).
        self.teacher_tactile_hist_buf = torch.zeros(
            (self.num_envs, self.cfg.teacher_tactile_history_len, self.cfg.teacher_tactile_frame_dim),
            device=self.device, dtype=torch.float,
        )
        self.prev_teacher_tactile_buf = torch.zeros(
            (self.num_envs, self.cfg.tactile_current_dim),
            device=self.device, dtype=torch.float,
        )
        n_taxels = int(self.cfg.student_tactile_num_taxels)
        self.student_prev_contact_b = torch.zeros(
            (self.num_envs, n_taxels), device=self.device, dtype=torch.float
        )
        self.student_contact_duration = torch.zeros(
            (self.num_envs, n_taxels), device=self.device, dtype=torch.float
        )
        self.student_hysteresis_contact_b = torch.zeros(
            (self.num_envs, n_taxels), device=self.device, dtype=torch.float
        )
        self.prev_graph_force = torch.zeros(
            (self.num_envs, n_taxels, 3), device=self.device, dtype=torch.float
        )
        num_tactile_fingers = len(self.cfg.tactile_tip_body_names)
        self.graph_prev_centroid = torch.zeros(
            (self.num_envs, num_tactile_fingers, 2), device=self.device, dtype=torch.float
        )
        self.graph_prev_contact_valid = torch.zeros(
            (self.num_envs, num_tactile_fingers), device=self.device, dtype=torch.bool
        )
        self.graph_shift_ema = torch.zeros_like(self.graph_prev_centroid)
        self._graph_finger_positions = self._build_graph_finger_positions()
        # per-finger contact duty cycle (EMA of the tanh contact indicator),
        # updated once per control step in _get_rewards
        self.contact_duty_ema = torch.zeros(
            (self.num_envs, len(self.cfg.tactile_tip_body_names)), device=self.device
        )
        self.tactile_tip_body_ids = [
            self.hand.body_names.index(body_name) for body_name in self.cfg.tactile_tip_body_names
        ]
        self.visible_task_reward_abs_ema = torch.tensor(0.0, device=self.device)
        self.visible_progress_abs_ema = torch.tensor(0.0, device=self.device)
        self.visible_contact_adaptive_scale = torch.tensor(1.0, device=self.device)
        self._visible_adaptive_updates = 0
        self._setup_coord_endogenous_buffers()
        self._tactile_taxel_visualizer = None
        self._tactile_force_tip_visualizer = None
        self._tactile_debug_draw = None
        self._tactile_vis_warned = False
        self._setup_tactile_visualizers()

    def _setup_coord_endogenous_buffers(self) -> None:
        """Allocate buffers for the task-progress-gated intrinsic utility."""
        num_fingers = len(self.cfg.tactile_tip_body_names)
        self._coord_enabled = bool(self.cfg.enable_coord_endogenous_reward)
        self._coord_global_steps = 0
        self.coord_finger_mask = None
        self.coord_finger_mask_sum = None
        self.coord_load_hist = None
        self.coord_load_hist_count = None
        if not self._coord_enabled:
            self.coord_omega_hist = None
            self.coord_b_prev = None
            self.coord_release_age = None
            return
        masked_joint_names = tuple(getattr(self.cfg, "masked_action_joint_names", ()))
        finger_mask = torch.ones(num_fingers, device=self.device, dtype=torch.float)
        for finger_idx, body_name in enumerate(self.cfg.tactile_tip_body_names):
            finger_name = self._finger_name_from_body_name(body_name)
            if finger_name is None:
                continue
            if any(f"_{finger_name}_" in joint_name for joint_name in masked_joint_names):
                finger_mask[finger_idx] = 0.0
        self.coord_finger_mask = finger_mask
        self.coord_finger_mask_sum = finger_mask.sum().clamp(min=1.0)
        w_q = int(self.cfg.coord_w_q)
        self.coord_omega_hist = torch.zeros(
            (self.num_envs, w_q), device=self.device, dtype=torch.float
        )
        self.coord_load_hist = torch.zeros(
            (self.num_envs, int(self.cfg.coord_load_window), num_fingers),
            device=self.device,
            dtype=torch.float,
        )
        self.coord_load_hist_count = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float
        )
        self.coord_b_prev = torch.zeros(
            (self.num_envs, num_fingers), device=self.device, dtype=torch.float
        )
        self.coord_release_age = torch.full(
            (self.num_envs, num_fingers),
            fill_value=float(self.cfg.coord_delta_h + 1),
            device=self.device,
            dtype=torch.float,
        )

    def _finger_name_from_body_name(self, body_name: str) -> str | None:
        """Return canonical finger name parsed from a Revo3 fingertip body."""
        for finger_name in ("thumb", "index", "middle", "ring", "little"):
            if f"_{finger_name}_" in body_name:
                return finger_name
        return None

    def _build_graph_finger_positions(self) -> tuple[torch.Tensor, ...]:
        """Return normalized physical-node coordinates for contact-shift features."""
        if self.cfg.tactile_layout != ESTIMATED_OFFICIAL_LAYOUT:
            return ()
        positions = []
        for finger_name in self.cfg.tactile_active_finger_names:
            xy = torch.tensor(
                normalized_estimated_official_centers_xy(finger_name),
                device=self.device,
                dtype=torch.float,
            )
            positions.append(xy)
        return tuple(positions)

    def _setup_scene(self):
        # Mirrors Revo3HandScrewEnv._setup_scene, inserting the SDF collision
        # override (before cloning) and the TacSL fingertip sensors.
        self.hand = Articulation(self.cfg.robot_cfg)
        self.object = Articulation(self.cfg.object_cfg)
        self._apply_sdf_collision_to_screw_body()
        # Authoring hook for task variants that extend the hand articulation
        # (e.g. the two-axis translation stage). It must run on the env_0
        # prototype before cloning so every cloned env inherits the topology.
        self._author_robot_stage_overrides()
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(usd_path=_LOCAL_GROUND_USD))
        self.scene.clone_environments(copy_from_source=False)
        self._initialize_object_radius_randomization()
        self.scene.filter_collisions(global_prim_paths=["/World/ground"])
        self.scene.articulations["hand"] = self.hand
        self.scene.articulations["object"] = self.object
        self._contact_sensor = []
        for id in range(len(self.cfg.contact_sensor)):
            self._contact_sensor.append(ContactSensor(self.cfg.contact_sensor[id]))
            self.scene.sensors[f"contact_sensor_{id}"] = self._contact_sensor[id]
        self._nut_contact_sensor = ContactSensor(self.cfg.nut_contact_sensor)
        self.scene.sensors["nut_contact_sensor"] = self._nut_contact_sensor
        # TacSL fingertip array sensors (force field only)
        self._tactile_sensor = []
        for sensor_name, sensor_cfg in zip(self.cfg.tactile_vis_sensor_names, self.cfg.tactile_sensor):
            sensor = sensor_cfg.class_type(sensor_cfg)
            self._tactile_sensor.append(sensor)
            self.scene.sensors[sensor_name] = sensor
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _author_robot_stage_overrides(self) -> None:
        """Extend the spawned hand articulation before environment cloning.

        The base tactile task keeps the hand exactly as authored in USD. The
        two-axis translation variant overrides this to add its prismatic stage.
        """
        return None

    def _apply_sdf_collision_to_screw_body(self):
        """Switch the rotating nut/handle collision mesh to PhysX SDF approximation.

        TacSL's ``_find_contact_object_components`` only accepts meshes whose
        ``UsdPhysics.MeshCollisionAPI`` approximation is "sdf", and the force
        field needs an SDF shape view for penetration queries.

        The URDF importer authors ``CollisionAPI + MeshCollisionAPI(convexHull)``
        on the ``collisions/.../node_*`` Xform and gives the collision mesh a
        non-identity local pose (URDF collision origin offset + STL scale).
        TacSL queries the SDF with points expressed in the rigid body frame, so
        a new collision Mesh prim is authored directly under the screw body
        with the mesh-to-body transform baked into its points (identity local
        pose). The original compound convex collider is disabled to avoid
        duplicate contact shapes.
        """
        stage = self.scene.stage
        body_path = f"{self.scene.env_prim_paths[0]}/object/{self.cfg.nut_body_name}"
        body_prim = stage.GetPrimAtPath(body_path)
        if not body_prim.IsValid():
            raise RuntimeError(f"Screw body prim not found at '{body_path}'.")

        # locate the URDF importer's compound collision Xform (may be an instance proxy)
        col_prim = None
        for prim in Usd.PrimRange(body_prim, Usd.TraverseInstanceProxies()):
            if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                col_prim = prim
                break
        if col_prim is None:
            raise RuntimeError(
                f"No mesh collision found under '{body_path}' to convert to SDF for TacSL force field. "
                "The rotating screw body must use a mesh collision (not a primitive shape)."
            )
        col_path = col_prim.GetPath()
        src_mesh_prim = None
        for prim in Usd.PrimRange(col_prim, Usd.TraverseInstanceProxies()):
            if prim.IsA(UsdGeom.Mesh):
                src_mesh_prim = prim
                break
        if src_mesh_prim is None:
            raise RuntimeError(f"No Mesh prim found under collision node '{col_path}'.")

        # bake the mesh-to-body transform (URDF collision origin + STL scale) into the points
        src_mesh = UsdGeom.Mesh(src_mesh_prim)
        xform_cache = UsdGeom.XformCache()
        mesh_to_body = xform_cache.GetLocalToWorldTransform(src_mesh_prim) * xform_cache.GetLocalToWorldTransform(
            body_prim
        ).GetInverse()
        baked_points = Vt.Vec3fArray(
            [Gf.Vec3f(mesh_to_body.Transform(Gf.Vec3d(p))) for p in src_mesh.GetPointsAttr().Get()]
        )

        sdf_mesh_path = body_prim.GetPath().AppendChild("tacsl_sdf_collision").AppendChild("mesh")
        sdf_mesh = UsdGeom.Mesh.Define(stage, sdf_mesh_path)
        sdf_mesh.CreatePointsAttr(baked_points)
        sdf_mesh.CreateFaceVertexCountsAttr(src_mesh.GetFaceVertexCountsAttr().Get())
        sdf_mesh.CreateFaceVertexIndicesAttr(src_mesh.GetFaceVertexIndicesAttr().Get())
        sdf_mesh.CreateSubdivisionSchemeAttr("none")
        sdf_mesh.CreatePurposeAttr(UsdGeom.Tokens.guide)  # collision-only, never rendered

        UsdPhysics.CollisionAPI.Apply(sdf_mesh.GetPrim()).CreateCollisionEnabledAttr().Set(True)
        UsdPhysics.MeshCollisionAPI.Apply(sdf_mesh.GetPrim()).CreateApproximationAttr().Set("sdf")
        sdf_api = PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(sdf_mesh.GetPrim())
        sdf_api.CreateSdfResolutionAttr().Set(int(self.cfg.tactile_sdf_resolution))

        # de-instance so the original collider (an instance proxy) can be disabled
        for prim in Usd.PrimRange(body_prim):
            if prim.IsInstanceable():
                prim.SetInstanceable(False)
        UsdPhysics.CollisionAPI(stage.GetPrimAtPath(col_path)).CreateCollisionEnabledAttr().Set(False)

        print(
            f"[INFO] TacSL: replaced convex collider '{col_path}' with baked SDF mesh "
            f"(resolution={self.cfg.tactile_sdf_resolution}) at '{sdf_mesh_path.pathString}'.",
            flush=True,
        )

    def _initialize_object_radius_randomization(self) -> None:
        """Capture the nominal TacSL mesh before applying per-environment scales.

        Cloned environments can inherit geometry from env 0. Retaining one
        nominal point array prevents env 0's scale from being compounded when
        the remaining SDF mesh overrides are authored.
        """
        body_path = f"{self.scene.env_prim_paths[0]}/object/{self.cfg.nut_body_name}"
        sdf_mesh_path = f"{body_path}/tacsl_sdf_collision/mesh"
        sdf_mesh = UsdGeom.Mesh.Get(self.scene.stage, sdf_mesh_path)
        if not sdf_mesh.GetPrim().IsValid():
            raise RuntimeError(f"TacSL SDF mesh not found at '{sdf_mesh_path}'.")
        self._nominal_tactile_sdf_points = tuple(sdf_mesh.GetPointsAttr().Get())
        try:
            super()._initialize_object_radius_randomization()
        finally:
            del self._nominal_tactile_sdf_points

    def _apply_object_radius_scale_to_env(self, env_index: int, scale: float) -> None:
        """Apply one radius scale to both rendered geometry and the TacSL SDF.

        SDF points are baked in the rotating rigid body's local frame because
        TacSL submits its queries in that frame. Scaling those points directly
        keeps tactile penetration depth aligned with the PhysX collision shape.

        Args:
            env_index: Index of the cloned parallel environment.
            scale: Multiplicative radius scale applied to local X and Y.
        """
        super()._apply_object_radius_scale_to_env(env_index, scale)
        body_path = (
            f"{self.scene.env_prim_paths[env_index]}/object/{self.cfg.nut_body_name}"
        )
        sdf_mesh_path = f"{body_path}/tacsl_sdf_collision/mesh"
        sdf_mesh = UsdGeom.Mesh.Get(self.scene.stage, sdf_mesh_path)
        if not sdf_mesh.GetPrim().IsValid():
            raise RuntimeError(f"TacSL SDF mesh not found at '{sdf_mesh_path}'.")
        scaled_points = Vt.Vec3fArray(
            [Gf.Vec3f(float(point[0]) * scale, float(point[1]) * scale, float(point[2]))
             for point in self._nominal_tactile_sdf_points]
        )
        sdf_mesh.GetPointsAttr().Set(scaled_points)

    def compute_observations(self):
        # capture reset flags before the base class consumes them
        at_reset_env_ids = self.at_reset_buf.nonzero(as_tuple=False).squeeze(-1)
        obs_buf = super().compute_observations()
        tactile = (
            self._compute_physical_tactile_features()
            if self.cfg.tactile_layout == ESTIMATED_OFFICIAL_LAYOUT
            else self._compute_tactile_array_features()
        )
        teacher_force = tactile
        if self.cfg.enable_visible_contact_noise:
            teacher_force = apply_force_vector_noise(
                tactile.view(self.num_envs, -1, 3),
                self.cfg.visible_contact_force_noise_frac,
            ).reshape_as(tactile)
            teacher_force = torch.clamp(
                teacher_force,
                -self.cfg.tactile_force_clip,
                self.cfg.tactile_force_clip,
            )
        # ``Fn`` originates from TacSL's non-negative compression magnitude.
        # Preserve that physical invariant after optional vector noise as well.
        teacher_force = teacher_force.view(self.num_envs, -1, 3)
        teacher_force[..., 0].clamp_min_(0.0)
        teacher_force = teacher_force.reshape_as(tactile)
        if self.cfg.tactile_layout == ESTIMATED_OFFICIAL_LAYOUT:
            teacher_frame, struct_frame = self._compute_graph_tactile_frames(
                force_clean=tactile,
                force_teacher=teacher_force,
            )
        else:
            # Shared d_norm from clean contact; teacher=[F,d], student=[b,delta_b,d].
            teacher_frame, struct_frame = self._compute_teacher_student_tactile_frames(
                force_clean=tactile,
                force_teacher=teacher_force,
            )
        teacher_tactile = self._build_teacher_tactile_priv(teacher_frame, at_reset_env_ids)
        self.priv_info_buf[:, self.cfg.tactile_priv_offset:] = teacher_tactile
        self.teacher_tactile_hist_buf[:] = torch.cat(
            [self.teacher_tactile_hist_buf[:, 1:], teacher_frame.unsqueeze(1)], dim=1)
        if len(at_reset_env_ids) > 0:
            self.teacher_tactile_hist_buf[at_reset_env_ids] = teacher_frame[at_reset_env_ids].unsqueeze(
                1
            ).repeat(1, self.cfg.teacher_tactile_history_len, 1)
        self.extras["tactile/priv_abs_mean"] = teacher_force.abs().mean()
        self.extras["tactile/priv_abs_max"] = teacher_force.abs().max()
        if self.cfg.tactile_teacher_use_delta:
            self.extras["tactile/priv_delta_abs_mean"] = teacher_tactile[:, teacher_frame.shape[-1]:].abs().mean()

        self.student_tactile_hist_buf[:] = torch.cat(
            [self.student_tactile_hist_buf[:, 1:], struct_frame.unsqueeze(1)], dim=1)
        if len(at_reset_env_ids) > 0:
            self.student_tactile_hist_buf[at_reset_env_ids] = struct_frame[at_reset_env_ids].unsqueeze(1).repeat(
                1, self.cfg.student_tactile_history_len, 1)
        if self.cfg.tactile_layout == ESTIMATED_OFFICIAL_LAYOUT:
            graph_nodes = int(self.cfg.tactile_graph_total_nodes)
            student_nodes = struct_frame[:, : graph_nodes * 5].view(self.num_envs, graph_nodes, 5)
            self.extras["tactile/student_contact_mean"] = student_nodes[..., 0].mean()
            self.extras["tactile/student_duration_mean"] = student_nodes[..., 3].mean()
            self.extras["tactile/teacher_duration_mean"] = student_nodes[..., 3].mean()
        else:
            self.extras["tactile/student_contact_mean"] = struct_frame[:, 0::3].mean()
            self.extras["tactile/student_duration_mean"] = struct_frame[:, 2::3].mean()
            self.extras["tactile/teacher_duration_mean"] = teacher_frame[:, 3::4].mean()
        self._update_tactile_debug_visualization()
        return obs_buf

    def _compute_physical_tactile_features(self) -> torch.Tensor:
        """Return each estimated physical sensor's integrated 3D force once.

        The estimated layout evaluates several SDF quadrature points per circular
        sensor. ``Revo3VisuoTactileSensor`` integrates those samples into the
        private physical-force tensors used here; the broadcast 16x16 compatibility
        grid is deliberately bypassed.
        """
        features = []
        expected_counts = tuple(int(value) for value in self.cfg.tactile_graph_sensor_counts)
        for finger_idx, sensor in enumerate(self._tactile_sensor):
            # Accessing data refreshes the TacSL sensor and its physical aggregates.
            data = sensor.data
            normal = getattr(sensor, "_physical_tactile_normal_force", None)
            shear = getattr(sensor, "_physical_tactile_shear_force", None)
            if normal is None or shear is None:
                raise RuntimeError(
                    "estimated_official requires per-physical-sensor force aggregates; "
                    "the TacSL sensor did not expose them after refresh"
                )
            expected = expected_counts[finger_idx]
            if normal.shape != (self.num_envs, expected) or shear.shape != (
                self.num_envs,
                expected,
                2,
            ):
                raise RuntimeError(
                    f"Unexpected physical tactile shape for finger {finger_idx}: "
                    f"normal={tuple(normal.shape)}, shear={tuple(shear.shape)}, "
                    f"expected ({self.num_envs},{expected}) and ({self.num_envs},{expected},2)"
                )
            features.append(torch.cat([normal.unsqueeze(-1), shear], dim=-1))

        tactile = torch.cat(features, dim=1) * float(self.cfg.tactile_force_scale)
        tactile = torch.clamp(
            tactile,
            -float(self.cfg.tactile_force_clip),
            float(self.cfg.tactile_force_clip),
        )
        return tactile.reshape(self.num_envs, -1)

    def _compute_graph_tactile_frames(
        self,
        force_clean: torch.Tensor,
        force_teacher: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build physical-node teacher and student frames for the static GNN.

        The public channels are ``[b, on, off, duration, eta]``. The teacher
        additionally receives ``[Fn, Ft1, Ft2, delta_Fn, delta_|Ft|]``. A
        per-finger ``[shift_u, shift_v, shift_valid, contact_ratio]`` context is
        appended after all nodes in both frames.
        """
        total_nodes = int(self.cfg.tactile_graph_total_nodes)
        force_c = force_clean.view(self.num_envs, total_nodes, 3)
        force_t = force_teacher.view(self.num_envs, total_nodes, 3)
        force_magnitude = torch.linalg.vector_norm(force_c, dim=-1)

        previous_hysteresis = self.student_hysteresis_contact_b
        contact_on = force_magnitude > float(self.cfg.student_tactile_contact_threshold)
        contact_off = force_magnitude < float(self.cfg.student_tactile_contact_off_threshold)
        contact_clean = torch.where(
            contact_on,
            torch.ones_like(previous_hysteresis),
            torch.where(contact_off, torch.zeros_like(previous_hysteresis), previous_hysteresis),
        )
        self.student_hysteresis_contact_b.copy_(contact_clean)

        contact_b = contact_clean
        if self.cfg.enable_visible_contact_noise and self.cfg.student_tactile_flip_prob > 0.0:
            flip_mask = torch.rand_like(contact_b) < float(self.cfg.student_tactile_flip_prob)
            contact_b = torch.where(flip_mask, 1.0 - contact_b, contact_b)

        previous_contact = self.student_prev_contact_b
        contact_established = ((previous_contact < 0.5) & (contact_b > 0.5)).to(contact_b.dtype)
        contact_released = ((previous_contact > 0.5) & (contact_b < 0.5)).to(contact_b.dtype)
        self.student_contact_duration.copy_(
            torch.where(
                contact_b > 0.5,
                self.student_contact_duration + 1.0,
                torch.zeros_like(self.student_contact_duration),
            )
        )
        duration_tau = float(self.cfg.student_tactile_duration_tau)
        duration_max = float(self.cfg.student_tactile_duration_max)
        duration_norm = torch.log1p(self.student_contact_duration / duration_tau)
        duration_norm = duration_norm / math.log1p(duration_max / duration_tau)
        duration_norm = duration_norm.clamp(0.0, 1.0)

        shift_contexts = []
        eta_chunks = []
        start = 0
        beta = float(self.cfg.tactile_shift_ema_beta)
        shift_max = float(self.cfg.tactile_shift_max)
        for finger_idx, (count, positions) in enumerate(
            zip(self.cfg.tactile_graph_sensor_counts, self._graph_finger_positions)
        ):
            count = int(count)
            finger_contact = contact_b[:, start : start + count]
            contact_count = finger_contact.sum(dim=-1, keepdim=True)
            contact_valid = contact_count.squeeze(-1) > 0.0
            centroid = torch.sum(
                finger_contact.unsqueeze(-1) * positions.unsqueeze(0), dim=1
            ) / contact_count.clamp_min(1.0)
            centroid = torch.where(contact_valid.unsqueeze(-1), centroid, torch.zeros_like(centroid))
            shift_valid = contact_valid & self.graph_prev_contact_valid[:, finger_idx]
            normalized_shift = (
                (centroid - self.graph_prev_centroid[:, finger_idx]) / shift_max
            ).clamp(-1.0, 1.0)
            normalized_shift = torch.where(
                shift_valid.unsqueeze(-1), normalized_shift, torch.zeros_like(normalized_shift)
            )
            shift_ema = beta * self.graph_shift_ema[:, finger_idx] + (1.0 - beta) * normalized_shift
            shift_ema = torch.where(
                shift_valid.unsqueeze(-1), shift_ema, torch.zeros_like(shift_ema)
            )
            self.graph_shift_ema[:, finger_idx].copy_(shift_ema)
            self.graph_prev_centroid[:, finger_idx].copy_(centroid)
            self.graph_prev_contact_valid[:, finger_idx].copy_(contact_valid)

            eta = torch.sum(
                (positions.unsqueeze(0) - centroid.unsqueeze(1)) * shift_ema.unsqueeze(1),
                dim=-1,
            ).clamp(-1.0, 1.0)
            eta = torch.where(shift_valid.unsqueeze(-1), eta, torch.zeros_like(eta))
            eta_chunks.append(eta)
            shift_contexts.append(
                torch.cat(
                    [
                        shift_ema,
                        shift_valid.to(contact_b.dtype).unsqueeze(-1),
                        contact_count / float(count),
                    ],
                    dim=-1,
                )
            )
            start += count

        eta = torch.cat(eta_chunks, dim=1)
        shift_context = torch.stack(shift_contexts, dim=1).reshape(self.num_envs, -1)
        common_nodes = torch.stack(
            [contact_b, contact_established, contact_released, duration_norm, eta], dim=-1
        )

        # Keep the scaled/clipped three-dimensional TacSL force unchanged.  In
        # particular, do not log-compress either the non-negative normal force
        # or the signed shear components before exposing them to the teacher.
        raw_force = force_t.clone()
        raw_force[..., 0].clamp_min_(0.0)
        previous_force = self.prev_graph_force
        delta_normal = (raw_force[..., 0] - previous_force[..., 0]).clamp(-1.0, 1.0)
        shear_magnitude = torch.linalg.vector_norm(raw_force[..., 1:3], dim=-1)
        previous_shear_magnitude = torch.linalg.vector_norm(previous_force[..., 1:3], dim=-1)
        delta_shear_magnitude = (shear_magnitude - previous_shear_magnitude).clamp(-1.0, 1.0)
        force_nodes = torch.cat(
            [
                raw_force,
                delta_normal.unsqueeze(-1),
                delta_shear_magnitude.unsqueeze(-1),
            ],
            dim=-1,
        )

        self.student_prev_contact_b.copy_(contact_b)
        self.prev_graph_force.copy_(raw_force)
        student_frame = torch.cat(
            [common_nodes.reshape(self.num_envs, -1), shift_context], dim=-1
        )
        teacher_nodes = torch.cat([common_nodes, force_nodes], dim=-1)
        teacher_frame = torch.cat(
            [teacher_nodes.reshape(self.num_envs, -1), shift_context], dim=-1
        )
        if student_frame.shape[-1] != int(self.cfg.student_tactile_frame_dim):
            raise RuntimeError(
                f"Graph student frame width {student_frame.shape[-1]} != configured "
                f"{self.cfg.student_tactile_frame_dim}"
            )
        if teacher_frame.shape[-1] != int(self.cfg.teacher_tactile_frame_dim):
            raise RuntimeError(
                f"Graph teacher frame width {teacher_frame.shape[-1]} != configured "
                f"{self.cfg.teacher_tactile_frame_dim}"
            )
        return teacher_frame, student_frame

    def _compute_teacher_student_tactile_frames(
        self,
        force_clean: torch.Tensor,
        force_teacher: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build teacher ``[F, d]`` and student ``[b, Δb, d]`` frames.

        Contact duration ``d_norm`` is shared: updated from clean-force contact so
        Stage1 teacher duration is not affected by student bit-flip noise. Student
        may still flip ``b`` (and thus ``Δb``) under ``enable_visible_contact_noise``.

        Args:
            force_clean: Pooled TacSL forces ``(num_envs, tactile_force_dim)``.
            force_teacher: Teacher forces (optionally magnitude/direction noise).

        Returns:
            ``(teacher_frame, student_frame)`` with shapes
            ``(num_envs, teacher_tactile_frame_dim)`` and
            ``(num_envs, student_tactile_frame_dim)``.
        """
        n_taxels = int(self.cfg.student_tactile_num_taxels)
        force_c = force_clean.view(self.num_envs, n_taxels, 3)
        force_t = force_teacher.view(self.num_envs, n_taxels, 3)
        contact_gt = (torch.linalg.norm(force_c, dim=-1) > self.cfg.student_tactile_contact_threshold).float()

        self.student_contact_duration = torch.where(
            contact_gt > 0.5,
            self.student_contact_duration + 1.0,
            torch.zeros_like(self.student_contact_duration),
        )
        duration_norm = (
            self.student_contact_duration / float(self.cfg.student_tactile_duration_tau)
        ).clamp(0.0, 1.0)

        teacher_frame = torch.cat([force_t, duration_norm.unsqueeze(-1)], dim=-1).reshape(
            self.num_envs, -1
        )

        contact_b = contact_gt
        if self.cfg.enable_visible_contact_noise and self.cfg.student_tactile_flip_prob > 0.0:
            flip_mask = torch.rand_like(contact_b) < float(self.cfg.student_tactile_flip_prob)
            contact_b = torch.where(flip_mask, 1.0 - contact_b, contact_b)
        delta_b = contact_b - self.student_prev_contact_b
        self.student_prev_contact_b = contact_b
        student_frame = torch.stack([contact_b, delta_b, duration_norm], dim=-1).reshape(
            self.num_envs, -1
        )
        return teacher_frame, student_frame

    def _get_observations(self) -> dict:
        obs_dict = super()._get_observations()
        # Stage2 student inputs (real-robot sensing only): the first 42 dims of each
        # lag-history frame are joint_pos + current targets — no contact forces
        obs_dict["student_proprio_hist"] = self.obs_buf_lag_history[
            :, -self.cfg.student_proprio_history_len:, : self.cfg.student_proprio_frame_dim].clone()
        obs_dict["student_tactile_hist"] = self.student_tactile_hist_buf.clone()
        # Teacher temporal TF input: [Fx, Fy, Fz, d_norm] history (same T as student).
        obs_dict["tactile_hist"] = self.teacher_tactile_hist_buf.clone()
        return obs_dict

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = self.hand._ALL_INDICES
        super()._reset_idx(env_ids)
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
        self.contact_duty_ema[env_ids] = 0.0
        if self._coord_enabled:
            self.coord_omega_hist[env_ids] = 0.0
            self.coord_load_hist[env_ids] = 0.0
            self.coord_load_hist_count[env_ids] = 0.0
            self.coord_b_prev[env_ids] = 0.0
            self.coord_release_age[env_ids] = float(self.cfg.coord_delta_h + 1)

    def _get_rewards(self) -> torch.Tensor:
        total_reward = super()._get_rewards()
        self._update_contact_buffer_diagnostics()
        base_task_reward = total_reward.detach()
        contribution_diagnostics = None

        if self.cfg.enable_visible_contact_reward or self._coord_enabled:
            contribution_diagnostics = self.get_finger_coordination_diagnostics()
            active_finger_names = tuple(self.cfg.tactile_active_finger_names)
            for finger_idx, finger_name in enumerate(active_finger_names):
                self.extras[f"tactile/{finger_name}_signed_axis_torque"] = (
                    contribution_diagnostics["signed_axis_torque"][:, finger_idx].mean()
                )
                self.extras[f"tactile/{finger_name}_positive_torque_contribution"] = (
                    contribution_diagnostics["positive_torque_contribution"][:, finger_idx].mean()
                )
                self.extras[f"tactile/{finger_name}_negative_torque_contribution"] = (
                    contribution_diagnostics["negative_torque_contribution"][:, finger_idx].mean()
                )

        if self.cfg.enable_visible_contact_reward:
            tactile_state = self._compute_tactile_contact_state()
            visible_contact_progress = self._compute_visible_contact_reward(
                tactile_state,
                contribution_diagnostics["positive_torque_contribution"],
            )
            visible_contact_reward = self._apply_visible_contact_adaptive_scale(
                visible_contact_progress,
                base_task_reward,
            )
            total_reward = total_reward + visible_contact_reward
            self.extras["visible_contact_reward"] = visible_contact_reward.mean()

        if self.cfg.multi_contact_reward_scale != 0.0:
            total_reward = total_reward + self._compute_multi_contact_reward()

        if self._coord_enabled:
            coord_task_reward = total_reward.detach()
            coord_reward = self._compute_coord_endogenous_reward(
                coord_task_reward,
                contribution_diagnostics,
            )
            total_reward = total_reward + coord_reward
            self.extras["tactile/coord_reward"] = coord_reward.mean()
            self.extras["tactile/task_reward_per_env"] = coord_task_reward
            self.extras["tactile/coord_reward_per_env"] = coord_reward.detach()

        if (
            self.cfg.enable_visible_contact_reward
            or self.cfg.multi_contact_reward_scale != 0.0
            or self._coord_enabled
        ):
            self.extras["total_reward"] = total_reward.mean()
        if self.cfg.enable_visible_contact_reward:
            self.extras["visible_contact_reward_ratio"] = (
                visible_contact_reward.mean()
                / total_reward.mean().abs().clamp_min(1.0e-6)
            )
        return total_reward

    def _update_contact_buffer_diagnostics(self) -> None:
        """Record current and historical detailed-contact buffer utilization.

        IsaacLab reuses the PhysX contact-count tensor for detailed queries. Since
        friction is queried after contact points, this samples the successful
        friction query and reports the fullest per-DIP view without a CPU sync.
        """
        utilizations = []
        capacities = []
        for sensor in self._contact_sensor:
            view = sensor.contact_physx_view
            count_buffer = getattr(view, "_contact_count_buffer", None)
            capacity = int(view.max_contact_data_count)
            if count_buffer is None or capacity < 1:
                continue
            used = count_buffer.to(device=self.device, dtype=torch.float).sum()
            utilizations.append(used / float(capacity))
            capacities.append(capacity)

        if not utilizations:
            return
        current = torch.stack(utilizations).max()
        self._contact_buffer_peak_utilization = torch.maximum(
            self._contact_buffer_peak_utilization,
            current,
        )
        self.extras["tactile/contact_buffer_utilization"] = current
        self.extras["tactile/contact_buffer_peak_utilization"] = (
            self._contact_buffer_peak_utilization.detach()
        )
        self.extras["tactile/contact_buffer_capacity_per_dip"] = torch.tensor(
            max(capacities),
            device=self.device,
            dtype=torch.float,
        )

    def _finger_weight_tensor(self, values: Sequence[float]) -> torch.Tensor:
        """Build a per-finger weight vector padded/truncated to ``num_fingertips``."""
        num_fingers = len(self.cfg.tactile_tip_body_names)
        weights = torch.tensor(values, dtype=torch.float, device=self.device)
        if weights.numel() < num_fingers:
            pad = torch.zeros((num_fingers - weights.numel(),), dtype=torch.float, device=self.device)
            weights = torch.cat([weights, pad], dim=0)
        elif weights.numel() > num_fingers:
            weights = weights[:num_fingers]
        return weights

    def _select_active_finger_slots(self, values: torch.Tensor, dim: int = 1) -> torch.Tensor:
        """Select task-active tactile fingers from a tensor that may still contain all five fingers."""
        expected = len(self.cfg.tactile_tip_body_names)
        if values.shape[dim] == expected:
            return values
        indices = torch.tensor(
            tuple(getattr(self.cfg, "tactile_active_finger_indices", range(values.shape[dim]))),
            dtype=torch.long,
            device=values.device,
        )
        if indices.numel() != expected or int(indices.max().item()) >= values.shape[dim]:
            raise RuntimeError(
                f"Cannot align finger tensor shape {tuple(values.shape)} with active tactile fingers "
                f"{tuple(getattr(self.cfg, 'tactile_active_finger_names', ()))}."
            )
        return values.index_select(dim=dim, index=indices)

    def _get_tactile_force_magnitude(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-physical-sensor force magnitudes and validity mask before pooling.

        Returns:
            Magnitudes of shape ``(num_envs, num_fingers, max_sensors)`` and a
            boolean validity mask of shape ``(num_fingers, max_sensors)``.
        """
        magnitudes = []
        for sensor in self._tactile_sensor:
            data = sensor.data
            normal = getattr(sensor, "_physical_tactile_normal_force", data.tactile_normal_force)
            shear = getattr(sensor, "_physical_tactile_shear_force", data.tactile_shear_force)
            taxels = torch.cat([normal.unsqueeze(-1), shear], dim=-1)
            magnitudes.append(taxels.norm(dim=-1).view(self.num_envs, -1))
        if not magnitudes:
            empty = torch.zeros(
                (self.num_envs, len(self.cfg.tactile_tip_body_names), 0),
                device=self.device,
            )
            return empty, torch.zeros(
                (len(self.cfg.tactile_tip_body_names), 0),
                dtype=torch.bool,
                device=self.device,
            )
        max_sensors = max(force.shape[-1] for force in magnitudes)
        tactile = torch.zeros(
            (self.num_envs, len(magnitudes), max_sensors),
            dtype=magnitudes[0].dtype,
            device=self.device,
        )
        valid = torch.zeros(
            (len(magnitudes), max_sensors), dtype=torch.bool, device=self.device
        )
        for finger_idx, force in enumerate(magnitudes):
            tactile[:, finger_idx, : force.shape[-1]] = force
            valid[finger_idx, : force.shape[-1]] = True
        num_fingers = len(self.cfg.tactile_tip_body_names)
        if tactile.shape[1] < num_fingers:
            pad = torch.zeros(
                (self.num_envs, num_fingers - tactile.shape[1], tactile.shape[-1]),
                dtype=tactile.dtype,
                device=tactile.device,
            )
            tactile = torch.cat([tactile, pad], dim=1)
            valid = torch.cat(
                [
                    valid,
                    torch.zeros(
                        (num_fingers - valid.shape[0], max_sensors),
                        dtype=torch.bool,
                        device=self.device,
                    ),
                ],
                dim=0,
            )
        elif tactile.shape[1] > num_fingers:
            tactile = tactile[:, :num_fingers]
            valid = valid[:num_fingers]
        return tactile, valid

    def _get_dip_physical_contact(self) -> torch.Tensor:
        """Per-finger object-filtered ContactSensor contact on elastomer/DIP bodies."""
        contact_force = torch.norm(self._contact_forces_w(), dim=-1)
        contact_force = self._select_active_finger_slots(contact_force, dim=1)
        return contact_force > float(self.cfg.visible_contact_force_min)

    def _contact_force_and_center_w(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reconstruct per-finger 3D force and contact center from TacSL taxels.

        TacSL reports normal and shear components in each taxel's local frame.
        They are rotated into world coordinates and summed per finger. This avoids
        PhysX's unstable detailed-friction query while retaining tangential force.
        The filtered ContactSensor force and DIP origin are used as a fallback.

        Returns:
            Force on each finger in world coordinates, contact centers in
            env-relative world coordinates, and a valid-center mask.
        """
        eps = 1.0e-8
        origins = self.scene.env_origins
        forces = []
        centers = []
        valid_centers = []
        tactile_sensor_count = 0

        active_indices = tuple(
            getattr(self.cfg, "tactile_active_finger_indices", range(len(self._tactile_sensor)))
        )
        for finger_idx, tactile_sensor in enumerate(self._tactile_sensor):
            body_id = self.tactile_tip_body_ids[finger_idx]
            fallback_center = self.hand.data.body_pos_w[:, body_id] - origins
            tactile_data = tactile_sensor.data
            normal = getattr(tactile_sensor, "_physical_tactile_normal_force", None)
            shear = getattr(tactile_sensor, "_physical_tactile_shear_force", None)
            local_points = getattr(tactile_sensor, "_tactile_physical_center_pos_local", None)

            if normal is None or shear is None or local_points is None:
                normal = getattr(tactile_data, "tactile_normal_force", None)
                shear = getattr(tactile_data, "tactile_shear_force", None)
                positions_w = getattr(tactile_data, "tactile_points_pos_w", None)
            else:
                body_pos_w = self.hand.data.body_pos_w[:, body_id]
                body_quat_w = self.hand.data.body_quat_w[:, body_id]
                point_count = local_points.shape[0]
                positions_w = quat_apply(
                    body_quat_w.unsqueeze(1).expand(-1, point_count, -1),
                    local_points.unsqueeze(0).expand(self.num_envs, -1, -1),
                ) + body_pos_w.unsqueeze(1)

            quaternions_w = getattr(tactile_data, "tactile_points_quat_w", None)
            if normal is None or shear is None or positions_w is None or quaternions_w is None:
                contact_idx = active_indices[finger_idx]
                contact_data = self._contact_sensor[contact_idx].data
                force_by_filter = getattr(contact_data, "force_matrix_w", None)
                if force_by_filter is None:
                    force = contact_data.net_forces_w[:, 0, :]
                else:
                    force = torch.nan_to_num(force_by_filter[:, 0, :, :]).sum(dim=1)
                forces.append(force)
                centers.append(fallback_center)
                valid_centers.append(force.norm(dim=-1) > eps)
                continue

            point_count = normal.shape[1]
            local_force = torch.cat([shear, normal.unsqueeze(-1)], dim=-1)
            force_by_taxel = quat_apply(
                quaternions_w[:, :point_count, :],
                local_force,
            )
            force_by_taxel = torch.nan_to_num(force_by_taxel)
            positions = positions_w[:, :point_count, :] - origins.unsqueeze(1)
            finite_position = torch.isfinite(positions).all(dim=-1)
            force_weight = force_by_taxel.norm(dim=-1) * finite_position.float()
            weight_sum = force_weight.sum(dim=-1, keepdim=True)
            weighted_center = (
                torch.nan_to_num(positions) * force_weight.unsqueeze(-1)
            ).sum(dim=1) / weight_sum.clamp_min(eps)
            center_valid = weight_sum.squeeze(-1) > eps
            center = torch.where(
                center_valid.unsqueeze(-1), weighted_center, fallback_center
            )
            forces.append(force_by_taxel.sum(dim=1))
            centers.append(center)
            valid_centers.append(center_valid)
            tactile_sensor_count += 1

        self.extras["tactile/contact_force_from_tacsl"] = torch.tensor(
            float(tactile_sensor_count == len(self._tactile_sensor)),
            device=self.device,
        )
        return (
            torch.stack(forces, dim=1),
            torch.stack(centers, dim=1),
            torch.stack(valid_centers, dim=1),
        )

    def _compute_finger_axis_contribution(
        self,
        force_on_finger_w: torch.Tensor,
        contact_center_w: torch.Tensor,
        object_center_w: torch.Tensor,
        object_axis_w: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert per-finger contact wrenches into unified task contributions.

        Args:
            force_on_finger_w: Complete contact force acting on each finger.
            contact_center_w: Per-finger contact center in env-relative world coordinates.
            object_center_w: Rotating-object center in env-relative world coordinates.
            object_axis_w: Unit rotation axis in world coordinates.

        Returns:
            Signed axis torque, normalized positive and negative contributions,
            and physical-contact confidence.
        """
        force_on_object_w = (
            float(self.cfg.finger_contact_force_on_object_sign) * force_on_finger_w
        )
        lever_arm = contact_center_w - object_center_w.unsqueeze(1)
        torque_w = torch.cross(lever_arm, force_on_object_w, dim=-1)
        signed_axis_torque = (
            float(self.cfg.coord_rot_dir)
            * torch.sum(torque_w * object_axis_w.unsqueeze(1), dim=-1)
        )
        torque_ref = max(float(self.cfg.finger_contribution_torque_ref), 1.0e-8)
        normalized_torque = signed_axis_torque / torque_ref
        positive_contribution = normalized_torque.clamp(0.0, 1.0)
        negative_contribution = (-normalized_torque).clamp(0.0, 1.0)
        force_ref = max(float(self.cfg.finger_physical_contact_force_ref), 1.0e-8)
        physical_contact_confidence = torch.tanh(force_on_finger_w.norm(dim=-1) / force_ref)
        return (
            signed_axis_torque,
            positive_contribution,
            negative_contribution,
            physical_contact_confidence,
        )

    def _compute_tactile_contact_state(self) -> dict[str, torch.Tensor]:
        """Compare TacSL taxel activation against ContactSensor physical contact."""
        tactile_magnitude, valid = self._get_tactile_force_magnitude()
        num_fingers = len(self.cfg.tactile_tip_body_names)
        if tactile_magnitude.shape[-1] == 0:
            empty_bool = torch.zeros((self.num_envs, num_fingers), dtype=torch.bool, device=self.device)
            zero = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
            self.extras["tactile/visible_contact_ratio"] = zero.mean()
            return {
                "dip_physical_contact": empty_bool,
                "per_finger_ratio": torch.zeros((self.num_envs, num_fingers), device=self.device),
            }

        active = (
            tactile_magnitude > float(self.cfg.tactile_visible_contact_threshold)
        ) & valid.unsqueeze(0)
        active_count = active.float().sum(dim=(1, 2))
        total_taxels = valid.sum().clamp_min(1).float()
        visible_ratio = active_count / total_taxels
        dip_physical_contact = self._get_dip_physical_contact()
        per_finger_ratio = active.float().sum(dim=-1) / valid.sum(dim=-1).clamp_min(1).float()
        self.extras["tactile/visible_contact_ratio"] = visible_ratio.mean()
        self.extras["tactile/dip_physical_contact_ratio"] = dip_physical_contact.float().mean()
        return {
            "dip_physical_contact": dip_physical_contact,
            "per_finger_ratio": per_finger_ratio,
        }

    def _compute_visible_contact_reward(
        self,
        tactile_contact_state: dict[str, torch.Tensor],
        positive_torque_contribution: torch.Tensor,
    ) -> torch.Tensor:
        """Reward visible contacts only when their finger contributes positive torque.

        The reward only considers fingers whose object-filtered ContactSensor is
        active. For each such finger, the active-taxel ratio is capped at
        ``tactile_visible_contact_ratio_cap``. A square-root positive-torque gate
        keeps useful contacts while rejecting zero/reverse-torque contacts.
        Finger weights are normalized so progress stays in ``[0, 1]`` for every
        task, independent of active finger count. Adaptive scaling is applied by
        :meth:`_apply_visible_contact_adaptive_scale`.

        Args:
            tactile_contact_state: Physical-contact and visible-taxel state.
            positive_torque_contribution: Normalized positive task contribution per finger.

        Returns:
            Per-environment task-aligned visible-contact progress in ``[0, 1]``.
        """
        physical_contact = tactile_contact_state["dip_physical_contact"].float()
        per_finger_ratio = tactile_contact_state["per_finger_ratio"]
        visible_weights = self._finger_weight_tensor(self.cfg.tactile_visible_contact_finger_weights)
        visible_weight_sum = visible_weights.sum().clamp_min(1.0e-6)
        ratio_cap = max(float(self.cfg.tactile_visible_contact_ratio_cap), 1.0e-6)
        visible_progress = torch.clamp(per_finger_ratio, min=0.0, max=ratio_cap) / ratio_cap
        contribution_gate = positive_torque_contribution.clamp(0.0, 1.0).pow(
            float(self.cfg.visible_contact_contribution_power)
        )
        visible_reward_progress = (
            visible_progress * physical_contact * visible_weights.unsqueeze(0)
        ).sum(dim=-1) / visible_weight_sum
        task_aligned_progress = (
            visible_progress
            * physical_contact
            * contribution_gate
            * visible_weights.unsqueeze(0)
        ).sum(dim=-1) / visible_weight_sum
        target_contact_count = physical_contact.mul(visible_weights.unsqueeze(0)).sum(dim=-1)
        contact_weight_sum = physical_contact.mul(visible_weights.unsqueeze(0)).sum(dim=-1).clamp_min(1.0)
        self.extras["tactile/target_finger_active_ratio"] = (
            (per_finger_ratio * physical_contact * visible_weights.unsqueeze(0)).sum(dim=-1)
            / contact_weight_sum
        ).mean()
        self.extras["tactile/target_finger_contact_count"] = target_contact_count.mean()
        self.extras["tactile/target_finger_visible_progress"] = visible_reward_progress.mean()
        self.extras["tactile/task_aligned_visible_progress"] = task_aligned_progress.mean()
        self.extras["tactile/visible_contribution_gate"] = contribution_gate.mean()
        self.extras["tactile/visible_reward_retention"] = (
            task_aligned_progress.sum()
            / visible_reward_progress.sum().clamp_min(1.0e-6)
        )
        return task_aligned_progress

    def _current_visible_contact_target_ratio(self) -> float:
        """Return the visible-contact target share at the curriculum step.

        Returns:
            The target mean absolute ratio against the base task reward.
        """
        initial_ratio = float(self.cfg.visible_contact_target_ratio_initial)
        mid_ratio = float(self.cfg.visible_contact_target_ratio_mid)
        final_ratio = float(self.cfg.visible_contact_target_ratio_final)
        progress = self._domain_randomization_curriculum_progress()
        warmup_progress = float(
            self.cfg.visible_contact_target_ratio_warmup_progress
        )
        mid_progress = float(self.cfg.visible_contact_target_ratio_mid_progress)
        if progress <= warmup_progress:
            return initial_ratio
        if progress <= mid_progress:
            phase = (progress - warmup_progress) / max(
                mid_progress - warmup_progress, 1.0e-6
            )
            return initial_ratio + (mid_ratio - initial_ratio) * phase
        phase = (progress - mid_progress) / max(1.0 - mid_progress, 1.0e-6)
        return mid_ratio + (final_ratio - mid_ratio) * min(phase, 1.0)

    def _apply_visible_contact_adaptive_scale(
        self,
        progress: torch.Tensor,
        task_reward: torch.Tensor,
    ) -> torch.Tensor:
        """Scale sparse useful contacts toward a bounded task-reward share.

        Args:
            progress: Task-aligned visible-contact progress in ``[0, 1]``.
            task_reward: Base reward before visible-contact shaping.

        Returns:
            Adaptively scaled and per-environment clipped visible-contact reward.
        """
        eps = 1.0e-6
        task_abs = task_reward.detach().abs().mean()
        progress_abs = progress.detach().abs().mean()
        ema = float(self.cfg.visible_contact_adaptive_ema)
        self._visible_adaptive_updates += 1
        if self._visible_adaptive_updates == 1:
            self.visible_task_reward_abs_ema.copy_(task_abs)
            self.visible_progress_abs_ema.copy_(progress_abs)
        else:
            self.visible_task_reward_abs_ema.mul_(ema).add_(task_abs * (1.0 - ema))
            self.visible_progress_abs_ema.mul_(ema).add_(progress_abs * (1.0 - ema))

        target_ratio = self._current_visible_contact_target_ratio()
        scale = (
            target_ratio
            * self.visible_task_reward_abs_ema
            / self.visible_progress_abs_ema.clamp_min(eps)
        )
        scale = torch.clamp(
            scale,
            min=float(self.cfg.visible_contact_adaptive_scale_min),
            max=float(self.cfg.visible_contact_adaptive_scale_max),
        )
        self.visible_contact_adaptive_scale.copy_(scale)

        reward_unclipped = progress * scale
        dynamic_clip = torch.maximum(
            task_reward.detach().abs()
            * float(self.cfg.visible_contact_reward_clip_ratio),
            torch.full_like(
                progress,
                float(self.cfg.visible_contact_reward_dynamic_clip_min),
            ),
        )
        reward = torch.minimum(reward_unclipped, dynamic_clip)
        self.extras["visible_contact_reward_scale"] = scale.detach()
        self.extras["visible_contact_target_ratio"] = torch.tensor(
            target_ratio, device=self.device
        )
        self.extras["visible_contact_reward_abs_ratio"] = (
            reward.detach().abs().mean() / task_abs.clamp_min(eps)
        )
        self.extras["visible_contact_reward_clip_ratio"] = (
            reward_unclipped > dynamic_clip
        ).float().mean()
        self.extras["visible_contact_task_reward_abs_ema"] = (
            self.visible_task_reward_abs_ema.detach()
        )
        self.extras["visible_contact_progress_abs_ema"] = (
            self.visible_progress_abs_ema.detach()
        )
        return reward

    def _current_coord_q_floor(self) -> float:
        """Return the fixed-schedule early coordination progress floor."""
        start = int(self.cfg.coord_guide_curriculum_start)
        end = int(self.cfg.coord_guide_curriculum_end)
        if end <= start:
            progress = 1.0
        else:
            agent_steps = int(self.common_step_counter) * int(self.num_envs)
            progress = (agent_steps - start) / float(end - start)
            progress = min(max(progress, 0.0), 1.0)
        initial = float(self.cfg.coord_q_floor_initial)
        final = float(self.cfg.coord_q_floor_final)
        return initial + (final - initial) * progress

    def _current_coord_presence_floor(self) -> float:
        """Return the early floor that linearizes useful-torque utility."""
        start = int(self.cfg.coord_guide_curriculum_start)
        end = int(self.cfg.coord_guide_curriculum_end)
        if end <= start:
            progress = 1.0
        else:
            agent_steps = int(self.common_step_counter) * int(self.num_envs)
            progress = (agent_steps - start) / float(end - start)
            progress = min(max(progress, 0.0), 1.0)
        initial = float(self.cfg.coord_presence_floor_initial)
        final = float(self.cfg.coord_presence_floor_final)
        return initial + (final - initial) * progress

    def _coord_presence_gate(
        self,
        normalized_total_torque: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        """Blend a smooth useful-torque gate with its early curriculum floor."""
        half_saturation = float(self.cfg.coord_presence_half_saturation)
        positive_torque = normalized_total_torque.clamp_min(0.0)
        raw_gate = positive_torque / (
            positive_torque + half_saturation
        )
        floor = self._current_coord_presence_floor()
        gate = floor + (1.0 - floor) * raw_gate
        return raw_gate, gate, floor

    def _coord_effective_torque_reward_weight(self) -> float:
        """Return the fixed physical coefficient for useful axis torque."""
        return float(self.cfg.coord_effective_torque_reward_weight)

    def _coord_effective_torque_guide(
        self,
        efficient_torque: torch.Tensor,
        nominal_capacity_sum: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, float, torch.Tensor]:
        """Return the bounded physical guide and its weighted score contribution."""
        normalized_torque = (
            efficient_torque.sum(dim=-1)
            / nominal_capacity_sum.clamp_min(1.0e-6)
        )
        reference = float(self.cfg.coord_effective_torque_guide_ref)
        guide = (normalized_torque / reference).clamp(0.0, 1.0).pow(
            float(self.cfg.coord_effective_torque_guide_power)
        )
        weight = self._coord_effective_torque_reward_weight()
        return normalized_torque, guide, weight, weight * guide

    def _current_coord_intrinsic_weight(self) -> float:
        """Return the fixed curriculum weight for the intrinsic utility."""
        agent_steps = int(self.common_step_counter) * int(self.num_envs)
        warmup_end = int(self.cfg.coord_intrinsic_warmup_end)
        decay_start = int(self.cfg.coord_intrinsic_decay_start)
        decay_end = int(self.cfg.coord_intrinsic_decay_end)
        initial = float(self.cfg.coord_intrinsic_weight_initial)
        peak = float(self.cfg.coord_intrinsic_weight_peak)
        final = float(self.cfg.coord_intrinsic_weight_final)

        if agent_steps < warmup_end and warmup_end > 0:
            progress = agent_steps / float(warmup_end)
            return initial + (peak - initial) * progress
        if agent_steps < decay_start:
            return peak
        if agent_steps < decay_end and decay_end > decay_start:
            progress = (agent_steps - decay_start) / float(decay_end - decay_start)
            return peak + (final - peak) * progress
        return final

    def _combine_coord_intrinsic_score(
        self,
        effective_torque_bonus: torch.Tensor,
        quality_score: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        """Combine fixed useful torque with curriculum-weighted coordination quality."""
        quality_weight = self._current_coord_intrinsic_weight()
        score = (
            effective_torque_bonus + quality_weight * quality_score
        ).clamp(-1.0, 1.0)
        return score, quality_weight

    def _apply_coord_intrinsic_reward(
        self,
        intrinsic_score: torch.Tensor,
        task_reward: torch.Tensor,
        reward_weight: float | None = None,
        positive_reward_budget: float | None = None,
    ) -> torch.Tensor:
        """Apply a bounded intrinsic score with an optional compatibility weight.

        Args:
            intrinsic_score: Per-environment score in ``[-1, 1]``.
            task_reward: Non-coordination reward used only for diagnostics.

        Returns:
            Signed intrinsic reward per environment.
        """
        eps = 1.0e-6
        task_abs = task_reward.detach().abs().mean()
        weight = (
            self._current_coord_intrinsic_weight()
            if reward_weight is None
            else float(reward_weight)
        )
        bounded_score = intrinsic_score.clamp(-1.0, 1.0)
        reward = weight * bounded_score
        reward_budget = (
            abs(weight)
            if positive_reward_budget is None
            else max(float(positive_reward_budget), 0.0)
        )

        self.extras["tactile/coord_intrinsic_weight"] = torch.tensor(
            weight, device=self.device
        )
        self.extras["tactile/coord_positive_reward_budget"] = torch.tensor(
            reward_budget, device=self.device
        )
        self.extras["tactile/coord_reward_budget_abs_ratio"] = (
            torch.tensor(reward_budget, device=self.device)
            / task_abs.clamp_min(eps)
        )
        self.extras["tactile/coord_intrinsic_score"] = bounded_score.mean()
        self.extras["tactile/coord_nonzero_ratio"] = (
            reward.detach().abs() > eps
        ).float().mean()
        self.extras["tactile/coord_positive_ratio"] = (
            reward.detach() > eps
        ).float().mean()
        self.extras["tactile/coord_negative_ratio"] = (
            reward.detach() < -eps
        ).float().mean()
        self.extras["tactile/coord_reward_abs_ratio"] = (
            reward.detach().abs().mean() / task_abs.clamp_min(eps)
        )
        if "pose_diff_penalty" in self.extras and "torque_penalty" in self.extras:
            pose_penalty = self.extras["pose_diff_penalty"].detach().abs()
            torque_penalty = self.extras["torque_penalty"].detach().abs()
            penalty_reference = 0.5 * (
                pose_penalty * abs(float(self.cfg.pose_diff_penalty_scale))
                + torque_penalty * abs(float(self.cfg.torque_penalty_scale))
            )
            self.extras["tactile/coord_penalty_reference"] = penalty_reference
            self.extras["tactile/coord_weight_penalty_ratio"] = (
                reward_budget / penalty_reference.clamp_min(eps)
            )
        return reward

    def _capacity_weighted_load_utility(
        self,
        normalized_load: torch.Tensor,
        torque_capacity: torch.Tensor,
        nominal_capacity_sum: torch.Tensor,
    ) -> torch.Tensor:
        """Return a monotone concave utility for useful per-finger torque.

        The perspective form ``capacity * (1 - exp(-load / saturation))``
        favors sharing a fixed torque in proportion to each finger's current
        lever-arm capacity, without requiring equal absolute contributions.
        """
        saturation = max(float(self.cfg.coord_load_saturation), 1.0e-6)
        bounded_load = normalized_load.clamp(
            min=0.0,
            max=float(self.cfg.coord_load_max),
        )
        per_finger_utility = torque_capacity * (
            1.0 - torch.exp(-bounded_load / saturation)
        )
        return (
            per_finger_utility.sum(dim=-1)
            / nominal_capacity_sum.clamp_min(1.0e-6)
        ).clamp(0.0, 1.0)

    def _coord_contact_confidence(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-finger contact confidence ``b`` and raw taxel force magnitudes.

        Returns:
            ``b`` of shape ``(num_envs, num_fingers)`` in ``[0, 1]``, and
            magnitudes ``(num_envs, num_fingers, num_taxels)``.
        """
        magnitudes, valid = self._get_tactile_force_magnitude()
        active = (
            (magnitudes > float(self.cfg.coord_taxel_thr)) & valid.unsqueeze(0)
        ).float()
        n_active = active.sum(dim=-1)
        b = (n_active / float(self.cfg.coord_n_sat)).clamp(0.0, 1.0)
        return b, magnitudes

    def _coord_contact_centers_w(self, magnitudes: torch.Tensor) -> torch.Tensor:
        """Force-weighted taxel centroids in env-relative world frame.

        Args:
            magnitudes: Raw per-taxel force norms ``(num_envs, num_fingers, N)``.

        Returns:
            Contact centers ``(num_envs, num_fingers, 3)``. Falls back to
            fingertip link origins when a finger has near-zero force.
        """
        eps = 1.0e-6
        centers = []
        origins = self.scene.env_origins
        for finger_idx, sensor in enumerate(self._tactile_sensor):
            local_points = getattr(
                sensor,
                "_tactile_physical_center_pos_local",
                getattr(sensor, "_tactile_pos_local", None),
            )
            body_id = self.tactile_tip_body_ids[finger_idx]
            tip_pos = self.hand.data.body_pos_w[:, body_id] - origins
            if local_points is None or local_points.numel() == 0:
                centers.append(tip_pos)
                continue
            w = magnitudes[:, finger_idx, : local_points.shape[0]]
            w_sum = w.sum(dim=-1, keepdim=True).clamp(min=eps)
            c_local = (w.unsqueeze(-1) * local_points.unsqueeze(0)).sum(dim=1) / w_sum
            quat_w = self.hand.data.body_quat_w[:, body_id]
            c_w = quat_apply(quat_w, c_local) + tip_pos
            no_contact = w_sum.squeeze(-1) <= (eps * 10.0)
            c_w = torch.where(no_contact.unsqueeze(-1), tip_pos, c_w)
            centers.append(c_w)
        num_fingers = len(self.cfg.tactile_tip_body_names)
        active_fingertip_pos = self._select_active_finger_slots(self.fingertip_pos, dim=1)
        while len(centers) < num_fingers:
            centers.append(active_fingertip_pos[:, len(centers)])
        return torch.stack(centers[:num_fingers], dim=1)

    def get_finger_coordination_diagnostics(self) -> dict[str, torch.Tensor]:
        """Compute per-finger contact and effective task contribution signals.

        This is the read-only diagnostic form of the instantaneous terms used
        by the endogenous coordination reward. It does not update the handover,
        angular-velocity history, curriculum, or potential buffers.

        Returns:
            Per-environment contact confidence/state, signed axis torque,
            normalized task contribution, and intermediate kinematic terms.
        """
        eps = 1.0e-6
        num_fingers = len(self.cfg.tactile_tip_body_names)
        tactile_confidence, magnitudes = self._coord_contact_confidence()
        tactile_contact_center = self._coord_contact_centers_w(magnitudes)
        force_on_finger_w, physical_contact_center, physical_center_valid = (
            self._contact_force_and_center_w()
        )
        p_contact = torch.where(
            physical_center_valid.unsqueeze(-1),
            physical_contact_center,
            tactile_contact_center,
        )

        nut_body_pos = self.object.data.body_pos_w[:, self.nut_body_idx] - self.scene.env_origins
        nut_body_quat = self.object.data.body_quat_w[:, self.nut_body_idx]
        axis_local = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
        axis_local[:, 2] = 1.0
        axis_w = quat_apply(nut_body_quat, axis_local)
        axis_w = axis_w / axis_w.norm(dim=-1, keepdim=True).clamp(min=eps)

        (
            signed_axis_torque,
            positive_torque_contribution,
            negative_torque_contribution,
            physical_contact_confidence,
        ) = self._compute_finger_axis_contribution(
            force_on_finger_w,
            p_contact,
            nut_body_pos,
            axis_w,
        )
        contact_confidence = torch.maximum(
            tactile_confidence,
            physical_contact_confidence,
        )

        rho_ref = (
            float(self.cfg.coord_rho_ref_scale)
            * float(self.cfg.coord_obj_radius)
            * self.object_radius_scales.unsqueeze(1)
        )
        r_vec = p_contact - nut_body_pos.unsqueeze(1)
        rho = torch.linalg.norm(torch.cross(r_vec, axis_w.unsqueeze(1), dim=-1), dim=-1)
        w_geom = (rho / rho_ref.clamp(min=eps)).clamp(0.0, 1.0)
        force_on_object_w = (
            float(self.cfg.finger_contact_force_on_object_sign) * force_on_finger_w
        )
        torque_w = torch.cross(r_vec, force_on_object_w, dim=-1)
        torque_magnitude = torch.linalg.norm(torque_w, dim=-1)
        positive_axis_torque = signed_axis_torque.clamp_min(0.0)
        axis_torque_efficiency = (
            positive_axis_torque / torque_magnitude.clamp_min(eps)
        ).clamp(0.0, 1.0)

        r_hat = r_vec / r_vec.norm(dim=-1, keepdim=True).clamp(min=eps)
        d_rot = float(self.cfg.coord_rot_dir)
        t_target = d_rot * torch.cross(axis_w.unsqueeze(1).expand_as(r_hat), r_hat, dim=-1)

        tip_ids = self.tactile_tip_body_ids
        v_tip = self.hand.data.body_lin_vel_w[:, tip_ids]
        v_com = self.object.data.body_lin_vel_w[:, self.nut_body_idx]
        omega_vec = self.nut_dof_vel_cf.unsqueeze(-1) * axis_w
        v_obj = v_com.unsqueeze(1) + torch.cross(
            omega_vec.unsqueeze(1).expand_as(r_vec), r_vec, dim=-1
        )
        v_rel = v_tip - v_obj
        w_motion = (
            torch.sum(v_rel * t_target, dim=-1) / max(float(self.cfg.coord_v_ref), eps)
        ).clamp(0.0, 1.0)
        motion_floor = float(self.cfg.coord_motion_floor)
        if motion_floor > 0.0:
            w_motion = motion_floor + (1.0 - motion_floor) * w_motion

        finger_mask = getattr(self, "coord_finger_mask", None)
        if finger_mask is None:
            finger_mask = torch.ones(num_fingers, device=self.device, dtype=torch.float)
        finger_mask_b = finger_mask.unsqueeze(0)

        return {
            "contact_confidence": contact_confidence,
            "tactile_contact_confidence": tactile_confidence,
            "physical_contact_confidence": physical_contact_confidence,
            "contact_state": (
                contact_confidence >= float(self.cfg.coord_b_th)
            ) & (finger_mask_b > 0.5),
            "geometry_weight": w_geom,
            "motion_weight": w_motion,
            "signed_axis_torque": signed_axis_torque,
            "positive_axis_torque": positive_axis_torque,
            "torque_magnitude": torque_magnitude,
            "axis_torque_efficiency": axis_torque_efficiency,
            "lever_radius": rho,
            "force_magnitude": torch.linalg.norm(force_on_object_w, dim=-1),
            "positive_torque_contribution": positive_torque_contribution,
            "negative_torque_contribution": negative_torque_contribution,
            "effective_contribution": positive_torque_contribution * finger_mask_b,
            "contact_center": p_contact,
            "force_on_finger": force_on_finger_w,
            "nut_body_position": nut_body_pos,
            "radial_direction": r_hat,
            "relative_velocity": v_rel,
        }

    def _compute_coord_endogenous_reward(
        self,
        task_reward: torch.Tensor,
        diagnostics: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Compute a size-aware multi-finger coordination intrinsic reward.

        Useful axis torque is normalized by each contact's lever-arm capacity.
        A monotone concave utility favors comfortable load sharing without
        requiring every finger, while a short load history credits sequential
        finger gait. The fixed curriculum weight decays after mid training.

        Args:
            task_reward: Non-coordination task reward used for diagnostics.
            diagnostics: Optional shared per-finger contribution diagnostics.

        Returns:
            Per-env coordination reward ``(num_envs,)``.
        """
        eps = 1.0e-6
        self._coord_global_steps += 1
        num_fingers = len(self.cfg.tactile_tip_body_names)
        d_rot = float(self.cfg.coord_rot_dir)

        # Ensure kinematics used below are current (base reward already refreshed).
        if diagnostics is None:
            diagnostics = self.get_finger_coordination_diagnostics()
        b = diagnostics["contact_confidence"]
        positive_axis_torque = diagnostics["positive_axis_torque"]
        torque_magnitude = diagnostics["torque_magnitude"]
        axis_efficiency = diagnostics["axis_torque_efficiency"]
        lever_radius = diagnostics["lever_radius"]
        force_magnitude = diagnostics["force_magnitude"]
        nut_body_pos = diagnostics["nut_body_position"]
        r_hat = diagnostics["radial_direction"]
        v_rel = diagnostics["relative_velocity"]
        omega_scalar = self.nut_dof_vel_cf
        finger_mask = self.coord_finger_mask
        finger_mask_b = finger_mask.unsqueeze(0)
        active_fingers = float(self.coord_finger_mask_sum.item())

        nominal_radius = (
            float(self.cfg.coord_obj_radius)
            * self.object_radius_scales.unsqueeze(1)
        )
        comfort_force = float(self.cfg.coord_force_comfort_ref)
        nominal_capacity = nominal_radius * comfort_force * finger_mask_b
        nominal_capacity_sum = nominal_capacity.sum(dim=-1)
        lever_ratio = (
            lever_radius / nominal_radius.clamp_min(eps)
        ).clamp(
            min=float(self.cfg.coord_lever_ratio_min),
            max=float(self.cfg.coord_lever_ratio_max),
        )
        torque_capacity = nominal_capacity * lever_ratio
        efficient_torque = (
            positive_axis_torque
            * axis_efficiency.pow(float(self.cfg.coord_axis_efficiency_power))
            * finger_mask_b
        )
        (
            normalized_effective_torque,
            effective_torque_guide,
            effective_torque_guide_weight,
            effective_torque_guide_bonus,
        ) = self._coord_effective_torque_guide(
            efficient_torque,
            nominal_capacity_sum,
        )
        normalized_load = efficient_torque / torque_capacity.clamp_min(eps)
        instant_utility = self._capacity_weighted_load_utility(
            normalized_load,
            torque_capacity,
            nominal_capacity_sum,
        )
        history_load = normalized_load.clamp(0.0, float(self.cfg.coord_load_max))
        self.coord_load_hist[:] = torch.cat(
            [self.coord_load_hist[:, 1:], history_load.unsqueeze(1)], dim=1
        )
        self.coord_load_hist_count.add_(1.0).clamp_(
            max=float(self.cfg.coord_load_window)
        )
        window_load = (
            self.coord_load_hist.sum(dim=1)
            / self.coord_load_hist_count.unsqueeze(-1).clamp_min(1.0)
        )
        window_utility = self._capacity_weighted_load_utility(
            window_load,
            nominal_capacity,
            nominal_capacity_sum,
        )
        instant_mix = float(self.cfg.coord_instantaneous_mix)
        load_share_utility = (
            instant_mix * instant_utility
            + (1.0 - instant_mix) * window_utility
        )
        normalized_total_torque = (
            positive_axis_torque.mul(finger_mask_b).sum(dim=-1)
            / nominal_capacity_sum.clamp_min(eps)
        )
        (
            raw_presence_gate,
            presence_gate,
            presence_floor,
        ) = self._coord_presence_gate(
            normalized_total_torque
        )
        load_share_utility = load_share_utility * presence_gate

        omega_pos = torch.clamp(d_rot * omega_scalar, min=0.0)
        self.coord_omega_hist[:] = torch.cat(
            [self.coord_omega_hist[:, 1:], omega_pos.unsqueeze(1)], dim=1
        )
        omega_bar = self.coord_omega_hist.mean(dim=-1)
        omega_min = float(self.cfg.coord_omega_min)
        radius_scale = self.object_radius_scales.clamp_min(eps)
        omega_ref = float(self.cfg.coord_omega_ref) / radius_scale
        omega_min_scaled = omega_min / radius_scale
        omega_span = (omega_ref - omega_min_scaled).clamp_min(eps)
        q_task_raw = ((omega_bar - omega_min_scaled) / omega_span).clamp(0.0, 1.0)
        q_task = torch.pow(q_task_raw, float(self.cfg.coord_q_power))
        q_floor = self._current_coord_q_floor()
        q_guide = q_floor + (1.0 - q_floor) * q_task
        tactile_conf = (b * finger_mask_b).sum(dim=-1) / max(active_fingers, eps)

        # Slip proxy: relative speed magnitude in the tangent plane (remove radial).
        v_radial = torch.sum(v_rel * r_hat, dim=-1, keepdim=True) * r_hat
        v_tang = v_rel - v_radial
        v_slip = v_tang.norm(dim=-1)
        contact_weight = b * finger_mask_b
        v_slip_bar = (contact_weight * v_slip).sum(dim=-1) / (contact_weight.sum(dim=-1) + eps)
        s_slip = torch.exp(
            -float(self.cfg.coord_k_slip) * v_slip_bar / max(float(self.cfg.coord_v_slip_ref), eps)
        )
        # Soft drop proxy, aligned to the task's active fingertip mask.
        active_fingertip_pos = self._select_active_finger_slots(self.fingertip_pos, dim=1)
        tip_dist = (active_fingertip_pos - nut_body_pos.unsqueeze(1)).norm(dim=-1)
        masked_tip_dist = torch.where(finger_mask_b > 0.5, tip_dist, torch.zeros_like(tip_dist))
        max_task_tip_dist = masked_tip_dist.max(dim=-1).values
        dist_margin = max(float(self.cfg.coord_drop_finger_soft_margin), eps)
        dist_gate = 1.0 - (
            (max_task_tip_dist - float(self.cfg.coord_drop_finger_dist)) / dist_margin
        ).clamp(0.0, 1.0)
        nut_force = torch.linalg.norm(self._nut_contact_sensor.data.net_forces_w[:, 0, :], dim=-1)
        force_min = float(self.cfg.coord_drop_nut_force_min)
        force_scale = max(float(self.cfg.coord_drop_nut_force_soft_scale) - force_min, eps)
        force_gate = ((nut_force - force_min) / force_scale).clamp(0.0, 1.0)
        stability_floor = float(self.cfg.coord_stability_floor)
        dist_gate = stability_floor + (1.0 - stability_floor) * dist_gate
        force_gate = stability_floor + (1.0 - stability_floor) * force_gate
        s_stable = s_slip * dist_gate * force_gate

        g_handover = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        alpha_g = float(self.cfg.coord_alpha_g)
        if alpha_g > 0.0:
            b_th = float(self.cfg.coord_b_th)
            delta_h = float(self.cfg.coord_delta_h)
            e_plus = ((self.coord_b_prev < b_th) & (b >= b_th)).float() * finger_mask_b
            e_minus = ((self.coord_b_prev >= b_th) & (b < b_th)).float() * finger_mask_b
            # Age since last release: reset to 0 on release, else increment.
            self.coord_release_age = torch.where(
                e_minus > 0.5,
                torch.zeros_like(self.coord_release_age),
                self.coord_release_age + 1.0,
            )
            # For each newly contacting finger j, score best other finger i released recently.
            # G = max_{i!=j} e+_j * 1{age_i <= Δh} * exp(-age_i / Δh)
            age = self.coord_release_age
            in_window = (age <= delta_h).float()
            decay = torch.exp(-age / max(delta_h, 1.0)) * in_window * finger_mask_b
            # Exclude self: zero diagonal contribution per env via broadcasting.
            # score[e,j,i] = e_plus[e,j] * decay[e,i] * (i != j)
            eye = torch.eye(num_fingers, device=self.device, dtype=torch.float)
            pair = e_plus.unsqueeze(-1) * decay.unsqueeze(1) * (1.0 - eye.unsqueeze(0))
            g_handover = pair.amax(dim=(-1, -2))

        self.coord_b_prev = b * finger_mask_b

        force_ratio = force_magnitude / max(comfort_force, eps)
        overload_raw = torch.relu(force_ratio - 1.0).square()
        overload = overload_raw / (1.0 + overload_raw)
        overload = (
            overload * b * finger_mask_b
        ).sum(dim=-1) / max(active_fingers, eps)
        off_axis_torque = (torque_magnitude - positive_axis_torque).clamp_min(0.0)
        waste_load = off_axis_torque / torque_capacity.clamp_min(eps)
        waste = (
            (1.0 - torch.exp(-waste_load.clamp(0.0, float(self.cfg.coord_load_max))))
            * b
            * finger_mask_b
        ).sum(dim=-1) / max(active_fingers, eps)

        alpha_h = float(self.cfg.coord_alpha_h)
        positive_utility = (
            alpha_h * load_share_utility + alpha_g * g_handover
        ).clamp(0.0, 1.0)
        effort_cost = (
            float(self.cfg.coord_overload_penalty_weight) * overload
            + float(self.cfg.coord_waste_penalty_weight) * waste
        )
        quality_score = q_guide * s_stable * (positive_utility - effort_cost)
        intrinsic_score, quality_weight = self._combine_coord_intrinsic_score(
            effective_torque_guide_bonus,
            quality_score,
        )
        r_coord = self._apply_coord_intrinsic_reward(
            intrinsic_score,
            task_reward,
            reward_weight=1.0,
            positive_reward_budget=(
                effective_torque_guide_weight + quality_weight
            ),
        )
        self.extras["tactile/coord_intrinsic_weight"] = torch.tensor(
            quality_weight, device=self.device
        )
        self.extras["tactile/coord_quality_weight"] = torch.tensor(
            quality_weight, device=self.device
        )

        useful_load = history_load
        effective_fingers = (
            useful_load.sum(dim=-1).square()
            / useful_load.square().sum(dim=-1).clamp_min(eps)
        ).clamp(0.0, active_fingers)

        self.extras["tactile/coord_quality"] = positive_utility.mean()
        self.extras["tactile/coord_phi"] = intrinsic_score.mean()
        self.extras["tactile/coord_load_utility"] = load_share_utility.mean()
        self.extras["tactile/coord_effective_torque_normalized"] = (
            normalized_effective_torque.mean()
        )
        self.extras["tactile/coord_effective_torque_guide"] = (
            effective_torque_guide.mean()
        )
        self.extras["tactile/coord_effective_torque_reward_weight"] = torch.tensor(
            effective_torque_guide_weight, device=self.device
        )
        self.extras["tactile/coord_effective_torque_guide_bonus"] = (
            effective_torque_guide_bonus.mean()
        )
        self.extras["tactile/coord_presence_floor"] = torch.tensor(
            presence_floor, device=self.device
        )
        self.extras["tactile/coord_presence_gate_raw"] = raw_presence_gate.mean()
        self.extras["tactile/coord_presence_gate"] = presence_gate.mean()
        self.extras["tactile/coord_instant_utility"] = instant_utility.mean()
        self.extras["tactile/coord_window_utility"] = window_utility.mean()
        self.extras["tactile/coord_effective_fingers"] = effective_fingers.mean()
        self.extras["tactile/coord_axis_efficiency"] = (
            axis_efficiency * finger_mask_b
        ).sum() / (finger_mask_b.sum() * self.num_envs).clamp_min(eps)
        self.extras["tactile/coord_overload"] = overload.mean()
        self.extras["tactile/coord_waste"] = waste.mean()
        self.extras["tactile/coord_capacity"] = nominal_capacity_sum.mean()
        self.extras["tactile/coord_q_raw"] = q_task_raw.mean()
        self.extras["tactile/coord_q"] = q_task.mean()
        self.extras["tactile/coord_q_floor"] = torch.tensor(q_floor, device=self.device)
        self.extras["tactile/coord_q_guide"] = q_guide.mean()
        self.extras["tactile/coord_tactile_conf"] = tactile_conf.mean()
        self.extras["tactile/coord_s"] = s_stable.mean()
        self.extras["tactile/coord_G"] = g_handover.mean()
        self.extras["tactile/coord_raw_reward"] = intrinsic_score.mean()
        self.extras["tactile/coord_C_sum"] = positive_axis_torque.sum(dim=-1).mean()
        self.extras["tactile/coord_presence"] = presence_gate.mean()
        self.extras["tactile/coord_s_slip"] = s_slip.mean()
        self.extras["tactile/coord_s_dist"] = dist_gate.mean()
        self.extras["tactile/coord_s_force"] = force_gate.mean()
        self.extras["tactile/coord_finger_mask_count"] = self.coord_finger_mask_sum.detach()
        return r_coord

    def _compute_multi_contact_reward(self) -> torch.Tensor:
        """Bounded bonus for multi-finger contact that produces valve rotation.

        r_coord = scale * coordination * gate, where coordination ramps 0->1 as the
        soft finger count (sum of per-finger EMA duty cycles) goes min->max fingers,
        and gate ramps 0->1 with the valve angle traveled over the last rot_window
        control steps. Both factors saturate: force magnitude and rotation speed
        stay priced by torque_penalty and rotate_reward, this term only buys the
        sustained >=3-finger drive posture.
        """
        num_fingers = len(self.cfg.tactile_tip_body_names)
        if self.cfg.tactile_layout == ESTIMATED_OFFICIAL_LAYOUT:
            physical_force, valid = self._get_tactile_force_magnitude()
            physical_force = physical_force * float(self.cfg.tactile_force_scale)
            physical_force = physical_force.masked_fill(~valid.unsqueeze(0), 0.0)
            peak_force = physical_force.max(dim=-1).values
        else:
            pool_rows, pool_cols = self.cfg.tactile_array_pool
            tactile = self._compute_tactile_array_features()
            # Per-finger regular-grid blocks are channel-major: (3, pooled cells).
            cell_force = tactile.view(
                self.num_envs, num_fingers, 3, pool_rows * pool_cols
            ).norm(dim=2)
            peak_force = cell_force.max(dim=-1).values
        contact = torch.tanh(peak_force / self.cfg.multi_contact_tau)
        lam = self.cfg.multi_contact_ema_lambda
        self.contact_duty_ema[:] = lam * self.contact_duty_ema + (1.0 - lam) * contact

        soft_count = self.contact_duty_ema.sum(dim=-1)
        lo, hi = self.cfg.multi_contact_min_fingers, self.cfg.multi_contact_max_fingers
        coordination = torch.clamp((soft_count - lo) / (hi - lo), 0.0, 1.0)

        # nut_dof_pos_history is appended in compute_observations (after rewards),
        # so index -rot_window holds the angle exactly rot_window control steps ago;
        # resets refill the history, zeroing the gate for fresh episodes.
        dtheta = self.nut_dof_pos - self.nut_dof_pos_history[:, -self.cfg.multi_contact_rot_window]
        gate = torch.clamp(dtheta / self.cfg.multi_contact_rot_ref, 0.0, 1.0)

        r_coord = self.cfg.multi_contact_reward_scale * coordination * gate
        self.extras["tactile/soft_finger_count"] = soft_count.mean()
        self.extras["tactile/coord_factor"] = coordination.mean()
        self.extras["tactile/rot_gate"] = gate.mean()
        self.extras["tactile/multi_contact_reward"] = r_coord.mean()
        return r_coord

    def _compute_tactile_array_features(self) -> torch.Tensor:
        """Pooled per-finger (normal, shear_x, shear_y) taxel grids, scaled and clipped.

        Returns:
            Tensor of shape (num_envs, num_fingers * pool_h * pool_w * 3).
        """
        rows, cols = self.cfg.tactile_array_size
        pool_rows, pool_cols = self.cfg.tactile_array_pool
        kernel = (rows // pool_rows, cols // pool_cols)
        features = []
        for sensor in self._tactile_sensor:
            data = sensor.data
            normal = data.tactile_normal_force              # (E, rows*cols)
            shear = data.tactile_shear_force                # (E, rows*cols, 2)
            taxels = torch.cat([normal.unsqueeze(-1), shear], dim=-1)
            taxels = taxels.view(self.num_envs, rows, cols, 3).permute(0, 3, 1, 2)
            pooled = F.avg_pool2d(taxels, kernel_size=kernel)
            features.append(pooled.reshape(self.num_envs, -1))
        tactile = torch.cat(features, dim=-1) * self.cfg.tactile_force_scale
        return torch.clamp(tactile, -self.cfg.tactile_force_clip, self.cfg.tactile_force_clip)

    def _build_teacher_tactile_priv(self, tactile: torch.Tensor, at_reset_env_ids: torch.Tensor) -> torch.Tensor:
        """Teacher tactile priv frame with optional one-step force delta."""
        if not self.cfg.tactile_teacher_use_delta:
            self.prev_teacher_tactile_buf[:] = tactile
            return tactile

        delta = tactile - self.prev_teacher_tactile_buf
        if len(at_reset_env_ids) > 0:
            delta[at_reset_env_ids] = 0.0
        self.prev_teacher_tactile_buf[:] = tactile
        return torch.cat([tactile, delta], dim=-1)

    def _setup_tactile_visualizers(self) -> None:
        """Create optional Isaac GUI visualizers for tactile taxels and contact-force vectors."""

        if not (
            getattr(self.cfg, "tactile_visualize_taxel_points", False)
            or getattr(self.cfg, "tactile_visualize_contact_forces", False)
        ):
            return

        try:
            from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
        except Exception as exc:
            print(f"[WARN] TacSL GUI visualization unavailable: {exc}", flush=True)
            return

        if getattr(self.cfg, "tactile_visualize_taxel_points", False):
            try:
                self._setup_tactile_taxel_spheres()
            except Exception as exc:
                print(f"[WARN] Could not create tactile taxel spheres: {exc}", flush=True)

        if getattr(self.cfg, "tactile_visualize_contact_forces", False):
            marker_cfg = VisualizationMarkersCfg(
                prim_path="/Visuals/TactileContactForceTips",
                markers={
                    "force_tip": sim_utils.SphereCfg(
                        radius=float(self.cfg.tactile_contact_force_marker_radius),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.05, 0.05)),
                    ),
                },
            )
            self._tactile_force_tip_visualizer = VisualizationMarkers(marker_cfg)

        try:
            from isaacsim.util.debug_draw import _debug_draw as debug_draw
        except ImportError:
            try:
                import omni.isaac.debug_draw._debug_draw as debug_draw
            except ImportError:
                debug_draw = None
        if debug_draw is not None:
            self._tactile_debug_draw = debug_draw.acquire_debug_draw_interface()

    def _setup_tactile_taxel_spheres(self) -> None:
        """Create stable USD sphere prims for the selected environment's physical taxels."""

        env_index = int(getattr(self.cfg, "tactile_vis_env_index", 0))
        if env_index < 0 or env_index >= self.num_envs:
            return
        positions_w = self._tactile_taxel_positions_w(env_index)
        if positions_w is None or positions_w.numel() == 0:
            return

        root_path = "/Visuals/TactileTaxelSpheres"
        sim_utils.create_prim(root_path, prim_type="Xform")
        color = Vt.Vec3fArray([Gf.Vec3f(0.1, 0.25, 1.0)])
        radius = float(self.cfg.tactile_taxel_marker_radius)
        for taxel_index, position in enumerate(positions_w.detach().cpu().tolist()):
            prim = sim_utils.create_prim(
                f"{root_path}/Taxel_{taxel_index:04d}",
                prim_type="Sphere",
                position=position,
                attributes={"radius": radius},
            )
            UsdGeom.Gprim(prim).CreateDisplayColorAttr().Set(color)

        self._tactile_taxel_visualizer = XformPrimView(
            f"{root_path}/Taxel_.*",
            device=self.device,
            sync_usd_on_fabric_write=True,
        )
        if self._tactile_taxel_visualizer.count != positions_w.shape[0]:
            raise RuntimeError(
                "Unexpected tactile sphere count: "
                f"{self._tactile_taxel_visualizer.count} != {positions_w.shape[0]}"
            )
        print(
            f"[INFO] Tactile GUI: created {positions_w.shape[0]} blue physical-taxel spheres "
            f"for env {env_index}.",
            flush=True,
        )

    def _contact_forces_w(self) -> torch.Tensor:
        """Return object-filtered per-fingertip contact forces in world frame."""

        forces = []
        for sensor in self._contact_sensor:
            force_by_filter = getattr(sensor.data, "force_matrix_w", None)
            if force_by_filter is None:
                force = sensor.data.net_forces_w[:, 0, :]
            else:
                force = torch.nan_to_num(force_by_filter[:, 0, :, :]).sum(dim=1)
            forces.append(force)
        return torch.stack(forces, dim=1)

    def _tactile_taxel_positions_w(self, env_index: int) -> torch.Tensor | None:
        """Return all tactile taxel positions for one environment in world frame."""

        positions = []
        for finger_idx, sensor in enumerate(self._tactile_sensor):
            local_points = getattr(
                sensor,
                "_tactile_physical_center_pos_local",
                getattr(sensor, "_tactile_pos_local", None),
            )
            if local_points is None:
                return None
            body_id = self.tactile_tip_body_ids[finger_idx]
            pos_w = self.hand.data.body_pos_w[env_index, body_id].unsqueeze(0)
            quat_w = self.hand.data.body_quat_w[env_index, body_id].unsqueeze(0)
            positions.append(quat_apply(quat_w.repeat(local_points.shape[0], 1), local_points) + pos_w)
        return torch.cat(positions, dim=0)

    def _update_tactile_debug_visualization(self) -> None:
        """Refresh optional Isaac GUI visualization for taxel grids and 3D contact forces."""

        if self._tactile_taxel_visualizer is None and self._tactile_force_tip_visualizer is None:
            return
        env_index = int(getattr(self.cfg, "tactile_vis_env_index", 0))
        if env_index < 0 or env_index >= self.num_envs:
            if not self._tactile_vis_warned:
                print(
                    f"[WARN] tactile_vis_env_index={env_index} outside [0, {self.num_envs}); disabling tactile GUI vis.",
                    flush=True,
                )
                self._tactile_vis_warned = True
            return

        try:
            if self._tactile_taxel_visualizer is not None:
                taxel_pos_w = self._tactile_taxel_positions_w(env_index)
                if taxel_pos_w is not None:
                    self._tactile_taxel_visualizer.set_world_poses(positions=taxel_pos_w)
                    if self._tactile_debug_draw is not None:
                        self._tactile_debug_draw.clear_points()
                        points = [tuple(point) for point in taxel_pos_w.detach().cpu().tolist()]
                        colors = [(0.1, 0.25, 1.0, 1.0)] * len(points)
                        point_size = max(
                            5.0,
                            8.0 * float(self.cfg.tactile_taxel_marker_radius) / 0.0005,
                        )
                        self._tactile_debug_draw.draw_points(
                            points,
                            colors,
                            [point_size] * len(points),
                        )

            if self._tactile_force_tip_visualizer is not None:
                contact_forces_w = self._select_active_finger_slots(
                    self._contact_forces_w(), dim=1
                )[env_index]
                finger_pos_w = self.hand.data.body_pos_w[env_index, self.tactile_tip_body_ids]
                force_norm = contact_forces_w.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                force_len = (force_norm * float(self.cfg.tactile_contact_force_vis_scale)).clamp(
                    max=float(self.cfg.tactile_contact_force_vis_max_len)
                )
                force_end_w = finger_pos_w + contact_forces_w / force_norm * force_len
                self._tactile_force_tip_visualizer.visualize(translations=force_end_w)

                if self._tactile_debug_draw is not None:
                    self._tactile_debug_draw.clear_lines()
                    starts = [tuple(p) for p in finger_pos_w.detach().cpu().tolist()]
                    ends = [tuple(p) for p in force_end_w.detach().cpu().tolist()]
                    colors = [(1.0, 0.05, 0.05, 1.0)] * len(starts)
                    sizes = [4.0] * len(starts)
                    self._tactile_debug_draw.draw_lines(starts, ends, colors, sizes)
        except Exception as exc:
            if not self._tactile_vis_warned:
                print(f"[WARN] TacSL GUI visualization failed (suppressing further warnings): {exc}", flush=True)
                self._tactile_vis_warned = True
