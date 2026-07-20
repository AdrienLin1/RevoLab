# Copyright (c) 2026, BrainCo.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""DirectRLEnv for Revo3 HORA screw/valve tasks with TacSL fingertip array sensing.

Extends Revo3HandScrewEnv with the tactile-revo3 TacSL force-field arrays:
five 16x16 taxel grids (normal + 2D shear per taxel) on the fingertip
elastomers. The pooled arrays are written into the tail of ``priv_info`` — the
HORA Stage-1 teacher observation. The actor observation (141 dims) and all
terminations/randomizations of the base task are unchanged. Rewards are
unchanged unless ``multi_contact_reward_scale`` is nonzero (valve tasks), which
adds a bounded multi-finger coordination bonus computed from the tactile
arrays (see ``_compute_multi_contact_reward``).

The TacSL force field computes per-taxel penetration against the rotating
nut/handle via PhysX SDF queries, which requires that body's collision mesh to
use the "sdf" approximation. The approximation override is applied to the
env_0 prototype before cloning, so every cloned env inherits it.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sensors import ContactSensor
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics, Vt

from .revo3_hand_screw_env import _LOCAL_GROUND_USD, Revo3HandScrewEnv
from .revo3_hand_screw_tactile_env_cfg import Revo3HandScrewTactileMixinCfg


class Revo3HandScrewTactileEnv(Revo3HandScrewEnv):
    cfg: Revo3HandScrewTactileMixinCfg

    def __init__(self, cfg: Revo3HandScrewTactileMixinCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        # Stage2 student: binary tactile history, one {0,1} frame per control step
        self.student_tactile_hist_buf = torch.zeros(
            (self.num_envs, self.cfg.student_tactile_history_len, self.cfg.student_tactile_frame_dim),
            device=self.device, dtype=torch.float,
        )
        # per-finger contact duty cycle (EMA of the tanh contact indicator),
        # updated once per control step in _get_rewards
        self.contact_duty_ema = torch.zeros(
            (self.num_envs, len(self.cfg.tactile_tip_body_names)), device=self.device
        )

    def _setup_scene(self):
        # Mirrors Revo3HandScrewEnv._setup_scene, inserting the SDF collision
        # override (before cloning) and the TacSL fingertip sensors.
        self.hand = Articulation(self.cfg.robot_cfg)
        self.object = Articulation(self.cfg.object_cfg)
        self._apply_sdf_collision_to_screw_body()
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(usd_path=_LOCAL_GROUND_USD))
        self.scene.clone_environments(copy_from_source=False)
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

    def compute_observations(self):
        # capture reset flags before the base class consumes them
        at_reset_env_ids = self.at_reset_buf.nonzero(as_tuple=False).squeeze(-1)
        obs_buf = super().compute_observations()
        tactile = self._compute_tactile_array_features()
        self.priv_info_buf[:, self.cfg.tactile_priv_offset:] = tactile
        self.extras["tactile/priv_abs_mean"] = tactile.abs().mean()
        self.extras["tactile/priv_abs_max"] = tactile.abs().max()

        # student binary tactile history: threshold the 240 pooled channels to {0,1},
        # drop the oldest frame, append the current one
        binary_tactile = torch.where(
            tactile.abs() > self.cfg.student_tactile_contact_threshold, 1.0, 0.0)
        self.student_tactile_hist_buf[:] = torch.cat(
            [self.student_tactile_hist_buf[:, 1:], binary_tactile.unsqueeze(1)], dim=1)
        # envs that just reset: fill the whole history with the first post-reset frame
        # (same convention as obs_buf_lag_history)
        if len(at_reset_env_ids) > 0:
            self.student_tactile_hist_buf[at_reset_env_ids] = binary_tactile[at_reset_env_ids].unsqueeze(1).repeat(
                1, self.cfg.student_tactile_history_len, 1)
        self.extras["tactile/student_binary_mean"] = binary_tactile.mean()
        return obs_buf

    def _get_observations(self) -> dict:
        obs_dict = super()._get_observations()
        # Stage2 student inputs (real-robot sensing only): the first 42 dims of each
        # lag-history frame are joint_pos + current targets — no contact forces
        obs_dict["student_proprio_hist"] = self.obs_buf_lag_history[
            :, -self.cfg.student_proprio_history_len:, : self.cfg.student_proprio_frame_dim].clone()
        obs_dict["student_tactile_hist"] = self.student_tactile_hist_buf.clone()
        return obs_dict

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = self.hand._ALL_INDICES
        super()._reset_idx(env_ids)
        self.student_tactile_hist_buf[env_ids] = 0.0
        self.contact_duty_ema[env_ids] = 0.0

    def _get_rewards(self) -> torch.Tensor:
        total_reward = super()._get_rewards()
        if self.cfg.multi_contact_reward_scale == 0.0:
            return total_reward
        total_reward = total_reward + self._compute_multi_contact_reward()
        self.extras["total_reward"] = total_reward.mean()
        return total_reward

    def _compute_multi_contact_reward(self) -> torch.Tensor:
        """Bounded bonus for multi-finger contact that produces valve rotation.

        r_coord = scale * coordination * gate, where coordination ramps 0->1 as the
        soft finger count (sum of per-finger EMA duty cycles) goes min->max fingers,
        and gate ramps 0->1 with the valve angle traveled over the last rot_window
        control steps. Both factors saturate: force magnitude and rotation speed
        stay priced by torque_penalty and rotate_reward, this term only buys the
        sustained >=3-finger drive posture.
        """
        pool_rows, pool_cols = self.cfg.tactile_array_pool
        num_fingers = len(self.cfg.tactile_tip_body_names)
        tactile = self._compute_tactile_array_features()
        # per-finger blocks are channel-major: (3, pool_rows*pool_cols)
        cell_force = tactile.view(self.num_envs, num_fingers, 3, pool_rows * pool_cols).norm(dim=2)
        contact = torch.tanh(cell_force.max(dim=-1).values / self.cfg.multi_contact_tau)
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
