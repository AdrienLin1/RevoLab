# Copyright (c) 2026, BrainCo.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tactile sensor constants and config helpers for the right Revo3 hand."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import isaaclab.sim as sim_utils
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils import math as math_utils
from pxr import Gf, Usd, UsdGeom, UsdPhysics

from ... import mdp
from .....tactile_layout import (
    ESTIMATED_OFFICIAL_LAYOUT,
    REGULAR_GRID_LAYOUT,
    estimated_official_calibration_bounds_xy,
    estimated_official_centers_xy,
    estimated_official_sensor_patch_quadrature,
    estimated_official_sensor_plane_xy,
    estimated_official_slot_center_indices,
    finger_name_from_tip_body_path,
    resolve_tactile_layout_name,
    validate_tactile_layout_name,
)

try:
    from isaaclab_contrib.sensors.tacsl_sensor import VisuoTactileSensor as _VisuoTactileSensor
    from isaaclab_contrib.sensors.tacsl_sensor import VisuoTactileSensorCfg as _VisuoTactileSensorCfg
except ImportError:
    _VisuoTactileSensor = None
    _VisuoTactileSensorCfg = None


TACTILE_FINGER_ORDER = ("little", "ring", "middle", "index", "thumb")

TACTILE_DIP_BODIES = tuple(f"right_{finger}_DIP_Link" for finger in TACTILE_FINGER_ORDER)
TACTILE_TIP_BODIES = tuple(f"right_{finger}_tip_Link" for finger in TACTILE_FINGER_ORDER)

TACTILE_FORCE_SENSOR_NAMES = tuple(f"{finger}_tactile_force" for finger in TACTILE_FINGER_ORDER)
TACTILE_VIS_SENSOR_NAMES = tuple(f"{finger}_tactile_sensor" for finger in TACTILE_FINGER_ORDER)

TACTILE_USD_PATH = (
    Path(__file__).resolve().parents[8] / "assets" / "usd" / "dexsuite" / "Tianji_Revo3_Right_tactile.usda"
)
TACTILE_CUBE_USD_PATH = (
    Path(__file__).resolve().parents[8] / "assets" / "usd" / "dexsuite" / "tactile_cube_sdf.usda"
)


if _VisuoTactileSensorCfg is not None:

    @configclass
    class Revo3VisuoTactileSensorCfg(_VisuoTactileSensorCfg):
        """TacSL config extended with Revo3 layout selection metadata."""

        tactile_layout: str = REGULAR_GRID_LAYOUT
        tactile_layout_finger: str = ""

else:
    Revo3VisuoTactileSensorCfg = None


if _VisuoTactileSensor is not None:

    class Revo3VisuoTactileSensor(_VisuoTactileSensor):
        """TacSL sensor variant that initializes the no-contact camera baseline automatically."""

        def _update_buffers_impl(self, env_ids):
            """Refresh one consistent force batch and integrate circular sensor patches."""

            if self.cfg.enable_force_field and len(env_ids) != self._num_envs:
                env_ids = torch.arange(self._num_envs, device=self._device)
            super()._update_buffers_impl(env_ids)
            self._aggregate_estimated_patch_forces()

        def _aggregate_estimated_patch_forces(self) -> None:
            """Integrate circular-patch samples and broadcast one result per fixed slot."""

            if getattr(self, "_tactile_layout_name", None) != ESTIMATED_OFFICIAL_LAYOUT:
                return
            sensor_count = int(self._tactile_patch_sensor_count)
            samples_per_sensor = int(self._tactile_patch_samples_per_sensor)
            query_count = sensor_count * samples_per_sensor
            weights = self._tactile_patch_weights.view(1, 1, samples_per_sensor)

            normal_samples = self._data.tactile_normal_force[:, :query_count].view(
                self._num_envs, sensor_count, samples_per_sensor
            )
            shear_samples = self._data.tactile_shear_force[:, :query_count].view(
                self._num_envs, sensor_count, samples_per_sensor, 2
            )
            depth_samples = self._data.penetration_depth[:, :query_count].view(
                self._num_envs, sensor_count, samples_per_sensor
            )
            physical_normal = torch.sum(normal_samples * weights, dim=-1)
            physical_shear = torch.sum(shear_samples * weights.unsqueeze(-1), dim=-2)
            physical_depth = torch.sum(depth_samples * weights, dim=-1)

            self._physical_tactile_normal_force = physical_normal
            self._physical_tactile_shear_force = physical_shear
            self._physical_tactile_penetration_depth = physical_depth
            slot_indices = self._tactile_slot_center_indices
            self._data.tactile_normal_force.copy_(physical_normal.index_select(1, slot_indices))
            self._data.tactile_shear_force.copy_(physical_shear.index_select(1, slot_indices))
            self._data.penetration_depth.copy_(physical_depth.index_select(1, slot_indices))

        def _find_tip_body_prim(self):
            """Find the rigid fingertip body that owns the elastomer visual geometry."""

            elastomer_prim = self._parent_prims[0]
            current_prim = elastomer_prim.GetParent()
            while current_prim and current_prim.IsValid():
                if current_prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    return current_prim
                current_prim = current_prim.GetParent()
                if not current_prim or not current_prim.IsValid() or current_prim.GetPath().pathString == "/":
                    break
            raise RuntimeError(
                f"No rigid fingertip body found above elastomer at path: "
                f"{elastomer_prim.GetPath().pathString}"
            )

        @staticmethod
        def _transform_points_between_prims(points: np.ndarray, source_prim, target_prim) -> np.ndarray:
            """Transform USD mesh-local points into another prim's local frame."""

            xform_cache = UsdGeom.XformCache()
            source_to_world = xform_cache.GetLocalToWorldTransform(source_prim)
            world_to_target = xform_cache.GetLocalToWorldTransform(target_prim).GetInverse()

            transformed_points = []
            for point in points:
                world_point = source_to_world.Transform(Gf.Vec3d(float(point[0]), float(point[1]), float(point[2])))
                target_point = world_to_target.Transform(world_point)
                transformed_points.append([target_point[0], target_point[1], target_point[2]])

            return np.asarray(transformed_points, dtype=np.float32)

        @staticmethod
        def _quat_from_rotation_matrix(rotation_matrix: np.ndarray) -> np.ndarray:
            """Convert a 3x3 rotation matrix to an Isaac Lab wxyz quaternion."""

            trace = float(np.trace(rotation_matrix))
            if trace > 0.0:
                scale = np.sqrt(trace + 1.0) * 2.0
                return np.array(
                    [
                        0.25 * scale,
                        (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / scale,
                        (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / scale,
                        (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / scale,
                    ],
                    dtype=np.float32,
                )

            if rotation_matrix[0, 0] > rotation_matrix[1, 1] and rotation_matrix[0, 0] > rotation_matrix[2, 2]:
                scale = np.sqrt(1.0 + rotation_matrix[0, 0] - rotation_matrix[1, 1] - rotation_matrix[2, 2]) * 2.0
                quat = [
                    (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / scale,
                    0.25 * scale,
                    (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / scale,
                    (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / scale,
                ]
            elif rotation_matrix[1, 1] > rotation_matrix[2, 2]:
                scale = np.sqrt(1.0 + rotation_matrix[1, 1] - rotation_matrix[0, 0] - rotation_matrix[2, 2]) * 2.0
                quat = [
                    (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / scale,
                    (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / scale,
                    0.25 * scale,
                    (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / scale,
                ]
            else:
                scale = np.sqrt(1.0 + rotation_matrix[2, 2] - rotation_matrix[0, 0] - rotation_matrix[1, 1]) * 2.0
                quat = [
                    (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / scale,
                    (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / scale,
                    (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / scale,
                    0.25 * scale,
                ]

            return np.asarray(quat, dtype=np.float32)

        def _rotation_quat_between_prims(self, source_prim, target_prim) -> torch.Tensor:
            """Return the source prim orientation expressed in the target prim frame."""

            source_basis = np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
            target_basis = self._transform_points_between_prims(source_basis, source_prim, target_prim)
            target_axes = target_basis[1:] - target_basis[0]
            target_axes /= np.linalg.norm(target_axes, axis=1, keepdims=True).clip(min=1e-9)
            rotation_matrix = target_axes.T
            u, _, vh = np.linalg.svd(rotation_matrix)
            rotation_matrix = u @ vh
            if np.linalg.det(rotation_matrix) < 0.0:
                u[:, -1] *= -1.0
                rotation_matrix = u @ vh

            quat = self._quat_from_rotation_matrix(rotation_matrix)
            quat /= np.linalg.norm(quat).clip(min=1e-9)
            return torch.tensor(quat, dtype=torch.float32, device=self._device)

        def _basis_between_prims(self, source_prim, target_prim) -> tuple[np.ndarray, np.ndarray]:
            """Return source origin and unit axes expressed in target prim frame."""

            source_basis = np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
            target_basis = self._transform_points_between_prims(source_basis, source_prim, target_prim)
            target_axes = target_basis[1:] - target_basis[0]
            target_axes /= np.linalg.norm(target_axes, axis=1, keepdims=True).clip(min=1e-9)
            return target_basis[0], target_axes

        @staticmethod
        def _is_descendant_of(prim, ancestor_prim) -> bool:
            """Check whether prim lives under ancestor_prim in the USD hierarchy."""

            ancestor_path = ancestor_prim.GetPath()
            current_prim = prim
            while current_prim and current_prim.IsValid():
                if current_prim.GetPath() == ancestor_path:
                    return True
                current_prim = current_prim.GetParent()
            return False

        def _collect_mesh_prims(self, root_prim, *, excluded_ancestor=None) -> list:
            """Collect mesh prims under a root prim while skipping excluded descendants."""

            mesh_prims = []
            for prim in Usd.PrimRange(root_prim, Usd.TraverseInstanceProxies()):
                if excluded_ancestor is not None and prim != root_prim and self._is_descendant_of(
                    prim, excluded_ancestor
                ):
                    continue
                if prim.IsA(UsdGeom.Mesh):
                    mesh_prims.append(prim)
            return mesh_prims

        @staticmethod
        def _mesh_point_count(mesh_prim) -> int:
            """Return the number of authored points on a USD mesh prim."""

            points = UsdGeom.Mesh(mesh_prim).GetPointsAttr().Get()
            return 0 if points is None else len(points)

        def _score_surface_mesh(self, mesh_prim) -> tuple[int, int]:
            """Rank candidate fingertip surface meshes."""

            path = mesh_prim.GetPath().pathString.lower()
            score = 0
            if "tactile" in path or "gel_surface" in path:
                score -= 100
            if "visual" in path or "mesh" in path:
                score += 20
            if "collision" in path or mesh_prim.HasAPI(UsdPhysics.CollisionAPI):
                score -= 5
            return score, self._mesh_point_count(mesh_prim)

        def _find_fingertip_surface_mesh_prim(self, tip_body_prim, elastomer_prim):
            """Find the fingertip mesh used to curve the tactile force-field samples."""

            mesh_prims = self._collect_mesh_prims(tip_body_prim, excluded_ancestor=elastomer_prim)
            if mesh_prims:
                return max(mesh_prims, key=self._score_surface_mesh)

            tip_path = tip_body_prim.GetPath().pathString
            dip_body_prim = tip_body_prim.GetStage().GetPrimAtPath(tip_path.replace("_tip_Link", "_DIP_Link"))
            if not dip_body_prim or not dip_body_prim.IsValid():
                return None

            mesh_prims = self._collect_mesh_prims(dip_body_prim)
            if not mesh_prims:
                return None
            return max(mesh_prims, key=self._score_surface_mesh)

        @staticmethod
        def _mesh_triangles(mesh_prim) -> list[tuple[int, int, int]]:
            """Triangulate USD mesh face indices by fan splitting."""

            usd_mesh = UsdGeom.Mesh(mesh_prim)
            face_counts = usd_mesh.GetFaceVertexCountsAttr().Get()
            face_indices = usd_mesh.GetFaceVertexIndicesAttr().Get()
            if face_counts is None or face_indices is None:
                return []

            triangles = []
            offset = 0
            for count in face_counts:
                count = int(count)
                if count >= 3:
                    first_idx = int(face_indices[offset])
                    for face_idx in range(1, count - 1):
                        triangles.append(
                            (
                                first_idx,
                                int(face_indices[offset + face_idx]),
                                int(face_indices[offset + face_idx + 1]),
                            )
                        )
                offset += count
            return triangles

        @staticmethod
        def _triangle_height_at_uv(point_uv: np.ndarray, triangle_coords: np.ndarray) -> float | None:
            """Interpolate normal height where a vertical ray crosses one triangle."""

            point_a = triangle_coords[0]
            edge_ab = triangle_coords[1, :2] - point_a[:2]
            edge_ac = triangle_coords[2, :2] - point_a[:2]
            point_ap = point_uv - point_a[:2]
            det = edge_ab[0] * edge_ac[1] - edge_ac[0] * edge_ab[1]
            if abs(float(det)) < 1e-12:
                return None

            bary_b = (point_ap[0] * edge_ac[1] - edge_ac[0] * point_ap[1]) / det
            bary_c = (edge_ab[0] * point_ap[1] - point_ap[0] * edge_ab[1]) / det
            bary_a = 1.0 - bary_b - bary_c
            eps = 1e-6
            if bary_a < -eps or bary_b < -eps or bary_c < -eps:
                return None
            return float(bary_a * point_a[2] + bary_b * triangle_coords[1, 2] + bary_c * triangle_coords[2, 2])

        @staticmethod
        def _nearest_surface_height(point_uv: np.ndarray, surface_coords: np.ndarray, cell_radius: float) -> float:
            """Estimate a surface height from nearby mesh vertices."""

            distances_sq = np.sum((surface_coords[:, :2] - point_uv) ** 2, axis=1)
            radius_sq = float(cell_radius * cell_radius)
            candidate_mask = distances_sq <= radius_sq
            if np.count_nonzero(candidate_mask) < 4:
                num_neighbors = min(16, surface_coords.shape[0])
                neighbor_idxs = np.argpartition(distances_sq, num_neighbors - 1)[:num_neighbors]
                return float(np.max(surface_coords[neighbor_idxs, 2]))
            return float(np.max(surface_coords[candidate_mask, 2]))

        @staticmethod
        def _analytic_fingertip_surface_height(flat_coords: np.ndarray, surface_coords: np.ndarray) -> np.ndarray:
            """Create a shallow fingertip pad dome when the authored gel mesh is still flat."""

            uv_min = surface_coords[:, :2].min(axis=0)
            uv_max = surface_coords[:, :2].max(axis=0)
            uv_center = (uv_min + uv_max) * 0.5
            uv_half_size = ((uv_max - uv_min) * 0.5).clip(min=1e-6)

            normalized_uv = (flat_coords[:, :2] - uv_center) / uv_half_size
            radius_sq = normalized_uv[:, 0] ** 2 + 0.55 * normalized_uv[:, 1] ** 2
            dome_profile = np.clip(1.0 - radius_sq, 0.0, 1.0)
            base_height = float(np.max(surface_coords[:, 2]))
            crown_height = 0.0025
            return base_height + crown_height * dome_profile

        def _project_points_to_elastomer_surface(
            self,
            flat_points_mesh: np.ndarray,
            elastomer_mesh_prim,
            axis_idxs: list[int],
            slim_axis: int,
            grid_axes: list[np.ndarray],
        ) -> np.ndarray:
            """Project local tactile grid points onto the authored elastomer mesh surface."""

            usd_mesh = UsdGeom.Mesh(elastomer_mesh_prim)
            surface_points = usd_mesh.GetPointsAttr().Get()
            if surface_points is None:
                return flat_points_mesh

            surface_points = np.asarray(surface_points, dtype=np.float32)
            if surface_points.size == 0:
                return flat_points_mesh

            coord_idxs = [axis_idxs[0], axis_idxs[1], slim_axis]
            surface_coords = surface_points[:, coord_idxs]
            flat_coords = flat_points_mesh[:, coord_idxs]
            triangles = self._mesh_triangles(elastomer_mesh_prim)

            grid_steps = []
            for grid_axis in grid_axes:
                if len(grid_axis) > 1:
                    grid_steps.append(float(np.min(np.diff(np.sort(grid_axis)))))
            cell_radius = max(grid_steps, default=0.002) * 1.25

            projected_coords = flat_coords.copy()
            for point_idx, flat_coord in enumerate(flat_coords):
                point_uv = flat_coord[:2]
                surface_height = None
                for triangle in triangles:
                    triangle_coords = surface_coords[list(triangle)]
                    height = self._triangle_height_at_uv(point_uv, triangle_coords)
                    if height is None:
                        continue
                    if surface_height is None or height > surface_height:
                        surface_height = height
                if surface_height is None:
                    surface_height = self._nearest_surface_height(point_uv, surface_coords, cell_radius)
                projected_coords[point_idx, 2] = surface_height

            if float(np.ptp(surface_coords[:, 2])) < 1e-6:
                projected_coords[:, 2] = self._analytic_fingertip_surface_height(flat_coords, surface_coords)
                print(
                    "[INFO] TacSL tactile gel_surface is flat; using analytic fingertip dome heights for "
                    f"{elastomer_mesh_prim.GetPath().pathString}.",
                    flush=True,
                )

            projected_points = flat_points_mesh.copy()
            projected_points[:, coord_idxs] = projected_coords
            return projected_points.astype(np.float32)

        def _project_points_to_fingertip_surface(
            self,
            flat_points_tip: np.ndarray,
            elastomer_mesh_prim,
            tip_body_prim,
            axis_idxs: list[int],
            slim_axis: int,
            grid_axes: list[np.ndarray],
        ) -> np.ndarray:
            """Project a flat tactile grid onto the fingertip surface along the sensor normal."""

            elastomer_prim = self._parent_prims[0]
            surface_mesh_prim = self._find_fingertip_surface_mesh_prim(tip_body_prim, elastomer_prim)
            if surface_mesh_prim is None:
                return flat_points_tip

            surface_points = UsdGeom.Mesh(surface_mesh_prim).GetPointsAttr().Get()
            if surface_points is None:
                return flat_points_tip

            surface_points = np.asarray(surface_points, dtype=np.float32)
            if surface_points.size == 0:
                return flat_points_tip

            origin_tip, mesh_axes_tip = self._basis_between_prims(elastomer_mesh_prim, tip_body_prim)
            surface_points_tip = self._transform_points_between_prims(
                surface_points,
                surface_mesh_prim,
                tip_body_prim,
            )

            basis_axes = np.stack(
                [
                    mesh_axes_tip[axis_idxs[0]],
                    mesh_axes_tip[axis_idxs[1]],
                    mesh_axes_tip[slim_axis],
                ],
                axis=0,
            )
            flat_coords = (flat_points_tip - origin_tip) @ basis_axes.T
            surface_coords = (surface_points_tip - origin_tip) @ basis_axes.T
            triangles = self._mesh_triangles(surface_mesh_prim)
            triangle_coords = surface_coords[np.asarray(triangles, dtype=np.int64)] if triangles else np.empty((0, 3, 3))
            triangle_uv_min = triangle_coords[:, :, :2].min(axis=1) - 1e-6
            triangle_uv_max = triangle_coords[:, :, :2].max(axis=1) + 1e-6

            grid_steps = []
            for grid_axis in grid_axes:
                if len(grid_axis) > 1:
                    grid_steps.append(float(np.min(np.diff(np.sort(grid_axis)))))
            cell_radius = max(grid_steps, default=0.002) * 1.25
            surface_clearance = 0.0005

            projected_coords = flat_coords.copy()
            for point_idx, flat_coord in enumerate(flat_coords):
                point_uv = flat_coord[:2]
                surface_height = None
                candidate_mask = np.all((point_uv >= triangle_uv_min) & (point_uv <= triangle_uv_max), axis=1)
                candidate_triangles = triangle_coords[candidate_mask]
                if candidate_triangles.size:
                    point_a = candidate_triangles[:, 0]
                    edge_ab = candidate_triangles[:, 1, :2] - point_a[:, :2]
                    edge_ac = candidate_triangles[:, 2, :2] - point_a[:, :2]
                    point_ap = point_uv - point_a[:, :2]
                    det = edge_ab[:, 0] * edge_ac[:, 1] - edge_ac[:, 0] * edge_ab[:, 1]
                    non_degenerate_mask = np.abs(det) >= 1e-12
                    if np.any(non_degenerate_mask):
                        candidate_triangles = candidate_triangles[non_degenerate_mask]
                        edge_ab = edge_ab[non_degenerate_mask]
                        edge_ac = edge_ac[non_degenerate_mask]
                        point_ap = point_ap[non_degenerate_mask]
                        det = det[non_degenerate_mask]
                        bary_b = (point_ap[:, 0] * edge_ac[:, 1] - edge_ac[:, 0] * point_ap[:, 1]) / det
                        bary_c = (edge_ab[:, 0] * point_ap[:, 1] - point_ap[:, 0] * edge_ab[:, 1]) / det
                        bary_a = 1.0 - bary_b - bary_c
                        inside_mask = (bary_a >= -1e-6) & (bary_b >= -1e-6) & (bary_c >= -1e-6)
                        if np.any(inside_mask):
                            heights = (
                                bary_a[inside_mask] * candidate_triangles[inside_mask, 0, 2]
                                + bary_b[inside_mask] * candidate_triangles[inside_mask, 1, 2]
                                + bary_c[inside_mask] * candidate_triangles[inside_mask, 2, 2]
                            )
                            surface_height = float(np.max(heights))
                if surface_height is None:
                    surface_height = self._nearest_surface_height(point_uv, surface_coords, cell_radius)
                projected_coords[point_idx, 2] = surface_height + surface_clearance

            projected_points_tip = origin_tip + projected_coords @ basis_axes
            return projected_points_tip.astype(np.float32)

        def _initialize_camera_tactile(self):
            super()._initialize_camera_tactile()
            self.get_initial_render()

        def _create_physx_views(self) -> None:
            """Create PhysX views using the fingertip link body, not the elastomer Xform."""

            tip_body_prim = self._find_tip_body_prim()
            tip_body_pattern = tip_body_prim.GetPath().pathString.replace("env_0", "env_*")
            self._elastomer_body_view = self._physics_sim_view.create_rigid_body_view([tip_body_pattern])
            self._elastomer_com_b = self._elastomer_body_view.get_coms().to(self._device).split([3, 4], dim=-1)[0]

            if self.cfg.contact_object_prim_path_expr is None:
                return

            contact_object_mesh, contact_object_rigid_body = self._find_contact_object_components()
            num_query_points = self.cfg.tactile_array_size[0] * self.cfg.tactile_array_size[1]
            mesh_path_pattern = contact_object_mesh.GetPath().pathString.replace("env_0", "env_*")
            self._contact_object_sdf_view = self._physics_sim_view.create_sdf_shape_view(
                mesh_path_pattern, num_query_points
            )

            body_path_pattern = contact_object_rigid_body.GetPath().pathString.replace("env_0", "env_*")
            self._contact_object_body_view = self._physics_sim_view.create_rigid_body_view([body_path_pattern])
            self._contact_object_com_b = self._contact_object_body_view.get_coms().to(self._device).split(
                [3, 4], dim=-1
            )[0]

        def _generate_tactile_points(self, num_divs: list, margin: float, visualize: bool):
            """Generate tactile points that follow the fingertip surface height."""

            elastomer_prim_path = self._parent_prims[0].GetPath().pathString
            tip_body_prim = self._find_tip_body_prim()

            def is_visual_mesh(prim) -> bool:
                return prim.IsA(UsdGeom.Mesh) and not prim.HasAPI(UsdPhysics.CollisionAPI)

            elastomer_mesh_prim = sim_utils.get_first_matching_child_prim(
                elastomer_prim_path,
                predicate=is_visual_mesh,
            )
            if elastomer_mesh_prim is None:
                raise RuntimeError(f"No visual mesh found under elastomer at path: {elastomer_prim_path}")

            usd_mesh = UsdGeom.Mesh(elastomer_mesh_prim)
            points = np.asarray(usd_mesh.GetPointsAttr().Get(), dtype=np.float32)
            mesh_bounds = np.array([points.min(axis=0), points.max(axis=0)], dtype=np.float32)
            elastomer_dims = np.diff(mesh_bounds, axis=0).squeeze()
            slim_axis = int(np.argmin(elastomer_dims))
            axis_idxs = [idx for idx in range(3) if idx != slim_axis]

            grid_axes = []
            for axis_idx, num_steps in zip(axis_idxs, num_divs):
                min_val = float(mesh_bounds[0, axis_idx] + margin)
                max_val = float(mesh_bounds[1, axis_idx] - margin)
                if max_val <= min_val:
                    center = float((mesh_bounds[0, axis_idx] + mesh_bounds[1, axis_idx]) * 0.5)
                    min_val = max_val = center
                grid_axes.append(np.linspace(min_val, max_val, int(num_steps), dtype=np.float32))

            layout_name = validate_tactile_layout_name(
                getattr(self.cfg, "tactile_layout", REGULAR_GRID_LAYOUT)
            )
            finger_name = getattr(self.cfg, "tactile_layout_finger", None)
            if not finger_name:
                finger_name = finger_name_from_tip_body_path(
                    tip_body_prim.GetPath().pathString
                )
            if layout_name == ESTIMATED_OFFICIAL_LAYOUT:
                longitudinal_bounds, lateral_bounds = estimated_official_calibration_bounds_xy(
                    finger_name
                )
                grid_axis_x = np.linspace(
                    *longitudinal_bounds, int(num_divs[0]), dtype=np.float32
                )
                grid_axis_y = np.linspace(
                    *lateral_bounds, int(num_divs[1]), dtype=np.float32
                )
                slot_center_indices = estimated_official_slot_center_indices(
                    finger_name,
                    grid_axis_x,
                    grid_axis_y,
                )
                physical_finger_xy = estimated_official_centers_xy(finger_name)
                physical_plane_xy = estimated_official_sensor_plane_xy(
                    finger_name, physical_finger_xy
                )
                finger_xy = physical_finger_xy[slot_center_indices]
                plane_xy = estimated_official_sensor_plane_xy(finger_name, finger_xy)
            else:
                plane_xy = np.asarray(
                    [(x, y) for x in grid_axes[0] for y in grid_axes[1]],
                    dtype=np.float32,
                )
                finger_xy = plane_xy
            self._tactile_layout_name = layout_name
            self._tactile_layout_finger = finger_name
            self._tactile_plane_xy = torch.tensor(
                finger_xy,
                dtype=torch.float32,
                device=self._device,
            )
            self._tactile_sensor_plane_xy = torch.tensor(
                plane_xy,
                dtype=torch.float32,
                device=self._device,
            )

            if layout_name == ESTIMATED_OFFICIAL_LAYOUT:
                center_points = np.zeros((len(physical_plane_xy), 3), dtype=np.float32)
                center_points[:, axis_idxs[0]] = physical_plane_xy[:, 0]
                center_points[:, axis_idxs[1]] = physical_plane_xy[:, 1]
                center_points[:, slim_axis] = float(
                    (mesh_bounds[0, slim_axis] + mesh_bounds[1, slim_axis]) * 0.5
                )
                center_points = self._transform_points_between_prims(
                    center_points, elastomer_mesh_prim, tip_body_prim
                )
                center_points = self._project_points_to_fingertip_surface(
                    center_points,
                    elastomer_mesh_prim,
                    tip_body_prim,
                    axis_idxs,
                    slim_axis,
                    grid_axes,
                )

                patch_offsets, patch_weights = estimated_official_sensor_patch_quadrature()
                _, mesh_axes_tip = self._basis_between_prims(
                    elastomer_mesh_prim, tip_body_prim
                )
                patch_offsets_tip = (
                    patch_offsets[:, :1] * mesh_axes_tip[axis_idxs[0]][None, :]
                    + patch_offsets[:, 1:] * mesh_axes_tip[axis_idxs[1]][None, :]
                )
                tactile_points = (
                    center_points[:, None, :] + patch_offsets_tip[None, :, :]
                ).reshape(-1, 3)
                expected_points = int(num_divs[0] * num_divs[1])
                if tactile_points.shape[0] > expected_points:
                    raise RuntimeError(
                        f"Circular patch quadrature needs {tactile_points.shape[0]} SDF queries, "
                        f"but only {expected_points} are available."
                    )
                padding = expected_points - tactile_points.shape[0]
                if padding:
                    tactile_points = np.concatenate(
                        (tactile_points, np.repeat(center_points[:1], padding, axis=0)),
                        axis=0,
                    )
                self._tactile_physical_center_pos_local = torch.tensor(
                    center_points, dtype=torch.float32, device=self._device
                )
                self._tactile_slot_center_indices = torch.tensor(
                    slot_center_indices, dtype=torch.long, device=self._device
                )
                self._tactile_patch_weights = torch.tensor(
                    patch_weights, dtype=torch.float32, device=self._device
                )
                self._tactile_patch_sensor_count = len(center_points)
                self._tactile_patch_samples_per_sensor = len(patch_weights)
                self._tactile_patch_radius_m = float(
                    np.linalg.norm(patch_offsets, axis=-1).max()
                )
                print(
                    f"[INFO] TacSL {finger_name}: {len(center_points)} circular sensors, "
                    f"radius={self._tactile_patch_radius_m * 1.0e3:.2f} mm, "
                    f"{len(patch_weights)} SDF samples per sensor within {expected_points} queries.",
                    flush=True,
                )
            else:
                tactile_points = np.zeros((len(plane_xy), 3), dtype=np.float32)
                tactile_points[:, axis_idxs[0]] = plane_xy[:, 0]
                tactile_points[:, axis_idxs[1]] = plane_xy[:, 1]
                tactile_points[:, slim_axis] = float(
                    (mesh_bounds[0, slim_axis] + mesh_bounds[1, slim_axis]) * 0.5
                )
                tactile_points = self._transform_points_between_prims(
                    tactile_points, elastomer_mesh_prim, tip_body_prim
                )
                tactile_points = self._project_points_to_fingertip_surface(
                    tactile_points,
                    elastomer_mesh_prim,
                    tip_body_prim,
                    axis_idxs,
                    slim_axis,
                    grid_axes,
                )
            self._tactile_pos_local = torch.tensor(tactile_points, dtype=torch.float32, device=self._device)
            self.num_tactile_points = self._tactile_pos_local.shape[0]
            expected_points = int(self.cfg.tactile_array_size[0] * self.cfg.tactile_array_size[1])
            if self.num_tactile_points != expected_points:
                raise RuntimeError(
                    f"Number of tactile points does not match expected: {self.num_tactile_points} != {expected_points}"
                )

            mesh_quat_body = self._rotation_quat_between_prims(elastomer_mesh_prim, tip_body_prim)
            tacsl_quat_mesh = math_utils.quat_from_euler_xyz(
                torch.tensor(0.0, device=self._device),
                torch.tensor(0.0, device=self._device),
                torch.tensor(-torch.pi, device=self._device),
            )
            tactile_quat_body = math_utils.quat_mul(mesh_quat_body, tacsl_quat_mesh)
            self._tactile_quat_local = tactile_quat_body.unsqueeze(0).repeat(self.num_tactile_points, 1)

else:

    class Revo3VisuoTactileSensor:
        """Placeholder used only when TacSL contrib sensors are not importable."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "TacSL tactile camera support requires "
                "`isaaclab_contrib.sensors.tacsl_sensor.VisuoTactileSensor`."
            )


@dataclass(frozen=True)
class TactileCameraSettings:
    """Camera settings shared by the five fingertip TacSL sensors."""

    height: int = 320
    width: int = 240
    update_period: float = 0.0


def make_tactile_force_sensor_cfgs(body_path_prefix: str = "") -> dict[str, ContactSensorCfg]:
    """Create one ContactSensorCfg per fingertip DIP link."""

    if body_path_prefix and not body_path_prefix.endswith("/"):
        body_path_prefix = f"{body_path_prefix}/"

    return {
        sensor_name: ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/" + body_path_prefix + body_name,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        )
        for sensor_name, body_name in zip(TACTILE_FORCE_SENSOR_NAMES, TACTILE_DIP_BODIES)
    }


def _load_tacsl_dependencies():
    """Load TacSL classes only when a tactile-camera env is constructed."""

    try:
        from isaaclab.sensors import TiledCameraCfg
        from isaaclab_assets.sensors import GELSIGHT_R15_CFG
        from isaaclab_contrib.sensors.tacsl_sensor import VisuoTactileSensorCfg
    except ImportError as exc:
        raise ImportError(
            "TacSL tactile camera support requires Isaac Lab contrib sensors. "
            "Install/enable the Isaac Lab extension that provides "
            "`isaaclab_contrib.sensors.tacsl_sensor.VisuoTactileSensorCfg` and "
            "`isaaclab_assets.sensors.GELSIGHT_R15_CFG`, plus `isaaclab.sensors.TiledCameraCfg`."
        ) from exc

    if Revo3VisuoTactileSensorCfg is None:
        raise ImportError("Revo3 TacSL config extension is unavailable.")
    return GELSIGHT_R15_CFG, TiledCameraCfg, Revo3VisuoTactileSensorCfg, Revo3VisuoTactileSensor


def make_tacsl_sensor_cfgs(
    *,
    camera_settings: TactileCameraSettings = TactileCameraSettings(),
    enable_camera_tactile: bool = True,
    enable_rgb: bool = False,
    enable_force_field: bool = False,
    debug_vis: bool = False,
    visualize_sdf_closest_pts: bool = False,
    tactile_array_size: tuple[int, int] = (16, 16),
    tactile_layout: str = REGULAR_GRID_LAYOUT,
    body_path_prefix: str = "",
):
    """Create one VisuoTactileSensorCfg per fingertip.

    The initial training path uses depth plus ContactSensor net forces. TacSL force fields
    stay disabled until the object SDF collision setup is verified.
    """

    if body_path_prefix and not body_path_prefix.endswith("/"):
        body_path_prefix = f"{body_path_prefix}/"

    tactile_layout = resolve_tactile_layout_name(tactile_layout)
    gelsight_cfg, tiled_camera_cfg, visuo_tactile_sensor_cfg, sensor_class = _load_tacsl_dependencies()
    sensor_cfgs = {}
    for finger, sensor_name, tip_body in zip(TACTILE_FINGER_ORDER, TACTILE_VIS_SENSOR_NAMES, TACTILE_TIP_BODIES):
        elastomer_path = f"{{ENV_REGEX_NS}}/Robot/{body_path_prefix}{tip_body}/tactile_elastomer"
        camera_cfg = None
        if enable_camera_tactile:
            camera_cfg = tiled_camera_cfg(
                prim_path=f"{elastomer_path}/cam",
                update_period=camera_settings.update_period,
                height=camera_settings.height,
                width=camera_settings.width,
                data_types=["distance_to_image_plane"],
                spawn=None,
            )
        sensor_cfg = visuo_tactile_sensor_cfg(
            class_type=sensor_class,
            prim_path=f"{elastomer_path}/tactile_sensor",
            history_length=0,
            render_cfg=gelsight_cfg,
            enable_camera_tactile=enable_camera_tactile,
            enable_force_field=enable_force_field,
            tactile_array_size=tactile_array_size,
            tactile_margin=0.003,
            contact_object_prim_path_expr="{ENV_REGEX_NS}/Object",
            normal_contact_stiffness=1.0,
            friction_coefficient=2.0,
            tangential_stiffness=0.1,
            camera_cfg=camera_cfg,
            debug_vis=debug_vis,
            visualize_sdf_closest_pts=visualize_sdf_closest_pts,
            tactile_layout=tactile_layout,
            tactile_layout_finger=finger,
        )
        sensor_cfgs[sensor_name] = sensor_cfg

    return sensor_cfgs


@configclass
class Revo3TactileObsCfg(ObsGroup):
    """Shaped tactile observations for debugging, export, and tactile policies."""

    tactile_force_3d = ObsTerm(
        func=mdp.fingers_contact_force_b_3d,
        params={"contact_sensor_names": list(TACTILE_FORCE_SENSOR_NAMES)},
        clip=(-20.0, 20.0),
    )
    tactile_depth: ObsTerm | None = None
    tactile_rgb: ObsTerm | None = None
    tactile_normal_force: ObsTerm | None = None
    tactile_shear_force: ObsTerm | None = None

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = False


@configclass
class Revo3TacslTactileMixinCfg:
    """Reusable TacSL tactile camera mixin for Revo3-based Dexsuite tasks."""

    enable_tacsl_tactile: bool = True
    enable_tactile_camera: bool = True
    enable_tactile_depth: bool = True
    enable_tactile_rgb: bool = False
    enable_tactile_force_field: bool = False
    tactile_debug_vis: bool = False
    tactile_visualize_sdf_closest_pts: bool = False
    tactile_image_height: int = 320
    tactile_image_width: int = 240
    tactile_array_size: tuple[int, int] = (16, 16)
    tactile_layout: str = REGULAR_GRID_LAYOUT
    tactile_body_path_prefix: str = ""
    tactile_vis_sensor_names: tuple[str, ...] = TACTILE_VIS_SENSOR_NAMES

    def __post_init__(self):
        super().__post_init__()
        self._configure_revo3_tacsl_tactile()

    def _configure_revo3_tacsl_tactile(self):
        self.observations.tactile = Revo3TactileObsCfg()

        if not self.enable_tacsl_tactile:
            return

        tactile_sensor_cfgs = make_tacsl_sensor_cfgs(
            camera_settings=TactileCameraSettings(
                height=self.tactile_image_height,
                width=self.tactile_image_width,
            ),
            enable_camera_tactile=self.enable_tactile_camera,
            enable_rgb=self.enable_tactile_rgb,
            enable_force_field=self.enable_tactile_force_field,
            debug_vis=self.tactile_debug_vis,
            visualize_sdf_closest_pts=self.tactile_visualize_sdf_closest_pts,
            tactile_array_size=self.tactile_array_size,
            tactile_layout=self.tactile_layout,
            body_path_prefix=self.tactile_body_path_prefix,
        )
        for sensor_name, sensor_cfg in tactile_sensor_cfgs.items():
            setattr(self.scene, sensor_name, sensor_cfg)

        if self.enable_tactile_camera and self.enable_tactile_depth:
            self.observations.tactile.tactile_depth = ObsTerm(
                func=mdp.tactile_depth_image,
                params={"tactile_sensor_names": list(self.tactile_vis_sensor_names)},
            )
        if self.enable_tactile_camera and self.enable_tactile_rgb:
            self.observations.tactile.tactile_rgb = ObsTerm(
                func=mdp.tactile_rgb_image,
                params={"tactile_sensor_names": list(self.tactile_vis_sensor_names)},
            )
        if self.enable_tactile_force_field:
            self.observations.tactile.tactile_normal_force = ObsTerm(
                func=mdp.tactile_normal_force,
                params={"tactile_sensor_names": list(self.tactile_vis_sensor_names)},
            )
            self.observations.tactile.tactile_shear_force = ObsTerm(
                func=mdp.tactile_shear_force,
                params={"tactile_sensor_names": list(self.tactile_vis_sensor_names)},
            )
