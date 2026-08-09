# Copyright (c) 2026, BrainCo.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""DirectRLEnv for Revo3 right hand screw manipulation (HORA-style).

Ported from dexscrew XHandHora (Isaac Gym VecTask) to Isaac Lab, reusing the
observation/priv-info/proprio-hist interfaces of hora_rotation so the HORA
PPO / ProprioAdapt stack works unchanged.

Observation (141 dims) — 3-frame sliding window, 47 dims/frame:
  [0:21]   joint positions, unscaled to [-1,1], +-0.02 rad noise
  [21:42]  current joint targets (delta-accumulated, clamped to joint limits)
  [42:47]  contact forces on 5 DIP fingertips (smoothed, latency-randomized)

Action (21 dims) — delta position control, task-masked fingers zeroed:
  action in [-1,1] -> target += (1/24)*action -> clamp(joint_limits)
  torque = p_gain*(target - pos) - d_gain*vel
  Optional action_delay in [0,1] (per-env, resampled at reset): within each
  control step's decimation physics substeps, the first delay*decimation
  substeps apply the previous target frame; the rest apply the new frame.

Reward (dexscrew compute_hand_reward):
  rotate:     clip(nut_joint_vel, -4, 4) * scale
  rotate_pen: max(nut_joint_vel - thres, 0) * scale (thres may follow a curriculum)
  pose_diff:  sum((q - q_init)^2, thumb excluded) * scale
  torque:     sum(torque^2) * scale
  work:       (sum(|torque|*|qdot|))^2 * scale
  proximity:  clamp(1 - mean(thumb_dist,index_dist)/thr, 0, 1) * scale
  (dexscrew's pc_z_dist penalty is omitted: the screw base is anchored, so the
   point-cloud z-extent is constant and only adds a constant reward offset.)

Termination (dexscrew check_termination):
  finger_dist: thumb or index fingertip farther than thr from the nut grasp center
  stagnation:  var(nut_joint_pos history) < eps after history_len steps
  no_contact:  zero net contact force on the nut for history_len steps
  screw_limit: nut joint within margin of its upper limit
  timeout:     800 steps @ 20 Hz
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_apply, saturate
from pxr import Gf, UsdGeom

if TYPE_CHECKING:
    from .revo3_hand_screw_env_cfg import Revo3HandScrewEnvCfg

_LOCAL_GROUND_USD = str(
    Path(__file__).resolve().parents[6] / "assets" / "usd" / "ground" / "default_environment.usd"
)


def apply_force_vector_noise(forces: torch.Tensor, noise_frac: float) -> torch.Tensor:
    """Add relative-magnitude force noise with a random 3D direction.

    Keeps the original force vector and adds
    ``noise_frac * |F| * u``, where ``u`` is a random unit vector. Zero-force
    samples stay zero (no spurious contact from noise alone).

    Args:
        forces: Force vectors of shape ``(..., 3)``, any leading batch dims.
        noise_frac: Relative noise magnitude (e.g. ``0.05`` → 5% of ``|F|``).

    Returns:
        Noisy force tensor with the same shape as ``forces``.
    """
    if noise_frac <= 0.0:
        return forces
    mag = torch.linalg.norm(forces, dim=-1, keepdim=True)
    rand_dir = torch.randn_like(forces)
    rand_dir = rand_dir / torch.linalg.norm(rand_dir, dim=-1, keepdim=True).clamp(min=1.0e-8)
    return forces + (float(noise_frac) * mag) * rand_dir


class Revo3HandScrewEnv(DirectRLEnv):
    cfg: Revo3HandScrewEnvCfg

    def __init__(self, cfg: Revo3HandScrewEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.num_hand_dofs = self.hand.num_joints
        self.object_radius_scale_ids = self.object_radius_scale_ids.to(self.device)
        self.object_radius_scales = self.object_radius_scales.to(self.device)

        # Canonical init joint pose from assets.py — pose_diff penalty reference & reset pose
        self.init_joint_pos = torch.zeros((1, self.num_hand_dofs), device=self.device)
        _cfg_pos = self.cfg.robot_cfg.init_state.joint_pos
        if _cfg_pos:
            for _name, _val in _cfg_pos.items():
                if _name in self.hand.joint_names:
                    self.init_joint_pos[0, self.hand.joint_names.index(_name)] = float(_val)

        # buffers for position targets
        self.prev_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)
        # Previous control-step targets held for the delayed portion of physics substeps.
        self.delayed_targets = torch.zeros(
            (self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device
        )
        # Per-env delay fraction in [0, 1] control steps (resampled at reset).
        self.action_delay = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self._physics_substep_idx = 0

        # data buffers (HORA interface)
        self.obs_buf_lag_history = torch.zeros((self.num_envs, 80, self.cfg.observation_space // 3), device=self.device, dtype=torch.float)
        self.at_reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.proprio_hist_buf = torch.zeros((self.num_envs, self.cfg.prop_hist_len, self.cfg.observation_space // 3), device=self.device, dtype=torch.float)
        self.priv_info_buf = torch.zeros((self.num_envs, self.cfg.priv_info_dim), device=self.device, dtype=torch.float)

        # actuated joints
        self.actuated_dof_indices = list()
        for joint_name in cfg.actuated_joint_names:
            self.actuated_dof_indices.append(self.hand.joint_names.index(joint_name))
        self.actuated_dof_indices.sort()

        # action mask (dexscrew masks unused fingers per task)
        self.action_mask = torch.ones(self.num_hand_dofs, device=self.device)
        for joint_name in self.cfg.masked_action_joint_names:
            self.action_mask[self.hand.joint_names.index(joint_name)] = 0.0
        # Park task-disabled fingers at zero instead of retaining the grasp pose.
        self.init_joint_pos.mul_(self.action_mask.unsqueeze(0))

        # pose-diff penalty mask (exclude thumb DOFs, dexscrew behavior)
        self.pose_diff_mask = torch.ones(self.num_hand_dofs, device=self.device)
        for i, joint_name in enumerate(self.hand.joint_names):
            if self.cfg.pose_diff_exclude_substring in joint_name:
                self.pose_diff_mask[i] = 0.0

        # finger bodies (order: thumb, index, middle, ring, little)
        self.finger_bodies = list()
        for body_name in self.cfg.fingertip_body_names:
            self.finger_bodies.append(self.hand.body_names.index(body_name))
        self.num_fingertips = len(self.finger_bodies)
        proximity_finger_ids = tuple(self.cfg.proximity_fingertip_indices)
        if not proximity_finger_ids or any(i < 0 or i >= self.num_fingertips for i in proximity_finger_ids):
            raise ValueError(
                f"proximity_fingertip_indices must contain valid fingertip indices, got {proximity_finger_ids}"
            )
        self.proximity_finger_ids = torch.tensor(
            proximity_finger_ids, dtype=torch.long, device=self.device
        )

        # screw object indices
        self.nut_body_idx = self.object.body_names.index(self.cfg.nut_body_name)
        self.nut_joint_idx = self.object.joint_names.index(self.cfg.nut_joint_name)
        self.nut_ref_offset = torch.tensor(self.cfg.nut_ref_offset, dtype=torch.float, device=self.device).repeat(self.num_envs, 1)
        self.num_object_bodies = len(self.object.body_names)

        # joint limits
        joint_pos_limits = self.hand.root_physx_view.get_dof_limits().to(self.device)
        self.hand_dof_lower_limits = joint_pos_limits[..., 0] * self.cfg.dof_limits_scale
        self.hand_dof_upper_limits = joint_pos_limits[..., 1] * self.cfg.dof_limits_scale

        # Hardcoded PD gains — not reading from URDF/USD baked-in defaults
        ndof = self.num_hand_dofs
        self.p_gain = torch.ones((self.num_envs, ndof), device=self.device) * self.cfg.pgain
        self.d_gain = torch.ones((self.num_envs, ndof), device=self.device) * self.cfg.dgain
        self.torques = torch.zeros((self.num_envs, ndof), device=self.device)

        # nut state buffers
        self.nut_dof_pos_prev = torch.zeros(self.num_envs, device=self.device)
        self.nut_dof_vel_cf = torch.zeros(self.num_envs, device=self.device)
        self.nut_dof_pos_history = torch.zeros((self.num_envs, self.cfg.nut_termination_history_len), device=self.device)
        self.nut_contact_history = torch.zeros((self.num_envs, self.cfg.nut_termination_history_len), device=self.device)
        self.default_nut_ref_pos = torch.zeros((self.num_envs, 3), device=self.device)

        # random force buffer (applied to the nut link)
        self.rb_forces = torch.zeros((self.num_envs, self.num_object_bodies, 3), dtype=torch.float, device=self.device)

        # contact buffers (fingertip tactile)
        self._contact_body_ids = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
        self._contact_body_ids_disable = torch.tensor(self.cfg.disable_tactile_ids, dtype=torch.long)
        self.last_contacts = torch.zeros((self.num_envs, len(self._contact_body_ids)), dtype=torch.float, device=self.device)
        self.object_joint_friction_scale = torch.ones(self.num_envs, dtype=torch.float, device=self.device)
        self.object_joint_friction_torque = torch.full(
            (self.num_envs,),
            float(self.cfg.object_joint_friction_default),
            dtype=torch.float,
            device=self.device,
        )
        self.material_friction = torch.full(
            (self.num_envs,),
            float(self.cfg.randomize_friction_lower),
            dtype=torch.float,
            device=self.device,
        )
        self.material_friction_upper = torch.tensor(
            float(self.cfg.randomize_friction_upper),
            dtype=torch.float,
            device=self.device,
        )

        # Randomize startup physics; material friction is resampled again at reset
        # so its upper bound can follow the training curriculum.
        if self.cfg.randomize_friction:
            self._randomize_material_friction(torch.arange(self.num_envs, device=self.device))
        if self.cfg.randomize_com:
            rand_com = torch.empty([self.num_envs, 3]).uniform_(self.cfg.randomize_com_lower, self.cfg.randomize_com_upper)
            coms = self.object.root_physx_view.get_coms().clone()
            coms[:, self.nut_body_idx, :3] += rand_com
            self.object.root_physx_view.set_coms(coms, torch.arange(self.num_envs, device="cpu"))
            self.priv_info_buf[:, 5:8] = coms[:, self.nut_body_idx, :3].to(self.device)
        if self.cfg.randomize_mass:
            rand_mass = torch.empty(self.num_envs).uniform_(self.cfg.randomize_mass_lower, self.cfg.randomize_mass_upper)
            masses = self.object.root_physx_view.get_masses().clone()
            masses[:, self.nut_body_idx] = rand_mass
            self.object.root_physx_view.set_masses(masses, torch.arange(self.num_envs, device="cpu"))
            self.priv_info_buf[:, 4] = rand_mass.to(self.device)
        if self.cfg.randomize_object_joint_friction:
            self._randomize_object_joint_friction(torch.arange(self.num_envs, device=self.device))
        else:
            self.object_joint_friction_torque = self.set_object_joint_friction_torque(
                torch.full((self.num_envs,), float(self.cfg.object_joint_friction_default))
            )
        self.nut_mass = self.object.root_physx_view.get_masses()[:, self.nut_body_idx].to(self.device)

    # ------------------------------------------------------------------
    # scene
    # ------------------------------------------------------------------

    def _setup_scene(self):
        self.hand = Articulation(self.cfg.robot_cfg)
        self.object = Articulation(self.cfg.object_cfg)
        # ground plane: the screw object stands on the floor (dexscrew layout)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(usd_path=_LOCAL_GROUND_USD))
        self.scene.clone_environments(copy_from_source=False)
        self._initialize_object_radius_randomization()
        # replicate_physics=False cannot use the cloner's automatic env IDs, so
        # filter explicitly while retaining collisions with the shared ground.
        self.scene.filter_collisions(global_prim_paths=["/World/ground"])
        self.scene.articulations["hand"] = self.hand
        self.scene.articulations["object"] = self.object
        # fingertip tactile sensors (filtered against the nut link)
        self._contact_sensor = []
        for id in range(len(self.cfg.contact_sensor)):
            self._contact_sensor.append(ContactSensor(self.cfg.contact_sensor[id]))
            self.scene.sensors[f"contact_sensor_{id}"] = self._contact_sensor[id]
        # net contact force on the nut (for no-contact termination)
        self._nut_contact_sensor = ContactSensor(self.cfg.nut_contact_sensor)
        self.scene.sensors["nut_contact_sensor"] = self._nut_contact_sensor
        # lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _initialize_object_radius_randomization(self) -> None:
        """Assign persistent radius scales and author them into every environment.

        Scale levels are distributed as evenly as possible, randomly permuted,
        and applied only while the parallel scene is initialized. Reset logic
        never resamples or rewrites these values.
        """
        scale_levels = torch.tensor(
            self.cfg.object_radius_scale_levels, dtype=torch.float32, device="cpu"
        )
        num_envs = len(self.scene.env_prim_paths)
        if self.cfg.randomize_object_radius:
            repeats = (num_envs + len(scale_levels) - 1) // len(scale_levels)
            scale_ids = torch.arange(len(scale_levels), dtype=torch.long).repeat(repeats)[:num_envs]
            scale_ids = scale_ids[torch.randperm(num_envs)]
        else:
            nominal_id = int(torch.argmin(torch.abs(scale_levels - 1.0)).item())
            scale_ids = torch.full((num_envs,), nominal_id, dtype=torch.long)

        override_scale = float(getattr(self.cfg, "object_radius_scale_override", 0.0))
        if override_scale > 0.0:
            override_env_index = int(
                getattr(self.cfg, "object_radius_scale_override_env_index", -1)
            )
            if not 0 <= override_env_index < num_envs:
                raise ValueError(
                    "object_radius_scale_override_env_index="
                    f"{override_env_index} outside [0, {num_envs})."
                )
            matching_ids = torch.nonzero(
                torch.isclose(
                    scale_levels,
                    torch.tensor(override_scale, dtype=scale_levels.dtype),
                ),
                as_tuple=False,
            ).flatten()
            if len(matching_ids) == 0:
                configured = ", ".join(f"{float(value):g}" for value in scale_levels.tolist())
                raise ValueError(
                    f"object_radius_scale_override={override_scale:g} is not one of the "
                    f"configured scale levels: {configured}"
                )
            scale_ids[override_env_index] = int(matching_ids[0].item())

        self.object_radius_scale_ids = scale_ids
        self.object_radius_scales = scale_levels[scale_ids]
        for env_index, scale in enumerate(self.object_radius_scales.tolist()):
            self._apply_object_radius_scale_to_env(env_index, float(scale))

        if self.cfg.print_object_radius_scale_ids:
            scale_table = ", ".join(
                f"{scale_id}={float(scale):.1f}"
                for scale_id, scale in enumerate(scale_levels.tolist())
            )
            print(f"[INFO] Object radius scale IDs: {scale_table}", flush=True)
            for env_index, scale_id in enumerate(scale_ids.tolist()):
                print(
                    f"env {env_index} object_radius_scale id: {scale_id} "
                    f"(scale={float(scale_levels[scale_id]):.1f})",
                    flush=True,
                )

    def _apply_object_radius_scale_to_env(self, env_index: int, scale: float) -> None:
        """Scale one rotating prism in XY while preserving its height and axis.

        The visual and original collision roots remain instanceable, so the
        five scale levels reuse the imported mesh data instead of duplicating
        complete object assets.

        Args:
            env_index: Index of the cloned parallel environment.
            scale: Multiplicative radius scale applied to local X and Y.
        """
        body_path = (
            f"{self.scene.env_prim_paths[env_index]}/object/{self.cfg.nut_body_name}"
        )
        body_prim = self.scene.stage.GetPrimAtPath(body_path)
        if not body_prim.IsValid():
            raise RuntimeError(f"Rotating object body not found at '{body_path}'.")

        for child_name in ("visuals", "collisions"):
            geometry_root = body_prim.GetChild(child_name)
            if not geometry_root.IsValid():
                raise RuntimeError(
                    f"Object radius randomization requires '{body_path}/{child_name}'."
                )
            xformable = UsdGeom.Xformable(geometry_root)
            scale_attr = geometry_root.GetAttribute("xformOp:scale:objectRadius")
            if scale_attr.IsValid():
                scale_attr.Set(Gf.Vec3d(scale, scale, 1.0))
            else:
                xformable.AddScaleOp(opSuffix="objectRadius").Set(
                    Gf.Vec3d(scale, scale, 1.0)
                )

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # New control step: policy target updates immediately; physics substeps may
        # still apply delayed_targets for the first action_delay * decimation steps.
        self._physics_substep_idx = 0
        actions = saturate(actions, torch.tensor(-self.cfg.clip_actions), torch.tensor(self.cfg.clip_actions))
        self.actions = actions.clone() * self.action_mask[self.actuated_dof_indices]
        targets = self.prev_targets[:, self.actuated_dof_indices] + self.cfg.action_scale * self.actions
        self.cur_targets[:, self.actuated_dof_indices] = saturate(
            targets,
            self.hand_dof_lower_limits[:, self.actuated_dof_indices],
            self.hand_dof_upper_limits[:, self.actuated_dof_indices],
        )
        self.nut_dof_pos_prev[:] = self.nut_dof_pos

        # random disturbance forces on the nut link (dexscrew update_rigid_body_force)
        if self.cfg.force_scale > 0.0:
            self.rb_forces *= torch.pow(torch.tensor(self.cfg.force_decay, dtype=torch.float32), self.physics_dt / self.cfg.force_decay_interval)
            prob = self.cfg.random_force_prob_scalar
            force_indices = (torch.less(torch.rand(self.num_envs, device=self.device), prob)).nonzero()
            self.rb_forces[force_indices, self.nut_body_idx, :] = torch.randn(
                self.rb_forces[force_indices, self.nut_body_idx, :].shape, device=self.device,
            ) * self.nut_mass[force_indices, None] * self.cfg.force_scale
            self.object.permanent_wrench_composer.set_forces_and_torques(
                forces=self.rb_forces,
                torques=torch.zeros_like(self.rb_forces),
            )

    def _apply_action(self) -> None:
        """Apply PD / position targets for one physics substep.

        Uses ``delayed_targets`` for substeps before ``action_delay * decimation``
        and ``cur_targets`` afterwards (per env). Commits the new target frame only
        after the last substep of the control step.
        """
        self._refresh_lab()
        switch_at = self.action_delay * float(self.cfg.decimation)
        use_delayed = self._physics_substep_idx < switch_at
        applied_targets = torch.where(
            use_delayed.unsqueeze(-1), self.delayed_targets, self.cur_targets
        )
        if self.cfg.torque_control:
            self.torques = (
                self.p_gain * (applied_targets - self.hand_dof_pos)
                - self.d_gain * self.hand_dof_vel
            )
            self.hand.set_joint_effort_target(
                self.torques[:, self.actuated_dof_indices], joint_ids=self.actuated_dof_indices
            )
        else:
            self.hand.set_joint_position_target(
                applied_targets[:, self.actuated_dof_indices], joint_ids=self.actuated_dof_indices
            )

        self._physics_substep_idx += 1
        if self._physics_substep_idx >= self.cfg.decimation:
            # End of control step: advance delta-target chain and delay buffer.
            self.prev_targets[:, self.actuated_dof_indices] = self.cur_targets[
                :, self.actuated_dof_indices
            ]
            self.delayed_targets[:, self.actuated_dof_indices] = self.cur_targets[
                :, self.actuated_dof_indices
            ]

    # ------------------------------------------------------------------
    # observations
    # ------------------------------------------------------------------

    def _get_observations(self) -> dict:
        self._refresh_lab()
        obs = self.compute_observations()
        return {
            "obs":          obs,
            "priv_info":    self.priv_info_buf.clone(),
            "proprio_hist": self.proprio_hist_buf.clone(),
        }

    def compute_observations(self):
        # fingertip contact forces with smoothing + latency (same as hora_rotation)
        # net_forces_w_history: (E, hist, 3) per sensor -> (E, hist, 5, 3)
        net_contact_forces_history = torch.cat(
            [self._contact_sensor[id].data.net_forces_w_history[:, :, 0, :].unsqueeze(2) for id in self._contact_body_ids], dim=2)
        # Smooth in 3D (W) so optional magnitude/direction noise can run before the norm.
        smooth_forces_w = (
            net_contact_forces_history[:, 0, :] * self.cfg.contact_smooth
            + net_contact_forces_history[:, 1, :] * (1.0 - self.cfg.contact_smooth)
        )
        smooth_forces_w[:, self._contact_body_ids_disable] = 0.0
        if self.cfg.enable_contact_noise and not self.cfg.binary_contact:
            smooth_forces_w = apply_force_vector_noise(
                smooth_forces_w, self.cfg.contact_force_noise_frac
            )
        smooth_contact_forces = torch.linalg.norm(smooth_forces_w, dim=-1)
        if self.cfg.binary_contact:
            binary_contacts = torch.where(smooth_contact_forces > self.cfg.contact_threshold, 1.0, 0.0)
            latency_samples = torch.rand_like(self.last_contacts)
            latency = torch.where(latency_samples < self.cfg.contact_latency, 1.0, 0.0)
            self.last_contacts = self.last_contacts * latency + binary_contacts * (1 - latency)
            mask = torch.rand_like(self.last_contacts)
            mask = torch.where(mask < self.cfg.contact_sensor_noise, 0.0, 1.0)
            sensed_contacts = torch.where(self.last_contacts > 0.1, mask * self.last_contacts, self.last_contacts)
        else:
            latency_samples = torch.rand_like(self.last_contacts)
            latency = torch.where(latency_samples < self.cfg.contact_latency, 1.0, 0.0)
            self.last_contacts = self.last_contacts * latency + smooth_contact_forces * (1 - latency)
            sensed_contacts = self.last_contacts.clone()

        if not self.cfg.enable_tactile:
            sensed_contacts[:] = 0.0

        # sliding window of (joint_pos, targets, contacts)
        prev_obs_buf = self.obs_buf_lag_history[:, 1:].clone()
        joint_noise_matrix = (torch.rand(self.hand_dof_pos.shape, device=self.device) * 2.0 - 1.0) * self.cfg.joint_noise_scale
        cur_obs_buf = unscale(
            joint_noise_matrix + self.hand_dof_pos,
            self.hand_dof_lower_limits,
            self.hand_dof_upper_limits
        ).clone().unsqueeze(1)
        cur_tar_buf = self.cur_targets[:, None]
        cur_obs_buf = torch.cat([cur_obs_buf, cur_tar_buf], dim=-1)
        cur_obs_buf = torch.cat([cur_obs_buf, sensed_contacts.clone().unsqueeze(1)], dim=-1)
        self.obs_buf_lag_history[:] = torch.cat([prev_obs_buf, cur_obs_buf], dim=1)

        # refill buffers for envs that were just reset
        at_reset_env_ids = self.at_reset_buf.nonzero(as_tuple=False).squeeze(-1)
        ndof = self.num_hand_dofs
        self.obs_buf_lag_history[at_reset_env_ids, :, 0:ndof] = unscale(
            self.hand_dof_pos[at_reset_env_ids],
            self.hand_dof_lower_limits[at_reset_env_ids],
            self.hand_dof_upper_limits[at_reset_env_ids],
        ).clone().unsqueeze(1)
        self.obs_buf_lag_history[at_reset_env_ids, :, ndof:ndof*2] = self.hand_dof_pos[at_reset_env_ids].unsqueeze(1)
        self.obs_buf_lag_history[at_reset_env_ids, :, ndof*2:ndof*2+5] = sensed_contacts[at_reset_env_ids].unsqueeze(1)
        obs_buf = (self.obs_buf_lag_history[:, -3:].reshape(self.num_envs, -1)).clone()

        # Stage2: zero contacts in actor obs, proprio_hist retains real contact history
        if not self.cfg.enable_contact_in_obs:
            obs_single = ndof * 2 + 5
            for f in range(3):
                obs_buf[:, f * obs_single + ndof * 2:f * obs_single + ndof * 2 + 5] = 0.0

        self.proprio_hist_buf[:] = self.obs_buf_lag_history[:, -self.cfg.prop_hist_len:].clone()

        # nut termination histories (dexscrew updates them in compute_observations)
        nut_contact_force = torch.norm(self._nut_contact_sensor.data.net_forces_w[:, 0, :], dim=-1)
        self.nut_dof_pos_history[:] = torch.cat(
            [self.nut_dof_pos_history[:, 1:], self.nut_dof_pos.unsqueeze(1)], dim=1)
        self.nut_contact_history[:] = torch.cat(
            [self.nut_contact_history[:, 1:], nut_contact_force.unsqueeze(1)], dim=1)
        if len(at_reset_env_ids) > 0:
            self.nut_dof_pos_history[at_reset_env_ids] = self.nut_dof_pos[at_reset_env_ids].unsqueeze(1).repeat(1, self.cfg.nut_termination_history_len)
            self.nut_contact_history[at_reset_env_ids] = nut_contact_force[at_reset_env_ids].unsqueeze(1).repeat(1, self.cfg.nut_termination_history_len)
            self.nut_dof_pos_prev[at_reset_env_ids] = self.nut_dof_pos[at_reset_env_ids]
            self.default_nut_ref_pos[at_reset_env_ids] = self.nut_ref_pos[at_reset_env_ids]
        self.at_reset_buf[at_reset_env_ids] = 0

        # privileged info
        self.priv_info_buf[:, 0:3] = self.nut_ref_pos - self.default_nut_ref_pos
        self.priv_info_buf[:, 8] = torch.sin(self.nut_dof_pos)
        self.priv_info_buf[:, 9] = torch.cos(self.nut_dof_pos)
        self.priv_info_buf[:, 10] = torch.clamp(self.nut_dof_vel_cf, -10.0, 10.0) / 10.0

        return obs_buf

    # ------------------------------------------------------------------
    # rewards
    # ------------------------------------------------------------------

    def _get_current_angvel_penalty_threshold(self) -> float:
        thres = self.cfg.angvel_penalty_threshold
        if isinstance(thres, (int, float)):
            return float(thres)
        init_t, final_t, curr_start, curr_end = thres
        agent_steps = self.common_step_counter * self.num_envs
        if curr_end > 0:
            progress = (agent_steps - curr_start) / (curr_end - curr_start)
            progress = min(max(progress, 0.0), 1.0)
            progress = round(progress * 20) / 20
        else:
            progress = 1.0
        return init_t + (final_t - init_t) * progress

    def _get_rewards(self) -> torch.Tensor:
        self._refresh_lab()
        # nut joint velocity at control frequency (finite difference, dexscrew style)
        nut_dof_linvel = (self.nut_dof_pos - self.nut_dof_pos_prev) / self.step_dt
        self.nut_dof_vel_cf = nut_dof_linvel

        rotate_reward = torch.clip(nut_dof_linvel, max=self.cfg.angvel_clip_max, min=self.cfg.angvel_clip_min)
        current_thres = self._get_current_angvel_penalty_threshold()
        rotate_penalty = torch.where(nut_dof_linvel > current_thres, nut_dof_linvel - current_thres, torch.zeros_like(nut_dof_linvel))

        pose_diff_penalty = (
            ((self.hand_dof_pos[:, self.actuated_dof_indices] - self.init_joint_pos[:, self.actuated_dof_indices]) ** 2)
            * self.pose_diff_mask[self.actuated_dof_indices]
        ).sum(-1)
        torque_penalty = (self.torques[:, self.actuated_dof_indices] ** 2).sum(-1)
        work_penalty = ((torch.abs(self.torques[:, self.actuated_dof_indices]) * torch.abs(self.hand_dof_vel[:, self.actuated_dof_indices])).sum(-1)) ** 2

        # Proximity to the grasp center. dexscrew uses thumb/index; task variants
        # may include more fingertips without changing the reward formulation.
        thumb_dist = torch.norm(self.fingertip_pos[:, 0] - self.nut_ref_pos, dim=-1)
        index_dist = torch.norm(self.fingertip_pos[:, 1] - self.nut_ref_pos, dim=-1)
        proximity_fingertip_pos = torch.index_select(
            self.fingertip_pos, dim=1, index=self.proximity_finger_ids
        )
        mean_dist = torch.norm(
            proximity_fingertip_pos - self.nut_ref_pos.unsqueeze(1), dim=-1
        ).mean(dim=1)
        proximity_reward = torch.clamp(1.0 - mean_dist / self.cfg.reset_dist_threshold, min=0.0, max=1.0)

        total_reward = (
            rotate_reward * self.cfg.rotate_reward_scale
            + rotate_penalty * self.cfg.rotate_penalty_scale
            + pose_diff_penalty * self.cfg.pose_diff_penalty_scale
            + torque_penalty * self.cfg.torque_penalty_scale
            + work_penalty * self.cfg.work_penalty_scale
            + proximity_reward * self.cfg.proximity_reward_scale
        )

        self.extras["rotation_reward"] = rotate_reward.mean()
        self.extras["rotate_penalty"] = rotate_penalty.mean()
        self.extras["pose_diff_penalty"] = pose_diff_penalty.mean()
        self.extras["torque_penalty"] = torque_penalty.mean()
        self.extras["work_penalty"] = work_penalty.mean()
        self.extras["proximity_reward"] = proximity_reward.mean()
        self.extras["curriculum/angvel_penalty_threshold"] = torch.tensor(current_thres)
        self.extras["screw/angular_velocity"] = nut_dof_linvel.mean()
        self.extras["metrics/angular_velocity_per_env"] = nut_dof_linvel.detach()
        self.extras["screw/angular_position"] = self.nut_dof_pos.mean()
        self.extras["screw/positive_vel_ratio"] = (nut_dof_linvel > 0).float().mean()
        self.extras["screw/thumb_nut_dist"] = thumb_dist.mean()
        self.extras["screw/index_nut_dist"] = index_dist.mean()
        self.extras["screw/proximity_finger_dist"] = mean_dist.mean()
        self.extras["randomization/friction_mean"] = self.material_friction.mean()
        self.extras["randomization/friction_upper"] = self.material_friction_upper
        self.extras["randomization/object_joint_friction_scale"] = self.object_joint_friction_scale.mean()
        self.extras["randomization/object_joint_friction_torque"] = self.object_joint_friction_torque.mean()
        scale_lower, scale_upper = self._current_object_joint_friction_scale_bounds()
        self.extras["curriculum/domain_randomization_progress"] = torch.tensor(
            self._domain_randomization_curriculum_progress()
        )
        self.extras["curriculum/object_joint_friction_scale_lower"] = torch.tensor(scale_lower)
        self.extras["curriculum/object_joint_friction_scale_upper"] = torch.tensor(scale_upper)
        self.extras["total_reward"] = total_reward.mean()
        return total_reward

    # ------------------------------------------------------------------
    # termination / reset
    # ------------------------------------------------------------------

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._refresh_lab()
        time_out = self.episode_length_buf >= self.max_episode_length

        if self.cfg.enable_finger_dist_reset:
            thumb_dist = torch.norm(self.fingertip_pos[:, 0] - self.nut_ref_pos, dim=-1)
            index_dist = torch.norm(self.fingertip_pos[:, 1] - self.nut_ref_pos, dim=-1)
            finger_dist_reset = (thumb_dist > self.cfg.reset_dist_threshold) | (index_dist > self.cfg.reset_dist_threshold)
        else:
            finger_dist_reset = torch.zeros_like(time_out)

        hist_filled = self.episode_length_buf >= self.cfg.nut_termination_history_len
        nut_pos_var = torch.var(self.nut_dof_pos_history, dim=1)
        stagnation_reset = (nut_pos_var < self.cfg.nut_stagnation_eps) & hist_filled
        no_contact_reset = torch.all(self.nut_contact_history <= 1e-3, dim=1) & hist_filled

        screw_at_limit = self.nut_dof_pos > (self.cfg.screw_upper_limit - self.cfg.screw_limit_margin)

        terminated = finger_dist_reset | stagnation_reset | no_contact_reset | screw_at_limit

        self.extras["reset/finger_dist"] = finger_dist_reset.float().mean()
        self.extras["reset/nut_stagnation"] = stagnation_reset.float().mean()
        self.extras["reset/nut_no_contact"] = no_contact_reset.float().mean()
        self.extras["reset/screw_at_limit"] = screw_at_limit.float().mean()
        self.extras["reset/time_out"] = time_out.float().mean()
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.hand._ALL_INDICES
        super()._reset_idx(env_ids)

        # pd randomize: dexscrew-style absolute per-DOF uniform ranges
        if self.cfg.randomize_pd_gains:
            self.p_gain[env_ids] = torch.empty((len(env_ids), self.num_hand_dofs), device=self.device).uniform_(
                self.cfg.randomize_p_gain_lower, self.cfg.randomize_p_gain_upper)
            self.d_gain[env_ids] = torch.empty((len(env_ids), self.num_hand_dofs), device=self.device).uniform_(
                self.cfg.randomize_d_gain_lower, self.cfg.randomize_d_gain_upper)
        if self.cfg.randomize_friction:
            self._randomize_material_friction(env_ids)
        if self.cfg.randomize_object_joint_friction:
            self._randomize_object_joint_friction(env_ids)

        # reset screw object: root at default pose (+ optional XY noise), nut joint back to zero
        object_default_state = self.object.data.default_root_state.clone()[env_ids]
        object_default_state[:, 0:3] += self.scene.env_origins[env_ids]
        if self.cfg.randomize_object_xy_position and self.cfg.object_xy_position_noise > 0.0:
            xy_noise = float(self.cfg.object_xy_position_noise)
            xy_offset = torch.empty((len(env_ids), 2), device=self.device).uniform_(-xy_noise, xy_noise)
            object_default_state[:, 0:2] += xy_offset
            self.extras["randomization/object_xy_offset_norm"] = xy_offset.norm(dim=-1).mean()
        self.object.write_root_pose_to_sim(object_default_state[:, :7], env_ids)
        self.object.write_root_velocity_to_sim(torch.zeros_like(object_default_state[:, 7:]), env_ids)
        nut_zero = torch.zeros((len(env_ids), self.object.num_joints), device=self.device)
        self.object.write_joint_state_to_sim(nut_zero, torch.zeros_like(nut_zero), env_ids=env_ids)
        self.rb_forces[env_ids] = 0.0

        # reset hand: root at default pose, joints = init pose + uniform noise
        hand_default_state = self.hand.data.default_root_state.clone()[env_ids]
        hand_default_state[:, 0:3] += self.scene.env_origins[env_ids]
        self.hand.write_root_state_to_sim(hand_default_state, env_ids)
        half_range = (self.hand_dof_upper_limits[env_ids] - self.hand_dof_lower_limits[env_ids]) / 2.0
        joint_noise = (torch.rand((len(env_ids), self.num_hand_dofs), device=self.device) * 2.0 - 1.0) \
            * self.cfg.reset_joint_noise_frac * half_range
        joint_noise.mul_(self.action_mask.unsqueeze(0))
        dof_pos = saturate(
            self.init_joint_pos.expand(len(env_ids), -1) + joint_noise,
            self.hand_dof_lower_limits[env_ids],
            self.hand_dof_upper_limits[env_ids],
        )
        dof_vel = torch.zeros_like(self.hand.data.default_joint_vel[env_ids])
        # targets hold the canonical init pose (dexscrew: prev_targets = noiseless pose)
        init_pose = self.init_joint_pos.expand(len(env_ids), -1)
        self.prev_targets[env_ids] = init_pose
        self.cur_targets[env_ids] = init_pose
        self.delayed_targets[env_ids] = init_pose
        delay_low, delay_high = self.cfg.action_delay
        self.action_delay[env_ids] = torch.empty(
            len(env_ids), device=self.device, dtype=torch.float
        ).uniform_(delay_low, delay_high)
        self.hand.set_joint_position_target(dof_pos, env_ids=env_ids)
        self.hand.write_joint_state_to_sim(dof_pos, dof_vel, env_ids=env_ids)
        self._refresh_lab()

        # reset data buffers
        self.last_contacts[env_ids] = 0
        self.proprio_hist_buf[env_ids] = 0
        self.at_reset_buf[env_ids] = 1
        self.extras["randomization/action_delay_mean"] = self.action_delay.mean()

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _refresh_lab(self):
        self.fingertip_pos = self.hand.data.body_pos_w[:, self.finger_bodies]
        self.fingertip_pos = self.fingertip_pos - self.scene.env_origins.repeat((1, self.num_fingertips)).reshape(self.num_envs, self.num_fingertips, 3)
        self.hand_dof_pos = self.hand.data.joint_pos
        self.hand_dof_vel = self.hand.data.joint_vel

        self.nut_dof_pos = self.object.data.joint_pos[:, self.nut_joint_idx]
        self.nut_dof_vel = self.object.data.joint_vel[:, self.nut_joint_idx]
        nut_body_pos = self.object.data.body_pos_w[:, self.nut_body_idx] - self.scene.env_origins
        nut_body_quat = self.object.data.body_quat_w[:, self.nut_body_idx]
        self.nut_ref_pos = nut_body_pos + quat_apply(nut_body_quat, self.nut_ref_offset)

    def _domain_randomization_curriculum_progress(self) -> float:
        """Return domain-randomization curriculum progress in ``[0, 1]``.

        Returns:
            Interpolation progress from global agent steps. When curriculum is
            disabled, or ``end <= start``, returns ``1.0`` (final difficulty).
        """
        if not bool(getattr(self.cfg, "domain_randomization_curriculum_enable", False)):
            return 1.0
        start = int(getattr(self.cfg, "domain_randomization_curriculum_start", 0))
        end = int(getattr(self.cfg, "domain_randomization_curriculum_end", 0))
        if end <= start:
            return 1.0
        agent_steps = self.common_step_counter * self.num_envs
        progress = (agent_steps - start) / float(end - start)
        return min(max(progress, 0.0), 1.0)

    def _current_material_friction_upper(self) -> float:
        upper = float(self.cfg.randomize_friction_upper)
        if not bool(getattr(self.cfg, "domain_randomization_curriculum_enable", False)):
            return upper
        initial_upper = float(getattr(self.cfg, "randomize_friction_initial_upper", upper))
        progress = self._domain_randomization_curriculum_progress()
        return initial_upper + (upper - initial_upper) * progress

    def _current_object_joint_friction_scale_bounds(self) -> tuple[float, float]:
        """Return the current passive-joint friction scale sampling bounds.

        Returns:
            ``(lower, upper)`` scale multipliers applied to
            ``object_joint_friction_default``.
        """
        final_lower = float(self.cfg.object_joint_friction_scale_lower)
        final_upper = float(self.cfg.object_joint_friction_scale_upper)
        if not bool(getattr(self.cfg, "domain_randomization_curriculum_enable", False)):
            return final_lower, final_upper
        initial_lower = float(self.cfg.object_joint_friction_initial_scale_lower)
        initial_upper = float(self.cfg.object_joint_friction_initial_scale_upper)
        progress = self._domain_randomization_curriculum_progress()
        scale_lower = initial_lower + (final_lower - initial_lower) * progress
        scale_upper = initial_upper + (final_upper - initial_upper) * progress
        if scale_upper < scale_lower:
            scale_upper = scale_lower
        return scale_lower, scale_upper

    def _randomize_object_joint_friction(self, env_ids: Sequence[int] | torch.Tensor) -> None:
        """Sample passive object joint friction torque for the given environments.

        Args:
            env_ids: Environment indices to resample.
        """
        env_ids_t = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).reshape(-1)
        if len(env_ids_t) == 0:
            return
        scale_lower, scale_upper = self._current_object_joint_friction_scale_bounds()
        rand_joint_friction_scale = torch.empty(len(env_ids_t), device=self.device).uniform_(
            scale_lower,
            scale_upper,
        )
        self.object_joint_friction_scale[env_ids_t] = rand_joint_friction_scale
        self.object_joint_friction_torque[env_ids_t] = (
            float(self.cfg.object_joint_friction_default) * rand_joint_friction_scale
        )
        self.set_object_joint_friction_torque(self.object_joint_friction_torque)

    def _randomize_material_friction(self, env_ids: Sequence[int] | torch.Tensor) -> None:
        env_ids_t = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).reshape(-1)
        if len(env_ids_t) == 0:
            return
        upper = self._current_material_friction_upper()
        lower = float(self.cfg.randomize_friction_lower)
        if upper < lower:
            raise ValueError(
                f"randomize_friction upper ({upper}) must be >= lower ({lower})"
            )
        rand_friction = torch.empty(len(env_ids_t), device=self.device).uniform_(lower, upper)
        self.material_friction[env_ids_t] = rand_friction
        self.material_friction_upper.fill_(upper)

        n_obj_mats = self.object.root_physx_view.get_material_properties().shape[1]
        self.set_friction(self.object, rand_friction.unsqueeze(1).repeat(1, n_obj_mats), env_ids_t)
        n_hand_mats = self.hand.root_physx_view.get_material_properties().shape[1]
        self.set_friction(self.hand, rand_friction.unsqueeze(1).repeat(1, n_hand_mats), env_ids_t)
        self.priv_info_buf[env_ids_t, 3] = rand_friction

    def set_friction(self, asset, value, env_ids: Sequence[int] | torch.Tensor | int):
        materials = asset.root_physx_view.get_material_properties()
        if isinstance(env_ids, int):
            env_ids_t = torch.arange(env_ids, device=self.device)
        else:
            env_ids_t = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).reshape(-1)
        value_t = torch.as_tensor(value, dtype=materials.dtype, device=materials.device)
        material_env_ids = env_ids_t.to(device=materials.device)
        materials[material_env_ids, :, 0] = value_t  # static friction
        materials[material_env_ids, :, 1] = value_t  # dynamic friction
        asset.root_physx_view.set_material_properties(materials, env_ids_t.cpu())

    def set_object_joint_friction_torque(self, value: torch.Tensor) -> torch.Tensor:
        values = torch.as_tensor(value, dtype=torch.float).reshape(self.num_envs)
        env_ids = torch.arange(self.num_envs, device="cpu")
        root_view = self.object.root_physx_view
        if hasattr(root_view, "get_dof_friction_properties") and hasattr(root_view, "set_dof_friction_properties"):
            friction_props = root_view.get_dof_friction_properties().clone()
            joint_values = values.to(device=friction_props.device, dtype=friction_props.dtype)
            friction_props[:, self.nut_joint_idx, 0] = joint_values  # static friction effort
            friction_props[:, self.nut_joint_idx, 1] = joint_values  # dynamic friction effort
            root_view.set_dof_friction_properties(friction_props, env_ids)
            return joint_values.to(self.device)

        friction_coeffs = root_view.get_dof_friction_coefficients().clone()
        joint_values = values.to(device=friction_coeffs.device, dtype=friction_coeffs.dtype)
        friction_coeffs[:, self.nut_joint_idx] = joint_values
        root_view.set_dof_friction_coefficients(friction_coeffs, env_ids)
        return joint_values.to(self.device)


@torch.jit.script
def unscale(x, lower, upper):
    return (2.0 * x - upper - lower) / (upper - lower)
