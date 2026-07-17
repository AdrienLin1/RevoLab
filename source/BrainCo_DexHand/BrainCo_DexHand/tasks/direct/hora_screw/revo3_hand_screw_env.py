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

if TYPE_CHECKING:
    from .revo3_hand_screw_env_cfg import Revo3HandScrewEnvCfg

_LOCAL_GROUND_USD = str(
    Path(__file__).resolve().parents[6] / "assets" / "usd" / "ground" / "default_environment.usd"
)


class Revo3HandScrewEnv(DirectRLEnv):
    cfg: Revo3HandScrewEnvCfg

    def __init__(self, cfg: Revo3HandScrewEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.num_hand_dofs = self.hand.num_joints

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

        # randomize physics params (once at startup, like dexscrew per-env creation)
        if self.cfg.randomize_friction:
            # dexscrew: hand and object share the SAME per-env friction ~ U(0.5, 8.0)
            rand_friction = torch.empty(self.num_envs).uniform_(self.cfg.randomize_friction_lower, self.cfg.randomize_friction_upper)
            rand_friction = rand_friction.reshape(self.num_envs, 1)
            n_obj_mats = self.object.root_physx_view.get_material_properties().shape[1]
            self.set_friction(self.object, rand_friction.clone().repeat(1, n_obj_mats), self.num_envs)
            n_hand_mats = self.hand.root_physx_view.get_material_properties().shape[1]
            self.set_friction(self.hand, rand_friction.clone().repeat(1, n_hand_mats), self.num_envs)
            self.priv_info_buf[:, 3] = rand_friction.squeeze().to(self.device)
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

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
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
        self._refresh_lab()
        if self.cfg.torque_control:
            self.torques = self.p_gain * (self.cur_targets - self.hand_dof_pos) - self.d_gain * self.hand_dof_vel
            self.hand.set_joint_effort_target(self.torques[:, self.actuated_dof_indices], joint_ids=self.actuated_dof_indices)
        else:
            self.hand.set_joint_position_target(self.cur_targets[:, self.actuated_dof_indices], joint_ids=self.actuated_dof_indices)
        self.prev_targets[:, self.actuated_dof_indices] = self.cur_targets[:, self.actuated_dof_indices]

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
        net_contact_forces_history = torch.cat(
            [self._contact_sensor[id].data.net_forces_w_history[:, :, 0, :].unsqueeze(2) for id in self._contact_body_ids], dim=2)
        norm_contact_forces_history = torch.norm(net_contact_forces_history, dim=-1)
        smooth_contact_forces = norm_contact_forces_history[:, 0, :] * self.cfg.contact_smooth + norm_contact_forces_history[:, 1, :] * (1 - self.cfg.contact_smooth)
        smooth_contact_forces[:, self._contact_body_ids_disable] = 0.0
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

        # proximity of thumb & index fingertips to the nut grasp center
        thumb_dist = torch.norm(self.fingertip_pos[:, 0] - self.nut_ref_pos, dim=-1)
        index_dist = torch.norm(self.fingertip_pos[:, 1] - self.nut_ref_pos, dim=-1)
        mean_dist = 0.5 * (thumb_dist + index_dist)
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
        self.extras["screw/angular_position"] = self.nut_dof_pos.mean()
        self.extras["screw/positive_vel_ratio"] = (nut_dof_linvel > 0).float().mean()
        self.extras["screw/thumb_nut_dist"] = thumb_dist.mean()
        self.extras["screw/index_nut_dist"] = index_dist.mean()
        self.extras["total_reward"] = total_reward.mean()
        return total_reward

    # ------------------------------------------------------------------
    # termination / reset
    # ------------------------------------------------------------------

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._refresh_lab()
        time_out = self.episode_length_buf >= self.max_episode_length

        thumb_dist = torch.norm(self.fingertip_pos[:, 0] - self.nut_ref_pos, dim=-1)
        index_dist = torch.norm(self.fingertip_pos[:, 1] - self.nut_ref_pos, dim=-1)
        finger_dist_reset = (thumb_dist > self.cfg.reset_dist_threshold) | (index_dist > self.cfg.reset_dist_threshold)

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

        # reset screw object: root at default pose, nut joint back to zero
        object_default_state = self.object.data.default_root_state.clone()[env_ids]
        object_default_state[:, 0:3] += self.scene.env_origins[env_ids]
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
        dof_pos = saturate(
            self.init_joint_pos.expand(len(env_ids), -1) + joint_noise,
            self.hand_dof_lower_limits[env_ids],
            self.hand_dof_upper_limits[env_ids],
        )
        dof_vel = torch.zeros_like(self.hand.data.default_joint_vel[env_ids])
        # targets hold the canonical init pose (dexscrew: prev_targets = noiseless pose)
        self.prev_targets[env_ids] = self.init_joint_pos.expand(len(env_ids), -1)
        self.cur_targets[env_ids] = self.init_joint_pos.expand(len(env_ids), -1)
        self.hand.set_joint_position_target(dof_pos, env_ids=env_ids)
        self.hand.write_joint_state_to_sim(dof_pos, dof_vel, env_ids=env_ids)
        self._refresh_lab()

        # reset data buffers
        self.last_contacts[env_ids] = 0
        self.proprio_hist_buf[env_ids] = 0
        self.at_reset_buf[env_ids] = 1

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

    def set_friction(self, asset, value, num_envs):
        materials = asset.root_physx_view.get_material_properties()
        materials[..., 0] = value  # static friction
        materials[..., 1] = value  # dynamic friction
        env_ids = torch.arange(num_envs, device="cpu")
        asset.root_physx_view.set_material_properties(materials, env_ids)


@torch.jit.script
def unscale(x, lower, upper):
    return (2.0 * x - upper - lower) / (upper - lower)
