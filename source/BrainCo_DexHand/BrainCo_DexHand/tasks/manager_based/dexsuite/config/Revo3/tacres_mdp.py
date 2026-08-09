# Copyright (c) 2026, BrainCo.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""TacRes-specific MDP terms: tactile features, force-impulse perturbations, and privileged obs.

Implements the tactile preprocessing pipeline and the perturbation curriculum described in
``FINAL_PROPOSAL_20260705_1547.md`` on top of the existing Revo3 fingertip contact sensors.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _resolve_env_ids(env: ManagerBasedRLEnv, env_ids) -> torch.Tensor:
    if env_ids is None or env_ids == slice(None):
        return torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    if not torch.is_tensor(env_ids):
        return torch.tensor(env_ids, device=env.device, dtype=torch.long)
    return env_ids.to(device=env.device, dtype=torch.long)


def _sensor_forces_w(env: ManagerBasedRLEnv, contact_sensor_names: list[str]) -> torch.Tensor:
    """Object-filtered contact forces of the fingertip sensors, stacked as (num_envs, num_sensors, 3)."""
    forces = [env.scene.sensors[name].data.force_matrix_w.view(env.num_envs, 3) for name in contact_sensor_names]
    return torch.stack(forces, dim=1)


class tactile_finger_features(ManagerTermBase):
    """Per-fingertip 5-dim tactile features with EMA smoothing (proposal preprocessing steps a-d).

    Per finger: ``[log1p(|f|), f_n/(|f|+eps), |f_t|/(|f_n|+eps), contact, contact_change]`` where the
    force is EMA-smoothed and expressed in the fingertip link frame. Output shape ``(num_envs, 5 * num_fingers)``.
    The tactile history tau_hist (K frames) is realized through the observation group's ``history_length``.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        sensor_names = cfg.params["contact_sensor_names"]
        body_names = cfg.params["body_names"]
        if len(sensor_names) != len(body_names):
            raise ValueError("contact_sensor_names and body_names must align one-to-one.")
        robot: Articulation = env.scene[cfg.params.get("asset_cfg", SceneEntityCfg("robot")).name]
        self._body_ids, _ = robot.find_bodies(list(body_names), preserve_order=True)
        self._robot = robot
        self._num_fingers = len(sensor_names)
        self._ema = torch.zeros(env.num_envs, self._num_fingers, 3, device=env.device)
        self._prev_contact = torch.zeros(env.num_envs, self._num_fingers, device=env.device)

    def reset(self, env_ids=None):
        ids = _resolve_env_ids(self._env, env_ids)
        self._ema[ids] = 0.0
        self._prev_contact[ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        contact_sensor_names: list[str],
        body_names: list[str],
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        ema_alpha: float = 0.8,
        contact_threshold: float = 1.0,
        ratio_clip: float = 10.0,
        eps: float = 1.0e-6,
    ) -> torch.Tensor:
        forces_w = _sensor_forces_w(env, contact_sensor_names)
        # rotate into the fingertip link frames (normal axis assumed local z)
        body_quat_w = self._robot.data.body_quat_w[:, self._body_ids]
        forces_l = quat_apply_inverse(body_quat_w, forces_w)
        # EMA smoothing
        self._ema = ema_alpha * self._ema + (1.0 - ema_alpha) * forces_l
        f = self._ema
        f_norm = torch.linalg.norm(f, dim=-1)
        f_n = f[..., 2]
        f_t = torch.linalg.norm(f[..., :2], dim=-1)
        contact = (torch.linalg.norm(forces_w, dim=-1) > contact_threshold).float()
        contact_change = (contact - self._prev_contact).abs()
        self._prev_contact = contact
        features = torch.stack(
            (
                torch.log1p(f_norm),
                f_n / (f_norm + eps),
                (f_t / (f_n.abs() + eps)).clamp(max=ratio_clip),
                contact,
                contact_change,
            ),
            dim=-1,
        )
        return features.view(env.num_envs, -1)


class tactile_event_features(ManagerTermBase):
    """Compact contact-event feature vector e_t (proposal preprocessing step e).

    ``[max_b |df_b|, sum_b |contact change|, max_b tangential/normal ratio, |d(net force)|,
    num contacts, log1p(max |f_b|)]`` computed from EMA-smoothed link-frame forces.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        sensor_names = cfg.params["contact_sensor_names"]
        body_names = cfg.params["body_names"]
        robot: Articulation = env.scene[cfg.params.get("asset_cfg", SceneEntityCfg("robot")).name]
        self._body_ids, _ = robot.find_bodies(list(body_names), preserve_order=True)
        self._robot = robot
        self._num_fingers = len(sensor_names)
        self._ema = torch.zeros(env.num_envs, self._num_fingers, 3, device=env.device)
        self._prev_ema = torch.zeros_like(self._ema)
        self._prev_contact = torch.zeros(env.num_envs, self._num_fingers, device=env.device)

    def reset(self, env_ids=None):
        ids = _resolve_env_ids(self._env, env_ids)
        self._ema[ids] = 0.0
        self._prev_ema[ids] = 0.0
        self._prev_contact[ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        contact_sensor_names: list[str],
        body_names: list[str],
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        ema_alpha: float = 0.8,
        contact_threshold: float = 1.0,
        ratio_clip: float = 10.0,
        eps: float = 1.0e-6,
    ) -> torch.Tensor:
        forces_w = _sensor_forces_w(env, contact_sensor_names)
        body_quat_w = self._robot.data.body_quat_w[:, self._body_ids]
        forces_l = quat_apply_inverse(body_quat_w, forces_w)
        self._prev_ema = self._ema
        self._ema = ema_alpha * self._ema + (1.0 - ema_alpha) * forces_l
        contact = (torch.linalg.norm(forces_w, dim=-1) > contact_threshold).float()

        force_diff = torch.linalg.norm(self._ema - self._prev_ema, dim=-1)
        contact_change = (contact - self._prev_contact).abs()
        f_n = self._ema[..., 2]
        f_t = torch.linalg.norm(self._ema[..., :2], dim=-1)
        tan_normal_ratio = (f_t / (f_n.abs() + eps)).clamp(max=ratio_clip)
        net_force_diff = torch.linalg.norm(self._ema.sum(dim=1) - self._prev_ema.sum(dim=1), dim=-1)
        self._prev_contact = contact

        features = torch.stack(
            (
                force_diff.amax(dim=1),
                contact_change.sum(dim=1),
                tan_normal_ratio.amax(dim=1),
                net_force_diff,
                contact.sum(dim=1),
                torch.log1p(torch.linalg.norm(self._ema, dim=-1).amax(dim=1)),
            ),
            dim=-1,
        )
        return features


class tactile_event_features_windowed(ManagerTermBase):
    """Two-block contrast event features over a longer horizon (anti-jitter event detector).

    Per-step force diffs are dominated by PhysX contact-solver jitter (measured ~5.5±10.7 N/step),
    so this term compares block statistics of the last ``window`` steps against the previous
    ``window`` steps. Averaging n steps shrinks the jitter by ~sqrt(n) while sustained perturbation
    signatures (pulls, friction-drop slip) survive the averaging.

    Features (7): [max_f |Δmean force mag|, ‖Δmean net force‖, max_f |Δcontact duty|,
    max_f Δ(tan/normal ratio), contact changes in recent window, recent net-force-norm std,
    log1p(max current force)].
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        sensor_names = cfg.params["contact_sensor_names"]
        body_names = cfg.params["body_names"]
        robot: Articulation = env.scene[cfg.params.get("asset_cfg", SceneEntityCfg("robot")).name]
        self._body_ids, _ = robot.find_bodies(list(body_names), preserve_order=True)
        self._robot = robot
        self._num_fingers = len(sensor_names)
        window = int(cfg.params.get("window", 12))
        self._window = window
        self._buf_force = torch.zeros(env.num_envs, 2 * window, self._num_fingers, 3, device=env.device)
        self._buf_contact = torch.zeros(env.num_envs, 2 * window, self._num_fingers, device=env.device)

    def reset(self, env_ids=None):
        ids = _resolve_env_ids(self._env, env_ids)
        self._buf_force[ids] = 0.0
        self._buf_contact[ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        contact_sensor_names: list[str],
        body_names: list[str],
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        window: int = 12,
        contact_threshold: float = 1.0,
        ratio_clip: float = 10.0,
        eps: float = 1.0e-6,
    ) -> torch.Tensor:
        forces_w = _sensor_forces_w(env, contact_sensor_names)
        body_quat_w = self._robot.data.body_quat_w[:, self._body_ids]
        forces_l = quat_apply_inverse(body_quat_w, forces_w)
        contact = (torch.linalg.norm(forces_w, dim=-1) > contact_threshold).float()

        # shift ring buffers and append the newest frame
        self._buf_force = torch.roll(self._buf_force, shifts=-1, dims=1)
        self._buf_force[:, -1] = forces_l
        self._buf_contact = torch.roll(self._buf_contact, shifts=-1, dims=1)
        self._buf_contact[:, -1] = contact

        w = self._window
        recent_f, prev_f = self._buf_force[:, w:], self._buf_force[:, :w]
        recent_c, prev_c = self._buf_contact[:, w:], self._buf_contact[:, :w]

        # block statistics
        mag_recent = torch.linalg.norm(recent_f, dim=-1).mean(dim=1)      # (E, F)
        mag_prev = torch.linalg.norm(prev_f, dim=-1).mean(dim=1)
        net_recent = recent_f.mean(dim=1).sum(dim=1)                       # (E, 3)
        net_prev = prev_f.mean(dim=1).sum(dim=1)
        duty_recent = recent_c.mean(dim=1)                                 # (E, F)
        duty_prev = prev_c.mean(dim=1)

        def _slip_ratio(block: torch.Tensor) -> torch.Tensor:
            f_n = block[..., 2]
            f_t = torch.linalg.norm(block[..., :2], dim=-1)
            return (f_t / (f_n.abs() + eps)).clamp(max=ratio_clip).mean(dim=1)  # (E, F)

        slip_recent = _slip_ratio(recent_f)
        slip_prev = _slip_ratio(prev_f)

        contact_changes = (recent_c[:, 1:] - recent_c[:, :-1]).abs().sum(dim=(1, 2))
        net_norm_std = torch.linalg.norm(recent_f.sum(dim=2), dim=-1).std(dim=1)

        # per-step raw features complement the block statistics (joint AUC probe evidence)
        step_diff = self._buf_force[:, -1] - self._buf_force[:, -2]
        raw_max_diff = torch.linalg.norm(step_diff, dim=-1).amax(dim=1)
        raw_net_diff = torch.linalg.norm(step_diff.sum(dim=1), dim=-1)
        raw_max_mag = torch.linalg.norm(forces_l, dim=-1).amax(dim=1)

        features = torch.stack(
            (
                (mag_recent - mag_prev).abs().amax(dim=1),
                torch.linalg.norm(net_recent - net_prev, dim=-1),
                (duty_recent - duty_prev).abs().amax(dim=1),
                (slip_recent - slip_prev).abs().amax(dim=1),
                contact_changes,
                net_norm_std,
                torch.log1p(torch.linalg.norm(forces_l, dim=-1).amax(dim=1)),
                raw_max_diff,
                raw_net_diff,
                raw_max_mag,
            ),
            dim=-1,
        )
        return features


class object_event_features(ManagerTermBase):
    """Object kinematic event cues: [speed, |accel|] of the object in the robot root frame.

    Stands in for a perception-derived centroid-velocity signal (the object point cloud is already
    part of the actor's observation); joint-AUC probing showed these two cues add ~5 points of
    event separability on top of the tactile features.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._prev_vel = torch.zeros(env.num_envs, 3, device=env.device)

    def reset(self, env_ids=None):
        ids = _resolve_env_ids(self._env, env_ids)
        self._prev_vel[ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ) -> torch.Tensor:
        robot: Articulation = env.scene[robot_cfg.name]
        obj: RigidObject = env.scene[object_cfg.name]
        vel_b = quat_apply_inverse(robot.data.root_quat_w, obj.data.root_lin_vel_w)
        accel = torch.linalg.norm(vel_b - self._prev_vel, dim=-1)
        self._prev_vel = vel_b.clone()
        return torch.stack((torch.linalg.norm(vel_b, dim=-1), accel), dim=-1)


class ObjectFrictionDrop(ManagerTermBase):
    """Mid-episode sudden friction drop on the object (reset-mode event term).

    With probability ``probability`` an episode gets one friction-drop event: at a random time
    ``t_drop`` the object's static/dynamic friction is scaled by a factor drawn from
    ``magnitude_range`` (e.g. 0.3 = drop to 30%) for the rest of the episode. Original per-env
    material properties (from the startup randomization) are restored on the next reset.
    The interval-mode term :func:`apply_object_friction_drop` performs the actual switch.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.drop_time = torch.full((env.num_envs,), -1.0, device=env.device)
        self.drop_scale = torch.ones(env.num_envs, device=env.device)
        self.applied = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._base_materials: torch.Tensor | None = None

    def _materials_view(self, env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg):
        obj: RigidObject = env.scene[object_cfg.name]
        return obj.root_physx_view

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids,
        probability: float = 0.7,
        magnitude_range: tuple[float, float] = (0.3, 0.5),
        start_time_range: tuple[float, float] = (0.5, 2.5),
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ) -> None:
        ids = _resolve_env_ids(env, env_ids)
        view = self._materials_view(env, object_cfg)
        # lazily cache the post-startup-randomization materials as the restore point
        if self._base_materials is None:
            self._base_materials = view.get_material_properties().clone()
        # restore original friction for envs that dropped in their previous episode
        applied_ids = ids[self.applied[ids]]
        if len(applied_ids) > 0:
            materials = view.get_material_properties()
            cpu_ids = applied_ids.cpu()
            materials[cpu_ids] = self._base_materials[cpu_ids]
            view.set_material_properties(materials, cpu_ids)
        self.applied[ids] = False
        # sample this episode's schedule
        num = len(ids)
        active = torch.rand(num, device=env.device) < probability
        t_drop = (
            torch.rand(num, device=env.device) * (start_time_range[1] - start_time_range[0]) + start_time_range[0]
        )
        self.drop_time[ids] = torch.where(active, t_drop, torch.full_like(t_drop, -1.0))
        self.drop_scale[ids] = (
            torch.rand(num, device=env.device) * (magnitude_range[1] - magnitude_range[0]) + magnitude_range[0]
        )

    def episode_time(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        return env.episode_length_buf.float() * env.step_dt

    def active_flag(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        t = self.episode_time(env)
        return ((self.drop_time >= 0.0) & (t >= self.drop_time)).float()

    def gate_label(self, env: ManagerBasedRLEnv, label_window_s: float = 0.3) -> torch.Tensor:
        # friction stays low until episode end, so the label is simply the active window
        return self.active_flag(env)

    def current_force(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        # interface parity with ObjectForcePerturbation (no external force in this family)
        return torch.zeros(env.num_envs, 3, device=env.device)


def apply_object_friction_drop(
    env: ManagerBasedRLEnv,
    env_ids,
    perturbation_term: str = "tacres_perturbation",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> None:
    """Interval-mode event (fires every step) that switches object friction at the scheduled time."""
    term: ObjectFrictionDrop = _get_perturbation_term(env, perturbation_term)
    t = term.episode_time(env)
    trigger = (term.drop_time >= 0.0) & (t >= term.drop_time) & (~term.applied)
    trigger_ids = trigger.nonzero().flatten()
    if len(trigger_ids) == 0:
        return
    obj: RigidObject = env.scene[object_cfg.name]
    view = obj.root_physx_view
    materials = view.get_material_properties()
    cpu_ids = trigger_ids.cpu()
    scales = term.drop_scale[trigger_ids].cpu().unsqueeze(-1)
    materials[cpu_ids, :, :2] = materials[cpu_ids, :, :2] * scales.unsqueeze(-1)
    view.set_material_properties(materials, cpu_ids)
    term.applied[trigger_ids] = True


class ObjectForcePerturbation(ManagerTermBase):
    """Samples per-episode external force impulses on the object (reset-mode event term).

    With probability ``probability`` an episode receives 1..``max_impulses`` impulses. Each impulse has
    a random start time, duration, unit direction, and magnitude drawn from ``magnitude_range`` (which
    the ADR curriculum widens from 2N towards 8N). The interval-mode term
    :func:`apply_object_force_perturbation` applies the scheduled forces each step.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        max_impulses = int(cfg.params.get("max_impulses", 2))
        self.max_impulses = max_impulses
        self.impulse_start = torch.full((env.num_envs, max_impulses), -1.0, device=env.device)
        self.impulse_end = torch.full((env.num_envs, max_impulses), -1.0, device=env.device)
        self.impulse_force = torch.zeros(env.num_envs, max_impulses, 3, device=env.device)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids,
        probability: float = 0.7,
        max_impulses: int = 2,
        magnitude_range: tuple[float, float] = (2.0, 2.0),
        duration_range: tuple[float, float] = (0.1, 0.2),
        start_time_range: tuple[float, float] = (0.3, 3.4),
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ) -> None:
        ids = _resolve_env_ids(env, env_ids)
        num = len(ids)
        device = env.device
        k = self.max_impulses

        active_episode = torch.rand(num, device=device) < probability
        num_impulses = torch.randint(1, k + 1, (num,), device=device)
        impulse_active = (
            torch.arange(k, device=device).unsqueeze(0) < num_impulses.unsqueeze(1)
        ) & active_episode.unsqueeze(1)

        start = (
            torch.rand(num, k, device=device) * (start_time_range[1] - start_time_range[0]) + start_time_range[0]
        )
        duration = torch.rand(num, k, device=device) * (duration_range[1] - duration_range[0]) + duration_range[0]
        magnitude = (
            torch.rand(num, k, device=device) * (magnitude_range[1] - magnitude_range[0]) + magnitude_range[0]
        )
        direction = torch.randn(num, k, 3, device=device)
        direction = direction / torch.linalg.norm(direction, dim=-1, keepdim=True).clamp(min=1.0e-6)

        inactive = ~impulse_active
        start = start.masked_fill(inactive, -1.0)
        self.impulse_start[ids] = start
        self.impulse_end[ids] = torch.where(inactive, torch.full_like(start, -1.0), start + duration)
        self.impulse_force[ids] = direction * magnitude.unsqueeze(-1) * impulse_active.unsqueeze(-1).float()

    def episode_time(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        return env.episode_length_buf.float() * env.step_dt

    def current_force(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        """Force to apply at the current step, shape (num_envs, 3)."""
        t = self.episode_time(env).unsqueeze(1)
        in_window = (t >= self.impulse_start) & (t < self.impulse_end) & (self.impulse_start >= 0.0)
        return (self.impulse_force * in_window.unsqueeze(-1).float()).sum(dim=1)

    def active_flag(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        """1.0 while an impulse is being applied, shape (num_envs,)."""
        t = self.episode_time(env).unsqueeze(1)
        in_window = (t >= self.impulse_start) & (t < self.impulse_end) & (self.impulse_start >= 0.0)
        return in_window.any(dim=1).float()

    def gate_label(self, env: ManagerBasedRLEnv, label_window_s: float = 0.3) -> torch.Tensor:
        """Warm-start gate label y_t = 1 within [t_inj, max(t_end, t_inj + label_window_s)].

        For short impulses this is the [t_inj, t_inj + 0.3s] reaction window of the proposal; for
        sustained perturbations (long durations) the label covers the whole active window.
        """
        t = self.episode_time(env).unsqueeze(1)
        label_end = torch.maximum(self.impulse_end, self.impulse_start + label_window_s)
        in_window = (t >= self.impulse_start) & (t < label_end) & (self.impulse_start >= 0.0)
        return in_window.any(dim=1).float()


def _get_perturbation_term(env: ManagerBasedRLEnv, term_name: str) -> ObjectForcePerturbation:
    return getattr(env.event_manager.cfg, term_name).func


def apply_object_force_perturbation(
    env: ManagerBasedRLEnv,
    env_ids,
    perturbation_term: str = "tacres_perturbation",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> None:
    """Interval-mode event (fires every step) that writes the scheduled impulse forces to the object."""
    term = _get_perturbation_term(env, perturbation_term)
    force = term.current_force(env)
    obj: RigidObject = env.scene[object_cfg.name]
    obj.set_external_force_and_torque(force.unsqueeze(1), torch.zeros_like(force).unsqueeze(1))


def perturbation_state(
    env: ManagerBasedRLEnv,
    perturbation_term: str = "tacres_perturbation",
) -> torch.Tensor:
    """Privileged perturbation observation: [active flag, applied force xyz], shape (num_envs, 4)."""
    term = _get_perturbation_term(env, perturbation_term)
    return torch.cat((term.active_flag(env).unsqueeze(1), term.current_force(env)), dim=1)


def perturbation_gate_label(
    env: ManagerBasedRLEnv,
    perturbation_term: str = "tacres_perturbation",
    label_window_s: float = 0.3,
) -> torch.Tensor:
    """Training-only gate warm-start label, shape (num_envs, 1)."""
    term = _get_perturbation_term(env, perturbation_term)
    return term.gate_label(env, label_window_s).unsqueeze(1)


def object_root_state_b(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Privileged object state in the robot root frame: pos(3), quat(4), lin vel(3), ang vel(3)."""
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    quat_r = robot.data.root_quat_w
    pos_b = quat_apply_inverse(quat_r, obj.data.root_pos_w - robot.data.root_pos_w)
    lin_vel_b = quat_apply_inverse(quat_r, obj.data.root_lin_vel_w)
    ang_vel_b = quat_apply_inverse(quat_r, obj.data.root_ang_vel_w)
    return torch.cat((pos_b, obj.data.root_quat_w, lin_vel_b, ang_vel_b), dim=1)


def fingertip_force_excess(
    env: ManagerBasedRLEnv,
    contact_sensor_names: list[str],
    threshold: float = 20.0,
) -> torch.Tensor:
    """Positive excess of the max fingertip force magnitude over ``threshold`` (for r_force penalty)."""
    force_mag = torch.linalg.norm(_sensor_forces_w(env, contact_sensor_names), dim=-1)
    return (force_mag.amax(dim=1) - threshold).clamp(min=0.0)
