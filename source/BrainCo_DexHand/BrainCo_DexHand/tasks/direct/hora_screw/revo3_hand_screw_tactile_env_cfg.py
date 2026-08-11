# Copyright (c) 2026, BrainCo.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Environment configs for Revo3 HORA screw/valve tasks with TacSL fingertips.

Tactile variants of the nut-bolt / screwdriver / valve tasks. They inherit the
base task knobs and add layout-aware TacSL fingertip force fields:

* The hand USD is swapped for the tactile overlay
  (``assets/usd/tactile_dexscrew/revo3_right_tactile.usda``), which references
  the shared ``revo3_right.usd`` and adds a ``tactile_elastomer`` Xform under
  each ``right_*_tip_Link``. Same joints/bodies, so all base-task logic holds.
* One ``VisuoTactileSensorCfg`` (force field only, no tactile camera) is
  created per fingertip against the rotating nut/handle body.
* ``regular_grid`` keeps the checkpoint-compatible 16x16 query grid and CNN
  path. Its force field is average-pooled to ``tactile_array_pool``.
* ``estimated_official`` queries the versioned 21/31 physical circular patches
  directly and preserves every physical node for the static GNN path. Teacher
  frames contain per-node public state plus privileged force channels; student
  frames contain only public state and per-finger shift/context channels.
* Both layouts keep ten control frames. Their spatial encoders produce the same
  per-finger State/Shift token contract before current-frame self-attention and
  history cross-attention.

The TacSL force field queries the contact object's SDF, so the tactile env
switches the nut/handle collision mesh approximation to "sdf" at setup time
(the base tasks keep the URDF importer's convex collision).
"""

from __future__ import annotations

import os
from pathlib import Path

from isaaclab.utils import configclass

from .revo3_hand_screw_env_cfg import (
    Revo3HandScrewDriverEnvCfg,
    Revo3HandScrewNutBoltEnvCfg,
    Revo3HandValveDriver40EnvCfg,
    Revo3HandVavleDriverEnvCfg,
)
from ...tactile_layout import (
    ESTIMATED_OFFICIAL_LAYOUT,
    REGULAR_GRID_LAYOUT,
    estimated_official_centers_xy,
    resolve_tactile_layout_name,
)

try:
    from isaaclab_assets.sensors import GELSIGHT_R15_CFG

    from .tacsl_sensor import Revo3VisuoTactileSensor, Revo3VisuoTactileSensorCfg
except ImportError:
    GELSIGHT_R15_CFG = None
    Revo3VisuoTactileSensor = None
    Revo3VisuoTactileSensorCfg = None

_REPO_ROOT = Path(__file__).resolve().parents[6]
REVO3_TACTILE_USD = str(_REPO_ROOT / "assets" / "usd" / "tactile_dexscrew" / "revo3_right_tactile.usda")

# Same order as Revo3HandScrewEnvCfg.fingertip_body_names (thumb first).
TACTILE_FINGER_ORDER = ("thumb", "index", "middle", "ring", "little")
TACTILE_TIP_BODIES = tuple(f"right_{finger}_tip_Link" for finger in TACTILE_FINGER_ORDER)
TACTILE_VIS_SENSOR_NAMES = tuple(f"{finger}_tactile_sensor" for finger in TACTILE_FINGER_ORDER)


def _finger_name_from_joint_name(joint_name: str) -> str | None:
    """Map a Revo3 joint name to its canonical tactile finger name."""
    for finger_name in TACTILE_FINGER_ORDER:
        if f"_{finger_name}_" in joint_name:
            return finger_name
    return None


def _active_tactile_finger_names(masked_action_joint_names) -> tuple[str, ...]:
    """Derive task-active tactile fingers from the action-joint mask."""
    masked_fingers = {
        finger_name
        for joint_name in (masked_action_joint_names or ())
        if (finger_name := _finger_name_from_joint_name(str(joint_name))) is not None
    }
    return tuple(finger_name for finger_name in TACTILE_FINGER_ORDER if finger_name not in masked_fingers)


def _env_bool(name: str, default: bool) -> bool:
    """Read an optional boolean environment override."""
    value = os.getenv(name)
    if value is None:
        return bool(default)
    return value.strip().lower() in {"1", "true", "yes", "on"}


@configclass
class Revo3HandScrewTactileMixinCfg:
    """Add layout-aware TacSL sensing and observation history to a screw task."""

    # ---- capacity-aware multi-finger coordination intrinsic reward (MASTER SWITCH) ----
    # Each finger's positive object-axis torque is normalized by its current
    # lever-arm capacity. A monotone concave utility rewards efficient load
    # sharing, while force overload and off-axis torque receive a small cost.
    # The signed intrinsic reward is added to the ordinary environment reward,
    # keeping one PPO return, one value baseline, and one advantage stream.
    # Enabled by default. Independent of legacy multi_contact_reward; for valve prefer one or the other.
    enable_coord_endogenous_reward = True  # 內生多指协作奖励
    # Homotopy curriculum: guide the search strongly in early/mid training,
    # then leave only a weak late tie-breaker instead of requiring all fingers.
    coord_intrinsic_weight_initial = 0.10
    coord_intrinsic_weight_peak = 0.30
    # Late quality preference remains meaningful; the effective-torque branch
    # below has its own fixed physical coefficient and does not use this decay.
    coord_intrinsic_weight_final = 0.18
    coord_intrinsic_warmup_end = 5_000_000
    coord_intrinsic_decay_start = 60_000_000
    coord_intrinsic_decay_end = 90_000_000
    # Give useful positive-torque coordination a dense early signal, then remove
    # the floor so late training requires measured task progress.
    coord_q_floor_initial = 0.20
    coord_q_floor_final = 0.0
    # Keep small useful torque dense early; anneal to the smooth physical
    # torque-presence gate together with the rotation-progress floor.
    coord_presence_floor_initial = 1.0
    coord_presence_floor_final = 0.0
    # A separate bounded guide guarantees a positive contribution from useful
    # axis torque even before rotation/stability gates become informative.
    coord_effective_torque_guide_ref = 0.005
    coord_effective_torque_guide_power = 0.5
    coord_effective_torque_reward_weight = 0.12
    coord_guide_curriculum_start = 0
    coord_guide_curriculum_end = 15_000_000
    coord_alpha_h = 1.0
    coord_alpha_g = 0.0
    coord_load_window = 16  # control steps; credits sequential finger gait
    coord_instantaneous_mix = 0.60
    coord_force_comfort_ref = 2.5  # N; comfort force used to form lever-arm capacity
    coord_load_saturation = 0.50  # normalized load at which marginal utility decays
    coord_load_max = 3.0  # numerical cap for utility/history and waste diagnostics
    coord_lever_ratio_min = 0.25
    coord_lever_ratio_max = 1.50
    coord_axis_efficiency_power = 1.0
    coord_overload_penalty_weight = 0.20
    coord_waste_penalty_weight = 0.10
    coord_n_sat = 6  # active unique sensors for b→1 in the selected layout
    coord_taxel_thr = 1.0e-5  # |F| threshold on raw TacSL (pre-scale) for active taxel
    # Half-saturation of the smooth useful-torque presence gate. The torque is
    # capacity-normalized, so this scale remains comparable across object sizes.
    coord_presence_half_saturation = 0.001
    coord_obj_radius = 0.02  # m; ρ_ref = coord_rho_ref_scale * this
    coord_rho_ref_scale = 0.7
    coord_v_ref = 0.03  # m/s; tangential motion consistency
    coord_motion_floor = 0.10  # keep stable rolling contacts from zeroing C_f
    # Unified per-finger task contribution: positive object-axis torque from the
    # complete ContactSensor force (normal + friction) at the tracked contact point.
    finger_contribution_torque_ref = 0.05  # N*m; contribution saturates at this torque
    finger_contact_force_on_object_sign = -1.0  # ContactSensor reports force on the finger
    finger_physical_contact_force_ref = 1.0  # N; physical-contact confidence saturation
    coord_omega_ref = 1.0  # rad/s; nutbolt/screwdriver default
    coord_omega_min = 0.05  # rad/s; below this, only the scheduled guide floor remains
    coord_q_power = 2.0  # sharpen rotation progress gate without changing sign
    coord_rot_dir = 1.0  # +1 matches rotate_reward on positive nut_dof_vel_cf
    coord_w_q = 5  # control steps for ω+ smoothing
    coord_k_slip = 0.75
    coord_v_slip_ref = 0.04  # m/s
    coord_b_th = 0.2  # contact establish/release threshold on b
    coord_delta_h = 8  # handover window (control steps)
    coord_drop_finger_dist = 0.08  # m; soft drop proxy (matches finger_dist_reset)
    coord_drop_finger_soft_margin = 0.04  # m; linear falloff beyond dist
    coord_drop_nut_force_min = 1.0e-3  # N; soft drop if nut contact below this
    coord_drop_nut_force_soft_scale = 5.0e-3  # N; force gate reaches 1 above this
    coord_stability_floor = 0.05  # nonzero floor for distance/contact stability gates
    # ---- tactile array (TacSL force field) ----
    # Keep the checkpoint-compatible regular grid by default. Set this to
    # "estimated_official" (or export REVO3_TACTILE_LAYOUT=estimated_official)
    # to snap all query slots to the versioned per-finger circle-center estimates.
    tactile_layout = REGULAR_GRID_LAYOUT
    tactile_array_size = (16, 16)
    # pooled grid appended to priv_info: per finger pool_h*pool_w*(normal+2 shear)
    tactile_array_pool = (4, 4)
    # F_n = normal_contact_stiffness(=1.0) * penetration_depth[m] ~ 1e-3; rescale to O(1)
    tactile_force_scale = 200.0
    tactile_force_clip = 5.0
    # Teacher temporal TF uses multi-frame history (no channel-wise force delta).
    # Keep False so priv_info = base(11) + current teacher frame(320) = 331.
    # Teacher frame per taxel: [Fx, Fy, Fz, d_norm] (3D force + contact duration).
    tactile_teacher_use_delta = False
    teacher_tactile_channels = 4
    teacher_tactile_frame_dim = 320  # 80 taxels * 4; filled/validated in _configure
    # Shared history length with the student tactile encoder (control steps).
    teacher_tactile_history_len = 10   #10
    # PhysX SDF resolution for the rotating nut/handle collision mesh
    tactile_sdf_resolution = 256
    tactile_debug_vis = False
    tactile_visualize_taxel_points = False
    tactile_visualize_contact_forces = False
    tactile_vis_env_index = 0
    tactile_taxel_marker_radius = 0.0005
    tactile_taxel_active_threshold = 0.001
    tactile_contact_force_marker_radius = 0.004
    tactile_contact_force_vis_scale = 0.01
    tactile_contact_force_vis_max_len = 0.08

    tactile_tip_body_names = TACTILE_TIP_BODIES
    tactile_vis_sensor_names = TACTILE_VIS_SENSOR_NAMES
    pack_active_tactile_only = True
    tactile_active_finger_names = TACTILE_FINGER_ORDER
    tactile_active_finger_indices = tuple(range(len(TACTILE_FINGER_ORDER)))
    tactile_sensor = []

    # filled by _configure_revo3_tactile()
    tactile_priv_offset = 0
    tactile_force_dim = 0  # pooled TacSL only: taxels * 3
    tactile_current_dim = 0  # teacher priv/hist frame (= force + d)
    tactile_priv_dim = 0

    # ---- multi-finger coordination reward (0 disables; valve tasks enable) ----
    # r_coord = scale * sat(soft_finger_count - min, 0, max - min)/(max - min)
    #                 * clip(delta_theta_window / rot_ref, 0, 1)
    # Bounded posture bonus: pays only when >=3 fingers hold a sustained duty-cycle
    # contact on the nut/valve AND the valve actually traveled over the window.
    # Force magnitude and speed are deliberately saturated away — they stay priced
    # by torque_penalty and rotate_reward respectively.
    multi_contact_reward_scale = 0.0
    # per-finger contact indicator = tanh(strength / tau); strength = max pooled-cell
    # force norm in tactile_force_scale units (single-taxel touch pools to ~0.0125)
    multi_contact_tau = 0.1
    # EMA duty cycle: time constant ~1/(1-lambda) control steps (~0.33 s at 0.85)
    multi_contact_ema_lambda = 0.85
    # soft finger count band: 0 below min (2-finger pinch earns nothing),
    # saturates at max (<5 keeps one finger free for gaiting)
    multi_contact_min_fingers = 1.0
    multi_contact_max_fingers = 3.0
    # rotation gate: valve angle traveled over the last window control steps,
    # normalized by rot_ref (0.5 rad / 10 steps = full gate at ~1 rad/s average)
    multi_contact_rot_window = 10
    multi_contact_rot_ref = 0.5

    # ---- Stage2 DAgger student observation (real-robot sensing only) ----
    # proprio frame = 21 joint_pos + 21 current joint targets (no contact forces)
    student_proprio_frame_dim = 42
    # Public command channels appended after joint_pos + targets.  Screw/valve
    # tasks have none; target-speed rotation sets this to one.
    student_proprio_command_dim = 0
    student_proprio_history_len = 3
    student_proprio_history_dim = 126
    # Structural tactile frame (sim2real-safe, no force magnitude):
    # per pooled taxel channels [b, Δb, d_norm] with
    #   b = 1{||F|| > thr}, Δb = b_t - b_{t-1}, d_norm = min(duration/tau, 1).
    # Layout matches 5 * 4 * 4 * 3 = 240 (same width as pooled TacSL).
    student_tactile_frame_dim = 240
    student_tactile_history_len = 10   #10
    student_tactile_raw_history_dim = 2400  #2400
    student_tactile_encoder_output_dim = 64
    student_obs_dim = 190
    # contact threshold on |scaled pooled tactile| (tactile_force_scale units);
    # single-taxel penetration ~1e-3 m scales to 0.2, diluted 16x by 4x4 avg-pool
    student_tactile_contact_threshold = 0.001
    # Physical-node graph contact hysteresis, applied to scaled force magnitude.
    student_tactile_contact_off_threshold = 0.0005
    # Normalize consecutive-contact duration (control steps) into [0, 1].
    student_tactile_duration_tau = 20.0
    student_tactile_duration_max = 100.0
    tactile_shift_ema_beta = 0.7
    tactile_shift_max = 0.2
    # Taxels per hand used when packing structural channels (5 fingers x 4x4).
    student_tactile_num_taxels = 80
    tactile_graph_sensor_counts = ()
    tactile_graph_total_nodes = 0
    tactile_graph_common_channels = 5
    tactile_graph_force_channels = 5
    tactile_graph_context_channels = 4
    # TacSL already provides per-taxel 3D force and world pose for recovering
    # object-axis torque. Keep PhysX detailed buffers disabled: Isaac Sim 5.0's
    # friction backend can fail even with one environment and corrupt the scene.
    contact_sensor_track_contact_points = False
    contact_sensor_track_friction_forces = False
    contact_sensor_max_contact_data_count_per_prim = 8

    # ---- visible contact reward (TacSL taxel activation on object-touching fingertips) ----
    # Reward fingertips that physically contact the rotating target object and activate
    # TacSL taxels. Per-finger active-taxel ratio saturates at 0.4 and contributes
    # up to 1.0 before applying normalized finger weights. When the shared domain-randomization
    # curriculum is enabled, its progress ramps the target reward ratio from the
    # initial to final value. Adaptive scaling raises the
    # value of efficient intermittent contacts as their batch frequency falls.
    # Enabled on all tactile screw and valve tasks with one shared ratio curriculum.
    enable_visible_contact_reward = False
    visible_contact_force_min = 0.02  # N; ContactSensor force gate (dexscrew contact_force_min)
    tactile_visible_contact_threshold = 1.0e-5
    tactile_visible_contact_ratio_cap = 0.40
    tactile_visible_contact_finger_weights = (1.0, 1.0, 1.0, 0.0, 0.0)
    visible_contact_target_ratio_initial = 0.01
    visible_contact_target_ratio_mid = 0.03
    visible_contact_target_ratio_final = 0.05
    visible_contact_target_ratio_warmup_progress = 0.10
    visible_contact_target_ratio_mid_progress = 0.50
    visible_contact_adaptive_ema = 0.99
    visible_contact_adaptive_scale_min = 1.0
    visible_contact_adaptive_scale_max = 300.0
    visible_contact_reward_clip_ratio = 0.10
    visible_contact_reward_dynamic_clip_min = 0.02
    # A square-root gate keeps useful positive contacts at O(1) reward while
    # zero/reverse-torque contacts receive no independent visible bonus.
    visible_contact_contribution_power = 0.5

    # ---- TacSL / visible-contact observation noise (independent of enable_contact_noise) ----
    # Default off. Teacher: add force noise of size ``frac * |F|`` with random direction
    # (F' = F + frac*|F|*u; original direction kept).
    # Student: flip contact bit b with student_tactile_flip_prob (Δb follows flipped b).
    enable_visible_contact_noise = True
    visible_contact_force_noise_frac = 0.05
    student_tactile_flip_prob = 0.05

    def __post_init__(self):
        super().__post_init__()
        self._configure_revo3_tactile()

    def _configure_revo3_tactile(self):
        if Revo3VisuoTactileSensorCfg is None or Revo3VisuoTactileSensor is None or GELSIGHT_R15_CFG is None:
            raise ImportError(
                "The tactile screw tasks require TacSL contrib sensors: "
                "`isaaclab_contrib.sensors.tacsl_sensor` and `isaaclab_assets.sensors.GELSIGHT_R15_CFG`."
            )

        # swap the hand USD for the tactile overlay (adds elastomer prims on tip links)
        self.robot_cfg = self.robot_cfg.replace(
            spawn=self.robot_cfg.spawn.replace(usd_path=REVO3_TACTILE_USD)
        )

        rows, cols = self.tactile_array_size
        pool_rows, pool_cols = self.tactile_array_pool
        if rows % pool_rows != 0 or cols % pool_cols != 0:
            raise ValueError(
                f"tactile_array_pool {self.tactile_array_pool} must divide tactile_array_size {self.tactile_array_size}"
            )
        self.tactile_layout = resolve_tactile_layout_name(self.tactile_layout)

        self.pack_active_tactile_only = _env_bool(
            "HORA_PACK_ACTIVE_TACTILE_ONLY",
            bool(self.pack_active_tactile_only),
        )
        if self.pack_active_tactile_only:
            active_fingers = _active_tactile_finger_names(
                getattr(self, "masked_action_joint_names", ())
            )
        else:
            active_fingers = tuple(TACTILE_FINGER_ORDER)
        if not active_fingers:
            raise ValueError("At least one tactile finger must remain active.")
        self.tactile_active_finger_names = active_fingers
        self.tactile_active_finger_indices = tuple(
            TACTILE_FINGER_ORDER.index(finger_name) for finger_name in active_fingers
        )
        self.tactile_tip_body_names = tuple(
            f"right_{finger_name}_tip_Link" for finger_name in active_fingers
        )
        self.tactile_vis_sensor_names = tuple(
            f"{finger_name}_tactile_sensor" for finger_name in active_fingers
        )
        weights = tuple(getattr(self, "tactile_visible_contact_finger_weights", ()))
        if len(weights) == len(TACTILE_FINGER_ORDER):
            self.tactile_visible_contact_finger_weights = tuple(
                weights[index] for index in self.tactile_active_finger_indices
            )

        # Extend priv_info with the current teacher tactile frame.  The regular
        # layout preserves the pooled 4x4 contract; estimated_official uses each
        # physical 21/31-node sensor exactly once.
        base_priv_dim = int(self.priv_info_dim)
        self.tactile_priv_offset = base_priv_dim
        if self.tactile_layout == ESTIMATED_OFFICIAL_LAYOUT:
            self.tactile_graph_sensor_counts = tuple(
                int(len(estimated_official_centers_xy(finger_name)))
                for finger_name in self.tactile_active_finger_names
            )
            self.tactile_graph_total_nodes = int(sum(self.tactile_graph_sensor_counts))
            n_taxels = self.tactile_graph_total_nodes
            node_channels = int(self.tactile_graph_common_channels) + int(
                self.tactile_graph_force_channels
            )
            context_dim = len(self.tactile_active_finger_names) * int(
                self.tactile_graph_context_channels
            )
            expected_teacher_frame = n_taxels * node_channels + context_dim
            expected_student_frame = (
                n_taxels * int(self.tactile_graph_common_channels) + context_dim
            )
        else:
            self.tactile_graph_sensor_counts = ()
            self.tactile_graph_total_nodes = 0
            n_taxels = len(self.tactile_tip_body_names) * pool_rows * pool_cols
            if int(self.teacher_tactile_channels) != 4:
                raise ValueError(
                    f"teacher_tactile_channels ({self.teacher_tactile_channels}) must be 4 "
                    f"([Fn, Ft1, Ft2, d_norm] per pooled taxel)"
                )
            expected_teacher_frame = n_taxels * int(self.teacher_tactile_channels)
            expected_student_frame = n_taxels * 3

        self.tactile_force_dim = n_taxels * 3
        self.teacher_tactile_frame_dim = expected_teacher_frame
        self.tactile_current_dim = int(self.teacher_tactile_frame_dim)
        self.tactile_priv_dim = self.tactile_current_dim * (2 if self.tactile_teacher_use_delta else 1)
        self.priv_info_dim = base_priv_dim + self.tactile_priv_dim

        # Student structural frame: pooled [b,delta_b,d] for the grid layout or
        # physical-node [b,on,off,d,eta] plus per-finger shift context for GNN.
        expected_student_taxels = n_taxels
        self.student_tactile_num_taxels = expected_student_taxels
        self.student_tactile_frame_dim = expected_student_frame
        self.student_tactile_raw_history_dim = (
            self.student_tactile_history_len * self.student_tactile_frame_dim
        )
        self.student_obs_dim = (
            self.student_proprio_history_dim + self.student_tactile_encoder_output_dim
        )
        if self.student_tactile_duration_tau <= 0.0:
            raise ValueError(
                f"student_tactile_duration_tau ({self.student_tactile_duration_tau}) must be positive"
            )
        if self.student_tactile_duration_max <= 0.0:
            raise ValueError("student_tactile_duration_max must be positive")
        if not 0.0 <= float(self.student_tactile_contact_off_threshold) < float(
            self.student_tactile_contact_threshold
        ):
            raise ValueError(
                "student_tactile_contact_off_threshold must be non-negative and below the on threshold"
            )
        if not 0.0 <= float(self.tactile_shift_ema_beta) < 1.0:
            raise ValueError("tactile_shift_ema_beta must be in [0, 1)")
        if self.tactile_shift_max <= 0.0:
            raise ValueError("tactile_shift_max must be positive")
        if self.teacher_tactile_history_len != self.student_tactile_history_len:
            raise ValueError(
                f"teacher_tactile_history_len ({self.teacher_tactile_history_len}) must equal "
                f"student_tactile_history_len ({self.student_tactile_history_len}) so teacher/student "
                f"temporal transformers stay aligned"
            )
        expected_proprio_frame_dim = (
            2 * self.action_space + int(self.student_proprio_command_dim)
        )
        if self.student_proprio_frame_dim != expected_proprio_frame_dim:
            raise ValueError(
                f"student_proprio_frame_dim ({self.student_proprio_frame_dim}) must equal "
                "joint_pos + targets + public command channels = "
                f"{expected_proprio_frame_dim}"
            )
        if self.student_proprio_history_dim != self.student_proprio_history_len * self.student_proprio_frame_dim:
            raise ValueError("student_proprio_history_dim != history_len * frame_dim")
        if self.student_tactile_raw_history_dim != self.student_tactile_history_len * self.student_tactile_frame_dim:
            raise ValueError("student_tactile_raw_history_dim != history_len * frame_dim")
        if self.student_obs_dim != self.student_proprio_history_dim + self.student_tactile_encoder_output_dim:
            raise ValueError("student_obs_dim != proprio_history_dim + tactile_encoder_output_dim")

        if self.enable_visible_contact_reward:
            visible_initial_ratio = float(self.visible_contact_target_ratio_initial)
            visible_mid_ratio = float(self.visible_contact_target_ratio_mid)
            visible_final_ratio = float(self.visible_contact_target_ratio_final)
            if not (
                0.0
                < visible_initial_ratio
                <= visible_mid_ratio
                <= visible_final_ratio
                < 1.0
            ):
                raise ValueError(
                    "visible contact target ratios must satisfy "
                    "0 < visible_contact_target_ratio_initial <= "
                    "visible_contact_target_ratio_mid <= "
                    "visible_contact_target_ratio_final < 1"
                )
            visible_warmup_progress = float(
                self.visible_contact_target_ratio_warmup_progress
            )
            visible_mid_progress = float(
                self.visible_contact_target_ratio_mid_progress
            )
            if not 0.0 <= visible_warmup_progress < visible_mid_progress <= 1.0:
                raise ValueError(
                    "visible contact ratio curriculum progress must satisfy "
                    "0 <= warmup < mid <= 1"
                )
            if not 0.0 <= float(self.visible_contact_adaptive_ema) < 1.0:
                raise ValueError("visible_contact_adaptive_ema must be in [0, 1)")
            if (
                float(self.visible_contact_adaptive_scale_min) <= 0.0
                or float(self.visible_contact_adaptive_scale_max)
                < float(self.visible_contact_adaptive_scale_min)
            ):
                raise ValueError(
                    "visible contact adaptive scales must be positive and ordered"
                )
            if float(self.visible_contact_reward_clip_ratio) <= 0.0:
                raise ValueError("visible_contact_reward_clip_ratio must be positive")
            if float(self.visible_contact_reward_dynamic_clip_min) < 0.0:
                raise ValueError(
                    "visible_contact_reward_dynamic_clip_min must be non-negative"
                )
            if not 0.0 < float(self.visible_contact_contribution_power) <= 1.0:
                raise ValueError("visible_contact_contribution_power must be in (0, 1]")
            visible_weights = tuple(float(value) for value in self.tactile_visible_contact_finger_weights)
            if any(value < 0.0 for value in visible_weights) or sum(visible_weights) <= 0.0:
                raise ValueError(
                    "tactile_visible_contact_finger_weights must be non-negative "
                    "and contain at least one positive weight"
                )

        if self.finger_contribution_torque_ref <= 0.0:
            raise ValueError("finger_contribution_torque_ref must be positive")
        if self.finger_physical_contact_force_ref <= 0.0:
            raise ValueError("finger_physical_contact_force_ref must be positive")
        if float(self.finger_contact_force_on_object_sign) not in (-1.0, 1.0):
            raise ValueError("finger_contact_force_on_object_sign must be -1 or 1")
        if int(self.contact_sensor_max_contact_data_count_per_prim) < 1:
            raise ValueError("contact_sensor_max_contact_data_count_per_prim must be positive")

        if self.multi_contact_reward_scale != 0.0:
            if self.multi_contact_tau <= 0.0:
                raise ValueError(f"multi_contact_tau ({self.multi_contact_tau}) must be positive")
            if not 0.0 <= self.multi_contact_ema_lambda < 1.0:
                raise ValueError(
                    f"multi_contact_ema_lambda ({self.multi_contact_ema_lambda}) must be in [0, 1)"
                )
            if self.multi_contact_max_fingers <= self.multi_contact_min_fingers:
                raise ValueError(
                    f"multi_contact_max_fingers ({self.multi_contact_max_fingers}) must exceed "
                    f"multi_contact_min_fingers ({self.multi_contact_min_fingers})"
                )
            if self.multi_contact_rot_ref <= 0.0:
                raise ValueError(f"multi_contact_rot_ref ({self.multi_contact_rot_ref}) must be positive")
            if not 1 <= self.multi_contact_rot_window <= self.nut_termination_history_len:
                raise ValueError(
                    f"multi_contact_rot_window ({self.multi_contact_rot_window}) must be in "
                    f"[1, nut_termination_history_len={self.nut_termination_history_len}]"
                )

        if self.enable_coord_endogenous_reward:
            weights = (
                float(self.coord_intrinsic_weight_initial),
                float(self.coord_intrinsic_weight_peak),
                float(self.coord_intrinsic_weight_final),
            )
            if any(weight < 0.0 for weight in weights):
                raise ValueError("coord intrinsic weights must be non-negative")
            if self.coord_intrinsic_weight_peak < max(
                self.coord_intrinsic_weight_initial,
                self.coord_intrinsic_weight_final,
            ):
                raise ValueError(
                    "coord_intrinsic_weight_peak must be the largest curriculum weight"
                )
            if not (
                0
                <= int(self.coord_intrinsic_warmup_end)
                <= int(self.coord_intrinsic_decay_start)
                <= int(self.coord_intrinsic_decay_end)
            ):
                raise ValueError(
                    "coord intrinsic curriculum steps must satisfy "
                    "0 <= warmup_end <= decay_start <= decay_end"
                )
            for name in (
                "coord_q_floor_initial",
                "coord_q_floor_final",
                "coord_presence_floor_initial",
                "coord_presence_floor_final",
                "coord_effective_torque_reward_weight",
            ):
                value = float(getattr(self, name))
                if not 0.0 <= value <= 1.0:
                    raise ValueError(f"{name} ({value}) must be in [0, 1]")
            if int(self.coord_guide_curriculum_end) < int(self.coord_guide_curriculum_start):
                raise ValueError(
                    "coord_guide_curriculum_end must be >= coord_guide_curriculum_start"
                )
            if self.coord_effective_torque_guide_ref <= 0.0:
                raise ValueError("coord_effective_torque_guide_ref must be positive")
            if self.coord_effective_torque_guide_power <= 0.0:
                raise ValueError("coord_effective_torque_guide_power must be positive")
            if abs(float(self.coord_alpha_h) + float(self.coord_alpha_g) - 1.0) > 1.0e-5:
                raise ValueError(
                    f"coord_alpha_h + coord_alpha_g must equal 1, got "
                    f"{self.coord_alpha_h} + {self.coord_alpha_g}"
                )
            if not 1 <= int(self.coord_load_window) <= 256:
                raise ValueError("coord_load_window must be in [1, 256]")
            if not 0.0 <= float(self.coord_instantaneous_mix) <= 1.0:
                raise ValueError("coord_instantaneous_mix must be in [0, 1]")
            if self.coord_force_comfort_ref <= 0.0:
                raise ValueError("coord_force_comfort_ref must be positive")
            if self.coord_load_saturation <= 0.0 or self.coord_load_max <= 0.0:
                raise ValueError("coord load saturation and maximum must be positive")
            if not 0.0 < self.coord_lever_ratio_min <= self.coord_lever_ratio_max:
                raise ValueError("coord lever-ratio bounds must be positive and ordered")
            if self.coord_axis_efficiency_power <= 0.0:
                raise ValueError("coord_axis_efficiency_power must be positive")
            if (
                self.coord_overload_penalty_weight < 0.0
                or self.coord_waste_penalty_weight < 0.0
            ):
                raise ValueError("coord effort penalty weights must be non-negative")
            if self.coord_n_sat <= 0:
                raise ValueError(f"coord_n_sat ({self.coord_n_sat}) must be positive")
            if self.coord_presence_half_saturation <= 0.0:
                raise ValueError(
                    "coord_presence_half_saturation must be positive"
                )
            if self.coord_obj_radius <= 0.0 or self.coord_rho_ref_scale <= 0.0:
                raise ValueError("coord_obj_radius and coord_rho_ref_scale must be positive")
            if self.coord_v_ref <= 0.0 or self.coord_omega_ref <= 0.0:
                raise ValueError("coord_v_ref and coord_omega_ref must be positive")
            if not 0.0 <= float(self.coord_omega_min) < float(self.coord_omega_ref):
                raise ValueError(
                    f"coord_omega_min ({self.coord_omega_min}) must be in [0, coord_omega_ref)"
                )
            if self.coord_q_power <= 0.0:
                raise ValueError(f"coord_q_power ({self.coord_q_power}) must be positive")
            if not 0.0 <= float(self.coord_motion_floor) < 1.0:
                raise ValueError(f"coord_motion_floor ({self.coord_motion_floor}) must be in [0, 1)")
            if not 1 <= int(self.coord_w_q) <= 64:
                raise ValueError(f"coord_w_q ({self.coord_w_q}) must be in [1, 64]")
            if self.coord_k_slip <= 0.0 or self.coord_v_slip_ref <= 0.0:
                raise ValueError("coord_k_slip and coord_v_slip_ref must be positive")
            if not 0.0 < float(self.coord_b_th) < 1.0:
                raise ValueError(f"coord_b_th ({self.coord_b_th}) must be in (0, 1)")
            if int(self.coord_delta_h) < 1:
                raise ValueError(f"coord_delta_h ({self.coord_delta_h}) must be >= 1")
            if self.coord_drop_finger_soft_margin <= 0.0:
                raise ValueError(
                    f"coord_drop_finger_soft_margin ({self.coord_drop_finger_soft_margin}) must be positive"
                )
            if self.coord_drop_nut_force_soft_scale <= 0.0:
                raise ValueError(
                    f"coord_drop_nut_force_soft_scale ({self.coord_drop_nut_force_soft_scale}) must be positive"
                )
            if self.coord_drop_nut_force_soft_scale <= self.coord_drop_nut_force_min:
                raise ValueError(
                    f"coord_drop_nut_force_soft_scale ({self.coord_drop_nut_force_soft_scale}) must exceed "
                    f"coord_drop_nut_force_min ({self.coord_drop_nut_force_min})"
                )
            if not 0.0 <= float(self.coord_stability_floor) < 1.0:
                raise ValueError(f"coord_stability_floor ({self.coord_stability_floor}) must be in [0, 1)")

        contact_object_prim_path_expr = getattr(
            self, "tactile_contact_object_prim_path_expr", None
        )
        if contact_object_prim_path_expr is None:
            contact_object_prim_path_expr = (
                f"/World/envs/env_.*/object/{self.nut_body_name}"
            )

        self.tactile_sensor = []
        for finger_name, tip_body in zip(
            self.tactile_active_finger_names,
            self.tactile_tip_body_names,
        ):
            elastomer_path = f"/World/envs/env_.*/hand/{tip_body}/tactile_elastomer"
            sensor_cfg = Revo3VisuoTactileSensorCfg(
                class_type=Revo3VisuoTactileSensor,
                prim_path=f"{elastomer_path}/tactile_sensor",
                history_length=0,
                render_cfg=GELSIGHT_R15_CFG,
                enable_camera_tactile=False,
                enable_force_field=True,
                tactile_array_size=self.tactile_array_size,
                tactile_margin=0.0025,
                contact_object_prim_path_expr=contact_object_prim_path_expr,
                normal_contact_stiffness=1.0,
                friction_coefficient=2.0,
                tangential_stiffness=0.1,
                camera_cfg=None,
                debug_vis=self.tactile_debug_vis,
                tactile_layout=self.tactile_layout,
                tactile_layout_finger=finger_name,
            )
            self.tactile_sensor.append(sensor_cfg)


@configclass
class Revo3HandScrewNutBoltTactileEnvCfg(Revo3HandScrewTactileMixinCfg, Revo3HandScrewNutBoltEnvCfg):
    """Nut-bolt task + TacSL fingertip arrays in the teacher observation."""

    enable_visible_contact_reward = True
    # Intrinsic coord quality defaults: α_H=1, α_G=0, ω_ref=1 rad/s.
    coord_alpha_h = 1.0
    coord_alpha_g = 0.0
    coord_omega_ref = 1.0


@configclass
class Revo3HandScrewDriverTactileEnvCfg(Revo3HandScrewTactileMixinCfg, Revo3HandScrewDriverEnvCfg):
    """Screwdriver task + TacSL fingertip arrays in the teacher observation."""

    enable_visible_contact_reward = True
    tactile_visible_contact_finger_weights = (1.0, 1.0, 0.8, 0.8, 0.0)
    coord_obj_radius = 0.016  # 16 mm handle circumradius at object_radius scale 1.0
    coord_alpha_h = 1.0
    coord_alpha_g = 0.0
    coord_omega_ref = 1.0


@configclass
class Revo3HandVavleDriverTactileEnvCfg(Revo3HandScrewTactileMixinCfg, Revo3HandVavleDriverEnvCfg):
    """Five-finger hexagonal valve task + TacSL fingertip arrays."""

    # All five fingers are actuated; normalized weights keep the reward maximum
    # equal to the shared visible-contact curriculum scale.
    tactile_visible_contact_finger_weights = (1.0, 1.0, 1.0, 1.0, 1.0)

    coord_obj_radius = 0.035  # 35 mm handle circumradius at object_radius scale 1.0
    # posture bonus: worth up to ~0.6 rad/s of speed (scale / rotate_reward_scale)
    multi_contact_reward_scale = 0.0
    # Prefer either legacy multi_contact OR endogenous coord (not both at full strength).
    # Recommended endogenous: enable_coord_endogenous_reward=True, multi_contact_reward_scale=0.
    coord_alpha_h = 0.4
    coord_alpha_g = 0.6
    coord_omega_ref = 1.0


@configclass
class Revo3HandValveDriver40TactileEnvCfg(Revo3HandScrewTactileMixinCfg, Revo3HandValveDriver40EnvCfg):
    """The tactile valve task with a 40 mm handle circumradius."""

    multi_contact_reward_scale = 1.5
    # Prefer either legacy multi_contact OR endogenous coord (not both at full strength).
    coord_alpha_h = 0.4
    coord_alpha_g = 0.6
    coord_omega_ref = 1
    coord_obj_radius = 0.04  # 40 mm handle
