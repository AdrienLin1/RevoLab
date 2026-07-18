# Copyright (c) 2026, BrainCo.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Environment configs for Revo3 HORA screw/valve tasks with TacSL fingertip arrays.

Tactile variants of the nut-bolt / screwdriver / valve tasks. They inherit the
base task knobs and add the TacSL 16x16 fingertip force-field arrays from
tactile-revo3:

* The hand USD is swapped for the tactile overlay
  (``assets/usd/tactile_dexscrew/revo3_right_tactile.usda``), which references
  the shared ``revo3_right.usd`` and adds a ``tactile_elastomer`` Xform under
  each ``right_*_tip_Link``. Same joints/bodies, so all base-task logic holds.
* One ``VisuoTactileSensorCfg`` (force field only, no tactile camera) is
  created per fingertip against the rotating nut/handle body.
* The per-taxel (normal, shear_x, shear_y) arrays are average-pooled to
  ``tactile_array_pool`` and appended to ``priv_info`` — the HORA Stage-1
  teacher observation. The actor obs (141 dims) is untouched.

The TacSL force field queries the contact object's SDF, so the tactile env
switches the nut/handle collision mesh approximation to "sdf" at setup time
(the base tasks keep the URDF importer's convex collision).
"""

from __future__ import annotations

from pathlib import Path

from isaaclab.utils import configclass

from .revo3_hand_screw_env_cfg import (
    Revo3HandScrewDriverEnvCfg,
    Revo3HandScrewNutBoltEnvCfg,
    Revo3HandValveDriver40EnvCfg,
    Revo3HandVavleDriverEnvCfg,
)

try:
    from isaaclab_assets.sensors import GELSIGHT_R15_CFG
    from isaaclab_contrib.sensors.tacsl_sensor import VisuoTactileSensorCfg

    from .tacsl_sensor import Revo3VisuoTactileSensor
except ImportError:
    GELSIGHT_R15_CFG = None
    VisuoTactileSensorCfg = None
    Revo3VisuoTactileSensor = None

_REPO_ROOT = Path(__file__).resolve().parents[6]
REVO3_TACTILE_USD = str(_REPO_ROOT / "assets" / "usd" / "tactile_dexscrew" / "revo3_right_tactile.usda")

# Same order as Revo3HandScrewEnvCfg.fingertip_body_names (thumb first).
TACTILE_FINGER_ORDER = ("thumb", "index", "middle", "ring", "little")
TACTILE_TIP_BODIES = tuple(f"right_{finger}_tip_Link" for finger in TACTILE_FINGER_ORDER)
TACTILE_VIS_SENSOR_NAMES = tuple(f"{finger}_tactile_sensor" for finger in TACTILE_FINGER_ORDER)


@configclass
class Revo3HandScrewTactileMixinCfg:
    """Adds TacSL fingertip array sensing (teacher/priv obs only) to a screw task cfg."""

    # ---- tactile array (TacSL force field) ----
    tactile_array_size = (16, 16)
    # pooled grid appended to priv_info: per finger pool_h*pool_w*(normal+2 shear)
    tactile_array_pool = (4, 4)
    # F_n = normal_contact_stiffness(=1.0) * penetration_depth[m] ~ 1e-3; rescale to O(1)
    tactile_force_scale = 200.0
    tactile_force_clip = 5.0
    # PhysX SDF resolution for the rotating nut/handle collision mesh
    tactile_sdf_resolution = 256
    tactile_debug_vis = False
    # 这个是用来控制是否可视化整列传感点

    tactile_tip_body_names = TACTILE_TIP_BODIES
    tactile_vis_sensor_names = TACTILE_VIS_SENSOR_NAMES
    tactile_sensor = []

    # filled by _configure_revo3_tactile()
    tactile_priv_offset = 0
    tactile_priv_dim = 0

    # ---- Stage2 DAgger student observation (real-robot sensing only) ----
    # proprio frame = 21 joint_pos + 21 current joint targets (no contact forces)
    student_proprio_frame_dim = 42
    student_proprio_history_len = 3
    student_proprio_history_dim = 126
    # binary tactile frame = the 240 pooled TacSL channels thresholded to {0, 1}
    student_tactile_frame_dim = 240
    student_tactile_history_len = 10
    student_tactile_raw_history_dim = 2400
    student_tactile_encoder_output_dim = 240
    student_obs_dim = 366
    # contact threshold on |scaled pooled tactile| (tactile_force_scale units);
    # single-taxel penetration ~1e-3 m scales to 0.2, diluted 16x by 4x4 avg-pool
    student_tactile_contact_threshold = 0.01

    def __post_init__(self):
        super().__post_init__()
        self._configure_revo3_tactile()

    def _configure_revo3_tactile(self):
        if VisuoTactileSensorCfg is None or Revo3VisuoTactileSensor is None or GELSIGHT_R15_CFG is None:
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

        # extend priv_info (HORA teacher observation) with the pooled arrays
        base_priv_dim = int(self.priv_info_dim)
        self.tactile_priv_offset = base_priv_dim
        self.tactile_priv_dim = len(self.tactile_tip_body_names) * pool_rows * pool_cols * 3
        self.priv_info_dim = base_priv_dim + self.tactile_priv_dim

        # student observation dims must stay consistent with the sim quantities
        if self.student_tactile_frame_dim != self.tactile_priv_dim:
            raise ValueError(
                f"student_tactile_frame_dim ({self.student_tactile_frame_dim}) must equal the pooled "
                f"TacSL tactile dim ({self.tactile_priv_dim})"
            )
        if self.student_proprio_frame_dim != 2 * self.action_space:
            raise ValueError(
                f"student_proprio_frame_dim ({self.student_proprio_frame_dim}) must equal "
                f"joint_pos + targets = 2 * action_space ({2 * self.action_space})"
            )
        if self.student_proprio_history_dim != self.student_proprio_history_len * self.student_proprio_frame_dim:
            raise ValueError("student_proprio_history_dim != history_len * frame_dim")
        if self.student_tactile_raw_history_dim != self.student_tactile_history_len * self.student_tactile_frame_dim:
            raise ValueError("student_tactile_raw_history_dim != history_len * frame_dim")
        if self.student_obs_dim != self.student_proprio_history_dim + self.student_tactile_encoder_output_dim:
            raise ValueError("student_obs_dim != proprio_history_dim + tactile_encoder_output_dim")

        self.tactile_sensor = []
        for tip_body in self.tactile_tip_body_names:
            elastomer_path = f"/World/envs/env_.*/hand/{tip_body}/tactile_elastomer"
            self.tactile_sensor.append(
                VisuoTactileSensorCfg(
                    class_type=Revo3VisuoTactileSensor,
                    prim_path=f"{elastomer_path}/tactile_sensor",
                    history_length=0,
                    render_cfg=GELSIGHT_R15_CFG,
                    enable_camera_tactile=False,
                    enable_force_field=True,
                    tactile_array_size=self.tactile_array_size,
                    tactile_margin=0.003,
                    contact_object_prim_path_expr=f"/World/envs/env_.*/object/{self.nut_body_name}",
                    normal_contact_stiffness=1.0,
                    friction_coefficient=2.0,
                    tangential_stiffness=0.1,
                    camera_cfg=None,
                    debug_vis=self.tactile_debug_vis,
                )
            )


@configclass
class Revo3HandScrewNutBoltTactileEnvCfg(Revo3HandScrewTactileMixinCfg, Revo3HandScrewNutBoltEnvCfg):
    """Nut-bolt task + TacSL fingertip arrays in the teacher observation."""

    pass


@configclass
class Revo3HandScrewDriverTactileEnvCfg(Revo3HandScrewTactileMixinCfg, Revo3HandScrewDriverEnvCfg):
    """Screwdriver task + TacSL fingertip arrays in the teacher observation."""

    pass


@configclass
class Revo3HandVavleDriverTactileEnvCfg(Revo3HandScrewTactileMixinCfg, Revo3HandVavleDriverEnvCfg):
    """Five-finger hexagonal valve task + TacSL fingertip arrays."""

    pass


@configclass
class Revo3HandValveDriver40TactileEnvCfg(Revo3HandScrewTactileMixinCfg, Revo3HandValveDriver40EnvCfg):
    """The unchanged tactile valve task with a 40 mm handle circumradius."""

    pass
