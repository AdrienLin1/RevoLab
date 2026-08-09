# Copyright (c) 2026, BrainCo.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configs for tactile ball/cylinder continuous in-hand rotation.

The tasks retain the HORA in-hand rotation dynamics and add the same TacSL
teacher/student observation contract used by the tactile screw tasks. Positive
axis angular velocity remains unbounded in the rotation reward so faster stable
rotation always receives additional credit.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.utils import configclass

from .assets import (
    BALL_INIT_POS,
    CYLINDER_INIT_POS,
    OBJECT_INIT_ROT,
    REVO3_HAND_BALL_CFG,
    REVO3_HAND_CYLINDER_CFG,
)
from .revo3_hand_hora_env_cfg import Revo3HandHoraEnvCfg
from ..hora_screw.revo3_hand_screw_tactile_env_cfg import (
    REVO3_TACTILE_USD,
    Revo3HandScrewTactileMixinCfg,
)


_TACTILE_RIGID = sim_utils.RigidBodyPropertiesCfg(
    kinematic_enabled=False,
    disable_gravity=False,
    enable_gyroscopic_forces=True,
    solver_position_iteration_count=8,
    solver_velocity_iteration_count=0,
    sleep_threshold=0.005,
    stabilization_threshold=0.0025,
    max_depenetration_velocity=1000.0,
)
_TACTILE_MASS = sim_utils.MassPropertiesCfg(mass=0.10)
_TACTILE_COLLISION = sim_utils.CollisionPropertiesCfg(
    collision_enabled=True,
    contact_offset=0.002,
    rest_offset=0.0,
)
_TACTILE_MATERIAL = sim_utils.RigidBodyMaterialCfg(
    static_friction=1.0,
    dynamic_friction=1.0,
)


TACTILE_BALL_OBJECT_CFG = RigidObjectCfg(
    prim_path="/World/envs/env_.*/object",
    spawn=sim_utils.MeshSphereCfg(
        radius=0.030,
        rigid_props=_TACTILE_RIGID,
        mass_props=_TACTILE_MASS,
        collision_props=_TACTILE_COLLISION,
        physics_material=_TACTILE_MATERIAL,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.55, 0.95)),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=BALL_INIT_POS, rot=OBJECT_INIT_ROT),
)


TACTILE_CYLINDER_OBJECT_CFG = RigidObjectCfg(
    prim_path="/World/envs/env_.*/object",
    spawn=sim_utils.MeshCylinderCfg(
        radius=0.030,
        height=0.070,
        axis="Z",
        rigid_props=_TACTILE_RIGID,
        mass_props=_TACTILE_MASS,
        collision_props=_TACTILE_COLLISION,
        physics_material=_TACTILE_MATERIAL,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.80, 0.55, 0.20)),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=CYLINDER_INIT_POS, rot=OBJECT_INIT_ROT),
)


@configclass
class Revo3HandTactileRotateEnvCfg(
    Revo3HandScrewTactileMixinCfg,
    Revo3HandHoraEnvCfg,
):
    """Add TacSL observations to continuous HORA rotation."""

    # Teacher actor frame: joint positions, current targets, and five legacy
    # contact magnitudes. Rotation speed is intentionally not observed.
    observation_space = 141  # 3 frames x (21 + 21 + 5)

    # Restore the original HORA rotation objective while removing its positive
    # speed ceiling. Reverse rotation remains bounded by angvel_clip_min.
    rotate_reward_scale = 2.5
    unbounded_positive_rotate_reward = True
    object_linvel_penalty_scale = -0.6
    object_pos_reward_scale = 0.003

    # All five fingers remain active for ball/cylinder in-hand rotation.
    masked_action_joint_names = ()
    pack_active_tactile_only = True
    enable_visible_contact_reward = False
    enable_coord_endogenous_reward = False
    multi_contact_reward_scale = 0.0

    # Student proprio contains joint state only; tactile history is separate.
    student_proprio_frame_dim = 42
    student_proprio_command_dim = 0
    student_proprio_history_len = 3
    student_proprio_history_dim = 126
    student_obs_dim = 190

    # The generated object mesh itself is the contact rigid body.
    tactile_contact_object_prim_path_expr = "/World/envs/env_.*/object"

    # Match hora_screw's persistent five-bucket geometry randomization. The
    # concrete task controls which mesh axes are scaled.
    randomize_object_size = True
    object_size_scale_levels = (0.8, 0.9, 1.0, 1.1, 1.2)
    object_size_scale_axes = (True, True, True)

    # Shared reset/domain randomization from hora_screw. Screw-only passive
    # joint resistance is intentionally excluded for free rigid objects.
    action_delay = (0.0, 1.0)
    reset_joint_noise_frac = 0.1
    randomize_object_xy_position = True
    object_xy_position_noise = 0.005
    enable_contact_noise = True
    contact_force_noise_frac = 0.02
    randomize_p_gain_scale_lower = 0.9
    randomize_p_gain_scale_upper = 1.1
    randomize_d_gain_scale_lower = 0.9
    randomize_d_gain_scale_upper = 1.1

    def __post_init__(self):
        """Configure the base HORA task and then attach TacSL sensors."""
        super().__post_init__()
        delay_low, delay_high = self.action_delay
        if not 0.0 <= float(delay_low) <= float(delay_high) <= 1.0:
            raise ValueError(
                "action_delay must satisfy 0 <= low <= high <= 1, got "
                f"{self.action_delay}"
            )


@configclass
class Revo3HandTactileRotateBallEnvCfg(Revo3HandTactileRotateEnvCfg):
    """Continuous tactile in-hand rotation with a ball."""

    object_size_scale_axes = (True, True, True)

    def __post_init__(self):
        """Select ball grasp, geometry, and the tactile hand overlay."""
        super().__post_init__()
        self.robot_cfg = REVO3_HAND_BALL_CFG.replace(
            spawn=REVO3_HAND_BALL_CFG.spawn.replace(usd_path=REVO3_TACTILE_USD)
        )
        self.object_cfg = TACTILE_BALL_OBJECT_CFG
        self.grasp_cache_path = "assets/grasp_cache/hora/revo3_right_grasp_ball"


@configclass
class Revo3HandTactileRotateCylinderEnvCfg(Revo3HandTactileRotateEnvCfg):
    """Continuous tactile in-hand rotation with a cylinder."""

    object_size_scale_axes = (True, True, False)

    def __post_init__(self):
        """Select cylinder grasp, geometry, and the tactile hand overlay."""
        super().__post_init__()
        self.robot_cfg = REVO3_HAND_CYLINDER_CFG.replace(
            spawn=REVO3_HAND_CYLINDER_CFG.spawn.replace(usd_path=REVO3_TACTILE_USD)
        )
        self.object_cfg = TACTILE_CYLINDER_OBJECT_CFG
        self.grasp_cache_path = "assets/grasp_cache/hora/revo3_right_grasp_cylinder"
