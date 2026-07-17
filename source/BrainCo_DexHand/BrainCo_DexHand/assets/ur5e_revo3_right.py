# Copyright (c) 2026, BrainCo.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg


REPO_ROOT = Path(__file__).resolve().parents[4]
ASSETS_DIR = REPO_ROOT / "assets"

UR5E_REVO3_RIGHT_USD = ASSETS_DIR / "usd/ur5e_revo3/ur5e_revo3_right.usd"
UR5E_REVO3_RIGHT_TACTILE_USD = ASSETS_DIR / "usd/ur5e_revo3/ur5e_revo3_right_tactile.usda"

UR5E_ARM_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

REVO3_RIGHT_FINGER_JOINT_NAMES = [
    "right_thumb_CMP_joint",
    "right_thumb_CMR_joint",
    "right_thumb_MCP_joint",
    "right_thumb_PIP_joint",
    "right_thumb_DIP_joint",
    "right_index_MPR_joint",
    "right_index_MCP_joint",
    "right_index_PIP_joint",
    "right_index_DIP_joint",
    "right_middle_MPR_joint",
    "right_middle_MCP_joint",
    "right_middle_PIP_joint",
    "right_middle_DIP_joint",
    "right_ring_MPR_joint",
    "right_ring_MCP_joint",
    "right_ring_PIP_joint",
    "right_ring_DIP_joint",
    "right_little_MPR_joint",
    "right_little_MCP_joint",
    "right_little_PIP_joint",
    "right_little_DIP_joint",
]

UR5E_DEFAULT_JOINT_POS = {
    "shoulder_pan_joint": 0.0,
    "shoulder_lift_joint": -1.35,
    "elbow_joint": 1.75,
    "wrist_1_joint": -1.95,
    "wrist_2_joint": -1.57,
    "wrist_3_joint": 0.0,
}

REVO3_RIGHT_DEFAULT_FINGER_JOINT_POS = {
    "right_thumb_CMP_joint": 0.43,
    "right_thumb_CMR_joint": 1.57,
    "right_thumb_MCP_joint": 0.0,
    "right_thumb_PIP_joint": 0.13,
    "right_thumb_DIP_joint": 0.28,
    "right_index_MPR_joint": 0.0,
    "right_index_MCP_joint": 0.5,
    "right_index_PIP_joint": 0.55,
    "right_index_DIP_joint": 0.31,
    "right_middle_MPR_joint": 0.0,
    "right_middle_MCP_joint": 0.5,
    "right_middle_PIP_joint": 0.55,
    "right_middle_DIP_joint": 0.31,
    "right_ring_MPR_joint": 0.0,
    "right_ring_MCP_joint": 0.5,
    "right_ring_PIP_joint": 0.55,
    "right_ring_DIP_joint": 0.31,
    "right_little_MPR_joint": 0.0,
    "right_little_MCP_joint": 0.5,
    "right_little_PIP_joint": 0.55,
    "right_little_DIP_joint": 0.31,
}

UR5E_REVO3_DEFAULT_JOINT_POS = {
    **UR5E_DEFAULT_JOINT_POS,
    **REVO3_RIGHT_DEFAULT_FINGER_JOINT_POS,
}

UR5E_ARM_EFFORT_LIMITS = {
    "shoulder_pan_joint": 150.0,
    "shoulder_lift_joint": 150.0,
    "elbow_joint": 150.0,
    "wrist_1_joint": 28.0,
    "wrist_2_joint": 28.0,
    "wrist_3_joint": 28.0,
}
UR5E_ARM_VELOCITY_LIMITS = {name: 3.1416 for name in UR5E_ARM_JOINT_NAMES}
REVO3_RIGHT_FINGER_EFFORT_LIMITS = {name: 5.0 for name in REVO3_RIGHT_FINGER_JOINT_NAMES}
REVO3_RIGHT_FINGER_VELOCITY_LIMITS = {name: 10.0 for name in REVO3_RIGHT_FINGER_JOINT_NAMES}


UR5E_REVO3_RIGHT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(UR5E_REVO3_RIGHT_USD),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            retain_accelerations=True,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1000.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=1,
            sleep_threshold=0.005,
            stabilization_threshold=0.0005,
            fix_root_link=True,
        ),
        joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos=UR5E_REVO3_DEFAULT_JOINT_POS,
    ),
    actuators={
        "ur5e_arm": ImplicitActuatorCfg(
            joint_names_expr=UR5E_ARM_JOINT_NAMES,
            stiffness=100.0,
            damping=10.0,
            friction=1.0,
            effort_limit_sim=UR5E_ARM_EFFORT_LIMITS,
            velocity_limit_sim=UR5E_ARM_VELOCITY_LIMITS,
        ),
        "revo3_right_fingers": ImplicitActuatorCfg(
            joint_names_expr=REVO3_RIGHT_FINGER_JOINT_NAMES,
            stiffness=3.0,
            damping=0.1,
            friction=0.01,
            effort_limit_sim=REVO3_RIGHT_FINGER_EFFORT_LIMITS,
            velocity_limit_sim=REVO3_RIGHT_FINGER_VELOCITY_LIMITS,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)

UR5E_REVO3_RIGHT_TACTILE_CFG = UR5E_REVO3_RIGHT_CFG.replace(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(UR5E_REVO3_RIGHT_TACTILE_USD),
        activate_contact_sensors=True,
        rigid_props=UR5E_REVO3_RIGHT_CFG.spawn.rigid_props,
        articulation_props=UR5E_REVO3_RIGHT_CFG.spawn.articulation_props,
        joint_drive_props=UR5E_REVO3_RIGHT_CFG.spawn.joint_drive_props,
    )
)
