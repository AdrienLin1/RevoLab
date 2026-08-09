# Copyright (c) 2026, BrainCo.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""UR5e + Revo3 right-hand Dexsuite grasp/lift environment configuration."""

import copy

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from BrainCo_DexHand.assets.ur5e_revo3_right import (
    REVO3_RIGHT_FINGER_JOINT_NAMES,
    UR5E_ARM_JOINT_NAMES,
    UR5E_REVO3_RIGHT_CFG,
    UR5E_REVO3_RIGHT_TACTILE_CFG,
)

from . import dexsuite_env_cfg_grasp_tianji as dexsuite
from . import mdp
from .config.Revo3.dexsuite_revo3_env_cfg_grasp import (
    Revo3ReorientRewardCfg,
    disable_collision_pairs_filtered,
)
from .config.Revo3.tactile import (
    TACTILE_CUBE_USD_PATH,
    TACTILE_FORCE_SENSOR_NAMES,
    TACTILE_TIP_BODIES,
    Revo3TacslTactileMixinCfg,
    make_tactile_force_sensor_cfgs,
)


UR5E_REVO3_PALM_BODY_NAME = "right_palm"
UR5E_REVO3_HAND_TIP_BODIES = [
    UR5E_REVO3_PALM_BODY_NAME,
    *TACTILE_TIP_BODIES,
]
UR5E_REVO3_TACTILE_FORCE_SENSOR_NAMES = list(TACTILE_FORCE_SENSOR_NAMES)
DEXSUITE_TABLE_TOP_Z = 0.76
DEXSUITE_CUBE_SIZE = 0.06
UR5E_REVO3_ROOT_XY = (0.5, 0.0)
UR5E_REVO3_OBJECT_INIT_POS = (0.0, 0.0, DEXSUITE_TABLE_TOP_Z + DEXSUITE_CUBE_SIZE * 0.5)


@configclass
class UR5eRevo3RelJointPosActionCfg:
    """Relative joint position control for the UR5e arm and Revo3 right hand."""

    action = mdp.RelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=[*UR5E_ARM_JOINT_NAMES, *REVO3_RIGHT_FINGER_JOINT_NAMES],
        scale=0.1,
    )


@configclass
class UR5eEventCfg(dexsuite.EventCfg):
    """UR5e events inherit the complete Tianji Dexsuite event set without adding or removing terms."""

    pass


@configclass
class UR5eRevo3MixinCfg:
    """Mixin that attaches a UR5e arm with a Revo3 right hand."""

    rewards: Revo3ReorientRewardCfg = Revo3ReorientRewardCfg()
    actions: UR5eRevo3RelJointPosActionCfg = UR5eRevo3RelJointPosActionCfg()

    def __post_init__(self: dexsuite.DexsuiteReorientEnvCfg):
        super().__post_init__()

        self.commands.object_pose.body_name = UR5E_REVO3_PALM_BODY_NAME
        self.commands.object_pose.debug_vis = True
        self.commands.object_pose.ranges.pos_x = (0.35, 0.45)
        self.commands.object_pose.ranges.pos_y = (-0.06, 0.06)
        self.commands.object_pose.ranges.pos_z = (0.20, 0.35)

        self.scene.robot = UR5E_REVO3_RIGHT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        base_joint_pos = dict(UR5E_REVO3_RIGHT_CFG.init_state.joint_pos) if (
            hasattr(UR5E_REVO3_RIGHT_CFG.init_state, "joint_pos")
            and UR5E_REVO3_RIGHT_CFG.init_state.joint_pos is not None
        ) else {}
        self.scene.robot.init_state = ArticulationCfg.InitialStateCfg(
            pos=(*UR5E_REVO3_ROOT_XY, DEXSUITE_TABLE_TOP_Z),
            rot=(0.0, 0.0, 0.0, 1.0),
            joint_pos=base_joint_pos,
        )
        self.scene.object = copy.deepcopy(self.scene.object)
        self.scene.object.init_state = RigidObjectCfg.InitialStateCfg(pos=UR5E_REVO3_OBJECT_INIT_POS)
        self.events.reset_object.params["pose_range"]["x"] = [0.05, 0.15]
        self.events.reset_object.params["pose_range"]["y"] = [-0.05, 0.05]
        self.events.reset_object.params["pose_range"]["z"] = [0.0, 0.005]

        self.events.reset_robot_joints = EventTerm(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[*UR5E_ARM_JOINT_NAMES, *REVO3_RIGHT_FINGER_JOINT_NAMES],
                ),
                "position_range": [0.0, 0.0],
                "velocity_range": [0.0, 0.0],
            },
        )
        self.events.reset_robot_wrist_joint = None

        for sensor_name, sensor_cfg in make_tactile_force_sensor_cfgs("ur5e_robot/revo3_right").items():
            setattr(self.scene, sensor_name, sensor_cfg)

        self.observations.proprio.contact = ObsTerm(
            func=mdp.fingers_contact_force_b,
            params={"contact_sensor_names": UR5E_REVO3_TACTILE_FORCE_SENSOR_NAMES},
            clip=(-20.0, 20.0),
        )

        self.observations.proprio.hand_tips_state_b.params["body_asset_cfg"].body_names = UR5E_REVO3_HAND_TIP_BODIES

        if hasattr(self.rewards, "fingers_to_object"):
            self.rewards.fingers_to_object.params["asset_cfg"] = SceneEntityCfg(
                "robot",
                body_names=UR5E_REVO3_HAND_TIP_BODIES,
            )
        if hasattr(self.rewards, "fingers_to_object_delta"):
            self.rewards.fingers_to_object_delta.params["asset_cfg"] = SceneEntityCfg(
                "robot",
                body_names=UR5E_REVO3_HAND_TIP_BODIES,
            )

        self.events.disable_collision_pairs_filtered = EventTerm(
            func=disable_collision_pairs_filtered,
            mode="prestartup",
            params={
                "asset_root": "Robot/ur5e_robot/revo3_right",
                "group_links": {
                    "palm": ["right_hand_base_link", "right_palm"],
                    "mcp": [
                        "right_thumb_MCP_Link",
                        "right_index_MCP_Link",
                        "right_middle_MCP_Link",
                        "right_ring_MCP_Link",
                        "right_little_MCP_Link",
                    ],
                },
                "filtered_group_pairs": [("palm", "mcp")],
                "self_filtered_groups": [],
            },
        )


@configclass
class DexsuiteUR5eRevo3LiftEnvCfg(UR5eRevo3MixinCfg, dexsuite.DexsuiteLiftEnvCfg):
    """Configuration for UR5e + Revo3 right-hand lift training."""

    events: UR5eEventCfg = UR5eEventCfg()


@configclass
class DexsuiteUR5eRevo3LiftEnvCfg_PLAY(UR5eRevo3MixinCfg, dexsuite.DexsuiteLiftEnvCfg_PLAY):
    """Configuration for UR5e + Revo3 right-hand lift evaluation/play."""

    events: UR5eEventCfg = UR5eEventCfg()

    def __post_init__(self):
        super().__post_init__()
        self.commands.object_pose.resampling_time_range = (2.0, 3.0)
        self.commands.object_pose.debug_vis = True
        self.commands.object_pose.position_only = True


@configclass
class UR5eRevo3TactileMixinCfg(Revo3TacslTactileMixinCfg):
    """Mixin that enables TacSL tactile cameras on the UR5e-mounted Revo3 hand."""

    enable_tactile_camera: bool = False
    enable_tactile_depth: bool = False
    enable_tactile_rgb: bool = False
    enable_tactile_force_field: bool = True
    tactile_debug_vis: bool = True
    tactile_visualize_sdf_closest_pts: bool = False
    tactile_body_path_prefix: str = "ur5e_robot/revo3_right"

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot.spawn = UR5E_REVO3_RIGHT_TACTILE_CFG.spawn
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(TACTILE_CUBE_USD_PATH),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=0,
                    disable_gravity=False,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=UR5E_REVO3_OBJECT_INIT_POS),
        )


@configclass
class DexsuiteUR5eRevo3LiftTactileEnvCfg(UR5eRevo3TactileMixinCfg, DexsuiteUR5eRevo3LiftEnvCfg):
    """Configuration for UR5e + Revo3 right-hand lift training with TacSL tactile sensors."""

    pass


@configclass
class DexsuiteUR5eRevo3LiftTactileEnvCfg_PLAY(UR5eRevo3TactileMixinCfg, DexsuiteUR5eRevo3LiftEnvCfg_PLAY):
    """Configuration for UR5e + Revo3 right-hand lift play with TacSL tactile sensors."""
