"""Asset configs for Revo3 right hand HORA screw/valve tasks.

Ported from dexscrew XHandHoraNutBolt / XHandHoraScrewDriver (Isaac Gym) to Isaac Lab.

Geometry (matches the dexscrew setup): the screw object stands UPRIGHT on the
ground — base disc on the floor, vertical bolt/shaft, rotating nut/handle sleeve
at the top. The hand FLOATS in the air above the object, palm facing down, and
finger-gaits the sleeve around the vertical screw axis.

The hand pose is the proven HORA cylinder-rotation grasp flipped 180 deg about
the world Y axis: palm-up with the object 0.135 m above the root becomes
palm-down with the object 0.135 m below the root. The relative hand-object
grasp geometry (validated by the cylinder task) is preserved exactly.
"""
from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sim.converters import UrdfConverterCfg

from .xy_stage import (
    XY_STAGE_ACTUATOR_EXPR,
    XY_STAGE_ACTUATOR_GROUP,
    XY_STAGE_JOINT_NAMES,
)
from .yaw_stage import (
    YAW_STAGE_ACTUATOR_EXPR,
    YAW_STAGE_ACTUATOR_GROUP,
    YAW_STAGE_JOINT_NAME,
)

_REPO_ROOT = Path(__file__).resolve().parents[6]
_REVO3_USD = str(_REPO_ROOT / "assets" / "usd" / "revo3_right.usd")
_TRINUT_URDF = str(_REPO_ROOT / "assets" / "urdf" / "screw" / "trinut" / "trinut.urdf")
_DRIVER_URDF = str(_REPO_ROOT / "assets" / "urdf" / "screw" / "driver" / "driver_8.urdf")
_VAVLE_DRIVER_URDF = str(
    _REPO_ROOT / "assets" / "urdf" / "screw" / "vavledriver" / "vavledriver_hex.urdf"
)
_VALVE_DRIVER_25_URDF = str(
    _REPO_ROOT / "assets" / "urdf" / "screw" / "vavledriver" / "valvedriver_hex_25.urdf"
)
_VALVE_DRIVER_40_URDF = str(
    _REPO_ROOT / "assets" / "urdf" / "screw" / "vavledriver" / "valvedriver_hex_40.urdf"
)

# Hand orientation: HORA palm-up grasp quat (0.59636781, 0.37992820, -0.37992820,
# 0.59636781) pre-rotated by R_y(180 deg) -> palm faces straight down (-Z).
HAND_INIT_ROT = (0.37992820, 0.59636781, 0.59636781, -0.37992820)
# Hand root positions: grasp center + (0, 0.08, 0.135), i.e. the flipped image of
# the palm-up arrangement where the grasp sat at root + (0, -0.08, 0.135).
#   trinut sleeve center: (0, 0.015, 0.0687) -> hand root (0, 0.08, 0.204)
#   driver handle center: (0, 0, 0.07)   -> hand root (0, 0.08, 0.205)
HAND_INIT_POS_NUTBOLT = (-0.015, 0.08, 0.204)  # -x方向靠近大拇指   
HAND_INIT_POS_DRIVER = (0.005, 0.08, 0.205) #  (-0.010, 0.08, 0.205）
# The valve grasp center is at z=0.06.  Move the palm 2 mm closer in Y than
# the cylinder-derived pose so all five fingers begin near the 52 mm flats.
HAND_INIT_POS_VAVLE_DRIVER = (0.0, 0.078, 0.195)

# Screw objects stand upright near the env origin, base disc on the ground.
OBJECT_INIT_ROT = (1.0, 0.0, 0.0, 0.0)
# Shift NutBolt 15 mm in world +Y toward the index finger's flexion plane.
TRINUT_INIT_POS = (0.0, 0.0, 0.0)
DRIVER_INIT_POS = (0.0, 0.0, 0.0)
VAVLE_DRIVER_INIT_POS = (0.0, 0.0, 0.0)


def _make_hand_cfg(joint_pos: dict[str, float], init_pos: tuple[float, float, float]) -> ArticulationCfg:
    return ArticulationCfg(
        prim_path="/World/envs/env_.*/hand",
        spawn=sim_utils.UsdFileCfg(
            usd_path=_REVO3_USD,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                retain_accelerations=False,
                enable_gyroscopic_forces=False,
                angular_damping=0.01,
                max_depenetration_velocity=1000.0,
                max_contact_impulse=1e32,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=1,
                sleep_threshold=0.005,
                stabilization_threshold=0.0005,
                fix_root_link=True,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=init_pos,
            rot=HAND_INIT_ROT,
            joint_pos=joint_pos,
        ),
        actuators={
            "fingers": ImplicitActuatorCfg(
                joint_names_expr=["right_.*"],
                effort_limit_sim=1.0,
                stiffness=0.0,
                damping=0.0,
                friction=0.01,
                armature=0.001,
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )


# Grasp init poses: derived from the proven HORA cylinder grasp (r=0.03) with
# slightly more finger flexion because the screw sleeves are thinner
# (trinut circumradius ~0.021, driver handle R = 0.016 m at scale 1.0).
# Keep index MPR neutral; NutBolt alignment is tuned through the object's world
# Y position instead of changing the finger's abduction/adduction angle.
REVO3_HAND_NUTBOLT_CFG = _make_hand_cfg({
    "right_thumb_CMP_joint":  1.65, "right_thumb_CMR_joint":  1.35,
    "right_thumb_MCP_joint":  0.45, "right_thumb_PIP_joint":  0.30,
    "right_thumb_DIP_joint":  0.05,
    "right_index_MPR_joint":  0.00, "right_index_MCP_joint":  1.30,
    "right_index_PIP_joint":  0.40, "right_index_DIP_joint":  0.05,
    "right_middle_MPR_joint": 0.00, "right_middle_MCP_joint": 1.05,
    "right_middle_PIP_joint": 0.30, "right_middle_DIP_joint": 0.05,
    "right_ring_MPR_joint":   0.00, "right_ring_MCP_joint":   0.00,
    "right_ring_PIP_joint":   0.00, "right_ring_DIP_joint":   0.00,
    "right_little_MPR_joint": 0.00, "right_little_MCP_joint": 0.00,
    "right_little_PIP_joint": 0.00, "right_little_DIP_joint": 0.00,
}, HAND_INIT_POS_NUTBOLT)

REVO3_HAND_DRIVER_CFG = _make_hand_cfg({
    "right_thumb_CMP_joint":  1.65, "right_thumb_CMR_joint":  1.35,
    "right_thumb_MCP_joint":  0.55, "right_thumb_PIP_joint":  0.35,
    "right_thumb_DIP_joint":  0.05,
    "right_index_MPR_joint": -0.25, "right_index_MCP_joint":  1.35,
    "right_index_PIP_joint":  0.45, "right_index_DIP_joint":  0.05,
    "right_middle_MPR_joint": 0.00, "right_middle_MCP_joint": 1.20,
    "right_middle_PIP_joint": 0.40, "right_middle_DIP_joint": 0.05,
    "right_ring_MPR_joint":   0.20, "right_ring_MCP_joint":   1.35,
    "right_ring_PIP_joint":   0.45, "right_ring_DIP_joint":   0.05,
    "right_little_MPR_joint": 0.00, "right_little_MCP_joint": 0.00,
    "right_little_PIP_joint": 0.00, "right_little_DIP_joint": 0.00,
}, HAND_INIT_POS_DRIVER)

# Five-finger wrap for the larger valve handle (hex circumradius 0.035 m).  Keep
# the index MPR neutral so it curls vertically toward the +X flat, and add a
# little middle-PIP flexion so its tactile surface starts near the -Y flat.
REVO3_HAND_VAVLE_DRIVER_CFG = _make_hand_cfg({
    "right_thumb_CMP_joint":  1.60, "right_thumb_CMR_joint":  1.30,
    "right_thumb_MCP_joint":  0.38, "right_thumb_PIP_joint":  0.22,
    "right_thumb_DIP_joint":  0.03,
    "right_index_MPR_joint":  -0.20, "right_index_MCP_joint":  1.18,
    "right_index_PIP_joint":  0.28, "right_index_DIP_joint":  0.03,
    "right_middle_MPR_joint": 0.00, "right_middle_MCP_joint": 0.94,
    "right_middle_PIP_joint": 0.35, "right_middle_DIP_joint": 0.03,
    "right_ring_MPR_joint":   0.18, "right_ring_MCP_joint":   0.96,
    "right_ring_PIP_joint":   0.24, "right_ring_DIP_joint":   0.03,
    "right_little_MPR_joint": 0.22, "right_little_MCP_joint": 1.18,
    "right_little_PIP_joint": 0.30, "right_little_DIP_joint": 0.03,
}, HAND_INIT_POS_VAVLE_DRIVER)


def make_xy_stage_hand_cfg(
    base_cfg: ArticulationCfg,
    *,
    joint_limit: float,
    effort_limit: float,
    velocity_limit: float,
    armature: float = 0.0,
    friction: float = 0.0,
) -> ArticulationCfg:
    """Return a hand config that also owns the two-axis translation stage.

    The stage joints are authored into the articulation at scene-setup time
    (see ``Revo3HandScrewTactileXYEnv._author_robot_stage_overrides``); this
    config only adds the matching actuator group and their zero reset pose.

    The stage actuator is deliberately a separate group with the
    ``stage_.*_joint`` pattern so it can never be captured by the ``right_.*``
    finger group. Like the fingers it runs with zero implicit PD stiffness and
    damping: the environment applies explicit, effort-limited PD torques, and
    ``effort_limit_sim`` enforces the same bound inside PhysX.

    Args:
        base_cfg: Hand configuration without the stage.
        joint_limit: Prismatic hard limit (metres) used for validation only.
        effort_limit: Maximum stage actuator force, newtons.
        velocity_limit: Maximum stage joint velocity, m/s.
        armature: Extra reflected inertia on the stage joints, kg.
        friction: Stage joint friction force, newtons.

    Returns:
        A new ``ArticulationCfg`` with the stage actuator and reset pose.
    """
    if joint_limit <= 0.0:
        raise ValueError(f"XY stage joint_limit must be positive, got {joint_limit}")
    if effort_limit <= 0.0:
        raise ValueError(f"XY stage effort_limit must be positive, got {effort_limit}")
    if velocity_limit <= 0.0:
        raise ValueError(f"XY stage velocity_limit must be positive, got {velocity_limit}")

    joint_pos = dict(base_cfg.init_state.joint_pos)
    for joint_name in XY_STAGE_JOINT_NAMES:
        joint_pos[joint_name] = 0.0
    actuators = dict(base_cfg.actuators)
    if XY_STAGE_ACTUATOR_GROUP in actuators:
        raise ValueError(
            f"Actuator group {XY_STAGE_ACTUATOR_GROUP!r} already exists on the base hand"
        )
    actuators[XY_STAGE_ACTUATOR_GROUP] = ImplicitActuatorCfg(
        joint_names_expr=[XY_STAGE_ACTUATOR_EXPR],
        effort_limit_sim=float(effort_limit),
        velocity_limit_sim=float(velocity_limit),
        stiffness=0.0,
        damping=0.0,
        friction=float(friction),
        armature=float(armature),
    )
    return base_cfg.replace(
        init_state=base_cfg.init_state.replace(joint_pos=joint_pos),
        actuators=actuators,
    )


def make_yaw_stage_hand_cfg(
    base_cfg: ArticulationCfg,
    *,
    joint_limit: float,
    effort_limit: float,
    velocity_limit: float,
    armature: float = 0.0,
    friction: float = 0.0,
) -> ArticulationCfg:
    """Return a hand config that also owns the world-Z yaw stage joint.

    The yaw joint itself is authored into the articulation at scene-setup time
    (see ``Revo3HandYawStageMixin._author_yaw_stage_joint``); this config only
    adds the matching actuator group and its zero reset pose.

    The yaw actuator is a **separate** group matched by the exact expression
    ``stage_yaw_joint``. It must never share the XY group: the XY group carries
    a 120 N *linear* effort limit while yaw carries a sub-newton-metre *torque*
    limit, and mixing them would hand yaw an effectively unbounded drive. The
    XY group's pattern is correspondingly exact (``stage_[xy]_joint``).

    Like the fingers it runs with zero implicit PD stiffness and damping: the
    environment applies explicit, effort-limited PD torques, and
    ``effort_limit_sim`` enforces the same bound inside PhysX.

    Args:
        base_cfg: Hand configuration without the yaw joint.
        joint_limit: Revolute hard limit (radians) used for validation only.
        effort_limit: Maximum yaw actuator torque, N*m.
        velocity_limit: Maximum yaw joint velocity, rad/s.
        armature: Extra reflected inertia on the yaw joint, kg*m^2.
        friction: Yaw joint friction torque, N*m.

    Returns:
        A new ``ArticulationCfg`` with the yaw actuator and reset pose.
    """
    if joint_limit <= 0.0:
        raise ValueError(f"Yaw stage joint_limit must be positive, got {joint_limit}")
    if effort_limit <= 0.0:
        raise ValueError(f"Yaw stage effort_limit must be positive, got {effort_limit}")
    if velocity_limit <= 0.0:
        raise ValueError(f"Yaw stage velocity_limit must be positive, got {velocity_limit}")

    joint_pos = dict(base_cfg.init_state.joint_pos)
    joint_pos[YAW_STAGE_JOINT_NAME] = 0.0
    actuators = dict(base_cfg.actuators)
    if YAW_STAGE_ACTUATOR_GROUP in actuators:
        raise ValueError(
            f"Actuator group {YAW_STAGE_ACTUATOR_GROUP!r} already exists on the base hand"
        )
    actuators[YAW_STAGE_ACTUATOR_GROUP] = ImplicitActuatorCfg(
        joint_names_expr=[YAW_STAGE_ACTUATOR_EXPR],
        effort_limit_sim=float(effort_limit),
        velocity_limit_sim=float(velocity_limit),
        stiffness=0.0,
        damping=0.0,
        friction=float(friction),
        armature=float(armature),
    )
    return base_cfg.replace(
        init_state=base_cfg.init_state.replace(joint_pos=joint_pos),
        actuators=actuators,
    )


def _make_screw_cfg(urdf_path: str, init_pos: tuple[float, float, float],
                    nut_joint_name: str) -> ArticulationCfg:
    """Fixed-base articulated screw object with one passive revolute joint."""
    return ArticulationCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=urdf_path,
            fix_base=True,
            merge_fixed_joints=False,
            make_instanceable=False,
            # Passive joint: no drive. Joint friction comes from the URDF/actuator cfg.
            joint_drive=UrdfConverterCfg.JointDriveCfg(
                target_type="none",
                gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
            ),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                max_depenetration_velocity=1000.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=1,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                fix_root_link=True,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=init_pos,
            rot=OBJECT_INIT_ROT,
            joint_pos={nut_joint_name: 0.0},
        ),
        actuators={
            "screw": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                effort_limit_sim=2.0,
                stiffness=0.0,
                damping=0.0,
                friction=0.2,
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )


SCREW_TRINUT_CFG = _make_screw_cfg(_TRINUT_URDF, TRINUT_INIT_POS, "nut_joint")
SCREW_DRIVER_CFG = _make_screw_cfg(_DRIVER_URDF, DRIVER_INIT_POS, "handle_to_shaft")
SCREW_VAVLE_DRIVER_CFG = _make_screw_cfg(
    _VAVLE_DRIVER_URDF, VAVLE_DRIVER_INIT_POS, "valve_to_shaft"
)
SCREW_VALVE_DRIVER_25_CFG = _make_screw_cfg(
    _VALVE_DRIVER_25_URDF, VAVLE_DRIVER_INIT_POS, "valve_to_shaft"
)
SCREW_VALVE_DRIVER_40_CFG = _make_screw_cfg(
    _VALVE_DRIVER_40_URDF, VAVLE_DRIVER_INIT_POS, "valve_to_shaft"
)
