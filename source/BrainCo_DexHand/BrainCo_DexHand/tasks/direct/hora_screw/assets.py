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

_REPO_ROOT = Path(__file__).resolve().parents[6]
_REVO3_USD = str(_REPO_ROOT / "assets" / "usd" / "revo3_right.usd")
_TRINUT_URDF = str(_REPO_ROOT / "assets" / "urdf" / "screw" / "trinut" / "trinut.urdf")
_DRIVER_URDF = str(_REPO_ROOT / "assets" / "urdf" / "screw" / "driver" / "driver_8.urdf")
_VAVLE_DRIVER_URDF = str(
    _REPO_ROOT / "assets" / "urdf" / "screw" / "vavledriver" / "vavledriver_hex.urdf"
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
SCREW_VALVE_DRIVER_40_CFG = _make_screw_cfg(
    _VALVE_DRIVER_40_URDF, VAVLE_DRIVER_INIT_POS, "valve_to_shaft"
)
