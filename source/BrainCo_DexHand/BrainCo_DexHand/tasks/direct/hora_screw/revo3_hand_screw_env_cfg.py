"""Environment config for Revo3 right hand HORA screw/valve tasks.

Ported from dexscrew XHandHoraNutBolt / XHandHoraScrewDriver (Isaac Gym) into the
Isaac Lab DirectRLEnv workflow, following the structure of hora_rotation.
"""
from __future__ import annotations

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

from .assets import (
    REVO3_HAND_DRIVER_CFG,
    REVO3_HAND_NUTBOLT_CFG,
    REVO3_HAND_VAVLE_DRIVER_CFG,
    SCREW_DRIVER_CFG,
    SCREW_VALVE_DRIVER_40_CFG,
    SCREW_TRINUT_CFG,
    SCREW_VAVLE_DRIVER_CFG,
)


@configclass
class Revo3HandScrewEnvCfg(DirectRLEnvCfg):
    # 800 control steps @ 20 Hz, matching dexscrew episodeLength=800
    episode_length_s = 40.0
    action_space = 21
    observation_space = 141  # 3 frames x 47 dims (21 joint_pos + 21 targets + 5 contacts)
    prop_hist_len = 30
    priv_info_dim = 11
    state_space = 0
    asymmetric_obs = False
    decimation = 12
    clip_obs = 5.0
    clip_actions = 1.0
    action_scale = 1 / 24
    torque_control = True
    # dexscrew controller gains (pgain 3, dgain 0.01). NOTE: deliberately NOT the
    # hora_rotation values (2.0/0.2) — that D is ~2x critical damping and kills the
    # fast finger-gaiting this task needs; dexscrew runs heavily underdamped.
    pgain: float = 3.0
    dgain: float = 0.01

    # Full gravity (dexscrew uses -9.81; the screw object is anchored, and the
    # hand has disable_gravity=True). No gravity curriculum for screw tasks.
    gravity_curriculum = False
    sim: SimulationCfg = SimulationCfg(
        # Render once per control step.  A smaller interval causes several renders
        # during one DirectRLEnv step and needlessly slows viewer training.
        dt=1 / 240, render_interval=decimation,
        gravity=(0.0, 0.0, -9.81),
        physx=PhysxCfg(
            solver_type=1, max_position_iteration_count=8, max_velocity_iteration_count=1,
            # The task applies a persistent, randomly changing wrench to the nut.
            # Applying it in every TGS iteration gives more accurate velocities.
            enable_external_forces_every_iteration=True,
            bounce_threshold_velocity=0.2,
            gpu_max_rigid_contact_count=8388608, gpu_max_rigid_patch_count=5 * 2**18,
        ),
    )

    # Filled by task variants (__post_init__)
    robot_cfg: ArticulationCfg = REVO3_HAND_NUTBOLT_CFG
    object_cfg: ArticulationCfg = SCREW_TRINUT_CFG

    actuated_joint_names = [
        "right_thumb_CMP_joint", "right_thumb_CMR_joint",
        "right_thumb_MCP_joint", "right_thumb_PIP_joint", "right_thumb_DIP_joint",
        "right_index_MPR_joint", "right_index_MCP_joint",
        "right_index_PIP_joint", "right_index_DIP_joint",
        "right_middle_MPR_joint", "right_middle_MCP_joint",
        "right_middle_PIP_joint", "right_middle_DIP_joint",
        "right_ring_MPR_joint", "right_ring_MCP_joint",
        "right_ring_PIP_joint", "right_ring_DIP_joint",
        "right_little_MPR_joint", "right_little_MCP_joint",
        "right_little_PIP_joint", "right_little_DIP_joint",
    ]
    fingertip_body_names = [
        "right_thumb_DIP_Link", "right_index_DIP_Link",
        "right_middle_DIP_Link", "right_ring_DIP_Link", "right_little_DIP_Link",
    ]
    elastomer_body_names = [
        "right_thumb_DIP_Link", "right_index_DIP_Link",
        "right_middle_DIP_Link", "right_ring_DIP_Link", "right_little_DIP_Link",
    ]
    contact_sensor = []
    nut_contact_sensor: ContactSensorCfg = None

    # Screw object description
    nut_body_name = "nut"          # rotating link of the screw articulation
    nut_joint_name = "nut_joint"   # passive revolute joint
    # Offset (nut link frame) from the nut link origin to the sleeve grasp center;
    # used for finger-distance termination and proximity reward.
    nut_ref_offset = (0.0, 0.0, -0.03125)

    # Actions of these joints are masked to 0 (dexscrew masks unused fingers;
    # masked fingers still hold their init pose via PD).
    masked_action_joint_names: list = []
    # Joints containing this substring are excluded from the pose-diff penalty
    # (dexscrew masks the thumb DOFs).
    pose_diff_exclude_substring = "thumb"

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        # Physics replication stays disabled because mass/COM/material properties
        # are randomized independently per environment.  Collision filtering is
        # therefore performed explicitly after cloning in _setup_scene().
        num_envs=16384, env_spacing=0.75, replicate_physics=False,
        filter_collisions=False)

    # ---- rewards (dexscrew XHandHoraNutBolt defaults; ScrewDriver overrides) ----
    angvel_clip_min = -4.0
    angvel_clip_max = 4.0
    # float for fixed threshold, or (init, final, start_step, end_step) curriculum
    angvel_penalty_threshold = 10.0
    rotate_reward_scale = 6.0
    pose_diff_penalty_scale = -0.5
    torque_penalty_scale = -0.1
    work_penalty_scale = -0.01
    rotate_penalty_scale = -0.3
    proximity_reward_scale = 2.0
    # Fingertips used by the proximity reward. Existing tasks preserve the
    # dexscrew thumb/index behavior; the valve variant uses all five fingers.
    proximity_fingertip_indices = (0, 1)

    # ---- terminations ----
    # thumb/index fingertip distance to the nut grasp center. dexscrew uses 0.05 with
    # true tip frames; Revo3 only exposes DIP link origins (~2-3 cm behind the pad,
    # measured rest distance ~0.04-0.05), hence the larger threshold.
    reset_dist_threshold = 0.08
    # When False, the thumb/index distance reset is skipped entirely; the
    # threshold above still normalizes the proximity reward.
    enable_finger_dist_reset = True
    nut_termination_history_len = 70
    nut_stagnation_eps = 0.003
    screw_upper_limit = 628.3185
    screw_limit_margin = 5.0

    # ---- reset ----
    # uniform joint noise at reset, as a fraction of half joint range (dexscrew: 10%)
    reset_joint_noise_frac = 0.1

    # ---- observation noise / tactile ----
    joint_noise_scale = 0.02
    enable_tactile = True
    enable_contact_in_obs = True   # Stage2 sets False: actor sees zero contact, adapt_tconv retains contact history
    binary_contact = False
    disable_tactile_ids = []
    contact_smooth = 0.5
    contact_threshold = 0.05
    contact_latency = 0.005
    contact_sensor_noise = 0.01
    dof_limits_scale = 0.9

    # ---- domain randomization ----
    # dexscrew-style absolute per-DOF PD randomization (+-10% around base gains)
    randomize_pd_gains = True
    randomize_p_gain_lower = 2.7
    randomize_p_gain_upper = 3.3
    randomize_d_gain_lower = 0.009
    randomize_d_gain_upper = 0.011
    # dexscrew-style friction: hand and object get the SAME per-env value drawn
    # from U(lower, upper). High friction is load-bearing here: turning the nut
    # against its 0.2 Nm joint friction needs ~10 N tangential force at r~0.02 m,
    # which is only transmissible with rubber-grade friction coefficients.
    randomize_friction = True
    randomize_friction_lower = 0.5
    randomize_friction_upper = 8.0
    randomize_com = True
    randomize_com_lower = -0.002
    randomize_com_upper = 0.002
    randomize_mass = True
    randomize_mass_lower = 0.04
    randomize_mass_upper = 0.06

    # random forces on the nut link (dexscrew: forceScale=2.0)
    force_scale = 2.0
    random_force_prob_scalar = 0.25
    force_decay = 0.9
    force_decay_interval = 0.08

    debug_show_axes = False
    grasp_cache_path = ""  # unused; kept for scripts/hora/train.py metadata compat

    def __post_init__(self):
        super().__post_init__()
        self.contact_sensor = []
        for name in self.elastomer_body_names:
            self.contact_sensor.append(ContactSensorCfg(
                prim_path=f"/World/envs/env_.*/hand/{name}",
                history_length=3,
                track_contact_points=False,
                filter_prim_paths_expr=[f"/World/envs/env_.*/object/{self.nut_body_name}"],
            ))
        self.nut_contact_sensor = ContactSensorCfg(
            prim_path=f"/World/envs/env_.*/object/{self.nut_body_name}",
            history_length=3,
            track_contact_points=False,
        )


@configclass
class Revo3HandScrewNutBoltEnvCfg(Revo3HandScrewEnvCfg):
    """Rotate a triangular-prism nut around a fixed bolt (dexscrew XHandHoraNutBolt)."""

    def __post_init__(self):
        self.robot_cfg = REVO3_HAND_NUTBOLT_CFG
        self.object_cfg = SCREW_TRINUT_CFG
        self.nut_body_name = "nut"
        self.nut_joint_name = "nut_joint"
        # trinut sleeve center in nut link frame (sleeve spans z in [-0.08, 0.0175])
        self.nut_ref_offset = (0.0, 0.0, -0.03125)
        # dexscrew NutBolt uses thumb+index+middle only (ring & little masked)
        self.masked_action_joint_names = [
            "right_ring_MPR_joint", "right_ring_MCP_joint",
            "right_ring_PIP_joint", "right_ring_DIP_joint",
            "right_little_MPR_joint", "right_little_MCP_joint",
            "right_little_PIP_joint", "right_little_DIP_joint",
        ]
        # rewards: dexscrew XHandHoraNutBolt.yaml
        self.angvel_penalty_threshold = 10.0
        self.rotate_reward_scale = 6.0
        self.pose_diff_penalty_scale = -0.5
        self.torque_penalty_scale = -0.1
        self.work_penalty_scale = -0.01
        self.rotate_penalty_scale = -0.3
        self.proximity_reward_scale = 2.0
        self.nut_termination_history_len = 70
        super().__post_init__()


@configclass
class Revo3HandScrewDriverEnvCfg(Revo3HandScrewEnvCfg):
    """Spin a screwdriver handle around its fixed shaft (dexscrew XHandHoraScrewDriver)."""

    def __post_init__(self):
        self.robot_cfg = REVO3_HAND_DRIVER_CFG
        self.object_cfg = SCREW_DRIVER_CFG
        self.nut_body_name = "handle"
        self.nut_joint_name = "handle_to_shaft"
        # driver handle center in handle link frame (handle spans z in [0.01, 0.11])
        self.nut_ref_offset = (0.0, 0.0, 0.06)
        # dexscrew ScrewDriver masks the pinky only -> mask the little finger
        self.masked_action_joint_names = [
            "right_little_MPR_joint", "right_little_MCP_joint",
            "right_little_PIP_joint", "right_little_DIP_joint",
        ]
        # rewards: dexscrew XHandHoraScrewDriver.yaml
        self.angvel_penalty_threshold = (7.5, 15.0, 30_000_000, 60_000_000)
        self.rotate_reward_scale = 2.5
        self.pose_diff_penalty_scale = -0.1
        self.torque_penalty_scale = -3.0
        self.work_penalty_scale = -0.01
        self.rotate_penalty_scale = -0.3
        self.proximity_reward_scale = 2.0
        self.nut_termination_history_len = 60
        super().__post_init__()


@configclass
class Revo3HandVavleDriverEnvCfg(Revo3HandScrewEnvCfg):
    """Turn a hand-sized hexagonal valve handle with all 21 hand joints."""

    def __post_init__(self):
        self.robot_cfg = REVO3_HAND_VAVLE_DRIVER_CFG
        self.object_cfg = SCREW_VAVLE_DRIVER_CFG
        self.nut_body_name = "valve"
        self.nut_joint_name = "valve_to_shaft"
        # The valve mesh spans z=[0.0, 0.08] in the rotating link frame.
        self.nut_ref_offset = (0.0, 0.0, 0.04)

        # Unlike screwdriver_tactile, every joint receives its policy action.
        # Keep the common screw-task 0.9 joint-range margin.
        self.masked_action_joint_names = []
        self.dof_limits_scale = 0.9

        # Keep screwdriver reward scales/curriculum. Only the proximity term is
        # evaluated over all fingers to encourage a whole-hand valve wrap.
        self.angvel_penalty_threshold = (7.5, 15.0, 30_000_000, 60_000_000)
        self.rotate_reward_scale = 2.5
        # No init-pose anchoring: all five fingers drive the valve, so they are
        # free to leave the reset grasp. extras["pose_diff_penalty"] still logs
        # the raw (unscaled) deviation.
        self.pose_diff_penalty_scale = 0.0
        self.torque_penalty_scale = -3.0
        self.work_penalty_scale = -0.01
        self.rotate_penalty_scale = -0.3
        self.proximity_reward_scale = 2.0
        self.proximity_fingertip_indices = (0, 1, 2, 3, 4)
        # No thumb/index distance reset: the whole-hand wrap is not required to
        # keep those two tips near the grasp center. Stagnation, no-contact and
        # joint-limit resets still apply.
        self.enable_finger_dist_reset = False
        self.nut_termination_history_len = 60
        super().__post_init__()


@configclass
class Revo3HandValveDriver40EnvCfg(Revo3HandVavleDriverEnvCfg):
    """The valve task with only the hex handle circumradius changed to 40 mm."""

    def __post_init__(self):
        super().__post_init__()
        # Everything else is inherited unchanged from the 30 mm valve task.
        self.object_cfg = SCREW_VALVE_DRIVER_40_CFG
