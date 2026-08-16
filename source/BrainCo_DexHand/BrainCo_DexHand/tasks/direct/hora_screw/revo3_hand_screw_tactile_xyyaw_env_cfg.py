# Copyright (c) 2026, BrainCo.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Environment config for the tactile valve task with an XY + yaw stage.

This variant keeps ``valvedriver_tactile`` and ``valvedriver_tactile_xy``
byte-for-byte intact and extends the two-axis stage with a real world-Z
revolute joint at the hand mount::

    world (fixed)
      -> stage_x_joint   (prismatic, world X)
      -> stage_x_carriage
      -> stage_y_joint   (prismatic, world Y)
      -> stage_y_carriage
      -> stage_yaw_joint (revolute,  world Z)
      -> right_hand_base_link (hand mount)
      -> Revo3 palm + 21 finger joints

Only the action space widens (23 -> 24). The 141-dim teacher observation, the
privileged info layout, the 1170-dim teacher tactile frame and the 42-dim
student proprio frame are all unchanged, so an existing 21-D
``valvedriver_tactile_frame813`` master checkpoint still loads strictly.

Action layout::

    action[:, 0:21]  -> 21 finger joints
    action[:, 21:23] -> arm end-effector world X / Y translation
    action[:, 23:24] -> arm end-effector yaw about world Z
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .revo3_hand_screw_tactile_env_cfg import Revo3HandVavleDriverTactileEnvCfg
from .revo3_hand_screw_tactile_xy_env_cfg import Revo3HandScrewTactileXYMixinCfg
from .revo3_hand_screw_tactile_yaw_env_cfg import Revo3HandScrewTactileYawMixinCfg
from .xy_stage import NUM_XY_DOFS
from .yaw_stage import (
    NUM_XYYAW_DOFS,
    XYYAW_STAGE_JOINT_NAMES,
    XYYAW_STAGE_WORLD_AXES,
)


@configclass
class Revo3HandScrewTactileXYYawMixinCfg(
    Revo3HandScrewTactileYawMixinCfg,
    Revo3HandScrewTactileXYMixinCfg,
):
    """Combine the two-axis translation stage with the world-Z yaw joint.

    The MRO runs ``__post_init__`` as base -> XY -> yaw, so the XY actuator
    group is attached first and the yaw group second; the two groups are
    matched by disjoint, exact expressions (``stage_[xy]_joint`` and
    ``stage_yaw_joint``) and therefore never share effort, velocity, armature
    or friction settings.
    """

    # ---- action layout -------------------------------------------------
    # action[:, :21] fingers, [21:23] XY translation, [23:24] yaw.
    action_space = 21 + NUM_XYYAW_DOFS
    finger_action_space = 21

    # ---- stage identity (also read by HierarchicalPPO) ------------------
    # Fixed action order; every DOF lookup resolves these names, never the
    # articulation's internal joint ordering.
    stage_joint_names = XYYAW_STAGE_JOINT_NAMES
    stage_world_axes = XYYAW_STAGE_WORLD_AXES

    def _stage_action_space(self) -> int:
        """Return the number of trailing stage action channels of this task."""
        return NUM_XYYAW_DOFS

    def _num_preceding_stage_dofs(self) -> int:
        """Return how many stage DOFs precede yaw in the action vector."""
        return NUM_XY_DOFS


@configclass
class Revo3HandVavleDriverTactileXYYawEnvCfg(
    Revo3HandScrewTactileXYYawMixinCfg,
    Revo3HandVavleDriverTactileEnvCfg,
):
    """Five-finger tactile valve task on a physical XY + yaw stage."""
