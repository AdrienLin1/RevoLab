# Copyright (c) 2026, BrainCo.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tactile valve task with a physical XY translation + world-Z yaw stage.

The environment extends :class:`Revo3HandScrewTactileXYEnv` with a real
revolute joint between the Y carriage and the hand mount::

    world (fixed by the asset's global root joint)
      -> stage_x_joint   (prismatic, world X, limits +/- xy_joint_limit)
      -> stage_x_carriage
      -> stage_y_joint   (prismatic, world Y, limits +/- xy_joint_limit)
      -> stage_y_carriage
      -> stage_yaw_joint (revolute,  world Z, limits +/- yaw_joint_limit)
      -> right_hand_base_link  (hand mount)
      -> Revo3 palm and its 21 finger joints

There is no root teleport anywhere in the control loop: the hand root pose is
only written at reset (unchanged base behaviour). Horizontal motion comes
exclusively from effort-limited PD forces on the two prismatic joints and yaw
from an effort-limited PD torque on the revolute joint, whose axis passes
through the hand base mount rather than the valve centre.

Action layout (fixed, resolved by joint NAME, never by articulation ordering)::

    action[:, 0:21]  -> finger joints
    action[:, 21:23] -> XY stage (world X / Y position-target increments)
    action[:, 23:24] -> yaw stage (world Z position-target increment)

Everything upstream of the yaw channel takes the untouched ``valvedriver_
tactile_xy`` code path, so the two-DOF baseline is bit-for-bit preserved.
"""

from __future__ import annotations

from .revo3_hand_screw_tactile_xy_env import Revo3HandScrewTactileXYEnv
from .revo3_hand_screw_tactile_xyyaw_env_cfg import Revo3HandScrewTactileXYYawMixinCfg
from .revo3_hand_screw_tactile_yaw_env import Revo3HandYawStageMixin

# Observation channels of the two translation DOFs, in generic block order.
_XY_OBS_BLOCKS: tuple[str, ...] = (
    "xy_position",
    "xy_velocity",
    "xy_target",
    "previous_xy_action",
    "xy_workspace_margin",
)


class Revo3HandScrewTactileXYYawEnv(
    Revo3HandYawStageMixin,
    Revo3HandScrewTactileXYEnv,
):
    """Tactile valve environment on two prismatic joints plus a yaw joint."""

    cfg: Revo3HandScrewTactileXYYawMixinCfg

    def _preceding_stage_obs_blocks(self) -> tuple[str, ...]:
        """The XY channels precede yaw in every generic stage block."""
        return _XY_OBS_BLOCKS

    def set_xy_curriculum_progress(self, progress: float) -> tuple[float, float]:
        """Drive the whole stage from the legacy XY entry point.

        Any caller that still speaks the two-DOF API (for example an older
        playback script) must not be able to ramp translation while leaving yaw
        behind, so this override pushes the shared progress to both DOFs.

        Args:
            progress: Ramp progress in ``[0, 1]``.

        Returns:
            The resolved XY ``(workspace, action_scale)`` in metres.
        """
        values = self.set_stage_curriculum_progress(progress)
        return values["xy_workspace"], values["xy_action_scale"]
