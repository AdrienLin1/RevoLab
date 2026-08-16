# Copyright (c) 2026, BrainCo.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Follower observation contract for the hierarchical valve policy.

The follower is a small, deployable end-effector policy.  It sees only what a
real arm controller could see at the same instant ``t``.  Its width is
parameterized by the number of stage DOFs it commands, so the 2-D translation
task and the 3-D translation+yaw task share one contract without either
changing the other's layout:

============================  =============================================
channel block                 dims
============================  =============================================
``executed_hand_action``      21
``tactile_latent``            128 (master's GRU hidden state of THIS forward)
stage position                ``D``  (joint positions, normalized)
stage velocity                ``D``  (joint velocities, normalized)
stage target                  ``D``  (position targets, normalized)
previous stage action         ``D``  (executed action of the previous cycle)
stage workspace margin        ``D``  (normalized distance to the boundary)
============================  =============================================

Total: ``149 + 5 * D`` -> **154** for 1 DOF (yaw only), **159** for 2 DOF (XY)
and **164** for 3 DOF (XY+yaw).

The 2-DOF layout keeps its original ``xy_*`` block names and its original
``FOLLOWER_OBS_SPEC`` / ``FOLLOWER_OBS_DIM`` / ``FOLLOWER_OBS_SLICES``
constants so existing XY checkpoints, tests and call sites are untouched.  Any
other width uses the generic ``stage_*`` block names.

Deliberately excluded (they would either leak privileged state or make the
follower depend on the master's private representation):
the master's 141-D observation, the 21 finger joint positions, the 21 finger
joint targets, the master's actor features, ``priv_info``, the raw teacher
tactile frame, and the attended per-finger tokens.
"""

from __future__ import annotations

import torch

TACTILE_LATENT_DIM: int = 128
EXECUTED_HAND_ACTION_DIM: int = 21
XY_ACTION_DIM: int = 2

# Generic per-stage-DOF blocks, in observation order. Used verbatim as the
# environment observation keys of every non-2-DOF stage task.
STAGE_OBS_BLOCKS: tuple[str, ...] = (
    "stage_position",
    "stage_velocity",
    "stage_target",
    "previous_stage_action",
    "stage_workspace_margin",
)
# Legacy 2-DOF block names. Kept exactly as shipped so the XY task's
# observation keys, checkpoints and slices never move.
XY_OBS_BLOCKS: tuple[str, ...] = (
    "xy_position",
    "xy_velocity",
    "xy_target",
    "previous_xy_action",
    "xy_workspace_margin",
)
# Widths a hierarchical follower may command: yaw only, XY, XY+yaw.
SUPPORTED_STAGE_DOFS: tuple[int, ...] = (1, 2, 3)

FOLLOWER_OBS_SPEC: tuple[tuple[str, int], ...] = (
    ("executed_hand_action", EXECUTED_HAND_ACTION_DIM),
    ("tactile_latent", TACTILE_LATENT_DIM),
    ("xy_position", XY_ACTION_DIM),
    ("xy_velocity", XY_ACTION_DIM),
    ("xy_target", XY_ACTION_DIM),
    ("previous_xy_action", XY_ACTION_DIM),
    ("xy_workspace_margin", XY_ACTION_DIM),
)
FOLLOWER_OBS_DIM: int = sum(width for _name, width in FOLLOWER_OBS_SPEC)
assert FOLLOWER_OBS_DIM == 159, FOLLOWER_OBS_DIM

FOLLOWER_OBS_SLICES: dict[str, slice] = {}
_offset = 0
for _name, _width in FOLLOWER_OBS_SPEC:
    FOLLOWER_OBS_SLICES[_name] = slice(_offset, _offset + _width)
    _offset += _width
del _offset, _name, _width


def stage_obs_blocks(num_stage_dofs: int) -> tuple[str, ...]:
    """Return the per-DOF block names used for a given stage width.

    Args:
        num_stage_dofs: Number of stage DOFs the follower commands.

    Returns:
        The five block names, in observation order. The 2-DOF stage keeps its
        historical ``xy_*`` names; every other width uses ``stage_*``.

    Raises:
        ValueError: If the width is not supported.
    """
    validate_stage_dofs(num_stage_dofs)
    return XY_OBS_BLOCKS if int(num_stage_dofs) == XY_ACTION_DIM else STAGE_OBS_BLOCKS


def validate_stage_dofs(num_stage_dofs: int) -> int:
    """Return the validated stage width.

    Raises:
        ValueError: If the width is not one of :data:`SUPPORTED_STAGE_DOFS`.
    """
    width = int(num_stage_dofs)
    if width not in SUPPORTED_STAGE_DOFS:
        raise ValueError(
            f"Unsupported hierarchical stage width {width}; supported widths are "
            f"{SUPPORTED_STAGE_DOFS} (1 = yaw only, 2 = XY, 3 = XY + yaw)"
        )
    return width


def follower_obs_spec(num_stage_dofs: int) -> tuple[tuple[str, int], ...]:
    """Return the ``(block, width)`` observation spec for a stage width."""
    width = validate_stage_dofs(num_stage_dofs)
    blocks = (
        ("executed_hand_action", EXECUTED_HAND_ACTION_DIM),
        ("tactile_latent", TACTILE_LATENT_DIM),
    )
    return blocks + tuple((name, width) for name in stage_obs_blocks(width))


def follower_obs_dim(num_stage_dofs: int) -> int:
    """Return the follower observation width for a stage width."""
    return sum(width for _name, width in follower_obs_spec(num_stage_dofs))


def follower_obs_slices(num_stage_dofs: int) -> dict[str, slice]:
    """Return the block name -> slice map for a stage width."""
    slices: dict[str, slice] = {}
    offset = 0
    for name, width in follower_obs_spec(num_stage_dofs):
        slices[name] = slice(offset, offset + width)
        offset += width
    return slices


assert follower_obs_dim(2) == FOLLOWER_OBS_DIM == 159
assert follower_obs_dim(3) == 164
assert follower_obs_dim(1) == 154


def validate_tactile_latent(tactile_latent) -> torch.Tensor:
    """Fail fast unless the master produced a ``[B, 128]`` GRU hidden state.

    Args:
        tactile_latent: Candidate latent from the master forward pass.

    Returns:
        The validated tensor.

    Raises:
        RuntimeError: If the latent is missing or not exactly 128-dimensional.
    """
    if tactile_latent is None:
        raise RuntimeError(
            "HierarchicalPPO requires the master to expose a 128-dimensional tactile "
            "latent. The configured teacher produced none: set "
            "network.tactile_encoder.type='finger_attention_gru' so the master returns "
            "its final GRU hidden state. Silently substituting other features is not "
            "supported."
        )
    if not isinstance(tactile_latent, torch.Tensor):
        raise RuntimeError(
            "Master tactile latent must be a torch.Tensor, got "
            f"{type(tactile_latent).__name__}"
        )
    if tactile_latent.ndim != 2 or tactile_latent.shape[-1] != TACTILE_LATENT_DIM:
        raise RuntimeError(
            "Master tactile latent must have shape [B, "
            f"{TACTILE_LATENT_DIM}] (FingerAttentionGRUTactileEncoder GRU hidden state), "
            f"got {tuple(tactile_latent.shape)}"
        )
    return tactile_latent


def build_follower_obs(
    *,
    executed_hand_action: torch.Tensor,
    tactile_latent: torch.Tensor,
    xy_position: torch.Tensor,
    xy_velocity: torch.Tensor,
    xy_target: torch.Tensor,
    previous_xy_action: torch.Tensor,
    xy_workspace_margin: torch.Tensor,
) -> torch.Tensor:
    """Assemble the strict 159-D follower observation for time ``t``.

    Every input must describe the state **before** ``env.step`` of this control
    cycle: the executed hand action is the clipped action that is about to be
    sent, and the tactile latent comes from the same master forward pass.

    Args:
        executed_hand_action: Clipped 21-D hand action of this cycle.
        tactile_latent: Master's 128-D GRU hidden state (detached by caller).
        xy_position: Normalized stage joint positions.
        xy_velocity: Normalized stage joint velocities.
        xy_target: Normalized stage position targets.
        previous_xy_action: Executed XY action of the previous cycle.
        xy_workspace_margin: Normalized distance to the soft workspace boundary.

    Returns:
        Tensor of shape ``(B, 159)``.

    Raises:
        RuntimeError: If any block has the wrong width or the total is not 159.
    """
    validate_tactile_latent(tactile_latent)
    blocks = {
        "executed_hand_action": executed_hand_action,
        "tactile_latent": tactile_latent,
        "xy_position": xy_position,
        "xy_velocity": xy_velocity,
        "xy_target": xy_target,
        "previous_xy_action": previous_xy_action,
        "xy_workspace_margin": xy_workspace_margin,
    }
    ordered = []
    for name, width in FOLLOWER_OBS_SPEC:
        value = blocks[name]
        if value.ndim != 2 or value.shape[-1] != width:
            raise RuntimeError(
                f"Follower observation block {name!r} must have shape [B, {width}], "
                f"got {tuple(value.shape)}"
            )
        ordered.append(value)
    follower_obs = torch.cat(ordered, dim=-1)
    if follower_obs.shape[-1] != FOLLOWER_OBS_DIM:
        raise RuntimeError(
            f"Follower observation width {follower_obs.shape[-1]} != {FOLLOWER_OBS_DIM}"
        )
    return follower_obs


def follower_obs_from_env(
    obs_dict,
    *,
    executed_hand_action: torch.Tensor,
    tactile_latent: torch.Tensor,
) -> torch.Tensor:
    """Build the follower observation from an env observation dictionary.

    Args:
        obs_dict: Environment observation containing the ``xy_*`` channels.
        executed_hand_action: Clipped hand action of this control cycle.
        tactile_latent: Master's 128-D latent from the same forward pass.

    Returns:
        Tensor of shape ``(B, 159)``.

    Raises:
        KeyError: If the environment does not publish the XY state channels.
    """
    required = ("xy_position", "xy_velocity", "xy_target", "previous_xy_action", "xy_workspace_margin")
    missing = [key for key in required if key not in obs_dict]
    if missing:
        raise KeyError(
            "HierarchicalPPO requires the XY-stage observation channels "
            f"{required}; missing {missing}. Use --task valvedriver_tactile_xy."
        )
    return build_follower_obs(
        executed_hand_action=executed_hand_action,
        tactile_latent=tactile_latent,
        xy_position=obs_dict["xy_position"],
        xy_velocity=obs_dict["xy_velocity"],
        xy_target=obs_dict["xy_target"],
        previous_xy_action=obs_dict["previous_xy_action"],
        xy_workspace_margin=obs_dict["xy_workspace_margin"],
    )


def build_stage_follower_obs(
    *,
    executed_hand_action: torch.Tensor,
    tactile_latent: torch.Tensor,
    stage_blocks,
    num_stage_dofs: int,
) -> torch.Tensor:
    """Assemble the follower observation of an arbitrary supported stage width.

    Every input must describe the state **before** ``env.step`` of this control
    cycle: the executed hand action is the clipped action that is about to be
    sent, and the tactile latent comes from the same master forward pass.

    Args:
        executed_hand_action: Clipped 21-D hand action of this cycle.
        tactile_latent: Master's 128-D GRU hidden state (detached by caller).
        stage_blocks: Mapping from the width's five stage block names (see
            :func:`stage_obs_blocks`) to ``[B, D]`` tensors.
        num_stage_dofs: Number of stage DOFs the follower commands.

    Returns:
        Tensor of shape ``(B, 149 + 5 * num_stage_dofs)``.

    Raises:
        RuntimeError: If any block has the wrong width or is missing.
    """
    validate_tactile_latent(tactile_latent)
    spec = follower_obs_spec(num_stage_dofs)
    blocks = {
        "executed_hand_action": executed_hand_action,
        "tactile_latent": tactile_latent,
        **dict(stage_blocks),
    }
    ordered = []
    for name, width in spec:
        value = blocks.get(name)
        if value is None:
            raise RuntimeError(
                f"Follower observation block {name!r} is missing for a "
                f"{int(num_stage_dofs)}-DOF stage; expected blocks "
                f"{[block for block, _width in spec]}"
            )
        if value.ndim != 2 or value.shape[-1] != width:
            raise RuntimeError(
                f"Follower observation block {name!r} must have shape [B, {width}], "
                f"got {tuple(value.shape)}"
            )
        ordered.append(value)
    follower_obs = torch.cat(ordered, dim=-1)
    expected = follower_obs_dim(num_stage_dofs)
    if follower_obs.shape[-1] != expected:
        raise RuntimeError(
            f"Follower observation width {follower_obs.shape[-1]} != {expected}"
        )
    return follower_obs


def stage_follower_obs_from_env(
    obs_dict,
    *,
    executed_hand_action: torch.Tensor,
    tactile_latent: torch.Tensor,
    num_stage_dofs: int,
) -> torch.Tensor:
    """Build the follower observation of any supported stage width from an env.

    The environment channels are looked up by the width's block names only, so
    this reader can never consult the master observation, ``priv_info`` or the
    tactile history even accidentally.

    Args:
        obs_dict: Environment observation containing the stage state channels.
        executed_hand_action: Clipped hand action of this control cycle.
        tactile_latent: Master's 128-D latent from the same forward pass.
        num_stage_dofs: Number of stage DOFs the follower commands.

    Returns:
        Tensor of shape ``(B, 149 + 5 * num_stage_dofs)``.

    Raises:
        KeyError: If the environment does not publish the stage state channels.
    """
    required = stage_obs_blocks(num_stage_dofs)
    missing = [key for key in required if key not in obs_dict]
    if missing:
        raise KeyError(
            f"HierarchicalPPO requires the {int(num_stage_dofs)}-DOF stage "
            f"observation channels {required}; missing {missing}. Use a stage task "
            "(--task valvedriver_tactile_xy / valvedriver_tactile_xyyaw / "
            "valvedriver_tactile_yaw)."
        )
    return build_stage_follower_obs(
        executed_hand_action=executed_hand_action,
        tactile_latent=tactile_latent,
        stage_blocks={key: obs_dict[key] for key in required},
        num_stage_dofs=num_stage_dofs,
    )
