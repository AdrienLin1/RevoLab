# Copyright (c) 2026, BrainCo.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Follower observation contract for the hierarchical valve policy.

The follower is a small, deployable 2-D horizontal-translation policy.  It sees
only what a real arm controller could see at the same instant ``t``:

===========================  ====  ==========================================
channel block                dims  meaning
===========================  ====  ==========================================
``executed_hand_action``       21  clipped 21-D hand action of THIS cycle
``tactile_latent``            128  master's GRU hidden state of THIS forward
``xy_position``                 2  stage joint positions (normalized)
``xy_velocity``                 2  stage joint velocities (normalized)
``xy_target``                   2  stage position targets (normalized)
``previous_xy_action``          2  executed XY action of the previous cycle
``xy_workspace_margin``         2  normalized distance to the soft boundary
===========================  ====  ==========================================

Total: **159**.

Deliberately excluded (they would either leak privileged state or make the
follower depend on the master's private representation):
the master's 141-D observation, the 21 finger joint positions, the 21 finger
joint targets, the master's actor features, ``priv_info``, the raw teacher
tactile frame, and the 160-D attended per-finger tokens.
"""

from __future__ import annotations

import torch

TACTILE_LATENT_DIM: int = 128
EXECUTED_HAND_ACTION_DIM: int = 21
XY_ACTION_DIM: int = 2

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
