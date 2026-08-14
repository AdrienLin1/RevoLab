# Copyright (c) 2026, BrainCo.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rollout storage for the hierarchical master/follower PPO trainer.

One rollout step stores both policies' data for the *same* environment state
``s_t``: the master's 21-D hand decision and the follower's 2-D horizontal
translation decision, plus the single 23-D action actually sent to the
environment.

Master and follower share the team reward but keep completely separate value
functions, advantages and PPO ratios. This class therefore runs two GAE
streams over one reward/done stream.
"""

from __future__ import annotations

import torch


def transform_op(arr: torch.Tensor) -> torch.Tensor:
    """Swap the time/env axes and flatten them into the batch dimension."""
    size = arr.size()
    return arr.transpose(0, 1).reshape(size[0] * size[1], *size[2:])


class HierarchicalExperienceBuffer:
    """Two-policy rollout buffer with independent master/follower advantages."""

    def __init__(
        self,
        num_envs: int,
        horizon_length: int,
        device,
        *,
        master_obs_dim: int,
        master_action_dim: int,
        priv_info_dim: int,
        tactile_hist_shape: tuple[int, int],
        follower_obs_dim: int,
        follower_action_dim: int,
        follower_critic_priv_dim: int,
        master_minibatch_size: int,
        follower_minibatch_size: int,
    ):
        self.device = device
        self.num_envs = int(num_envs)
        self.transitions_per_env = int(horizon_length)
        self.batch_size = self.num_envs * self.transitions_per_env
        self.master_obs_dim = int(master_obs_dim)
        self.master_action_dim = int(master_action_dim)
        self.priv_info_dim = int(priv_info_dim)
        self.follower_obs_dim = int(follower_obs_dim)
        self.follower_action_dim = int(follower_action_dim)
        self.follower_critic_priv_dim = int(follower_critic_priv_dim)
        self.env_action_dim = self.master_action_dim + self.follower_action_dim

        if tactile_hist_shape is None:
            raise ValueError(
                "HierarchicalExperienceBuffer requires the teacher tactile history "
                "shape; the hierarchical trainer only supports the structured "
                "finger_attention_gru master."
            )
        self.tactile_hist_shape = tuple(int(value) for value in tactile_hist_shape)

        for name, value in (
            ("master_minibatch_size", master_minibatch_size),
            ("follower_minibatch_size", follower_minibatch_size),
        ):
            if int(value) <= 0 or self.batch_size % int(value) != 0:
                raise ValueError(
                    f"{name} ({value}) must be positive and divide the rollout batch "
                    f"size ({self.batch_size})"
                )
        self.master_minibatch_size = int(master_minibatch_size)
        self.follower_minibatch_size = int(follower_minibatch_size)
        self.num_master_minibatches = self.batch_size // self.master_minibatch_size
        self.num_follower_minibatches = self.batch_size // self.follower_minibatch_size

        def _zeros(*trailing, dtype=torch.float32):
            return torch.zeros(
                (self.transitions_per_env, self.num_envs, *trailing),
                dtype=dtype,
                device=self.device,
            )

        self.storage_dict = {
            # ---- shared transition data ----
            # ``rewards`` is the raw scaled team reward. The two ``*_rewards``
            # streams are that same reward plus each policy's OWN truncation
            # bootstrap, so one policy's value estimate never leaks into the
            # other's advantage.
            "rewards": _zeros(1),
            "master_rewards": _zeros(1),
            "follower_rewards": _zeros(1),
            "dones": _zeros(dtype=torch.uint8),
            "timeouts": _zeros(dtype=torch.float32),
            "env_actions": _zeros(self.env_action_dim),
            # ---- master ----
            "obses": _zeros(self.master_obs_dim),
            "priv_info": _zeros(self.priv_info_dim),
            "tactile_hist": _zeros(*self.tactile_hist_shape),
            "master_actions": _zeros(self.master_action_dim),
            "master_executed_actions": _zeros(self.master_action_dim),
            "master_neglogpacs": _zeros(),
            "master_values": _zeros(1),
            "master_mus": _zeros(self.master_action_dim),
            "master_sigmas": _zeros(self.master_action_dim),
            "master_returns": _zeros(1),
            # ---- follower ----
            "follower_obses": _zeros(self.follower_obs_dim),
            "follower_critic_priv": _zeros(max(self.follower_critic_priv_dim, 1)),
            "follower_actions": _zeros(self.follower_action_dim),
            "follower_executed_actions": _zeros(self.follower_action_dim),
            "follower_neglogpacs": _zeros(),
            "follower_values": _zeros(1),
            "follower_mus": _zeros(self.follower_action_dim),
            "follower_sigmas": _zeros(self.follower_action_dim),
            "follower_returns": _zeros(1),
        }
        self.data_dict: dict[str, torch.Tensor] | None = None
        self.rollout_stats: dict[str, float] = {}

    # ------------------------------------------------------------------
    # collection
    # ------------------------------------------------------------------

    def update_data(self, name: str, index: int, value: torch.Tensor) -> None:
        """Write one rollout step of a stored field."""
        if name not in self.storage_dict:
            raise KeyError(f"Unknown hierarchical rollout field: {name}")
        self.storage_dict[name][index, :] = value

    # ------------------------------------------------------------------
    # returns
    # ------------------------------------------------------------------

    def _compute_gae(self, reward_key, value_key, return_key, last_values, gamma, tau) -> None:
        last_gae_lam = 0.0
        advantages = torch.zeros_like(self.storage_dict[reward_key])
        for t in reversed(range(self.transitions_per_env)):
            if t == self.transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.storage_dict[value_key][t + 1]
            next_nonterminal = (1.0 - self.storage_dict["dones"].float()[t]).unsqueeze(1)
            delta = (
                self.storage_dict[reward_key][t]
                + gamma * next_values * next_nonterminal
                - self.storage_dict[value_key][t]
            )
            advantages[t] = last_gae_lam = (
                delta + gamma * tau * next_nonterminal * last_gae_lam
            )
            self.storage_dict[return_key][t, :] = (
                advantages[t] + self.storage_dict[value_key][t]
            )

    def compute_returns(self, master_last_values, follower_last_values, gamma, tau) -> None:
        """Run two independent GAE passes over the shared team reward."""
        self._compute_gae(
            "master_rewards",
            "master_values",
            "master_returns",
            master_last_values,
            gamma,
            tau,
        )
        self._compute_gae(
            "follower_rewards",
            "follower_values",
            "follower_returns",
            follower_last_values,
            gamma,
            tau,
        )

    @staticmethod
    def _normalize_advantage(advantage: torch.Tensor) -> torch.Tensor:
        std = advantage.std(unbiased=False)
        if not torch.isfinite(std) or std <= 1.0e-8:
            return torch.zeros_like(advantage)
        return (advantage - advantage.mean()) / std

    def prepare_training(self, normalize_advantage: bool = True) -> dict[str, torch.Tensor]:
        """Flatten the rollout and build both advantage streams."""
        self.data_dict = {key: transform_op(value) for key, value in self.storage_dict.items()}
        for prefix in ("master", "follower"):
            raw = self.data_dict[f"{prefix}_returns"] - self.data_dict[f"{prefix}_values"]
            advantages = (
                self._normalize_advantage(raw) if normalize_advantage else raw
            ).squeeze(1)
            self.data_dict[f"{prefix}_advantages"] = advantages
            self.rollout_stats[f"{prefix}_adv_raw_std"] = raw.std(unbiased=False).item()
        return self.data_dict

    # ------------------------------------------------------------------
    # minibatches
    # ------------------------------------------------------------------

    def _slice(self, keys, start: int, end: int) -> dict[str, torch.Tensor]:
        if self.data_dict is None:
            raise RuntimeError("prepare_training() must run before slicing minibatches")
        return {key: self.data_dict[key][start:end] for key in keys}

    _MASTER_KEYS = (
        "obses",
        "priv_info",
        "tactile_hist",
        "master_actions",
        "master_neglogpacs",
        "master_values",
        "master_mus",
        "master_sigmas",
        "master_returns",
        "master_advantages",
    )
    _FOLLOWER_KEYS = (
        "follower_obses",
        "follower_critic_priv",
        "follower_actions",
        "follower_neglogpacs",
        "follower_values",
        "follower_mus",
        "follower_sigmas",
        "follower_returns",
        "follower_advantages",
    )

    def master_minibatch(self, index: int) -> dict[str, torch.Tensor]:
        start = index * self.master_minibatch_size
        end = start + self.master_minibatch_size
        self._last_master_range = (start, end)
        return self._slice(self._MASTER_KEYS, start, end)

    def follower_minibatch(self, index: int) -> dict[str, torch.Tensor]:
        start = index * self.follower_minibatch_size
        end = start + self.follower_minibatch_size
        self._last_follower_range = (start, end)
        return self._slice(self._FOLLOWER_KEYS, start, end)

    def update_master_mu_sigma(self, mu: torch.Tensor, sigma: torch.Tensor) -> None:
        start, end = self._last_master_range
        self.data_dict["master_mus"][start:end] = mu
        self.data_dict["master_sigmas"][start:end] = sigma

    def update_follower_mu_sigma(self, mu: torch.Tensor, sigma: torch.Tensor) -> None:
        start, end = self._last_follower_range
        self.data_dict["follower_mus"][start:end] = mu
        self.data_dict["follower_sigmas"][start:end] = sigma
