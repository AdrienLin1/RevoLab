# Copyright (c) 2026, BrainCo.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""TacRes actor: frozen proprio base policy + tactile-gated bounded residual correction."""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.models import MLPModel
from rsl_rl.modules import MLP, EmpiricalNormalization
from rsl_rl.modules.distribution import Distribution
from rsl_rl.utils import resolve_callable, unpad_trajectories


class TacResActor(nn.Module):
    """Actor implementing ``mu = a_base + alpha * g * tanh(MLP_res(...))`` with a frozen base policy.

    - ``base``: an :class:`MLPModel` with the exact architecture of the phase-1 actor, so its weights
      (including observation normalizer and action std) load directly from a phase-1 checkpoint.
      It is frozen at construction time and only used for deterministic forward inference.
    - ``residual``: MLP over [normalized residual observations, a_base] with tanh-bounded output.
    - ``gate``: MLP over normalized contact-event features with sigmoid output in [0, 1].
    - The exploration distribution is a Gaussian over the composed mean with a learnable (log) std.

    The constructor signature matches what ``PPO.construct_algorithm`` passes to the actor class.
    """

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        # residual network (uses the generic MLP model cfg fields)
        hidden_dims: tuple[int, ...] | list[int] = (512, 256),
        activation: str = "elu",
        obs_normalization: bool = True,
        distribution_cfg: dict | None = None,
        # TacRes-specific configuration
        base_obs_groups: list[str] | None = None,
        residual_obs_groups: list[str] | None = None,
        gate_obs_groups: list[str] | None = None,
        base_hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128),
        base_activation: str = "elu",
        base_obs_normalization: bool = True,
        base_distribution_cfg: dict | None = None,
        gate_hidden_dims: tuple[int, ...] | list[int] = (64, 32),
        alpha: float = 0.1,
        fixed_gate: float = -1.0,
    ) -> None:
        super().__init__()
        if base_obs_groups is None or residual_obs_groups is None or gate_obs_groups is None:
            raise ValueError("TacResActor requires base_obs_groups, residual_obs_groups, and gate_obs_groups.")
        for group in (*base_obs_groups, *residual_obs_groups, *gate_obs_groups):
            if group not in obs.keys():
                raise ValueError(f"TacResActor observation group '{group}' not found in env observations.")

        self.alpha = alpha
        # B2-style ablation: when >= 0, the gate is that constant (e.g. 1.0 for always-on residual)
        # and the gate network is bypassed (kept in the module for checkpoint compatibility).
        # Negative values (default) enable the learned gate.
        self.fixed_gate = float(fixed_gate) if fixed_gate is not None and fixed_gate >= 0 else None
        self.base_obs_groups = list(base_obs_groups)
        self.residual_obs_groups = list(residual_obs_groups)
        self.gate_obs_groups = list(gate_obs_groups)

        # frozen base policy (architecture identical to the phase-1 actor)
        if base_distribution_cfg is None:
            base_distribution_cfg = {"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"}
        sub_obs_groups = dict(obs_groups)
        sub_obs_groups["tacres_base"] = self.base_obs_groups
        self.base = MLPModel(
            obs,
            sub_obs_groups,
            "tacres_base",
            output_dim,
            hidden_dims=base_hidden_dims,
            activation=base_activation,
            obs_normalization=base_obs_normalization,
            distribution_cfg=dict(base_distribution_cfg),
        )
        self.base.requires_grad_(False)
        self.base.eval()

        # residual network: input = [normalized residual obs, a_base]
        self.obs_normalization = obs_normalization
        residual_obs_dim = sum(obs[g].shape[-1] for g in self.residual_obs_groups)
        gate_obs_dim = sum(obs[g].shape[-1] for g in self.gate_obs_groups)
        if obs_normalization:
            self.residual_obs_normalizer = EmpiricalNormalization(residual_obs_dim)
            self.gate_obs_normalizer = EmpiricalNormalization(gate_obs_dim)
        else:
            self.residual_obs_normalizer = nn.Identity()
            self.gate_obs_normalizer = nn.Identity()
        self.residual_mlp = MLP(residual_obs_dim + output_dim, output_dim, hidden_dims, activation)
        self.gate_mlp = MLP(gate_obs_dim, 1, gate_hidden_dims, activation)
        # zero-init the residual head: mu == a_base at initialization (functional equivalence to the
        # base policy at t=0) and the pre-tanh activations start in the linear regime, which prevents
        # early tanh saturation from noisy advantages while the fresh critic is still uninformative.
        residual_head = [m for m in self.residual_mlp if isinstance(m, nn.Linear)][-1]
        nn.init.zeros_(residual_head.weight)
        nn.init.zeros_(residual_head.bias)

        # exploration distribution over the composed mean (learnable log std, proposal init -1.0)
        if distribution_cfg is None:
            distribution_cfg = {"class_name": "GaussianDistribution", "init_std": 0.3679, "std_type": "log"}
        distribution_cfg = dict(distribution_cfg)
        dist_class: type[Distribution] = resolve_callable(distribution_cfg.pop("class_name"))  # type: ignore
        self.distribution: Distribution = dist_class(output_dim, **distribution_cfg)
        if self.distribution.input_dim != output_dim:
            raise ValueError("TacResActor only supports distributions whose input is the action mean.")

        # caches from the last forward pass (used by TacResPPO auxiliary losses)
        self.last_gate: torch.Tensor | None = None
        self.last_residual: torch.Tensor | None = None
        self.last_residual_pre: torch.Tensor | None = None
        self.last_base_action: torch.Tensor | None = None

    # -- forward pass ------------------------------------------------------------------------

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state=None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        obs = unpad_trajectories(obs, masks) if masks is not None else obs
        # frozen base action (no grad may ever flow into the base policy)
        with torch.no_grad():
            base_action = self.base(obs)
        base_action = base_action.detach()
        # gated bounded residual
        residual_obs = torch.cat([obs[g] for g in self.residual_obs_groups], dim=-1)
        residual_in = torch.cat([self.residual_obs_normalizer(residual_obs), base_action], dim=-1)
        residual_pre = self.residual_mlp(residual_in)
        residual = torch.tanh(residual_pre)
        if self.fixed_gate is not None:
            gate = torch.full_like(residual[..., :1], self.fixed_gate)
        else:
            gate_obs = torch.cat([obs[g] for g in self.gate_obs_groups], dim=-1)
            gate = torch.sigmoid(self.gate_mlp(self.gate_obs_normalizer(gate_obs)))
        mean = base_action + self.alpha * gate * residual
        # cache for auxiliary losses
        self.last_gate = gate
        self.last_residual = residual
        self.last_residual_pre = residual_pre
        self.last_base_action = base_action
        if stochastic_output:
            self.distribution.update(mean)
            return self.distribution.sample()
        return mean

    # -- normalization / mode handling -------------------------------------------------------

    def update_normalization(self, obs: TensorDict) -> None:
        """Update residual/gate input normalizers. The base normalizer stays frozen (base is eval)."""
        if self.obs_normalization:
            residual_obs = torch.cat([obs[g] for g in self.residual_obs_groups], dim=-1)
            self.residual_obs_normalizer.update(residual_obs)
            gate_obs = torch.cat([obs[g] for g in self.gate_obs_groups], dim=-1)
            self.gate_obs_normalizer.update(gate_obs)

    def train(self, mode: bool = True) -> "TacResActor":
        super().train(mode)
        # the base policy (incl. its empirical normalizer) must never leave eval mode
        self.base.eval()
        return self

    # -- distribution interface (mirrors MLPModel) --------------------------------------------

    @property
    def output_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def output_std(self) -> torch.Tensor:
        return self.distribution.std

    @property
    def output_entropy(self) -> torch.Tensor:
        return self.distribution.entropy

    @property
    def output_distribution_params(self) -> tuple[torch.Tensor, ...]:
        return self.distribution.params

    def get_output_log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(outputs)

    def get_kl_divergence(self, old_params, new_params) -> torch.Tensor:
        return self.distribution.kl_divergence(old_params, new_params)

    # -- recurrent interface (no-ops, MLP-only) -----------------------------------------------

    def reset(self, dones: torch.Tensor | None = None, hidden_state=None) -> None:
        pass

    def get_hidden_state(self):
        return None

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        pass

    # -- export -------------------------------------------------------------------------------

    def as_jit(self) -> nn.Module:
        raise NotImplementedError("JIT export is not implemented for TacResActor yet.")

    def as_onnx(self, verbose: bool) -> nn.Module:
        raise NotImplementedError("ONNX export is not implemented for TacResActor yet.")

    # -- checkpoint helpers ---------------------------------------------------------------------

    def load_base_checkpoint(self, checkpoint_path: str, device: str) -> None:
        """Initialize the frozen base policy from a phase-1 checkpoint.

        Only ``actor_state_dict`` is read; the phase-1 critic is intentionally discarded because the
        phase-2 critic must be trained from scratch on the perturbed, tactile-privileged setting.
        """
        loaded = torch.load(checkpoint_path, weights_only=False, map_location=device)
        if "actor_state_dict" not in loaded:
            raise KeyError(
                f"Base checkpoint '{checkpoint_path}' has no 'actor_state_dict' "
                "(expected an rsl-rl >= 5.0 OnPolicyRunner checkpoint from the TacRes phase-1 task)."
            )
        self.base.load_state_dict(loaded["actor_state_dict"], strict=True)
        self.base.requires_grad_(False)
        self.base.eval()
