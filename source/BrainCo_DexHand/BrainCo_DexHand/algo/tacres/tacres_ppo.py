# Copyright (c) 2026, BrainCo.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO variant for TacRes phase 2: frozen base policy, gate warm-start, and residual regularization."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain

from rsl_rl.algorithms import PPO
from rsl_rl.utils import resolve_optimizer

from .tacres_actor import TacResActor


class TacResPPO(PPO):
    """PPO that trains only the residual, gate, critic, and action-std parameters.

    Extensions over vanilla PPO:

    - Loads the frozen base policy from a phase-1 checkpoint (actor weights only; the phase-1 critic
      is never loaded because the phase-2 critic must be re-initialized and trained from scratch).
    - Rebuilds the optimizer over trainable parameters only, excluding all frozen base parameters.
    - Adds the annealed auxiliary losses from the proposal:
      ``w(t) * BCE(g, y) + lambda_g(t) * E[g] + lambda_r * E[|da|^2]``.
    """

    actor: TacResActor

    def __init__(
        self,
        actor: TacResActor,
        critic,
        storage,
        base_checkpoint: str = "",
        gate_label_group: str = "tacres_gate_label",
        gate_warm_coef: float = 1.0,
        gate_warm_frac: float = 0.3,
        gate_warm_floor: float = 0.0,
        gate_sparsity_coef: float = 0.01,
        gate_sparsity_ramp_frac: float = 0.3,
        residual_l2_coef: float = 1.0e-3,
        schedule_total_iterations: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(actor, critic, storage, **kwargs)
        if self.rnd is not None or self.symmetry is not None:
            raise ValueError("TacResPPO does not support RND or symmetry extensions.")
        if not isinstance(actor, TacResActor):
            raise TypeError("TacResPPO requires a TacResActor as the actor model.")

        self.gate_label_group = gate_label_group
        self.gate_warm_coef = gate_warm_coef
        self.gate_warm_frac = gate_warm_frac
        self.gate_warm_floor = gate_warm_floor
        self.gate_sparsity_coef = gate_sparsity_coef
        self.gate_sparsity_ramp_frac = gate_sparsity_ramp_frac
        self.residual_l2_coef = residual_l2_coef
        self.total_iterations = max(int(schedule_total_iterations or 1), 1)
        self._iteration = 0
        self._base_grads_verified = False

        # initialize the frozen base policy from the phase-1 checkpoint
        if not base_checkpoint:
            raise ValueError(
                "TacRes phase 2 requires the phase-1 base checkpoint. Launch training with"
                " 'agent.algorithm.base_checkpoint=/path/to/model_<it>.pt'."
            )
        self.actor.load_base_checkpoint(base_checkpoint, self.device)
        print(f"[TacResPPO] Loaded frozen base policy from: {base_checkpoint}")
        print("[TacResPPO] Phase-1 critic deliberately NOT loaded; phase-2 critic trains from scratch.")

        # the optimizer must contain only trainable parameters (residual, gate, log_std, critic)
        trainable_params = [p for p in chain(self.actor.parameters(), self.critic.parameters()) if p.requires_grad]
        frozen_params = [p for p in self.actor.parameters() if not p.requires_grad]
        optimizer_name = kwargs.get("optimizer", "adam")
        self.optimizer = resolve_optimizer(optimizer_name)(trainable_params, lr=self.learning_rate)
        num_base = sum(p.numel() for p in self.actor.base.parameters())
        num_trainable = sum(p.numel() for p in trainable_params)
        assert all(not p.requires_grad for p in self.actor.base.parameters()), "Base policy must be frozen."
        print(
            f"[TacResPPO] Optimizer rebuilt: {len(trainable_params)} trainable tensors ({num_trainable} params);"
            f" excluded {len(frozen_params)} frozen base tensors ({num_base} params)."
        )

    # -- annealing schedules -------------------------------------------------------------------

    def _gate_warm_weight(self) -> float:
        """Linear 1 -> ``gate_warm_floor`` over the first ``gate_warm_frac`` of training.

        A nonzero floor keeps a weak event-alignment signal on the gate for the rest of training,
        which counteracts the mutual gate/residual collapse (sparsity closes the gate before the
        residual has become useful enough for PPO to defend it).
        """
        warm_iters = max(self.gate_warm_frac * self.total_iterations, 1.0)
        return self.gate_warm_coef * max(self.gate_warm_floor, 1.0 - self._iteration / warm_iters)

    def _gate_sparsity_weight(self) -> float:
        """0 during warm-start, then linear ramp to ``gate_sparsity_coef``."""
        warm_iters = self.gate_warm_frac * self.total_iterations
        if self._iteration < warm_iters:
            return 0.0
        ramp_iters = max(self.gate_sparsity_ramp_frac * self.total_iterations, 1.0)
        return self.gate_sparsity_coef * min(1.0, (self._iteration - warm_iters) / ramp_iters)

    # -- PPO update with auxiliary losses --------------------------------------------------------

    def update(self) -> dict[str, float]:  # noqa: C901
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        mean_gate_warm_loss = 0
        mean_gate = 0
        mean_residual_sq = 0
        mean_residual_pre_sq = 0

        gate_warm_weight = self._gate_warm_weight()
        gate_sparsity_weight = self._gate_sparsity_weight()

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for batch in generator:
            original_batch_size = batch.observations.batch_size[0]

            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)

            # recompute actions log prob and entropy under the current policy
            self.actor(
                batch.observations,
                masks=batch.masks,
                hidden_state=batch.hidden_states[0],
                stochastic_output=True,
            )
            actions_log_prob = self.actor.get_output_log_prob(batch.actions)
            values = self.critic(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1])
            distribution_params = tuple(p[:original_batch_size] for p in self.actor.output_distribution_params)
            entropy = self.actor.output_entropy[:original_batch_size]

            # KL-adaptive learning rate (same as vanilla PPO)
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = self.actor.get_kl_divergence(batch.old_distribution_params, distribution_params)
                    kl_mean = torch.mean(kl)

                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # surrogate loss
            ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))
            surrogate = -torch.squeeze(batch.advantages) * ratio
            surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # value function loss
            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
                value_losses = (values - batch.returns).pow(2)
                value_losses_clipped = (value_clipped - batch.returns).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (batch.returns - values).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()

            # TacRes auxiliary losses (gate warm-start BCE, gate sparsity, residual magnitude).
            # The magnitude penalty acts on the pre-tanh activations so it keeps a restoring
            # gradient even if tanh saturates (the post-tanh value has zero gradient there).
            gate = self.actor.last_gate
            residual_sq = self.actor.last_residual.pow(2).mean()
            residual_pre_sq = self.actor.last_residual_pre.pow(2).mean()
            loss = loss + self.residual_l2_coef * residual_pre_sq
            if self.actor.fixed_gate is None:
                gate_labels = batch.observations[self.gate_label_group]
                gate_warm_loss = F.binary_cross_entropy(gate.clamp(1e-6, 1.0 - 1e-6), gate_labels)
                loss = loss + gate_warm_weight * gate_warm_loss + gate_sparsity_weight * gate.mean()
            else:
                gate_warm_loss = torch.zeros((), device=self.device)

            # gradient step over trainable parameters only
            self.optimizer.zero_grad()
            loss.backward()

            if not self._base_grads_verified:
                leaked = [
                    name
                    for name, p in self.actor.base.named_parameters()
                    if p.grad is not None and p.grad.abs().max() > 0
                ]
                assert not leaked, f"Gradients leaked into the frozen base policy: {leaked}"
                self._base_grads_verified = True
                print("[TacResPPO] Verified: no gradients flow into the frozen base policy.")

            if self.is_multi_gpu:
                self.reduce_parameters()

            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()
            mean_gate_warm_loss += gate_warm_loss.item()
            mean_gate += gate.mean().item()
            mean_residual_sq += residual_sq.item()
            mean_residual_pre_sq += residual_pre_sq.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        self.storage.clear()
        self._iteration += 1

        return {
            "value": mean_value_loss / num_updates,
            "surrogate": mean_surrogate_loss / num_updates,
            "entropy": mean_entropy / num_updates,
            "gate_warm_bce": mean_gate_warm_loss / num_updates,
            "gate_mean": mean_gate / num_updates,
            "residual_sq": mean_residual_sq / num_updates,
            "residual_pre_sq": mean_residual_pre_sq / num_updates,
            "gate_warm_weight": gate_warm_weight,
            "gate_sparsity_weight": gate_sparsity_weight,
        }

    def train_mode(self) -> None:
        super().train_mode()
        # keep the frozen base policy (and its normalizer) in eval mode
        self.actor.base.eval()

    @staticmethod
    def construct_algorithm(obs, env, cfg: dict, device: str) -> "TacResPPO":
        """Construct TacResPPO, defaulting the annealing horizon to the runner's max_iterations."""
        if not cfg["algorithm"].get("schedule_total_iterations"):
            cfg["algorithm"]["schedule_total_iterations"] = cfg.get("max_iterations")
        return PPO.construct_algorithm(obs, env, cfg, device)
