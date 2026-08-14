# Copyright (c) 2026, BrainCo.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hierarchical PPO: a 21-D dexterous-hand master and a 2-D translation follower.

Both policies decide from the **same** environment state ``s_t`` and the joint
23-D action is executed by a single ``env.step`` per control cycle::

    master_result   = master.act(obs_t)                       # 21-D hand
    executed_hand   = clamp(master_result["actions"], -1, 1)
    tactile_latent  = master_result["tactile_latent"].detach() # [B, 128] GRU state
    follower_obs    = build_follower_obs(executed_hand, tactile_latent, xy_state_t)
    follower_result = follower.act(follower_obs)               # 2-D translation
    executed_xy     = clamp(follower_result["actions"], -1, 1)
    obs_{t+1}, r, done, info = env.step(cat([executed_hand, executed_xy]))

No physics step, render or sensor refresh happens between the two policy
forwards, and the tactile encoder runs exactly once per cycle.

Master and follower share the team reward but keep separate value functions,
input normalizers, optimizers, learning rates and PPO ratios::

    ratio_master   = exp(old_logp_master   - new_logp_master)
    ratio_follower = exp(old_logp_follower - new_logp_follower)

Curriculum (global, latched, fully checkpointed):

* **Stage 0** - master-only. The follower neither samples nor updates and the
  XY action is exactly ``[0, 0]``; the stage stays at zero under the same
  physical actuator. Once the EMA of the per-rollout signed mean angular
  velocity is strictly above ``activation_speed_threshold`` (0.8 rad/s) for
  ``activation_patience`` consecutive epochs, Stage 1 latches on permanently.
* **Stage 1** - follower-only. The master (weights *and* input normalizer) is
  frozen; the tactile latent is detached; the workspace and action scale ramp
  from their initial to their final values over ``xy_curriculum_ramp_steps``.
* **Stage 2** - optional joint fine-tuning. Enabled by YAML. The master actor
  trunk, action head and critic unfreeze at a small learning rate with a KL
  regularizer toward the Stage-1-start policy; the tactile encoder stays frozen
  by default. Disabled by default, so the run stays in Stage 1 forever - the
  clean follower-only ablation.
"""

from __future__ import annotations

import copy
import math
import os
import time

import torch
from tensorboardX import SummaryWriter

from BrainCo_DexHand.algo.hora.models.models import (
    ActorCritic,
    FollowerActorCritic,
    build_actor_critic_kwargs,
    validate_teacher_tactile_checkpoint_compatibility,
)
from BrainCo_DexHand.algo.hora.models.running_mean_std import RunningMeanStd
from BrainCo_DexHand.algo.hora.ppo.hierarchical_experience import (
    HierarchicalExperienceBuffer,
)
from BrainCo_DexHand.algo.hora.ppo.hierarchical_obs import (
    FOLLOWER_OBS_DIM,
    TACTILE_LATENT_DIM,
    follower_obs_from_env,
    validate_tactile_latent,
)
from BrainCo_DexHand.algo.hora.ppo.ppo import AdaptiveScheduler, policy_kl
from BrainCo_DexHand.algo.hora.utils.misc import (
    AverageScalarMeter,
    normalize_tensorboard_tag,
    tprint,
)

STAGE_MASTER = 0
STAGE_FOLLOWER = 1
STAGE_JOINT_FINETUNE = 2
STAGE_NAMES = {
    STAGE_MASTER: "stage0_master",
    STAGE_FOLLOWER: "stage1_follower",
    STAGE_JOINT_FINETUNE: "stage2_joint_finetune",
}

# Marker stored in every hierarchical checkpoint so a plain Stage-1 PPO
# checkpoint can never be mistaken for a full hierarchical resume.
CHECKPOINT_FORMAT = "hora_hierarchical_ppo_v1"


def _section(config, key):
    """Return a config sub-section as a mapping-like object (never ``None``)."""
    value = config.get(key, None) if hasattr(config, "get") else getattr(config, key, None)
    return value if value is not None else {}


def _get(section, key, default):
    """Read a scalar from an OmegaConf/dict section with a default."""
    if section is None:
        return default
    if hasattr(section, "get"):
        value = section.get(key, default)
    else:
        value = getattr(section, key, default)
    return default if value is None else value


def _resolve_minibatch_size(batch_size: int, preferred: int) -> int:
    """Return the largest divisor of ``batch_size`` not exceeding ``preferred``."""
    if batch_size <= 0 or preferred <= 0:
        raise ValueError(f"Invalid minibatch resolution: batch={batch_size}, preferred={preferred}")
    for size in range(min(batch_size, preferred), 0, -1):
        if batch_size % size == 0:
            return size
    raise RuntimeError("unreachable: 1 always divides a positive batch size")


class HierarchicalPPO:
    """Two-policy PPO trainer for the dexterous hand + 2-D translation stage."""

    def __init__(self, env, output_dif, full_config):
        self.device = full_config["rl_device"]
        self.full_config = full_config
        self.network_config = full_config.train.network
        self.ppo_config = full_config.train.ppo
        self.follower_config = _section(full_config.train, "follower")
        self.hier_config = _section(full_config.train, "hierarchical")

        # ---- environment / action contract ----
        self.env = env
        self.num_actors = int(self.ppo_config["num_actors"])
        action_space = self.env.action_space
        self.env_action_dim = int(action_space.shape[0])
        self.master_action_dim = int(
            getattr(self.env.cfg, "finger_action_space", self.env_action_dim - 2)
        )
        self.follower_action_dim = self.env_action_dim - self.master_action_dim
        if self.follower_action_dim != 2:
            raise ValueError(
                "HierarchicalPPO expects a 2-D horizontal translation follower: env "
                f"action_space={self.env_action_dim}, finger_action_space="
                f"{self.master_action_dim}"
            )
        self.observation_space = self.env.observation_space
        self.obs_shape = self.observation_space.shape

        # ---- master model (unchanged Stage-1 teacher) ----
        self.priv_info_dim = int(self.ppo_config["priv_info_dim"])
        net_config = build_actor_critic_kwargs(
            self.network_config,
            self.ppo_config,
            self.master_action_dim,
            self.obs_shape,
            self.obs_shape[0] // 3,
            False,
            env_cfg=self.env.cfg,
        )
        self.master = ActorCritic(net_config).to(self.device)
        if not getattr(self.master, "use_tactile_history", False):
            raise ValueError(
                "HierarchicalPPO requires the structured tactile master "
                "(network.tactile_encoder.type='finger_attention_gru')."
            )
        if self.master.tactile_latent_output_dim != TACTILE_LATENT_DIM:
            raise ValueError(
                "HierarchicalPPO requires a "
                f"{TACTILE_LATENT_DIM}-dimensional master tactile latent, got "
                f"{self.master.tactile_latent_output_dim}. Configure "
                "network.tactile_encoder.gru_hidden_dim=128."
            )
        self.tactile_hist_shape = (
            int(self.master.tactile_history_len),
            int(self.master.tactile_frame_dim),
        )

        # ---- follower model ----
        self.follower_obs_dim = FOLLOWER_OBS_DIM
        self.follower_critic_priv_dim = int(_get(self.follower_config, "critic_priv_dim", 11))
        if self.follower_critic_priv_dim > self.priv_info_dim:
            raise ValueError(
                f"follower.critic_priv_dim ({self.follower_critic_priv_dim}) exceeds "
                f"priv_info_dim ({self.priv_info_dim})"
            )
        self.follower = FollowerActorCritic(
            obs_dim=self.follower_obs_dim,
            actions_num=self.follower_action_dim,
            actor_units=list(_get(self.follower_config, "actor_units", [256, 128, 64])),
            critic_units=list(_get(self.follower_config, "critic_units", [256, 128, 64])),
            critic_priv_dim=self.follower_critic_priv_dim,
            init_log_sigma=float(_get(self.follower_config, "init_log_sigma", -1.0)),
        ).to(self.device)

        # ---- normalizers (independent per policy) ----
        self.normalize_input = bool(self.ppo_config["normalize_input"])
        self.normalize_value = bool(self.ppo_config["normalize_value"])
        self.master_running_mean_std = RunningMeanStd(self.obs_shape).to(self.device)
        self.master_value_mean_std = RunningMeanStd((1,)).to(self.device)
        self.follower_normalize_input = bool(
            _get(self.follower_config, "normalize_input", True)
        )
        self.follower_normalize_value = bool(
            _get(self.follower_config, "normalize_value", True)
        )
        self.follower_running_mean_std = RunningMeanStd((self.follower_obs_dim,)).to(self.device)
        self.follower_value_mean_std = RunningMeanStd((1,)).to(self.device)

        # ---- output dirs ----
        self.output_dir = output_dif
        self.nn_dir = os.path.join(self.output_dir, "hier_nn")
        self.tb_dif = os.path.join(self.output_dir, "hier_tb")
        os.makedirs(self.nn_dir, exist_ok=True)
        os.makedirs(self.tb_dif, exist_ok=True)
        self.writer = SummaryWriter(self.tb_dif)

        # ---- master PPO hyper-parameters (unchanged names) ----
        self.master_lr = float(self.ppo_config["learning_rate"])
        self.weight_decay = float(self.ppo_config.get("weight_decay", 0.0))
        self.tactile_encoder_lr_scale = float(
            self.ppo_config.get("tactile_encoder_lr_scale", 0.3)
        )
        self.master_e_clip = float(self.ppo_config["e_clip"])
        self.master_entropy_coef = float(self.ppo_config["entropy_coef"])
        self.master_critic_coef = float(self.ppo_config["critic_coef"])
        self.master_bounds_loss_coef = float(self.ppo_config["bounds_loss_coef"])
        self.master_kl_threshold = float(self.ppo_config["kl_threshold"])
        self.master_mini_epochs = int(self.ppo_config["mini_epochs"])
        self.master_grad_norm = float(self.ppo_config["grad_norm"])
        self.master_truncate_grads = bool(self.ppo_config["truncate_grads"])
        self.master_scheduler = AdaptiveScheduler(self.master_kl_threshold)

        # ---- follower PPO hyper-parameters ----
        self.follower_lr = float(_get(self.follower_config, "learning_rate", 3.0e-4))
        self.follower_weight_decay = float(_get(self.follower_config, "weight_decay", 0.0))
        self.follower_e_clip = float(_get(self.follower_config, "e_clip", 0.2))
        self.follower_entropy_coef = float(_get(self.follower_config, "entropy_coef", 0.005))
        self.follower_critic_coef = float(_get(self.follower_config, "critic_coef", 2.0))
        self.follower_bounds_loss_coef = float(
            _get(self.follower_config, "bounds_loss_coef", 0.0001)
        )
        self.follower_kl_threshold = float(_get(self.follower_config, "kl_threshold", 0.016))
        self.follower_mini_epochs = int(_get(self.follower_config, "mini_epochs", 5))
        self.follower_grad_norm = float(_get(self.follower_config, "grad_norm", 1.0))
        self.follower_truncate_grads = bool(_get(self.follower_config, "truncate_grads", True))
        self.follower_scheduler = AdaptiveScheduler(self.follower_kl_threshold)

        # ---- shared PPO collection settings ----
        self.gamma = float(self.ppo_config["gamma"])
        self.tau = float(self.ppo_config["tau"])
        self.value_bootstrap = bool(self.ppo_config["value_bootstrap"])
        self.normalize_advantage = bool(self.ppo_config["normalize_advantage"])
        self.clip_value = bool(self.ppo_config["clip_value"])
        self.reward_scale = float(self.ppo_config.get("reward_scale", 1.0))
        self.horizon_length = int(self.ppo_config["horizon_length"])
        self.batch_size = self.horizon_length * self.num_actors
        self.master_minibatch_size = _resolve_minibatch_size(
            self.batch_size, int(self.ppo_config["minibatch_size"])
        )
        self.follower_minibatch_size = _resolve_minibatch_size(
            self.batch_size,
            int(_get(self.follower_config, "minibatch_size", self.master_minibatch_size)),
        )
        self.max_agent_steps = int(self.ppo_config["max_agent_steps"])
        self.save_freq = int(self.ppo_config["save_frequency"])
        self.save_best_after = int(self.ppo_config["save_best_after"])

        # ---- curriculum state (fully checkpointed) ----
        self.current_stage = STAGE_MASTER
        self.activation_speed_threshold = float(
            _get(self.hier_config, "activation_speed_threshold", 0.8)
        )
        self.activation_patience = int(_get(self.hier_config, "activation_patience", 5))
        self.activation_speed_ema_beta = float(
            _get(self.hier_config, "activation_speed_ema_beta", 0.9)
        )
        if not 0.0 <= self.activation_speed_ema_beta < 1.0:
            raise ValueError("hierarchical.activation_speed_ema_beta must be in [0, 1)")
        if self.activation_patience < 1:
            raise ValueError("hierarchical.activation_patience must be >= 1")
        self.activation_speed_ema = 0.0
        self._activation_speed_ema_initialized = False
        self.activation_patience_counter = 0
        self.activation_agent_step = -1
        self.stage_start_agent_step = 0
        self.joint_finetune_enable = bool(
            _get(self.hier_config, "joint_finetune_enable", False)
        )
        self.follower_only_steps = int(
            _get(self.hier_config, "follower_only_steps", 50_000_000)
        )
        self.master_finetune_lr_scale = float(
            _get(self.hier_config, "master_finetune_lr_scale", 0.07)
        )
        if not 0.0 < self.master_finetune_lr_scale <= 1.0:
            raise ValueError("hierarchical.master_finetune_lr_scale must be in (0, 1]")
        self.master_kl_coef = float(_get(self.hier_config, "master_kl_coef", 1.0))
        self.freeze_tactile_encoder_in_joint_finetune = bool(
            _get(self.hier_config, "freeze_tactile_encoder_in_joint_finetune", True)
        )
        self.xy_curriculum_ramp_steps = int(
            _get(
                self.hier_config,
                "xy_curriculum_ramp_steps",
                getattr(self.env.cfg, "xy_curriculum_ramp_steps", 20_000_000),
            )
        )
        self.master_reference = None

        # ---- optimizers ----
        self.master_optimizer = self._build_master_optimizer()
        self.follower_optimizer = torch.optim.Adam(
            self.follower.parameters(),
            self.follower_lr,
            weight_decay=self.follower_weight_decay,
        )

        # ---- rollout storage ----
        self.storage = HierarchicalExperienceBuffer(
            self.num_actors,
            self.horizon_length,
            self.device,
            master_obs_dim=self.obs_shape[0],
            master_action_dim=self.master_action_dim,
            priv_info_dim=self.priv_info_dim,
            tactile_hist_shape=self.tactile_hist_shape,
            follower_obs_dim=self.follower_obs_dim,
            follower_action_dim=self.follower_action_dim,
            follower_critic_priv_dim=self.follower_critic_priv_dim,
            master_minibatch_size=self.master_minibatch_size,
            follower_minibatch_size=self.follower_minibatch_size,
        )

        # ---- episode statistics ----
        self.extra_info = {}
        self.episode_rewards = AverageScalarMeter(100)
        self.episode_raw_rewards = AverageScalarMeter(100)
        self.episode_lengths = AverageScalarMeter(100)
        self.current_rewards = torch.zeros(
            (self.num_actors, 1), dtype=torch.float32, device=self.device
        )
        self.current_raw_rewards = torch.zeros_like(self.current_rewards)
        self.current_lengths = torch.zeros(
            self.num_actors, dtype=torch.float32, device=self.device
        )
        self.dones = torch.ones((self.num_actors,), dtype=torch.uint8, device=self.device)
        self.obs = None
        self.epoch_num = 0
        self.agent_steps = 0
        self.best_rewards = -10000.0
        self.best_angular_velocity = -10000.0
        self.data_collect_time = 0.0
        self.rl_train_time = 0.0
        self.last_rollout_speed = 0.0
        self._apply_stage_freeze()

    # ------------------------------------------------------------------
    # construction helpers
    # ------------------------------------------------------------------

    def _build_master_optimizer(self):
        """Adam over the master with a reduced tactile-encoder learning rate."""
        tactile_encoder = getattr(self.master, "tactile_encoder", None)
        if tactile_encoder is None or abs(self.tactile_encoder_lr_scale - 1.0) < 1.0e-8:
            return torch.optim.Adam(
                self.master.parameters(), self.master_lr, weight_decay=self.weight_decay
            )
        encoder_params = list(tactile_encoder.parameters())
        encoder_ids = {id(p) for p in encoder_params}
        other_params = [p for p in self.master.parameters() if id(p) not in encoder_ids]
        return torch.optim.Adam(
            [
                {"params": other_params, "lr": self.master_lr},
                {
                    "params": encoder_params,
                    "lr": self.master_lr * self.tactile_encoder_lr_scale,
                },
            ],
            weight_decay=self.weight_decay,
        )

    def _set_master_lr(self, base_lr: float) -> None:
        self.master_lr = float(base_lr)
        groups = self.master_optimizer.param_groups
        groups[0]["lr"] = self.master_lr
        if len(groups) > 1:
            groups[1]["lr"] = self.master_lr * self.tactile_encoder_lr_scale

    def _set_follower_lr(self, base_lr: float) -> None:
        self.follower_lr = float(base_lr)
        for group in self.follower_optimizer.param_groups:
            group["lr"] = self.follower_lr

    # ------------------------------------------------------------------
    # curriculum
    # ------------------------------------------------------------------

    @property
    def master_frozen(self) -> bool:
        """The master is frozen for the whole follower-only stage."""
        return self.current_stage == STAGE_FOLLOWER

    @property
    def follower_active(self) -> bool:
        """The follower samples and updates from Stage 1 onward."""
        return self.current_stage >= STAGE_FOLLOWER

    def _apply_stage_freeze(self) -> None:
        """Apply the parameter/normalizer freeze pattern of the current stage."""
        if self.current_stage == STAGE_MASTER:
            self.master.requires_grad_(True)
            return
        if self.current_stage == STAGE_FOLLOWER:
            # Freeze the weights AND the input normalizer: a drifting
            # normalizer would silently change a "frozen" policy.
            self.master.requires_grad_(False)
            return
        # Stage 2: unfreeze the actor trunk, action head and critic. The tactile
        # encoder stays frozen by default so the perception front-end that the
        # follower conditions on does not move underneath it.
        self.master.requires_grad_(True)
        if self.freeze_tactile_encoder_in_joint_finetune:
            tactile_encoder = getattr(self.master, "tactile_encoder", None)
            if tactile_encoder is not None:
                tactile_encoder.requires_grad_(False)

    def _activate_follower(self) -> None:
        """Latch Stage 1 on. This transition is permanent."""
        self.current_stage = STAGE_FOLLOWER
        self.activation_agent_step = int(self.agent_steps)
        self.stage_start_agent_step = int(self.agent_steps)
        self._apply_stage_freeze()
        if self.joint_finetune_enable:
            # Snapshot the Stage-1-start master: the Stage-2 KL regularizer is
            # measured against exactly this policy.
            self.master_reference = copy.deepcopy(self.master).to(self.device)
            self.master_reference.requires_grad_(False)
            self.master_reference.eval()
        print(
            f"[INFO] Hierarchical curriculum: activating the XY follower at "
            f"agent_steps={self.agent_steps} "
            f"(speed EMA={self.activation_speed_ema:.4f} rad/s > "
            f"{self.activation_speed_threshold:.2f} for {self.activation_patience} epochs). "
            "The master is now frozen.",
            flush=True,
        )

    def _enter_joint_finetune(self) -> None:
        self.current_stage = STAGE_JOINT_FINETUNE
        self._apply_stage_freeze()
        print(
            f"[INFO] Hierarchical curriculum: entering joint fine-tuning at "
            f"agent_steps={self.agent_steps} (master lr scale "
            f"{self.master_finetune_lr_scale}, KL coef {self.master_kl_coef}).",
            flush=True,
        )

    def _update_curriculum(self, rollout_speed: float) -> None:
        """Update the latched global curriculum from one rollout's mean speed.

        Args:
            rollout_speed: Signed mean valve angular velocity of the rollout.
        """
        self.last_rollout_speed = float(rollout_speed)
        if self._activation_speed_ema_initialized:
            beta = self.activation_speed_ema_beta
            self.activation_speed_ema = (
                beta * self.activation_speed_ema + (1.0 - beta) * float(rollout_speed)
            )
        else:
            self.activation_speed_ema = float(rollout_speed)
            self._activation_speed_ema_initialized = True

        if self.current_stage == STAGE_MASTER:
            if self.activation_speed_ema > self.activation_speed_threshold:
                self.activation_patience_counter += 1
            else:
                self.activation_patience_counter = 0
            if self.activation_patience_counter >= self.activation_patience:
                self._activate_follower()
            return

        # Stage >= 1 is latched: a later speed drop never reverts the stage.
        if (
            self.current_stage == STAGE_FOLLOWER
            and self.joint_finetune_enable
            and (self.agent_steps - self.stage_start_agent_step) >= self.follower_only_steps
        ):
            self._enter_joint_finetune()

    def xy_curriculum_progress(self) -> float:
        """Return the workspace / action-scale ramp progress in ``[0, 1]``."""
        if self.current_stage == STAGE_MASTER:
            return 0.0
        if self.xy_curriculum_ramp_steps <= 0:
            return 1.0
        progress = (
            self.agent_steps - self.stage_start_agent_step
        ) / float(self.xy_curriculum_ramp_steps)
        return min(max(progress, 0.0), 1.0)

    def _push_xy_curriculum(self) -> tuple[float, float]:
        """Publish the current workspace / action scale to the environment."""
        progress = self.xy_curriculum_progress()
        setter = getattr(self.env, "set_xy_curriculum_progress", None)
        if setter is None:
            raise RuntimeError(
                "HierarchicalPPO requires an environment exposing "
                "set_xy_curriculum_progress(); use --task valvedriver_tactile_xy."
            )
        return setter(progress)

    # ------------------------------------------------------------------
    # train / eval mode
    # ------------------------------------------------------------------

    def set_eval(self):
        self.master.eval()
        self.follower.eval()
        if self.normalize_input:
            self.master_running_mean_std.eval()
        if self.normalize_value:
            self.master_value_mean_std.eval()
        if self.follower_normalize_input:
            self.follower_running_mean_std.eval()
        if self.follower_normalize_value:
            self.follower_value_mean_std.eval()

    def set_train(self):
        # A frozen master stays in eval mode; so does its input normalizer, so
        # the policy that the follower conditions on is genuinely stationary.
        if self.current_stage == STAGE_MASTER:
            self.master.train()
            if self.normalize_input:
                self.master_running_mean_std.train()
            if self.normalize_value:
                self.master_value_mean_std.train()
        else:
            self.master.eval()
            self.master_running_mean_std.eval()
            if self.current_stage == STAGE_JOINT_FINETUNE:
                self.master.train()
                if self.normalize_value:
                    self.master_value_mean_std.train()
        self.follower.train()
        if self.follower_normalize_input and self.follower_active:
            self.follower_running_mean_std.train()
        if self.follower_normalize_value and self.follower_active:
            self.follower_value_mean_std.train()

    # ------------------------------------------------------------------
    # policy forwards
    # ------------------------------------------------------------------

    @torch.no_grad()
    def master_act(self, obs_dict) -> dict:
        """One master forward: action distribution, value and tactile latent."""
        processed_obs = (
            self.master_running_mean_std(obs_dict["obs"])
            if self.normalize_input
            else obs_dict["obs"]
        )
        if "tactile_hist" not in obs_dict:
            raise KeyError(
                "HierarchicalPPO requires obs['tactile_hist'] from the tactile task."
            )
        result = self.master.act(
            {
                "obs": processed_obs,
                "priv_info": obs_dict["priv_info"],
                "tactile_hist": obs_dict["tactile_hist"],
            }
        )
        if self.normalize_value:
            result["values"] = self.master_value_mean_std(result["values"], True)
        validate_tactile_latent(result.get("tactile_latent"))
        return result

    @torch.no_grad()
    def follower_act(self, follower_obs, critic_priv) -> dict:
        """One follower forward on the strict 159-D observation."""
        if follower_obs.shape[-1] != self.follower_obs_dim:
            raise RuntimeError(
                f"Follower observation must be [B, {self.follower_obs_dim}], got "
                f"{tuple(follower_obs.shape)}"
            )
        processed = (
            self.follower_running_mean_std(follower_obs)
            if self.follower_normalize_input
            else follower_obs
        )
        result = self.follower.act(processed, critic_priv)
        if self.follower_normalize_value:
            result["values"] = self.follower_value_mean_std(result["values"], True)
        return result

    def _critic_priv(self, obs_dict):
        """Return the privileged slice that only the follower critic may read."""
        if self.follower_critic_priv_dim == 0:
            return None
        return obs_dict["priv_info"][:, : self.follower_critic_priv_dim]

    # ------------------------------------------------------------------
    # rollout
    # ------------------------------------------------------------------

    def _as_per_env_column(self, value, name, expected_envs, dtype):
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value, device=self.device)
        value = value.to(device=self.device, dtype=dtype)
        if value.numel() != expected_envs:
            raise RuntimeError(
                f"{name} must contain one value per environment; got shape "
                f"{tuple(value.shape)} for {expected_envs} environments"
            )
        return value.reshape(expected_envs, 1)

    def play_steps(self) -> float:
        """Collect one rollout and return the signed mean angular velocity."""
        speed_sum = 0.0
        speed_count = 0
        zeros_xy = torch.zeros(
            (self.num_actors, self.follower_action_dim), device=self.device
        )

        for n in range(self.horizon_length):
            obs_dict = self.obs

            # ---- master decides from s_t ----
            master_result = self.master_act(obs_dict)
            sampled_hand_action = master_result["actions"]
            executed_hand_action = torch.clamp(sampled_hand_action, -1.0, 1.0)
            tactile_latent = master_result["tactile_latent"].detach()

            # ---- follower decides from the SAME s_t, no env interaction yet ----
            follower_obs = follower_obs_from_env(
                obs_dict,
                executed_hand_action=executed_hand_action,
                tactile_latent=tactile_latent,
            )
            if follower_obs.shape[-1] != FOLLOWER_OBS_DIM:
                raise RuntimeError(
                    f"follower_obs.shape[-1] == {follower_obs.shape[-1]} != {FOLLOWER_OBS_DIM}"
                )
            critic_priv = self._critic_priv(obs_dict)
            if self.follower_active:
                follower_result = self.follower_act(follower_obs, critic_priv)
                sampled_xy_action = follower_result["actions"]
                executed_xy_action = torch.clamp(sampled_xy_action, -1.0, 1.0)
            else:
                # Stage 0: the follower neither samples nor updates; the stage is
                # held at zero by the same physical actuator.
                follower_result = {
                    "actions": zeros_xy,
                    "neglogpacs": torch.zeros(self.num_actors, device=self.device),
                    "values": torch.zeros((self.num_actors, 1), device=self.device),
                    "mus": zeros_xy,
                    "sigmas": torch.ones_like(zeros_xy),
                }
                sampled_xy_action = zeros_xy
                executed_xy_action = zeros_xy

            # ---- store o_t and both decisions BEFORE the single env step ----
            self.storage.update_data("obses", n, obs_dict["obs"])
            self.storage.update_data("priv_info", n, obs_dict["priv_info"])
            self.storage.update_data("tactile_hist", n, obs_dict["tactile_hist"])
            self.storage.update_data("master_actions", n, sampled_hand_action.detach())
            self.storage.update_data("master_executed_actions", n, executed_hand_action.detach())
            self.storage.update_data("master_neglogpacs", n, master_result["neglogpacs"].detach())
            self.storage.update_data("master_values", n, master_result["values"].detach())
            self.storage.update_data("master_mus", n, master_result["mus"].detach())
            self.storage.update_data("master_sigmas", n, master_result["sigmas"].detach())
            self.storage.update_data("follower_obses", n, follower_obs.detach())
            if critic_priv is not None:
                self.storage.update_data("follower_critic_priv", n, critic_priv.detach())
            self.storage.update_data("follower_actions", n, sampled_xy_action.detach())
            self.storage.update_data(
                "follower_executed_actions", n, executed_xy_action.detach()
            )
            self.storage.update_data(
                "follower_neglogpacs", n, follower_result["neglogpacs"].detach()
            )
            self.storage.update_data("follower_values", n, follower_result["values"].detach())
            self.storage.update_data("follower_mus", n, follower_result["mus"].detach())
            self.storage.update_data("follower_sigmas", n, follower_result["sigmas"].detach())

            # ---- exactly one environment interaction per control cycle ----
            joint_action = torch.cat([executed_hand_action, executed_xy_action], dim=-1)
            if joint_action.shape[-1] != self.env_action_dim:
                raise RuntimeError(
                    f"Joint action must be [B, {self.env_action_dim}], got "
                    f"{tuple(joint_action.shape)}"
                )
            self.storage.update_data("env_actions", n, joint_action.detach())
            self.obs, rewards, self.dones, infos = self.env.step(joint_action)
            rewards = rewards.unsqueeze(1)
            assert isinstance(infos, dict), "Info Should be a Dict"

            self.storage.update_data("dones", n, self.dones)
            shaped_rewards = self.reward_scale * rewards
            self.storage.update_data("rewards", n, shaped_rewards)
            master_rewards = shaped_rewards.clone()
            follower_rewards = shaped_rewards.clone()
            time_outs = None
            if "time_outs" in infos:
                time_outs = self._as_per_env_column(
                    infos["time_outs"], "time_outs", rewards.shape[0], rewards.dtype
                )
                self.storage.update_data("timeouts", n, time_outs.squeeze(1))
            if self.value_bootstrap and time_outs is not None:
                # Each policy bootstraps truncated episodes with its OWN value.
                master_rewards = master_rewards + self.gamma * time_outs * (
                    master_result["values"].detach()
                )
                follower_rewards = follower_rewards + self.gamma * time_outs * (
                    follower_result["values"].detach()
                )
            self.storage.update_data("master_rewards", n, master_rewards)
            self.storage.update_data("follower_rewards", n, follower_rewards)

            speed_sum += self._rollout_speed_from_infos(infos)
            speed_count += 1

            self.current_rewards += self.reward_scale * rewards
            self.current_raw_rewards += rewards
            self.current_lengths += 1
            done_indices = self.dones.nonzero(as_tuple=False)
            self.episode_rewards.update(self.current_rewards[done_indices])
            self.episode_raw_rewards.update(self.current_raw_rewards[done_indices])
            self.episode_lengths.update(self.current_lengths[done_indices])

            self.extra_info = {}
            for key, value in infos.items():
                if isinstance(value, (float, int)) or (
                    isinstance(value, torch.Tensor) and len(value.shape) == 0
                ):
                    if isinstance(value, torch.Tensor):
                        value = value.item()
                    if isinstance(key, str) and key.startswith("rew/"):
                        self.extra_info[key] = float(value) * self.reward_scale
                    else:
                        self.extra_info[key] = value

            not_dones = 1.0 - self.dones.float()
            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_raw_rewards = self.current_raw_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones

        # ---- bootstrap values after the rollout ----
        last_master = self.master_act(self.obs)
        last_follower_obs = follower_obs_from_env(
            self.obs,
            executed_hand_action=torch.clamp(last_master["actions"], -1.0, 1.0),
            tactile_latent=last_master["tactile_latent"].detach(),
        )
        last_follower = self.follower_act(last_follower_obs, self._critic_priv(self.obs))

        self.agent_steps += self.batch_size
        self.storage.compute_returns(
            last_master["values"],
            last_follower["values"],
            self.gamma,
            self.tau,
        )
        self.storage.prepare_training(normalize_advantage=self.normalize_advantage)
        self._normalize_stored_values()
        return speed_sum / max(speed_count, 1)

    def _rollout_speed_from_infos(self, infos) -> float:
        """Return the signed mean angular velocity of one environment step."""
        per_env = infos.get("metrics/angular_velocity_per_env")
        if isinstance(per_env, torch.Tensor) and per_env.numel() > 0:
            return float(per_env.float().mean().item())
        scalar = infos.get("screw/angular_velocity")
        if isinstance(scalar, torch.Tensor):
            return float(scalar.float().mean().item())
        if isinstance(scalar, (int, float)):
            return float(scalar)
        raise RuntimeError(
            "The hierarchical curriculum needs the signed valve angular velocity; "
            "the environment published neither 'metrics/angular_velocity_per_env' "
            "nor 'screw/angular_velocity'."
        )

    def _normalize_stored_values(self) -> None:
        """Apply the running value normalizers to both stored value streams."""
        data = self.storage.data_dict
        if self.normalize_value:
            self.master_value_mean_std.train()
            data["master_values"] = self.master_value_mean_std(data["master_values"])
            data["master_returns"] = self.master_value_mean_std(data["master_returns"])
            self.master_value_mean_std.eval()
        # In Stage 0 the follower never samples, so its stored values are all
        # zero. Feeding those into the running normalizer would collapse its
        # variance long before the follower is ever used.
        if self.follower_normalize_value and self.follower_active:
            self.follower_value_mean_std.train()
            data["follower_values"] = self.follower_value_mean_std(data["follower_values"])
            data["follower_returns"] = self.follower_value_mean_std(data["follower_returns"])
            self.follower_value_mean_std.eval()

    # ------------------------------------------------------------------
    # updates
    # ------------------------------------------------------------------

    def _bounds_loss(self, mu: torch.Tensor, coef: float) -> torch.Tensor:
        if coef <= 0.0:
            return torch.zeros((), device=self.device)
        soft_bound = 1.1
        mu_loss_high = torch.clamp_max(mu - soft_bound, 0.0) ** 2
        mu_loss_low = torch.clamp_max(-mu + soft_bound, 0.0) ** 2
        return (mu_loss_low + mu_loss_high).sum(dim=-1).mean()

    def update_master(self):
        """Run the master PPO update (Stage 0, and Stage 2 fine-tuning)."""
        actor_losses, critic_losses, entropies, kls = [], [], [], []
        kl_regularizers = []
        joint_finetune = self.current_stage == STAGE_JOINT_FINETUNE
        if joint_finetune:
            # In Stage 2 the master learning rate is slaved to the follower's
            # (which was already adapted this epoch), not to its own KL schedule.
            self._set_master_lr(self.follower_lr * self.master_finetune_lr_scale)
        for _ in range(self.master_mini_epochs):
            epoch_kls = []
            for index in range(self.storage.num_master_minibatches):
                batch = self.storage.master_minibatch(index)
                obs = (
                    self.master_running_mean_std(batch["obses"])
                    if self.normalize_input
                    else batch["obses"]
                )
                batch_dict = {
                    "prev_actions": batch["master_actions"],
                    "obs": obs,
                    "priv_info": batch["priv_info"],
                    "tactile_hist": batch["tactile_hist"],
                }
                result = self.master(batch_dict)
                mu, sigma = result["mus"], result["sigmas"]
                advantage = batch["master_advantages"]

                # Master PPO ratio, independent of the follower's.
                ratio = torch.exp(batch["master_neglogpacs"] - result["prev_neglogp"])
                clipped_ratio = torch.clamp(
                    ratio, 1.0 - self.master_e_clip, 1.0 + self.master_e_clip
                )
                actor_loss = torch.max(
                    -advantage * ratio, -advantage * clipped_ratio
                ).mean()

                values = result["values"]
                value_preds = batch["master_values"]
                returns = batch["master_returns"]
                value_pred_clipped = value_preds + (values - value_preds).clamp(
                    -self.master_e_clip, self.master_e_clip
                )
                critic_loss = torch.max(
                    (values - returns) ** 2, (value_pred_clipped - returns) ** 2
                ).mean()
                entropy = result["entropy"].mean()
                bounds_loss = self._bounds_loss(mu, self.master_bounds_loss_coef)

                loss = (
                    actor_loss
                    + 0.5 * critic_loss * self.master_critic_coef
                    - entropy * self.master_entropy_coef
                    + bounds_loss * self.master_bounds_loss_coef
                )
                kl_regularizer = torch.zeros((), device=self.device)
                if joint_finetune and self.master_reference is not None:
                    with torch.no_grad():
                        reference = self.master_reference(batch_dict)
                    # KL(new || Stage-1-start reference): keeps fine-tuning close
                    # to the policy the follower was trained against.
                    kl_regularizer = policy_kl(
                        mu, sigma, reference["mus"], reference["sigmas"]
                    )
                    loss = loss + self.master_kl_coef * kl_regularizer
                    kl_regularizers.append(kl_regularizer.detach())

                self.master_optimizer.zero_grad()
                loss.backward()
                if self.master_truncate_grads:
                    torch.nn.utils.clip_grad_norm_(
                        self.master.parameters(), self.master_grad_norm
                    )
                self.master_optimizer.step()

                with torch.no_grad():
                    epoch_kls.append(
                        policy_kl(
                            mu.detach(),
                            sigma.detach(),
                            batch["master_mus"],
                            batch["master_sigmas"],
                        )
                    )
                actor_losses.append(actor_loss.detach())
                critic_losses.append(critic_loss.detach())
                entropies.append(entropy.detach())
                self.storage.update_master_mu_sigma(mu.detach(), sigma.detach())

            average_kl = (
                torch.mean(torch.stack(epoch_kls))
                if epoch_kls
                else torch.zeros((), device=self.device)
            )
            kls.append(average_kl)
            if not joint_finetune:
                self._set_master_lr(
                    self.master_scheduler.update(self.master_lr, average_kl.item())
                )
        return actor_losses, critic_losses, entropies, kls, kl_regularizers

    def update_follower(self):
        """Run the follower PPO update with its own ratio and optimizer."""
        actor_losses, critic_losses, entropies, kls = [], [], [], []
        for _ in range(self.follower_mini_epochs):
            epoch_kls = []
            for index in range(self.storage.num_follower_minibatches):
                batch = self.storage.follower_minibatch(index)
                obs = (
                    self.follower_running_mean_std(batch["follower_obses"])
                    if self.follower_normalize_input
                    else batch["follower_obses"]
                )
                critic_priv = (
                    batch["follower_critic_priv"]
                    if self.follower_critic_priv_dim > 0
                    else None
                )
                result = self.follower(
                    {
                        "obs": obs,
                        # The conditioning hand action and tactile latent are
                        # stored values: no gradient ever flows back to the
                        # master through the follower loss.
                        "prev_actions": batch["follower_actions"],
                        "critic_priv": critic_priv,
                    }
                )
                mu, sigma = result["mus"], result["sigmas"]
                advantage = batch["follower_advantages"]

                ratio = torch.exp(batch["follower_neglogpacs"] - result["prev_neglogp"])
                clipped_ratio = torch.clamp(
                    ratio, 1.0 - self.follower_e_clip, 1.0 + self.follower_e_clip
                )
                actor_loss = torch.max(
                    -advantage * ratio, -advantage * clipped_ratio
                ).mean()

                values = result["values"]
                value_preds = batch["follower_values"]
                returns = batch["follower_returns"]
                value_pred_clipped = value_preds + (values - value_preds).clamp(
                    -self.follower_e_clip, self.follower_e_clip
                )
                critic_loss = torch.max(
                    (values - returns) ** 2, (value_pred_clipped - returns) ** 2
                ).mean()
                entropy = result["entropy"].mean()
                bounds_loss = self._bounds_loss(mu, self.follower_bounds_loss_coef)

                loss = (
                    actor_loss
                    + 0.5 * critic_loss * self.follower_critic_coef
                    - entropy * self.follower_entropy_coef
                    + bounds_loss * self.follower_bounds_loss_coef
                )
                self.follower_optimizer.zero_grad()
                loss.backward()
                if self.follower_truncate_grads:
                    torch.nn.utils.clip_grad_norm_(
                        self.follower.parameters(), self.follower_grad_norm
                    )
                self.follower_optimizer.step()

                with torch.no_grad():
                    epoch_kls.append(
                        policy_kl(
                            mu.detach(),
                            sigma.detach(),
                            batch["follower_mus"],
                            batch["follower_sigmas"],
                        )
                    )
                actor_losses.append(actor_loss.detach())
                critic_losses.append(critic_loss.detach())
                entropies.append(entropy.detach())
                self.storage.update_follower_mu_sigma(mu.detach(), sigma.detach())

            average_kl = (
                torch.mean(torch.stack(epoch_kls))
                if epoch_kls
                else torch.zeros((), device=self.device)
            )
            kls.append(average_kl)
            self._set_follower_lr(
                self.follower_scheduler.update(self.follower_lr, average_kl.item())
            )
        return actor_losses, critic_losses, entropies, kls

    def train_epoch(self) -> dict:
        """Collect one rollout and update whichever policies the stage allows."""
        collect_start = time.time()
        self.set_eval()
        rollout_speed = self.play_steps()
        collect_time = time.time() - collect_start
        self.data_collect_time += collect_time

        learn_start = time.time()
        self.set_train()
        stats = {
            "master_actor": [],
            "master_critic": [],
            "master_entropy": [],
            "master_kl": [],
            "master_kl_reg": [],
            "follower_actor": [],
            "follower_critic": [],
            "follower_entropy": [],
            "follower_kl": [],
        }
        # The follower updates first so the Stage-2 master learning rate can be
        # slaved to this epoch's follower learning rate. The two losses are
        # independent (separate parameters, separate ratios), so the order does
        # not otherwise affect the update.
        if self.follower_active:
            (
                stats["follower_actor"],
                stats["follower_critic"],
                stats["follower_entropy"],
                stats["follower_kl"],
            ) = self.update_follower()
        # Stage 1 freezes the master completely: no optimizer step at all.
        if self.current_stage in (STAGE_MASTER, STAGE_JOINT_FINETUNE):
            (
                stats["master_actor"],
                stats["master_critic"],
                stats["master_entropy"],
                stats["master_kl"],
                stats["master_kl_reg"],
            ) = self.update_master()
        learn_time = time.time() - learn_start
        self.rl_train_time += learn_time
        stats["rollout_speed"] = rollout_speed
        stats["collect_time"] = collect_time
        stats["learn_time"] = learn_time
        return stats

    # ------------------------------------------------------------------
    # logging
    # ------------------------------------------------------------------

    @staticmethod
    def _mean_or_none(items):
        if not items:
            return None
        return torch.mean(torch.stack(items)).item()

    def write_stats(self, stats, workspace: float, action_scale: float) -> None:
        step = self.agent_steps
        if self.rl_train_time > 0:
            self.writer.add_scalar(
                "performance/RLTrainFPS", self.agent_steps / self.rl_train_time, step
            )
        if self.data_collect_time > 0:
            self.writer.add_scalar(
                "performance/EnvStepFPS", self.agent_steps / self.data_collect_time, step
            )
        scalar_map = {
            "losses/master_actor": self._mean_or_none(stats["master_actor"]),
            "losses/master_critic": self._mean_or_none(stats["master_critic"]),
            "losses/master_entropy": self._mean_or_none(stats["master_entropy"]),
            "losses/follower_actor": self._mean_or_none(stats["follower_actor"]),
            "losses/follower_critic": self._mean_or_none(stats["follower_critic"]),
            "losses/follower_entropy": self._mean_or_none(stats["follower_entropy"]),
            "info/master_kl": self._mean_or_none(stats["master_kl"]),
            "info/follower_kl": self._mean_or_none(stats["follower_kl"]),
            "info/master_kl_regularizer": self._mean_or_none(stats["master_kl_reg"]),
        }
        for tag, value in scalar_map.items():
            if value is not None:
                self.writer.add_scalar(tag, value, step)
        self.writer.add_scalar("info/master_lr", self.master_lr, step)
        self.writer.add_scalar("info/follower_lr", self.follower_lr, step)

        self.writer.add_scalar("curriculum/hierarchical_stage", self.current_stage, step)
        self.writer.add_scalar(
            "curriculum/activation_speed_ema", self.activation_speed_ema, step
        )
        self.writer.add_scalar(
            "curriculum/activation_patience_counter",
            self.activation_patience_counter,
            step,
        )
        self.writer.add_scalar("curriculum/xy_workspace", workspace, step)
        self.writer.add_scalar("curriculum/xy_action_scale", action_scale, step)
        self.writer.add_scalar(
            "curriculum/xy_curriculum_progress", self.xy_curriculum_progress(), step
        )
        self.writer.add_scalar(
            "hierarchical/master_frozen", float(self.master_frozen), step
        )
        self.writer.add_scalar(
            "hierarchical/joint_finetune_enabled", float(self.joint_finetune_enable), step
        )
        self.writer.add_scalar(
            "hierarchical/follower_active", float(self.follower_active), step
        )
        self.writer.add_scalar(
            "screw/rollout_signed_mean_angular_velocity", stats["rollout_speed"], step
        )
        for key, value in self.extra_info.items():
            self.writer.add_scalar(normalize_tensorboard_tag(key), value, step)

    # ------------------------------------------------------------------
    # checkpoints
    # ------------------------------------------------------------------

    def save(self, name: str) -> None:
        """Persist models, optimizers, normalizers and the complete curriculum."""
        payload = {
            "format": CHECKPOINT_FORMAT,
            "master": self.master.state_dict(),
            "follower": self.follower.state_dict(),
            "master_optimizer": self.master_optimizer.state_dict(),
            "follower_optimizer": self.follower_optimizer.state_dict(),
            "master_running_mean_std": self.master_running_mean_std.state_dict(),
            "master_value_mean_std": self.master_value_mean_std.state_dict(),
            "follower_running_mean_std": self.follower_running_mean_std.state_dict(),
            "follower_value_mean_std": self.follower_value_mean_std.state_dict(),
            "current_stage": int(self.current_stage),
            "activation_speed_ema": float(self.activation_speed_ema),
            "activation_speed_ema_initialized": bool(self._activation_speed_ema_initialized),
            "activation_patience_counter": int(self.activation_patience_counter),
            "activation_agent_step": int(self.activation_agent_step),
            "stage_start_agent_step": int(self.stage_start_agent_step),
            "xy_curriculum_progress": float(self.xy_curriculum_progress()),
            "xy_curriculum_ramp_steps": int(self.xy_curriculum_ramp_steps),
            "agent_steps": int(self.agent_steps),
            "epoch_num": int(self.epoch_num),
            "best_rewards": float(self.best_rewards),
            "best_angular_velocity": float(self.best_angular_velocity),
            "master_lr": float(self.master_lr),
            "follower_lr": float(self.follower_lr),
            "joint_finetune_enable": bool(self.joint_finetune_enable),
        }
        if self.master_reference is not None:
            payload["master_reference"] = self.master_reference.state_dict()
        torch.save(payload, f"{name}.pth")

    def restore_train(self, fn: str) -> None:
        """Resume a complete hierarchical run, curriculum state included."""
        if not fn:
            return
        checkpoint = torch.load(fn, map_location=self.device)
        if checkpoint.get("format") != CHECKPOINT_FORMAT:
            raise RuntimeError(
                f"'{fn}' is not a HierarchicalPPO checkpoint (format="
                f"{checkpoint.get('format')!r}). To warm-start only the master from a "
                "21-D Stage-1 teacher checkpoint use --master_checkpoint instead of "
                "--checkpoint."
            )
        required = (
            "master",
            "follower",
            "master_optimizer",
            "follower_optimizer",
            "master_running_mean_std",
            "follower_running_mean_std",
            "current_stage",
            "agent_steps",
            "epoch_num",
        )
        missing = [key for key in required if key not in checkpoint]
        if missing:
            raise RuntimeError(
                f"Strict hierarchical resume failed: missing keys {missing} in {fn}"
            )

        validate_teacher_tactile_checkpoint_compatibility(
            checkpoint["master"], self.master.state_dict(), checkpoint_path=str(fn)
        )
        self.master.load_state_dict(checkpoint["master"], strict=True)
        self.follower.load_state_dict(checkpoint["follower"], strict=True)
        self.master_running_mean_std.load_state_dict(checkpoint["master_running_mean_std"])
        self.follower_running_mean_std.load_state_dict(
            checkpoint["follower_running_mean_std"]
        )
        if "master_value_mean_std" in checkpoint:
            self.master_value_mean_std.load_state_dict(checkpoint["master_value_mean_std"])
        if "follower_value_mean_std" in checkpoint:
            self.follower_value_mean_std.load_state_dict(
                checkpoint["follower_value_mean_std"]
            )

        self.current_stage = int(checkpoint["current_stage"])
        self.activation_speed_ema = float(checkpoint.get("activation_speed_ema", 0.0))
        self._activation_speed_ema_initialized = bool(
            checkpoint.get("activation_speed_ema_initialized", False)
        )
        self.activation_patience_counter = int(
            checkpoint.get("activation_patience_counter", 0)
        )
        self.activation_agent_step = int(checkpoint.get("activation_agent_step", -1))
        self.stage_start_agent_step = int(checkpoint.get("stage_start_agent_step", 0))
        self.agent_steps = int(checkpoint["agent_steps"])
        self.epoch_num = int(checkpoint["epoch_num"])
        self.best_rewards = float(checkpoint.get("best_rewards", -10000.0))
        self.best_angular_velocity = float(
            checkpoint.get("best_angular_velocity", -10000.0)
        )
        self.master_lr = float(checkpoint.get("master_lr", self.master_lr))
        self.follower_lr = float(checkpoint.get("follower_lr", self.follower_lr))

        if "master_reference" in checkpoint:
            self.master_reference = copy.deepcopy(self.master).to(self.device)
            self.master_reference.load_state_dict(checkpoint["master_reference"], strict=True)
            self.master_reference.requires_grad_(False)
            self.master_reference.eval()
        elif self.current_stage >= STAGE_FOLLOWER and self.joint_finetune_enable:
            self.master_reference = copy.deepcopy(self.master).to(self.device)
            self.master_reference.requires_grad_(False)
            self.master_reference.eval()

        self._apply_stage_freeze()
        self.master_optimizer.load_state_dict(checkpoint["master_optimizer"])
        self.follower_optimizer.load_state_dict(checkpoint["follower_optimizer"])
        self._set_master_lr(self.master_lr)
        self._set_follower_lr(self.follower_lr)
        if hasattr(self.env, "common_step_counter"):
            self.env.common_step_counter = self.agent_steps // self.num_actors
        print(
            "[INFO] Restored hierarchical state: "
            f"stage={STAGE_NAMES[self.current_stage]}, agent_steps={self.agent_steps}, "
            f"epoch={self.epoch_num}, speed_ema={self.activation_speed_ema:.4f}, "
            f"patience={self.activation_patience_counter}, "
            f"stage_start_agent_step={self.stage_start_agent_step}, "
            f"xy_progress={self.xy_curriculum_progress():.4f}",
            flush=True,
        )

    def restore_master_checkpoint(self, fn: str) -> None:
        """Strictly warm-start the master from an existing 21-D Stage-1 teacher.

        Only the master weights and its normalization state are consumed. The
        optimizer, epoch counters and curriculum state are deliberately NOT
        taken from the Stage-1 run: this is a warm start, not a resume.
        """
        if not fn:
            return
        checkpoint = torch.load(fn, map_location=self.device)
        if checkpoint.get("format") == CHECKPOINT_FORMAT:
            raise RuntimeError(
                f"'{fn}' is a full hierarchical checkpoint. Pass it to --checkpoint "
                "for a complete resume instead of --master_checkpoint."
            )
        if "model" not in checkpoint:
            raise RuntimeError(
                f"'{fn}' is not a Stage-1 PPO teacher checkpoint (no 'model' key)."
            )
        validate_teacher_tactile_checkpoint_compatibility(
            checkpoint["model"], self.master.state_dict(), checkpoint_path=str(fn)
        )
        self.master.load_state_dict(checkpoint["model"], strict=True)
        for key, module in (
            ("running_mean_std", self.master_running_mean_std),
            ("value_mean_std", self.master_value_mean_std),
        ):
            if key not in checkpoint:
                raise RuntimeError(
                    f"Stage-1 teacher checkpoint '{fn}' is missing '{key}'; the master "
                    "normalization state is required for a strict warm start."
                )
            module.load_state_dict(checkpoint[key])
        print(
            f"[INFO] Warm-started the {self.master_action_dim}-D master from '{fn}' "
            "(weights + normalization only). Curriculum starts at Stage 0; the "
            "follower activates once the speed EMA crosses "
            f"{self.activation_speed_threshold:.2f} rad/s.",
            flush=True,
        )

    def restore_test(self, fn: str) -> None:
        """Load a hierarchical checkpoint for evaluation."""
        checkpoint = torch.load(fn, map_location=self.device)
        if checkpoint.get("format") != CHECKPOINT_FORMAT:
            raise RuntimeError(f"'{fn}' is not a HierarchicalPPO checkpoint.")
        self.master.load_state_dict(checkpoint["master"], strict=True)
        self.follower.load_state_dict(checkpoint["follower"], strict=True)
        self.master_running_mean_std.load_state_dict(checkpoint["master_running_mean_std"])
        self.follower_running_mean_std.load_state_dict(
            checkpoint["follower_running_mean_std"]
        )
        self.current_stage = int(checkpoint.get("current_stage", STAGE_FOLLOWER))
        self.stage_start_agent_step = int(checkpoint.get("stage_start_agent_step", 0))
        self.agent_steps = int(checkpoint.get("agent_steps", 0))

    @torch.no_grad()
    def test(self) -> None:
        """Roll out the deterministic hierarchical policy indefinitely."""
        self.set_eval()
        self._push_xy_curriculum()
        obs_dict = self.env.reset()
        while True:
            processed = (
                self.master_running_mean_std(obs_dict["obs"])
                if self.normalize_input
                else obs_dict["obs"]
            )
            # One master forward produces both the action and the latent, so the
            # tactile encoder runs exactly once per control cycle here too.
            mu, latent = self.master.act_inference_with_latent(
                {
                    "obs": processed,
                    "priv_info": obs_dict["priv_info"],
                    "tactile_hist": obs_dict["tactile_hist"],
                }
            )
            executed_hand_action = torch.clamp(mu, -1.0, 1.0)
            latent = validate_tactile_latent(latent).detach()
            follower_obs = follower_obs_from_env(
                obs_dict,
                executed_hand_action=executed_hand_action,
                tactile_latent=latent,
            )
            if self.follower_active:
                follower_input = (
                    self.follower_running_mean_std(follower_obs)
                    if self.follower_normalize_input
                    else follower_obs
                )
                executed_xy = torch.clamp(
                    self.follower.act_inference(follower_input), -1.0, 1.0
                )
            else:
                executed_xy = torch.zeros(
                    (follower_obs.shape[0], self.follower_action_dim), device=self.device
                )
            obs_dict, _r, _done, _info = self.env.step(
                torch.cat([executed_hand_action, executed_xy], dim=-1)
            )

    # ------------------------------------------------------------------
    # main loop
    # ------------------------------------------------------------------

    def train(self) -> None:
        start_time = time.time()
        last_time = time.time()
        self.obs = self.env.reset()
        if self.agent_steps == 0:
            self.agent_steps = self.batch_size
        total_iters = max(1, math.ceil(self.max_agent_steps / self.batch_size))

        while self.agent_steps < self.max_agent_steps:
            self.epoch_num += 1
            iter_start = time.time()
            workspace, action_scale = self._push_xy_curriculum()
            stats = self.train_epoch()
            self.storage.data_dict = None
            self._update_curriculum(stats["rollout_speed"])

            self.write_stats(stats, workspace, action_scale)
            mean_rewards = self.episode_rewards.get_mean()
            mean_raw_rewards = self.episode_raw_rewards.get_mean()
            mean_lengths = self.episode_lengths.get_mean()
            self.writer.add_scalar("episode_rewards/step", mean_rewards, self.agent_steps)
            self.writer.add_scalar(
                "episode_rewards_raw/step", mean_raw_rewards, self.agent_steps
            )
            self.writer.add_scalar("episode_lengths/step", mean_lengths, self.agent_steps)

            all_fps = self.agent_steps / max(time.time() - start_time, 1.0e-6)
            last_fps = self.batch_size / max(time.time() - last_time, 1.0e-6)
            last_time = time.time()
            tprint(
                f"Agent Steps: {int(self.agent_steps // 1e6):04}M | FPS: {all_fps:.1f} | "
                f"Last FPS: {last_fps:.1f} | Stage: {STAGE_NAMES[self.current_stage]} | "
                f"SpeedEMA: {self.activation_speed_ema:.3f} | "
                f"Best reward: {self.best_rewards:.2f} | Best speed: "
                f"{self.best_angular_velocity:.3f}"
            )
            print("", flush=True)
            self._print_epoch_log(
                total_iters=total_iters,
                stats=stats,
                iter_time=time.time() - iter_start,
                elapsed=time.time() - start_time,
                mean_rewards=mean_rewards,
                mean_lengths=mean_lengths,
                workspace=workspace,
                action_scale=action_scale,
            )

            if self.save_freq > 0 and self.epoch_num % self.save_freq == 0:
                self.save(
                    os.path.join(
                        self.nn_dir,
                        f"ep_{self.epoch_num}_step_{int(self.agent_steps // 1e6):04}M"
                        f"_reward_{mean_rewards:.2f}",
                    )
                )
            self.save(os.path.join(self.nn_dir, "last"))

            if mean_rewards > self.best_rewards and self.epoch_num >= self.save_best_after:
                print(f"save current best reward: {mean_rewards:.2f}")
                self.best_rewards = mean_rewards
                self.save(os.path.join(self.nn_dir, "best_reward"))
            # best_speed tracks the SMOOTHED signed angular velocity, never the
            # episode reward, so the speed-ceiling experiment is judged on speed.
            if (
                self.activation_speed_ema > self.best_angular_velocity
                and self.epoch_num >= self.save_best_after
            ):
                print(f"save current best speed: {self.activation_speed_ema:.4f} rad/s")
                self.best_angular_velocity = self.activation_speed_ema
                self.save(os.path.join(self.nn_dir, "best_speed"))

        print("max steps achieved")

    def _print_epoch_log(
        self,
        total_iters,
        stats,
        iter_time,
        elapsed,
        mean_rewards,
        mean_lengths,
        workspace,
        action_scale,
    ) -> None:
        width = 100
        pad = 34
        fps = int(self.batch_size / max(1.0e-6, stats["collect_time"] + stats["learn_time"]))
        eta = max(0.0, (total_iters - self.epoch_num) * (elapsed / max(1, self.epoch_num)))
        lines = [
            "#" * width,
            f" Hierarchical iteration {self.epoch_num}/{total_iters} ".center(width, " "),
            "",
            f"{'Computation:':>{pad}} {fps} steps/s "
            f"(collection: {stats['collect_time']:.3f}s, learning: {stats['learn_time']:.3f}s)",
            f"{'Stage:':>{pad}} {STAGE_NAMES[self.current_stage]}",
            f"{'Master frozen:':>{pad}} {self.master_frozen}",
            f"{'Follower active:':>{pad}} {self.follower_active}",
            f"{'Rollout signed mean omega:':>{pad}} {stats['rollout_speed']:.4f} rad/s",
            f"{'Activation speed EMA:':>{pad}} {self.activation_speed_ema:.4f} rad/s",
            f"{'Activation patience:':>{pad}} "
            f"{self.activation_patience_counter}/{self.activation_patience}",
            f"{'XY workspace / action scale:':>{pad}} "
            f"{workspace:.4f} m / {action_scale:.4f} m",
            f"{'Mean reward:':>{pad}} {mean_rewards:.4f}",
            f"{'Mean episode length:':>{pad}} {mean_lengths:.4f}",
        ]
        for key in sorted(self.extra_info):
            if not key.startswith(("xy/", "screw/", "curriculum/")):
                continue
            value = self.extra_info[key]
            if isinstance(value, torch.Tensor):
                value = value.item()
            if isinstance(value, (int, float)):
                lines.append(f"{key + ':':>{pad}} {float(value):.6f}")
        lines.extend(
            [
                "-" * width,
                f"{'Total timesteps:':>{pad}} {self.agent_steps}",
                f"{'Iteration time:':>{pad}} {iter_time:.2f}s",
                f"{'Time elapsed:':>{pad}} {time.strftime('%H:%M:%S', time.gmtime(elapsed))}",
                f"{'ETA:':>{pad}} {time.strftime('%H:%M:%S', time.gmtime(eta))}",
            ]
        )
        print("\n".join(lines))
