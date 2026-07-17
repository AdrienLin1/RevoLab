# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class DexsuiteRevo3PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner configuration for Revo3 dexsuite tasks."""

    num_steps_per_env = 32
    # rsl-rl >= 5.0 selects the actor network's input from the "actor" observation set. Without an
    # explicit "actor" key it silently falls back to just the group named "policy" (small proprio
    # vector), which mismatches checkpoints trained with the full observation. Name the set "actor"
    # so the actor consumes policy+proprio+perception (matches the pretrained Lift checkpoint).
    obs_groups = {"actor": ["policy", "proprio", "perception"], "critic": ["policy", "proprio", "perception"]}
    max_iterations = 15000
    save_interval = 250
    experiment_name = "dexsuite_tianji"

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class DexsuiteUr5eRevo3PPORunnerCfg(DexsuiteRevo3PPORunnerCfg):
    """PPO runner configuration for UR5e + Revo3 dexsuite tasks."""

    experiment_name = "dexsuite_ur5e"
