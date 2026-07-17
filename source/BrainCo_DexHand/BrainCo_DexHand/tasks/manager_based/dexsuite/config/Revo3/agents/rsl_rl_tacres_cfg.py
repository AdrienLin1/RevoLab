# Copyright (c) 2026, BrainCo.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL runner configurations for the TacRes two-phase training pipeline."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from .rsl_rl_ppo_cfg import DexsuiteRevo3PPORunnerCfg


@configclass
class TacResPhase1PPORunnerCfg(DexsuiteRevo3PPORunnerCfg):
    """Phase 1: train the proprio-only base policy (same PPO setup as the original Lift task).

    The environment removes the contact-force observation, so actor and critic are both
    proprioception-only. The resulting checkpoint is loaded as the frozen base policy in phase 2.
    """

    experiment_name = "tacres_phase1"


@configclass
class TacResActorCfg(RslRlMLPModelCfg):
    """Configuration of :class:`BrainCo_DexHand.algo.tacres.TacResActor`.

    ``hidden_dims``/``activation``/``obs_normalization`` configure the residual trunk; the
    ``base_*`` fields must exactly match the phase-1 actor so its checkpoint loads strictly.
    """

    class_name: str = "BrainCo_DexHand.algo.tacres:TacResActor"

    # residual network pi_res (proposal: MLP[512, 256])
    hidden_dims: list[int] = [512, 256]
    activation: str = "elu"
    obs_normalization: bool = True
    # exploration distribution over the composed mean: learnable log-std, init log(std) = -1.0
    distribution_cfg = RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.3679, std_type="log")

    # observation sub-sets consumed by the three actor components
    base_obs_groups: list[str] = ["policy", "proprio", "perception"]
    residual_obs_groups: list[str] = ["policy", "proprio", "tacres_tactile"]
    gate_obs_groups: list[str] = ["tacres_event"]

    # frozen base policy architecture (must match the phase-1 actor / checkpoint exactly)
    base_hidden_dims: list[int] = [512, 256, 128]
    base_activation: str = "elu"
    base_obs_normalization: bool = True
    base_distribution_cfg = RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0, std_type="scalar")

    # gate network g (proposal: MLP[64, 32])
    gate_hidden_dims: list[int] = [64, 32]

    # residual scale alpha in the normalized action space
    alpha: float = 0.1

    # B2-style ablation: set to 1.0 for an always-on residual (gate network bypassed);
    # negative (default) uses the learned gate
    fixed_gate: float = -1.0


@configclass
class TacResPPOAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """PPO algorithm configuration with the TacRes auxiliary-loss schedule."""

    class_name: str = "BrainCo_DexHand.algo.tacres:TacResPPO"

    # phase-1 checkpoint that initializes the frozen base policy; must be provided at launch, e.g.
    #   train.py --task ...TacRes-Phase2-v0 agent.algorithm.base_checkpoint=/path/to/model_100.pt
    base_checkpoint: str = ""

    # observation group holding the training-only warm-start labels
    gate_label_group: str = "tacres_gate_label"
    # L_gate-warm: BCE(g, y) weight annealed 1 -> `gate_warm_floor` over the first `gate_warm_frac`
    # of training (a small floor guards against gate/residual mutual collapse)
    gate_warm_coef: float = 1.0
    gate_warm_frac: float = 0.3
    gate_warm_floor: float = 0.0
    # lambda_g: gate sparsity weight ramped 0 -> `gate_sparsity_coef` after the warm-start anneal
    gate_sparsity_coef: float = 0.01
    gate_sparsity_ramp_frac: float = 0.3
    # lambda_r: residual magnitude regularization
    residual_l2_coef: float = 1.0e-3
    # annealing horizon; defaults to the runner's max_iterations when unset
    schedule_total_iterations: int | None = None


@configclass
class TacResPhase2PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Phase 2: frozen base + residual/gate training under the perturbation curriculum."""

    num_steps_per_env = 24
    max_iterations = 15000
    save_interval = 250
    experiment_name = "tacres_phase2"

    # The "actor" set is the union of groups consumed by the TacResActor components (base/residual/
    # gate sub-sets are configured on the actor cfg). The critic is privileged and freshly initialized.
    obs_groups = {
        "actor": ["policy", "proprio", "perception", "tacres_tactile", "tacres_event"],
        "critic": ["policy", "proprio", "perception", "tacres_privileged"],
    }

    actor = TacResActorCfg()

    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=None,
    )

    algorithm = TacResPPOAlgorithmCfg(
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
