# Copyright (c) 2026, BrainCo.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Environment configurations for the TacRes two-phase training pipeline.

Phase 1 trains a proprio-only base policy (no fingertip contact forces in actor or critic).
Phase 2 adds the force-impulse perturbation curriculum, shaped tactile observations for the
residual/gate networks, and privileged observations for the newly initialized critic.
"""

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from ... import mdp
from . import tacres_mdp
from .dexsuite_revo3_env_cfg_grasp import DexsuiteRevo3LiftEnvCfg, DexsuiteRevo3LiftEnvCfg_PLAY
from .tactile import TACTILE_DIP_BODIES, TACTILE_FORCE_SENSOR_NAMES

_TACTILE_SENSOR_NAMES = list(TACTILE_FORCE_SENSOR_NAMES)
_TACTILE_BODY_NAMES = list(TACTILE_DIP_BODIES)

# Common params shared by the tactile feature terms.
_TACTILE_TERM_PARAMS = {
    "contact_sensor_names": _TACTILE_SENSOR_NAMES,
    "body_names": _TACTILE_BODY_NAMES,
    "ema_alpha": 0.8,
    "contact_threshold": 1.0,
}


##
# Phase 1: proprio-only base policy
##


@configclass
class DexsuiteRevo3LiftTacResPhase1EnvCfg(DexsuiteRevo3LiftEnvCfg):
    """Lift task without any contact-force observation (TacRes base-policy training).

    The fingertip contact sensors stay in the scene (rewards use them), but the 5-finger 3D
    contact-force term is removed from the ``proprio`` group, so neither actor nor critic sees it.
    No perturbation curriculum is enabled; the regular domain randomization is kept.
    """

    def __post_init__(self):
        super().__post_init__()
        self.observations.proprio.contact = None


@configclass
class DexsuiteRevo3LiftTacResPhase1EnvCfg_PLAY(DexsuiteRevo3LiftEnvCfg_PLAY):
    """Evaluation variant of the TacRes phase-1 environment."""

    def __post_init__(self):
        super().__post_init__()
        self.observations.proprio.contact = None


##
# Phase 2: perturbation curriculum + tactile observations
##


@configclass
class TacResTactileObsCfg(ObsGroup):
    """Shaped fingertip tactile features (tau_hist input of the residual network, K=10 history)."""

    features = ObsTerm(func=tacres_mdp.tactile_finger_features, params=dict(_TACTILE_TERM_PARAMS))

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True
        self.history_length = 10
        self.flatten_history_dim = True


@configclass
class TacResEventObsCfg(ObsGroup):
    """Contact-event features e_t (gate network input)."""

    event = ObsTerm(func=tacres_mdp.tactile_event_features, params=dict(_TACTILE_TERM_PARAMS))

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class TacResPrivilegedObsCfg(ObsGroup):
    """Privileged critic observations: current tactile frame, perturbation state, true object state."""

    contact_forces = ObsTerm(
        func=mdp.fingers_contact_force_b,
        params={"contact_sensor_names": _TACTILE_SENSOR_NAMES},
        clip=(-20.0, 20.0),
    )
    perturbation = ObsTerm(func=tacres_mdp.perturbation_state)
    object_state = ObsTerm(func=tacres_mdp.object_root_state_b, clip=(-10.0, 10.0))

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class TacResGateLabelObsCfg(ObsGroup):
    """Training-only warm-start gate labels (consumed by TacResPPO, not by any network input)."""

    label = ObsTerm(func=tacres_mdp.perturbation_gate_label, params={"label_window_s": 0.3})

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


def _configure_tacres_phase2(cfg) -> None:
    """Attach phase-2 observation groups, perturbation events, curriculum, and reward shaping."""
    # The frozen base policy is competent at full difficulty but cannot adapt to lighter-gravity
    # levels, so ADR demotion triggers an unrecoverable downward spiral (observed empirically).
    # Phase 2 therefore starts at max difficulty and never demotes.
    cfg.curriculum.adr.params["init_difficulty"] = 10
    cfg.curriculum.adr.params["promotion_only"] = True
    # observation groups for residual / gate / critic / warm-start labels
    cfg.observations.tacres_tactile = TacResTactileObsCfg()
    cfg.observations.tacres_event = TacResEventObsCfg()
    cfg.observations.tacres_privileged = TacResPrivilegedObsCfg()
    cfg.observations.tacres_gate_label = TacResGateLabelObsCfg()

    # per-episode force-impulse schedule (sampled on reset) + per-step application
    cfg.events.tacres_perturbation = EventTerm(
        func=tacres_mdp.ObjectForcePerturbation,
        mode="reset",
        params={
            "probability": 0.7,
            "max_impulses": 2,
            "magnitude_range": (2.0, 2.0),
            "duration_range": (0.1, 0.2),
            "start_time_range": (0.3, 3.4),
            "object_cfg": SceneEntityCfg("object"),
        },
    )
    cfg.events.tacres_perturbation_apply = EventTerm(
        func=tacres_mdp.apply_object_force_perturbation,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        is_global_time=True,
        params={
            "perturbation_term": "tacres_perturbation",
            "object_cfg": SceneEntityCfg("object"),
        },
    )

    # impulse magnitude curriculum: 2N -> up to 8N, driven by the existing ADR difficulty term
    cfg.curriculum.tacres_impulse_magnitude_adr = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "events.tacres_perturbation.params.magnitude_range",
            "modify_fn": mdp.initial_final_interpolate_fn,
            "modify_params": {
                "initial_value": (2.0, 2.0),
                "final_value": (2.0, 8.0),
                "difficulty_term_str": "adr",
            },
        },
    )

    # reward shaping from the proposal: r_drop and r_force
    cfg.rewards.drop_penalty = RewTerm(
        func=mdp.is_terminated_term,
        weight=-10.0,
        params={"term_keys": "object_out_of_bound"},
    )
    cfg.rewards.fingertip_force_penalty = RewTerm(
        func=tacres_mdp.fingertip_force_excess,
        weight=-0.01,
        params={"contact_sensor_names": _TACTILE_SENSOR_NAMES, "threshold": 20.0},
    )


@configclass
class DexsuiteRevo3LiftTacResPhase2EnvCfg(DexsuiteRevo3LiftTacResPhase1EnvCfg):
    """Phase-2 environment: proprio-only base obs plus tactile/privileged groups and perturbations."""

    def __post_init__(self):
        super().__post_init__()
        _configure_tacres_phase2(self)


@configclass
class DexsuiteRevo3LiftTacResPhase2EnvCfg_PLAY(DexsuiteRevo3LiftTacResPhase1EnvCfg_PLAY):
    """Evaluation variant of the TacRes phase-2 environment."""

    def __post_init__(self):
        super().__post_init__()
        _configure_tacres_phase2(self)


##
# Phase-2 variants: sustained perturbation families + windowed event features
##


def _use_windowed_event_features(cfg, window: int = 12) -> None:
    """Swap the per-step event features for the windowed contrast detector + object kinematics."""
    params = dict(_TACTILE_TERM_PARAMS)
    params.pop("ema_alpha", None)
    params["window"] = window
    cfg.observations.tacres_event.event = ObsTerm(
        func=tacres_mdp.tactile_event_features_windowed, params=params
    )
    cfg.observations.tacres_event.object_kinematics = ObsTerm(func=tacres_mdp.object_event_features)


@configclass
class DexsuiteRevo3LiftTacResPhase2PullEnvCfg(DexsuiteRevo3LiftTacResPhase2EnvCfg):
    """Sustained-pull family: 0.8-1.6 s external forces on the object instead of short impulses."""

    def __post_init__(self):
        super().__post_init__()
        _use_windowed_event_features(self)
        # one sustained pull per perturbed episode; fixed magnitude range (no ADR coupling)
        self.events.tacres_perturbation.params.update(
            {
                "max_impulses": 1,
                "magnitude_range": (3.0, 6.0),
                "duration_range": (0.8, 1.6),
                "start_time_range": (0.3, 2.2),
            }
        )
        self.curriculum.tacres_impulse_magnitude_adr = None


@configclass
class DexsuiteRevo3LiftTacResPhase2FrictionEnvCfg(DexsuiteRevo3LiftTacResPhase2EnvCfg):
    """Friction-drop family: object friction suddenly scaled down mid-episode (slip induction)."""

    def __post_init__(self):
        super().__post_init__()
        _use_windowed_event_features(self)
        self.events.tacres_perturbation = EventTerm(
            func=tacres_mdp.ObjectFrictionDrop,
            mode="reset",
            params={
                "probability": 0.7,
                "magnitude_range": (0.3, 0.5),
                "start_time_range": (0.5, 2.5),
                "object_cfg": SceneEntityCfg("object"),
            },
        )
        self.events.tacres_perturbation_apply = EventTerm(
            func=tacres_mdp.apply_object_friction_drop,
            mode="interval",
            interval_range_s=(0.0, 0.0),
            is_global_time=True,
            params={
                "perturbation_term": "tacres_perturbation",
                "object_cfg": SceneEntityCfg("object"),
            },
        )
        self.curriculum.tacres_impulse_magnitude_adr = None
