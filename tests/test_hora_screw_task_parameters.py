import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_CFG_PATH = (
    REPO_ROOT
    / "source/BrainCo_DexHand/BrainCo_DexHand/tasks/direct/hora_screw/revo3_hand_screw_env_cfg.py"
)
ENV_PATH = ENV_CFG_PATH.with_name("revo3_hand_screw_env.py")
TACTILE_CFG_PATH = (
    REPO_ROOT
    / "source/BrainCo_DexHand/BrainCo_DexHand/tasks/direct/hora_screw/revo3_hand_screw_tactile_env_cfg.py"
)
TACTILE_ENV_PATH = (
    REPO_ROOT
    / "source/BrainCo_DexHand/BrainCo_DexHand/tasks/direct/hora_screw/revo3_hand_screw_tactile_env.py"
)
TACTILE_AGENT_DIR = TACTILE_ENV_PATH.parent / "agents"


def _literal_self_assignments(path: Path, class_name: str) -> dict[str, object]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    class_node = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    assignments: dict[str, object] = {}
    for node in ast.walk(class_node):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not (
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "self"
        ):
            continue
        try:
            assignments[target.attr] = ast.literal_eval(node.value)
        except (ValueError, TypeError):
            continue
    return assignments


def _literal_class_assignments(path: Path, class_name: str) -> dict[str, object]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    class_node = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    assignments: dict[str, object] = {}
    for node in class_node.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name):
            try:
                assignments[target.id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                continue
    return assignments


def _load_method(path: Path, class_name: str, method_name: str):
    """Load one class method from source without importing its runtime dependencies.

    Args:
        path: Python source file containing the class.
        class_name: Name of the class that owns the method.
        method_name: Name of the method to compile.

    Returns:
        The compiled unbound method.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    class_node = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method_node = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )
    module = ast.fix_missing_locations(ast.Module(body=[method_node], type_ignores=[]))
    def quat_apply(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
        xyz = quaternion[..., 1:]
        cross = torch.cross(xyz, vector, dim=-1)
        return vector + 2.0 * (
            quaternion[..., :1] * cross + torch.cross(xyz, cross, dim=-1)
        )

    namespace = {"torch": torch, "quat_apply": quat_apply}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[method_name]


def test_screwdriver_keeps_physical_resistance_and_aligns_torque_penalty():
    base = _literal_class_assignments(ENV_CFG_PATH, "Revo3HandScrewEnvCfg")
    driver = _literal_self_assignments(ENV_CFG_PATH, "Revo3HandScrewDriverEnvCfg")

    assert base["object_joint_friction_default"] == 0.2
    assert "object_joint_friction_default" not in driver
    assert driver["torque_penalty_scale"] == base["torque_penalty_scale"] == -0.1


def test_object_radius_override_changes_only_the_selected_environment():
    """Preserve random scales outside the explicitly selected analysis environment."""
    initialize = _load_method(
        ENV_PATH,
        "Revo3HandScrewEnv",
        "_initialize_object_radius_randomization",
    )
    common_cfg = {
        "object_radius_scale_levels": (0.8, 0.9, 1.0, 1.1, 1.2),
        "randomize_object_radius": True,
        "print_object_radius_scale_ids": False,
    }
    baseline = SimpleNamespace(
        cfg=SimpleNamespace(
            **common_cfg,
            object_radius_scale_override=0.0,
            object_radius_scale_override_env_index=-1,
        ),
        scene=SimpleNamespace(env_prim_paths=[f"env_{index}" for index in range(7)]),
        _apply_object_radius_scale_to_env=lambda *_: None,
    )
    selected = SimpleNamespace(
        cfg=SimpleNamespace(
            **common_cfg,
            object_radius_scale_override=1.2,
            object_radius_scale_override_env_index=6,
        ),
        scene=SimpleNamespace(env_prim_paths=[f"env_{index}" for index in range(7)]),
        _apply_object_radius_scale_to_env=lambda *_: None,
    )

    torch.manual_seed(42)
    initialize(baseline)
    torch.manual_seed(42)
    initialize(selected)

    torch.testing.assert_close(selected.object_radius_scales[:6], baseline.object_radius_scales[:6])
    assert selected.object_radius_scales[6].item() == pytest.approx(1.2)
    assert baseline.object_radius_scales[6].item() != pytest.approx(1.2)


@pytest.mark.parametrize(
    "config_name",
    (
        "Revo3HandScrewTactile.yaml",
        "Revo3HandScrewTactileGRU.yaml",
        "Revo3HandScrewTactileGRUSmoke.yaml",
        "Revo3HandScrewTactileSmoke.yaml",
    ),
)
def test_tactile_agents_use_standard_single_return_ppo(config_name):
    """Keep coordination inside the environment reward, not a second PPO loss."""
    config_text = (TACTILE_AGENT_DIR / config_name).read_text(encoding="utf-8")

    assert "separate_coord_advantage: False" in config_text
    assert "coord_advantage_coef_" not in config_text
    assert "coord_value_loss_coef" not in config_text
    assert "tactile_layout: estimated_official" in config_text


def test_valve_reward_parameters_match_five_finger_task():
    base = _literal_class_assignments(ENV_CFG_PATH, "Revo3HandScrewEnvCfg")
    valve = _literal_self_assignments(ENV_CFG_PATH, "Revo3HandVavleDriverEnvCfg")
    tactile = _literal_class_assignments(
        TACTILE_CFG_PATH, "Revo3HandVavleDriverTactileEnvCfg"
    )

    weights = tactile["tactile_visible_contact_finger_weights"]
    assert weights == (1.0, 1.0, 1.0, 1.0, 1.0)
    assert "visible_contact_reward_scale" not in tactile
    assert valve["pose_diff_penalty_scale"] == 0.0
    assert valve["torque_penalty_scale"] == base["torque_penalty_scale"] == -0.1


def test_screwdriver_visible_contact_weights_cover_each_active_finger():
    tactile = _literal_class_assignments(
        TACTILE_CFG_PATH, "Revo3HandScrewDriverTactileEnvCfg"
    )

    weights = tactile["tactile_visible_contact_finger_weights"]
    assert all(weight > 0.0 for weight in weights[:4])
    assert weights[4] == 0.0


def test_tactile_reward_scales_prioritize_visible_contact_and_coordination():
    """Guide coordination early, then retain only a weak late tie-breaker."""
    tactile = _literal_class_assignments(
        TACTILE_CFG_PATH, "Revo3HandScrewTactileMixinCfg"
    )

    assert tactile["visible_contact_target_ratio_initial"] == pytest.approx(0.01)
    assert tactile["visible_contact_target_ratio_mid"] == pytest.approx(0.03)
    assert tactile["visible_contact_target_ratio_final"] == pytest.approx(0.05)
    assert tactile["visible_contact_adaptive_scale_max"] == 300.0
    assert tactile["visible_contact_reward_clip_ratio"] == pytest.approx(0.10)
    assert tactile["coord_intrinsic_weight_initial"] == pytest.approx(0.10)
    assert tactile["coord_intrinsic_weight_peak"] == pytest.approx(0.30)
    assert tactile["coord_intrinsic_weight_final"] == pytest.approx(0.18)
    assert tactile["coord_intrinsic_warmup_end"] == 5_000_000
    assert tactile["coord_intrinsic_decay_start"] == 60_000_000
    assert tactile["coord_intrinsic_decay_end"] == 90_000_000
    assert tactile["coord_load_window"] == 16
    assert tactile["coord_force_comfort_ref"] == pytest.approx(2.5)
    assert tactile["coord_q_floor_initial"] == pytest.approx(0.20)
    assert tactile["coord_q_floor_final"] == pytest.approx(0.0)
    assert tactile["coord_presence_floor_initial"] == pytest.approx(1.0)
    assert tactile["coord_presence_floor_final"] == pytest.approx(0.0)
    assert tactile["coord_effective_torque_guide_ref"] == pytest.approx(0.005)
    assert tactile["coord_effective_torque_guide_power"] == pytest.approx(0.5)
    assert tactile["coord_effective_torque_reward_weight"] == pytest.approx(0.12)
    assert tactile["coord_presence_half_saturation"] == pytest.approx(0.001)
    assert tactile["coord_guide_curriculum_end"] == 15_000_000
    assert "coord_lambda_max" not in tactile
    assert "coord_dense_scale" not in tactile
    assert "coord_delta_clip" not in tactile
    assert "coord_gamma" not in tactile
    assert "coord_reward_adaptive_ema" not in tactile
    assert "coord_reward_target_final" not in tactile
    for class_name in (
        "Revo3HandScrewNutBoltTactileEnvCfg",
        "Revo3HandScrewDriverTactileEnvCfg",
        "Revo3HandVavleDriverTactileEnvCfg",
        "Revo3HandValveDriver40TactileEnvCfg",
    ):
        task = _literal_class_assignments(TACTILE_CFG_PATH, class_name)
        assert "visible_contact_target_ratio_initial" not in task
        assert "visible_contact_target_ratio_final" not in task


def test_capacity_weighted_coord_utility_prefers_load_sharing():
    """Prefer comfortable sharing without reducing any useful torque."""
    compute_utility = _load_method(
        TACTILE_ENV_PATH,
        "Revo3HandScrewTactileEnv",
        "_capacity_weighted_load_utility",
    )
    env = SimpleNamespace(
        cfg=SimpleNamespace(
            coord_load_saturation=0.5,
            coord_load_max=3.0,
        )
    )
    capacity = torch.ones((1, 3))
    capacity_sum = torch.tensor([3.0])
    single_finger = compute_utility(
        env, torch.tensor([[1.0, 0.0, 0.0]]), capacity, capacity_sum
    )
    shared = compute_utility(
        env, torch.full((1, 3), 1.0 / 3.0), capacity, capacity_sum
    )
    increased = compute_utility(
        env, torch.tensor([[1.0, 0.1, 0.0]]), capacity, capacity_sum
    )

    assert shared.item() > single_finger.item()
    assert increased.item() > single_finger.item()


def test_capacity_weighted_coord_utility_is_object_scale_invariant():
    """Keep normalized utility unchanged when radius and torque scale together."""
    compute_utility = _load_method(
        TACTILE_ENV_PATH,
        "Revo3HandScrewTactileEnv",
        "_capacity_weighted_load_utility",
    )
    env = SimpleNamespace(
        cfg=SimpleNamespace(
            coord_load_saturation=0.5,
            coord_load_max=3.0,
        )
    )
    normalized_load = torch.tensor([[0.4, 0.2, 0.0]])
    small_capacity = torch.tensor([[0.04, 0.04, 0.04]])
    large_capacity = 1.5 * small_capacity

    small = compute_utility(
        env, normalized_load, small_capacity, small_capacity.sum(dim=-1)
    )
    large = compute_utility(
        env, normalized_load, large_capacity, large_capacity.sum(dim=-1)
    )

    torch.testing.assert_close(small, large)


def test_coord_intrinsic_reward_is_signed_fixed_weight_and_bounded():
    """Apply one fixed rollout weight without policy-statistic normalization."""
    apply_reward = _load_method(
        TACTILE_ENV_PATH,
        "Revo3HandScrewTactileEnv",
        "_apply_coord_intrinsic_reward",
    )
    env = SimpleNamespace(
        _current_coord_intrinsic_weight=lambda: 0.3,
        extras={},
        device=torch.device("cpu"),
    )

    reward = apply_reward(
        env,
        torch.tensor([1.0, -0.5, 2.0, -2.0]),
        torch.full((4,), 7.0),
    )

    assert reward.tolist() == pytest.approx([0.3, -0.15, 0.3, -0.3])
    assert env.extras["tactile/coord_positive_reward_budget"].item() == pytest.approx(0.3)
    assert env.extras["tactile/coord_reward_budget_abs_ratio"].item() == pytest.approx(
        0.3 / 7.0
    )
    assert env.extras["tactile/coord_positive_ratio"].item() == pytest.approx(0.5)
    assert env.extras["tactile/coord_negative_ratio"].item() == pytest.approx(0.5)


def test_coord_intrinsic_weight_peaks_then_anneals():
    """Guide early/mid exploration without imposing late all-finger behavior."""
    current_weight = _load_method(
        TACTILE_ENV_PATH,
        "Revo3HandScrewTactileEnv",
        "_current_coord_intrinsic_weight",
    )
    env = SimpleNamespace(
        cfg=SimpleNamespace(
            coord_intrinsic_weight_initial=0.10,
            coord_intrinsic_weight_peak=0.30,
            coord_intrinsic_weight_final=0.18,
            coord_intrinsic_warmup_end=10,
            coord_intrinsic_decay_start=30,
            coord_intrinsic_decay_end=50,
        ),
        common_step_counter=0,
        num_envs=10,
    )

    assert current_weight(env) == pytest.approx(0.10)
    env.common_step_counter = 1
    assert current_weight(env) == pytest.approx(0.30)
    env.common_step_counter = 3
    assert current_weight(env) == pytest.approx(0.30)
    env.common_step_counter = 4
    assert current_weight(env) == pytest.approx(0.24)
    env.common_step_counter = 5
    assert current_weight(env) == pytest.approx(0.18)


def test_effective_torque_bonus_bypasses_quality_weight():
    """Keep the useful-torque floor fixed while only quality is annealed."""
    combine_score = _load_method(
        TACTILE_ENV_PATH,
        "Revo3HandScrewTactileEnv",
        "_combine_coord_intrinsic_score",
    )
    env = SimpleNamespace(_current_coord_intrinsic_weight=lambda: 0.18)

    score, quality_weight = combine_score(
        env,
        torch.tensor([0.12, 0.12, 0.12]),
        torch.tensor([0.0, 1.0, -1.0]),
    )

    torch.testing.assert_close(score, torch.tensor([0.12, 0.30, -0.06]))
    assert quality_weight == pytest.approx(0.18)


def test_coord_q_floor_anneals_by_global_agent_steps():
    """Decay early coordination guidance independently of policy statistics."""
    current_floor = _load_method(
        TACTILE_ENV_PATH,
        "Revo3HandScrewTactileEnv",
        "_current_coord_q_floor",
    )
    env = SimpleNamespace(
        cfg=SimpleNamespace(
            coord_guide_curriculum_start=0,
            coord_guide_curriculum_end=100,
            coord_q_floor_initial=0.2,
            coord_q_floor_final=0.0,
        ),
        common_step_counter=0,
        num_envs=10,
    )

    assert current_floor(env) == pytest.approx(0.2)
    env.common_step_counter = 5
    assert current_floor(env) == pytest.approx(0.1)
    env.common_step_counter = 10
    assert current_floor(env) == pytest.approx(0.0)


def test_coord_presence_floor_anneals_to_physical_gate():
    """Make weak useful torque dense early and restore the physical late gate."""
    current_floor = _load_method(
        TACTILE_ENV_PATH,
        "Revo3HandScrewTactileEnv",
        "_current_coord_presence_floor",
    )
    env = SimpleNamespace(
        cfg=SimpleNamespace(
            coord_guide_curriculum_start=0,
            coord_guide_curriculum_end=100,
            coord_presence_floor_initial=1.0,
            coord_presence_floor_final=0.0,
        ),
        common_step_counter=0,
        num_envs=10,
    )

    assert current_floor(env) == pytest.approx(1.0)
    env.common_step_counter = 5
    assert current_floor(env) == pytest.approx(0.5)
    env.common_step_counter = 10
    assert current_floor(env) == pytest.approx(0.0)


def test_coord_presence_gate_is_dense_early_and_smooth_late():
    """Avoid early suppression and retain a useful late marginal gradient."""
    presence_gate = _load_method(
        TACTILE_ENV_PATH,
        "Revo3HandScrewTactileEnv",
        "_coord_presence_gate",
    )
    env = SimpleNamespace(
        cfg=SimpleNamespace(coord_presence_half_saturation=0.001),
        _current_coord_presence_floor=lambda: 1.0,
    )
    normalized_torque = torch.tensor([0.0, 0.001, 0.005])

    raw_gate, early_gate, floor = presence_gate(env, normalized_torque)

    torch.testing.assert_close(raw_gate, torch.tensor([0.0, 0.5, 5.0 / 6.0]))
    torch.testing.assert_close(early_gate, torch.ones(3))
    assert floor == pytest.approx(1.0)

    env._current_coord_presence_floor = lambda: 0.0
    _, late_gate, floor = presence_gate(env, normalized_torque)
    torch.testing.assert_close(late_gate, raw_gate)
    assert floor == pytest.approx(0.0)


def test_coord_effective_torque_reward_weight_is_fixed():
    """Keep useful torque independent from the quality-weight schedule."""
    current_weight = _load_method(
        TACTILE_ENV_PATH,
        "Revo3HandScrewTactileEnv",
        "_coord_effective_torque_reward_weight",
    )
    env = SimpleNamespace(
        cfg=SimpleNamespace(coord_effective_torque_reward_weight=0.12),
    )

    assert current_weight(env) == pytest.approx(0.12)


def test_coord_effective_torque_guide_is_bounded_and_capacity_normalized():
    """Reward only efficient torque with a fixed, object-relative scale."""
    compute_guide = _load_method(
        TACTILE_ENV_PATH,
        "Revo3HandScrewTactileEnv",
        "_coord_effective_torque_guide",
    )
    env = SimpleNamespace(
        cfg=SimpleNamespace(
            coord_effective_torque_guide_ref=0.1,
            coord_effective_torque_guide_power=0.5,
        ),
        _coord_effective_torque_reward_weight=lambda: 0.25,
    )
    efficient_torque = torch.tensor(
        [[0.0, 0.0], [0.025, 0.0], [0.05, 0.05]]
    )
    capacity_sum = torch.ones(3)

    normalized, guide, weight, bonus = compute_guide(
        env, efficient_torque, capacity_sum
    )

    torch.testing.assert_close(normalized, torch.tensor([0.0, 0.025, 0.1]))
    torch.testing.assert_close(guide, torch.tensor([0.0, 0.5, 1.0]))
    assert weight == pytest.approx(0.25)
    torch.testing.assert_close(bonus, torch.tensor([0.0, 0.125, 0.25]))


def test_visible_contact_target_ratio_uses_shared_domain_curriculum():
    base = _literal_class_assignments(ENV_CFG_PATH, "Revo3HandScrewEnvCfg")
    tactile = _literal_class_assignments(
        TACTILE_CFG_PATH, "Revo3HandScrewTactileMixinCfg"
    )
    current_ratio = _load_method(
        TACTILE_ENV_PATH,
        "Revo3HandScrewTactileEnv",
        "_current_visible_contact_target_ratio",
    )
    curriculum_progress = _load_method(
        ENV_CFG_PATH.parent / "revo3_hand_screw_env.py",
        "Revo3HandScrewEnv",
        "_domain_randomization_curriculum_progress",
    )

    assert base["domain_randomization_curriculum_enable"] is True
    assert base["domain_randomization_curriculum_start"] == 0
    assert base["domain_randomization_curriculum_end"] == 15_000_000
    assert tactile["visible_contact_target_ratio_initial"] == pytest.approx(0.01)

    env = SimpleNamespace(
        cfg=SimpleNamespace(
            domain_randomization_curriculum_enable=True,
            domain_randomization_curriculum_start=0,
            domain_randomization_curriculum_end=15_000_000,
            visible_contact_target_ratio_initial=0.01,
            visible_contact_target_ratio_mid=0.03,
            visible_contact_target_ratio_final=0.05,
            visible_contact_target_ratio_warmup_progress=0.10,
            visible_contact_target_ratio_mid_progress=0.50,
        ),
        common_step_counter=0,
        num_envs=100,
    )
    env._domain_randomization_curriculum_progress = lambda: curriculum_progress(env)
    assert current_ratio(env) == pytest.approx(0.01)
    env.common_step_counter = 15_000
    assert current_ratio(env) == pytest.approx(0.01)
    env.common_step_counter = 45_000
    assert current_ratio(env) == pytest.approx(0.02)
    env.common_step_counter = 75_000
    assert current_ratio(env) == pytest.approx(0.03)
    env.common_step_counter = 150_000
    assert current_ratio(env) == pytest.approx(0.05)

    env.cfg.domain_randomization_curriculum_enable = False
    env.common_step_counter = 0
    assert current_ratio(env) == pytest.approx(0.05)


def test_visible_contact_adaptive_scale_values_sparse_contacts_more():
    """Raise the value of each useful contact when contact frequency falls."""
    apply_scale = _load_method(
        TACTILE_ENV_PATH,
        "Revo3HandScrewTactileEnv",
        "_apply_visible_contact_adaptive_scale",
    )
    env = SimpleNamespace(
        cfg=SimpleNamespace(
            visible_contact_adaptive_ema=0.0,
            visible_contact_adaptive_scale_min=1.0e-3,
            visible_contact_adaptive_scale_max=300.0,
            visible_contact_reward_clip_ratio=10.0,
            visible_contact_reward_dynamic_clip_min=0.0,
        ),
        visible_task_reward_abs_ema=torch.tensor(0.0),
        visible_progress_abs_ema=torch.tensor(0.0),
        visible_contact_adaptive_scale=torch.tensor(1.0),
        _visible_adaptive_updates=0,
        _current_visible_contact_target_ratio=lambda: 0.05,
        extras={},
        device=torch.device("cpu"),
    )
    task_reward = torch.full((4,), 10.0)

    dense_reward = apply_scale(env, torch.ones(4), task_reward)
    dense_scale = env.visible_contact_adaptive_scale.item()
    sparse_reward = apply_scale(
        env, torch.tensor([1.0, 0.0, 0.0, 0.0]), task_reward
    )
    sparse_scale = env.visible_contact_adaptive_scale.item()

    assert dense_scale == pytest.approx(0.5)
    assert sparse_scale == pytest.approx(2.0)
    assert sparse_scale == pytest.approx(4.0 * dense_scale)
    assert dense_reward.tolist() == pytest.approx([0.5, 0.5, 0.5, 0.5])
    assert sparse_reward.tolist() == pytest.approx([2.0, 0.0, 0.0, 0.0])
    assert sparse_reward.mean().item() / task_reward.abs().mean().item() == pytest.approx(0.05)


def test_visible_contact_adaptive_scale_clips_each_environment():
    """Prevent an individual contact reward from dominating base reward."""
    apply_scale = _load_method(
        TACTILE_ENV_PATH,
        "Revo3HandScrewTactileEnv",
        "_apply_visible_contact_adaptive_scale",
    )
    env = SimpleNamespace(
        cfg=SimpleNamespace(
            visible_contact_adaptive_ema=0.0,
            visible_contact_adaptive_scale_min=1.0,
            visible_contact_adaptive_scale_max=300.0,
            visible_contact_reward_clip_ratio=0.10,
            visible_contact_reward_dynamic_clip_min=0.02,
        ),
        visible_task_reward_abs_ema=torch.tensor(0.0),
        visible_progress_abs_ema=torch.tensor(0.0),
        visible_contact_adaptive_scale=torch.tensor(1.0),
        _visible_adaptive_updates=0,
        _current_visible_contact_target_ratio=lambda: 0.05,
        extras={},
        device=torch.device("cpu"),
    )
    task_reward = torch.tensor([10.0, 2.0, 0.0, -4.0])

    reward = apply_scale(env, torch.ones(4), task_reward)

    assert reward.tolist() == pytest.approx([1.0, 0.2, 0.02, 0.4])
    assert env.extras["visible_contact_reward_clip_ratio"].item() == pytest.approx(0.75)


def test_finger_axis_contribution_uses_signed_object_torque():
    """Verify positive and reverse finger torques map to separate contributions."""
    compute_contribution = _load_method(
        TACTILE_ENV_PATH,
        "Revo3HandScrewTactileEnv",
        "_compute_finger_axis_contribution",
    )
    env = SimpleNamespace(
        cfg=SimpleNamespace(
            finger_contact_force_on_object_sign=-1.0,
            coord_rot_dir=1.0,
            finger_contribution_torque_ref=1.0,
            finger_physical_contact_force_ref=1.0,
        )
    )
    force_on_finger = torch.tensor([[[0.0, -1.0, 0.0], [0.0, 1.0, 0.0]]])
    contact_center = torch.tensor([[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
    object_center = torch.zeros((1, 3))
    object_axis = torch.tensor([[0.0, 0.0, 1.0]])

    signed, positive, negative, physical = compute_contribution(
        env,
        force_on_finger,
        contact_center,
        object_center,
        object_axis,
    )

    torch.testing.assert_close(signed, torch.tensor([[1.0, -1.0]]))
    torch.testing.assert_close(positive, torch.tensor([[1.0, 0.0]]))
    torch.testing.assert_close(negative, torch.tensor([[0.0, 1.0]]))
    assert torch.all(physical > 0.0)


def test_task_aligned_visible_progress_normalizes_weights_and_rejects_reverse_contact():
    """Normalize raw progress while gating zero and reverse contributions."""
    tactile = _literal_class_assignments(
        TACTILE_CFG_PATH, "Revo3HandScrewTactileMixinCfg"
    )
    compute_reward = _load_method(
        TACTILE_ENV_PATH,
        "Revo3HandScrewTactileEnv",
        "_compute_visible_contact_reward",
    )
    weights = tactile["tactile_visible_contact_finger_weights"][:3]
    env = SimpleNamespace(
        cfg=SimpleNamespace(
            tactile_visible_contact_finger_weights=weights,
            tactile_visible_contact_ratio_cap=tactile[
                "tactile_visible_contact_ratio_cap"
            ],
            visible_contact_contribution_power=tactile[
                "visible_contact_contribution_power"
            ],
        ),
        extras={},
        _finger_weight_tensor=lambda values: torch.tensor(values),
    )
    state = {
        "dip_physical_contact": torch.ones((1, 3), dtype=torch.bool),
        "per_finger_ratio": torch.full(
            (1, 3), tactile["tactile_visible_contact_ratio_cap"]
        ),
    }
    full_progress = compute_reward(env, state, torch.ones((1, 3)))
    zero_progress = compute_reward(env, state, torch.zeros((1, 3)))
    quarter_contribution_progress = compute_reward(
        env, state, torch.full((1, 3), 0.25)
    )

    assert full_progress.item() == pytest.approx(1.0)
    assert zero_progress.item() == pytest.approx(0.0)
    assert quarter_contribution_progress.item() == pytest.approx(0.5)


@pytest.mark.parametrize(
    "weights",
    [
        (1.0, 1.0, 1.0),
        (1.0, 1.0, 0.8, 0.8),
        (1.0, 1.0, 1.0, 1.0, 1.0),
    ],
)
def test_visible_contact_progress_maximum_is_independent_of_finger_count(weights):
    """Normalize three-, four-, and five-finger progress to the same maximum."""
    tactile = _literal_class_assignments(
        TACTILE_CFG_PATH, "Revo3HandScrewTactileMixinCfg"
    )
    compute_reward = _load_method(
        TACTILE_ENV_PATH,
        "Revo3HandScrewTactileEnv",
        "_compute_visible_contact_reward",
    )
    finger_count = len(weights)
    env = SimpleNamespace(
        cfg=SimpleNamespace(
            tactile_visible_contact_finger_weights=weights,
            tactile_visible_contact_ratio_cap=tactile[
                "tactile_visible_contact_ratio_cap"
            ],
            visible_contact_contribution_power=tactile[
                "visible_contact_contribution_power"
            ],
        ),
        extras={},
        _finger_weight_tensor=lambda values: torch.tensor(values),
    )
    state = {
        "dip_physical_contact": torch.ones((1, finger_count), dtype=torch.bool),
        "per_finger_ratio": torch.full(
            (1, finger_count), tactile["tactile_visible_contact_ratio_cap"]
        ),
    }

    progress = compute_reward(env, state, torch.ones((1, finger_count)))

    assert progress.item() == pytest.approx(1.0)


def test_tacsl_data_reconstructs_complete_force_and_contact_center():
    """Verify TacSL normal/shear components reconstruct world-frame contact data."""
    contact_force_and_center = _load_method(
        TACTILE_ENV_PATH,
        "Revo3HandScrewTactileEnv",
        "_contact_force_and_center_w",
    )
    tactile_data = SimpleNamespace(
        tactile_normal_force=torch.tensor([[2.0, 2.0]]),
        tactile_shear_force=torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]),
        tactile_points_pos_w=torch.tensor([[[1.0, 2.0, 3.0], [3.0, 2.0, 3.0]]]),
        tactile_points_quat_w=torch.tensor(
            [[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]]
        ),
    )
    env = SimpleNamespace(
        num_envs=1,
        device=torch.device("cpu"),
        hand=SimpleNamespace(
            data=SimpleNamespace(
                body_pos_w=torch.zeros((1, 1, 3)),
                body_quat_w=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]),
            )
        ),
        tactile_tip_body_ids=[0],
        scene=SimpleNamespace(env_origins=torch.tensor([[0.5, 0.5, 0.5]])),
        cfg=SimpleNamespace(tactile_active_finger_indices=(0,)),
        _tactile_sensor=[SimpleNamespace(data=tactile_data)],
        _contact_sensor=[],
        extras={},
    )

    force, center, valid = contact_force_and_center(env)

    torch.testing.assert_close(force, torch.tensor([[[1.0, 1.0, 4.0]]]))
    torch.testing.assert_close(center, torch.tensor([[[1.5, 1.5, 2.5]]]))
    assert valid.item() is True
    assert env.extras["tactile/contact_force_from_tacsl"].item() == 1.0


def test_tactile_tasks_avoid_unstable_physx_friction_tracking():
    """Ensure torque contribution does not call PhysX detailed-friction buffers."""
    base = _literal_class_assignments(ENV_CFG_PATH, "Revo3HandScrewEnvCfg")
    tactile = _literal_class_assignments(
        TACTILE_CFG_PATH, "Revo3HandScrewTactileMixinCfg"
    )

    assert base["contact_sensor_track_contact_points"] is False
    assert base["contact_sensor_track_friction_forces"] is False
    assert tactile["contact_sensor_track_contact_points"] is False
    assert tactile["contact_sensor_track_friction_forces"] is False
    assert tactile["contact_sensor_max_contact_data_count_per_prim"] == 8
    assert tactile["finger_contribution_torque_ref"] > 0.0


def test_contact_force_helper_prefers_object_filtered_force_matrix():
    """Exclude contacts with non-target objects when a filter matrix is available."""
    contact_forces = _load_method(
        TACTILE_ENV_PATH,
        "Revo3HandScrewTactileEnv",
        "_contact_forces_w",
    )
    sensor = SimpleNamespace(
        data=SimpleNamespace(
            force_matrix_w=torch.tensor([[[[1.0, 2.0, 3.0]]]]),
            net_forces_w=torch.tensor([[[9.0, 9.0, 9.0]]]),
        )
    )
    env = SimpleNamespace(_contact_sensor=[sensor])

    torch.testing.assert_close(
        contact_forces(env),
        torch.tensor([[[1.0, 2.0, 3.0]]]),
    )


def test_contact_buffer_diagnostics_tracks_current_and_historical_peak():
    """Verify detailed-buffer diagnostics retain the highest successful usage."""
    update_diagnostics = _load_method(
        TACTILE_ENV_PATH,
        "Revo3HandScrewTactileEnv",
        "_update_contact_buffer_diagnostics",
    )
    count_buffer = torch.tensor([[10, 6]], dtype=torch.int32)
    env = SimpleNamespace(
        device=torch.device("cpu"),
        _contact_sensor=[
            SimpleNamespace(
                contact_physx_view=SimpleNamespace(
                    _contact_count_buffer=count_buffer,
                    max_contact_data_count=32,
                )
            )
        ],
        _contact_buffer_peak_utilization=torch.tensor(0.0),
        extras={},
    )

    update_diagnostics(env)
    assert env.extras["tactile/contact_buffer_utilization"].item() == pytest.approx(0.5)
    assert env.extras["tactile/contact_buffer_peak_utilization"].item() == pytest.approx(0.5)
    assert env.extras["tactile/contact_buffer_capacity_per_dip"].item() == 32

    count_buffer[:] = 1
    update_diagnostics(env)
    assert env.extras["tactile/contact_buffer_utilization"].item() == pytest.approx(0.0625)
    assert env.extras["tactile/contact_buffer_peak_utilization"].item() == pytest.approx(0.5)


def test_contact_buffer_diagnostics_ignores_uninitialized_detailed_buffer():
    """Keep diagnostics inert before PhysX creates its optional count buffer."""
    update_diagnostics = _load_method(
        TACTILE_ENV_PATH,
        "Revo3HandScrewTactileEnv",
        "_update_contact_buffer_diagnostics",
    )
    env = SimpleNamespace(
        device=torch.device("cpu"),
        _contact_sensor=[
            SimpleNamespace(
                contact_physx_view=SimpleNamespace(max_contact_data_count=32)
            )
        ],
        _contact_buffer_peak_utilization=torch.tensor(0.0),
        extras={},
    )

    update_diagnostics(env)
    assert env.extras == {}
