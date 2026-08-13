"""Static and numerical tests for continuous tactile rotation tasks."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import pytest
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
ROTATION_DIR = (
    REPO_ROOT
    / "source/BrainCo_DexHand/BrainCo_DexHand/tasks/direct/hora_rotation"
)
CFG_PATH = ROTATION_DIR / "revo3_hand_tactile_rotate_env_cfg.py"
ENV_PATH = ROTATION_DIR / "revo3_hand_tactile_rotate_env.py"
HORA_ENV_PATH = ROTATION_DIR / "revo3_hand_hora_env.py"
REGISTRY_PATH = ROTATION_DIR / "__init__.py"
TRAIN_PATH = REPO_ROOT / "scripts/hora/train.py"
YAML_PATH = ROTATION_DIR / "agents/Revo3HandTactileRotate.yaml"
YAML_DIR = ROTATION_DIR / "agents"
PHYSICAL_LAYOUT_PATH = (
    REPO_ROOT
    / "source/BrainCo_DexHand/BrainCo_DexHand/tasks/tactile_layouts"
    / "revo3_right_official_diagram_estimate_v5.json"
)


def _class_literals(path: Path, class_name: str) -> dict[str, object]:
    """Return literal class assignments without importing Isaac Sim."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    result = {}
    for node in class_node.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        if not isinstance(node.targets[0], ast.Name):
            continue
        try:
            result[node.targets[0].id] = ast.literal_eval(node.value)
        except (ValueError, TypeError):
            continue
    return result


def _load_function(path: Path, function_name: str):
    """Compile one top-level function without importing simulator dependencies."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace = {"torch": torch}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[function_name]


def test_tactile_rotate_config_uses_unbounded_rotation_without_a_command():
    """Keep rotation as the primary reward and remove target-speed inputs."""
    cfg = _class_literals(CFG_PATH, "Revo3HandTactileRotateEnvCfg")
    source = ENV_PATH.read_text(encoding="utf-8")

    assert cfg["rotate_reward_scale"] == pytest.approx(2.5)
    assert cfg["unbounded_positive_rotate_reward"] is True
    assert cfg["object_linvel_penalty_scale"] == pytest.approx(-0.6)
    assert cfg["object_pos_reward_scale"] == pytest.approx(0.003)
    assert cfg["observation_space"] == 141
    assert cfg["student_proprio_command_dim"] == 0
    assert cfg["student_proprio_frame_dim"] == 42
    assert cfg["student_proprio_history_dim"] == 126
    assert cfg["student_obs_dim"] == 190
    assert "tactile_layout = ESTIMATED_OFFICIAL_LAYOUT" in CFG_PATH.read_text(encoding="utf-8")
    assert "self.priv_info_buf[:, 8]" not in source
    assert "target_angvel" not in source
    assert "compute_target_angvel_terms" not in source


def test_unbounded_rotate_reward_keeps_increasing_with_positive_speed():
    """Reward every positive speed increase while bounding reverse rotation."""
    compute_reward = _load_function(HORA_ENV_PATH, "compute_rotate_reward")
    axis_speed = torch.tensor([-2.0, -0.25, 0.5, 1.0, 4.0])

    unbounded = compute_reward(axis_speed, -0.5, 0.5, True)
    legacy = compute_reward(axis_speed, -0.5, 0.5, False)

    assert unbounded.tolist() == pytest.approx([-0.5, -0.25, 0.5, 1.0, 4.0])
    assert legacy.tolist() == pytest.approx([-0.5, -0.25, 0.5, 0.5, 0.5])


def test_tactile_rotate_matches_shared_screw_randomization_contract():
    """Keep free-object-compatible reset randomizations aligned with screw tasks."""
    cfg = _class_literals(CFG_PATH, "Revo3HandTactileRotateEnvCfg")

    assert cfg["object_size_scale_levels"] == (0.8, 0.9, 1.0, 1.1, 1.2)
    assert cfg["randomize_object_size"] is True
    assert cfg["action_delay"] == (0.0, 1.0)
    assert cfg["reset_joint_noise_frac"] == pytest.approx(0.1)
    assert cfg["object_xy_position_noise"] == pytest.approx(0.005)
    assert cfg["enable_contact_noise"] is True
    assert cfg["contact_force_noise_frac"] == pytest.approx(0.02)

    ball_cfg = _class_literals(CFG_PATH, "Revo3HandTactileRotateBallEnvCfg")
    cylinder_cfg = _class_literals(CFG_PATH, "Revo3HandTactileRotateCylinderEnvCfg")
    assert ball_cfg["object_size_scale_axes"] == (True, True, True)
    assert cylinder_cfg["object_size_scale_axes"] == (True, True, False)


def test_every_cloned_object_mesh_gets_an_explicit_sdf_and_size_override():
    """Prevent env-0-only SDF authoring from regressing in parallel scenes."""
    source = ENV_PATH.read_text(encoding="utf-8")

    assert "for env_index, scale in enumerate(self.object_size_scales.tolist())" in source
    assert "self._configure_object_sdf_collision(env_index)" in source
    assert "mesh.GetPointsAttr().Set(scaled_points)" in source
    assert "UsdPhysics.CollisionAPI.Apply(mesh_prim)" in source


def test_ball_and_cylinder_are_registered_and_routed_through_tactile_dagger():
    """Expose both assets through Gym and the existing T-S trainer."""
    registry = REGISTRY_PATH.read_text(encoding="utf-8")
    train = TRAIN_PATH.read_text(encoding="utf-8")

    assert "BrainCo-Direct-Revo3-TactileRotate-Ball-v0" in registry
    assert "BrainCo-Direct-Revo3-TactileRotate-Cylinder-v0" in registry
    assert "Revo3HandTactileRotateEnv" in registry
    assert "rotate_ball_tactile" in train
    assert "rotate_cylinder_tactile" in train
    assert "args.task in _TACTILE_TASKS" in train
    assert "agent_cls = TactileDAgger" in train


def test_tactile_rotate_yaml_uses_physical_layout_and_tactile_teacher():
    """Use the 115-node physical layout with the tactile teacher/student networks."""
    config = yaml.safe_load(YAML_PATH.read_text(encoding="utf-8"))

    assert config["tactile_layout"] == "estimated_official"
    assert config["ppo"]["reward_scale"] == pytest.approx(1.0)
    assert config["ppo"]["priv_info_dim"] == 1178
    assert config["network"]["tactile_encoder"]["type"] == "mlp"
    assert config["network"]["student_tactile_encoder"]["type"] == "conv1d"


def test_tactile_rotate_physical_layout_has_valve_compatible_115_nodes():
    """Keep rotation and five-finger valve tasks on the same tactile node contract."""
    config = yaml.safe_load(YAML_PATH.read_text(encoding="utf-8"))
    layout = json.loads(PHYSICAL_LAYOUT_PATH.read_text(encoding="utf-8"))
    finger_order = ("thumb", "index", "middle", "ring", "little")
    sensor_counts = tuple(len(layout["fingers"][name]["sensors"]) for name in finger_order)

    assert sensor_counts == (31, 21, 21, 21, 21)
    assert sum(sensor_counts) == 115
    teacher_tactile_dim = sum(sensor_counts) * 10 + len(finger_order) * 4
    assert teacher_tactile_dim == 1170
    assert config["ppo"]["priv_info_dim"] == 8 + teacher_tactile_dim


def test_tactile_rotate_yamls_cover_mlp_teacher_and_student_encoders():
    """Expose the public MLP teacher and conv1d/GRU student choices."""
    expected = {
        "Revo3HandTactileRotate.yaml": ("mlp", "conv1d"),
        "Revo3HandTactileRotateGRU.yaml": ("mlp", "gru"),
    }

    observed = {}
    for config_name in expected:
        config = yaml.safe_load((YAML_DIR / config_name).read_text(encoding="utf-8"))
        observed[config_name] = (
            config["network"]["tactile_encoder"]["type"],
            config["network"]["student_tactile_encoder"]["type"],
        )
        assert config["tactile_layout"] == "estimated_official"
        assert config["ppo"]["reward_scale"] == pytest.approx(1.0)
        assert config["ppo"]["priv_info_dim"] == 1178

    assert observed == expected
    assert {teacher for teacher, _ in observed.values()} == {"mlp"}
    assert {student for _, student in observed.values()} == {"conv1d", "gru"}


def test_student_history_contains_joint_state_without_contact_or_command():
    """Keep student proprio limited to normalized positions and joint targets."""
    tree = ast.parse(ENV_PATH.read_text(encoding="utf-8"))
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "Revo3HandTactileRotateEnv"
    )
    method = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == "_get_observations"
    )
    source = ast.get_source_segment(ENV_PATH.read_text(encoding="utf-8"), method)

    assert ": 2 * self.num_hand_dofs" in source
    assert "command_index" not in source
    assert "torch.cat" not in source
