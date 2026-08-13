"""Static integration tests for the frame813 structured tactile teacher config."""

from __future__ import annotations

import ast
import datetime
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = REPO_ROOT / "scripts/hora/train.py"
PLAY_PATH = REPO_ROOT / "scripts/hora/play.py"
SCREW_DIR = (
    REPO_ROOT
    / "source/BrainCo_DexHand/BrainCo_DexHand/tasks/direct/hora_screw"
)
ROTATION_DIR = (
    REPO_ROOT
    / "source/BrainCo_DexHand/BrainCo_DexHand/tasks/direct/hora_rotation"
)
FRAME813_PATH = SCREW_DIR / "agents/valvedriver_tactile_frame813.yaml"
PPO_PATH = (
    REPO_ROOT
    / "source/BrainCo_DexHand/BrainCo_DexHand/algo/hora/ppo/ppo.py"
)
TACTILE_DAGGER_PATH = (
    REPO_ROOT
    / "source/BrainCo_DexHand/BrainCo_DexHand/algo/hora/padapt/tactile_dagger.py"
)
LAYOUT_PATH = (
    REPO_ROOT
    / "source/BrainCo_DexHand/BrainCo_DexHand/tasks/tactile_layouts"
    / "revo3_right_official_diagram_estimate_v5.json"
)
TARGET_TASKS = (
    "nutbolt_tactile",
    "screwdriver_tactile",
    "valvedriver_tactile",
    "valvedriver_tactile25",
    "valvedriver_tactile_40",
    "rotate_ball_tactile",
    "rotate_cylinder_tactile",
)


def _compile_functions(path: Path, *function_names: str, namespace=None):
    """Compile selected top-level functions without importing Isaac Sim scripts."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in function_names
    ]
    assert {node.name for node in selected} == set(function_names)
    module = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    compiled_namespace = dict(namespace or {})
    exec(compile(module, str(path), "exec"), compiled_namespace)
    return tuple(compiled_namespace[name] for name in function_names)


def _task_choices(path: Path) -> tuple[str, ...]:
    """Read argparse task choices without executing the entry point."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
            continue
        if ast.literal_eval(node.args[0]) != "--task":
            continue
        choices = next(keyword.value for keyword in node.keywords if keyword.arg == "choices")
        return tuple(ast.literal_eval(choices))
    raise AssertionError(f"No --task parser declaration in {path}")


def _masked_joint_names(class_name: str) -> tuple[str, ...]:
    """Read one task's action-joint mask without importing its Isaac config."""
    path = SCREW_DIR / "revo3_hand_screw_env_cfg.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    post_init = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == "__post_init__"
    )
    for node in post_init.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if (
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "self"
            and target.attr == "masked_action_joint_names"
        ):
            return tuple(ast.literal_eval(node.value))
    raise AssertionError(f"No action mask in {class_name}")


def _active_fingers(masked_joint_names: tuple[str, ...]) -> tuple[str, ...]:
    canonical = ("thumb", "index", "middle", "ring", "little")
    return tuple(
        finger
        for finger in canonical
        if not any(f"_{finger}_" in joint_name for joint_name in masked_joint_names)
    )


def test_frame813_yaml_is_an_explicit_structured_teacher_config():
    config = yaml.safe_load(FRAME813_PATH.read_text(encoding="utf-8"))
    encoder = config["network"]["tactile_encoder"]

    assert config["tactile_layout"] == "estimated_official"
    assert config["network"]["mlp"]["units"] == [512, 256, 128]
    assert config["network"]["priv_mlp"]["units"] == [256, 128, 32]
    assert encoder == {
        "type": "finger_attention_gru",
        "node_channels": ["u", "v", "b", "d", "Fn", "Ft1", "Ft2"],
        "node_input_dim": 7,
        "node_source_channels": [0, 3, 5, 6, 7],
        "teacher_node_frame_channels": 10,
        "finger_context_channels": 4,
        "finger_token_dim": 32,
        "finger_mlp_hidden_dim": 64,
        "attention_heads": 4,
        "attention_ff_dim": 64,
        "gru_hidden_dim": 128,
        "gru_num_layers": 1,
        "gru_bidirectional": False,
        "history_len": 10,
    }
    assert config["network"]["student_tactile_encoder"] == {
        "type": "conv1d",
        "gated_fusion": False,
        "distill_dim": 32,
        "output_dim": 128,
    }
    assert config["ppo"]["priv_info"] is True
    assert config["ppo"]["proprio_adapt"] is False
    assert config["ppo"]["tactile_distill_coef"] == pytest.approx(0.0)


def test_existing_teacher_and_student_yaml_semantics_are_unchanged():
    expected = {
        SCREW_DIR / "agents/Revo3HandScrewTactile.yaml": ("mlp", "conv1d"),
        SCREW_DIR / "agents/Revo3HandScrewTactileGRU.yaml": ("mlp", "gru"),
        ROTATION_DIR / "agents/Revo3HandTactileRotate.yaml": ("mlp", "conv1d"),
        ROTATION_DIR / "agents/Revo3HandTactileRotateGRU.yaml": ("mlp", "gru"),
    }

    for path, encoder_types in expected.items():
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert (
            config["network"]["tactile_encoder"]["type"],
            config["network"]["student_tactile_encoder"]["type"],
        ) == encoder_types


@pytest.mark.parametrize("script_path", (TRAIN_PATH, PLAY_PATH))
def test_train_and_play_load_screw_config_for_every_target_task(script_path: Path):
    """Exercise each script's real config resolver without starting AppLauncher."""
    assert set(TARGET_TASKS).issubset(_task_choices(script_path))
    args = SimpleNamespace(
        train_cfg="valvedriver_tactile_frame813",
        tactile_layout=None,
        algo="PPO",
        checkpoint="/tmp/frame813-test-checkpoint.pth",
        output_name="frame813_test",
        num_envs=8,
        test=True,
        device="cpu",
        task="",
    )
    namespace = {
        "REPO_ROOT": REPO_ROOT,
        "args": args,
        "OmegaConf": OmegaConf,
        "os": os,
        "resolve_agent_tactile_layout": lambda cfg: str(cfg.tactile_layout),
        "validate_tactile_layout_name": lambda value: value,
        "_cap_layout_minibatch_size": lambda cfg, size: (size, None),
        "_resolve_minibatch_size": lambda rollout, preferred: preferred,
        "_is_stage2_checkpoint": lambda path: False,
    }
    (build_full_config,) = _compile_functions(
        script_path, "_build_full_config", namespace=namespace
    )

    for task_name in TARGET_TASKS:
        args.task = task_name
        args.tactile_layout = None
        full_config = build_full_config(seed=42)
        assert full_config.train.network.tactile_encoder.type == "finger_attention_gru"
        assert full_config.train.tactile_layout == "estimated_official"


@pytest.mark.parametrize(
    ("task_name", "finger_names", "counts", "base_priv_dim", "full_priv_dim"),
    (
        ("nutbolt_tactile", ("thumb", "index", "middle"), (31, 21, 21), 11, 753),
        (
            "screwdriver_tactile",
            ("thumb", "index", "middle", "ring"),
            (31, 21, 21, 21),
            11,
            967,
        ),
        (
            "valvedriver_tactile",
            ("thumb", "index", "middle", "ring", "little"),
            (31, 21, 21, 21, 21),
            11,
            1181,
        ),
        (
            "valvedriver_tactile25",
            ("thumb", "index", "middle", "ring", "little"),
            (31, 21, 21, 21, 21),
            11,
            1181,
        ),
        (
            "valvedriver_tactile_40",
            ("thumb", "index", "middle", "ring", "little"),
            (31, 21, 21, 21, 21),
            11,
            1181,
        ),
        (
            "rotate_ball_tactile",
            ("thumb", "index", "middle", "ring", "little"),
            (31, 21, 21, 21, 21),
            8,
            1178,
        ),
        (
            "rotate_cylinder_tactile",
            ("thumb", "index", "middle", "ring", "little"),
            (31, 21, 21, 21, 21),
            8,
            1178,
        ),
    ),
)
def test_task_resolved_dimensions_and_saved_runtime_metadata(
    tmp_path: Path,
    task_name: str,
    finger_names: tuple[str, ...],
    counts: tuple[int, ...],
    base_priv_dim: int,
    full_priv_dim: int,
):
    layout = json.loads(LAYOUT_PATH.read_text(encoding="utf-8"))
    observed_counts = tuple(
        len(layout["fingers"][finger_name]["sensors"])
        for finger_name in finger_names
    )
    total_nodes = sum(counts)
    teacher_frame_dim = total_nodes * 10 + len(finger_names) * 4

    assert observed_counts == counts
    assert base_priv_dim + teacher_frame_dim == full_priv_dim

    if task_name == "nutbolt_tactile":
        assert _active_fingers(_masked_joint_names("Revo3HandScrewNutBoltEnvCfg")) == finger_names
    elif task_name == "screwdriver_tactile":
        assert _active_fingers(_masked_joint_names("Revo3HandScrewDriverEnvCfg")) == finger_names
    elif task_name.startswith("valvedriver_tactile"):
        assert _active_fingers(_masked_joint_names("Revo3HandVavleDriverEnvCfg")) == finger_names

    attach_runtime, save_metadata = _compile_functions(
        TRAIN_PATH,
        "_attach_env_runtime_to_config",
        "_save_run_metadata",
        namespace={"OmegaConf": OmegaConf, "datetime": datetime, "os": os},
    )
    student_frame_dim = total_nodes * 5 + len(finger_names) * 4
    env_cfg = SimpleNamespace(
        sim=SimpleNamespace(dt=1.0 / 120.0),
        decimation=4,
        grasp_cache_path=f"assets/grasp_cache/hora/{task_name}",
        tactile_layout="estimated_official",
        action_scale=0.04167,
        action_space=21,
        observation_space=141,
        priv_info_dim=full_priv_dim,
        tactile_priv_offset=base_priv_dim,
        tactile_priv_dim=teacher_frame_dim,
        tactile_active_finger_names=finger_names,
        tactile_graph_sensor_counts=counts,
        tactile_graph_total_nodes=total_nodes,
        tactile_graph_common_channels=5,
        tactile_graph_context_channels=4,
        teacher_tactile_history_len=10,
        teacher_tactile_frame_dim=teacher_frame_dim,
        student_tactile_duration_tau=20.0,
        student_tactile_duration_max=100.0,
        tactile_shift_ema_beta=0.7,
        tactile_shift_max=0.2,
        masked_action_joint_names=(),
        student_proprio_command_dim=0,
        student_proprio_history_len=3,
        student_proprio_frame_dim=42,
        student_tactile_history_len=10,
        student_tactile_frame_dim=student_frame_dim,
    )
    full_config = OmegaConf.create(
        {"train": {"ppo": {"priv_info_dim": full_priv_dim}}}
    )

    attach_runtime(full_config, env_cfg, task_name)
    save_metadata(str(tmp_path), full_config)
    saved_path = next(tmp_path.glob("config_*.yaml"))
    saved = OmegaConf.load(saved_path)

    assert saved.train.ppo.priv_info_dim == full_priv_dim
    assert saved.env_runtime.priv_info_dim == full_priv_dim
    assert saved.env_runtime.tactile_priv_offset == base_priv_dim
    assert saved.env_runtime.tactile_priv_dim == teacher_frame_dim
    assert saved.env_runtime.teacher_tactile_frame_dim == teacher_frame_dim
    assert saved.env_runtime.teacher_tactile_history_len == 10
    assert tuple(saved.env_runtime.tactile_active_finger_names) == finger_names
    assert tuple(saved.env_runtime.tactile_graph_sensor_counts) == counts
    assert saved.env_runtime.tactile_graph_total_nodes == total_nodes


def test_stage2_mask_validation_defines_the_active_finger_count():
    tree = ast.parse(TRAIN_PATH.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_validate_stage1_teacher_for_tactile_stage2"
    )
    assigned_names = {
        target.id
        for node in ast.walk(function)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    assert "env_num_fingers" in assigned_names


@pytest.mark.parametrize(
    "env_path",
    (
        SCREW_DIR / "revo3_hand_screw_tactile_env.py",
        ROTATION_DIR / "revo3_hand_tactile_rotate_env.py",
    ),
    ids=("screw", "rotation"),
)
def test_teacher_history_allocation_and_reset_replicate_the_current_frame(
    env_path: Path,
):
    """Keep reset histories at the runtime-resolved ten-frame teacher shape."""
    source = env_path.read_text(encoding="utf-8")

    assert "self.cfg.teacher_tactile_history_len" in source
    assert "self.cfg.teacher_tactile_frame_dim" in source
    assert "self.teacher_tactile_hist_buf[at_reset_env_ids]" in source
    assert ".repeat(1, self.cfg.teacher_tactile_history_len, 1)" in source
    assert 'obs_dict["tactile_hist"] = self.teacher_tactile_hist_buf.clone()' in source


def test_teacher_history_is_threaded_through_ppo_playback_and_dagger():
    """Cover rollout, minibatch, test/play, and frozen-teacher input plumbing."""
    ppo_source = PPO_PATH.read_text(encoding="utf-8")
    play_source = PLAY_PATH.read_text(encoding="utf-8")
    dagger_source = TACTILE_DAGGER_PATH.read_text(encoding="utf-8")

    assert 'getattr(self.model, "use_tactile_history", False)' in ppo_source
    assert "tactile_hist_shape=tactile_hist_shape" in ppo_source
    assert "self.storage.update_data('tactile_hist', n, self.obs['tactile_hist'])" in ppo_source
    assert "tactile_hist = batch.get('tactile_hist')" in ppo_source
    assert "batch_dict['tactile_hist'] = tactile_hist" in ppo_source
    assert "input_dict['tactile_hist'] = obs_dict['tactile_hist']" in ppo_source

    assert "input_dict['tactile_hist'] = obs_dict['tactile_hist']" in play_source
    assert "teacher_input['tactile_hist'] = obs_dict['tactile_hist']" in dagger_source
    assert "tactile_hist=obs_dict.get('tactile_hist')" in dagger_source
