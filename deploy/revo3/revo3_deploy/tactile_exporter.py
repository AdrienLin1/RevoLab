from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml

from revo3_deploy.input_builder import GRAPH_CONTEXT_CHANNELS, GRAPH_NODE_CHANNELS
from revo3_deploy.robot_profile import JOINT_DIM
from revo3_deploy.sdk_hand_io import REVO3_TOUCH_MODULES
from revo3_deploy.task_registry import (
    TACTILE_STUDENT_COMMAND_OBSERVATION_SCHEMA,
    TACTILE_STUDENT_OBSERVATION_SCHEMA,
    TIP_MODULE_IDS,
    build_action_mask,
    resolve_tactile_task,
    student_proprio_frame_channels,
)

PROPRIO_HISTORY_LEN = 3
TACTILE_HISTORY_LEN = 10
TACTILE_LAYOUT = "estimated_official"
TACTILE_LAYOUT_VERSION = "revo3_right_official_diagram_estimate_v5"


def export_tactile_student(
    *,
    checkpoint_path: str | Path,
    config_path: str | Path | None = None,
    profile_path: str | Path,
    output_dir: str | Path,
    task: str,
    layout_path: str | Path | None = None,
    sensor_map_path: str | Path | None = None,
    opset: int = 17,
    dynamic_batch: bool = True,
) -> tuple[Path, Path]:
    """Export a HORA tactile DAgger student and its hardware contract.

    Args:
        checkpoint_path: Stage-2 ``.ckpt`` containing student and proprio RMS state.
        config_path: Saved HORA config, or ``None`` to discover it beside the checkpoint.
        profile_path: Revo3 deployment profile containing policy joint order.
        output_dir: Directory for ``policy.onnx`` and ``policy.yaml``.
        task: Supported tactile training task, alias, or Gym ID.
        layout_path: Optional physical sensor layout JSON override.
        sensor_map_path: Optional verified SDK-channel to diagram-ID mapping YAML.
        opset: ONNX operator-set version.
        dynamic_batch: Whether the exported batch axis is dynamic.

    Returns:
        Paths to the exported ONNX model and policy YAML.
    """

    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    profile_path = Path(profile_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    for path, label in ((checkpoint_path, "checkpoint"), (profile_path, "profile")):
        if not path.is_file():
            raise FileNotFoundError(f"Tactile export {label} not found: {path}")
    config_path = resolve_tactile_config_path(
        checkpoint_path,
        config_path,
        task=task,
    )

    config = _load_yaml(config_path)
    profile = _load_yaml(profile_path)
    layout_json_path = _resolve_layout_path(layout_path)
    with layout_json_path.open("r", encoding="utf-8") as stream:
        layout_data = json.load(stream)
    resolved_sensor_map_path = None
    sensor_map_data = None
    if sensor_map_path:
        resolved_sensor_map_path = Path(sensor_map_path).expanduser().resolve()
        if not resolved_sensor_map_path.is_file():
            raise FileNotFoundError(
                f"Tactile SDK channel mapping not found: {resolved_sensor_map_path}"
            )
        sensor_map_data = _load_yaml(resolved_sensor_map_path)

    spec = build_tactile_export_spec(
        config,
        profile,
        layout_data,
        sensor_map_data,
        task=task,
    )
    if resolved_sensor_map_path is not None:
        spec["sensor_mapping"]["source"] = str(resolved_sensor_map_path)
    source_root = _repo_root() / "source" / "BrainCo_DexHand"
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

    import torch

    from BrainCo_DexHand.algo.hora.models.models import TactileStudentPolicy
    from BrainCo_DexHand.algo.hora.models.running_mean_std import RunningMeanStd

    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    for key in ("student", "proprio_mean_std"):
        if key not in checkpoint:
            raise RuntimeError(
                f"Stage-2 tactile checkpoint is missing {key!r}: {checkpoint_path}"
            )

    model = TactileStudentPolicy(copy.deepcopy(spec["student_model_kwargs"]))
    model.load_state_dict(checkpoint["student"], strict=False)
    model.cpu().eval()
    _replace_rms_norm_for_onnx(model, torch)
    proprio_rms = RunningMeanStd(
        (spec["proprio_history_len"], spec["proprio_frame_dim"])
    )
    proprio_rms.load_state_dict(checkpoint["proprio_mean_std"], strict=True)
    proprio_rms.cpu().eval()

    class TactileStudentExportWrapper(torch.nn.Module):
        def __init__(self, student, normalizer):
            """Store the trained student and its proprio normalizer."""

            super().__init__()
            self.student = student
            self.normalizer = normalizer

        def forward(self, student_proprio_hist, student_tactile_hist):
            """Normalize proprioception and return clipped student actions."""

            normalized_proprio = self.normalizer(student_proprio_hist)
            action = self.student(normalized_proprio, student_tactile_hist)
            return torch.clamp(action, -1.0, 1.0)

    wrapper = TactileStudentExportWrapper(model, proprio_rms).eval()
    proprio = torch.zeros(
        (1, spec["proprio_history_len"], spec["proprio_frame_dim"]),
        dtype=torch.float32,
    )
    tactile = torch.zeros(
        (1, spec["tactile_history_len"], spec["tactile_frame_dim"]),
        dtype=torch.float32,
    )
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            "student_proprio_hist": {0: "B"},
            "student_tactile_hist": {0: "B"},
            "action": {0: "B"},
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "policy.onnx"
    policy_path = output_dir / "policy.yaml"
    with tempfile.NamedTemporaryFile(
        prefix="policy_",
        suffix=".onnx",
        dir=output_dir,
        delete=False,
    ) as temp_stream:
        temporary_onnx_path = Path(temp_stream.name)
    mha_fastpath_enabled = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                (proprio, tactile),
                str(temporary_onnx_path),
                opset_version=int(opset),
                input_names=["student_proprio_hist", "student_tactile_hist"],
                output_names=["action"],
                dynamic_axes=dynamic_axes,
                do_constant_folding=True,
            )
        spec["onnx_verification"] = _verify_exported_onnx(
            wrapper,
            temporary_onnx_path,
            spec,
            torch,
            dynamic_batch=dynamic_batch,
        )
        os.replace(temporary_onnx_path, onnx_path)
        spec["onnx_sha256"] = _sha256_file(onnx_path)
    finally:
        torch.backends.mha.set_fastpath_enabled(mha_fastpath_enabled)
        temporary_onnx_path.unlink(missing_ok=True)

    policy = build_tactile_policy_metadata(
        spec=spec,
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        layout_path=layout_json_path,
        task=task,
        policy_rate_hz=spec["policy_rate_hz"],
        dynamic_batch=dynamic_batch,
    )
    _write_yaml_atomic(policy_path, policy)
    return onnx_path, policy_path


def build_tactile_export_spec(
    config: dict[str, Any],
    profile: dict[str, Any],
    layout_data: dict[str, Any],
    sensor_map_data: dict[str, Any] | None = None,
    *,
    task: str | None = None,
) -> dict[str, Any]:
    """Build and validate the student model and physical-sensor export spec.

    Args:
        config: Saved HORA training configuration.
        profile: Revo3 deployment profile.
        layout_data: Versioned estimated-official sensor layout.
        sensor_map_data: Optional SDK channel mapping for the diagram sensor IDs.
        task: Tactile task identifier; falls back to new config runtime metadata.

    Returns:
        Validated model dimensions, model kwargs, fingers, and joint action mask.
    """

    train = _require_mapping(config, "train")
    network = _require_mapping(train, "network")
    ppo = _require_mapping(train, "ppo")
    layout_cfg = copy.deepcopy(_require_mapping(network, "tactile_layout_encoder"))
    student_cfg = copy.deepcopy(_require_mapping(network, "student_tactile_encoder"))
    mlp_cfg = _require_mapping(network, "mlp")

    layout_name = str(layout_cfg.get("layout", ""))
    env_runtime = _require_mapping(config, "env_runtime")
    runtime_task = str(
        env_runtime.get("task") or env_runtime.get("task_name") or ""
    ).strip()
    requested_task = str(task or runtime_task).strip()
    if not requested_task:
        raise ValueError(
            "Tactile export requires a task name because observation fingers and action "
            "joints are task-dependent."
        )
    task_spec = resolve_tactile_task(requested_task)
    if runtime_task:
        runtime_task_spec = resolve_tactile_task(runtime_task)
        if runtime_task_spec.canonical_name != task_spec.canonical_name:
            raise ValueError(
                f"Requested task {task_spec.canonical_name!r} differs from saved config "
                f"task {runtime_task_spec.canonical_name!r}."
            )
    runtime_layout = str(env_runtime.get("tactile_layout", ""))
    if layout_name != TACTILE_LAYOUT or runtime_layout != TACTILE_LAYOUT:
        raise ValueError(
            "Real fingertip deployment requires an estimated_official student checkpoint; "
            f"network={layout_name!r}, env_runtime={runtime_layout!r}."
        )
    if layout_data.get("layout_version") != TACTILE_LAYOUT_VERSION:
        raise ValueError(
            f"Expected tactile layout {TACTILE_LAYOUT_VERSION!r}, got "
            f"{layout_data.get('layout_version')!r}."
        )

    finger_names = tuple(str(value) for value in layout_cfg.get("finger_names") or ())
    if not finger_names or len(set(finger_names)) != len(finger_names):
        raise ValueError("tactile_layout_encoder.finger_names must be non-empty and unique.")
    finger_names = task_spec.validate_observation_fingers(finger_names)
    task_contract = task_spec.build_contract(finger_names)
    _validate_saved_runtime_contract(env_runtime, task_contract)
    layout_fingers = _require_mapping(layout_data, "fingers")
    fingers = []
    total_nodes = 0
    for name in finger_names:
        if name not in TIP_MODULE_IDS or name not in layout_fingers:
            raise ValueError(f"Unsupported tactile finger in checkpoint config: {name!r}")
        module_id = TIP_MODULE_IDS[name]
        module_name, module_count = REVO3_TOUCH_MODULES[module_id]
        sensors = layout_fingers[name].get("sensors") or []
        official_ids = [int(sensor["official_sensor_id"]) for sensor in sensors]
        if official_ids != list(range(1, module_count + 1)):
            raise ValueError(
                f"{name} official sensor IDs must be 1..{module_count}, got {official_ids}."
            )
        positions = [
            [float(value) for value in sensor["position_finger_longitudinal_lateral_m"]]
            for sensor in sensors
        ]
        fingers.append(
            {
                "name": name,
                "module_id": module_id,
                "module_name": module_name,
                "sensor_count": module_count,
                "official_sensor_ids": official_ids,
                "sensor_positions_xy": positions,
            }
        )
        total_nodes += module_count

    sensor_mapping = _apply_sensor_mapping(fingers, sensor_map_data)

    tactile_frame_dim = total_nodes * GRAPH_NODE_CHANNELS + len(fingers) * GRAPH_CONTEXT_CHANNELS
    if total_nodes != int(task_contract["total_nodes"]):
        raise ValueError(
            f"Task {task_spec.canonical_name!r} expects {task_contract['total_nodes']} "
            f"physical nodes, but the layout contains {total_nodes}."
        )
    if tactile_frame_dim != int(task_contract["tactile_frame_dim"]):
        raise ValueError(
            f"Task {task_spec.canonical_name!r} expects tactile frame "
            f"{task_contract['tactile_frame_dim']}, got {tactile_frame_dim}."
        )
    configured_frame_dim = int(layout_cfg.get("student_frame_dim", 0))
    if configured_frame_dim != tactile_frame_dim:
        raise ValueError(
            f"Configured student tactile frame {configured_frame_dim} != physical layout "
            f"{total_nodes}*{GRAPH_NODE_CHANNELS}+{len(fingers)}*{GRAPH_CONTEXT_CHANNELS}="
            f"{tactile_frame_dim}."
        )

    policy_joint_order = tuple(str(value) for value in profile.get("policy_joint_order") or ())
    if len(policy_joint_order) != JOINT_DIM or len(set(policy_joint_order)) != JOINT_DIM:
        raise ValueError("Deployment profile policy_joint_order must contain 21 unique joints.")
    action_mask = build_action_mask(policy_joint_order, task_spec.action_fingers)

    transformer_cfg = student_cfg.get("transformer") or {}
    tactile_history_len = int(
        transformer_cfg.get(
            "history_len",
            (_require_mapping(network, "tactile_encoder")).get(
                "history_len", TACTILE_HISTORY_LEN
            ),
        )
    )
    proprio_frame_dim = int(task_contract["student_proprio_frame_dim"])
    _validate_saved_history_contract(
        env_runtime,
        tactile_history_len,
        proprio_frame_dim,
        task_spec.public_command_channels,
    )
    tactile_emb_dim = int(
        transformer_cfg.get("output_dim", student_cfg.get("output_dim", 64))
    )
    configured_num_fingers = transformer_cfg.get("num_fingers")
    if configured_num_fingers is not None and int(configured_num_fingers) != len(fingers):
        raise ValueError(
            f"student_tactile_encoder.transformer.num_fingers={configured_num_fingers} "
            f"does not match task observation fingers {finger_names}."
        )
    layout_cfg.setdefault("graph", {})["compile_graph"] = False
    student_model_kwargs = {
        "actor_units": [int(value) for value in mlp_cfg.get("units") or ()],
        "actions_num": JOINT_DIM,
        "proprio_hist_len": PROPRIO_HISTORY_LEN,
        "proprio_frame_dim": proprio_frame_dim,
        "tactile_frame_dim": tactile_frame_dim,
        "tactile_hist_len": tactile_history_len,
        "tactile_emb_dim": tactile_emb_dim,
        "tactile_encoder_type": str(student_cfg.get("type", "transformer")),
        "tactile_encoder_cfg": student_cfg,
        "tactile_layout_encoder_cfg": layout_cfg,
        "gated_fusion": bool(student_cfg.get("gated_fusion", True)),
        "distill_dim": int(student_cfg.get("distill_dim", 64)),
    }
    if not student_model_kwargs["actor_units"]:
        raise ValueError("train.network.mlp.units must be non-empty.")
    if str(student_model_kwargs["tactile_encoder_type"]) != "transformer":
        raise ValueError("estimated_official physical-node deployment requires transformer/GNN student.")
    if not bool(ppo.get("normalize_input", True)):
        raise ValueError("Tactile export expects the trained proprio_mean_std normalizer.")

    action_scale = float(profile.get("action_scale", 1.0 / 24.0))
    saved_action_scale = env_runtime.get("action_scale")
    if saved_action_scale is not None and not abs(
        float(saved_action_scale) - action_scale
    ) <= 1.0e-12:
        raise ValueError(
            f"Saved env_runtime.action_scale={saved_action_scale} differs from "
            f"deployment profile action_scale={action_scale}."
        )
    policy_rate_hz = float(env_runtime.get("policy_rate_hz", task_spec.policy_rate_hz))
    if policy_rate_hz <= 0.0:
        raise ValueError("Saved tactile policy_rate_hz must be positive.")
    if not abs(policy_rate_hz - task_spec.policy_rate_hz) <= 1.0e-12:
        raise ValueError(
            f"Saved policy_rate_hz={policy_rate_hz} differs from task "
            f"{task_spec.canonical_name!r} rate {task_spec.policy_rate_hz}."
        )
    configured_action_dim = env_runtime.get("action_dim")
    if configured_action_dim is not None and int(configured_action_dim) != JOINT_DIM:
        raise ValueError(
            f"Saved env_runtime.action_dim={configured_action_dim} differs from "
            f"deployment action dimension {JOINT_DIM}."
        )

    return {
        "task_contract": task_contract,
        "policy_joint_order": policy_joint_order,
        "action_scale": action_scale,
        "action_mask": action_mask,
        "fingers": fingers,
        "proprio_history_len": PROPRIO_HISTORY_LEN,
        "proprio_frame_dim": proprio_frame_dim,
        "public_command_channels": list(task_spec.public_command_channels),
        "tactile_history_len": tactile_history_len,
        "tactile_frame_dim": tactile_frame_dim,
        "total_nodes": total_nodes,
        "sensor_mapping": sensor_mapping,
        "duration_tau": float(env_runtime.get("student_tactile_duration_tau", 20.0)),
        "duration_max": float(env_runtime.get("student_tactile_duration_max", 100.0)),
        "shift_ema_beta": float(env_runtime.get("tactile_shift_ema_beta", 0.7)),
        "shift_max": float(env_runtime.get("tactile_shift_max", 0.2)),
        "policy_rate_hz": policy_rate_hz,
        "student_model_kwargs": student_model_kwargs,
    }


def build_tactile_policy_metadata(
    *,
    spec: dict[str, Any],
    checkpoint_path: Path,
    config_path: Path,
    layout_path: Path,
    task: str,
    policy_rate_hz: float,
    dynamic_batch: bool,
) -> dict[str, Any]:
    """Build the deploy-time YAML contract for an exported tactile student."""

    batch_dim: str | int = "B" if dynamic_batch else 1
    return {
        "export": {
            "stage": "tactile_student",
            "source_checkpoint": str(checkpoint_path),
            "source_config": str(config_path),
            "task": spec["task_contract"]["canonical_name"],
            "requested_task": str(task),
        },
        "task_contract": copy.deepcopy(spec["task_contract"]),
        "observation_contract": {
            "schema": (
                TACTILE_STUDENT_COMMAND_OBSERVATION_SCHEMA
                if spec["public_command_channels"]
                else TACTILE_STUDENT_OBSERVATION_SCHEMA
            ),
            "student_proprio": {
                "history_len": spec["proprio_history_len"],
                "frame_dim": spec["proprio_frame_dim"],
                "frame_channels": student_proprio_frame_channels(
                    spec["public_command_channels"]
                ),
                "contains_raw_3d_force": False,
            },
            "student_tactile": {
                "history_len": spec["tactile_history_len"],
                "frame_dim": spec["tactile_frame_dim"],
                "representation": "derived_public_contact_graph",
                "contains_raw_3d_force": False,
            },
        },
        "artifacts": {
            "onnx": "policy.onnx",
            "onnx_sha256": spec.get("onnx_sha256"),
        },
        "io_contract": {
            "inputs": [
                {
                    "name": "student_proprio_hist",
                    "shape": [
                        batch_dim,
                        spec["proprio_history_len"],
                        spec["proprio_frame_dim"],
                    ],
                    "dtype": "float32",
                },
                {
                    "name": "student_tactile_hist",
                    "shape": [
                        batch_dim,
                        spec["tactile_history_len"],
                        spec["tactile_frame_dim"],
                    ],
                    "dtype": "float32",
                },
            ],
            "outputs": [
                {
                    "name": "action",
                    "shape": [batch_dim, JOINT_DIM],
                    "dtype": "float32",
                }
            ],
            "action_semantics": "delta",
            "action_formula": (
                "cur_targets = prev_targets + action_scale * clipped_action, "
                "apply action_mask_right_hand, then clamp to joint limits"
            ),
            "action_clip": [-1.0, 1.0],
            "action_scale": spec["action_scale"],
            "action_mask_right_hand": spec["action_mask"],
            "policy_rate_hz": float(policy_rate_hz),
            "joint_order_right_hand": list(spec["policy_joint_order"]),
            "dynamic_batch": bool(dynamic_batch),
        },
        "normalization": {
            "baked_in_onnx": True,
            "proprio_source": "checkpoint.proprio_mean_std",
            "tactile": "structural channels are not normalized during training",
        },
        "verification": copy.deepcopy(
            spec.get(
                "onnx_verification",
                {"performed": False},
            )
        ),
        "tactile": {
            "layout": TACTILE_LAYOUT,
            "layout_version": TACTILE_LAYOUT_VERSION,
            "layout_source": str(layout_path),
            "sensor_value_order": (
                "SDK arrays are reordered by sdk_channel_ids_by_official_sensor_id "
                "before graph feature construction"
            ),
            "sensor_mapping": spec["sensor_mapping"],
            "node_channels": ["contact", "established", "released", "duration", "eta"],
            "finger_context_channels": [
                "shift_x",
                "shift_y",
                "shift_valid",
                "contact_ratio",
            ],
            "proprio_history_len": spec["proprio_history_len"],
            "proprio_frame_dim": spec["proprio_frame_dim"],
            "history_len": spec["tactile_history_len"],
            "frame_dim": spec["tactile_frame_dim"],
            "total_nodes": spec["total_nodes"],
            "duration_tau": spec["duration_tau"],
            "duration_max": spec["duration_max"],
            "shift_ema_beta": spec["shift_ema_beta"],
            "shift_max": spec["shift_max"],
            "fingers": spec["fingers"],
        },
}


def _apply_sensor_mapping(
    fingers: list[dict[str, Any]],
    sensor_map_data: dict[str, Any] | None,
) -> dict[str, Any]:
    """Attach and validate a diagram-ID to SDK-channel permutation."""

    if sensor_map_data is None:
        for finger in fingers:
            count = int(finger["sensor_count"])
            finger["sdk_channel_ids_by_official_sensor_id"] = list(
                range(1, count + 1)
            )
        return {
            "verified": False,
            "source": "identity_assumption_not_hardware_verified",
            "mapping_semantics": (
                "list index = official_sensor_id - 1; value = 1-based SDK channel ID"
            ),
        }

    if sensor_map_data.get("layout_version") != TACTILE_LAYOUT_VERSION:
        raise ValueError(
            f"Sensor map layout_version must be {TACTILE_LAYOUT_VERSION!r}."
        )
    mapped_fingers = sensor_map_data.get("fingers")
    if not isinstance(mapped_fingers, dict):
        raise ValueError("Sensor map must contain a fingers mapping.")
    for finger in fingers:
        name = str(finger["name"])
        mapped = mapped_fingers.get(name)
        if not isinstance(mapped, dict):
            raise ValueError(f"Sensor map is missing active finger {name!r}.")
        if int(mapped.get("module_id", -1)) != int(finger["module_id"]):
            raise ValueError(
                f"Sensor map module_id for {name} must be {finger['module_id']}."
            )
        channels = [
            int(value)
            for value in mapped.get("sdk_channel_ids_by_official_sensor_id") or ()
        ]
        count = int(finger["sensor_count"])
        if sorted(channels) != list(range(1, count + 1)):
            raise ValueError(
                f"Sensor map for {name} must be a permutation of SDK channels 1..{count}."
            )
        finger["sdk_channel_ids_by_official_sensor_id"] = channels
    verified = sensor_map_data.get("verified", False)
    if not isinstance(verified, bool):
        raise ValueError("Sensor map verified must be a YAML boolean.")
    return {
        "verified": verified,
        "source": "provided_sensor_map",
        "mapping_semantics": (
            "list index = official_sensor_id - 1; value = 1-based SDK channel ID"
        ),
    }


def resolve_tactile_config_path(
    checkpoint_path: str | Path,
    config_path: str | Path | None = None,
    *,
    task: str | None = None,
) -> Path:
    """Resolve an explicit config or discover the latest config beside a checkpoint.

    Stage-2 checkpoints are stored under ``RUN_DIR/stage2_nn`` while their
    complete training snapshots are stored as ``RUN_DIR/config_*.yaml``.

    Args:
        checkpoint_path: HORA Stage-2 checkpoint path.
        config_path: Optional explicit saved training config.
        task: Optional task used to filter configs with saved runtime metadata.

    Returns:
        Absolute path to the selected saved config.
    """

    if config_path:
        resolved = Path(config_path).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Tactile export config not found: {resolved}")
        return resolved

    checkpoint = Path(checkpoint_path).expanduser().resolve()
    search_dirs = (checkpoint.parent.parent, checkpoint.parent)
    candidates = {
        candidate.resolve()
        for directory in search_dirs
        for candidate in directory.glob("config_*.yaml")
        if candidate.is_file()
    }
    if not candidates:
        raise FileNotFoundError(
            "Could not discover a saved HORA config_*.yaml beside checkpoint "
            f"{checkpoint}. Pass --config explicitly."
        )
    matching = []
    unspecified = []
    requested_task = resolve_tactile_task(task) if task else None
    for candidate in candidates:
        candidate_cfg = _load_yaml(candidate)
        runtime_cfg = candidate_cfg.get("env_runtime")
        saved_task = (
            str(runtime_cfg.get("task") or runtime_cfg.get("task_name") or "").strip()
            if isinstance(runtime_cfg, dict)
            else ""
        )
        if not saved_task:
            unspecified.append(candidate)
            continue
        if (
            requested_task is None
            or _try_resolve_canonical_task(saved_task)
            == requested_task.canonical_name
        ):
            matching.append(candidate)
    eligible = matching or unspecified
    if not eligible:
        raise ValueError(
            f"Found saved configs beside {checkpoint}, but none match task "
            f"{requested_task.canonical_name!r}. Pass --config explicitly if the "
            "checkpoint was moved."
        )
    return max(eligible, key=lambda path: (path.stat().st_mtime_ns, path.name))


def _validate_saved_runtime_contract(
    env_runtime: dict[str, Any], task_contract: dict[str, Any]
) -> None:
    """Validate optional exact observation metadata written by newer training runs.

    Args:
        env_runtime: Saved training environment runtime values.
        task_contract: Task contract inferred from the requested task and network.
    """

    expected_sequences = {
        "tactile_observation_fingers": task_contract["observation_fingers"],
        "action_fingers": task_contract["action_fingers"],
    }
    for key, expected in expected_sequences.items():
        configured = env_runtime.get(key)
        if configured is not None and list(configured) != list(expected):
            raise ValueError(
                f"Saved env_runtime.{key}={list(configured)} differs from task contract "
                f"{list(expected)}."
            )
    expected_scalars = {
        "student_tactile_frame_dim": task_contract["tactile_frame_dim"],
        "student_tactile_total_nodes": task_contract["total_nodes"],
    }
    for key, expected in expected_scalars.items():
        configured = env_runtime.get(key)
        if configured is not None and int(configured) != int(expected):
            raise ValueError(
                f"Saved env_runtime.{key}={configured} differs from task contract "
                f"{expected}."
            )
    configured_range = env_runtime.get("target_angvel_range_rad_s")
    expected_range = task_contract.get("target_angvel_range_rad_s")
    if configured_range is not None and list(configured_range) != list(
        expected_range or ()
    ):
        raise ValueError(
            "Saved env_runtime.target_angvel_range_rad_s differs from task contract "
            f"{expected_range}."
        )


def _validate_saved_history_contract(
    env_runtime: dict[str, Any],
    tactile_history_len: int,
    proprio_frame_dim: int,
    public_command_channels: tuple[str, ...],
) -> None:
    """Validate saved temporal observation shapes against the deploy builder.

    Args:
        env_runtime: Saved training environment runtime values.
        tactile_history_len: History length reconstructed from the student network.
        proprio_frame_dim: Task-specific proprioceptive frame width.
        public_command_channels: Ordered task command channels.
    """

    expected = {
        "student_proprio_history_len": PROPRIO_HISTORY_LEN,
        "student_proprio_frame_dim": proprio_frame_dim,
        "student_tactile_history_len": tactile_history_len,
    }
    for key, value in expected.items():
        configured = env_runtime.get(key)
        if configured is not None and int(configured) != int(value):
            raise ValueError(
                f"Saved env_runtime.{key}={configured} differs from deploy contract {value}."
            )
    expected_schema = (
        TACTILE_STUDENT_COMMAND_OBSERVATION_SCHEMA
        if public_command_channels
        else TACTILE_STUDENT_OBSERVATION_SCHEMA
    )
    configured_schema = env_runtime.get("student_observation_schema")
    if configured_schema is not None and str(configured_schema) != expected_schema:
        raise ValueError(
            f"Saved env_runtime.student_observation_schema={configured_schema!r} differs "
            f"from deploy contract {expected_schema!r}."
        )
    configured_commands = env_runtime.get("student_public_command_channels")
    if configured_commands is not None and list(configured_commands) != list(
        public_command_channels
    ):
        raise ValueError(
            "Saved env_runtime.student_public_command_channels differs from the "
            f"task contract {list(public_command_channels)}."
        )
    configured_force_dim = env_runtime.get("student_proprio_raw_force_dim")
    if configured_force_dim is not None and int(configured_force_dim) != 0:
        raise ValueError(
            "Tactile student proprio must not contain raw force channels; expected "
            "env_runtime.student_proprio_raw_force_dim=0."
        )


def _resolve_layout_path(path: str | Path | None) -> Path:
    """Resolve the built-in physical layout or a user override."""

    if path is None:
        path = (
            _repo_root()
            / "source"
            / "BrainCo_DexHand"
            / "BrainCo_DexHand"
            / "tasks"
            / "tactile_layouts"
            / f"{TACTILE_LAYOUT_VERSION}.json"
        )
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Physical tactile layout not found: {resolved}")
    return resolved


def _replace_rms_norm_for_onnx(module, torch_module) -> None:
    """Replace ``nn.RMSNorm`` with an ONNX-opset-17-compatible equivalent.

    Args:
        module: Loaded student model whose weights are already restored.
        torch_module: Imported torch module, passed to keep torch an export-only dependency.
    """

    class OnnxRmsNorm(torch_module.nn.Module):
        def __init__(self, source):
            """Copy an RMSNorm layer into ONNX-compatible operations."""

            super().__init__()
            self.eps = (
                float(source.eps)
                if source.eps is not None
                else float(torch_module.finfo(source.weight.dtype).eps)
            )
            self.weight = torch_module.nn.Parameter(
                source.weight.detach().clone(),
                requires_grad=False,
            )

        def forward(self, value):
            """Apply root-mean-square normalization to the last axis."""

            mean_square = value.square().mean(dim=-1, keepdim=True)
            normalized = value * torch_module.rsqrt(mean_square + self.eps)
            return normalized * self.weight

    for name, child in tuple(module.named_children()):
        if isinstance(child, torch_module.nn.RMSNorm):
            setattr(module, name, OnnxRmsNorm(child))
        else:
            _replace_rms_norm_for_onnx(child, torch_module)


def _verify_exported_onnx(
    wrapper,
    onnx_path: Path,
    spec: dict[str, Any],
    torch_module,
    *,
    dynamic_batch: bool,
    atol: float = 1.0e-4,
) -> dict[str, Any]:
    """Compare exported ONNX actions against the loaded PyTorch policy.

    Args:
        wrapper: Export wrapper containing the normalizer and student.
        onnx_path: Temporary ONNX artifact to validate.
        spec: Validated model dimensions.
        torch_module: Imported torch module.
        dynamic_batch: Whether the ONNX batch dimension is dynamic.
        atol: Maximum accepted absolute action difference.

    Returns:
        Serializable verification result for ``policy.yaml``.
    """

    import onnxruntime as ort

    batch_size = 2 if dynamic_batch else 1
    generator = torch_module.Generator(device="cpu")
    generator.manual_seed(20260802)
    proprio = torch_module.randn(
        batch_size,
        int(spec["proprio_history_len"]),
        int(spec["proprio_frame_dim"]),
        generator=generator,
        dtype=torch_module.float32,
    )
    tactile = torch_module.rand(
        batch_size,
        int(spec["tactile_history_len"]),
        int(spec["tactile_frame_dim"]),
        generator=generator,
        dtype=torch_module.float32,
    )
    with torch_module.no_grad():
        expected = wrapper(proprio, tactile).cpu().numpy()
    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )
    actual = session.run(
        ["action"],
        {
            "student_proprio_hist": proprio.numpy(),
            "student_tactile_hist": tactile.numpy(),
        },
    )[0]
    if actual.shape != expected.shape:
        raise RuntimeError(
            f"ONNX verification shape {actual.shape} differs from PyTorch {expected.shape}."
        )
    max_abs_error = float(abs(actual - expected).max())
    if not max_abs_error <= float(atol):
        raise RuntimeError(
            f"ONNX verification max_abs_error={max_abs_error:.6g} exceeds "
            f"atol={atol:.6g}."
        )
    return {
        "performed": True,
        "reference": "pytorch_eval_cpu",
        "runtime": "onnxruntime_cpu",
        "seed": 20260802,
        "batch_size": batch_size,
        "atol": float(atol),
        "max_abs_error": max_abs_error,
    }


def _try_resolve_canonical_task(task: str) -> str | None:
    """Resolve a tactile task while treating other task families as non-matches."""

    try:
        return resolve_tactile_task(task).canonical_name
    except ValueError:
        return None


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one artifact."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_yaml_atomic(path: Path, value: dict[str, Any]) -> None:
    """Write YAML beside the destination and atomically replace it."""

    with tempfile.NamedTemporaryFile(
        mode="w",
        prefix=f"{path.stem}_",
        suffix=path.suffix,
        dir=path.parent,
        encoding="utf-8",
        delete=False,
    ) as stream:
        temporary_path = Path(stream.name)
        yaml.safe_dump(value, stream, sort_keys=False, allow_unicode=True)
    try:
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _repo_root() -> Path:
    """Return the RevoLab repository root."""

    return Path(__file__).resolve().parents[3]


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load one YAML mapping from disk."""

    with path.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping in {path}.")
    return value


def _require_mapping(value: dict[str, Any], key: str) -> dict[str, Any]:
    """Return a required child mapping or raise a contract error."""

    result = value.get(key)
    if not isinstance(result, dict):
        raise ValueError(f"Missing mapping {key!r} in tactile export configuration.")
    return result
