from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import onnxruntime as ort
import yaml

from revo3_deploy.input_builder import TactileStudentInputBuilder
from revo3_deploy.robot_profile import JOINT_DIM, Revo3Profile
from revo3_deploy.sdk_hand_io import REVO3_TOUCH_MODULES
from revo3_deploy.task_registry import (
    TACTILE_STUDENT_COMMAND_OBSERVATION_SCHEMA,
    TACTILE_STUDENT_OBSERVATION_SCHEMA,
    TactileTaskSpec,
    build_action_mask,
    resolve_tactile_task,
    student_proprio_frame_channels,
)

TACTILE_LAYOUT_VERSION = "revo3_right_official_diagram_estimate_v5"


class Revo3TactilePolicyRunner:
    """Run a TactileDAgger ONNX policy with physical Revo3 taxel history."""

    def __init__(
        self,
        onnx_path: str | Path,
        policy_path: str | Path,
        profile: Revo3Profile,
        task: str,
        contact_threshold_on: float | list[float] | np.ndarray,
        contact_threshold_off: float | list[float] | np.ndarray,
        use_gpu: bool = False,
        target_angvel: float | None = None,
    ) -> None:
        """Load and validate a task-specific physical tactile policy.

        Args:
            onnx_path: Exported tactile student ONNX model.
            policy_path: Exported task and observation contract YAML.
            profile: Validated physical hand joint profile.
            task: Runtime task name, alias, or Gym ID.
            contact_threshold_on: Per-taxel or scalar contact-on threshold.
            contact_threshold_off: Per-taxel or scalar contact-off threshold.
            use_gpu: Whether to prefer ONNX Runtime CUDA execution.
            target_angvel: Optional command for a task contract that exposes it.
        """

        self.onnx_path = Path(onnx_path)
        self.policy_path = Path(policy_path)
        self.policy_cfg = self._load_policy_cfg(self.policy_path)
        self._validate_artifact_contract(self.policy_cfg, self.onnx_path)
        self.policy_contract = self._validate_policy_contract(self.policy_cfg)
        self.tactile_cfg = self._validate_tactile_contract(self.policy_cfg)
        self.task_spec = resolve_tactile_task(task)
        self.task_contract = self._validate_task_contract(
            self.policy_cfg,
            self.task_spec,
            self.tactile_cfg,
        )
        self.task_name = self.task_spec.canonical_name
        public_command_values = self._resolve_public_command_values(target_angvel)
        self._validate_observation_schema(self.policy_cfg, self.task_spec)
        self._validate_observation_io_alignment(
            self.policy_contract,
            self.tactile_cfg,
        )

        policy_order = tuple(self.policy_contract["joint_order_right_hand"])
        if policy_order != profile.policy_joint_order:
            raise ValueError("Policy contract joint order differs from robot profile.")
        if not np.isclose(
            float(self.policy_contract["action_scale"]),
            profile.action_scale,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError("Policy contract action_scale differs from robot profile.")
        self.profile = profile

        fingers = tuple(self.tactile_cfg["fingers"])
        self.finger_names = tuple(str(finger["name"]) for finger in fingers)
        self.module_ids = tuple(int(finger["module_id"]) for finger in fingers)
        self.sdk_channel_indices = {
            int(finger["module_id"]): np.asarray(
                finger["sdk_channel_ids_by_official_sensor_id"], dtype=np.int64
            )
            - 1
            for finger in fingers
        }
        self.sensor_mapping_verified = bool(
            self.tactile_cfg["sensor_mapping"]["verified"]
        )
        self.builder = TactileStudentInputBuilder(
            joint_lower=profile.joint_lower_policy,
            joint_upper=profile.joint_upper_policy,
            joint_names=profile.policy_joint_order,
            finger_names=self.finger_names,
            module_ids=self.module_ids,
            sensor_positions=[finger["sensor_positions_xy"] for finger in fingers],
            contact_threshold_on=contact_threshold_on,
            contact_threshold_off=contact_threshold_off,
            action_scale=profile.action_scale,
            proprio_history_len=int(self.tactile_cfg["proprio_history_len"]),
            tactile_history_len=int(self.tactile_cfg["history_len"]),
            duration_tau=float(self.tactile_cfg["duration_tau"]),
            duration_max=float(self.tactile_cfg["duration_max"]),
            shift_ema_beta=float(self.tactile_cfg["shift_ema_beta"]),
            shift_max=float(self.tactile_cfg["shift_max"]),
            public_command_names=self.task_spec.public_command_channels,
            public_command_values=public_command_values,
        )
        configured_mask = self.policy_contract.get("action_mask_right_hand")
        if configured_mask is None:
            configured_mask = build_action_mask(
                profile.policy_joint_order, self.task_spec.action_fingers
            )
        self.action_mask = np.asarray(configured_mask, dtype=np.float32).reshape(-1)
        if self.action_mask.shape != (JOINT_DIM,) or not np.isin(
            self.action_mask, (0.0, 1.0)
        ).all():
            raise ValueError("action_mask_right_hand must contain 21 binary values.")
        if int(self.action_mask.sum()) <= 0:
            raise ValueError("Tactile contract does not activate any policy joints.")
        expected_mask = np.asarray(
            build_action_mask(profile.policy_joint_order, self.task_spec.action_fingers),
            dtype=np.float32,
        )
        if not np.array_equal(self.action_mask, expected_mask):
            raise ValueError(
                "action_mask_right_hand does not match task-controlled action fingers."
            )
        if self.builder.tactile_frame_dim != int(self.tactile_cfg["frame_dim"]):
            raise ValueError(
                f"Physical builder frame {self.builder.tactile_frame_dim} != policy tactile.frame_dim "
                f"{self.tactile_cfg['frame_dim']}."
            )
        if self.builder.proprio_frame_dim != int(
            self.tactile_cfg["proprio_frame_dim"]
        ):
            raise ValueError(
                f"Physical builder proprio frame {self.builder.proprio_frame_dim} != "
                f"policy tactile.proprio_frame_dim {self.tactile_cfg['proprio_frame_dim']}."
            )
        self.initialized = False

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if use_gpu
            else ["CPUExecutionProvider"]
        )
        self.session = ort.InferenceSession(str(self.onnx_path), providers=providers)
        self.input_names = [meta.name for meta in self.session.get_inputs()]
        self.output_names = [meta.name for meta in self.session.get_outputs()]
        self.output_name = self._resolve_output_name()
        self._validate_onnx_io_contract()

    @property
    def rate_hz(self) -> float:
        """Return the policy control rate from the deploy contract."""

        return float(
            self.policy_contract.get("policy_rate_hz") or self.profile.default_rate_hz
        )

    def step(
        self,
        measured_policy_pos_rad: np.ndarray,
        touch_modules: dict[int, np.ndarray],
    ) -> np.ndarray:
        """Run one closed-loop tactile policy step.

        Args:
            measured_policy_pos_rad: Measured joints in policy order and radians.
            touch_modules: Physical SDK touch arrays keyed by module ID.

        Returns:
            Updated target positions in policy order and radians.
        """

        if not self.initialized:
            inputs = self.builder.reset(
                measured_policy_pos_rad, self._touch_in_layout_order(touch_modules)
            )
            self.initialized = True
        else:
            inputs = self.builder.observe(
                measured_policy_pos_rad, self._touch_in_layout_order(touch_modules)
            )
        action = self.session.run([self.output_name], self._build_ort_feed(inputs))[0][0]
        return self.builder.action_to_target(action * self.action_mask)

    def _resolve_public_command_values(
        self, target_angvel: float | None
    ) -> tuple[float, ...]:
        """Validate runtime command values required by the selected task.

        Args:
            target_angvel: Requested angular velocity for a commanded task.

        Returns:
            Public command values in the task registry's channel order.
        """
        channels = self.task_spec.public_command_channels
        if not channels:
            if target_angvel is not None:
                raise ValueError(
                    f"Task {self.task_name!r} does not accept --target-angvel."
                )
            return ()
        if channels != ("target_angular_velocity_rad_s",):
            raise ValueError(f"Unsupported public command contract: {channels}.")
        if target_angvel is None:
            raise ValueError(
                f"Task {self.task_name!r} requires target_angvel in rad/s."
            )
        value = float(target_angvel)
        bounds = self.task_spec.target_angvel_range_rad_s
        if bounds is None:
            raise ValueError(
                f"Task {self.task_name!r} has no target angular velocity bounds."
            )
        if not bounds[0] <= value <= bounds[1]:
            raise ValueError(
                f"target_angvel must be in [{bounds[0]}, {bounds[1]}] rad/s, "
                f"got {value}."
            )
        return (value,)

    @staticmethod
    def _load_policy_cfg(path: Path) -> dict:
        """Load one deploy policy YAML mapping."""

        with path.open("r", encoding="utf-8") as stream:
            return yaml.safe_load(stream) or {}

    @staticmethod
    def _validate_artifact_contract(policy_cfg: dict, onnx_path: Path) -> None:
        """Verify that the ONNX file is the artifact bound to policy YAML.

        Args:
            policy_cfg: Loaded deploy policy metadata.
            onnx_path: ONNX artifact selected for this runtime.
        """

        artifacts = policy_cfg.get("artifacts")
        if not isinstance(artifacts, dict):
            raise ValueError("Tactile policy metadata is missing artifacts.")
        verification = policy_cfg.get("verification")
        if not isinstance(verification, dict) or verification.get("performed") is not True:
            raise ValueError(
                "Tactile policy metadata does not contain a successful ONNX verification."
            )
        atol = float(verification.get("atol", -1.0))
        max_abs_error = float(verification.get("max_abs_error", float("inf")))
        if not np.isfinite(atol) or atol < 0.0:
            raise ValueError("Tactile ONNX verification atol is invalid.")
        if not np.isfinite(max_abs_error) or max_abs_error > atol:
            raise ValueError(
                f"Tactile ONNX verification error {max_abs_error} exceeds atol {atol}."
            )
        expected_digest = str(artifacts.get("onnx_sha256") or "").lower()
        if len(expected_digest) != 64 or any(
            character not in "0123456789abcdef" for character in expected_digest
        ):
            raise ValueError(
                "policy.yaml artifacts.onnx_sha256 is missing or invalid; re-export "
                "the task-aware ONNX policy."
            )
        if not onnx_path.is_file():
            raise FileNotFoundError(f"Tactile ONNX artifact not found: {onnx_path}")
        digest = hashlib.sha256()
        with onnx_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        actual_digest = digest.hexdigest()
        if actual_digest != expected_digest:
            raise ValueError(
                "Selected ONNX SHA-256 differs from policy.yaml; do not mix artifacts "
                "from different tasks or exports."
            )

    @staticmethod
    def _validate_policy_contract(policy_cfg: dict) -> dict:
        """Validate common ONNX inputs, outputs, joints, and action semantics."""

        contract = policy_cfg.get("io_contract")
        if not isinstance(contract, dict):
            raise ValueError("policy.yaml must contain io_contract.")
        for key in ("inputs", "outputs", "joint_order_right_hand", "action_scale"):
            if key not in contract:
                raise ValueError(f"policy.yaml io_contract missing {key}.")
        joint_order = list(contract["joint_order_right_hand"])
        if len(joint_order) != JOINT_DIM or len(set(joint_order)) != JOINT_DIM:
            raise ValueError("joint_order_right_hand must contain 21 unique joints.")
        if str(contract.get("action_semantics", "delta")) != "delta":
            raise ValueError("Only delta action_semantics is supported.")
        if not np.isfinite(float(contract["action_scale"])) or float(
            contract["action_scale"]
        ) <= 0.0:
            raise ValueError("io_contract.action_scale must be positive and finite.")
        expected_inputs = ["student_proprio_hist", "student_tactile_hist"]
        configured_inputs = [str(value.get("name")) for value in contract["inputs"]]
        if configured_inputs != expected_inputs:
            raise ValueError(
                f"Tactile policy inputs must be {expected_inputs}, got {configured_inputs}."
            )
        return contract

    @staticmethod
    def _validate_tactile_contract(policy_cfg: dict) -> dict:
        """Validate physical sensor layout and channel metadata."""

        tactile = policy_cfg.get("tactile")
        if not isinstance(tactile, dict):
            raise ValueError("Tactile policy metadata must contain tactile.")
        if tactile.get("layout") != "estimated_official":
            raise ValueError("Only estimated_official physical-node tactile policies are supported.")
        if tactile.get("layout_version") != TACTILE_LAYOUT_VERSION:
            raise ValueError(
                f"Tactile policy layout_version must be {TACTILE_LAYOUT_VERSION!r}."
            )
        for key in (
            "fingers",
            "proprio_history_len",
            "proprio_frame_dim",
            "history_len",
            "frame_dim",
            "total_nodes",
            "duration_tau",
            "duration_max",
            "shift_ema_beta",
            "shift_max",
            "sensor_mapping",
        ):
            if key not in tactile:
                raise ValueError(f"Tactile policy metadata missing tactile.{key}.")
        fingers = tactile["fingers"]
        if not isinstance(fingers, list) or not fingers:
            raise ValueError("tactile.fingers must be a non-empty list.")
        for finger in fingers:
            for key in (
                "name",
                "module_id",
                "module_name",
                "sensor_count",
                "official_sensor_ids",
                "sensor_positions_xy",
            ):
                if key not in finger:
                    raise ValueError(f"Tactile finger metadata missing {key}.")
            module_id = int(finger["module_id"])
            if module_id not in REVO3_TOUCH_MODULES:
                raise ValueError(f"Invalid Revo3 touch module ID {module_id}.")
            expected_name, expected_count = REVO3_TOUCH_MODULES[module_id]
            finger_name = str(finger["name"])
            if expected_name != f"{finger_name}_tip":
                raise ValueError(
                    f"Finger {finger_name!r} cannot use Revo3 module {module_id} "
                    f"({expected_name!r})."
                )
            if str(finger["module_name"]) != expected_name:
                raise ValueError(
                    f"Module {module_id} name {finger['module_name']!r} != {expected_name!r}."
                )
            if int(finger["sensor_count"]) != expected_count:
                raise ValueError(
                    f"Module {module_id} sensor_count must be {expected_count}."
                )
            official_ids = [int(value) for value in finger["official_sensor_ids"]]
            if official_ids != list(range(1, expected_count + 1)):
                raise ValueError(
                    f"Module {module_id} official_sensor_ids must be 1..{expected_count}."
                )
            positions = np.asarray(finger["sensor_positions_xy"], dtype=np.float32)
            if positions.shape != (expected_count, 2):
                raise ValueError(
                    f"Module {module_id} sensor_positions_xy must be ({expected_count}, 2)."
                )
            channels = [
                int(value)
                for value in finger.get("sdk_channel_ids_by_official_sensor_id") or ()
            ]
            if sorted(channels) != list(range(1, expected_count + 1)):
                raise ValueError(
                    f"Module {module_id} SDK channel mapping must be a permutation of "
                    f"1..{expected_count}."
                )
        sensor_mapping = tactile["sensor_mapping"]
        if not isinstance(sensor_mapping, dict):
            raise ValueError("tactile.sensor_mapping must be a mapping.")
        for key in ("verified", "source", "mapping_semantics"):
            if key not in sensor_mapping:
                raise ValueError(f"tactile.sensor_mapping missing {key}.")
        if not isinstance(sensor_mapping["verified"], bool):
            raise ValueError("tactile.sensor_mapping.verified must be boolean.")
        total_nodes = sum(int(finger["sensor_count"]) for finger in fingers)
        if total_nodes != int(tactile["total_nodes"]):
            raise ValueError(
                f"tactile.total_nodes={tactile['total_nodes']} != finger total {total_nodes}."
            )
        return tactile

    @staticmethod
    def _validate_task_contract(
        policy_cfg: dict,
        task_spec: TactileTaskSpec,
        tactile: dict,
    ) -> dict:
        """Validate requested task semantics against exported tactile metadata.

        Args:
            policy_cfg: Loaded deploy policy metadata.
            task_spec: Canonical specification resolved from runtime ``--task``.
            tactile: Already validated tactile observation metadata.

        Returns:
            Validated serialized task contract.
        """

        contract = policy_cfg.get("task_contract")
        if not isinstance(contract, dict):
            raise ValueError(
                "Tactile policy metadata is missing task_contract; re-export with "
                "--task using the task-aware exporter."
            )
        export_cfg = policy_cfg.get("export")
        if not isinstance(export_cfg, dict):
            raise ValueError("Tactile policy metadata is missing export metadata.")
        exported_task = resolve_tactile_task(str(export_cfg.get("task", "")))
        if exported_task.canonical_name != task_spec.canonical_name:
            raise ValueError(
                f"policy.yaml export.task {exported_task.canonical_name!r} differs from "
                f"runtime task {task_spec.canonical_name!r}."
            )
        canonical_name = str(contract.get("canonical_name", ""))
        if canonical_name != task_spec.canonical_name:
            raise ValueError(
                f"Runtime task {task_spec.canonical_name!r} differs from exported task "
                f"{canonical_name!r}."
            )
        observation_fingers = task_spec.validate_observation_fingers(
            contract.get("observation_fingers") or ()
        )
        action_fingers = tuple(
            str(value) for value in contract.get("action_fingers") or ()
        )
        if action_fingers != task_spec.action_fingers:
            raise ValueError(
                f"Exported action fingers {action_fingers} differ from task "
                f"{task_spec.canonical_name!r} action fingers {task_spec.action_fingers}."
            )
        tactile_fingers = tuple(str(value["name"]) for value in tactile["fingers"])
        if tactile_fingers != observation_fingers:
            raise ValueError(
                f"task_contract observation fingers {observation_fingers} differ from "
                f"tactile.fingers {tactile_fingers}."
            )
        expected = task_spec.build_contract(observation_fingers)
        for key in (
            "total_nodes",
            "tactile_frame_dim",
            "student_proprio_frame_dim",
        ):
            if int(contract.get(key, -1)) != int(expected[key]):
                raise ValueError(
                    f"task_contract.{key}={contract.get(key)} differs from expected "
                    f"{expected[key]}."
                )
        public_commands = list(contract.get("public_command_channels") or ())
        if public_commands != expected["public_command_channels"]:
            raise ValueError(
                "task_contract.public_command_channels differs from the requested "
                f"task: {public_commands} != {expected['public_command_channels']}."
            )
        target_range = contract.get("target_angvel_range_rad_s")
        expected_target_range = expected["target_angvel_range_rad_s"]
        if target_range is None or expected_target_range is None:
            ranges_match = target_range is None and expected_target_range is None
        else:
            ranges_match = bool(
                np.allclose(
                    np.asarray(target_range, dtype=np.float64),
                    np.asarray(expected_target_range, dtype=np.float64),
                    rtol=0.0,
                    atol=1.0e-12,
                )
            )
        if not ranges_match:
            raise ValueError(
                "task_contract.target_angvel_range_rad_s differs from the requested "
                f"task: {target_range} != {expected_target_range}."
            )
        if not np.isclose(
            float(contract.get("policy_rate_hz", -1.0)),
            float(expected["policy_rate_hz"]),
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError(
                f"task_contract.policy_rate_hz={contract.get('policy_rate_hz')} differs "
                f"from task rate {expected['policy_rate_hz']}."
            )
        if not np.isclose(
            float(policy_cfg["io_contract"].get("policy_rate_hz", -1.0)),
            float(expected["policy_rate_hz"]),
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError("io_contract.policy_rate_hz differs from the task contract.")
        module_ids = [int(value) for value in contract.get("touch_module_ids") or ()]
        expected_modules = [int(value) for value in expected["touch_module_ids"]]
        tactile_modules = [int(value["module_id"]) for value in tactile["fingers"]]
        if module_ids != expected_modules or module_ids != tactile_modules:
            raise ValueError(
                "task_contract touch_module_ids do not match task observation fingers "
                "and tactile module metadata."
            )
        if int(tactile["total_nodes"]) != int(expected["total_nodes"]):
            raise ValueError("tactile.total_nodes differs from the requested task contract.")
        if int(tactile["frame_dim"]) != int(expected["tactile_frame_dim"]):
            raise ValueError("tactile.frame_dim differs from the requested task contract.")
        if int(tactile["proprio_frame_dim"]) != int(
            expected["student_proprio_frame_dim"]
        ):
            raise ValueError(
                "tactile.proprio_frame_dim differs from the requested task contract."
            )
        return contract

    @staticmethod
    def _validate_observation_io_alignment(contract: dict, tactile: dict) -> None:
        """Cross-check ONNX tensor metadata against temporal tactile semantics.

        Args:
            contract: Validated policy IO contract.
            tactile: Validated physical tactile observation metadata.
        """

        batch_dim: str | int = "B" if bool(contract.get("dynamic_batch", True)) else 1
        expected_inputs = [
            [
                batch_dim,
                int(tactile["proprio_history_len"]),
                int(tactile["proprio_frame_dim"]),
            ],
            [batch_dim, int(tactile["history_len"]), int(tactile["frame_dim"])],
        ]
        configured_inputs = contract.get("inputs")
        if not isinstance(configured_inputs, list) or len(configured_inputs) != 2:
            raise ValueError("Tactile io_contract must contain exactly two inputs.")
        for configured, expected in zip(configured_inputs, expected_inputs):
            if list(configured.get("shape") or ()) != expected:
                raise ValueError(
                    f"io_contract input {configured.get('name')!r} shape "
                    f"{configured.get('shape')} differs from tactile observation {expected}."
                )
        configured_outputs = contract.get("outputs")
        expected_output = [batch_dim, JOINT_DIM]
        if not isinstance(configured_outputs, list) or len(configured_outputs) != 1:
            raise ValueError("Tactile io_contract must contain exactly one output.")
        if list(configured_outputs[0].get("shape") or ()) != expected_output:
            raise ValueError(
                f"io_contract action shape {configured_outputs[0].get('shape')} differs "
                f"from expected {expected_output}."
            )

    @staticmethod
    def _validate_observation_schema(
        policy_cfg: dict, task_spec: TactileTaskSpec | None = None
    ) -> None:
        """Reject teacher/adaptation observations presented as student inputs.

        Args:
            policy_cfg: Loaded task-aware tactile policy metadata.
            task_spec: Resolved task, inferred from export metadata when omitted.
        """

        if task_spec is None:
            export_cfg = policy_cfg.get("export")
            if not isinstance(export_cfg, dict):
                raise ValueError("Tactile policy metadata is missing export metadata.")
            task_spec = resolve_tactile_task(str(export_cfg.get("task", "")))
        schema = policy_cfg.get("observation_contract")
        if not isinstance(schema, dict):
            raise ValueError("Tactile policy metadata is missing observation_contract.")
        expected_schema = (
            TACTILE_STUDENT_COMMAND_OBSERVATION_SCHEMA
            if task_spec.public_command_channels
            else TACTILE_STUDENT_OBSERVATION_SCHEMA
        )
        if schema.get("schema") != expected_schema:
            raise ValueError(
                "Unsupported tactile student observation schema; re-export the checkpoint."
            )
        proprio = schema.get("student_proprio")
        tactile = schema.get("student_tactile")
        if not isinstance(proprio, dict) or not isinstance(tactile, dict):
            raise ValueError(
                "observation_contract must describe student_proprio and student_tactile."
            )
        expected_frame_dim = 42 + len(task_spec.public_command_channels)
        if int(proprio.get("history_len", -1)) != 3 or int(
            proprio.get("frame_dim", -1)
        ) != expected_frame_dim:
            raise ValueError(
                "Tactile student proprio contract must be "
                f"[B,3,{expected_frame_dim}] for task {task_spec.canonical_name!r}."
            )
        expected_channels = student_proprio_frame_channels(
            task_spec.public_command_channels
        )
        if proprio.get("frame_channels") != expected_channels:
            raise ValueError(
                "Tactile student proprio channels differ from the requested task contract."
            )
        if proprio.get("contains_raw_3d_force") is not False:
            raise ValueError("Tactile student proprio must not contain raw 3D force.")
        if tactile.get("representation") != "derived_public_contact_graph":
            raise ValueError("Unsupported tactile student representation.")
        if tactile.get("contains_raw_3d_force") is not False:
            raise ValueError("Tactile student input must not contain raw 3D force.")

    def _touch_in_layout_order(
        self, touch_modules: dict[int, np.ndarray]
    ) -> dict[int, np.ndarray]:
        """Reorder each SDK module array into diagram sensor-ID order."""

        ordered = {}
        for module_id, indices in self.sdk_channel_indices.items():
            if module_id not in touch_modules:
                raise ValueError(f"Missing tactile module {module_id} from SDK snapshot.")
            values = np.asarray(touch_modules[module_id], dtype=np.float32).reshape(-1)
            if values.shape != indices.shape:
                raise ValueError(
                    f"Tactile module {module_id} contains {values.shape[0]} SDK channels; "
                    f"mapping expects {indices.shape[0]}."
                )
            ordered[module_id] = values[indices]
        return ordered

    def _build_ort_feed(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Select builder tensors in the exact ONNX input-name order."""

        return {name: inputs[name] for name in self.input_names}

    def _resolve_output_name(self) -> str:
        """Resolve the configured action tensor from ONNX outputs."""

        configured = [
            str(value.get("name"))
            for value in self.policy_contract.get("outputs", [])
            if value.get("name")
        ]
        for candidate in configured + ["action"]:
            if candidate in self.output_names:
                return candidate
        if len(self.output_names) == 1:
            return self.output_names[0]
        raise RuntimeError(f"Could not resolve action output from {self.output_names}.")

    def _validate_onnx_io_contract(self) -> None:
        """Validate live ONNX tensor metadata against policy YAML."""

        expected_inputs = [
            str(value.get("name")) for value in self.policy_contract.get("inputs", [])
        ]
        expected_outputs = [
            str(value.get("name")) for value in self.policy_contract.get("outputs", [])
        ]
        if expected_inputs != self.input_names:
            raise ValueError(
                f"ONNX inputs {self.input_names} do not match policy.yaml {expected_inputs}."
            )
        if expected_outputs != self.output_names:
            raise ValueError(
                f"ONNX outputs {self.output_names} do not match policy.yaml {expected_outputs}."
            )
        configured_inputs = self.policy_contract.get("inputs", [])
        for meta, configured in zip(self.session.get_inputs(), configured_inputs):
            self._validate_tensor_meta(meta, configured, "input")
        configured_outputs = self.policy_contract.get("outputs", [])
        for meta, configured in zip(self.session.get_outputs(), configured_outputs):
            self._validate_tensor_meta(meta, configured, "output")

    @staticmethod
    def _validate_tensor_meta(meta, configured: dict, kind: str) -> None:
        """Validate one ONNX tensor shape and dtype."""

        expected_shape = list(configured.get("shape") or [])
        actual_shape = list(meta.shape)
        if len(expected_shape) != len(actual_shape):
            raise ValueError(
                f"ONNX {kind} {meta.name} rank {actual_shape} != policy.yaml {expected_shape}."
            )
        for index, (actual, expected) in enumerate(zip(actual_shape, expected_shape)):
            if index == 0 and (expected == "B" or isinstance(actual, str) or actual is None):
                continue
            if int(actual) != int(expected):
                raise ValueError(
                    f"ONNX {kind} {meta.name} shape {actual_shape} != policy.yaml "
                    f"{expected_shape}."
                )
        if str(configured.get("dtype")) != "float32" or meta.type != "tensor(float)":
            raise ValueError(
                f"ONNX {kind} {meta.name} must be float32; model={meta.type}, "
                f"policy={configured.get('dtype')}."
            )
