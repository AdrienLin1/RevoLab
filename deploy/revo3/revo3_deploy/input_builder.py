from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence

import numpy as np

OBS_DIM = 126
HIST_LEN = 30
OBS_PER_STEP = 42
JOINT_DIM = 21
ACTION_SCALE = 1.0 / 24.0
TACTILE_HIST_LEN = 10
GRAPH_NODE_CHANNELS = 5
GRAPH_CONTEXT_CHANNELS = 4


class Stage2InputBuilder:
    """Build Stage-2 policy inputs from real robot joint positions."""

    def __init__(
        self,
        joint_lower: np.ndarray,
        joint_upper: np.ndarray,
        joint_names: tuple[str, ...] | list[str],
        action_scale: float = ACTION_SCALE,
    ) -> None:
        self.joint_names = tuple(joint_names)
        if len(self.joint_names) != JOINT_DIM:
            raise ValueError(f"joint_names must have {JOINT_DIM} entries.")
        if len(set(self.joint_names)) != JOINT_DIM:
            raise ValueError("joint_names contains duplicates.")

        self.joint_lower = self._as_vector(joint_lower, "joint_lower")
        self.joint_upper = self._as_vector(joint_upper, "joint_upper")
        if np.any(self.joint_upper <= self.joint_lower):
            raise ValueError("Every joint upper limit must be greater than lower.")

        self.action_scale = float(action_scale)
        self.current_target: np.ndarray | None = None
        self._frames: deque[np.ndarray] = deque(maxlen=HIST_LEN)

    def reset(self, joint_pos: np.ndarray, target: np.ndarray | None = None) -> dict[str, np.ndarray]:
        joint_pos = self._as_vector(joint_pos, "joint_pos")
        if target is None:
            target = joint_pos
        target = self._clip_target(self._as_vector(target, "target"))
        frame = self._build_frame(joint_pos, target)

        self.current_target = target.copy()
        self._frames.clear()
        for _ in range(HIST_LEN):
            self._frames.append(frame.copy())
        return self._get_policy_inputs()

    def observe(self, joint_pos: np.ndarray) -> dict[str, np.ndarray]:
        joint_pos = self._as_vector(joint_pos, "joint_pos")
        if self.current_target is None:
            return self.reset(joint_pos)
        self._frames.append(self._build_frame(joint_pos, self.current_target))
        return self._get_policy_inputs()

    def action_to_target(self, action: np.ndarray) -> np.ndarray:
        if self.current_target is None:
            raise RuntimeError("Call reset() before action_to_target().")
        action = np.clip(self._as_vector(action, "action"), -1.0, 1.0)
        self.current_target = self._clip_target(self.current_target + self.action_scale * action)
        return self.current_target.copy()

    def _get_policy_inputs(self) -> dict[str, np.ndarray]:
        if len(self._frames) != HIST_LEN:
            raise RuntimeError(f"History not ready: expected {HIST_LEN} frames.")
        hist = np.stack(list(self._frames), axis=0).astype(np.float32)
        return {
            "obs": hist[-3:].reshape(1, OBS_DIM),
            "proprio_hist": hist.reshape(1, HIST_LEN, OBS_PER_STEP),
        }

    def _build_frame(self, joint_pos: np.ndarray, target: np.ndarray) -> np.ndarray:
        q_norm = (2.0 * joint_pos - self.joint_upper - self.joint_lower) / (
            self.joint_upper - self.joint_lower
        )
        return np.concatenate([q_norm, target], axis=0).astype(np.float32)

    def _clip_target(self, target: np.ndarray) -> np.ndarray:
        return np.clip(target, self.joint_lower, self.joint_upper).astype(np.float32)

    @staticmethod
    def _as_vector(value, name: str) -> np.ndarray:
        vector = np.asarray(value, dtype=np.float32).reshape(-1)
        if vector.shape != (JOINT_DIM,):
            raise ValueError(f"{name} must have shape ({JOINT_DIM},), got {vector.shape}.")
        return vector


class TactileStudentInputBuilder:
    """Build TactileDAgger student inputs from joints and fingertip taxels.

    The implementation mirrors the physical-graph observation path in
    ``Revo3HandScrewTactileEnv``. Each tactile node contributes
    ``[contact, established, released, duration, eta]`` and each active finger
    appends ``[shift_x, shift_y, shift_valid, contact_ratio]``.

    Args:
        joint_lower: Policy-order lower joint limits used for joint normalization.
        joint_upper: Policy-order upper joint limits used for joint normalization.
        joint_names: Policy-order joint names.
        finger_names: Active fingers in the order expected by the ONNX model.
        module_ids: SDK touch module IDs matching ``finger_names``.
        sensor_positions: Per-finger physical taxel positions in official ID order.
        contact_threshold_on: Scalar or per-taxel threshold for contact establishment.
        contact_threshold_off: Scalar or per-taxel threshold for contact release.
        action_scale: Delta action scale applied to policy targets.
        proprio_history_len: Number of proprioceptive frames consumed by the student.
        tactile_history_len: Number of tactile frames consumed by the student.
        duration_tau: Contact duration time constant in control steps.
        duration_max: Contact duration upper reference in control steps.
        shift_ema_beta: EMA coefficient for contact-centroid motion.
        shift_max: Normalization range for one-step centroid motion.
        public_command_names: Ordered public scalar command channel names.
        public_command_values: Fixed command values appended to every proprio frame.
    """

    def __init__(
        self,
        joint_lower: np.ndarray,
        joint_upper: np.ndarray,
        joint_names: tuple[str, ...] | list[str],
        finger_names: Sequence[str],
        module_ids: Sequence[int],
        sensor_positions: Sequence[np.ndarray],
        contact_threshold_on: float | Sequence[float],
        contact_threshold_off: float | Sequence[float],
        action_scale: float = ACTION_SCALE,
        proprio_history_len: int = 3,
        tactile_history_len: int = TACTILE_HIST_LEN,
        duration_tau: float = 20.0,
        duration_max: float = 100.0,
        shift_ema_beta: float = 0.7,
        shift_max: float = 0.2,
        public_command_names: Sequence[str] = (),
        public_command_values: Sequence[float] = (),
    ) -> None:
        self.joint_names = tuple(joint_names)
        if len(self.joint_names) != JOINT_DIM or len(set(self.joint_names)) != JOINT_DIM:
            raise ValueError(f"joint_names must contain {JOINT_DIM} unique entries.")
        self.joint_lower = Stage2InputBuilder._as_vector(joint_lower, "joint_lower")
        self.joint_upper = Stage2InputBuilder._as_vector(joint_upper, "joint_upper")
        if np.any(self.joint_upper <= self.joint_lower):
            raise ValueError("Every joint upper limit must be greater than lower.")

        self.finger_names = tuple(str(name) for name in finger_names)
        self.module_ids = tuple(int(value) for value in module_ids)
        if not self.finger_names or len(self.finger_names) != len(self.module_ids):
            raise ValueError("finger_names and module_ids must have the same non-zero length.")
        if len(set(self.finger_names)) != len(self.finger_names):
            raise ValueError("finger_names contains duplicates.")
        if len(set(self.module_ids)) != len(self.module_ids):
            raise ValueError("module_ids contains duplicates.")

        if len(sensor_positions) != len(self.finger_names):
            raise ValueError("sensor_positions must contain one array per active finger.")
        self.sensor_positions = tuple(
            self._normalize_sensor_positions(value, name)
            for value, name in zip(sensor_positions, self.finger_names)
        )
        self.sensor_counts = tuple(int(value.shape[0]) for value in self.sensor_positions)
        self.total_nodes = int(sum(self.sensor_counts))
        self.tactile_frame_dim = (
            self.total_nodes * GRAPH_NODE_CHANNELS
            + len(self.finger_names) * GRAPH_CONTEXT_CHANNELS
        )

        self.contact_threshold_on = self._threshold_vector(
            contact_threshold_on, "contact_threshold_on"
        )
        self.contact_threshold_off = self._threshold_vector(
            contact_threshold_off, "contact_threshold_off"
        )
        if np.any(self.contact_threshold_off < 0.0):
            raise ValueError("contact_threshold_off must be non-negative.")
        if np.any(self.contact_threshold_off >= self.contact_threshold_on):
            raise ValueError("Every off threshold must be below its on threshold.")

        self.action_scale = float(action_scale)
        self.proprio_history_len = int(proprio_history_len)
        self.tactile_history_len = int(tactile_history_len)
        self.duration_tau = float(duration_tau)
        self.duration_max = float(duration_max)
        self.shift_ema_beta = float(shift_ema_beta)
        self.shift_max = float(shift_max)
        if self.proprio_history_len <= 0 or self.tactile_history_len <= 0:
            raise ValueError("History lengths must be positive.")
        if self.duration_tau <= 0.0 or self.duration_max <= 0.0:
            raise ValueError("duration_tau and duration_max must be positive.")
        if not 0.0 <= self.shift_ema_beta < 1.0:
            raise ValueError("shift_ema_beta must be in [0, 1).")
        if self.shift_max <= 0.0:
            raise ValueError("shift_max must be positive.")
        self.public_command_names = tuple(str(name) for name in public_command_names)
        self.public_command_values = np.asarray(
            tuple(float(value) for value in public_command_values),
            dtype=np.float32,
        )
        if len(set(self.public_command_names)) != len(self.public_command_names):
            raise ValueError("public_command_names contains duplicates.")
        if self.public_command_values.shape != (len(self.public_command_names),):
            raise ValueError(
                "public_command_values must contain one scalar per public command."
            )
        if not np.isfinite(self.public_command_values).all():
            raise ValueError("public_command_values must be finite.")
        self.proprio_frame_dim = 2 * JOINT_DIM + len(self.public_command_names)

        self.current_target: np.ndarray | None = None
        self._proprio_frames: deque[np.ndarray] = deque(maxlen=self.proprio_history_len)
        self._tactile_frames: deque[np.ndarray] = deque(maxlen=self.tactile_history_len)
        self._previous_contact = np.zeros(self.total_nodes, dtype=np.float32)
        self._hysteresis_contact = np.zeros(self.total_nodes, dtype=np.float32)
        self._contact_duration = np.zeros(self.total_nodes, dtype=np.float32)
        self._previous_centroid = np.zeros((len(self.finger_names), 2), dtype=np.float32)
        self._previous_contact_valid = np.zeros(len(self.finger_names), dtype=bool)
        self._shift_ema = np.zeros((len(self.finger_names), 2), dtype=np.float32)

    def reset(
        self,
        joint_pos: np.ndarray,
        touch_modules: Mapping[int, Sequence[float]],
        target: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Reset temporal state and repeat the first measured frame.

        Args:
            joint_pos: Measured positions in policy joint order.
            touch_modules: SDK module ID to raw taxel array mapping.
            target: Optional initial target in policy joint order.

        Returns:
            Batched ONNX inputs for the tactile student.
        """

        joint_pos = Stage2InputBuilder._as_vector(joint_pos, "joint_pos")
        if target is None:
            target = joint_pos
        target = self._clip_target(Stage2InputBuilder._as_vector(target, "target"))
        self.current_target = target.copy()
        self._reset_tactile_state()

        proprio_frame = self._build_proprio_frame(joint_pos, target)
        tactile_frame = self._build_tactile_frame(touch_modules)
        self._proprio_frames.clear()
        self._tactile_frames.clear()
        for _ in range(self.proprio_history_len):
            self._proprio_frames.append(proprio_frame.copy())
        for _ in range(self.tactile_history_len):
            self._tactile_frames.append(tactile_frame.copy())
        return self._get_policy_inputs()

    def observe(
        self,
        joint_pos: np.ndarray,
        touch_modules: Mapping[int, Sequence[float]],
    ) -> dict[str, np.ndarray]:
        """Append one measured proprioceptive and tactile frame.

        Args:
            joint_pos: Measured positions in policy joint order.
            touch_modules: SDK module ID to raw taxel array mapping.

        Returns:
            Batched ONNX inputs for the tactile student.
        """

        joint_pos = Stage2InputBuilder._as_vector(joint_pos, "joint_pos")
        if self.current_target is None:
            return self.reset(joint_pos, touch_modules)
        self._proprio_frames.append(self._build_proprio_frame(joint_pos, self.current_target))
        self._tactile_frames.append(self._build_tactile_frame(touch_modules))
        return self._get_policy_inputs()

    def action_to_target(self, action: np.ndarray) -> np.ndarray:
        """Advance the delta-action target and clamp it to policy joint limits.

        Args:
            action: Student action in policy joint order.

        Returns:
            Updated target in policy joint order.
        """

        if self.current_target is None:
            raise RuntimeError("Call reset() before action_to_target().")
        action = np.clip(Stage2InputBuilder._as_vector(action, "action"), -1.0, 1.0)
        self.current_target = self._clip_target(self.current_target + self.action_scale * action)
        return self.current_target.copy()

    def _get_policy_inputs(self) -> dict[str, np.ndarray]:
        if len(self._proprio_frames) != self.proprio_history_len:
            raise RuntimeError("Proprioceptive history is not ready.")
        if len(self._tactile_frames) != self.tactile_history_len:
            raise RuntimeError("Tactile history is not ready.")
        return {
            "student_proprio_hist": np.stack(self._proprio_frames, axis=0)[None].astype(
                np.float32
            ),
            "student_tactile_hist": np.stack(self._tactile_frames, axis=0)[None].astype(
                np.float32
            ),
        }

    def _build_proprio_frame(self, joint_pos: np.ndarray, target: np.ndarray) -> np.ndarray:
        q_norm = (2.0 * joint_pos - self.joint_upper - self.joint_lower) / (
            self.joint_upper - self.joint_lower
        )
        return np.concatenate(
            [q_norm, target, self.public_command_values], axis=0
        ).astype(np.float32)

    def _build_tactile_frame(
        self, touch_modules: Mapping[int, Sequence[float]]
    ) -> np.ndarray:
        raw = self._pack_touch_modules(touch_modules)
        contact_on = raw > self.contact_threshold_on
        contact_off = raw < self.contact_threshold_off
        contact = np.where(
            contact_on,
            1.0,
            np.where(contact_off, 0.0, self._hysteresis_contact),
        ).astype(np.float32)
        self._hysteresis_contact[:] = contact

        established = ((self._previous_contact < 0.5) & (contact > 0.5)).astype(np.float32)
        released = ((self._previous_contact > 0.5) & (contact < 0.5)).astype(np.float32)
        self._contact_duration = np.where(
            contact > 0.5,
            self._contact_duration + 1.0,
            0.0,
        ).astype(np.float32)
        duration = np.log1p(self._contact_duration / self.duration_tau)
        duration /= np.log1p(self.duration_max / self.duration_tau)
        duration = np.clip(duration, 0.0, 1.0).astype(np.float32)

        eta_chunks = []
        contexts = []
        start = 0
        for finger_index, (count, positions) in enumerate(
            zip(self.sensor_counts, self.sensor_positions)
        ):
            finger_contact = contact[start : start + count]
            contact_count = float(finger_contact.sum())
            contact_valid = contact_count > 0.0
            if contact_valid:
                centroid = (
                    finger_contact[:, None] * positions
                ).sum(axis=0) / max(contact_count, 1.0)
            else:
                centroid = np.zeros(2, dtype=np.float32)
            shift_valid = contact_valid and bool(self._previous_contact_valid[finger_index])
            if shift_valid:
                normalized_shift = np.clip(
                    (centroid - self._previous_centroid[finger_index]) / self.shift_max,
                    -1.0,
                    1.0,
                )
                shift_ema = (
                    self.shift_ema_beta * self._shift_ema[finger_index]
                    + (1.0 - self.shift_ema_beta) * normalized_shift
                )
            else:
                shift_ema = np.zeros(2, dtype=np.float32)
            self._shift_ema[finger_index] = shift_ema
            self._previous_centroid[finger_index] = centroid
            self._previous_contact_valid[finger_index] = contact_valid

            eta = np.sum((positions - centroid[None]) * shift_ema[None], axis=-1)
            if not shift_valid:
                eta.fill(0.0)
            eta_chunks.append(np.clip(eta, -1.0, 1.0).astype(np.float32))
            contexts.append(
                np.asarray(
                    [shift_ema[0], shift_ema[1], float(shift_valid), contact_count / count],
                    dtype=np.float32,
                )
            )
            start += count

        eta_all = np.concatenate(eta_chunks, axis=0)
        nodes = np.stack([contact, established, released, duration, eta_all], axis=-1)
        self._previous_contact[:] = contact
        frame = np.concatenate([nodes.reshape(-1), np.stack(contexts).reshape(-1)])
        if frame.shape != (self.tactile_frame_dim,):
            raise RuntimeError(
                f"Built tactile frame {frame.shape} != ({self.tactile_frame_dim},)."
            )
        return frame.astype(np.float32)

    def _pack_touch_modules(
        self, touch_modules: Mapping[int, Sequence[float]]
    ) -> np.ndarray:
        chunks = []
        for finger_name, module_id, expected_count in zip(
            self.finger_names, self.module_ids, self.sensor_counts
        ):
            if module_id not in touch_modules:
                raise ValueError(
                    f"Missing tactile module {module_id} for active finger {finger_name}."
                )
            values = np.asarray(touch_modules[module_id], dtype=np.float32).reshape(-1)
            if values.shape != (expected_count,):
                raise ValueError(
                    f"Tactile module {module_id} ({finger_name}) must contain "
                    f"{expected_count} values, got {values.shape}."
                )
            if not np.isfinite(values).all():
                raise ValueError(f"Tactile module {module_id} contains non-finite values.")
            chunks.append(values)
        return np.concatenate(chunks, axis=0)

    def _threshold_vector(
        self, value: float | Sequence[float], name: str
    ) -> np.ndarray:
        vector = np.asarray(value, dtype=np.float32).reshape(-1)
        if vector.shape == (1,):
            return np.full(self.total_nodes, float(vector[0]), dtype=np.float32)
        if vector.shape != (self.total_nodes,):
            raise ValueError(
                f"{name} must be scalar or contain {self.total_nodes} values, got {vector.shape}."
            )
        return vector

    @staticmethod
    def _normalize_sensor_positions(value: np.ndarray, finger_name: str) -> np.ndarray:
        positions = np.asarray(value, dtype=np.float32)
        if positions.ndim != 2 or positions.shape[0] <= 0 or positions.shape[1] != 2:
            raise ValueError(
                f"sensor_positions for {finger_name} must have shape (N, 2), got {positions.shape}."
            )
        if not np.isfinite(positions).all():
            raise ValueError(f"sensor_positions for {finger_name} contains non-finite values.")
        positions = positions - positions.mean(axis=0, keepdims=True)
        pairwise = positions[:, None] - positions[None, :]
        diameter = float(np.linalg.norm(pairwise, axis=-1).max())
        return (positions / max(diameter, 1.0e-6)).astype(np.float32)

    def _reset_tactile_state(self) -> None:
        self._previous_contact.fill(0.0)
        self._hysteresis_contact.fill(0.0)
        self._contact_duration.fill(0.0)
        self._previous_centroid.fill(0.0)
        self._previous_contact_valid.fill(False)
        self._shift_ema.fill(0.0)

    def _clip_target(self, target: np.ndarray) -> np.ndarray:
        return np.clip(target, self.joint_lower, self.joint_upper).astype(np.float32)
