from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


FINGER_ORDER = ("thumb", "index", "middle", "ring", "little")
TIP_MODULE_IDS = {
    "thumb": 1,
    "index": 3,
    "middle": 5,
    "ring": 7,
    "little": 9,
}
TIP_SENSOR_COUNTS = {
    "thumb": 31,
    "index": 21,
    "middle": 21,
    "ring": 21,
    "little": 21,
}
GRAPH_NODE_CHANNELS = 5
GRAPH_CONTEXT_CHANNELS = 4
TACTILE_STUDENT_OBSERVATION_SCHEMA = "hora_tactile_student_physical_graph_v1"
TACTILE_STUDENT_COMMAND_OBSERVATION_SCHEMA = (
    "hora_tactile_student_physical_graph_command_v2"
)
STUDENT_PROPRIO_FRAME_CHANNELS = {
    "joint_position_normalized": 21,
    "current_target_rad": 21,
    "raw_force": 0,
}


def student_proprio_frame_channels(
    public_command_channels: Sequence[str] = (),
) -> dict[str, int]:
    """Build ordered proprioceptive channel metadata for one task.

    Args:
        public_command_channels: Public scalar commands appended after joint state.

    Returns:
        Channel names and widths in student frame order.
    """
    channels = dict(STUDENT_PROPRIO_FRAME_CHANNELS)
    for name in public_command_channels:
        if name in channels:
            raise ValueError(f"Duplicate student proprio channel name: {name!r}.")
        channels[str(name)] = 1
    return channels


@dataclass(frozen=True)
class TactileTaskSpec:
    """Describe the observation and action fingers for one tactile task."""

    canonical_name: str
    default_observation_fingers: tuple[str, ...]
    action_fingers: tuple[str, ...]
    policy_rate_hz: float = 20.0
    aliases: tuple[str, ...] = ()
    public_command_channels: tuple[str, ...] = ()
    target_angvel_range_rad_s: tuple[float, float] | None = None

    def validate_observation_fingers(
        self, finger_names: Sequence[str]
    ) -> tuple[str, ...]:
        """Validate checkpoint observation fingers against this task.

        Training normally packs only action-enabled fingers. The explicit
        all-five sequence is also accepted for checkpoints trained with
        ``HORA_PACK_ACTIVE_TACTILE_ONLY=0``.

        Args:
            finger_names: Ordered tactile fingers stored in the training config.

        Returns:
            Validated observation fingers as a tuple.
        """

        observed = tuple(str(name) for name in finger_names)
        allowed = {self.default_observation_fingers, FINGER_ORDER}
        if observed not in allowed:
            raise ValueError(
                f"Task {self.canonical_name!r} expects observation fingers "
                f"{self.default_observation_fingers} or explicit all-five compatibility "
                f"{FINGER_ORDER}, got {observed}."
            )
        return observed

    def build_contract(self, observation_fingers: Sequence[str]) -> dict:
        """Build serializable task metadata for export and deployment checks.

        Args:
            observation_fingers: Ordered tactile fingers encoded by the student.

        Returns:
            Task identity, finger roles, module IDs, and expected dimensions.
        """

        observed = self.validate_observation_fingers(observation_fingers)
        total_nodes = sum(TIP_SENSOR_COUNTS[name] for name in observed)
        frame_dim = (
            total_nodes * GRAPH_NODE_CHANNELS
            + len(observed) * GRAPH_CONTEXT_CHANNELS
        )
        return {
            "canonical_name": self.canonical_name,
            "observation_fingers": list(observed),
            "action_fingers": list(self.action_fingers),
            "touch_module_ids": [TIP_MODULE_IDS[name] for name in observed],
            "policy_rate_hz": float(self.policy_rate_hz),
            "total_nodes": total_nodes,
            "tactile_frame_dim": frame_dim,
            "public_command_channels": list(self.public_command_channels),
            "student_proprio_frame_dim": 42 + len(self.public_command_channels),
            "target_angvel_range_rad_s": (
                list(self.target_angvel_range_rad_s)
                if self.target_angvel_range_rad_s is not None
                else None
            ),
            "all_five_observation_compatibility": (
                observed == FINGER_ORDER
                and observed != self.default_observation_fingers
            ),
        }


TACTILE_TASK_SPECS = (
    TactileTaskSpec(
        canonical_name="rotate_ball_tactile",
        default_observation_fingers=FINGER_ORDER,
        action_fingers=FINGER_ORDER,
        aliases=("BrainCo-Direct-Revo3-TactileRotate-Ball-v0",),
    ),
    TactileTaskSpec(
        canonical_name="rotate_cylinder_tactile",
        default_observation_fingers=FINGER_ORDER,
        action_fingers=FINGER_ORDER,
        aliases=("BrainCo-Direct-Revo3-TactileRotate-Cylinder-v0",),
    ),
    TactileTaskSpec(
        canonical_name="nutbolt_tactile",
        default_observation_fingers=("thumb", "index", "middle"),
        action_fingers=("thumb", "index", "middle"),
        aliases=(
            "BrainCo-Direct-Revo3-HoraNutBoltTactile-v0",
            "RevoHoraNutBoltTactile-v0",
        ),
    ),
    TactileTaskSpec(
        canonical_name="screwdriver_tactile",
        default_observation_fingers=("thumb", "index", "middle", "ring"),
        action_fingers=("thumb", "index", "middle", "ring"),
        aliases=(
            "BrainCo-Direct-Revo3-HoraScrewDriverTactile-v0",
            "RevoHoraScrewDriverTactile-v0",
        ),
    ),
    TactileTaskSpec(
        canonical_name="valvedriver_tactile",
        default_observation_fingers=FINGER_ORDER,
        action_fingers=FINGER_ORDER,
        aliases=(
            "vavledriver_tactile",
            "BrainCo-Direct-Revo3-HoraVavleDriverTactile-v0",
            "BrainCo-Direct-Revo3-HoraValveDriverTactile-v0",
            "RevoHoraVavleDriverTactile-v0",
            "RevoHoraValveDriverTactile-v0",
        ),
    ),
    TactileTaskSpec(
        canonical_name="valvedriver_tactile_40",
        default_observation_fingers=FINGER_ORDER,
        action_fingers=FINGER_ORDER,
        aliases=(
            "BrainCo-Direct-Revo3-HoraValveDriverTactile40-v0",
            "RevoHoraValveDriverTactile40-v0",
        ),
    ),
)

_TASK_BY_ALIAS = {
    alias.lower(): spec
    for spec in TACTILE_TASK_SPECS
    for alias in (spec.canonical_name, *spec.aliases)
}


def resolve_tactile_task(task: str) -> TactileTaskSpec:
    """Resolve a short task name, historical alias, or Gym ID.

    Args:
        task: User-supplied tactile task identifier.

    Returns:
        Canonical tactile task specification.
    """

    normalized = str(task).strip().lower()
    spec = _TASK_BY_ALIAS.get(normalized)
    if spec is None:
        supported = ", ".join(value.canonical_name for value in TACTILE_TASK_SPECS)
        raise ValueError(
            f"Unsupported tactile deployment task {task!r}. Supported tasks: {supported}."
        )
    return spec


def build_action_mask(
    policy_joint_order: Sequence[str], action_fingers: Sequence[str]
) -> list[float]:
    """Build a binary 21-joint mask from task-controlled fingers.

    Args:
        policy_joint_order: Joint names in model output order.
        action_fingers: Fingers whose action dimensions are enabled.

    Returns:
        Binary mask aligned with the policy joint order.
    """

    enabled = tuple(str(name) for name in action_fingers)
    return [
        1.0 if any(f"_{finger}_" in joint_name for finger in enabled) else 0.0
        for joint_name in policy_joint_order
    ]


def canonical_tactile_task_names() -> tuple[str, ...]:
    """Return canonical task names accepted by the deployment CLI.

    Returns:
        Ordered canonical task names.
    """

    return tuple(spec.canonical_name for spec in TACTILE_TASK_SPECS)
