"""Derive handover events, phases, and paper-level attention summaries."""

from __future__ import annotations

import numpy as np


PHASE_NAMES = (
    "stable_rotation",
    "source_release",
    "new_finger_takeover",
    "regrasp_complete",
)


def tactile_finger_feature_slices(
    frame_dim: int,
    num_fingers: int,
    *,
    sensor_counts: tuple[int, ...] | list[int] | None = None,
    node_channels: int | None = None,
    context_channels: int = 0,
) -> tuple[tuple[slice, ...], ...]:
    """Return frame slices belonging to each tactile finger.

    Regular-grid frames store one contiguous, equal-width block per finger.
    Physical-node graph frames instead store all variable-width node blocks
    first and append one fixed-width context block per finger.

    Args:
        frame_dim: Total width of one tactile frame.
        num_fingers: Number of tactile finger slots.
        sensor_counts: Optional physical-node count for every finger.
        node_channels: Channels stored for each physical node.
        context_channels: Appended context channels for each finger.

    Returns:
        Per-finger tuples of one or more non-overlapping frame slices.
    """
    frame_dim = int(frame_dim)
    num_fingers = int(num_fingers)
    context_channels = int(context_channels)
    if frame_dim <= 0 or num_fingers <= 0:
        raise ValueError("frame_dim and num_fingers must be positive.")

    if sensor_counts is None:
        finger_width, remainder = divmod(frame_dim, num_fingers)
        if remainder:
            raise ValueError(
                f"Regular tactile frame width {frame_dim} is not divisible by "
                f"num_fingers={num_fingers}."
            )
        return tuple(
            (slice(finger * finger_width, (finger + 1) * finger_width),)
            for finger in range(num_fingers)
        )

    counts = tuple(int(count) for count in sensor_counts)
    if len(counts) != num_fingers or any(count <= 0 for count in counts):
        raise ValueError(
            "sensor_counts must contain one positive node count per finger."
        )
    if node_channels is None or int(node_channels) <= 0:
        raise ValueError("node_channels must be positive for a graph tactile frame.")
    if context_channels < 0:
        raise ValueError("context_channels must be non-negative.")

    node_channels = int(node_channels)
    node_width = sum(counts) * node_channels
    expected_dim = node_width + num_fingers * context_channels
    if frame_dim != expected_dim:
        raise ValueError(
            f"Graph tactile frame width {frame_dim} does not match nodes "
            f"({sum(counts)} * {node_channels}) plus context "
            f"({num_fingers} * {context_channels}) = {expected_dim}."
        )

    result = []
    node_start = 0
    for finger, count in enumerate(counts):
        node_stop = node_start + count * node_channels
        slices = [slice(node_start, node_stop)]
        if context_channels:
            context_start = node_width + finger * context_channels
            slices.append(slice(context_start, context_start + context_channels))
        result.append(tuple(slices))
        node_start = node_stop
    return tuple(result)


def assign_rotation_cycle_labels(
    angular_position: np.ndarray,
    episode_index: np.ndarray,
    target_rotation: float = 2.0 * np.pi,
) -> np.ndarray:
    """Assign complete positive-rotation cycles inside episode boundaries.

    Each cycle starts at the previous cycle endpoint and ends at the first
    sample whose unwrapped angular position advances by ``target_rotation``.
    Samples outside complete cycles retain label ``-1``.

    Args:
        angular_position: Unwrapped object angle shaped ``(time,)``.
        episode_index: Episode identifier aligned with the angle sequence.
        target_rotation: Positive angular displacement defining one cycle.

    Returns:
        Integer cycle label for each sample, with ``-1`` for incomplete spans.
    """
    angle = np.asarray(angular_position, dtype=np.float64)
    episodes = np.asarray(episode_index)
    if angle.ndim != 1 or episodes.shape != angle.shape:
        raise ValueError("angular_position and episode_index must be aligned one-dimensional arrays.")
    if float(target_rotation) <= 0.0:
        raise ValueError("target_rotation must be positive.")

    labels = np.full(len(angle), -1, dtype=np.int64)
    next_cycle = 0
    boundaries = np.flatnonzero(np.r_[True, episodes[1:] != episodes[:-1], True])
    for episode_start, episode_stop in zip(boundaries[:-1], boundaries[1:]):
        cycle_start = int(episode_start)
        while cycle_start < int(episode_stop) - 1:
            relative = angle[cycle_start:int(episode_stop)] - angle[cycle_start]
            reached = np.flatnonzero(relative >= float(target_rotation))
            if len(reached) == 0:
                break
            cycle_stop = cycle_start + int(reached[0])
            if cycle_stop <= cycle_start:
                break
            labels[cycle_start:cycle_stop] = next_cycle
            next_cycle += 1
            cycle_start = cycle_stop
    return labels


def select_conditioned_donors(
    phase_labels: np.ndarray,
    matching_features: np.ndarray,
    group_ids: np.ndarray,
    query_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Select nearest phase-matched donors from independent trajectories.

    Matching features are standardized before Euclidean nearest-neighbor
    search. Donors must share the query phase and have a different group id,
    where a group should identify one episode trajectory.

    Args:
        phase_labels: Integer phase id for every candidate sample.
        matching_features: Candidate covariates shaped ``(samples, features)``.
        group_ids: Episode-trajectory id for every candidate sample.
        query_indices: Candidate-array indices requiring matched donors.

    Returns:
        Donor indices and standardized matching distances for each query.
    """
    phases = np.asarray(phase_labels)
    features = np.asarray(matching_features, dtype=np.float64)
    groups = np.asarray(group_ids)
    queries = np.asarray(query_indices, dtype=np.int64)
    if phases.ndim != 1 or groups.shape != phases.shape:
        raise ValueError("phase_labels and group_ids must be aligned one-dimensional arrays.")
    if features.ndim != 2 or features.shape[0] != len(phases):
        raise ValueError("matching_features must have shape (samples, features).")
    if np.any((queries < 0) | (queries >= len(phases))):
        raise ValueError("query_indices contains an out-of-range sample index.")

    scale = np.nanstd(features, axis=0)
    scale[~np.isfinite(scale) | (scale < 1.0e-8)] = 1.0
    center = np.nanmedian(features, axis=0)
    standardized = np.nan_to_num((features - center) / scale, nan=0.0)
    donors = np.empty(len(queries), dtype=np.int64)
    distances = np.empty(len(queries), dtype=np.float64)

    for output_index, query_index in enumerate(queries):
        valid = (phases == phases[query_index]) & (groups != groups[query_index])
        candidates = np.flatnonzero(valid)
        if len(candidates) == 0:
            raise ValueError(
                f"No independent donor is available for phase {int(phases[query_index])}; "
                "collect more environments, episodes, or seeds."
            )
        delta = standardized[candidates] - standardized[query_index]
        squared_distance = np.einsum("ij,ij->i", delta, delta)
        nearest = int(np.argmin(squared_distance))
        donors[output_index] = int(candidates[nearest])
        distances[output_index] = float(np.sqrt(squared_distance[nearest]))
    return donors, distances


def detect_handover_events(
    contact_state: np.ndarray,
    max_gap: int = 8,
) -> np.ndarray:
    """Detect directed release-to-acquisition handover events.

    An event at time ``t`` on ``events[t, target, source]`` means the source
    finger released within the previous ``max_gap`` steps and a different
    target finger established contact at ``t``.

    Args:
        contact_state: Binary array shaped ``(time, fingers)``.
        max_gap: Maximum release-to-acquisition interval in control steps.

    Returns:
        Boolean event tensor shaped ``(time, target, source)``.
    """
    contact = np.asarray(contact_state, dtype=bool)
    if contact.ndim != 2:
        raise ValueError("contact_state must have shape (time, fingers).")
    time_steps, num_fingers = contact.shape
    events = np.zeros((time_steps, num_fingers, num_fingers), dtype=bool)
    previous = np.vstack([np.zeros((1, num_fingers), dtype=bool), contact[:-1]])
    acquired = contact & ~previous
    released = ~contact & previous
    last_release = np.full(num_fingers, -10 * max(1, int(max_gap)), dtype=np.int64)

    for time_index in range(time_steps):
        last_release[released[time_index]] = time_index
        targets = np.flatnonzero(acquired[time_index])
        for target in targets:
            recent = (time_index - last_release <= max_gap) & (last_release <= time_index)
            recent[target] = False
            events[time_index, target, recent] = True
    return events


def build_phase_labels(
    contact_state: np.ndarray,
    handover_events: np.ndarray | None = None,
    max_gap: int = 8,
    takeover_steps: int = 4,
    regrasp_steps: int = 6,
) -> np.ndarray:
    """Assign the four paper phases around every detected handover.

    Stable rotation is the default. The interval from source release to target
    acquisition is labeled release, followed by fixed takeover and completed
    regrasp windows. Later events overwrite earlier recovery windows so a new
    release is never hidden by an old event.

    Args:
        contact_state: Binary array shaped ``(time, fingers)``.
        handover_events: Optional precomputed event tensor.
        max_gap: Release-to-acquisition matching window.
        takeover_steps: Steps assigned to new-finger takeover after acquisition.
        regrasp_steps: Steps assigned to completed regrasp after takeover.

    Returns:
        Integer phase id for every time step.
    """
    contact = np.asarray(contact_state, dtype=bool)
    events = (
        detect_handover_events(contact, max_gap=max_gap)
        if handover_events is None
        else np.asarray(handover_events, dtype=bool)
    )
    if events.shape != (contact.shape[0], contact.shape[1], contact.shape[1]):
        raise ValueError("handover_events shape must be (time, fingers, fingers).")

    labels = np.zeros(contact.shape[0], dtype=np.int8)
    previous = np.vstack([np.zeros((1, contact.shape[1]), dtype=bool), contact[:-1]])
    release_times = np.argwhere(~contact & previous)

    matched = []
    for acquire_time, target, source in np.argwhere(events):
        candidates = release_times[
            (release_times[:, 1] == source)
            & (release_times[:, 0] <= acquire_time)
            & (release_times[:, 0] >= acquire_time - max_gap)
        ]
        release_time = int(candidates[-1, 0]) if len(candidates) else int(acquire_time)
        matched.append((release_time, int(acquire_time), int(source), int(target)))

    for release_time, acquire_time, source, target in matched:
        del source, target
        labels[release_time:acquire_time] = 1
        takeover_end = min(len(labels), acquire_time + max(1, int(takeover_steps)))
        labels[acquire_time:takeover_end] = 2
        regrasp_end = min(len(labels), takeover_end + max(1, int(regrasp_steps)))
        labels[takeover_end:regrasp_end] = 3

    # Reapply release spans last so overlapping handovers preserve the most
    # semantically urgent phase.
    for release_time, acquire_time, source, target in matched:
        del source, target
        labels[release_time:acquire_time] = 1
    return labels


def phase_attention_means(
    attention: np.ndarray,
    phase_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Average directed attention separately for each handover phase.

    Args:
        attention: Array shaped ``(time, target, source)``.
        phase_labels: Integer phase id for every time step.

    Returns:
        Phase means shaped ``(4, fingers, fingers)`` and sample counts ``(4,)``.
    """
    attention = np.asarray(attention, dtype=np.float64)
    labels = np.asarray(phase_labels)
    if attention.ndim != 3 or attention.shape[0] != labels.shape[0]:
        raise ValueError("attention and phase_labels must share the time dimension.")
    means = np.full((len(PHASE_NAMES), *attention.shape[1:]), np.nan, dtype=np.float64)
    counts = np.zeros(len(PHASE_NAMES), dtype=np.int64)
    for phase_index in range(len(PHASE_NAMES)):
        selected = attention[labels == phase_index]
        counts[phase_index] = len(selected)
        if len(selected):
            means[phase_index] = selected.mean(axis=0)
    return means, counts


def select_critical_edges(
    attention: np.ndarray,
    handover_events: np.ndarray | None = None,
    top_k: int = 4,
    event_radius: int = 4,
) -> list[tuple[int, int]]:
    """Select source-to-target edges emphasized near observed handovers.

    Event-participating edges are ranked by frequency first. Remaining slots
    use mean attention within event neighborhoods, or the full rollout when no
    events occur. Diagonal edges are never returned.

    Args:
        attention: Array shaped ``(time, target, source)``.
        handover_events: Optional boolean event tensor with the same shape.
        top_k: Maximum number of directed edges to return.
        event_radius: Context radius around each handover event.

    Returns:
        Directed ``(source, target)`` pairs in plotting priority order.
    """
    attention = np.asarray(attention, dtype=np.float64)
    if attention.ndim != 3 or attention.shape[1] != attention.shape[2]:
        raise ValueError("attention must have shape (time, fingers, fingers).")
    num_fingers = attention.shape[1]
    events = None if handover_events is None else np.asarray(handover_events, dtype=bool)
    chosen: list[tuple[int, int]] = []

    if events is not None and events.shape == attention.shape:
        counts = events.sum(axis=0)
        ranked_events = sorted(
            (
                (int(counts[target, source]), source, target)
                for target in range(num_fingers)
                for source in range(num_fingers)
                if source != target and counts[target, source] > 0
            ),
            reverse=True,
        )
        chosen.extend((source, target) for _, source, target in ranked_events[:top_k])

        event_times = np.flatnonzero(events.any(axis=(1, 2)))
        neighborhood = np.zeros(attention.shape[0], dtype=bool)
        for time_index in event_times:
            lo = max(0, time_index - int(event_radius))
            hi = min(len(neighborhood), time_index + int(event_radius) + 1)
            neighborhood[lo:hi] = True
        score = attention[neighborhood].mean(axis=0) if neighborhood.any() else attention.mean(axis=0)
    else:
        score = attention.mean(axis=0)

    ranked_attention = sorted(
        (
            (float(score[target, source]), source, target)
            for target in range(num_fingers)
            for source in range(num_fingers)
            if source != target
        ),
        reverse=True,
    )
    for _, source, target in ranked_attention:
        edge = (source, target)
        if edge not in chosen:
            chosen.append(edge)
        if len(chosen) >= int(top_k):
            break
    return chosen
