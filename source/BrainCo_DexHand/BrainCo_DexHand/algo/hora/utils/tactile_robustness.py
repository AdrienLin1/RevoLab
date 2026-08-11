"""Play-time tactile robustness perturbations for HORA policy inputs.

The perturbations are applied after an environment produces observations, so
they do not change physics, rewards, or termination conditions.  A spatial
dropout mask is sampled once per environment and reused for the complete play
run, matching a failed or mis-installed taxel rather than frame-wise noise.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch


class TactileObservationPerturber:
    """Apply force scaling, taxel dropout, and sensor noise to play observations.

    Regular-grid frames use four teacher channels or three student channels per
    taxel.  Estimated-official graph frames use ten teacher channels or five
    student channels per node, followed by per-finger context channels.
    """

    def __init__(
        self,
        *,
        layout: str,
        policy_mode: str,
        num_envs: int,
        teacher_frame_dim: int,
        student_frame_dim: int,
        graph_total_nodes: int = 0,
        graph_sensor_counts: Sequence[int] = (),
        tactile_priv_offset: int = 11,
        tactile_priv_dim: int = 0,
        force_scale: float = 1.0,
        spatial_dropout: float = 0.0,
        noise_std: float = 0.0,
        binary_flip_prob: float = 0.0,
        graph_force_limit: float = 5.0,
        legacy_action_dim: int = 21,
        legacy_frame_dim: int = 47,
        active_finger_indices: Sequence[int] = (0, 1, 2, 3, 4),
        device: torch.device | str = "cpu",
    ) -> None:
        """Validate settings and create the persistent per-environment mask.

        Args:
            layout: ``regular_grid`` or ``estimated_official``.
            policy_mode: ``stage1_teacher`` or ``tactile_student``.
            num_envs: Number of parallel environments in the play batch.
            teacher_frame_dim: Width of one teacher tactile frame.
            student_frame_dim: Width of one student tactile frame.
            graph_total_nodes: Number of physical graph nodes for graph layout.
            graph_sensor_counts: Per-finger graph node counts.
            tactile_priv_offset: Start of tactile data in ``priv_info``.
            tactile_priv_dim: Total tactile width in ``priv_info``.
            force_scale: Multiplicative force scale ``k``.
            spatial_dropout: Fraction of taxels permanently disabled per env.
            noise_std: Gaussian standard deviation in scaled TacSL force units.
            binary_flip_prob: Probability of flipping a student contact bit.
            graph_force_limit: Absolute force limit for graph force perturbations.
            legacy_action_dim: Number of hand actions/joints in the legacy obs.
            legacy_frame_dim: Width of one legacy actor-observation frame.
            active_finger_indices: Original five-finger indices represented by
                the tactile layout.
            device: Torch device used by the observation tensors.
        """
        if layout not in ("regular_grid", "estimated_official"):
            raise ValueError(f"unsupported tactile layout: {layout!r}")
        if policy_mode not in ("stage1_teacher", "tactile_student"):
            raise ValueError(f"unsupported policy mode: {policy_mode!r}")
        if force_scale <= 0.0:
            raise ValueError("force_scale must be positive")
        if noise_std < 0.0:
            raise ValueError("noise_std must be non-negative")
        for name, value in (
            ("spatial_dropout", spatial_dropout),
            ("binary_flip_prob", binary_flip_prob),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")

        self.layout = layout
        self.policy_mode = policy_mode
        self.force_scale = float(force_scale)
        self.spatial_dropout = float(spatial_dropout)
        self.noise_std = float(noise_std)
        self.binary_flip_prob = float(binary_flip_prob)
        self.graph_force_limit = float(graph_force_limit)
        self.tactile_priv_offset = int(tactile_priv_offset)
        self.tactile_priv_dim = int(tactile_priv_dim)
        self.teacher_frame_dim = int(teacher_frame_dim)
        self.student_frame_dim = int(student_frame_dim)
        self.graph_sensor_counts = tuple(int(v) for v in graph_sensor_counts)
        self.active_finger_indices = tuple(int(v) for v in active_finger_indices)
        self.legacy_action_dim = int(legacy_action_dim)
        self.legacy_frame_dim = int(legacy_frame_dim)

        if layout == "estimated_official":
            self.num_taxels = int(graph_total_nodes)
            self.teacher_channels = 10
            self.student_channels = 5
        else:
            if teacher_frame_dim % 4 != 0 or student_frame_dim % 3 != 0:
                raise ValueError("regular-grid tactile frame widths must be divisible by 4 and 3")
            self.num_taxels = int(student_frame_dim // 3)
            if teacher_frame_dim // 4 != self.num_taxels:
                raise ValueError("teacher and student regular-grid taxel counts differ")
            self.teacher_channels = 4
            self.student_channels = 3
        if self.num_taxels <= 0:
            raise ValueError("tactile frame must contain at least one taxel/node")
        if self.graph_force_limit <= 0.0:
            raise ValueError("graph force limit must be positive")
        if len(self.active_finger_indices) == 0:
            raise ValueError("active_finger_indices must not be empty")
        if layout == "regular_grid" and self.num_taxels % len(self.active_finger_indices) != 0:
            raise ValueError("regular-grid taxels must divide evenly across active fingers")
        if layout == "estimated_official" and sum(self.graph_sensor_counts) != self.num_taxels:
            raise ValueError("graph_sensor_counts must sum to graph_total_nodes")
        if layout == "estimated_official" and len(self.graph_sensor_counts) != len(
            self.active_finger_indices
        ):
            raise ValueError("graph_sensor_counts must match active_finger_indices")

        if self.spatial_dropout <= 0.0:
            keep = torch.ones((int(num_envs), self.num_taxels), device=device, dtype=torch.bool)
        elif self.spatial_dropout >= 1.0:
            keep = torch.zeros((int(num_envs), self.num_taxels), device=device, dtype=torch.bool)
        else:
            drop_count = int(round(self.spatial_dropout * self.num_taxels))
            random_order = torch.rand(
                (int(num_envs), self.num_taxels), device=device
            ).argsort(dim=-1)
            keep = torch.ones(
                (int(num_envs), self.num_taxels), device=device, dtype=torch.bool
            )
            keep.scatter_(1, random_order[:, :drop_count], False)
        self.keep_mask = keep
        self.finger_keep_ratio = self._build_finger_keep_ratio()
        self._legacy_obs_cache = None
        self._teacher_history_cache = None
        self._student_history_cache = None

    def _build_finger_keep_ratio(self) -> torch.Tensor:
        """Compute surviving taxel fraction for each of the five legacy fingers."""
        ratio = torch.ones((self.keep_mask.shape[0], 5), device=self.keep_mask.device)
        if self.layout == "estimated_official" and self.graph_sensor_counts:
            start = 0
            for finger_idx, count in zip(self.active_finger_indices, self.graph_sensor_counts):
                ratio[:, finger_idx] = self.keep_mask[:, start : start + count].float().mean(dim=-1)
                start += count
        elif self.layout == "regular_grid":
            per_finger = self.num_taxels // max(len(self.active_finger_indices), 1)
            start = 0
            for finger_idx in self.active_finger_indices:
                ratio[:, finger_idx] = self.keep_mask[:, start : start + per_finger].float().mean(dim=-1)
                start += per_finger
        return ratio

    def __call__(
        self,
        obs_dict: dict[str, torch.Tensor],
        *,
        reset_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return a copy of ``obs_dict`` with configured tactile faults applied.

        Historical noise samples are cached so an old frame does not receive a
        different random fault every time the temporal window is evaluated.

        Args:
            obs_dict: Clean observation dictionary returned by the environment.
            reset_mask: Optional per-environment mask for histories refilled by
                an environment reset on the latest step.

        Returns:
            A shallow dictionary copy containing perturbed tensor values.
        """
        result = dict(obs_dict)
        if "obs" in result:
            result["obs"] = self._update_legacy_obs(result["obs"], reset_mask)

        if self.policy_mode == "stage1_teacher":
            current_teacher = None
            if "tactile_hist" in result:
                result["tactile_hist"] = self._update_temporal_history(
                    result["tactile_hist"],
                    cache_name="_teacher_history_cache",
                    perturb=self._perturb_teacher_frames,
                    reset_mask=reset_mask,
                    repair_teacher_transition=True,
                )
                current_teacher = result["tactile_hist"][:, -1]
            if "priv_info" in result and self.tactile_priv_dim > 0:
                priv = result["priv_info"].clone()
                end = min(priv.shape[-1], self.tactile_priv_offset + self.tactile_priv_dim)
                block = priv[..., self.tactile_priv_offset : end]
                chunks = []
                while block.shape[-1] >= self.teacher_frame_dim and self.teacher_frame_dim > 0:
                    if not chunks and current_teacher is not None:
                        chunks.append(current_teacher)
                    else:
                        chunks.append(self._perturb_teacher_frames(block[..., : self.teacher_frame_dim]))
                    block = block[..., self.teacher_frame_dim :]
                if block.shape[-1] > 0:
                    chunks.append(block)
                if chunks:
                    priv[..., self.tactile_priv_offset : end] = torch.cat(chunks, dim=-1)
                result["priv_info"] = priv
        else:
            if "student_tactile_hist" in result:
                result["student_tactile_hist"] = self._update_temporal_history(
                    result["student_tactile_hist"],
                    cache_name="_student_history_cache",
                    perturb=self._perturb_student_frames,
                    reset_mask=reset_mask,
                    repair_student_transition=True,
                )
        return result

    def _update_temporal_history(
        self,
        frames: torch.Tensor,
        *,
        cache_name: str,
        perturb,
        reset_mask: torch.Tensor | None,
        repair_student_transition: bool = False,
        repair_teacher_transition: bool = False,
    ) -> torch.Tensor:
        """Perturb only newly acquired history frames and preserve old noise.

        Args:
            frames: Clean history tensor shaped ``(E, T, D)``.
            cache_name: Attribute used to retain the perturbed history.
            perturb: Layout-specific frame perturbation callable.
            reset_mask: Optional per-environment reset mask.
            repair_student_transition: Recompute newest structural transition
                channels from the cached prior contact bits.
            repair_teacher_transition: Recompute newest graph force deltas from
                the cached prior force channels.

        Returns:
            Updated perturbed history tensor.
        """
        if frames.ndim != 3:
            return perturb(frames)
        cache = getattr(self, cache_name)
        if cache is None or cache.shape != frames.shape:
            updated = perturb(frames)
        else:
            newest = perturb(frames[:, -1:])
            if repair_student_transition:
                newest = self._repair_student_transition(cache[:, -1:], newest)
            if repair_teacher_transition:
                newest = self._repair_teacher_transition(cache[:, -1:], newest)
            updated = torch.cat([cache[:, 1:], newest], dim=1)
            if reset_mask is not None:
                resets = reset_mask.to(device=frames.device, dtype=torch.bool)
                if resets.any():
                    reset_histories = perturb(frames)
                    updated[resets] = reset_histories[resets]
        setattr(self, cache_name, updated.clone())
        return updated

    def _repair_student_transition(
        self,
        previous: torch.Tensor,
        newest: torch.Tensor,
    ) -> torch.Tensor:
        """Align newest delta/on/off channels with cached noisy contact bits.

        Args:
            previous: Cached prior student frame ``(E, 1, D)``.
            newest: Newly perturbed student frame ``(E, 1, D)``.

        Returns:
            Newest frame with temporally consistent event channels.
        """
        result = newest.clone()
        if self.layout == "regular_grid":
            old_nodes = previous.view(previous.shape[0], 1, self.num_taxels, 3)
            new_nodes = result.view(result.shape[0], 1, self.num_taxels, 3)
            new_nodes[..., 1] = new_nodes[..., 0] - old_nodes[..., 0]
        else:
            node_width = self.num_taxels * 5
            old_nodes = previous[..., :node_width].view(
                previous.shape[0], 1, self.num_taxels, 5
            )
            new_nodes = result[..., :node_width].view(
                result.shape[0], 1, self.num_taxels, 5
            )
            old_bits = old_nodes[..., 0]
            new_bits = new_nodes[..., 0]
            new_nodes[..., 1] = ((old_bits < 0.5) & (new_bits > 0.5)).to(new_bits.dtype)
            new_nodes[..., 2] = ((old_bits > 0.5) & (new_bits < 0.5)).to(new_bits.dtype)
        return result

    def _repair_teacher_transition(
        self,
        previous: torch.Tensor,
        newest: torch.Tensor,
    ) -> torch.Tensor:
        """Recompute newest graph force deltas from cached raw forces.

        Args:
            previous: Cached prior teacher frame ``(E, 1, D)``.
            newest: Newly perturbed teacher frame ``(E, 1, D)``.

        Returns:
            Newest frame with consistent normal and shear-magnitude deltas.
        """
        if self.layout != "estimated_official":
            return newest
        result = newest.clone()
        node_width = self.num_taxels * 10
        old_nodes = previous[..., :node_width].view(
            previous.shape[0], 1, self.num_taxels, 10
        )
        new_nodes = result[..., :node_width].view(
            result.shape[0], 1, self.num_taxels, 10
        )
        new_nodes[..., 8] = (new_nodes[..., 5] - old_nodes[..., 5]).clamp(-1.0, 1.0)
        old_shear = torch.linalg.vector_norm(old_nodes[..., 6:8], dim=-1)
        new_shear = torch.linalg.vector_norm(new_nodes[..., 6:8], dim=-1)
        new_nodes[..., 9] = (new_shear - old_shear).clamp(-1.0, 1.0)
        return result

    def _update_legacy_obs(
        self,
        obs: torch.Tensor,
        reset_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Update the three-frame legacy observation without re-noising history.

        Args:
            obs: Flattened legacy actor observation.
            reset_mask: Optional per-environment reset mask.

        Returns:
            Flattened perturbed actor observation.
        """
        if obs.ndim != 2 or obs.shape[-1] % 3 != 0:
            return obs
        frame_dim = obs.shape[-1] // 3
        clean = obs.view(obs.shape[0], 3, frame_dim)
        cache = self._legacy_obs_cache
        if cache is None or cache.shape != clean.shape:
            updated = self._perturb_legacy_frames(clean)
        else:
            newest = self._perturb_legacy_frames(clean[:, -1:])
            updated = torch.cat([cache[:, 1:], newest], dim=1)
            if reset_mask is not None:
                resets = reset_mask.to(device=obs.device, dtype=torch.bool)
                if resets.any():
                    reset_frames = self._perturb_legacy_frames(clean)
                    updated[resets] = reset_frames[resets]
        self._legacy_obs_cache = updated.clone()
        return updated.reshape_as(obs)

    def _perturb_legacy_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """Perturb five legacy contact magnitudes embedded in actor observations."""
        frame_dim = min(self.legacy_frame_dim, frames.shape[-1])
        if frame_dim < 2 * self.legacy_action_dim + 5:
            return frames
        result = frames.clone()
        contact = result[..., 2 * self.legacy_action_dim : 2 * self.legacy_action_dim + 5]
        contact = contact * self.force_scale
        if self.noise_std > 0.0:
            contact = contact + torch.randn_like(contact) * self.noise_std
        if self.spatial_dropout > 0.0:
            contact = contact * self.finger_keep_ratio.unsqueeze(1)
        result[..., 2 * self.legacy_action_dim : 2 * self.legacy_action_dim + 5] = contact
        return result

    def _perturb_teacher_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """Perturb force-bearing teacher frames while preserving duration/context."""
        if frames.shape[-1] != self.teacher_frame_dim:
            raise ValueError(
                f"teacher frame width {frames.shape[-1]} != {self.teacher_frame_dim}"
            )
        result = frames.clone()
        if self.layout == "regular_grid":
            values = result.view(*result.shape[:-1], self.num_taxels, 4)
            force = values[..., :3] * self.force_scale
            if self.noise_std > 0.0:
                force = force + torch.randn_like(force) * self.noise_std
            values[..., :3] = force
            values = self._apply_taxel_mask(values, channels=4)
            result = values.reshape_as(result)
        else:
            node_width = self.num_taxels * 10
            nodes = result[..., :node_width].view(*result.shape[:-1], self.num_taxels, 10)
            nodes = self._perturb_graph_teacher_force(nodes)
            nodes = self._apply_taxel_mask(nodes, channels=10)
            node_values = nodes.reshape(*result.shape[:-1], node_width)
            context = self._scale_graph_context(result[..., node_width:])
            result = torch.cat([node_values, context], dim=-1)
        return result

    def _perturb_graph_teacher_force(self, nodes: torch.Tensor) -> torch.Tensor:
        """Apply force faults directly to raw graph force channels.

        Args:
            nodes: Graph teacher nodes with common and force channels.

        Returns:
            Nodes whose force and force-delta channels follow the configured
            physical force perturbation.
        """
        result = nodes.clone()
        force_limit = self.graph_force_limit
        raw_force = result[..., 5:8] * self.force_scale
        if self.noise_std > 0.0:
            raw_force = raw_force + torch.randn_like(raw_force) * self.noise_std
        raw_normal = raw_force[..., 0].clamp(0.0, force_limit)
        raw_shear = raw_force[..., 1:3].clamp(-force_limit, force_limit)

        result[..., 5] = raw_normal
        result[..., 6:8] = raw_shear
        if result.ndim >= 4:
            normal = result[..., 5]
            shear = torch.linalg.vector_norm(result[..., 6:8], dim=-1)
            normal_delta = torch.cat(
                [result[..., :1, :, 8], normal[..., 1:, :] - normal[..., :-1, :]],
                dim=-2,
            )
            shear_delta = torch.cat(
                [result[..., :1, :, 9], shear[..., 1:, :] - shear[..., :-1, :]],
                dim=-2,
            )
            result[..., 8] = normal_delta.clamp(-1.0, 1.0)
            result[..., 9] = shear_delta.clamp(-1.0, 1.0)
        else:
            result[..., 8:10] = (result[..., 8:10] * self.force_scale).clamp(-1.0, 1.0)
        return result

    def _perturb_student_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """Perturb structural student frames and keep contact transitions coherent."""
        if frames.shape[-1] != self.student_frame_dim:
            raise ValueError(
                f"student frame width {frames.shape[-1]} != {self.student_frame_dim}"
            )
        result = frames.clone()
        if self.layout == "regular_grid":
            values = result.view(*result.shape[:-1], self.num_taxels, 3)
            if self.binary_flip_prob > 0.0:
                bits = values[..., 0]
                flip = torch.rand_like(bits) < self.binary_flip_prob
                bits = torch.where(flip, 1.0 - bits, bits)
                values[..., 0] = bits
                if values.ndim >= 4:
                    previous = torch.cat([bits[..., :1, :], bits[..., :-1, :]], dim=-2)
                    values[..., 1] = bits - previous
            values = self._apply_taxel_mask(values, channels=3)
            result = values.reshape_as(result)
        else:
            node_width = self.num_taxels * 5
            nodes = result[..., :node_width].view(*result.shape[:-1], self.num_taxels, 5)
            if self.binary_flip_prob > 0.0:
                bits = nodes[..., 0]
                flip = torch.rand_like(bits) < self.binary_flip_prob
                bits = torch.where(flip, 1.0 - bits, bits)
                nodes[..., 0] = bits
                if nodes.ndim >= 4:
                    previous = torch.cat([bits[..., :1, :], bits[..., :-1, :]], dim=-2)
                    nodes[..., 1] = ((previous < 0.5) & (bits > 0.5)).to(bits.dtype)
                    nodes[..., 2] = ((previous > 0.5) & (bits < 0.5)).to(bits.dtype)
                nodes[..., 3] = torch.where(bits > 0.5, nodes[..., 3], torch.zeros_like(nodes[..., 3]))
                nodes[..., 4] = torch.where(bits > 0.5, nodes[..., 4], torch.zeros_like(nodes[..., 4]))
            nodes = self._apply_taxel_mask(nodes, channels=5)
            node_values = nodes.reshape(*result.shape[:-1], node_width)
            context = self._scale_graph_context(result[..., node_width:])
            result = torch.cat([node_values, context], dim=-1)
        return result

    def _apply_taxel_mask(self, values: torch.Tensor, *, channels: int) -> torch.Tensor:
        """Zero all channels belonging to a permanently dropped taxel/node."""
        if values.shape[-1] != channels:
            raise ValueError(f"expected {channels} taxel channels, got {values.shape[-1]}")
        if self.spatial_dropout <= 0.0:
            return values
        if values.ndim == 3:
            mask = self.keep_mask.unsqueeze(-1)
        else:
            mask = self.keep_mask.unsqueeze(1).unsqueeze(-1)
        return values * mask.to(values.dtype)

    def _scale_graph_context(self, context: torch.Tensor) -> torch.Tensor:
        """Reduce graph contact-ratio context according to surviving nodes."""
        if context.shape[-1] == 0 or not self.graph_sensor_counts or self.spatial_dropout <= 0.0:
            return context
        result = context.clone().view(*context.shape[:-1], len(self.graph_sensor_counts), 4)
        ratios = []
        start = 0
        for count in self.graph_sensor_counts:
            ratios.append(self.keep_mask[:, start : start + count].float().mean(dim=-1))
            start += count
        ratio = torch.stack(ratios, dim=-1)
        if result.ndim == 3:
            result[..., 3] *= ratio
        else:
            result[..., 3] *= ratio.unsqueeze(1)
        return result.reshape_as(context)
