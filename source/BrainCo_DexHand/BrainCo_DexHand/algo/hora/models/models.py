"""HORA tactile models for public MLP teacher / conv1d-GRU student training."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class MLP(nn.Module):
    """Stacked linear layers with ELU activations."""

    def __init__(self, units, input_size):
        super().__init__()
        layers = []
        for output_size in units:
            layers.append(nn.Linear(input_size, output_size))
            layers.append(nn.ELU())
            input_size = output_size
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


def _cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _cfg_bool(cfg, key, default=False):
    value = _cfg_get(cfg, key, default)
    if isinstance(value, str):
        return value.lower() in ("1", "true", "yes", "on")
    return bool(value)


def resolve_tactile_encoder_type(network_config, legacy_key="tactile_transformer"):
    """Resolve the explicitly selected Stage-1 teacher tactile architecture."""
    del legacy_key
    tactile_cfg = _cfg_get(network_config, "tactile_encoder", None)
    encoder_type = str(_cfg_get(tactile_cfg, "type", "mlp")).lower()
    supported = ("mlp", "finger_attention_gru")
    if encoder_type not in supported:
        raise ValueError(
            f"Unsupported teacher tactile encoder type {encoder_type!r}; "
            f"expected one of {supported}."
        )
    return encoder_type, tactile_cfg


def resolve_student_tactile_encoder_type(network_config):
    """Resolve student tactile encoder type: ``conv1d`` or ``gru``."""
    student_cfg = _cfg_get(network_config, "student_tactile_encoder", None)
    if student_cfg is None:
        return "conv1d", None
    encoder_type = str(_cfg_get(student_cfg, "type", "conv1d")).lower()
    if encoder_type not in ("conv1d", "gru"):
        raise ValueError(
            f"Unsupported student tactile encoder type {encoder_type!r}; "
            "public release supports conv1d or gru only."
        )
    return encoder_type, student_cfg


def infer_teacher_tactile_encoder_type_from_state_dict(state_dict) -> str:
    """Infer Stage1 teacher tactile encoder type from a checkpoint state dict."""
    keys = tuple(str(key).removeprefix("module.") for key in state_dict.keys()) \
        if isinstance(state_dict, dict) else ()
    tactile_keys = tuple(key for key in keys if key.startswith("tactile_encoder."))
    if not tactile_keys:
        return "mlp"

    has_finger_mlps = any(key.startswith("tactile_encoder.finger_mlps.") for key in keys)
    has_attention = any(key.startswith("tactile_encoder.finger_attention.") for key in keys)
    has_gru = any(key.startswith("tactile_encoder.gru.") for key in keys)
    has_identity = "tactile_encoder.finger_identity_embedding" in keys
    if has_finger_mlps and has_attention and has_gru and has_identity:
        return "finger_attention_gru"

    preview = ", ".join(tactile_keys[:5])
    if len(tactile_keys) > 5:
        preview += ", ..."
    raise ValueError(
        "Checkpoint contains an unrecognized Stage-1 tactile_encoder module. "
        "Supported teacher types are 'mlp' and 'finger_attention_gru'; "
        f"observed keys: {preview}."
    )


def _state_dict_tensor(state_dict, key: str):
    """Return a tensor from either a plain or ``module.``-prefixed state dict."""
    if not isinstance(state_dict, dict):
        return None
    value = state_dict.get(key)
    if value is None:
        value = state_dict.get(f"module.{key}")
    return value


def _finger_attention_checkpoint_metadata_errors(checkpoint_state, runtime_state) -> list[str]:
    """Return architecture metadata mismatches for structured teacher checkpoints."""
    errors = []
    metadata_keys = (
        ("tactile_encoder.architecture_signature", "architecture signature", False),
        ("tactile_encoder.active_finger_ids", "active finger identities", False),
        ("tactile_encoder.finger_counts_tensor", "finger node counts", False),
        ("tactile_encoder.node_coordinates", "normalized node coordinates", True),
    )
    for key, label, floating in metadata_keys:
        checkpoint_value = _state_dict_tensor(checkpoint_state, key)
        runtime_value = _state_dict_tensor(runtime_state, key)
        if checkpoint_value is None or runtime_value is None:
            errors.append(
                f"missing {label}: checkpoint={checkpoint_value is not None}, "
                f"runtime={runtime_value is not None}"
            )
            continue
        if tuple(checkpoint_value.shape) != tuple(runtime_value.shape):
            errors.append(
                f"{label} shape: checkpoint={tuple(checkpoint_value.shape)}, "
                f"runtime={tuple(runtime_value.shape)}"
            )
            continue
        if floating:
            matches = torch.allclose(
                checkpoint_value.detach().cpu().float(),
                runtime_value.detach().cpu().float(),
                rtol=0.0,
                atol=1.0e-7,
            )
        else:
            matches = torch.equal(
                checkpoint_value.detach().cpu(),
                runtime_value.detach().cpu(),
            )
        if not matches:
            errors.append(
                f"{label}: checkpoint={checkpoint_value.detach().cpu().tolist()}, "
                f"runtime={runtime_value.detach().cpu().tolist()}"
            )

    for key, label in (
        ("tactile_encoder.gru.weight_ih_l0", "GRU input"),
        ("actor_mlp.mlp.0.weight", "Actor input"),
    ):
        checkpoint_value = _state_dict_tensor(checkpoint_state, key)
        runtime_value = _state_dict_tensor(runtime_state, key)
        if checkpoint_value is None or runtime_value is None:
            errors.append(f"missing {label} weight")
        elif tuple(checkpoint_value.shape) != tuple(runtime_value.shape):
            errors.append(
                f"{label} shape: checkpoint={tuple(checkpoint_value.shape)}, "
                f"runtime={tuple(runtime_value.shape)}"
            )
    return errors


def infer_teacher_tactile_architecture_from_state_dict(
    state_dict,
) -> tuple[str | None, str | None]:
    """Return the fixed layout/history contract encoded by a teacher checkpoint."""
    encoder_type = infer_teacher_tactile_encoder_type_from_state_dict(state_dict)
    if encoder_type == "finger_attention_gru":
        signature = _state_dict_tensor(
            state_dict,
            "tactile_encoder.architecture_signature",
        )
        history_len = int(signature[0].item()) if signature is not None else 10
        return "estimated_official", f"history_len={history_len}"
    return None, None


def validate_teacher_tactile_checkpoint_compatibility(
    checkpoint_state,
    runtime_state,
    checkpoint_path: str = "",
) -> None:
    """Reject teacher checkpoints built for a different tactile architecture."""
    try:
        checkpoint_type = infer_teacher_tactile_encoder_type_from_state_dict(
            checkpoint_state
        )
        runtime_type = infer_teacher_tactile_encoder_type_from_state_dict(runtime_state)
    except ValueError as exc:
        location = f"\n  checkpoint: {checkpoint_path}" if checkpoint_path else ""
        raise RuntimeError(
            "Could not validate the Stage-1 teacher checkpoint architecture."
            f"{location}\n  - {exc}"
        ) from exc

    errors = []
    if checkpoint_type != runtime_type:
        errors.append(
            f"tactile encoder type: checkpoint={checkpoint_type}, runtime={runtime_type}"
        )
    elif checkpoint_type == "finger_attention_gru":
        errors.extend(
            _finger_attention_checkpoint_metadata_errors(
                checkpoint_state,
                runtime_state,
            )
        )

    if errors:
        location = f"\n  checkpoint: {checkpoint_path}" if checkpoint_path else ""
        details = "\n".join(f"  - {error}" for error in errors)
        raise RuntimeError(
            "Checkpoint tactile architecture is incompatible with the runtime model."
            f"{location}\n{details}\n"
            "Legacy flat-MLP and finger_attention_gru Stage-1 checkpoints are "
            "intentionally incompatible. Use the exact task-resolved configuration "
            "that created this checkpoint, or train the new architecture from scratch."
        )


def _network_mlp_units(network_config, key):
    block = _cfg_get(network_config, key, None)
    units = _cfg_get(block, "units", None)
    if units is None:
        raise KeyError(f"train.network.{key}.units is required")
    return list(units)


def build_actor_critic_kwargs(
    network_config,
    ppo_config,
    actions_num,
    input_shape,
    obs_per_step,
    proprio_adapt,
    env_cfg=None,
):
    """Build kwargs for ``ActorCritic`` from train yaml network/ppo sections."""
    tactile_type, tactile_cfg = resolve_tactile_encoder_type(network_config)
    result = {
        "actor_units": _network_mlp_units(network_config, "mlp"),
        "priv_mlp_units": _network_mlp_units(network_config, "priv_mlp"),
        "actions_num": actions_num,
        "input_shape": input_shape,
        "priv_info": ppo_config["priv_info"],
        "proprio_adapt": proprio_adapt,
        "priv_info_dim": ppo_config["priv_info_dim"],
        "obs_per_step": obs_per_step,
        "tactile_encoder_type": tactile_type,
        "tactile_encoder_cfg": tactile_cfg,
    }
    if tactile_type == "finger_attention_gru":
        if env_cfg is None:
            raise ValueError(
                "tactile_encoder.type='finger_attention_gru' requires env_cfg so the model "
                "can consume task-resolved tactile metadata."
            )
        result["tactile_env_metadata"] = {
            "config_label": str(
                getattr(env_cfg, "task", env_cfg.__class__.__name__)
            ),
            "layout": str(getattr(env_cfg, "tactile_layout", "")),
            "base_priv_dim": int(getattr(env_cfg, "tactile_priv_offset", -1)),
            "active_finger_names": tuple(
                str(value)
                for value in getattr(env_cfg, "tactile_active_finger_names", ())
            ),
            "finger_counts": tuple(
                int(value)
                for value in getattr(env_cfg, "tactile_graph_sensor_counts", ())
            ),
            "finger_positions": tuple(
                getattr(env_cfg, "tactile_graph_finger_positions", ())
            ),
            "frame_dim": int(getattr(env_cfg, "teacher_tactile_frame_dim", -1)),
            "history_len": int(getattr(env_cfg, "teacher_tactile_history_len", -1)),
            "node_frame_channels": int(
                getattr(env_cfg, "tactile_graph_common_channels", 0)
            )
            + int(getattr(env_cfg, "tactile_graph_force_channels", 0)),
            "context_channels": int(
                getattr(env_cfg, "tactile_graph_context_channels", -1)
            ),
        }
    return result


class FingerAttentionGRUTactileEncoder(nn.Module):
    """Structured Stage-1 encoder: whole-finger MLP, attention, then GRU."""

    canonical_finger_names = ("thumb", "index", "middle", "ring", "little")
    node_channels = ("u", "v", "b", "d", "Fn", "Ft1", "Ft2")
    node_source_channels = (0, 3, 5, 6, 7)

    def __init__(self, encoder_cfg, env_metadata):
        super().__init__()
        self.config_label = str(_cfg_get(env_metadata, "config_label", "unknown"))
        self.layout = str(_cfg_get(env_metadata, "layout", ""))
        self.active_finger_names = tuple(
            str(value)
            for value in _cfg_get(env_metadata, "active_finger_names", ())
        )
        self.finger_counts = tuple(
            int(value) for value in _cfg_get(env_metadata, "finger_counts", ())
        )
        self.num_fingers = len(self.active_finger_names)
        self.total_nodes = int(sum(self.finger_counts))
        self.frame_dim = int(_cfg_get(env_metadata, "frame_dim", -1))
        self.history_len = int(_cfg_get(env_metadata, "history_len", -1))
        self.node_frame_channels = int(
            _cfg_get(env_metadata, "node_frame_channels", -1)
        )
        self.context_channels = int(_cfg_get(env_metadata, "context_channels", -1))

        self.finger_token_dim = int(_cfg_get(encoder_cfg, "finger_token_dim", 32))
        self.finger_mlp_hidden_dim = int(
            _cfg_get(encoder_cfg, "finger_mlp_hidden_dim", 64)
        )
        self.attention_heads = int(_cfg_get(encoder_cfg, "attention_heads", 4))
        self.attention_ff_dim = int(_cfg_get(encoder_cfg, "attention_ff_dim", 64))
        self.gru_hidden_dim = int(_cfg_get(encoder_cfg, "gru_hidden_dim", 128))
        self.gru_num_layers = int(_cfg_get(encoder_cfg, "gru_num_layers", 1))
        self.gru_bidirectional = _cfg_bool(
            encoder_cfg,
            "gru_bidirectional",
            False,
        )
        configured_history_len = int(_cfg_get(encoder_cfg, "history_len", 10))
        configured_node_channels = tuple(
            str(value)
            for value in _cfg_get(encoder_cfg, "node_channels", self.node_channels)
        )
        configured_node_input_dim = int(_cfg_get(encoder_cfg, "node_input_dim", 7))
        configured_source_channels = tuple(
            int(value)
            for value in _cfg_get(
                encoder_cfg,
                "node_source_channels",
                self.node_source_channels,
            )
        )
        configured_node_frame_channels = int(
            _cfg_get(encoder_cfg, "teacher_node_frame_channels", 10)
        )
        configured_context_channels = int(
            _cfg_get(encoder_cfg, "finger_context_channels", 4)
        )

        expected_frame_dim = self.total_nodes * 10 + self.num_fingers * 4
        self.expected_frame_dim = expected_frame_dim
        ordered_subset = tuple(
            name
            for name in self.canonical_finger_names
            if name in self.active_finger_names
        )
        if self.layout != "estimated_official":
            self._raise_configuration_error("layout must be 'estimated_official'")
        if self.num_fingers not in (3, 4, 5):
            self._raise_configuration_error("the active finger count must be 3, 4, or 5")
        if len(set(self.active_finger_names)) != self.num_fingers:
            self._raise_configuration_error("active finger names must be unique")
        if ordered_subset != self.active_finger_names:
            self._raise_configuration_error(
                "active fingers must be an ordered subset of "
                f"{self.canonical_finger_names}"
            )
        if len(self.finger_counts) != self.num_fingers:
            self._raise_configuration_error(
                "finger_counts must align one-to-one with active_finger_names"
            )
        expected_counts = tuple(
            31 if name == "thumb" else 21 for name in self.active_finger_names
        )
        if self.finger_counts != expected_counts:
            self._raise_configuration_error(
                f"expected physical sensor counts {expected_counts}"
            )
        if self.node_frame_channels != 10 or configured_node_frame_channels != 10:
            self._raise_configuration_error(
                "both environment and encoder node-frame channel counts must be 10"
            )
        if self.context_channels != 4 or configured_context_channels != 4:
            self._raise_configuration_error(
                "both environment and encoder per-finger context channel counts must be 4"
            )
        if self.frame_dim != expected_frame_dim:
            self._raise_configuration_error("teacher tactile frame width is inconsistent")
        if self.history_len != 10 or configured_history_len != 10:
            self._raise_configuration_error(
                "both environment and encoder tactile history lengths must be 10",
                configured_history_len=configured_history_len,
            )
        if configured_node_channels != self.node_channels:
            self._raise_configuration_error(
                f"node_channels must be {self.node_channels}"
            )
        if configured_node_input_dim != 7:
            self._raise_configuration_error("node_input_dim must be 7")
        if configured_source_channels != self.node_source_channels:
            self._raise_configuration_error(
                f"node_source_channels must be {self.node_source_channels}"
            )
        fixed_dimensions = {
            "finger_token_dim": (self.finger_token_dim, 32),
            "finger_mlp_hidden_dim": (self.finger_mlp_hidden_dim, 64),
            "attention_heads": (self.attention_heads, 4),
            "attention_ff_dim": (self.attention_ff_dim, 64),
            "gru_hidden_dim": (self.gru_hidden_dim, 128),
            "gru_num_layers": (self.gru_num_layers, 1),
        }
        mismatched_dimensions = [
            f"{name}={actual} (expected {expected})"
            for name, (actual, expected) in fixed_dimensions.items()
            if actual != expected
        ]
        if self.gru_bidirectional:
            mismatched_dimensions.append("gru_bidirectional=True (expected False)")
        if mismatched_dimensions:
            self._raise_configuration_error(
                "fixed architecture mismatch: " + ", ".join(mismatched_dimensions)
            )

        raw_positions = tuple(_cfg_get(env_metadata, "finger_positions", ()))
        if len(raw_positions) != self.num_fingers:
            self._raise_configuration_error(
                "finger_positions must align one-to-one with active_finger_names"
            )
        positions = []
        for finger_name, count, values in zip(
            self.active_finger_names,
            self.finger_counts,
            raw_positions,
        ):
            try:
                xy = torch.as_tensor(values, dtype=torch.float32)
            except (TypeError, ValueError) as exc:
                self._raise_configuration_error(
                    f"could not parse coordinates for {finger_name}: {exc}"
                )
            if tuple(xy.shape) != (count, 2):
                self._raise_configuration_error(
                    f"coordinates for {finger_name} have shape {tuple(xy.shape)}, "
                    f"expected {(count, 2)}"
                )
            if not bool(torch.isfinite(xy).all()):
                self._raise_configuration_error(
                    f"coordinates for {finger_name} contain non-finite values"
                )
            centered_error = float(xy.mean(dim=0).abs().max())
            diameter = float(torch.pdist(xy).max())
            if centered_error > 1.0e-5 or abs(diameter - 1.0) > 1.0e-5:
                self._raise_configuration_error(
                    f"coordinates for {finger_name} are not independently normalized "
                    f"(center_error={centered_error:.3g}, diameter={diameter:.6g})"
                )
            positions.append(xy)

        self.output_dim = self.gru_hidden_dim
        self.register_buffer("node_coordinates", torch.cat(positions, dim=0))
        self.register_buffer(
            "active_finger_ids",
            torch.tensor(
                [self.canonical_finger_names.index(name) for name in self.active_finger_names],
                dtype=torch.long,
            ),
        )
        self.register_buffer(
            "finger_counts_tensor",
            torch.tensor(self.finger_counts, dtype=torch.long),
        )
        architecture_signature = (
            self.history_len,
            self.frame_dim,
            self.node_frame_channels,
            self.context_channels,
            self.num_fingers,
            self.total_nodes,
            7,
            self.finger_mlp_hidden_dim,
            self.finger_token_dim,
            self.attention_heads,
            self.attention_ff_dim,
            self.gru_hidden_dim,
            self.gru_num_layers,
            int(self.gru_bidirectional),
            *self.node_source_channels,
        )
        self.register_buffer(
            "architecture_signature",
            torch.tensor(architecture_signature, dtype=torch.long),
        )
        self.register_buffer(
            "_node_source_channel_indices",
            torch.tensor(self.node_source_channels, dtype=torch.long),
            persistent=False,
        )

        self.finger_mlps = nn.ModuleDict(
            {
                name: MLP(
                    units=[self.finger_mlp_hidden_dim, self.finger_token_dim],
                    input_size=count * 7,
                )
                for name, count in zip(self.active_finger_names, self.finger_counts)
            }
        )
        self.finger_identity_embedding = nn.Parameter(
            torch.empty(len(self.canonical_finger_names), self.finger_token_dim)
        )
        nn.init.normal_(self.finger_identity_embedding, mean=0.0, std=0.02)
        self.finger_attention = nn.TransformerEncoderLayer(
            d_model=self.finger_token_dim,
            nhead=self.attention_heads,
            dim_feedforward=self.attention_ff_dim,
            dropout=0.0,
            batch_first=True,
        )
        self.gru = nn.GRU(
            input_size=self.num_fingers * self.finger_token_dim,
            hidden_size=self.gru_hidden_dim,
            num_layers=self.gru_num_layers,
            batch_first=True,
            bidirectional=self.gru_bidirectional,
            dropout=0.0,
        )

    def _raise_configuration_error(self, reason, configured_history_len=None):
        expected_frame_dim = self.total_nodes * 10 + self.num_fingers * 4
        history = self.history_len
        if configured_history_len is not None:
            history = f"env={self.history_len}, encoder={configured_history_len}"
        raise ValueError(
            "Invalid finger_attention_gru tactile configuration: "
            f"{reason}; task/config={self.config_label!r}; "
            f"active_finger_names={self.active_finger_names}; "
            f"finger_counts={self.finger_counts}; total_nodes={self.total_nodes}; "
            f"expected_frame_dim={expected_frame_dim}; "
            f"actual_frame_dim={self.frame_dim}; history_len={history}; "
            f"layout={self.layout!r}."
        )

    def _validate_history_shape(self, tactile_hist):
        if not isinstance(tactile_hist, torch.Tensor):
            self._raise_configuration_error(
                f"tactile_hist must be a torch.Tensor, got {type(tactile_hist).__name__}"
            )
        if tactile_hist.ndim != 3:
            self._raise_configuration_error(
                f"tactile_hist must have shape [B, 10, {self.frame_dim}], "
                f"got {tuple(tactile_hist.shape)}"
            )
        if tactile_hist.shape[1] != self.history_len or tactile_hist.shape[2] != self.frame_dim:
            self._raise_configuration_error(
                f"tactile_hist must have shape [B, {self.history_len}, {self.frame_dim}], "
                f"got {tuple(tactile_hist.shape)}"
            )

    def extract_node_inputs(self, tactile_hist: torch.Tensor) -> torch.Tensor:
        """Return ``[u,v,b,d,Fn,Ft1,Ft2]`` for every physical node."""
        self._validate_history_shape(tactile_hist)
        batch_size = tactile_hist.shape[0]
        node_width = self.total_nodes * self.node_frame_channels
        dynamic = tactile_hist[..., :node_width].reshape(
            batch_size,
            self.history_len,
            self.total_nodes,
            self.node_frame_channels,
        )
        dynamic = dynamic.index_select(-1, self._node_source_channel_indices)
        coordinates = self.node_coordinates.to(dtype=dynamic.dtype).view(
            1,
            1,
            self.total_nodes,
            2,
        )
        coordinates = coordinates.expand(batch_size, self.history_len, -1, -1)
        return torch.cat((coordinates, dynamic), dim=-1)

    def encode_fingers(self, node_inputs: torch.Tensor) -> torch.Tensor:
        """Flatten each whole finger and apply its own independent MLP."""
        expected_shape = (
            self.history_len,
            self.total_nodes,
            len(self.node_channels),
        )
        if node_inputs.ndim != 4 or tuple(node_inputs.shape[1:]) != expected_shape:
            self._raise_configuration_error(
                "node inputs must have shape "
                f"[B, {expected_shape[0]}, {expected_shape[1]}, {expected_shape[2]}], "
                f"got {tuple(node_inputs.shape)}"
            )
        finger_tokens = []
        node_offset = 0
        for finger_name, count in zip(self.active_finger_names, self.finger_counts):
            finger_nodes = node_inputs[:, :, node_offset : node_offset + count, :]
            finger_tokens.append(
                self.finger_mlps[finger_name](finger_nodes.flatten(start_dim=2))
            )
            node_offset += count
        tokens = torch.stack(finger_tokens, dim=2)
        identity = self.finger_identity_embedding.index_select(0, self.active_finger_ids)
        return tokens + identity.view(1, 1, self.num_fingers, self.finger_token_dim)

    def apply_finger_attention(self, finger_tokens: torch.Tensor) -> torch.Tensor:
        """Apply self-attention only across active fingers within each frame."""
        expected_shape = (
            self.history_len,
            self.num_fingers,
            self.finger_token_dim,
        )
        if finger_tokens.ndim != 4 or tuple(finger_tokens.shape[1:]) != expected_shape:
            self._raise_configuration_error(
                "finger tokens must have shape "
                f"[B, {expected_shape[0]}, {expected_shape[1]}, {expected_shape[2]}], "
                f"got {tuple(finger_tokens.shape)}"
            )
        batch_size = finger_tokens.shape[0]
        attended = self.finger_attention(
            finger_tokens.reshape(
                batch_size * self.history_len,
                self.num_fingers,
                self.finger_token_dim,
            )
        )
        return attended.reshape(
            batch_size,
            self.history_len,
            self.num_fingers,
            self.finger_token_dim,
        )

    def encode_temporal(self, attended_tokens: torch.Tensor) -> torch.Tensor:
        """Flatten active tokens per frame and return the final GRU hidden state."""
        if attended_tokens.ndim == 4:
            expected_shape = (
                self.history_len,
                self.num_fingers,
                self.finger_token_dim,
            )
            if tuple(attended_tokens.shape[1:]) != expected_shape:
                self._raise_configuration_error(
                    f"attended tokens have shape {tuple(attended_tokens.shape)}, "
                    f"expected [B, {expected_shape[0]}, {expected_shape[1]}, "
                    f"{expected_shape[2]}]"
                )
            sequence = attended_tokens.flatten(start_dim=2)
        elif attended_tokens.ndim == 3:
            expected_width = self.num_fingers * self.finger_token_dim
            if (
                attended_tokens.shape[1] != self.history_len
                or attended_tokens.shape[2] != expected_width
            ):
                self._raise_configuration_error(
                    f"GRU input has shape {tuple(attended_tokens.shape)}, expected "
                    f"[B, {self.history_len}, {expected_width}]"
                )
            sequence = attended_tokens
        else:
            self._raise_configuration_error(
                "attended tokens must be rank 3 or rank 4, "
                f"got rank {attended_tokens.ndim}"
            )
        _, hidden = self.gru(sequence)
        return hidden[-1]

    def forward(self, tactile_hist: torch.Tensor) -> torch.Tensor:
        node_inputs = self.extract_node_inputs(tactile_hist)
        finger_tokens = self.encode_fingers(node_inputs)
        attended_tokens = self.apply_finger_attention(finger_tokens)
        return self.encode_temporal(attended_tokens)


def build_student_tactile_policy_kwargs(network_config, actions_num, env_cfg):
    """Build kwargs for ``TactileStudentPolicy`` from train yaml + tactile env cfg."""
    encoder_type, encoder_cfg = resolve_student_tactile_encoder_type(network_config)
    emb_dim = int(env_cfg.student_tactile_encoder_output_dim)
    if encoder_cfg is not None:
        emb_dim = int(_cfg_get(encoder_cfg, "output_dim", emb_dim))
    return {
        "actor_units": _network_mlp_units(network_config, "mlp"),
        "actions_num": actions_num,
        "proprio_hist_len": int(env_cfg.student_proprio_history_len),
        "proprio_frame_dim": int(env_cfg.student_proprio_frame_dim),
        "tactile_frame_dim": int(env_cfg.student_tactile_frame_dim),
        "tactile_hist_len": int(env_cfg.student_tactile_history_len),
        "tactile_emb_dim": emb_dim,
        "tactile_encoder_type": encoder_type,
        "tactile_encoder_cfg": encoder_cfg,
        "gated_fusion": _cfg_bool(encoder_cfg, "gated_fusion", True),
        "distill_dim": int(_cfg_get(encoder_cfg, "distill_dim", 64)),
    }


class GatedTactileFusion(nn.Module):
    """Gate tactile latent residual onto a proprio / base embedding."""

    def __init__(self, base_dim: int, tactile_dim: int, out_dim: int):
        super().__init__()
        self.tactile_proj = nn.Linear(tactile_dim, out_dim)
        self.gate = nn.Sequential(
            nn.Linear(base_dim + tactile_dim, out_dim),
            nn.Sigmoid(),
        )
        if base_dim != out_dim:
            self.base_proj = nn.Linear(base_dim, out_dim)
        else:
            self.base_proj = nn.Identity()

    def forward(self, base_embed: torch.Tensor, tactile_embed: torch.Tensor) -> torch.Tensor:
        base = self.base_proj(base_embed)
        gate = self.gate(torch.cat([base_embed, tactile_embed], dim=-1))
        return base + gate * self.tactile_proj(tactile_embed)


class ProprioAdaptTConv(nn.Module):
    """Temporal conv encoder for non-tactile Stage2 ProprioAdapt."""

    def __init__(self, obs_per_step=47, output_dim=32):
        super().__init__()
        self.output_dim = int(output_dim)
        self.channel_transform = nn.Sequential(
            nn.Linear(obs_per_step, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
        )
        self.temporal_aggregation = nn.Sequential(
            nn.Conv1d(32, 32, (9,), stride=(2,)),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 32, (5,), stride=(1,)),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 32, (5,), stride=(1,)),
            nn.ReLU(inplace=True),
        )
        self.low_dim_proj = nn.Linear(32 * 3, self.output_dim)

    def forward(self, x):
        x = self.channel_transform(x)
        x = x.permute((0, 2, 1))
        x = self.temporal_aggregation(x)
        x = self.low_dim_proj(x.flatten(1))
        return x


class TactileHistoryEncoder(nn.Module):
    """Conv1d spatio-temporal encoder for student binary tactile history."""

    def __init__(self, frame_dim=240, history_len=10, output_dim=240):
        super().__init__()
        if history_len != 10:
            raise ValueError(
                f"TactileHistoryEncoder temporal convs expect history_len=10, got {history_len}"
            )
        self.frame_dim = frame_dim
        self.history_len = history_len
        self.output_dim = output_dim
        self.channel_transform = nn.Sequential(
            nn.Linear(frame_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
        )
        self.temporal_aggregation = nn.Sequential(
            nn.Conv1d(256, 256, (4,), stride=(2,)),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, (4,), stride=(1,)),
            nn.ReLU(inplace=True),
        )
        self.low_dim_proj = nn.Linear(256, output_dim)

    def forward(self, x):
        x = self.channel_transform(x)
        x = x.permute((0, 2, 1))
        x = self.temporal_aggregation(x)
        x = self.low_dim_proj(x.flatten(1))
        return x


class TactileHistoryGRUEncoder(nn.Module):
    """GRU spatio-temporal encoder for student binary tactile history."""

    def __init__(
        self,
        frame_dim=240,
        history_len=10,
        output_dim=240,
        hidden_dim=256,
        num_layers=1,
        bidirectional=False,
        dropout=0.0,
        pool="last",
    ):
        super().__init__()
        self.frame_dim = int(frame_dim)
        self.history_len = int(history_len)
        self.output_dim = int(output_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.bidirectional = bool(bidirectional)
        self.pool = str(pool).lower()
        if self.pool not in ("last", "mean"):
            raise ValueError(f"pool must be 'last' or 'mean', got {pool}")

        self.frame_proj = nn.Sequential(
            nn.Linear(self.frame_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.gru = nn.GRU(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=self.bidirectional,
            dropout=float(dropout) if self.num_layers > 1 else 0.0,
        )
        gru_out_dim = self.hidden_dim * (2 if self.bidirectional else 1)
        self.out_proj = nn.Linear(gru_out_dim, self.output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, history_len, frame_dim = x.shape
        if history_len != self.history_len:
            raise ValueError(f"expected history_len={self.history_len}, got {history_len}")
        if frame_dim != self.frame_dim:
            raise ValueError(f"expected frame_dim={self.frame_dim}, got {frame_dim}")

        seq = self.frame_proj(x)
        gru_out, _ = self.gru(seq)
        if self.pool == "mean":
            pooled = gru_out.mean(dim=1)
        else:
            pooled = gru_out[:, -1, :]
        return self.out_proj(pooled)


def _build_student_tactile_encoder(kwargs):
    encoder_type = str(kwargs.pop("tactile_encoder_type", "conv1d")).lower()
    encoder_cfg = kwargs.pop("tactile_encoder_cfg", None)
    kwargs.pop("tactile_layout_encoder_cfg", None)
    frame_dim = kwargs.pop("tactile_frame_dim")
    history_len = kwargs.pop("tactile_hist_len")
    output_dim = kwargs.pop("tactile_emb_dim")

    if encoder_type == "gru":
        gru_cfg = _cfg_get(encoder_cfg, "gru", encoder_cfg) or {}
        return TactileHistoryGRUEncoder(
            frame_dim=frame_dim,
            history_len=history_len,
            output_dim=output_dim,
            hidden_dim=int(_cfg_get(gru_cfg, "hidden_dim", 256)),
            num_layers=int(_cfg_get(gru_cfg, "num_layers", 1)),
            bidirectional=_cfg_bool(gru_cfg, "bidirectional", False),
            dropout=float(_cfg_get(gru_cfg, "dropout", 0.0)),
            pool=str(_cfg_get(gru_cfg, "pool", "last")),
        )

    if encoder_type != "conv1d":
        raise ValueError(f"Unsupported student tactile encoder type: {encoder_type}")
    return TactileHistoryEncoder(
        frame_dim=frame_dim,
        history_len=history_len,
        output_dim=output_dim,
    )


class TactileStudentPolicy(nn.Module):
    """DAgger student for tactile screw tasks (real-robot sensing only)."""

    def __init__(self, kwargs):
        super().__init__()
        actions_num = kwargs.pop("actions_num")
        self.units = kwargs.pop("actor_units")
        self.proprio_hist_len = kwargs.pop("proprio_hist_len")
        self.proprio_frame_dim = kwargs.pop("proprio_frame_dim")
        use_gated_fusion = bool(kwargs.pop("gated_fusion", True))
        distill_dim = int(kwargs.pop("distill_dim", 64))
        self.tactile_encoder_type = str(
            kwargs.get("tactile_encoder_type", "conv1d")
        ).lower()
        self.tactile_encoder = _build_student_tactile_encoder(kwargs)
        proprio_dim = self.proprio_hist_len * self.proprio_frame_dim
        self.tactile_emb_dim = int(self.tactile_encoder.output_dim)
        self.use_gated_fusion = use_gated_fusion
        if use_gated_fusion:
            self.proprio_proj = nn.Linear(proprio_dim, self.units[0])
            self.fusion = GatedTactileFusion(
                base_dim=self.units[0],
                tactile_dim=self.tactile_emb_dim,
                out_dim=self.units[0],
            )
            actor_input = self.units[0]
            self.obs_dim = proprio_dim + self.tactile_emb_dim
            remaining_units = self.units[1:] if len(self.units) > 1 else [self.units[0]]
            self.actor_mlp = MLP(units=remaining_units, input_size=actor_input)
            out_size = remaining_units[-1]
        else:
            self.obs_dim = proprio_dim + self.tactile_emb_dim
            self.actor_mlp = MLP(units=self.units, input_size=self.obs_dim)
            out_size = self.units[-1]
        self.mu = torch.nn.Linear(out_size, actions_num)
        self.value = torch.nn.Linear(out_size, 1)
        self.sigma = nn.Parameter(
            torch.zeros(actions_num, requires_grad=True, dtype=torch.float32)
        )
        self.distill_proj = nn.Linear(self.tactile_emb_dim, distill_dim)

        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Conv1d)):
                fan_out = module.kernel_size[0] * module.out_channels
                module.weight.data.normal_(mean=0.0, std=np.sqrt(2.0 / fan_out))
                if getattr(module, "bias", None) is not None:
                    torch.nn.init.zeros_(module.bias)
            if isinstance(module, nn.Linear) and getattr(module, "bias", None) is not None:
                torch.nn.init.zeros_(module.bias)

    def encode_tactile(self, tactile_hist: torch.Tensor) -> torch.Tensor:
        return self.tactile_encoder(tactile_hist)

    def encode_tactile_with_distill(
        self, tactile_hist: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tactile_emb = self.encode_tactile(tactile_hist)
        return tactile_emb, self.distill_proj(tactile_emb)

    def _build_obs_from_embedding(self, proprio_hist, tactile_emb):
        proprio_flat = proprio_hist.flatten(1)
        if self.use_gated_fusion:
            return self.fusion(self.proprio_proj(proprio_flat), tactile_emb)
        return torch.cat([proprio_flat, tactile_emb], dim=-1)

    def build_obs(self, proprio_hist, tactile_hist):
        tactile_emb = self.encode_tactile(tactile_hist)
        return self._build_obs_from_embedding(proprio_hist, tactile_emb), tactile_emb

    def _actor_features(self, proprio_hist, tactile_hist):
        tactile_emb = self.encode_tactile(tactile_hist)
        obs = self._build_obs_from_embedding(proprio_hist, tactile_emb)
        return self.actor_mlp(obs), tactile_emb

    def forward(self, proprio_hist, tactile_hist):
        features, _ = self._actor_features(proprio_hist, tactile_hist)
        return self.mu(features)

    def forward_with_latent(self, proprio_hist, tactile_hist):
        tactile_emb, distill_repr = self.encode_tactile_with_distill(tactile_hist)
        obs = self._build_obs_from_embedding(proprio_hist, tactile_emb)
        mu = self.mu(self.actor_mlp(obs))
        return mu, distill_repr

    def evaluate_dagger_ppo(self, proprio_hist, tactile_hist, actions):
        tactile_emb, distill_repr = self.encode_tactile_with_distill(tactile_hist)
        obs = self._build_obs_from_embedding(proprio_hist, tactile_emb)
        features = self.actor_mlp(obs)
        mu = self.mu(features)
        sigma = torch.exp(self.sigma).expand_as(mu)
        distr = torch.distributions.Normal(mu, sigma)
        values = self.value(features).squeeze(-1)
        neglogp = -distr.log_prob(actions).sum(dim=-1)
        entropy = distr.entropy().sum(dim=-1)
        return mu, distill_repr, values, neglogp, entropy, sigma

    def act(self, proprio_hist, tactile_hist, deterministic: bool = False):
        features, _ = self._actor_features(proprio_hist, tactile_hist)
        mu = self.mu(features)
        sigma = torch.exp(self.sigma).expand_as(mu)
        distr = torch.distributions.Normal(mu, sigma)
        actions = mu if deterministic else distr.sample()
        neglogp = -distr.log_prob(actions).sum(dim=-1)
        values = self.value(features).squeeze(-1)
        return actions, mu, sigma, values, neglogp

    def evaluate_ppo(self, proprio_hist, tactile_hist, actions):
        features, _ = self._actor_features(proprio_hist, tactile_hist)
        mu = self.mu(features)
        sigma = torch.exp(self.sigma).expand_as(mu)
        distr = torch.distributions.Normal(mu, sigma)
        neglogp = -distr.log_prob(actions).sum(dim=-1)
        entropy = distr.entropy().sum(dim=-1)
        values = self.value(features).squeeze(-1)
        return neglogp, entropy, values, mu, sigma


class ActorCritic(nn.Module):
    """Stage-1 teacher with legacy flat-MLP and structured tactile paths."""

    def __init__(self, kwargs):
        super().__init__()
        actions_num = kwargs.pop("actions_num")
        input_shape = kwargs.pop("input_shape")
        self.units = kwargs.pop("actor_units")
        self.priv_mlp = kwargs.pop("priv_mlp_units")
        obs_per_step = kwargs.pop("obs_per_step", 47)
        mlp_input_shape = input_shape[0]

        out_size = self.units[-1]
        self.priv_info = kwargs["priv_info"]
        self.priv_info_stage2 = kwargs["proprio_adapt"]
        self.tactile_encoder_type = str(kwargs.pop("tactile_encoder_type", "mlp")).lower()
        tactile_encoder_cfg = kwargs.pop("tactile_encoder_cfg", None)
        tactile_env_metadata = kwargs.pop("tactile_env_metadata", None)

        if self.tactile_encoder_type not in ("mlp", "finger_attention_gru"):
            raise ValueError(
                f"Unsupported teacher tactile encoder type: {self.tactile_encoder_type}"
            )
        if self.tactile_encoder_type == "finger_attention_gru" and not self.priv_info:
            raise ValueError(
                "tactile_encoder.type='finger_attention_gru' requires ppo.priv_info=true"
            )
        if self.tactile_encoder_type == "finger_attention_gru" and self.priv_info_stage2:
            raise ValueError(
                "tactile_encoder.type='finger_attention_gru' is a Stage-1 Teacher and "
                "requires ppo.proprio_adapt=false"
            )

        if self.priv_info:
            self.priv_info_dim = int(kwargs["priv_info_dim"])
            if self.tactile_encoder_type == "finger_attention_gru":
                if tactile_env_metadata is None:
                    raise ValueError(
                        "tactile_encoder.type='finger_attention_gru' requires "
                        "task-resolved tactile_env_metadata"
                    )
                self.tactile_encoder = FingerAttentionGRUTactileEncoder(
                    tactile_encoder_cfg,
                    tactile_env_metadata,
                )
                self.base_priv_dim = int(
                    _cfg_get(tactile_env_metadata, "base_priv_dim", -1)
                )
                self.tactile_priv_offset = self.base_priv_dim
                if self.base_priv_dim <= 0:
                    self.tactile_encoder._raise_configuration_error(
                        f"base_priv_dim must be positive, got {self.base_priv_dim}"
                    )
                expected_priv_info_dim = (
                    self.base_priv_dim + self.tactile_encoder.frame_dim
                )
                if self.priv_info_dim != expected_priv_info_dim:
                    self.tactile_encoder._raise_configuration_error(
                        "full priv_info width must equal the base privilege slice plus "
                        "one teacher tactile frame: "
                        f"base_priv_dim={self.base_priv_dim}, "
                        f"tactile_frame_dim={self.tactile_encoder.frame_dim}, "
                        f"expected_priv_info_dim={expected_priv_info_dim}, "
                        f"actual_priv_info_dim={self.priv_info_dim}"
                    )
                self.tactile_latent_dim = self.tactile_encoder.output_dim
                self.use_tactile_history = True
                self.tactile_history_len = self.tactile_encoder.history_len
                self.tactile_frame_dim = self.tactile_encoder.frame_dim
                mlp_input_shape += self.base_priv_dim + self.tactile_latent_dim
            else:
                # Keep this branch byte-for-byte compatible at the module/state level:
                # legacy teachers still flatten the entire priv_info through env_mlp.
                mlp_input_shape += self.priv_mlp[-1]
                self.env_mlp = MLP(units=self.priv_mlp, input_size=self.priv_info_dim)
                self.tactile_latent_dim = 0
                self.use_tactile_history = False
                self.tactile_history_len = 0
                self.tactile_frame_dim = 0

                if self.priv_info_stage2:
                    self.adapt_tconv = ProprioAdaptTConv(
                        obs_per_step=obs_per_step,
                        output_dim=self.priv_mlp[-1],
                    )

        self.actor_input_dim = int(mlp_input_shape)
        self.actor_mlp = MLP(units=self.units, input_size=mlp_input_shape)
        self.value = torch.nn.Linear(out_size, 1)
        self.mu = torch.nn.Linear(out_size, actions_num)
        self.sigma = nn.Parameter(
            torch.zeros(actions_num, requires_grad=True, dtype=torch.float32),
            requires_grad=True,
        )

        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Conv1d)):
                fan_out = module.kernel_size[0] * module.out_channels
                module.weight.data.normal_(mean=0.0, std=np.sqrt(2.0 / fan_out))
                if getattr(module, "bias", None) is not None:
                    torch.nn.init.zeros_(module.bias)
            if isinstance(module, nn.Linear) and getattr(module, "bias", None) is not None:
                torch.nn.init.zeros_(module.bias)
        nn.init.constant_(self.sigma, 0)

    def _encode_priv_info(self, priv_info, tactile_hist: torch.Tensor | None = None):
        if self.tactile_encoder_type == "mlp":
            return self.env_mlp(priv_info)
        if tactile_hist is None:
            raise KeyError(
                "finger_attention_gru Teacher requires obs_dict['tactile_hist']; "
                "detailed tactile data is never read from the priv_info tail"
            )
        return self.tactile_encoder(tactile_hist)

    def get_tactile_latent(self, tactile_hist: torch.Tensor) -> torch.Tensor:
        """Encode detailed history for optional Stage-2 latent distillation."""
        if self.tactile_encoder_type != "finger_attention_gru":
            raise RuntimeError(
                "The legacy flat-MLP Teacher has no separate tactile latent."
            )
        return self.tactile_encoder(tactile_hist)

    @property
    def tactile_latent_output_dim(self) -> int:
        """Width of the per-forward tactile latent, or 0 when there is none.

        Only the ``finger_attention_gru`` teacher exposes a latent: the final
        GRU hidden state produced by the same forward pass that computes the
        action distribution. The legacy flat-MLP teacher returns 0.
        """
        if self.tactile_encoder_type != "finger_attention_gru":
            return 0
        return int(self.tactile_latent_dim)

    @torch.no_grad()
    def act(self, obs_dict):
        mu, logstd, value, _, _, features, tactile_latent = self._actor_critic(
            obs_dict, return_features=True
        )
        sigma = torch.exp(logstd)
        distr = torch.distributions.Normal(mu, sigma)
        selected_action = distr.sample()
        return {
            "neglogpacs": -distr.log_prob(selected_action).sum(1),
            "values": value,
            "actions": selected_action,
            "mus": mu,
            "sigmas": sigma,
            "features": features,
            # Same-forward tactile latent so a follower policy never re-runs the
            # tactile encoder. ``None`` for the legacy flat-MLP teacher.
            "tactile_latent": tactile_latent,
        }

    @torch.no_grad()
    def act_inference(self, obs_dict):
        mu, _, _, _, _ = self._actor_critic(obs_dict)
        return mu

    @torch.no_grad()
    def act_inference_with_latent(self, obs_dict):
        """Return the deterministic action and the same-forward tactile latent.

        Used at evaluation time by a hierarchical follower so the tactile
        encoder still runs exactly once per control cycle.
        """
        mu, _logstd, _value, _extrin, _extrin_gt, _features, tactile_latent = (
            self._actor_critic(obs_dict, return_features=True)
        )
        return mu, tactile_latent

    def _actor_critic(self, obs_dict, return_features=False):
        obs = obs_dict["obs"]
        extrin, extrin_gt = None, None
        # Only the structured teacher has a tactile latent; it is exactly the
        # encoder output of this forward pass, never a recomputation.
        tactile_latent = None
        if self.priv_info:
            if self.tactile_encoder_type == "finger_attention_gru":
                priv_info = obs_dict["priv_info"]
                if priv_info.ndim != 2 or priv_info.shape[-1] != self.priv_info_dim:
                    raise ValueError(
                        "finger_attention_gru Teacher expected priv_info shape "
                        f"[B, {self.priv_info_dim}], got {tuple(priv_info.shape)}"
                    )
                if "tactile_hist" not in obs_dict:
                    raise KeyError(
                        "finger_attention_gru Teacher requires obs_dict['tactile_hist'] "
                        f"with shape [B, {self.tactile_history_len}, "
                        f"{self.tactile_frame_dim}]"
                    )
                extrin = self._encode_priv_info(
                    priv_info,
                    obs_dict["tactile_hist"],
                )
                tactile_latent = extrin
                # Base task privilege is intentionally passed through raw: no env_mlp,
                # learned projection, normalization, LayerNorm, or tanh.
                obs = torch.cat(
                    [obs, priv_info[:, : self.base_priv_dim], extrin],
                    dim=-1,
                )
            elif self.priv_info_stage2:
                extrin = self.adapt_tconv(obs_dict["proprio_hist"])
                extrin_gt = (
                    self._encode_priv_info(obs_dict["priv_info"])
                    if "priv_info" in obs_dict
                    else extrin
                )
                extrin_gt = torch.tanh(extrin_gt)
                extrin = torch.tanh(extrin)
                obs = torch.cat([obs, extrin], dim=-1)
            else:
                extrin = self._encode_priv_info(obs_dict["priv_info"])
                extrin = torch.tanh(extrin)
                obs = torch.cat([obs, extrin], dim=-1)

        x = self.actor_mlp(obs)
        value = self.value(x)
        mu = self.mu(x)
        sigma = self.sigma
        result = (mu, mu * 0 + sigma, value, extrin, extrin_gt)
        if return_features:
            return result + (x, tactile_latent)
        return result

    def forward(self, input_dict):
        prev_actions = input_dict.get("prev_actions", None)
        (
            mu,
            logstd,
            value,
            extrin,
            extrin_gt,
            features,
            tactile_latent,
        ) = self._actor_critic(input_dict, return_features=True)
        sigma = torch.exp(logstd)
        distr = torch.distributions.Normal(mu, sigma)
        entropy = distr.entropy().sum(dim=-1)
        prev_neglogp = -distr.log_prob(prev_actions).sum(1)
        return {
            "prev_neglogp": torch.squeeze(prev_neglogp),
            "values": value,
            "entropy": entropy,
            "mus": mu,
            "sigmas": sigma,
            "extrin": extrin,
            "extrin_gt": extrin_gt,
            "features": features,
            "tactile_latent": tactile_latent,
        }


class FollowerActorCritic(nn.Module):
    """Independent 2-D horizontal-translation follower policy.

    The actor consumes exactly the deployable 159-D follower observation
    (executed hand action + master tactile latent + XY stage self-state) and
    emits a diagonal Gaussian over the two normalized translation channels.

    The critic is *centralized*: it may additionally read a slice of the base
    privileged task info. That slice never reaches the actor, so the learned
    policy stays deployable while the value baseline can use privileged state.
    """

    def __init__(
        self,
        obs_dim: int,
        actions_num: int = 2,
        actor_units=(256, 128, 64),
        critic_units=(256, 128, 64),
        critic_priv_dim: int = 0,
        init_log_sigma: float = 0.0,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.actions_num = int(actions_num)
        self.critic_priv_dim = int(critic_priv_dim)
        if self.obs_dim <= 0:
            raise ValueError(f"Follower obs_dim must be positive, got {self.obs_dim}")
        if self.actions_num <= 0:
            raise ValueError(
                f"Follower actions_num must be positive, got {self.actions_num}"
            )
        if self.critic_priv_dim < 0:
            raise ValueError(
                f"Follower critic_priv_dim must be non-negative, got {self.critic_priv_dim}"
            )

        self.actor_units = list(actor_units)
        self.critic_units = list(critic_units)
        self.actor_mlp = MLP(units=self.actor_units, input_size=self.obs_dim)
        self.mu = nn.Linear(self.actor_units[-1], self.actions_num)
        self.sigma = nn.Parameter(
            torch.full((self.actions_num,), float(init_log_sigma), dtype=torch.float32),
            requires_grad=True,
        )
        self.critic_input_dim = self.obs_dim + self.critic_priv_dim
        self.critic_mlp = MLP(units=self.critic_units, input_size=self.critic_input_dim)
        self.value = nn.Linear(self.critic_units[-1], 1)

        for module in self.modules():
            if isinstance(module, nn.Linear) and getattr(module, "bias", None) is not None:
                torch.nn.init.zeros_(module.bias)

    def _check_actor_obs(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.ndim != 2 or obs.shape[-1] != self.obs_dim:
            raise RuntimeError(
                f"Follower actor observation must have shape [B, {self.obs_dim}], "
                f"got {tuple(obs.shape)}"
            )
        return obs

    def _critic_input(self, obs: torch.Tensor, critic_priv: torch.Tensor | None) -> torch.Tensor:
        if self.critic_priv_dim == 0:
            if critic_priv is not None and critic_priv.shape[-1] != 0:
                raise RuntimeError(
                    "Follower critic was built without privileged input but received "
                    f"{critic_priv.shape[-1]} extra channels"
                )
            return obs
        if critic_priv is None:
            raise RuntimeError(
                "Follower centralized critic requires "
                f"{self.critic_priv_dim} privileged channels, got None"
            )
        if critic_priv.ndim != 2 or critic_priv.shape[-1] != self.critic_priv_dim:
            raise RuntimeError(
                "Follower critic privileged input must have shape "
                f"[B, {self.critic_priv_dim}], got {tuple(critic_priv.shape)}"
            )
        return torch.cat([obs, critic_priv], dim=-1)

    def _distribution(self, obs: torch.Tensor):
        features = self.actor_mlp(self._check_actor_obs(obs))
        mu = self.mu(features)
        sigma = torch.exp(self.sigma).expand_as(mu)
        return mu, sigma, torch.distributions.Normal(mu, sigma)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, critic_priv: torch.Tensor | None = None) -> dict:
        """Sample one follower action and evaluate its centralized value."""
        mu, sigma, distr = self._distribution(obs)
        actions = distr.sample()
        values = self.value(self.critic_mlp(self._critic_input(obs, critic_priv)))
        return {
            "actions": actions,
            "neglogpacs": -distr.log_prob(actions).sum(dim=-1),
            "values": values,
            "mus": mu,
            "sigmas": sigma,
        }

    @torch.no_grad()
    def act_inference(self, obs: torch.Tensor) -> torch.Tensor:
        """Return the deterministic follower action."""
        mu, _sigma, _distr = self._distribution(obs)
        return mu

    def forward(self, input_dict: dict) -> dict:
        """Evaluate stored follower actions for one PPO minibatch."""
        obs = input_dict["obs"]
        prev_actions = input_dict["prev_actions"]
        mu, sigma, distr = self._distribution(obs)
        values = self.value(
            self.critic_mlp(self._critic_input(obs, input_dict.get("critic_priv")))
        )
        return {
            "prev_neglogp": torch.squeeze(-distr.log_prob(prev_actions).sum(dim=-1)),
            "values": values,
            "entropy": distr.entropy().sum(dim=-1),
            "mus": mu,
            "sigmas": sigma,
        }
