"""Unit tests for the structured frame813 Stage-1 tactile teacher."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import yaml

from BrainCo_DexHand.algo.hora.models.models import (
    ActorCritic,
    FingerAttentionGRUTactileEncoder,
    build_actor_critic_kwargs,
    infer_teacher_tactile_encoder_type_from_state_dict,
    resolve_tactile_encoder_type,
    validate_teacher_tactile_checkpoint_compatibility,
)
from BrainCo_DexHand.algo.hora.ppo.experience import ExperienceBuffer
REPO_ROOT = Path(__file__).resolve().parents[1]
AGENT_DIR = (
    REPO_ROOT
    / "source/BrainCo_DexHand/BrainCo_DexHand/tasks/direct/hora_screw/agents"
)
FRAME813_YAML = AGENT_DIR / "valvedriver_tactile_frame813.yaml"
LEGACY_YAML = AGENT_DIR / "Revo3HandScrewTactile.yaml"
CANONICAL_FINGERS = ("thumb", "index", "middle", "ring", "little")
TACTILE_LAYOUT_PATH = (
    REPO_ROOT / "source/BrainCo_DexHand/BrainCo_DexHand/tasks/tactile_layout.py"
)
_LAYOUT_SPEC = importlib.util.spec_from_file_location(
    "frame813_test_tactile_layout",
    TACTILE_LAYOUT_PATH,
)
assert _LAYOUT_SPEC is not None and _LAYOUT_SPEC.loader is not None
_TACTILE_LAYOUT = importlib.util.module_from_spec(_LAYOUT_SPEC)
_LAYOUT_SPEC.loader.exec_module(_TACTILE_LAYOUT)
normalized_estimated_official_centers_xy = (
    _TACTILE_LAYOUT.normalized_estimated_official_centers_xy
)

TASK_CASES = (
    pytest.param(
        ("thumb", "index", "middle"),
        (31, 21, 21),
        11,
        742,
        753,
        id="nutbolt-3-finger",
    ),
    pytest.param(
        ("thumb", "index", "middle", "ring"),
        (31, 21, 21, 21),
        11,
        956,
        967,
        id="screwdriver-4-finger",
    ),
    pytest.param(
        CANONICAL_FINGERS,
        (31, 21, 21, 21, 21),
        11,
        1170,
        1181,
        id="valve-5-finger",
    ),
    pytest.param(
        CANONICAL_FINGERS,
        (31, 21, 21, 21, 21),
        8,
        1170,
        1178,
        id="rotation-5-finger",
    ),
)


def _network_config(path: Path = FRAME813_YAML):
    return yaml.safe_load(path.read_text(encoding="utf-8"))["network"]


def _ppo_config(priv_info_dim: int, path: Path = FRAME813_YAML):
    ppo = yaml.safe_load(path.read_text(encoding="utf-8"))["ppo"]
    ppo["priv_info_dim"] = int(priv_info_dim)
    return ppo


def _env_cfg(
    names: tuple[str, ...],
    counts: tuple[int, ...],
    base_priv_dim: int,
    frame_dim: int,
    priv_info_dim: int,
):
    return SimpleNamespace(
        task=f"pytest_{len(names)}finger",
        tactile_layout="estimated_official",
        tactile_priv_offset=base_priv_dim,
        tactile_active_finger_names=names,
        tactile_graph_sensor_counts=counts,
        tactile_graph_total_nodes=sum(counts),
        tactile_graph_finger_positions=tuple(
            tuple(
                tuple(float(value) for value in row)
                for row in normalized_estimated_official_centers_xy(name)
            )
            for name in names
        ),
        teacher_tactile_frame_dim=frame_dim,
        teacher_tactile_history_len=10,
        tactile_graph_common_channels=5,
        tactile_graph_force_channels=5,
        tactile_graph_context_channels=4,
        priv_info_dim=priv_info_dim,
        observation_space=141,
    )


def _build_model(
    names: tuple[str, ...],
    counts: tuple[int, ...],
    base_priv_dim: int,
    frame_dim: int,
    priv_info_dim: int,
    *,
    network_path: Path = FRAME813_YAML,
):
    env_cfg = _env_cfg(names, counts, base_priv_dim, frame_dim, priv_info_dim)
    kwargs = build_actor_critic_kwargs(
        _network_config(network_path),
        _ppo_config(priv_info_dim, network_path),
        actions_num=21,
        input_shape=(141,),
        obs_per_step=47,
        proprio_adapt=False,
        env_cfg=env_cfg,
    )
    return ActorCritic(copy.deepcopy(kwargs)), env_cfg, kwargs


def _make_history(batch: int, history_len: int, total_nodes: int, frame_dim: int):
    """Create distinguishable values in every raw node channel and context."""

    history = torch.zeros(batch, history_len, frame_dim)
    nodes = history[..., : total_nodes * 10].view(batch, history_len, total_nodes, 10)
    batch_term = torch.arange(batch, dtype=torch.float32).view(batch, 1, 1, 1) * 100.0
    time_term = torch.arange(history_len, dtype=torch.float32).view(1, history_len, 1, 1)
    node_term = torch.arange(total_nodes, dtype=torch.float32).view(1, 1, total_nodes, 1) / 100.0
    channel_term = torch.arange(10, dtype=torch.float32).view(1, 1, 1, 10) / 1000.0
    nodes.copy_(batch_term + time_term + node_term + channel_term)
    history[..., total_nodes * 10 :] = 10_000.0
    return history


def _first_module(root: nn.Module, module_type):
    return next(module for module in root.modules() if isinstance(module, module_type))


def _finger_mlp(encoder: FingerAttentionGRUTactileEncoder, finger_name: str):
    finger_mlps = getattr(encoder, "finger_mlps")
    return finger_mlps[finger_name]


def test_teacher_encoder_type_resolution_is_explicit_and_strict():
    assert resolve_tactile_encoder_type(
        {"tactile_encoder": {"type": "mlp"}}
    )[0] == "mlp"
    assert resolve_tactile_encoder_type(
        {"tactile_encoder": {"type": "finger_attention_gru"}}
    )[0] == "finger_attention_gru"
    with pytest.raises(ValueError, match="unsupported_teacher"):
        resolve_tactile_encoder_type(
            {"tactile_encoder": {"type": "unsupported_teacher"}}
        )


@pytest.mark.parametrize(
    ("names", "counts", "base_priv_dim", "frame_dim", "priv_info_dim"),
    TASK_CASES,
)
def test_runtime_metadata_builds_task_resolved_teacher(
    names,
    counts,
    base_priv_dim,
    frame_dim,
    priv_info_dim,
):
    model, env_cfg, kwargs = _build_model(
        names, counts, base_priv_dim, frame_dim, priv_info_dim
    )

    assert isinstance(model.tactile_encoder, FingerAttentionGRUTactileEncoder)
    assert model.tactile_encoder_type == "finger_attention_gru"
    assert model.use_tactile_history is True
    assert model.tactile_history_len == 10
    assert model.tactile_frame_dim == frame_dim
    assert model.tactile_latent_dim == 128
    assert model.base_priv_dim == base_priv_dim
    assert model.tactile_priv_offset == base_priv_dim
    assert tuple(model.tactile_encoder.active_finger_names) == names
    assert tuple(model.tactile_encoder.finger_counts) == counts
    assert kwargs["tactile_env_metadata"]["active_finger_names"] == names
    assert kwargs["tactile_env_metadata"]["finger_counts"] == counts
    assert (
        kwargs["tactile_env_metadata"]["finger_positions"]
        == env_cfg.tactile_graph_finger_positions
    )
    assert not hasattr(model, "env_mlp")


@pytest.mark.parametrize(
    ("names", "counts", "base_priv_dim", "frame_dim", "priv_info_dim"),
    TASK_CASES[:3],
)
def test_each_finger_mlp_consumes_the_whole_ordered_finger_without_pooling(
    names,
    counts,
    base_priv_dim,
    frame_dim,
    priv_info_dim,
):
    model, _, _ = _build_model(names, counts, base_priv_dim, frame_dim, priv_info_dim)
    encoder = model.tactile_encoder
    parameter_ids: list[set[int]] = []

    for name, count in zip(names, counts):
        finger_mlp = _finger_mlp(encoder, name)
        linears = [module for module in finger_mlp.modules() if isinstance(module, nn.Linear)]
        assert [(layer.in_features, layer.out_features) for layer in linears] == [
            (count * 7, 64),
            (64, 32),
        ]
        parameter_ids.append({id(parameter) for parameter in finger_mlp.parameters()})

    for index, left in enumerate(parameter_ids):
        for right in parameter_ids[index + 1 :]:
            assert left.isdisjoint(right)


def test_node_feature_extraction_uses_uv_b_d_and_force_only():
    names = ("thumb", "index", "middle")
    counts = (31, 21, 21)
    model, env_cfg, _ = _build_model(names, counts, 11, 742, 753)
    encoder = model.tactile_encoder.eval()
    batch, history_len = 2, 10
    history = _make_history(batch, history_len, sum(counts), 742)
    captured: dict[str, torch.Tensor] = {}
    handles = []

    def capture_input(finger_name):
        def hook(_module, inputs):
            captured[finger_name] = inputs[0].detach().clone()

        return hook

    for name in names:
        handles.append(
            _finger_mlp(encoder, name).register_forward_pre_hook(
                capture_input(name)
            )
        )
    try:
        latent = encoder(history)
    finally:
        for handle in handles:
            handle.remove()

    assert latent.shape == (batch, 128)
    raw_nodes = history[..., : sum(counts) * 10].view(
        batch, history_len, sum(counts), 10
    )
    selected = raw_nodes[..., [0, 3, 5, 6, 7]]
    start = 0
    for name, count in zip(names, counts):
        stop = start + count
        xy = torch.tensor(
            env_cfg.tactile_graph_finger_positions[names.index(name)],
            dtype=history.dtype,
        ).view(1, 1, count, 2).expand(batch, history_len, -1, -1)
        expected = torch.cat((xy, selected[..., start:stop, :]), dim=-1).flatten(-2)
        assert captured[name].reshape(batch, history_len, -1).shape == expected.shape
        torch.testing.assert_close(
            captured[name].reshape(batch, history_len, -1), expected
        )
        start = stop

    ignored_change = history.clone()
    ignored_nodes = ignored_change[..., : sum(counts) * 10].view(
        batch, history_len, sum(counts), 10
    )
    ignored_nodes[..., [1, 2, 4, 8, 9]] += 1234.0
    ignored_change[..., sum(counts) * 10 :] -= 4321.0
    with torch.no_grad():
        original_latent = encoder(history)
        ignored_latent = encoder(ignored_change)
    torch.testing.assert_close(original_latent, ignored_latent, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    ("names", "counts", "frame_dim", "gru_input_dim"),
    (
        (("thumb", "index", "middle"), (31, 21, 21), 742, 96),
        (("thumb", "index", "middle", "ring"), (31, 21, 21, 21), 956, 128),
        (CANONICAL_FINGERS, (31, 21, 21, 21, 21), 1170, 160),
    ),
)
def test_attention_and_gru_keep_all_active_finger_tokens(
    names,
    counts,
    frame_dim,
    gru_input_dim,
):
    model, _, _ = _build_model(names, counts, 11, frame_dim, 11 + frame_dim)
    encoder = model.tactile_encoder.train()
    attention = _first_module(encoder, nn.MultiheadAttention)
    gru = _first_module(encoder, nn.GRU)
    captured = {}

    attention_handle = attention.register_forward_hook(
        lambda _module, inputs, output: captured.update(
            attention_input=inputs[0].detach(), attention_output=output[0].detach()
        )
    )
    gru_handle = gru.register_forward_hook(
        lambda _module, inputs, output: captured.update(
            gru_input=inputs[0].detach(),
            gru_output=output[0].detach(),
            gru_hidden=output[1].detach(),
        )
    )
    try:
        node_inputs = encoder.extract_node_inputs(torch.randn(2, 10, frame_dim))
        finger_tokens = encoder.encode_fingers(node_inputs)
        attended_tokens = encoder.apply_finger_attention(finger_tokens)
        latent = encoder.encode_temporal(attended_tokens)
    finally:
        attention_handle.remove()
        gru_handle.remove()

    finger_count = len(names)
    assert finger_tokens.shape == (2, 10, finger_count, 32)
    assert attended_tokens.shape == (2, 10, finger_count, 32)
    assert captured["attention_input"].shape == (2 * 10, finger_count, 32)
    assert captured["attention_output"].shape == (2 * 10, finger_count, 32)
    assert attention.embed_dim == 32
    assert attention.num_heads == 4
    transformer_layer = _first_module(encoder, nn.TransformerEncoderLayer)
    assert transformer_layer.linear1.out_features == 64
    assert transformer_layer.linear2.in_features == 64
    assert captured["gru_input"].shape == (2, 10, gru_input_dim)
    assert captured["gru_output"].shape == (2, 10, 128)
    assert captured["gru_hidden"].shape == (1, 2, 128)
    assert gru.input_size == gru_input_dim
    assert gru.hidden_size == 128
    assert gru.num_layers == 1
    assert gru.bidirectional is False
    assert gru.dropout == pytest.approx(0.0)
    assert latent.shape == (2, 128)


def test_encoder_keeps_five_learnable_identities_and_buffered_physical_coordinates():
    names = ("thumb", "index", "middle")
    counts = (31, 21, 21)
    model, env_cfg, _ = _build_model(names, counts, 11, 742, 753)
    encoder = model.tactile_encoder

    identity_parameters = [
        parameter
        for name, parameter in encoder.named_parameters()
        if "identity" in name or "finger_embedding" in name
    ]
    assert any(tuple(parameter.shape) == (5, 32) for parameter in identity_parameters)

    expected_xy = torch.cat(
        [
            torch.tensor(value, dtype=torch.float32)
            for value in env_cfg.tactile_graph_finger_positions
        ],
        dim=0,
    )
    coordinate_buffers = [
        value
        for name, value in encoder.named_buffers()
        if ("position" in name or "coord" in name or "xy" in name)
        and value.ndim == 2
        and value.shape[-1] == 2
    ]
    assert any(
        value.shape == expected_xy.shape
        and torch.allclose(value.cpu(), expected_xy)
        for value in coordinate_buffers
    )
    coordinate_parameter_names = [
        name
        for name, _parameter in encoder.named_parameters()
        if "position" in name or "coord" in name or "xy" in name
    ]
    assert coordinate_parameter_names == []


@pytest.mark.parametrize(
    ("names", "counts", "base_priv_dim", "frame_dim", "priv_info_dim", "actor_input_dim"),
    (
        (("thumb", "index", "middle"), (31, 21, 21), 11, 742, 753, 280),
        (CANONICAL_FINGERS, (31, 21, 21, 21, 21), 11, 1170, 1181, 280),
        (CANONICAL_FINGERS, (31, 21, 21, 21, 21), 8, 1170, 1178, 277),
    ),
)
def test_full_teacher_forward_directly_fuses_obs_base_priv_and_tactile_latent(
    names,
    counts,
    base_priv_dim,
    frame_dim,
    priv_info_dim,
    actor_input_dim,
):
    torch.manual_seed(11)
    model, _, _ = _build_model(names, counts, base_priv_dim, frame_dim, priv_info_dim)
    model.eval()
    obs = torch.randn(2, 141)
    priv_info = torch.randn(2, priv_info_dim)
    tactile_hist = torch.randn(2, 10, frame_dim)
    captured = {}
    encoder_handle = model.tactile_encoder.register_forward_hook(
        lambda _module, _inputs, output: captured.update(latent=output.detach().clone())
    )
    actor_handle = model.actor_mlp.register_forward_pre_hook(
        lambda _module, inputs: captured.update(actor_input=inputs[0].detach().clone())
    )
    try:
        mu, _logstd, value, _extrin, _extrin_gt = model._actor_critic(
            {"obs": obs, "priv_info": priv_info, "tactile_hist": tactile_hist}
        )
    finally:
        encoder_handle.remove()
        actor_handle.remove()

    fused = captured["actor_input"]
    assert fused.shape == (2, actor_input_dim)
    torch.testing.assert_close(fused[:, :141], obs)
    torch.testing.assert_close(fused[:, 141 : 141 + base_priv_dim], priv_info[:, :base_priv_dim])
    torch.testing.assert_close(fused[:, 141 + base_priv_dim :], captured["latent"])
    assert mu.shape == (2, 21)
    assert value.shape == (2, 1)
    first_actor_linear = _first_module(model.actor_mlp, nn.Linear)
    assert first_actor_linear.in_features == actor_input_dim
    assert not hasattr(model, "env_mlp")

    changed_tail = priv_info.clone()
    changed_tail[:, base_priv_dim:] += 1000.0
    changed_base = priv_info.clone()
    changed_base[:, 0] += 10.0
    with torch.no_grad():
        original_mu = model.act_inference(
            {"obs": obs, "priv_info": priv_info, "tactile_hist": tactile_hist}
        )
        tail_mu = model.act_inference(
            {"obs": obs, "priv_info": changed_tail, "tactile_hist": tactile_hist}
        )
        base_mu = model.act_inference(
            {"obs": obs, "priv_info": changed_base, "tactile_hist": tactile_hist}
        )
        changed_history = tactile_hist.clone()
        changed_history[:, -1, 5] += 10.0
        tactile_mu = model.act_inference(
            {"obs": obs, "priv_info": priv_info, "tactile_hist": changed_history}
        )
    torch.testing.assert_close(original_mu, tail_mu, rtol=0.0, atol=0.0)
    assert not torch.allclose(original_mu, base_mu)
    assert not torch.allclose(original_mu, tactile_mu)


def test_temporal_order_changes_latent_and_gru_receives_gradient():
    model, _, _ = _build_model(
        ("thumb", "index", "middle"), (31, 21, 21), 11, 742, 753
    )
    encoder = model.tactile_encoder.eval()
    history = torch.randn(3, 10, 742)
    history += torch.arange(10, dtype=history.dtype).view(1, 10, 1)

    with torch.no_grad():
        forward_latent = encoder(history)
        reversed_latent = encoder(history.flip(1))
    assert not torch.allclose(forward_latent, reversed_latent)

    encoder.train()
    encoder.zero_grad(set_to_none=True)
    loss = encoder(history).square().mean()
    loss.backward()
    gru = _first_module(encoder, nn.GRU)
    assert gru.weight_ih_l0.grad is not None
    assert torch.isfinite(gru.weight_ih_l0.grad).all()
    assert torch.count_nonzero(gru.weight_ih_l0.grad).item() > 0


def test_ppo_experience_buffer_round_trips_teacher_history():
    buffer = ExperienceBuffer(
        num_envs=2,
        horizon_length=2,
        batch_size=4,
        minibatch_size=2,
        obs_dim=141,
        act_dim=21,
        priv_dim=753,
        device="cpu",
        tactile_hist_shape=(10, 742),
    )
    first = torch.randn(2, 10, 742)
    second = torch.randn(2, 10, 742)
    buffer.update_data("tactile_hist", 0, first)
    buffer.update_data("tactile_hist", 1, second)
    buffer.prepare_training(normalize_advantage=False)

    expected = torch.stack((first, second), dim=0).transpose(0, 1).reshape(4, 10, 742)
    torch.testing.assert_close(buffer.data_dict["tactile_hist"], expected)
    assert buffer[0]["tactile_hist"].shape == (2, 10, 742)


def test_old_and_new_teacher_checkpoints_are_strictly_separated():
    new_model, _, _ = _build_model(
        ("thumb", "index", "middle"), (31, 21, 21), 11, 742, 753
    )
    old_model, _, _ = _build_model(
        ("thumb", "index", "middle"),
        (31, 21, 21),
        11,
        742,
        753,
        network_path=LEGACY_YAML,
    )

    old_clone, _, _ = _build_model(
        ("thumb", "index", "middle"),
        (31, 21, 21),
        11,
        742,
        753,
        network_path=LEGACY_YAML,
    )
    old_clone.load_state_dict(old_model.state_dict(), strict=True)
    assert infer_teacher_tactile_encoder_type_from_state_dict(old_model.state_dict()) == "mlp"
    assert (
        infer_teacher_tactile_encoder_type_from_state_dict(new_model.state_dict())
        == "finger_attention_gru"
    )

    for checkpoint_state, runtime_state in (
        (old_model.state_dict(), new_model.state_dict()),
        (new_model.state_dict(), old_model.state_dict()),
    ):
        with pytest.raises(RuntimeError) as exc_info:
            validate_teacher_tactile_checkpoint_compatibility(
                checkpoint_state,
                runtime_state,
                checkpoint_path="pytest_teacher.pth",
            )
        message = str(exc_info.value)
        assert "pytest_teacher.pth" in message
        assert "mlp" in message
        assert "finger_attention_gru" in message

    with pytest.raises(RuntimeError):
        new_model.load_state_dict(old_model.state_dict(), strict=True)
    with pytest.raises(RuntimeError):
        old_model.load_state_dict(new_model.state_dict(), strict=True)


def test_structured_checkpoint_rejects_a_different_active_finger_contract():
    three_finger, _, _ = _build_model(
        ("thumb", "index", "middle"), (31, 21, 21), 11, 742, 753
    )
    five_finger, _, _ = _build_model(
        CANONICAL_FINGERS, (31, 21, 21, 21, 21), 11, 1170, 1181
    )

    with pytest.raises(RuntimeError) as exc_info:
        validate_teacher_tactile_checkpoint_compatibility(
            three_finger.state_dict(),
            five_finger.state_dict(),
            checkpoint_path="nutbolt_frame813.pth",
        )

    message = str(exc_info.value)
    assert "nutbolt_frame813.pth" in message
    assert "active finger identities" in message or "finger node counts" in message
    assert "GRU input" in message


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    (
        ("tactile_layout", "regular_grid"),
        ("teacher_tactile_history_len", 9),
        ("teacher_tactile_frame_dim", 741),
        ("tactile_active_finger_names", ("index", "thumb", "middle")),
        ("tactile_graph_sensor_counts", (30, 21, 21)),
    ),
)
def test_invalid_runtime_contract_fails_with_complete_diagnostics(field, invalid_value):
    env_cfg = _env_cfg(("thumb", "index", "middle"), (31, 21, 21), 11, 742, 753)
    setattr(env_cfg, field, invalid_value)

    with pytest.raises(ValueError) as exc_info:
        kwargs = build_actor_critic_kwargs(
            _network_config(),
            _ppo_config(753),
            actions_num=21,
            input_shape=(141,),
            obs_per_step=47,
            proprio_adapt=False,
            env_cfg=env_cfg,
        )
        ActorCritic(kwargs)

    message = str(exc_info.value)
    for required in (
        "pytest_3finger",
        "active_finger_names",
        "finger_counts",
        "total_nodes",
        "expected_frame_dim",
        "actual_frame_dim",
        "history_len",
        "layout",
    ):
        assert required in message


def test_structured_teacher_rejects_unsynced_full_priv_info_width():
    env_cfg = _env_cfg(("thumb", "index", "middle"), (31, 21, 21), 11, 742, 753)
    kwargs = build_actor_critic_kwargs(
        _network_config(),
        _ppo_config(1181),
        actions_num=21,
        input_shape=(141,),
        obs_per_step=47,
        proprio_adapt=False,
        env_cfg=env_cfg,
    )

    with pytest.raises(ValueError, match="expected_priv_info_dim=753"):
        ActorCritic(kwargs)


@pytest.mark.parametrize(
    ("shape", "missing_dimension"),
    (((2, 9, 742), "history_len"), ((2, 10, 741), "frame_dim")),
)
def test_encoder_rejects_wrong_tactile_history_shape(shape, missing_dimension):
    model, _, _ = _build_model(
        ("thumb", "index", "middle"), (31, 21, 21), 11, 742, 753
    )
    with pytest.raises(ValueError, match=missing_dimension):
        model.tactile_encoder(torch.zeros(shape))
