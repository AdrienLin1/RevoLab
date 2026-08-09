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
    """Resolve teacher tactile encoder type; only ``mlp`` is supported publicly."""
    del legacy_key
    tactile_cfg = _cfg_get(network_config, "tactile_encoder", None)
    encoder_type = str(_cfg_get(tactile_cfg, "type", "mlp")).lower()
    if encoder_type != "mlp":
        raise ValueError(
            f"Unsupported teacher tactile encoder type {encoder_type!r}; "
            "public release supports tactile_encoder.type=mlp only."
        )
    return "mlp", tactile_cfg


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
    keys = state_dict.keys() if isinstance(state_dict, dict) else ()
    if any(str(key).startswith("tactile_encoder.") for key in keys):
        raise ValueError(
            "Checkpoint uses a tactile_encoder module; public release supports MLP teachers only."
        )
    return "mlp"


def infer_teacher_tactile_architecture_from_state_dict(
    state_dict,
) -> tuple[str | None, str | None]:
    """Return layout/history placeholders for MLP-only public checkpoints."""
    del state_dict
    return None, None


def validate_teacher_tactile_checkpoint_compatibility(
    checkpoint_state,
    runtime_state,
    checkpoint_path: str = "",
) -> None:
    """Reject teacher checkpoints built for a different tactile architecture."""
    checkpoint_type = infer_teacher_tactile_encoder_type_from_state_dict(checkpoint_state)
    runtime_type = infer_teacher_tactile_encoder_type_from_state_dict(runtime_state)
    if checkpoint_type == runtime_type:
        return
    location = f"\n  checkpoint: {checkpoint_path}" if checkpoint_path else ""
    raise RuntimeError(
        "Checkpoint tactile architecture is incompatible with the runtime model."
        f"{location}\n"
        f"  - tactile encoder type: checkpoint={checkpoint_type}, runtime={runtime_type}\n"
        "Use an MLP teacher checkpoint with Revo3HandScrewTactile or Revo3HandTactileRotate."
    )


def _network_mlp_units(network_config, key):
    block = _cfg_get(network_config, key, None)
    units = _cfg_get(block, "units", None)
    if units is None:
        raise KeyError(f"train.network.{key}.units is required")
    return list(units)


def build_actor_critic_kwargs(
    network_config, ppo_config, actions_num, input_shape, obs_per_step, proprio_adapt
):
    """Build kwargs for ``ActorCritic`` from train yaml network/ppo sections."""
    tactile_type, tactile_cfg = resolve_tactile_encoder_type(network_config)
    return {
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
    """Stage1 teacher policy with flat MLP privilege encoding."""

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
        kwargs.pop("tactile_encoder_cfg", None)

        if self.priv_info:
            mlp_input_shape += self.priv_mlp[-1]
            self.priv_info_dim = int(kwargs["priv_info_dim"])
            if self.tactile_encoder_type != "mlp":
                raise ValueError(
                    f"Unsupported teacher tactile encoder type: {self.tactile_encoder_type}"
                )
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
        del tactile_hist
        return self.env_mlp(priv_info)

    @torch.no_grad()
    def act(self, obs_dict):
        mu, logstd, value, _, _, features = self._actor_critic(
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
        }

    @torch.no_grad()
    def act_inference(self, obs_dict):
        mu, _, _, _, _ = self._actor_critic(obs_dict)
        return mu

    def _actor_critic(self, obs_dict, return_features=False):
        obs = obs_dict["obs"]
        extrin, extrin_gt = None, None
        if self.priv_info:
            if self.priv_info_stage2:
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
            return result + (x,)
        return result

    def forward(self, input_dict):
        prev_actions = input_dict.get("prev_actions", None)
        mu, logstd, value, extrin, extrin_gt, features = self._actor_critic(
            input_dict, return_features=True
        )
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
        }
