#!/usr/bin/env python
"""Convert a legacy rsl-rl checkpoint (combined ``model_state_dict`` with in-module obs
normalizers) into the rsl-rl >= 5.0 layout (separate ``actor_state_dict`` / ``critic_state_dict``).

The stock ``handle_deprecated_rsl_rl_checkpoint`` only converts the very old layout
(``actor.* / critic.* / std`` and nothing else). Checkpoints that also carry
``actor_obs_normalizer.* / critic_obs_normalizer.*`` inside ``model_state_dict`` trip its
"Unrecognized key" guard. This script handles that hybrid layout.

Key remap (per rsl_rl.models.mlp_model.MLPModel):
    actor.<i>.*              -> mlp.<i>.*
    actor_obs_normalizer.*   -> obs_normalizer.*
    std                      -> distribution.std_param            (actor only)
    critic.<i>.*             -> mlp.<i>.*
    critic_obs_normalizer.*  -> obs_normalizer.*

Usage:
    python scripts/rsl_rl/convert_rslrl_ckpt.py <in.pt> [out.pt]
"""

import os
import sys

import torch


def convert(in_path: str, out_path: str) -> None:
    loaded = torch.load(in_path, weights_only=False, map_location="cpu")
    if not isinstance(loaded, dict) or "model_state_dict" not in loaded:
        raise ValueError(f"'{in_path}' has no 'model_state_dict'; nothing to convert.")
    if "actor_state_dict" in loaded:
        raise ValueError(f"'{in_path}' already uses the new rsl-rl>=5.0 layout.")

    msd = loaded["model_state_dict"]
    actor: dict[str, torch.Tensor] = {}
    critic: dict[str, torch.Tensor] = {}

    for key, value in msd.items():
        if key == "std":
            actor["distribution.std_param"] = value
        elif key.startswith("actor_obs_normalizer."):
            actor["obs_normalizer." + key[len("actor_obs_normalizer.") :]] = value
        elif key.startswith("critic_obs_normalizer."):
            critic["obs_normalizer." + key[len("critic_obs_normalizer.") :]] = value
        elif key.startswith("actor."):
            actor["mlp." + key[len("actor.") :]] = value
        elif key.startswith("critic."):
            critic["mlp." + key[len("critic.") :]] = value
        else:
            raise ValueError(f"Unrecognized key '{key}' in model_state_dict; cannot convert safely.")

    converted = {
        "actor_state_dict": actor,
        "critic_state_dict": critic,
        "optimizer_state_dict": loaded.get("optimizer_state_dict"),
        "iter": loaded.get("iter", 0),
        "infos": loaded.get("infos"),
    }
    torch.save(converted, out_path)

    print(f"[convert] wrote {out_path}")
    print(f"[convert] actor keys ({len(actor)}):")
    for k in sorted(actor):
        print(f"   {k:32s} {tuple(actor[k].shape) if hasattr(actor[k], 'shape') else type(actor[k])}")
    print(f"[convert] critic keys ({len(critic)}):")
    for k in sorted(critic):
        print(f"   {k:32s} {tuple(critic[k].shape) if hasattr(critic[k], 'shape') else type(critic[k])}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    src = sys.argv[1]
    dst = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(src)[0] + "_rslrl5.pt"
    convert(src, dst)
