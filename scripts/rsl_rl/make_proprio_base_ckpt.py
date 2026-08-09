#!/usr/bin/env python
"""Build a proprio-only (TacRes phase-1) checkpoint from the pretrained Lift checkpoint.

The pretrained ``BrainCo-Dexsuite-Revo3-Right-Lift-v0`` actor/critic consume
policy(195) + proprio(745) + perception(960) = 1900 dims, where the proprio group is ordered
[contact(75), joint_pos(140), joint_vel(140), hand_tips_state_b(390)] (see the observation
manager table in the training log). The TacRes phase-1 task removes the contact term, so this
script deletes input columns [195, 270) from the first MLP layer and the corresponding entries
of the empirical observation normalizers (reverse net2net surgery). All other weights are kept.

The result is NOT equivalent to a policy trained without tactile input; it is a warm start that
must be briefly fine-tuned on the phase-1 task (use train.py --init_checkpoint).

Usage:
    python scripts/rsl_rl/make_proprio_base_ckpt.py <in.pt> <out.pt> [--start 195 --width 75]
"""

import argparse

import torch


def strip_columns(state: dict, start: int, width: int) -> dict:
    out = {}
    keep = None
    for key, value in state.items():
        if key == "mlp.0.weight":
            keep = torch.cat([value[:, :start], value[:, start + width:]], dim=1)
            out[key] = keep
        elif key.startswith("obs_normalizer._"):
            out[key] = torch.cat([value[:, :start], value[:, start + width:]], dim=1)
        else:
            out[key] = value
    assert keep is not None, "state dict has no mlp.0.weight"
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--start", type=int, default=195, help="First column of the contact slice.")
    parser.add_argument("--width", type=int, default=75, help="Width of the contact slice.")
    args = parser.parse_args()

    loaded = torch.load(args.input, weights_only=False, map_location="cpu")
    in_dim = loaded["actor_state_dict"]["mlp.0.weight"].shape[1]
    converted = {
        "actor_state_dict": strip_columns(loaded["actor_state_dict"], args.start, args.width),
        "critic_state_dict": strip_columns(loaded["critic_state_dict"], args.start, args.width),
        "iter": 0,
        "infos": None,
    }
    out_dim = converted["actor_state_dict"]["mlp.0.weight"].shape[1]
    torch.save(converted, args.output)
    print(f"Wrote {args.output}: actor/critic input {in_dim} -> {out_dim}"
          f" (removed columns [{args.start}, {args.start + args.width}))")


if __name__ == "__main__":
    main()
