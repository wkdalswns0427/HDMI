#!/usr/bin/env python3
"""Warm-start a PPO-ROA checkpoint into a task whose OBJECT observation group
has grown by N dimensions appended at the end.

The motivating case: wall_painting_goal adds `target_center_b` (3 floats) to the
object group, 30 -> 33. No existing checkpoint loads into it, so it would have to
train from scratch and re-learn grip and balance before it could learn anything
about following the target. This script expands a converged checkpoint instead:

  - every first layer that consumes the object group gets N zero-weight input
    columns, so at step zero the network computes exactly what it did before and
    ignores the new inputs until training gives them weight
  - the observation normalizer's running sum and sum-of-squares for the object
    group get N entries, seeded from a supplied prior so the new inputs are
    normalized sensibly from the first step

Where the columns go is the part that must be exact, and it differs by layer
(active_adaptation/learning/ppo/ppo_roa.py):

  encoder_priv   CatTensors([priv, object], sort=False)       -> end
  adapt_module   CatTensors([policy, command, object], False) -> end
  adapt_ema      same as adapt_module                         -> end
  critic         CatTensors([priv, policy, command, object])  -> SORTED by key,
                 i.e. [command, object, policy, priv]         -> inside

Within the object group the env concatenates terms in config order with no
sorting (envs/base.py, ObsGroup), so terms appended to a task's
observation.object list land last. Every insertion point is computed from the
group sizes stored in the checkpoint and asserted against each layer's actual
width -- a mismatch aborts rather than writing a subtly wrong checkpoint.

    python pipeline/scripts/expand_object_obs.py --src <ckpt> --dst <out> \\
        --prior-mean 0.3504 0.6579 0.4375 --prior-std 0.201 0.0916 0.2317
"""

import argparse
from pathlib import Path

import torch

# The policy state dict is one level deep: each component holds its own flat
# state dict, so a first layer is addressed as policy[component][key].
FIRST_LAYER_KEY = "module.1.module.0.0.weight"
FIRST_LAYERS = ("encoder_priv", "adapt_module", "adapt_ema", "critic")


def insert_zero_columns(w: torch.Tensor, at: int, n: int) -> torch.Tensor:
    zeros = torch.zeros(w.shape[0], n, dtype=w.dtype, device=w.device)
    return torch.cat([w[:, :at], zeros, w[:, at:]], dim=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--prior-mean", type=float, nargs="+", required=True,
                    help="expected mean of each new object dim")
    ap.add_argument("--prior-std", type=float, nargs="+", required=True,
                    help="expected std of each new object dim")
    args = ap.parse_args()

    mean = torch.tensor(args.prior_mean, dtype=torch.float32)
    std = torch.tensor(args.prior_std, dtype=torch.float32)
    assert mean.shape == std.shape, "prior mean and std need the same length"
    n = mean.numel()

    sd = torch.load(args.src, map_location="cpu", weights_only=False)
    pol = sd["policy"]
    vn = sd["vecnorm"]["_extra_state"]

    size = {k: vn[f"{k}_sum"].numel() for k in ("command", "policy", "object", "priv")}
    obj = size["object"]
    print(f"group sizes in checkpoint: {size}")
    print(f"adding {n} object dims: object {obj} -> {obj + n}\n")

    # Expected (input width, insertion column) per layer, from the group sizes.
    expect = {
        "encoder_priv": (size["priv"] + obj, size["priv"] + obj),
        "adapt_module": (size["policy"] + size["command"] + obj,
                         size["policy"] + size["command"] + obj),
        "adapt_ema": (size["policy"] + size["command"] + obj,
                      size["policy"] + size["command"] + obj),
    }
    critic_order = sorted(["priv", "policy", "command", "object"])
    assert critic_order == ["command", "object", "policy", "priv"], critic_order
    expect["critic"] = (sum(size.values()), size["command"] + obj)

    for name in FIRST_LAYERS:
        key = FIRST_LAYER_KEY
        w = pol[name][key]
        width, at = expect[name]
        if w.shape[1] != width:
            raise SystemExit(f"ABORT: {name} input width {w.shape[1]} != expected "
                             f"{width}; the concatenation layout is not what this "
                             f"script assumes. Nothing written.")
        pol[name][key] = insert_zero_columns(w, at, n)
        print(f"  {name:13s} {tuple(w.shape)} -> {tuple(pol[name][key].shape)}   "
              f"{n} zero cols at {at}")

    # Normalizer: seed the new dims so mean = prior mean, var = prior var under
    # the existing (decayed) count. mean = sum/cnt, var = ssq/cnt - mean^2.
    for label, container in (("vecnorm", sd["vecnorm"]["_extra_state"]),
                             ("env transform",
                              sd["env"]["transforms.2._extra_state"]
                              if "transforms.2._extra_state" in sd["env"] else None)):
        if container is None:
            print(f"  ({label}: not present, skipped)")
            continue
        cnt = float(container["object_count"])
        new_sum = mean * cnt
        new_ssq = (std ** 2 + mean ** 2) * cnt
        container["object_sum"] = torch.cat([container["object_sum"], new_sum.to(container["object_sum"])])
        container["object_ssq"] = torch.cat([container["object_ssq"], new_ssq.to(container["object_ssq"])])
        print(f"  {label:13s} object_sum/ssq {obj} -> {container['object_sum'].numel()}   "
              f"(count {cnt:.3g})")

    Path(args.dst).parent.mkdir(parents=True, exist_ok=True)
    torch.save(sd, args.dst)
    print(f"\nwrote {args.dst}")


if __name__ == "__main__":
    main()
