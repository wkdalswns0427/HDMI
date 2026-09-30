#!/usr/bin/env python
"""Bootstrap an object-annotation JSON from a converted robot motion.

The object trajectory is the manual part of HDMI's pipeline. This script does
not remove that work -- it makes it cheap by printing the diagnostics you need
to pick frames (per-wrist height and speed over time), guessing a plausible
carry window, and writing a template you then correct.

    # look at the motion and get a starting template
    python scripts/make_annotation.py <motion_dir> --object suitcase \
        --out annotations/move_suitcase.json

Always re-check the emitted frames against the video before training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

WRISTS = ["left_wrist_yaw_link", "right_wrist_yaw_link",
          "left_rubber_hand", "right_rubber_hand"]


def sparkline(x, width=60):
    """Tiny inline plot so frame picking does not need a plotting window."""
    blocks = "▁▂▃▄▅▆▇█"
    x = np.asarray(x, dtype=float)
    idx = np.linspace(0, len(x) - 1, min(width, len(x))).astype(int)
    v = x[idx]
    lo, hi = v.min(), v.max()
    if hi - lo < 1e-9:
        return blocks[0] * len(v)
    n = ((v - lo) / (hi - lo) * (len(blocks) - 1)).round().astype(int)
    return "".join(blocks[i] for i in n)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("motion_dir")
    p.add_argument("--object", required=True, help="object name, e.g. suitcase")
    p.add_argument("--asset", default=None, help="HDMI asset key (defaults to --object)")
    p.add_argument("--out", required=True)
    p.add_argument("--parent", default=None,
                   help="link that carries the object; default = the busier wrist")
    args = p.parse_args()

    path = Path(args.motion_dir)
    d = np.load(path / "motion.npz")
    meta = json.load(open(path / "meta.json"))
    names = meta["body_names"]
    bp = d["body_pos_w"]
    T, fps = bp.shape[0], meta["fps"]

    print(f"{T} frames @ {fps} fps ({T/fps:.2f}s)\n")

    avail = [w for w in WRISTS if w in names]
    if not avail:
        raise SystemExit(f"no wrist/hand link found among {names}")

    # Per-wrist height + speed traces: the grasp is usually where a hand drops
    # low, slows down, then travels while staying low-variance relative to it.
    stats = {}
    for w in avail:
        z = bp[:, names.index(w), 2]
        spd = np.linalg.norm(np.diff(bp[:, names.index(w)], axis=0), axis=-1) * fps
        spd = np.concatenate([spd, spd[-1:]])
        stats[w] = (z, spd)
        print(f"{w}")
        print(f"  z    {z.min():.2f}..{z.max():.2f} m  {sparkline(z)}")
        print(f"  spd  {spd.min():.2f}..{spd.max():.2f} m/s  {sparkline(spd)}")

    if args.parent:
        parent = args.parent
    else:
        # The hand that travels farthest is the one doing the manipulating.
        parent = max(avail, key=lambda w: np.linalg.norm(
            bp[:, names.index(w)] - bp[:, names.index(w)].mean(0), axis=-1).sum())
    print(f"\ncarrying link (guess): {parent}")

    z, spd = stats[parent]
    # Guess: carry spans the low-hand stretch between the first and last frame
    # where the hand is in the bottom third of its vertical range.
    lo = z.min() + 0.33 * (z.max() - z.min())
    low = np.where(z <= lo)[0]
    if len(low) > 2:
        pick, place = int(low[0]), int(low[-1])
    else:
        pick, place = T // 4, 3 * T // 4
    pick, place = max(1, pick), min(T - 2, place)
    print(f"carry window  (guess): frames {pick}..{place} "
          f"({pick/fps:.2f}s .. {place/fps:.2f}s)  <-- CHECK THIS AGAINST THE VIDEO")

    # Rest poses: put the object under the hand at pickup and at drop-off, on
    # the floor. These are placeholders keyed to the retargeted motion.
    hand = bp[:, names.index(parent)]
    start_pos = [round(float(hand[pick, 0]), 3), round(float(hand[pick, 1]), 3), 0.15]
    end_pos = [round(float(hand[place, 0]), 3), round(float(hand[place, 1]), 3), 0.15]

    ann = {
        "_comment": [
            "Object annotation for HDMI. Frame indices are on the RESAMPLED "
            f"timeline ({T} frames @ {fps} fps).",
            "Frames below are heuristic guesses -- verify against the source video.",
            "modes: static (fixed pose) | lerp (move between poses) | attach (ride a link)",
        ],
        "object": {"name": args.object, "asset": args.asset or args.object},
        "extra_objects": [],
        "segments": [
            {"start": 0, "end": pick - 1, "mode": "static",
             "pos": start_pos, "quat": [1, 0, 0, 0]},
            {"start": pick, "end": place, "mode": "attach", "parent": parent},
            {"start": place + 1, "end": T - 1, "mode": "static",
             "pos": end_pos, "quat": [1, 0, 0, 0]},
        ],
        "contact": [[pick, place]],
        "smoothing": {"window": 9},
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(ann, f, indent=2)
    print(f"\nwrote {out}\n  edit the frames/poses, then re-run scripts/gmr_to_hdmi.py "
          f"with --annotation {out}")


if __name__ == "__main__":
    main()
