#!/usr/bin/env python
"""Sanity-check an HDMI motion folder before spending GPU hours on it.

Catches the failure modes that otherwise show up as a policy that will not
train: unnormalised quaternions, a robot floating above or sunk below the
floor, velocity spikes from a retargeting glitch, contact flags that never
fire, and shape/meta disagreements.

    python scripts/validate_motion.py <motion_dir> [--reference <other_dir>]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REQUIRED = ["body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w",
            "joint_pos", "joint_vel"]

# Loose physical envelopes; a humanoid demo outside these is almost always a bug.
LIMITS = {"lin_vel": 12.0, "ang_vel": 40.0, "joint_vel": 30.0}


def summarize(path: Path) -> dict:
    d = np.load(path / "motion.npz")
    meta = json.load(open(path / "meta.json"))
    return {"d": d, "meta": meta}


def check(path: Path) -> tuple[list[str], list[str]]:
    errors, warns = [], []
    if not (path / "motion.npz").exists():
        return [f"missing {path/'motion.npz'}"], []
    if not (path / "meta.json").exists():
        return [f"missing {path/'meta.json'}"], []

    d = np.load(path / "motion.npz")
    meta = json.load(open(path / "meta.json"))

    for k in REQUIRED:
        if k not in d.files:
            errors.append(f"missing array {k!r}")
    if errors:
        return errors, warns

    T = d["body_pos_w"].shape[0]
    B = d["body_pos_w"].shape[1]
    J = d["joint_pos"].shape[1]

    # --- shapes vs meta ---
    if B != len(meta.get("body_names", [])):
        errors.append(f"{B} bodies in npz vs {len(meta.get('body_names',[]))} in meta.json")
    if J != len(meta.get("joint_names", [])):
        errors.append(f"{J} joints in npz vs {len(meta.get('joint_names',[]))} in meta.json")
    for k in REQUIRED:
        if d[k].shape[0] != T:
            errors.append(f"{k}: {d[k].shape[0]} frames, expected {T}")
    if "fps" not in meta:
        errors.append("meta.json has no fps")

    # --- finite ---
    for k in REQUIRED:
        if not np.isfinite(d[k]).all():
            errors.append(f"{k} contains NaN/Inf")

    # --- quaternions ---
    n = np.linalg.norm(d["body_quat_w"], axis=-1)
    if np.abs(n - 1).max() > 1e-5:
        errors.append(f"quaternions not unit-norm (max err {np.abs(n-1).max():.2e})")
    # A sign flip between frames breaks finite-difference angular velocity.
    dots = np.sum(d["body_quat_w"][1:] * d["body_quat_w"][:-1], axis=-1)
    if (dots < 0).any():
        warns.append(f"{int((dots<0).sum())} quaternion sign flips "
                     f"(angular velocities across them are wrong)")

    # --- ground contact ---
    minz = d["body_pos_w"][..., 2].min(axis=1)
    if minz.mean() > 0.15:
        warns.append(f"lowest body averages {minz.mean():.3f} m — robot may float")
    if minz.min() < -0.05:
        warns.append(f"lowest body reaches {minz.min():.3f} m — robot sinks through floor")

    # --- velocity envelopes ---
    for key, arr, lim in [
        ("lin_vel", np.linalg.norm(d["body_lin_vel_w"], axis=-1), LIMITS["lin_vel"]),
        ("ang_vel", np.linalg.norm(d["body_ang_vel_w"], axis=-1), LIMITS["ang_vel"]),
        ("joint_vel", np.abs(d["joint_vel"]), LIMITS["joint_vel"]),
    ]:
        if arr.max() > lim:
            warns.append(f"{key} peaks at {arr.max():.1f} (> {lim}) — likely a retarget glitch")

    # --- contact flag ---
    # Robot links are named "*_link" (plus "pelvis"); anything else in
    # body_names is an appended object body.
    obj_bodies = [n for n in meta.get("body_names", [])
                  if not n.endswith("_link") and n != "pelvis"]
    if "object_contact" in d.files:
        c = d["object_contact"]
        if c.shape != (T, 1):
            errors.append(f"object_contact shape {c.shape}, expected {(T,1)}")
        elif obj_bodies and not c.any():
            warns.append(f"object bodies {obj_bodies} present but object_contact "
                         f"is never True — the policy gets no contact signal")
    elif obj_bodies:
        warns.append(f"object bodies {obj_bodies} present but no object_contact array")

    return errors, warns


def travel(path: Path, body="pelvis"):
    """Horizontal path length / net displacement / speed of one body."""
    d = np.load(path / "motion.npz")
    meta = json.load(open(path / "meta.json"))
    if body not in meta["body_names"]:
        return None
    xyz = d["body_pos_w"][:, meta["body_names"].index(body)]
    fps = float(meta.get("fps", 50)) or 50.0
    dur = len(xyz) / fps
    step = np.linalg.norm(np.diff(xyz[:, :2], axis=0), axis=1)
    return {
        "xyz": xyz, "fps": fps, "duration": dur,
        "path_len": float(step.sum()),
        "net_disp": float(np.linalg.norm(xyz[-1, :2] - xyz[0, :2])),
        "mean_speed": float(step.sum() / dur) if dur else 0.0,
        "peak_speed": float((step * fps).max()) if len(step) else 0.0,
        "range_x": float(np.ptp(xyz[:, 0])), "range_y": float(np.ptp(xyz[:, 1])),
        "range_z": float(np.ptp(xyz[:, 2])),
    }


def plot_trajectory(out_png, runs):
    """Top-down pelvis path. `runs` is [(label, travel_dict), ...]."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Slots 1-2 of the validated categorical palette (blue, orange).
    HUES = ["#2a78d6", "#eb6834"]
    INK, MUTED, GRID = "#1a1a18", "#6b6b66", "#e2e1dd"

    fig, ax = plt.subplots(figsize=(6.4, 6.0), dpi=150)
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")

    for i, (label, t) in enumerate(runs):
        c = HUES[i % len(HUES)]
        x, y = t["xyz"][:, 0], t["xyz"][:, 1]
        ax.plot(x, y, color=c, lw=2.0, solid_capstyle="round",
                label=f"{label}  ({t['path_len']:.2f} m)", zorder=3 - i)
        # start hollow, end filled -- direction without an arrow overlay
        ax.plot(x[0], y[0], "o", ms=9, mfc="white", mec=c, mew=2.0, zorder=4)
        ax.plot(x[-1], y[-1], "o", ms=9, color=c, zorder=4)
        ax.annotate("start", (x[0], y[0]), textcoords="offset points",
                    xytext=(8, 6), color=MUTED, fontsize=9)
        ax.annotate("end", (x[-1], y[-1]), textcoords="offset points",
                    xytext=(8, 6), color=MUTED, fontsize=9)

    ax.set_aspect("equal", adjustable="datalim")   # metres must not be distorted
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_xlabel("x (m)", color=MUTED, fontsize=10)
    ax.set_ylabel("y (m)", color=MUTED, fontsize=10)
    ax.set_title("Pelvis path, top-down", color=INK, fontsize=12,
                 loc="left", pad=12)
    if len(runs) >= 2:
        leg = ax.legend(frameon=False, fontsize=9, loc="best")
        for t_ in leg.get_texts():
            t_.set_color(INK)
    fig.tight_layout()
    fig.savefig(out_png, facecolor="white")
    plt.close(fig)


def report(path: Path):
    d = np.load(path / "motion.npz")
    meta = json.load(open(path / "meta.json"))
    T = d["body_pos_w"].shape[0]
    fps = meta.get("fps", 0) or 1
    print(f"  frames  : {T} @ {fps} fps ({T/fps:.2f}s)")
    print(f"  bodies  : {d['body_pos_w'].shape[1]}  joints: {d['joint_pos'].shape[1]}")
    print(f"  min-z   : mean {d['body_pos_w'][...,2].min(axis=1).mean():.4f} m")
    print(f"  lin_vel : p99 {np.percentile(np.linalg.norm(d['body_lin_vel_w'],axis=-1),99):.2f} "
          f"max {np.linalg.norm(d['body_lin_vel_w'],axis=-1).max():.2f} m/s")
    print(f"  ang_vel : p99 {np.percentile(np.linalg.norm(d['body_ang_vel_w'],axis=-1),99):.2f} "
          f"max {np.linalg.norm(d['body_ang_vel_w'],axis=-1).max():.2f} rad/s")
    if "object_contact" in d.files:
        c = d["object_contact"]
        print(f"  contact : {int(c.sum())}/{T} frames ({100*c.mean():.1f}%)")
    t = travel(path)
    if t:
        print(f"  travel  : path {t['path_len']:.3f} m | net {t['net_disp']:.3f} m | "
              f"mean {t['mean_speed']:.3f} m/s | peak {t['peak_speed']:.2f} m/s")
        print(f"  range   : x {t['range_x']:.3f}  y {t['range_y']:.3f}  z {t['range_z']:.3f} m")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("motion_dir")
    p.add_argument("--reference", default=None,
                   help="a known-good motion folder to print alongside")
    p.add_argument("--plot", default=None, metavar="OUT.png",
                   help="write a top-down pelvis path plot; overlays --reference "
                        "when given, so executed vs reference is directly comparable")
    args = p.parse_args()

    path = Path(args.motion_dir)
    print(f"=== {path} ===")
    errors, warns = check(path)
    # Still print the stats when the core arrays are sound -- a malformed
    # object_contact (HDMI's recorder writes the reference slice, not the
    # executed length) shouldn't hide the trajectory numbers.
    try:
        report(path)
    except Exception as e:
        print(f"  (stats unavailable: {type(e).__name__}: {e})")

    if args.reference:
        ref = Path(args.reference)
        print(f"\n=== reference: {ref} ===")
        rerr, _ = check(ref)
        if not rerr:
            report(ref)

    if args.plot:
        runs = []
        t = travel(path)
        if t: runs.append((path.name, t))
        if args.reference:
            tr = travel(Path(args.reference))
            if tr: runs.append((Path(args.reference).name, tr))
        if runs:
            plot_trajectory(args.plot, runs)
            print(f"\n  wrote trajectory plot -> {args.plot}")

    print()
    for w in warns:
        print(f"  WARN  {w}")
    for e in errors:
        print(f"  ERROR {e}")
    if not errors and not warns:
        print("  OK — no issues found")
    elif not errors:
        print(f"\n  passed with {len(warns)} warning(s)")
    else:
        print(f"\n  FAILED with {len(errors)} error(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
