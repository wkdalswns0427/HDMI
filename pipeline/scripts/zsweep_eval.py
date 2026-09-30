#!/usr/bin/env python3
"""Phase 2: measure how far the wall-painting target can move in z before the
policy stops covering it.

Runs HDMI's play.py once per fixed z offset with the target region PINNED at
that offset (both ends of target_region_pos_range.z set equal), and reads the
final coverage off `stats/debug/paint_coverage` -- an unweighted mirror of
paint_coverage_delta whose episode-accumulated stat is the coverage ratio.

    python pipeline/scripts/zsweep_eval.py \
        --checkpoint HDMI/outputs/.../checkpoint_1000.pt \
        --offsets -0.20 -0.15 -0.10 -0.05 0.0 0.05 0.10 0.15 0.20

The reference clip sweeps z = 0.796 .. 1.649 m with the pelvis fixed at
0.809 m, so the interesting question is how far below 0.796 the policy holds
before it would need to bend or squat.
"""

import argparse
import json
import re
import subprocess
from pathlib import Path

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from hdmi_pipeline import paths

HDMI = paths.HDMI_ROOT
PY = paths.conda_python("hdmi")

# The stat play.py prints for the unweighted mirror term.
STAT = re.compile(r"\('stats', 'debug', 'paint_coverage'\)\s+([\d.eE+-]+)")


def run_one(checkpoint: str, z: float, task: str, num_envs: int,
            out_dir: Path) -> float | None:
    """One play.py run with the target pinned at `z`. Returns final coverage."""
    log = out_dir / f"z{z:+.3f}.log"

    cmd = [
        PY, "scripts/play.py",
        f"task=G1/hdmi/{task}",
        # ppo_roa does NOT exist -- the algo configs come from a structured
        # config store, not cfg/algo/*.yaml, and the trained checkpoints all
        # use ppo_roa_train. Getting this wrong fails instantly in hydra and
        # every offset silently parses as None.
        "algo=ppo_roa_train",
        f"checkpoint_path={checkpoint}",
        "headless=true",
        f"task.num_envs={num_envs}",
        # Pin the target instead of randomizing it.
        f"task.command.target_region_pos_range.z=[{z},{z}]",
        "task.command.target_region_scale_range.height=[1.0,1.0]",
    ]

    with open(log, "w") as fh:
        rc = subprocess.run(cmd, cwd=HDMI, stdout=fh, stderr=subprocess.STDOUT,
                            check=False).returncode

    text = log.read_text(errors="ignore")

    # Last reported episode value wins -- earlier ones are partial episodes.
    values = STAT.findall(text)
    if values:
        return float(values[-1])

    # No value parsed. Say why, loudly -- a silent None for every offset is
    # indistinguishable from "the policy scored nothing", and that cost an
    # hour once already.
    print(f"    !! no coverage parsed from {log.name} (exit {rc})")
    for line in text.splitlines():
        if ("Could not find" in line or "Error" in line
                or "Traceback" in line or "error" in line.lower()[:40]):
            print(f"    !! {line.strip()[:120]}")
            break
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True,
                    help="policy to evaluate (absolute path)")
    ap.add_argument("--task", default="wall_painting2_both",
                    help="task config under cfg/task/G1/hdmi. Must have the "
                         "SAME observation space the checkpoint was trained "
                         "with -- the z offset is applied by CLI override, so "
                         "sweep the training task, not wall_painting2_zsweep. "
                         "That config adds target_center_b (obs 30 -> 33) and "
                         "a policy trained without it fails to load vecnorm.")
    ap.add_argument("--offsets", type=float, nargs="+",
                    default=[-0.5, -0.4, -0.3, -0.2, -0.1, 0.0,
                             0.1, 0.2, 0.3, 0.4, 0.5],
                    help="target-region z offsets in metres. -0.5 puts the "
                         "target bottom at 0.296 m (squat territory), +0.5 "
                         "puts the top at 2.149 m (out of standing reach). "
                         "Past -0.6 the target passes through the floor.")
    ap.add_argument("--num_envs", type=int, default=64)
    ap.add_argument("--out", default=None,
                    help="directory for per-offset logs and results.json")
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else \
        Path(args.checkpoint).parent / "zsweep"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"checkpoint : {args.checkpoint}")
    print(f"task       : {args.task}")
    print(f"envs       : {args.num_envs}")
    print(f"logs       : {out_dir}\n")

    results = {}
    print(f"{'z offset [m]':>13}  {'coverage':>9}  {'vs z=0':>8}")
    print("-" * 34)

    for z in args.offsets:
        cov = run_one(args.checkpoint, z, args.task, args.num_envs, out_dir)
        results[f"{z:+.3f}"] = cov
        base = results.get("+0.000")
        rel = f"{cov / base * 100:7.1f}%" if (cov and base) else "      -"
        print(f"{z:>+13.3f}  {cov if cov is None else f'{cov:9.4f}'}  {rel}")

    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_dir / 'results.json'}")

    ok = {k: v for k, v in results.items() if v is not None}
    if ok and "+0.000" in ok:
        base = ok["+0.000"]
        held = [k for k, v in ok.items() if v >= 0.8 * base]
        if held:
            lo = min(float(k) for k in held)
            hi = max(float(k) for k in held)
            print(f"\nholds >=80% of the pinned-target coverage over "
                  f"z = {lo:+.2f} .. {hi:+.2f} m")
            print("Compare this band against what a bend/squat clip would add "
                  "(see pipeline/DOC_multi_demo_same_task.md) before filming.")


if __name__ == "__main__":
    main()
