#!/usr/bin/env python3
"""Measure how far the wall-painting target can move in z before a policy stops
covering it.

For each z offset, runs HDMI's play.py with the target region PINNED at that
offset and records, per block of finished episodes:

    coverage     stats/debug/paint_coverage -- unweighted mirror of
                 paint_coverage_delta, so its episode sum IS the final coverage
    episode_len  how long episodes last before terminating
    success      fraction of episodes that reach 90% of max length

Coverage alone cannot tell "paints the wrong place" from "drops the roller
early", so all three are reported.

    python pipeline/scripts/zsweep_eval.py --checkpoint <ckpt>

play.py never exits on its own -- its rollout is `for i in itertools.count()`
-- so each offset is run as a stream: read play.py's output, stop after
--blocks stats blocks (each block is num_envs finished episodes), then kill the
process group. IsaacSim also hangs in teardown, which the group kill covers.

The reference clip paints z = 0.796 .. 1.649 m with the pelvis fixed at
0.809 m. Offsets are applied to the target centre along world z, which for this
wall is the in-plane vertical, so the target stays on the wall.
"""

import argparse
import json
import os
import re
import signal
import statistics
import subprocess
import threading
import time
from pathlib import Path

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from hdmi_pipeline import paths

HDMI = paths.HDMI_ROOT
PY = paths.conda_python("hdmi")

# play.py prints each stat as `(<key tuple>) <value>`; one line per key.
PATTERNS = {
    "coverage": re.compile(r"\('stats', 'debug', 'paint_coverage'\)\s+([\d.eE+-]+)"),
    "episode_len": re.compile(r"\('stats', 'episode_len'\)\s+([\d.eE+-]+)"),
    "success": re.compile(r"\('stats', 'success'\)\s+([\d.eE+-]+)"),
}


def run_one(checkpoint: str, z: float, task: str, num_envs: int, blocks: int,
            timeout_s: float, out_dir: Path) -> dict:
    """Stream one play.py run with the target pinned at `z`.

    Returns {metric: [value per stats block]}; empty lists if nothing parsed.
    """
    log = out_dir / f"z{z:+.3f}.log"

    cmd = [
        PY, "scripts/play.py",
        f"task=G1/hdmi/{task}",
        # ppo_roa does NOT exist -- the algo configs come from a structured
        # config store, not cfg/algo/*.yaml, and the trained checkpoints all
        # use ppo_roa_train. Getting it wrong fails instantly in hydra.
        "algo=ppo_roa_train",
        f"checkpoint_path={checkpoint}",
        "headless=true",
        f"task.num_envs={num_envs}",
        # Pin the target: everything fixed except the swept z. Tasks like
        # wall_painting_goal also randomize the in-plane position and the width,
        # so those must be pinned too or the "pinned" target still moves.
        # `++` adds the key when the task config does not declare it.
        f"task.command.target_region_pos_range.z=[{z},{z}]",
        "++task.command.target_region_uv_range.u=[0.0,0.0]",
        "++task.command.target_region_uv_range.v=[0.0,0.0]",
        "task.command.target_region_scale_range.width=[1.0,1.0]",
        "task.command.target_region_scale_range.height=[1.0,1.0]",
    ]

    # Without this, play.py's stdout is block-buffered into the pipe and stats
    # arrive in large delayed chunks, or not at all before the kill.
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    env.pop("PYTHONNOUSERSITE", None)   # the hdmi env needs ~/.local

    results = {k: [] for k in PATTERNS}

    with open(log, "w") as fh:
        proc = subprocess.Popen(
            cmd, cwd=HDMI, env=env, text=True, bufsize=1,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            start_new_session=True,       # its own process group, for the kill
        )

        def kill_group():
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

        # A hung play.py produces no lines, so a timeout checked inside the
        # read loop would never fire. A timer kills it from outside instead.
        watchdog = threading.Timer(timeout_s, kill_group)
        watchdog.start()
        try:
            for line in proc.stdout:
                fh.write(line)
                for key, pat in PATTERNS.items():
                    m = pat.search(line)
                    if m:
                        results[key].append(float(m.group(1)))
                # A block is complete once its coverage line has been seen.
                if len(results["coverage"]) >= blocks:
                    break
        finally:
            watchdog.cancel()
            kill_group()
            proc.wait()

    if not results["coverage"]:
        # Say why, loudly: a silent empty result for every offset looks the
        # same as "the policy scored nothing", and that cost an hour once.
        text = log.read_text(errors="ignore")
        reason = next(
            (l.strip() for l in text.splitlines()
             if "Could not find" in l or "Traceback" in l
             or "Error executing" in l or "RuntimeError" in l),
            f"no stats block within {timeout_s:.0f}s",
        )
        print(f"    !! z={z:+.2f}: nothing parsed -- {reason[:110]}")
        print(f"    !! full log: {log}")

    return results


def summarize(values: list) -> tuple:
    if not values:
        return None, None
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std


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
    ap.add_argument("--num_envs", type=int, default=32)
    ap.add_argument("--blocks", type=int, default=3,
                    help="stats blocks to average per offset; each block is "
                         "num_envs finished episodes")
    ap.add_argument("--timeout", type=float, default=600.0,
                    help="seconds per offset before giving up on it")
    ap.add_argument("--out", default=None,
                    help="directory for per-offset logs and results.json")
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else \
        Path(args.checkpoint).parent / "zsweep"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"checkpoint : {args.checkpoint}")
    print(f"task       : {args.task}")
    print(f"envs       : {args.num_envs} x {args.blocks} blocks "
          f"= {args.num_envs * args.blocks} episodes per offset")
    print(f"logs       : {out_dir}\n")
    print(f"{'z [m]':>7}  {'coverage':>15}  {'vs z=0':>7}  "
          f"{'episode_len':>11}  {'success':>7}  {'time':>5}")
    print("-" * 64)

    results = {}
    for z in args.offsets:
        t0 = time.time()
        r = run_one(args.checkpoint, z, args.task, args.num_envs,
                    args.blocks, args.timeout, out_dir)
        cov, cov_sd = summarize(r["coverage"])
        el, _ = summarize(r["episode_len"])
        su, _ = summarize(r["success"])
        results[f"{z:+.3f}"] = {
            "coverage": cov, "coverage_std": cov_sd,
            "episode_len": el, "success": su,
            "blocks": r["coverage"],
        }

        base = results.get("+0.000", {}).get("coverage")
        rel = f"{cov / base * 100:6.1f}%" if (cov is not None and base) else "      -"
        cov_s = f"{cov:.4f} +-{cov_sd:.4f}" if cov is not None else "           none"
        el_s = f"{el:11.1f}" if el is not None else "          -"
        su_s = f"{su:7.3f}" if su is not None else "      -"
        print(f"{z:+7.2f}  {cov_s:>15}  {rel:>7}  {el_s}  {su_s}  "
              f"{time.time() - t0:4.0f}s", flush=True)

        # Write as we go, so a crash partway still leaves the finished rows.
        (out_dir / "results.json").write_text(json.dumps(results, indent=2))

    print(f"\nwrote {out_dir / 'results.json'}")

    ok = {k: v for k, v in results.items() if v["coverage"] is not None}
    if "+0.000" in ok:
        base = ok["+0.000"]["coverage"]
        held = [float(k) for k, v in ok.items() if v["coverage"] >= 0.8 * base]
        if held:
            print(f"\nholds >=80% of the z=0 coverage over "
                  f"z = {min(held):+.2f} .. {max(held):+.2f} m")
            print("That band is what one standing clip covers without retraining. "
                  "Compare the goal-conditioned policy against it.")


if __name__ == "__main__":
    main()
