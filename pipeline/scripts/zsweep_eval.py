#!/usr/bin/env python3
"""Measure how far the wall-painting target can move in z before a policy stops
covering it.

For each z offset, evaluates the policy with the target region PINNED at that
offset, one complete episode per env, every episode from frame 0 of the clip
(pipeline/scripts/eval_episodes.py). Per offset it reports:

    coverage     final coverage of the target, mean over episodes
    success      fraction of episodes that reach the end of the clip
                 (command.success: t >= motion_len - 1)
    cov|success  coverage over the successful episodes only -- painting
                 quality with the drop rate taken out
    episode_len  how long episodes last before terminating
    lost         fraction of episodes ended by the lost-contact termination

Coverage alone cannot tell "paints the wrong place" from "drops the roller
early", so both are reported, and cov|success separates them: a policy that
follows the target keeps cov|success up as the target moves, one that paints
the demonstrated spot loses it by the overlap.

    python pipeline/scripts/zsweep_eval.py --checkpoint <ckpt> --task <task>

An earlier version streamed play.py and averaged its first few printed stats
blocks. That is biased: play.py prints the mean of the first num_envs episodes
to FINISH, short failed episodes finish (and restart) many times before a long
successful one finishes once, so stopping early drops the successes. For the
goal-conditioned policy it read success 0.000 at z=0 where the true rate was
0.29. One episode per env has no such bias.

The reference clip paints z = 0.796 .. 1.649 m with the pelvis fixed at
0.809 m. Offsets are applied to the target centre along world z, which for this
wall is the in-plane vertical, so the target stays on the wall.
"""

import argparse
import json
import math
import os
import signal
import subprocess
import time
from pathlib import Path

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from hdmi_pipeline import paths

HDMI = paths.HDMI_ROOT
PY = paths.conda_python("hdmi")
EVAL = paths.PIPELINE / "scripts" / "eval_episodes.py"


def run_one(checkpoint: str, z: float, task: str, num_envs: int,
            timeout_s: float, out_dir: Path) -> dict | None:
    """Evaluate num_envs episodes with the target pinned at `z`.

    Returns eval_episodes.py's RESULT dict, or None if it produced none.
    """
    log = out_dir / f"z{z:+.3f}.log"

    cmd = [
        PY, str(EVAL),
        f"task=G1/hdmi/{task}",
        # ppo_roa does NOT exist -- the algo configs come from a structured
        # config store, not cfg/algo/*.yaml, and the trained checkpoints all
        # use ppo_roa_train. Getting it wrong fails instantly in hydra.
        "algo=ppo_roa_train",
        f"checkpoint_path={checkpoint}",
        "headless=true",
        f"task.num_envs={num_envs}",
        # Pin the target: everything fixed except the swept z. Tasks like
        # wall_painting_goal also randomize the in-plane position and the size,
        # so those must be pinned too or the "pinned" target still moves.
        # `++` adds the key when the task config does not declare it.
        f"task.command.target_region_pos_range.z=[{z},{z}]",
        "++task.command.target_region_uv_range.u=[0.0,0.0]",
        "++task.command.target_region_uv_range.v=[0.0,0.0]",
        "task.command.target_region_scale_range.width=[1.0,1.0]",
        "task.command.target_region_scale_range.height=[1.0,1.0]",
        # keep hydra's per-run directories out of outputs_play/
        # quoted: hydra's override grammar is not happy with a bare `+` in z+0.1
        f"hydra.run.dir='{out_dir / 'hydra' / f'z{z:+.3f}'}'",
    ]

    env = dict(os.environ)
    env.pop("PYTHONNOUSERSITE", None)   # the hdmi env needs ~/.local

    with open(log, "w") as fh:
        proc = subprocess.Popen(
            cmd, cwd=HDMI, env=env, stdout=fh, stderr=subprocess.STDOUT,
            start_new_session=True,       # its own process group, for the kill
        )
        try:
            proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            pass
        finally:
            # eval_episodes.py exits via os._exit, but if IsaacSim hangs
            # anywhere the group kill still clears it.
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()

    text = log.read_text(errors="ignore")
    for line in text.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[len("RESULT "):])

    # Say why, loudly: a silent empty row looks the same as "scored nothing".
    reason = next(
        (l.strip() for l in text.splitlines()
         if "Could not find" in l or "Traceback" in l
         or "Error executing" in l or "RuntimeError" in l),
        f"no RESULT line within {timeout_s:.0f}s",
    )
    print(f"    !! z={z:+.2f}: nothing parsed -- {reason[:110]}")
    print(f"    !! full log: {log}")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True,
                    help="policy to evaluate (absolute path)")
    ap.add_argument("--task", default="wall_painting2_both",
                    help="task config under cfg/task/G1/hdmi. Must have the "
                         "SAME observation space the checkpoint was trained "
                         "with -- the z offset is applied by CLI override, so "
                         "sweep the training task. wall_painting_goal adds "
                         "target_center_b (obs 30 -> 33); a checkpoint trained "
                         "without it fails to load vecnorm there.")
    ap.add_argument("--offsets", type=float, nargs="+",
                    default=[-0.5, -0.4, -0.3, -0.2, -0.1, 0.0,
                             0.1, 0.2, 0.3, 0.4, 0.5],
                    help="target-region z offsets in metres. -0.5 puts the "
                         "target bottom at 0.296 m (squat territory), +0.5 "
                         "puts the top at 2.149 m (out of standing reach). "
                         "Past -0.6 the target passes through the floor.")
    ap.add_argument("--num_envs", type=int, default=128,
                    help="episodes per offset (one per env)")
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
    print(f"episodes   : {args.num_envs} per offset, one per env, from frame 0")
    print(f"logs       : {out_dir}\n")
    print(f"{'z [m]':>6}  {'coverage (s.e.)':>15}  {'vs z=0':>6}  {'success':>7}  "
          f"{'cov|success':>11}  {'ep_len':>6}  {'lost':>5}  {'time':>5}")
    print("-" * 78)

    results = {}
    for z in args.offsets:
        t0 = time.time()
        r = run_one(args.checkpoint, z, args.task, args.num_envs,
                    args.timeout, out_dir)
        results[f"{z:+.3f}"] = r
        if r is not None:
            base = (results.get("+0.000") or {}).get("coverage")
            rel = f"{r['coverage'] / base * 100:5.1f}%" if base else "     -"
            sem = r["coverage_std"] / math.sqrt(r["episodes"])
            cgs = r["coverage_given_success"]
            cgs_s = f"{cgs:11.3f}" if cgs is not None else "          -"
            cov_s = f"{r['coverage']:.3f} ({sem:.3f})"
            print(f"{z:+6.2f}  {cov_s:>15}  {rel:>6}  "
                  f"{r['success']:7.3f}  {cgs_s}  {r['episode_len']:6.1f}  "
                  f"{r['lost_contact_term']:5.2f}  {time.time() - t0:4.0f}s",
                  flush=True)
        # Write as we go, so a crash partway still leaves the finished rows.
        (out_dir / "results.json").write_text(json.dumps(results, indent=2))

    print(f"\nwrote {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
