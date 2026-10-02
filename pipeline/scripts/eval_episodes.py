#!/usr/bin/env python3
"""Evaluate a checkpoint on exactly one episode per env, all from frame 0.

Why not play.py's printed stats: play.py prints the mean of the first num_envs
episodes to FINISH. An env that drops the roller at step 28 finishes, restarts
and finishes again ten times before an env that paints the whole 342-frame
stroke finishes once, so stopping after a few blocks never counts the long,
successful episodes. On a policy that fails early most of the time that reads
as success 0.000 when the true rate is not zero.

The rollout is HDMI's `scripts.helpers.evaluate` loop (eval mode, seeded,
deterministic actions) with two changes: it keeps each env's FIRST episode
only and stops as soon as every env has finished one, and it records joint
torque and velocity every step for the efficiency numbers. Every env
contributes one complete episode however long it lasts, so num_envs is the
sample size.

Per episode it reports:

    coverage, success, episode_len, both-hands contact, lost-contact ending
    iou, precision   painted area vs the target, using paint outside the
                     target (needs track_wall_paint; on below 257 envs):
                     iou = cov / (1 + out), precision = cov / (cov + out)
    wall_dist        mean roller-head distance from the wall while painting;
                     ~0.042 m (the roller radius) means it touched the wall
    wall_force       mean tool-wall normal force while painting, in N (solid
                     wall tasks only)
    work_J           absolute mechanical work, sum over joints and steps of
                     |tau * qdot| * dt, also split into legs / waist / arms
    mean_power_W, torque_rms_Nm, peak_torque_ratio (|tau| / effort limit)
    J_per_m2         work per square metre of target painted

Torque and velocity are the last physics substep of each policy step. The
step on which an env's episode ends is left out of the torque numbers: by the
time step_and_maybe_reset returns, that env has already been reset.

Takes the same hydra overrides as play.py, plus `+save_traces=<path.npz>` for
the per-step joint torque and velocity traces. Prints one line
`RESULT {json}` for zsweep_eval.py to parse:

    python pipeline/scripts/eval_episodes.py task=G1/hdmi/wall_painting_goal \
        algo=ppo_roa_train checkpoint_path=<ckpt> task.num_envs=128 headless=true
"""

import json
import os
import re
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

HDMI_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HDMI_ROOT))   # for `scripts.helpers`

# The algo configs (ppo_roa_train, ...) live in a structured config store that
# is filled on import, not in cfg/algo/. Without this hydra cannot find them.
import active_adaptation  # noqa: F401

# Joint groups by name; the patterns cover both G1 (waist_*) and H1-2 (torso).
JOINT_GROUPS = {
    "legs": r".*_(hip|knee|ankle)_.*",
    "waist": r"(waist|torso)_.*",
    "arms": r".*_(shoulder|elbow|wrist)_.*",
}


@hydra.main(config_path="../../cfg", config_name="play", version_base=None)
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    from isaaclab.app import AppLauncher
    AppLauncher(OmegaConf.to_container(cfg.app))

    import numpy as np
    import torch
    from torchrl.envs.utils import set_exploration_type, ExplorationType
    from scripts.helpers import make_env_policy

    env, agent, _ = make_env_policy(cfg)
    policy = agent.get_rollout_policy("eval")
    base = env.base_env
    robot = base.scene["robot"]
    cmd = base.command_manager
    n = base.num_envs
    dt = float(base.step_dt)
    joint_names = list(robot.joint_names)
    effort_limit = robot.data.joint_effort_limits[0].abs().clamp_min(1e-6).cpu()

    base.eval()
    env.eval()
    env.set_seed(cfg.seed)
    td = env.reset()

    first_done = torch.full((n,), -1, dtype=torch.long)
    stats = {}            # stat key -> [n] episode total of the first episode
    tau_steps, qd_steps = [], []

    torch.compiler.cudagraph_mark_step_begin()
    with torch.inference_mode(), set_exploration_type(ExplorationType.MODE):
        for t in range(base.max_episode_length):
            td = policy(td)
            out, td = env.step_and_maybe_reset(td)
            tau_steps.append(robot.data.applied_torque.to("cpu", torch.float32))
            qd_steps.append(robot.data.joint_vel.to("cpu", torch.float32))

            done = out["next", "done"].reshape(n).cpu()
            newly = done & (first_done < 0)
            if bool(newly.any()):
                for k, v in out["next", "stats"].items(True, True):
                    key = "/".join(k) if isinstance(k, tuple) else k
                    if key not in stats:
                        stats[key] = torch.full((n,), float("nan"))
                    stats[key][newly] = v.reshape(n, -1)[:, 0].float().cpu()[newly]
                first_done[newly] = t
            if bool((first_done >= 0).all()):
                break

    unfinished = int((first_done < 0).sum())
    if unfinished:
        raise RuntimeError(f"{unfinished} envs never finished an episode")

    # Stats are episode totals of the weighted per-step values; every debug
    # term is weighted 1.0, so they read directly.
    def stat(key):
        if key not in stats:
            return torch.full((n,), float("nan"))
        return stats[key].double()

    length = stat("episode_len")
    success = stat("success")
    coverage = stat("debug/paint_coverage")
    outside = stat("debug/paint_outside")
    paint_steps = stat("debug/paint_steps")
    tracked = bool(getattr(cmd, "track_wall_paint", False))
    if not tracked:
        outside = torch.full((n,), float("nan"), dtype=torch.float64)

    # Torque and velocity of each env's first episode, terminal step dropped.
    tau = torch.stack(tau_steps, dim=1).double()            # [n, T, J]
    qd = torch.stack(qd_steps, dim=1).double()
    steps = torch.arange(tau.shape[1]).view(1, -1)
    alive = (steps < first_done.view(-1, 1)).double()       # [n, T]
    n_alive = alive.sum(1).clamp_min(1.0)

    power = (tau * qd).abs() * alive.unsqueeze(-1)          # [n, T, J], W
    work = power.sum(dim=(1, 2)) * dt
    group_work = {}
    for name, pattern in JOINT_GROUPS.items():
        idx = [i for i, j in enumerate(joint_names) if re.fullmatch(pattern, j)]
        if idx:
            group_work[name] = power[:, :, idx].sum(dim=(1, 2)) * dt
    torque_rms = ((tau.square() * alive.unsqueeze(-1)).sum(dim=(1, 2))
                  / (n_alive * len(joint_names))).sqrt()
    peak_ratio = ((tau.abs() / effort_limit.double()) * alive.unsqueeze(-1)).amax(dim=(1, 2))

    target_area = (cmd.target_width[:, 0] * cmd.target_height[:, 0]).double().cpu()
    painted_m2 = coverage * target_area

    def ratio(a, b):
        return torch.where(b > 0, a / b.clamp_min(1e-12), torch.full_like(a, float("nan")))

    per_episode = {
        "coverage": coverage,
        "success": success,
        "episode_len": length,
        "both_hands_per_step": ratio(stat("debug/eef_contact_both"), length),
        "lost_contact_term": stat("termination/cum_lost_contact_steps"),
        "outside": outside,
        "iou": coverage / (1.0 + outside),
        "precision": ratio(coverage, coverage + outside),
        "wall_dist": ratio(stat("debug/paint_wall_dist"), paint_steps),
        "wall_force": ratio(stat("debug/paint_wall_force"), paint_steps),
        "work_J": work,
        "mean_power_W": work / (n_alive * dt),
        "torque_rms_Nm": torque_rms,
        "peak_torque_ratio": peak_ratio,
        "J_per_m2": ratio(work, painted_m2),
    }
    for name, w in group_work.items():
        per_episode[f"work_{name}_J"] = w

    ok = success > 0.5
    result = {"episodes": n, "wall_tracked": tracked,
              "target_area_m2": float(target_area.mean())}
    for key, x in per_episode.items():
        finite = ~torch.isnan(x)
        both = finite & ok
        result[key] = x[finite].mean().item() if finite.any() else None
        result[key + "_std"] = x[finite].std().item() if finite.sum() > 1 else None
        result[key + "_given_success"] = x[both].mean().item() if both.any() else None
    # Work per painted area as a ratio of totals. A per-episode mean of the
    # ratio blows up on episodes that paint almost nothing (one at 0.002
    # coverage reads ~10^6 J/m^2), so it is not summarized that way.
    def per_area(mask):
        area = painted_m2[mask].sum()
        return (work[mask].sum() / area).item() if area > 0 else None
    result["J_per_m2"] = per_area(torch.ones_like(ok))
    result["J_per_m2_given_success"] = per_area(ok)
    result["J_per_m2_std"] = None
    for key in ("coverage", "success", "iou", "work_J"):
        result[key + "_per_episode"] = [
            None if np.isnan(v) else round(float(v), 4) for v in per_episode[key].tolist()
        ]

    traces = cfg.get("save_traces")
    if traces:
        last = int(first_done.max())
        np.savez_compressed(
            traces,
            joint_names=np.array(joint_names),
            effort_limit=effort_limit.numpy(),
            dt=dt,
            first_done=first_done.numpy(),
            alive=alive[:, :last].bool().numpy(),
            torque=tau[:, :last].float().numpy().astype(np.float16),
            joint_vel=qd[:, :last].float().numpy().astype(np.float16),
            **{k: v.numpy() for k, v in per_episode.items()},
        )
        result["traces"] = str(traces)

    print("RESULT " + json.dumps(result), flush=True)
    # IsaacSim hangs in teardown; eval.py exits the same way.
    os._exit(0)


if __name__ == "__main__":
    main()
