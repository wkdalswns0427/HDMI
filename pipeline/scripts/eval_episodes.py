#!/usr/bin/env python3
"""Evaluate a checkpoint on exactly one episode per env, all from frame 0.

Why not play.py's printed stats: play.py prints the mean of the first num_envs
episodes to FINISH. An env that drops the roller at step 28 finishes, restarts
and finishes again ten times before an env that paints the whole 342-frame
stroke finishes once, so stopping after a few blocks never counts the long,
successful episodes. On a policy that fails early most of the time that reads
as success 0.000 when the true rate is not zero.

This runs HDMI's own `scripts.helpers.evaluate`, which steps every env for
max_episode_length and keeps only each env's FIRST episode. Every env
contributes one complete episode however long it lasts, so num_envs is the
sample size.

Takes the same hydra overrides as play.py and prints one line
`RESULT {json}` (per-episode means and stds) for zsweep_eval.py to parse:

    python pipeline/scripts/eval_episodes.py task=G1/hdmi/wall_painting_goal \
        algo=ppo_roa_train checkpoint_path=<ckpt> task.num_envs=128 headless=true
"""

import json
import os
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

HDMI_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HDMI_ROOT))   # for `scripts.helpers`

# The algo configs (ppo_roa_train, ...) live in a structured config store that
# is filled on import, not in cfg/algo/. Without this hydra cannot find them.
import active_adaptation  # noqa: F401


@hydra.main(config_path="../../cfg", config_name="play", version_base=None)
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    from isaaclab.app import AppLauncher
    AppLauncher(OmegaConf.to_container(cfg.app))

    import torch
    from scripts.helpers import make_env_policy, evaluate

    env, agent, _ = make_env_policy(cfg)
    policy = agent.get_rollout_policy("eval")
    _, trajs, stats, _ = evaluate(env, policy, seed=cfg.seed)

    done = trajs["next", "done"].squeeze(-1)
    unfinished = int((~done.any(dim=1)).sum())
    if unfinished:
        # evaluate() takes argmax of `done`, which is 0 for an env that never
        # finished, so its "episode" would be step 0. Refuse rather than mix it in.
        raise RuntimeError(f"{unfinished} envs never finished an episode")

    # evaluate() divides every stat except episode_len and success by the
    # episode length. Multiply back to get per-episode totals: the coverage
    # total IS the final coverage, a termination flag total is 0 or 1.
    length = stats["eval/episode_len"].float()
    def total(key):
        return stats[key].float() * length

    coverage = total("eval/debug/paint_coverage")
    success = stats["eval/success"].float()
    per_episode = {
        "coverage": coverage,
        "episode_len": length,
        "success": success,
        "both_hands_per_step": stats["eval/debug/eef_contact_both"].float(),
        "lost_contact_term": total("eval/termination/cum_lost_contact_steps"),
    }
    out = {"episodes": int(length.numel())}
    for k, v in per_episode.items():
        out[k] = v.mean().item()
        out[k + "_std"] = v.std().item()
    # Painting quality with the drop rate taken out: coverage over the
    # episodes that held the roller to the end of the clip.
    ok = success > 0.5
    out["coverage_given_success"] = coverage[ok].mean().item() if ok.any() else None
    out["coverage_per_episode"] = [round(x, 4) for x in coverage.tolist()]
    out["success_per_episode"] = [int(x) for x in success.tolist()]

    print("RESULT " + json.dumps(out), flush=True)
    # IsaacSim hangs in teardown; eval.py exits the same way.
    os._exit(0)


if __name__ == "__main__":
    main()
