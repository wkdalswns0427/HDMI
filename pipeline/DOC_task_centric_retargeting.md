**Task-centric retargeting: from a worker video to a goal-conditioned policy**

The framing document for this pipeline. `DOC_wall_painting_stages.md` is the stage-by-stage build log with commands and run history; this is what the thing is and what it claims.

**The thesis**

Motion retargeting asks: make the robot move like the human did. Task-centric retargeting asks: make the robot *do what the human was doing*, which is not the same question and does not have the same answer.

A worker painting a wall is not executing a trajectory. They are covering a region, and the trajectory is one of many ways they could have done it. Retarget the trajectory and you get a robot that reproduces one stroke. Retarget the *task* and you get a robot that covers the region — including regions the worker never painted.

The gap between those two is the whole point of this pipeline. Everything below either extracts the task from the video, or builds the machinery to reward it.

**What baseline HDMI gives you, and where it stops**

HDMI tracks a single reference motion and rewards imitation. Ten tracking terms compare the robot to the reference frame by frame, indexed by a phase variable. Two object terms keep the manipulated object on its demonstrated path. A contact term keeps the hands on it. A teacher/student split (regularized online adaptation) handles the sim-to-real gap.

Every one of those rewards is defined against the demonstration. Nothing in HDMI knows what the task *was*. The suitcase policy moves a suitcase along one path; the door policy opens one door through one arc. They are excellent trajectory trackers and they are specialists by construction — the upstream FAQ puts useful generalization at roughly 10-20 cm of object displacement, because contact targets live in the object frame and the object observation is real goal conditioning as far as it goes. Push past that and the ten tracking terms outvote the task.

HDMI also assumes you already have retargeted motion. It ships with data derived from mocap and from OMOMO. Getting from a phone video of a construction worker to `motion.npz` is not part of it.

**Three additions**

*One: a video front end.* GVHMR recovers SMPL-X from monocular RGB, GMR retargets to the robot's kinematics, and a MuJoCo forward-kinematics pass produces the per-link world poses and finite-difference velocities HDMI needs and GMR does not emit. That FK stage exists solely to close that gap and is the piece no upstream repo provides. Real work, genuinely useful, but it is plumbing in front of an unchanged method. It does not by itself make this a different pipeline.

*Two: an objective the demonstration does not contain.* The paint target is a rectangle of wall, reconstructed after the fact from where the roller actually went: fit a vertical plane through the roller-head centerline by SVD, push it one roller radius away from the human so it is the wall surface rather than the roller axis, then bound the projected sweep. Online, the roller's swept quad is rasterized into a 5 mm coverage mask every step and the newly covered fraction is the reward.

This term is different in kind from everything else in HDMI. It is an *outcome*: it scores what ended up on the wall, and it is indifferent to how the robot got there. The tracking terms say "be the worker"; the coverage term says "paint the wall." With the target where the demo painted, they agree. That agreement is what makes the task learnable at all, and their disagreement is what makes it interesting.

*Three: goal randomization.* The target region moves and resizes on the wall while the reference motion stays fixed. Position slides in wall coordinates (`target_region_uv_range`, u along the in-plane horizontal and v vertically) rather than world xyz, because a world x/y offset would push the region off the wall plane. Size scales 0.7x to 1.3x. The policy observes the region's size, its centre in the robot's root frame, the roller head's position inside it, and a 4x4 map of what has been covered so far.

That third piece is what converts an outcome objective into a goal-conditioned one. Without it the policy paints one rectangle in one place and the coverage reward is an elaborate way of rewarding the demonstrated stroke.

**The pipeline**

```
worker video (monocular RGB)
  |
  |  GVHMR                     SMPL-X, global trajectory
  |  GMR                       robot root pose + joint angles
  |  MuJoCo FK                 per-link poses + velocities   <- not in any upstream repo
  v
motion.npz + meta.json         the reference: what the human did
  |
  |  object annotation         MANUAL: which link holds the tool, when, where
  |  repack                    object body + per-frame contact flags
  v
motion.npz with the object
  |
  |  roller-head extraction    the tool's working surface, per frame
  |  plane fit (SVD, vertical) the wall the human was working on
  |  sweep bounding box        the region they covered           <- THE TASK
  v
paint_target_rectangle.npz     the goal: what the human was doing
  |
  |  randomize position + size on the wall
  |  coverage rasterizer, 5 mm, batched over envs
  v
goal-conditioned policy
```

The left column is retargeting the motion. The step marked THE TASK is where the goal gets recovered, and everything below it is task-centric rather than motion-centric.

**What is demonstrated and what is not**

Demonstrated, as of 2026-09-30:

The video-to-policy path works end to end on one clip. The coverage objective is correct — replay of the reference reaches 0.73 coverage, reproducibly, and the rasterizer is verified bitwise-identical to a reference implementation. Multi-motion loading works for N clips with per-motion goal regions.

A policy trained with it, evaluated from the first frame of the clip with the target where the demo painted it, completes the full 342-frame stroke in 76% of episodes and reaches 0.74 coverage, level with the replayed demonstration's 0.73. The episodes that finish cover 0.92.

Read training metrics with care. During training every episode starts at a random frame of the clip, and `success` means reaching the end from wherever it started, so the training-time figure (0.795 for this policy) overstates full-stroke performance. The evaluation numbers here start from frame 0 and count exactly one episode per env; averaging `play.py`'s printed stats instead undercounts long successful episodes and understated this policy at 0.68 and 0.69.

Not demonstrated by any of that, and this is the important part:

**That the outcome objective buys generalization the reference alone cannot.** Every number above comes from a policy painting the one rectangle, in the one place, that its demonstration painted. That is HDMI with an extra reward. The claim in the thesis — cover regions the worker never painted — needs a policy trained with the target moving. The first such test is below, under "The result", and it is a partial yes.

**The baseline: the fixed-target policy ignores the target**

Measured 2026-09-30, `pipeline/scripts/zsweep_eval.py`, 128 episodes per offset, one per env, from frame 0. The target is pinned at vertical offsets from -0.5 to +0.5 m and the fixed-target policy is evaluated at each. `cov|success` is coverage over the episodes that finish the stroke:

| z offset | coverage | success | cov\|success |
| --- | --- | --- | --- |
| -0.50 | 0.350 | 0.73 | 0.440 |
| -0.30 | 0.550 | 0.75 | 0.685 |
| -0.10 | 0.662 | 0.72 | 0.878 |
| 0.00 | 0.742 | 0.76 | 0.922 |
| +0.10 | 0.661 | 0.73 | 0.862 |
| +0.30 | 0.458 | 0.68 | 0.646 |
| +0.50 | 0.306 | 0.72 | 0.414 |

Two things make this a clean baseline rather than just a falling curve.

Coverage falls by almost exactly what geometry predicts for a policy that paints the same spot every time. If the painted region stays put and only the target moves, coverage at an offset is the z=0 coverage scaled by how much of the moved target still overlaps it. Measured coverage matches that prediction to within 0.95-1.16x across the whole range, and `cov|success` to within 1.06-1.15x; the small excess is the painted region being a little larger than the target rectangle, not adaptation.

Success does not move with the target — 0.68 to 0.79 at every offset. The robot drops the roller at the same rate wherever the goal is. The coverage loss is entirely the target walking away from where the robot paints.

So the policy is not goal-conditioned at all, and the sweep's own summary line — coverage holding above 80% of z=0 over +-0.2 m — is overlap, not generalization: a 0.85 m target shifted 0.2 m still covers about three quarters of the same wall.

**The experiment that decides it**

Train on `wall_painting_goal`, where the region slides +-0.35 m horizontally and +-0.40 m vertically on the wall and scales 0.7x to 1.3x, with the reference motion unchanged throughout. Then evaluate coverage at held-out target positions, against two baselines: the fixed-target policy evaluated at the same positions (`pipeline/scripts/zsweep_eval.py`), and the reference motion replayed there, which scores whatever the demonstrated stroke happens to cover at a region it was not aimed at.

The baseline makes the test sharp. Overlap alone predicts 0.31 coverage at +-0.5 m. A goal-conditioned policy that follows the target should hold well above that at the edges while giving up little at z=0; one that lands near 0.31-0.35 out there is still painting the demonstrated spot.

If the goal-conditioned policy beats the baseline across the range, the claim holds and the pipeline is doing something HDMI does not. If it only matches the fixed-target policy near zero offset and degrades the same way outside it, then the ten tracking terms are still in charge, the outcome objective is decoration, and the honest next move is to replace phase-indexed tracking with something that does not pin the arm to one trajectory — a motion set spanning the postures, or an adversarial motion prior (AMP, Peng et al. 2021) that rewards moving like the demonstration without dictating where.

One caveat for reading the comparison. The fixed-target policy is the product of about 1.05B frames over five runs; the goal-conditioned run gets 450M from scratch, because its extra observation changes the input size and no checkpoint carries over. A small shortfall at z=0 is partly less training. The edges are where the method shows.

Either result is worth having. The second one is worth having sooner rather than later.

**The result, 2026-09-30: it follows the target down, when it holds on**

The goal-conditioned policy after 450M frames from scratch, same sweep, 128 episodes per offset. Standard error on `cov|success` is at most 0.010:

| z offset | fixed-target coverage | cov\|success | goal-conditioned coverage | cov\|success |
| --- | --- | --- | --- | --- |
| -0.50 | 0.350 | 0.440 | 0.321 | **0.639** |
| -0.30 | 0.550 | 0.685 | 0.345 | **0.800** |
| -0.10 | 0.662 | 0.878 | 0.360 | 0.888 |
| 0.00 | 0.742 | 0.922 | 0.301 | 0.817 |
| +0.10 | 0.661 | 0.862 | 0.264 | 0.718 |
| +0.30 | 0.458 | 0.646 | 0.166 | 0.510 |
| +0.50 | 0.306 | 0.414 | 0.091 | 0.293 |

By the test above it fails: mean coverage is below the fixed-target policy at every offset, including the edges. The reason is grip, not the target. It finishes the stroke in 27-30% of episodes at every offset against the baseline's 68-79%.

Take the drop rate out and the thesis shows up. In the episodes that finish, the goal-conditioned policy covers 0.639 of a target moved 0.5 m down, against the baseline's 0.440 and against 0.338 for painting its own z=0 spot. The margin grows with the offset. That is a policy painting a region its demonstration never painted, because the objective told it where.

It does not follow upward. Above z=0 it paints like a fixed-spot policy and sits below the baseline, and its whole painted band is lower than the demo's, topping out near 1.55 m against the demo's 1.649 m. Up is mostly out of standing reach anyway, which is why the result is "down only" rather than "no".

So the effect is real in the reachable direction and masked by a weak grip. The comparison is not yet even: 450M frames with the product contact reward from a cold start, against 1.05B and a mean-then-product curriculum for the baseline. The next measurement is the same sweep after more training, with success from frame 0 as the number that has to move. Details and commands are in the 2026-09-30 evening run log of `DOC_wall_painting_stages.md`.

**Where postures come in**

The demonstration paints 0.796-1.649 m with the pelvis fixed at 0.809 m and the feet planted. All of it is arm reach. A target at v = -0.40 sits at 0.40 m, which no amount of arm reach covers from a standing stance.

So there is a ceiling on goal conditioning from one clip that is physical, not algorithmic, and filming the same task with knees bent and in a squat raises it. That is a motion set, and the two code changes it needs are in (see the multi-motion section of `DOC_wall_painting_stages.md`): `object_contact` concatenates across clips, and each clip carries its own goal region.

The ordering matters. Randomized targets on one clip tells you where the physical ceiling is. Filming before that measurement risks shooting two videos to solve a problem that was never the binding one.

**Honest summary**

Today this is HDMI with a video-derived reference and an outcome-based task objective. That is a real extension and a useful one, and it is not a different pipeline.

What would make it one is a policy that covers regions it was never shown, driven by an objective recovered from the video rather than by the trajectory in it.

Status, 2026-09-30 evening: the baseline is measured and is not goal-conditioned. The first goal-conditioned policy is measured and is, downward and only in the episodes where it keeps the roller, which is 27-30% of them. The next step is more training for grip, then the same sweep.
