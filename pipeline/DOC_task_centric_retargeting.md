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

The video-to-policy path works end to end on one clip. The coverage objective is correct — replay of the reference reaches 0.73 coverage, reproducibly, and the rasterizer is verified bitwise-identical to a reference implementation. A policy trained with it reaches 0.795 success and covers 45% of a 342-frame clip, converged. Multi-motion loading works for N clips with per-motion goal regions.

Not demonstrated, and this is the important part:

**That the outcome objective buys generalization the reference alone cannot.** Every number above comes from a policy painting the one rectangle, in the one place, that its demonstration painted. That is HDMI with an extra reward. The claim in the thesis — cover regions the worker never painted — has not been tested, because until the target started moving there was nothing to test.

**The experiment that decides it**

Train on `wall_painting_goal`, where the region slides +-0.35 m horizontally and +-0.40 m vertically on the wall and scales 0.7x to 1.3x, with the reference motion unchanged throughout. Then evaluate coverage at held-out target positions, against two baselines: the fixed-target policy evaluated at the same positions (`pipeline/scripts/zsweep_eval.py`), and the reference motion replayed there, which scores whatever the demonstrated stroke happens to cover at a region it was not aimed at.

If the goal-conditioned policy beats both across the range, the claim holds and the pipeline is doing something HDMI does not. If it only matches the fixed-target policy near zero offset and degrades the same way outside it, then the ten tracking terms are still in charge, the outcome objective is decoration, and the honest next move is a motion set or an AMP-style prior in place of phase-indexed tracking — see `DOC_goal_conditioned_study.md`.

Either result is worth having. The second one is worth having sooner rather than later.

**Where postures come in**

The demonstration paints 0.796-1.649 m with the pelvis fixed at 0.809 m and the feet planted. All of it is arm reach. A target at v = -0.40 sits at 0.40 m, which no amount of arm reach covers from a standing stance.

So there is a ceiling on goal conditioning from one clip that is physical, not algorithmic, and filming the same task with knees bent and in a squat raises it. That is a motion set (`DOC_multi_demo_same_task.md`), and the two code changes it needs are in: `object_contact` concatenates across clips, and each clip carries its own goal region.

The ordering matters. Randomized targets on one clip tells you where the physical ceiling is. Filming before that measurement risks shooting two videos to solve a problem that was never the binding one.

**Honest summary**

Today this is HDMI with a video-derived reference and an outcome-based task objective. That is a real extension and a useful one, and it is not a different pipeline.

What would make it one is a policy that covers regions it was never shown, driven by an objective recovered from the video rather than by the trajectory in it. The machinery for that now exists and the experiment is specified. It has not been run.
