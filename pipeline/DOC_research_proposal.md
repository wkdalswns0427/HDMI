**Retargeting the task, not the motion: research plan to RSS 2027**

Plan as of 2026-10-01. The hardware is a Unitree H1-2. Simulation runs on G1, where the preliminary results are, and on H1-2, the hardware platform, where shoveling already trains. The framing is in `DOC_task_centric_retargeting.md` and the build log in `DOC_wall_painting_stages.md`; this is what we will claim, how we will test it, and the schedule.

**The venue fits the project**

RSS 2027 (Athens, July 6-11) reviews hypotheses before results. Stage 1 is a 6-page anonymized extended abstract that must fully specify the problem, hypotheses and methodology, while experiments may be missing or preliminary. Papers that pass get an Extension Charter from the area chair, execute it, and submit an 8-page final paper; the committee checks the charter was followed. Negative results are explicitly not grounds for rejection. We already test a stated hypothesis against a null model, and the real-robot evaluation can land in the extension period.

| date | stage |
| --- | --- |
| Dec 4, 2026 | extended abstract: 5 pages + 1 of references, anonymized |
| Feb 5, 2027 | stage 1 decision: reject or invite to rebuttal |
| Feb 12 | rebuttal due |
| Feb 26 | stage 2 decision: reject or Extension Charter |
| Apr 16 | final paper, 8 pages, charter executed |
| Apr 19 / Apr 30 | supplementary video / acceptance |

Dates from roboticsconference.org as read on 2026-09-30; re-check before committing to them. The fork `github.com/wkdalswns0427/HDMI` is public under a personal account, so the submission must not link it; use an anonymized mirror if code is cited.

**Pitch**

Humanoid learning from human video retargets the human's motion, and the policy reproduces the demonstrated trajectory; HDMI randomizes initial object placement over only 10-20 cm. We retarget the task instead. From one monocular video we recover what the person accomplished (the wall region a roller covered, the pose a box ended in) and train a whole-body policy rewarded for that outcome under randomized goals, while tracking the demonstration for feasibility and style. The claim: a real humanoid achieves goals its demonstration never showed.

**Relation to HDMI**

We build on HDMI and say so. Its tracking rewards, object-frame contact targets, network and PPO-ROA teacher-student training are our backbone, and HDMI with and without our reward are baselines B1 and B2. New inputs, a different network or a fine-tuned student do not make a new method; the objective does. HDMI optimizes "match the demonstration"; we optimize "achieve the demonstrated outcome at a commanded goal", with the demonstration as a prior. The paper stands on whether that difference reaches goals tracking cannot (H1-H3), not on how much code differs.

**Hypotheses, pre-registered**

Null model for every coverage test: a policy that paints the same spot wherever the goal is scores its demo-goal coverage times the overlap of the moved goal with the original ("overlap-only"). Today's fixed-target policy sits on that line at 0.95-1.16x.

| | hypothesis | test | falsified if |
| --- | --- | --- | --- |
| H1 | goal-conditioned training on one video covers held-out reachable goals the demo never painted | T1 sim sweeps over v, u and size, on H1-2 and G1, 3 seeds | mean coverage over goals ≥0.3 m from the demo goal is <0.10 above the fixed-target policy, or demo-goal success drops >0.10 |
| H2 | the gain comes from goal conditioning, not randomization alone | remove the goal observation | the ablated policy also clears the overlap-only line |
| H3 | an outcome objective beats editing the demonstration | B3: IK-edit the reference to each goal, then track it | B3 matches ours within s.e. |
| H4 | posture clips extend reach | stand vs stand + bend + squat | the lower edge of the ≥80% band moves down <0.2 m |
| H5 | the recipe carries to a second outcome type | T3 box placement, H1's test | H1's criterion fails on T3 |
| H6 | it transfers to hardware | real H1-2: demo goal + 4 unseen, 10 trials each, ours vs fixed-target | ours beats fixed-target at fewer than 3 of the 4 unseen goals |

A negative H3 (editing the reference suffices) would still be a publishable result under this format.

**Preliminary results (simulation, G1, T1, one standing clip)**

The pipeline runs end to end: phone video → GVHMR → GMR → MuJoCo FK → HDMI, with the target recovered by an SVD plane fit and a sweep bound. Replaying the demo scores 0.73 coverage.

The fixed-target policy (HDMI plus the coverage reward) completes 76% of strokes from frame 0 and covers 0.74. When the target moves, its coverage follows the overlap-only line: it ignores the goal.

The goal-conditioned policy (450M frames from scratch, then 450M more) follows the goal downward. It completes 48-56% of strokes against the fixed-target policy's 66-78%, yet its mean coverage is higher at 0.5 m and 0.4 m down, and in completed strokes its IoU with the target is higher at every downward offset. It does not follow upward: the demonstrated stroke already reaches 1.65 m, near the top of standing reach. Against H1 it is close but not there: the margin clears 0.10 only at 0.5 m down, and demo-goal success is 0.25 below the fixed-target policy's.

| target moved | fixed-target coverage | goal-conditioned coverage | fixed-target IoU, completed strokes | goal-conditioned IoU, completed strokes |
| --- | --- | --- | --- | --- |
| 0.5 m down | 0.36 | **0.50** | 0.27 | **0.50** |
| 0.3 m down | 0.53 | **0.56** | 0.48 | **0.64** |
| demo goal | 0.75 | 0.52 | 0.80 | 0.73 |
| 0.3 m up | 0.50 | 0.30 | 0.45 | 0.44 |

Those numbers are projected paint: the rasterizer counted the roller's sweep at any distance and the wall had no collision, so while painting the fixed-target policy held the roller 1.4-3.5 cm off the wall and the goal-conditioned one pushed it up to 4 cm in. With a solid wall and paint only on contact (≥2 N), and 450M frames of fine-tuning, the goal-conditioned policy completes 65-77% of strokes from frame 0, at the fixed-target policy's level. It covers 0.45 of the target in a completed stroke at the demo goal with 0.97 precision. In completed strokes its coverage stays 1.39-1.62x its own overlap-only line from 0.2 m to 0.5 m down, so downward goal-following survives real contact. The H1-2 port, trained from scratch on the same video, completes 59-72% of strokes from frame 0 and shows the same downward following, 1.37-1.62x its line, though it covers less per stroke (0.26 vs 0.45 at the demo goal). The fixed-target baseline under the same physics is not trained yet, and only vertical offsets have been measured; horizontal and size sweeps are part of H1.

**Method**

Per task, only step 2 is manual:

- Video → robot reference: GVHMR, GMR, MuJoCo FK. GMR retargets the same video to G1 and to H1-2.
- Object annotation: which link holds the tool, and when. Minutes per clip; stated as a limitation.
- Outcome extraction: for coverage, the tool's working surface in contact → surface plane by SVD → covered region; for placement, the object's final pose.
- Training: randomize the goal with the reference fixed. The policy sees the goal in its root frame and task progress (4×4 coverage map, or object-to-goal error). Reward = HDMI tracking + contact + outcome progress. PPO teacher with privileged state, ROA student for deployment.
- Posture motion sets: several clips of the same task, each with its own goal region (implemented).

Three changes before the next training round, all forced by hardware:

- Wall contact. The canvas has no collision today and paint is counted from the roller's projection, so policies paint from 1.4-3.5 cm off the wall or from 4 cm inside it. Enable collision, count paint only while the roller touches the wall with force in a band, randomize wall position ±2 cm.
- End-effector. The H1-2's dexterous hands are not used. Where grip is needed, a task-specific static hand designed for the roller handle goes on the wrist, and the same rigid shape is modeled in sim on both robots. `pipeline/scripts/make_h1_2_eef.py` is where it goes: the handless H1-2 model has no wrist colliders, and shoveling already uses a capsule authored there. Grip is the current failure in sim, so a shape that captures the pole helps in both places.
- H1-2 port of T1. The pipeline and HDMI already run shovel_dirt on H1-2 (`h1_2_handless-eef_capsule`). The roller needs an H1-2 variant because GMR scales the motion to the robot and the grip span changes with it (the shovel's went from 0.438 m on G1 to 0.745 m on H1-2).

**Tasks**

| task | outcome | goal space | reference | status | hardware |
| --- | --- | --- | --- | --- | --- |
| T1 wall painting | coverage of a wall rectangle | u ±0.35 m, v -0.5..+0.4 m, size 0.7-1.3x | worker phone video, retargeted to G1 and H1-2; bend and squat clips to film | G1 trained, partial; H1-2 to port | H1-2, primary |
| T3 box placement | final box pose | placement position and height | own filmed clip | to film | H1-2, stretch |
| T2 floor painting | coverage of a floor region | position, size | `floor_painting` clip in repo | config exists | no |

T3 carries H5 because it is a different outcome type; T2 is cheap evidence that coverage extraction is not wall-specific.

**Baselines, ablations, protocol**

- Baselines: B1 HDMI tracking only. B2 HDMI + outcome reward, fixed goal (have, on G1). B3 IK-edited reference at the new goal, tracked. B4 motion prior (AMP; HDMI ships `amp_omomo-lsgan`) + outcome reward, no phase-locked tracking.
- Ablations: goal observation (H2), coverage-map observation, randomization range, posture clips (H4), contact reward form, outcome weight.
- Sim metrics: 128 episodes per goal, one per env, all from frame 0 (`pipeline/scripts/eval_episodes.py`; averaging `play.py`'s printed stats is biased toward short episodes). Coverage, full-stroke success, coverage in completed strokes, and the width of the band where coverage ≥80% of the demo-goal value. IoU and precision against the target, with paint outside it counted on a raster over the whole wall. Roller-to-wall distance while painting. Mechanical work and power by joint group (legs, waist, arms), RMS and peak torque, work per m² painted. Overlap-only curve on every plot. 3 seeds, mean ± s.e.
- Real metrics: 5 goals (demo region; 0.3 m down; 0.5 m down; 0.3 m left; 0.3 m right), 10 trials each for ours and B2, 100 trials in randomized order. Coverage measured two ways that cross-check: the tracked roller-head path rasterized by the sim code, and a photo of real water-based paint on replaceable paper, segmented inside the marked rectangle. Success = full stroke, no tool drop, no safety stop. Paired per-goal comparison with bootstrap 95% CIs.

**Real-robot plan**

Hardware: Unitree H1-2 with the task-specific static hand, roller matched to the H1-2 sim variant, plywood wall with replaceable paper and foam backing, gantry for early sessions.

Runtime state: proprioception and the reference motion are onboard. Every object observation (roller position and heading and the reference contact points in the base frame, roller head position inside the target, target centre in the base frame, target size, 4×4 coverage map) derives from three poses (robot base, roller, wall) plus the sim's rasterizer run onboard. Plan A: motion capture markers on pelvis, roller and wall. Plan B: AprilTags and a camera, with its noise and latency matched in training. Which one is decided by Oct 8.

Deployment: HDMI's sim2real stack (`github.com/EGalahad/sim2real`: MuJoCo sim2sim, Unitree SDK bridge, ONNX) documents G1 only, so H1-2 needs its robot config added (joint order, PD gains, SDK bridge) or our own control loop on the Unitree SDK. It also has no object-state input, so we build a tracking-to-policy bridge that computes the object observations. Policy export: `play.py export_policy=true`. Ladder: student distillation (`ppo_roa_finetune`) → MuJoCo sim2sim → gantry at the demo goal → free-standing at the demo goal → goal sweep.

**Schedule**

One owner (mj) on both tracks until ce joins. Training runs unattended on the GPU, so runs go overnight and hardware work fills the days; when ce joins, hardware is the track to hand over.

Phase A, to the abstract (Oct 1 - Dec 4):

| window | sim and method | hardware | gate |
| --- | --- | --- | --- |
| Oct 1-14 | sweep the G1 continuation; wall collision and force-gated paint; port T1 to H1-2 (retarget, roller variant, static-hand model) and train it | tracking decision (Oct 8); static hand v1; wall rig | A1: ≥0.60 full strokes from frame 0 in sim |
| Oct 15 - Nov 4 | film bend/squat and box placement; posture set (H4); u and size sweeps; B3; T3 through the pipeline | H1-2 deploy config; student distillation; sim2sim; tracking bridge | A2: H1's criterion met on H1-2 in sim, 1 seed |
| Nov 5-20 | 3-seed sweeps; H2 ablation; B1, B4 | gantry trials at the demo goal | A3: H1-2 completes one stroke on hardware (strengthens the abstract, not required) |
| Nov 21 - Dec 4 | write the abstract; internal review Nov 27 | footage of A3 | submit Dec 4 |

Phase B, in review (Dec 5 - Feb 26): T3 and T2 results; free-standing trials and first unseen goals on hardware; rebuttal Feb 5-12 with whatever is new.

Phase C, extension (Feb 26 - Apr 16): execute the charter. Expected: the 100-trial real goal sweep, remaining seeds and ablations, the 8-page paper; video by Apr 19.

If a gate slips, drop in this order: T2, then B4, then G1 seeds beyond one. H1-2, T1, B2, B3 and the hardware sweep are the paper.

**Risks**

| risk | mitigation |
| --- | --- |
| grip stays weak in sim | more frames worked once (the continuation); static hand that captures the pole; mean-then-product contact curriculum |
| wall contact breaks transfer | collision and force-gated paint in sim before hardware; compliant roller cover; force randomization |
| H1-2 deployment: HDMI's sim2real stack documents G1 only | start the H1-2 robot config in week 3; it gates every hardware step |
| goals above the demo are unreachable | define the goal space by reach and say so; posture clips extend it downward |
| "reward engineering" critique | one extraction per outcome type, shared across tasks; state what is manual |
| novelty vs DeepMimic/AMP (imitation + task reward) and HDMI/VideoMimic (humanoids from video) | the claim is the combination: goal inferred from one video, unseen goals, real humanoid tool use; B3 and B4 test it directly |
| one person on both tracks | GPU runs overnight; the drop order above; hand hardware to ce when available |
| compute | all planned runs ≈ 320 GPU-hours, about two weeks of the RTX 5090 nonstop; a second GPU halves the calendar time |

**Decisions, answered 2026-10-01**

1. End-effector: task-specific static hand on the H1-2, dexterous hands unused; the same shape modeled in sim.
2. Tracking: motion capture or AprilTags, decided by Oct 8.
3. Robot: H1-2. Session schedule to fix before gantry trials in November.
4. Owners: mj on both tracks until ce joins.
5. T3: our own filmed box-placement clip, so every task comes from video.

**Abstract outline (5 pages)**

Introduction and contributions 0.75 · related work 0.5 · method 1.25 · hypotheses and experimental design 1.5 (hypothesis table, baselines, protocol, statistics) · preliminary results 0.75 · what the extension will deliver 0.25.
