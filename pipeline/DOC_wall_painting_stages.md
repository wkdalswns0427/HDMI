### **wall painting existing worker video**

Same seven stages as shovel_dirt, plus one that shovel does not have. Shovel asks "hold the tool and reproduce the motion." Painting asks that *and* "cover this rectangle of wall", which is a goal the demonstration does not contain. Stage 8 is where that goal gets built.

**What is different from shovel_dirt, in one table**

| | shovel_dirt | wall_painting2 |
| --- | --- | --- |
| object | shovel, 2.5 kg capsule shaft | roller, 2.5 kg pole + head cylinder |
| grip | `two_hand` + `grip_offset` into the eef_L pocket | `two_hand`, **no** `grip_offset` (wrist origins used directly) |
| end-effector | `eef_L` (L-hook) | `eef_box` with rubber hands |
| grip span | 0.138 m drift over the clip | 0.503 m mean, near-constant |
| reward groups | tracking + object_tracking | tracking + object_tracking + **task** |
| what "success" means | reproduce the reference | **cover the target rectangle** |
| extra offline work | none | **Stage 8**: fit the wall, extract the target rectangle |
| extra online machinery | none | **Stage 9**: rasterize the roller sweep every step |

**Task formulation**

Three reward groups stack, and they are pulling in different directions on purpose.

`tracking` — ten terms, weight 0.5 each, comparing the robot to the reference clip frame by frame indexed by `ref_motion_phase`. Upper/lower body position and orientation, root position and orientation, body linear and angular velocity, joint position and velocity. This is what makes it look like the worker.

`object_tracking` — `object_pos_tracking` and `object_ori_tracking` (weight 1.0) keep the roller on its demonstrated path; `eef_contact_exp` (weight 0.5) rewards both hands staying on the shaft at the two contact points. Weight 0.5 not 1.0 because `eef_contact_exp` has `gain=5.0` and at weight 1.0 it scores ~3.5/step while the ten tracking terms together score ~2.67 — one contact term outweighing all of tracking.

`task` — `paint_coverage_delta`, weight 400. This is the new one. It rewards newly covered area on the wall, and it is the only term that does not care how the robot gets there. Weight 400 because the per-step delta is ~0.002 (0.2% of the rectangle), so 400 puts it at ~0.8/step, comparable to a tracking term.

The interesting tension: tracking says "be the worker", coverage says "paint the wall." With the target pinned where the demo painted, they agree. Move the target and they stop agreeing, which is exactly the experiment the machinery exists to run.

**Observations the policy gets that shovel's does not**

| obs | shape | what it says |
| --- | --- | --- |
| `target_region_size` | 2 | rectangle w, h normalized to the demonstrated size |
| `roller_head_uv` | 2 | roller head centre in target coordinates, ±1 at the edges |
| `paint_coverage_grid` | 16 | 4×4 map of which parts of the rectangle are painted |
| `target_center_b` | 3 | target centre in root frame — **off by default**, needed only when the target moves |

- **Stage 0: Export setup**

    ```jsx
    cd ~/mj_ws/simbench

    export SB=~/mj_ws/simbench
    export PY_GVHMR=~/miniconda3/envs/gvhmr/bin/python
    export PY_GMR=~/miniconda3/envs/gmr/bin/python
    export PY_HDMI=~/miniconda3/envs/hdmi/bin/python

    export CLIP="$SB/GVHMR/inputs/demo/construction video2/wall_painting2.mp4"
    export TASK=wall_painting2
    export ROBOT=unitree_g1
    export MOTION=$SB/HDMI/data/motion/data_for_sim/$TASK
    export WORK=$SB/HDMI/pipeline/work/$TASK
    mkdir -p "$WORK"
    ```

    `PYTHONNOUSERSITE=1` for GVHMR and GMR only. Never for hdmi — that env resolves `typing_extensions` out of `~/.local` and torch fails to import at line 34 without it.

- **Stage 1: Video .mp4 to GVHMR .pt**

    ```jsx
    cd $SB/GVHMR
    PYTHONNOUSERSITE=1 $PY_GVHMR tools/demo/demo.py --video="$CLIP"
    cd -
    ```

    Output lands in `outputs/demo/wall_painting2/hmr4d_results.pt`. Same torch-2.6 `weights_only` trap as shovel — `patch_gvhmr_torch26.sh` handles it.

    **Stage 1.1: drop down SMPL video**

    ```jsx
    cd $SB/GVHMR
    PYTHONNOUSERSITE=1 $PY_GVHMR $SB/HDMI/pipeline/scripts/render_smpl.py \
      "outputs/demo/wall_painting2/hmr4d_results.pt" \
      -o $SB/HDMI/pipeline/wall_painting2_human.mp4
    ```

- **Stage 2: Retarget joints**

    ```jsx
    cd $SB/HDMI/pipeline
    PYTHONNOUSERSITE=1 $PY_GMR scripts/gvhmr_to_gmr.py \
      --gvhmr_pred "$PRED" --robot $ROBOT --video "$CLIP" \
      --save_path "$WORK/$TASK.pkl"
    ```

- **Stage 3: Forward Kinematic solve validate**

    ```jsx
    PYTHONNOUSERSITE=1 $PY_GMR $SB/HDMI/pipeline/scripts/gmr_to_hdmi.py \
      --gmr_pkl "$WORK/$TASK.pkl" --robot $ROBOT \
      --out_dir "$MOTION" --target_fps 50
    // 342 frames @ 50 fps (6.84s), 28 bodies -- no object yet
    ```

    Worth knowing before anything else: **the pelvis does not move in this clip.** Over all 342 frames it sits at 0.809 m and shifts 7 mm; the knees shift 16 mm. All 0.853 m of painted wall comes from arm reach with the feet planted. That single fact drives everything downstream — the roller geometry, the target rectangle, and the question of whether a second clip at a lower posture is needed.

- **Stage 4: Train teacher** (body only, no roller)

    ```jsx
    cd $SB/HDMI
    $PY_HDMI scripts/train.py algo=ppo_roa_train task=G1/tracking/wall_painting2
    ```

    Optional for painting — the object is the point here, so you can go straight to Stage 6. Useful only as a sanity baseline that the retarget is trackable at all.

- **Stage 5: train student**

    ```jsx
    export TEACHER=.../checkpoint_final.pt
    $PY_HDMI scripts/train.py algo=ppo_roa_finetune task=G1/tracking/wall_painting2 \
      checkpoint_path=$TEACHER
    ```

    Same note as shovel: `enable_residual_distillation: True` means the student was already being distilled during teacher training, so it does not start from scratch.

- **Stage 6: Rigid Object → roller**

    **6.1 — author the asset.** Same nested-same-name USD layout constraint as the shovel: `/roller` Xform → `/roller/roller` rigid body → collision children, because `object_body_name` is used both as a prim path and as `body_names.index()`.

    ```jsx
    $PY_HDMI $SB/HDMI/pipeline/scripts/make_roller_usd.py
    // defaultPrim: /roller
    //   /roller/roller              APIs=['RigidBodyAPI','MassAPI']   mass 2.5
    //   /roller/roller/pole         Capsule   r=0.0209, 1.0 m, axis z
    //   /roller/roller/head         Cylinder  r=0.0425, l=0.2291, axis x
    //   + connector_1..4, joint_1..3

    // active_adaptation/assets/objects.py -> OBJECTS["roller"] = ROLLER_CFG
    ROLLER_CFG = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/roller",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ASSET_PATH}/objects/roller/roller.usd",
            activate_contact_sensors=True,
            mass_props=sim_utils.MassPropertiesCfg(mass=2.5),
        ),
    )
    ```

    The head constants matter later — head centre `[-0.000457, 0.000206, 0.270269]` in the roller's local frame, axis local +X, length 0.229137. Stage 9 hardcodes them.

    **6.2 — measure the grip, then annotate.**

    ```jsx
    PYTHONNOUSERSITE=1 $PY_GMR scripts/check_grip_span.py --motion_dir "$MOTION"
    // mean 0.5031 m   median 0.5028   std 0.0042   range 0.0201
    ```

    0.503 m with 4 mm of drift — far steadier than the shovel's 0.138 m. That is why the roller annotation can pin `axis_offset` at 0.700 m and leave `grip_offset` out entirely: the wrist link origins are usable grip points directly. Setting `grip_offset` here would repeat the shovel's lever-arm failure (contact loss 0.0019 → 0.0150 per step).

    ```jsx
    // pipeline/annotations/wall_painting2.json
    {
      "object": {"name": "roller", "asset": "roller"},
      "segments": [
        {"start": 0, "end": 341, "mode": "two_hand",
         "parent_a": "right_wrist_yaw_link",
         "parent_b": "left_wrist_yaw_link",
         "axis_offset": 0.700, "local_axis": "z"}
      ],
      "contact": [[0, 341]],
      "smoothing": {"window": 5},
      "contact_target_pos_offset": [[0.0, 0, -0.197], [0.0, 0, -0.700]],
      //                              ^ left (-0.197)   ^ right (-0.700)
      //  ORDER IS [left, right], matching contact_eef_body_name in
      //  base/hdmi-base.yaml. These were reversed until 2026-09-29. See the
      //  run log at the bottom of this doc.
      "contact_eef_pos_offset":    [[0.05, 0.0, 0.0], [0.05, 0.0, 0.0]]
    }
    ```

    **6.3 — repack with the object, then validate.**

    ```jsx
    PYTHONNOUSERSITE=1 $PY_GMR scripts/gmr_to_hdmi.py \
      --gmr_pkl "$WORK/$TASK.pkl" --robot unitree_g1 \
      --out_dir "$MOTION" --annotation annotations/wall_painting2.json --target_fps 50
    // [object] roller: contact on 342/342 frames
    //   bodies : 29  [... 'roller']      joints : 29      frames : 342 @ 50 fps

    PYTHONNOUSERSITE=1 $PY_GMR scripts/validate_motion.py "$MOTION"
    ```

    **6.4 — preview and task config.**

    ```jsx
    PYTHONNOUSERSITE=1 $PY_GMR scripts/render_motion.py "$MOTION" \
      --robot unitree_g1 -o $SB/HDMI/pipeline/wall_painting2_gmr.mp4

    PYTHONNOUSERSITE=1 $PY_GMR scripts/make_task_cfg.py \
      --motion_dir "$MOTION" --annotation annotations/wall_painting2.json \
      --name WallPainting2 --robot g1 \
      --robot_type g1_29dof_rubberhand-feet_sphere-eef_box-body_capsule \
      --out $SB/HDMI/cfg/task/G1/hdmi/wall_painting2.yaml
    ```

    Note the end-effector differs from shovel: `eef_box` with rubber hands, not the `eef_L` hook.

    **6.5 — confirm it spawns.**

    ```jsx
    cd $SB/HDMI
    $PY_HDMI scripts/play.py algo=ppo_roa_train task=G1/hdmi/wall_painting2 \
      +task.command.replay_motion=true task.num_envs=1
    // Using object type roller with asset .../objects/roller/roller.usd
    // Reward group: object_tracking
    ```

- **Stage 7: Train Teacher & Student with roller on contact**

    ```jsx
    $PY_HDMI scripts/train.py algo=ppo_roa_train task=G1/hdmi/wall_painting2 \
      save_interval=50

    export TEACHER=.../checkpoint_final.pt
    $PY_HDMI scripts/train.py algo=ppo_roa_finetune task=G1/hdmi/wall_painting2 \
      checkpoint_path=$TEACHER
    ```

    `save_interval=50` rather than the default 300. The first attempt at this task ran 112 minutes, reached iteration 120, and saved nothing, because 120 < 300.

- **Stage 8: Extract the paint target (offline, painting only)**

    Shovel has no equivalent. The goal — a rectangle of wall to cover — is not in the demonstration, so it has to be recovered from where the roller actually went. Three scripts in `pipeline/task_info/scripts/`, run in order.

    ```jsx
    PYTHONNOUSERSITE=1 $PY_GMR $SB/HDMI/pipeline/task_info/scripts/visualize_roller_swept_area.py
    // re-derives the two_hand roller pose from both wrist links and writes the
    // head centreline endpoints per frame -> roller_head_trajectory.npz  (342 frames)

    PYTHONNOUSERSITE=1 $PY_GMR $SB/HDMI/pipeline/task_info/scripts/fit_roller_plane.py
    // SVD plane through 25 samples per roller line, normal_z forced to 0 (vertical
    // wall), then pushed one roller radius AWAY from the human so it is the wall
    // surface rather than the roller axis -> roller_fitted_plane.npz
    //   rmse 0.0307 m   p95 0.0600 m   normal [0.8787, 0.4774, 0]

    PYTHONNOUSERSITE=1 $PY_GMR $SB/HDMI/pipeline/task_info/scripts/target_area.py
    // axis-aligned bbox of all projected endpoints -> paint_target_rectangle.npz
    //   0.6025 x 0.8529 m, centre [-0.7214, -0.2603, 1.2228], z 0.796 .. 1.649
    ```

    The bbox is exact, not an approximation: any interpolated point on the roller between two frames is a convex combination of four recorded endpoints, so the endpoint extrema enclose the whole sweep.

- **Stage 9: Coverage reward (online, painting only)**

    `target_region_path` in the task YAML points at `paint_target_rectangle.npz`, which switches on machinery in `RobotObjectTracking`:

    `_sample_target_region()` on reset — places the rectangle at the env origin plus a randomized offset, scales it, rebuilds the four corners, clears the paint mask.

    `_update_paint_coverage()` every step — projects the roller head's two endpoints into wall (u,v), forms the quad swept since the previous frame, rasterizes it at 5 mm into a boolean mask by a four-edge half-plane test, and reports `coverage_ratio` and `coverage_delta`.

    ```jsx
    // cfg/task/G1/hdmi/wall_painting2.yaml  (the lines shovel does not have)
    command:
      target_region_path: ".../pipeline/task_info/wall_painting2/paint_target_rectangle.npz"
      target_region_pos_range:   {x: [0,0], y: [0,0], z: [0,0]}
      target_region_scale_range: {width: [1,1], height: [1,1]}
      coverage_grid_size: 4
    observation:
      object:
        target_region_size: {}
        roller_head_uv: {}
        paint_coverage_grid: {}
    reward:
      task:
        paint_coverage_delta: {weight: 400.0}
    ```

    **Verify before training.** Replay at one env should reach ~0.73 coverage on the reference motion:

    ```jsx
    $PY_HDMI scripts/play.py algo=ppo_roa_train task=G1/hdmi/wall_painting2 \
      +task.command.replay_motion=true task.num_envs=1
    ```

    Four measured replays landed at 0.7307, 0.7378, 0.7364, 0.7367 — the spread is reset randomization (`object_pose_range` ±0.05 m, ±0.1 rad), not drift. Anything outside roughly 0.72–0.75 means the wall fit or the head geometry is wrong.

    **The rasterizer must stay batched.** It was originally a Python `for env_id in range(num_envs)` loop with two `.item()` GPU syncs inside. Invisible at one env in replay, fatal at 4096: 2362 rollout_fps and 56 s per iteration. Batched over envs it runs 47,300 fps and 2.8 s per iteration, a 19.6× end-to-end difference — 54 minutes for a full 150M-frame run instead of 17.6 hours. If you ever touch that function, keep it batched and re-check with `pipeline/scripts/` timing before launching a long run.

**Stage 10: Does it generalize in height? (open)**

The demonstration paints 0.796–1.649 m from a fixed stance. The question is whether the policy can paint a rectangle somewhere else. Two levers, cheapest first.

**Move the goal.** `cfg/task/G1/hdmi/wall_painting_goal.yaml` randomizes the target on every reset while the reference stays put: it slides +-0.35 m horizontally and +-0.40 m vertically along the wall (`target_region_uv_range`) and scales 0.7-1.3x, and it switches on `target_center_b`, which the policy needs once the target is no longer where the demo left it. That observation makes the input 33-dim instead of 30, so this task trains from scratch -- no fixed-target checkpoint loads into it.

Randomize position with `target_region_uv_range`, not `target_region_pos_range`. The latter is a world-frame xyz offset, and with this wall's normal at [0.879, 0.477, 0] a world x/y offset of the same size pushes the region up to 0.468 m off the wall plane. The uv offset stays on it to 1e-16 m.

```jsx
$PY_HDMI scripts/train.py algo=ppo_roa_train task=G1/hdmi/wall_painting_goal \
  total_frames=450_000_000 save_interval=50

$PY_HDMI pipeline/scripts/zsweep_eval.py --checkpoint <ckpt> --task <task>
// pins the target at z = -0.5 .. +0.5 m in 0.1 steps; 128 episodes each, one
// per env, all from frame 0 (pipeline/scripts/eval_episodes.py); ~2 min/offset
// reports coverage, success, coverage over successful episodes, episode_len
// --task must match what the checkpoint was TRAINED on: wall_painting2_both
// for the fixed-target policy, wall_painting_goal for the goal-conditioned one
```

**Add postures.** Film the same task with knees bent ~45° and in a squat, retarget each, and train on all three as a motion set. Projected bands: stand 0.80–1.65 (measured), bend ~0.58–1.43, squat ~0.44–1.29, union 0.44–1.65 — 42% more wall for two more shoots. Needed two code changes, both now in (see the multi-motion section below): concatenating `object_contact` across motions and making `target_region_path` per-motion so coverage 1.0 stays reachable in every episode.

Run the sweep before filming. It may show the band you care about is already covered.

**Run log — 2026-09-29: the first time it was actually trained**

Everything above describes the pipeline as designed. This is what happened the first time it ran end to end, because two things were wrong and neither showed up in any reward curve.

**The contact targets were reversed.** `contact_eef_body_name` is fixed in `base/hdmi-base.yaml` as `["left_wrist_yaw_link", "right_wrist_yaw_link"]`, so `contact_target_pos_offset` must be ordered [left, right]. The annotation had it [right, left]: slot 0 held -0.700 (the right wrist's position) and slot 1 held -0.197. Each hand was told to grip where the other belongs.

Measured from `motion.npz`, wrist positions expressed in the roller's local frame: `right_wrist_yaw_link` at z -0.6998, `left_wrist_yaw_link` at z -0.1967, span 0.5032 m. Against the config as written, each hand was off by 0.503 m — exactly the grip span. shovel_dirt gets this right and is the template: it puts `-axis_offset` in slot 1 (right).

**How it hid.** The failure is invisible in the reward curves. Over 1144 iterations the coverage reward rose 12.6% and the tracking terms drifted up slightly, which reads like slow learning. It was not. `episode_len` sat at exactly 25.000 for the entire run — `min_steps` on the termination conditions, meaning every episode died the instant it was allowed to. `success` was 0.000 throughout and `cum_lost_contact_steps` fired on 100% of episodes.

Two metrics that look reassuring and are not: `reward.loco/survival` is a constant 1.0 per step whenever the episode is alive, so it never indicates anything; and reward stats accumulate over the episode (`envs/mdp/base.py`: `stats[...].add_(self.weight * rew)`), so a per-step mean has to be derived by dividing by `episode_len` rather than read directly.

**Check `episode_len` before anything else.** It is the one number that separates "learning slowly" from "not learning". Pull it a few minutes into a run:

```jsx
// in the run dir, read the wandb history stream directly --
// wandb-summary.json is only written on a clean exit
train/stats/episode_len                       // 25.000 flat = dead
train/stats/termination/cum_lost_contact_steps
train/stats/debug/eef_contact_all             // divide by episode_len for per-step
train/stats/success
```

**Before and after the fix**, same config otherwise, 1144 iterations each:

| | swapped | corrected |
| --- | --- | --- |
| episode_len | 25.000 flat | 25.95 -> 48.90 |
| per-step contact | 0.0003 | 0.047 -> 0.230 |
| lost-contact terminations | 100% | 94.8% -> 73.9% |
| success | 0.000 | 0.000 -> 0.123 |

`episode_len` was still accelerating at the end (31.9 at iteration 900, 48.9 at 1144), so the run stopped mid-climb.

**It holds the roller one-handed.** Visible in `pipeline/vidoes/wall_painting2_policy_02_fixed.mp4`: one hand on the shaft, the other parked on the hip for most of the clip. This caps the contact metric at 0.5, because `eef_contact_all` is a mean over the two end-effectors — the final 0.230 is roughly one hand engaged half the time. The policy found a local optimum where half the contact reward is cheap to collect. Worth trying before more frames: make the contact reward multiplicative across hands instead of a mean, so half credit stops being available.

**The rasterizer was 19.6x slower than it needed to be.** Covered in Stage 9 above; the numbers came from here. Batched, it was verified bitwise-identical to the per-env version offline (40 steps x 24 envs, mixed rectangle sizes, zero difference in mask, ratio and grid) and across four live replays whose final coverage values (0.7307, 0.7378, 0.7364, 0.7367) are indistinguishable between versions.

**Artefacts from this run**

```jsx
// the only checkpoint worth using -- corrected contact targets
HDMI/outputs/2026-09-29/11-34-19-G1WallPainting2-ppo_roa/
  wandb/run-20260929_113423-6bvobnzi/files/checkpoint_final.pt

// LOOKS complete, is worthless -- trained with the swapped targets
HDMI/outputs/2026-09-28/19-11-47-G1WallPainting2-ppo_roa/   // 23 ckpts + final

pipeline/vidoes/wall_painting2_policy_02_fixed.mp4    // working policy, one-handed
pipeline/vidoes/wall_painting2_policy_01_broken.mp4   // swapped targets, roller loose
```

**Open, in order**

Make the contact reward multiplicative and retrain, to find out whether the one-handed grip is reward shaping or a physical limit of `eef_box` palms on a 0.503 m two-handed span with the roller's centre of mass (local z +0.05) above both hands.

Then continue from `checkpoint_final.pt` with `total_frames=300_000_000`, since the last run had not plateaued.

The z-sweep is premature until the policy holds the roller for a meaningful fraction of the 342-frame clip. At 48.9 steps it holds for about a second, so a sweep now would mostly measure how quickly it drops the roller at each height.

*Correction, 2026-09-30:* "48.9 steps = about a second of the clip" misreads the metric. Training episodes start at a random frame, so `episode_len` is not how far into the clip the robot gets. See the 2026-09-30 run log.

**Run log — 2026-09-29 continued: does more training fix the one-handed grip?**

No. Warm-started from the corrected run's `checkpoint_final.pt` for another 150M frames (`checkpoint_path` loads weights only; `total_frames` is a fresh budget, so 150M means 1144 more iterations, not a continuation of the old count).

| | run 2 corrected | run 3 continued |
| --- | --- | --- |
| episode_len | 48.9 | 60.7 |
| success | 0.123 | 0.207 |
| contact/step | 0.230 | 0.384 |
| wall time | 57.8 min | 64.6 min |

It converged. Over the final 320 iterations episode_len moved 60.0 -> 60.7, success 0.202 -> 0.207, contact 0.375 -> 0.384. Almost all the gain landed in the first 200 iterations, which is the warm start settling rather than new learning. The previous run's curve was still bending upward when it stopped; this one is flat.

The video (`pipeline/vidoes/wall_painting2_policy_03_cont.mp4`) shows the same grip as before: one hand on the shaft, the other bent at the waist, roller dropped partway through. Longer hold, same strategy.

**Why 0.384 is the tell.** `eef_contact_all` averages over the two end-effectors, so a *perfect* one-handed grip caps at 0.5. Converging at 0.384 means the policy is near the ceiling of the wrong strategy, not partway to the right one. Half the contact reward is cheap and the second hand is never worth reaching for.

**The fix being tested: make the contact reward multiplicative.** Two new classes in `rewards.py`, both additive so nothing existing changed.

`eef_contact_exp_both` — identical to `eef_contact_exp` except the per-eef terms are multiplied rather than averaged, so a one-handed grip scores zero. Out-of-contact-phase end-effectors contribute 1.0, the identity for a product, so gating is unchanged. Peak is still `gain`, so the 0.5 weight carries over.

`eef_contact_both` — binary both-hands-on indicator for reading as a stat, since `eef_contact_all` reports 0.5 for a one-handed grip and this reports 0.

```jsx
// cfg/task/G1/hdmi/wall_painting2_both.yaml
reward:
  object_tracking:
    eef_contact_exp: {weight: 0.0, enabled: false}
    eef_contact_exp_both: {weight: 0.5, gain: 5.0, pos_sigma: 0.3, frc_sigma: 40.0, frc_thres: 10.0}
  debug:
    eef_contact_both: {weight: 1.0, enabled: false}   // /episode_len = per-step both-hands fraction
```

**Warm-start from the one-handed policy, do not train from scratch.** A product reward gives an untrained policy near-zero signal everywhere — there is no gradient toward "get the first hand on" the way the mean provides. Starting from a policy that already places one hand gives the product something to build on.

```jsx
export CKPT=$PWD/outputs/2026-09-29/14-37-27-G1WallPainting2-ppo_roa/wandb/run-*/files/checkpoint_final.pt
$PY_HDMI scripts/train.py algo=ppo_roa_train task=G1/hdmi/wall_painting2_both \
  checkpoint_path=$CKPT save_interval=50
```

Baselines to beat: episode_len 60.7, success 0.207, contact/step 0.384.

**If the product reward fails**, the read is that an open `eef_box` palm cannot hold a 0.503 m two-handed span with the roller's centre of mass (local z +0.05) above both grips. That points at the asset rather than the reward: the `eef_L` hook shovel_dirt uses, or a lighter roller. Worth deciding before spending more runs on shaping.

**Two practical notes on `render.py`**

The process hangs in teardown after writing the video — observed sitting for over an hour with the GPU already released. The mp4 is complete and readable well before the process exits, so **check the file, not the process**:

```jsx
ffprobe -v error -show_entries format=duration -of csv=p=0 recording-<stamp>.mp4
// prints the duration once the moov atom is written; "moov atom not found" means still writing
```

Output is 2560x1440, 1000 frames, ~420 MB per render. Compress before keeping or sharing:

```jsx
ffmpeg -i recording-<stamp>.mp4 -vf scale=1280:-2 -c:v libx264 -crf 26 \
  -pix_fmt yuv420p -movflags +faststart out.mp4    // ~420 MB -> ~18 MB
```

**Run log — 2026-09-29 final: the product reward is what unlocked it**

Five runs in one day. Two changes produced all of the improvement, and neither was more training.

| | run 1 swapped | run 2 fixed | run 3 cont. | run 4 product | run 5 cont. |
| --- | --- | --- | --- | --- | --- |
| episode_len | 25.0 flat | 48.9 | 60.7 | 85.5 | **111.1** |
| success | 0.000 | 0.123 | 0.207 | 0.362 | **0.527** |
| both-hands contact | — | — | 0.190 | 0.447 | **0.546** |
| lost-contact term. | 100% | 73.9% | 73.9% | 53.6% | **35.3%** |

*Correction, 2026-09-30:* this table previously had a "% of the 342-frame clip" row computed as `episode_len` / 342. That is wrong: training episodes start at a random frame, so `episode_len` measures time from a random start, not progress through the clip. Full-stroke performance has to be measured in evaluation, which starts at frame 0 -- see the 2026-09-30 run log.

Run 3 (more frames on the mean reward) converged: its last 320 iterations moved episode_len by 0.7. Runs 4 and 5 (product reward) were still climbing when each stopped, run 5 steeper at the end than the middle. The ceiling has not been found.

**What each change bought.** Fixing the swapped contact targets took the task from not-learning to learning at all. Switching `eef_contact_exp` from a mean over the two hands to a product took it from a one-handed local optimum to a genuine two-handed grip. Frames alone, on the mean reward, plateaued.

**A metric trap worth repeating.** `eef_contact_all` is a mean over the two end-effectors, so a perfect one-handed grip reads 0.5 and looks like "halfway there". It is not — it is the ceiling of the wrong strategy. `eef_contact_both` (added in `rewards.py`) is the binary both-hands indicator and reads 0 for a one-handed grip. Use that one when the question is whether the robot has actually got two hands on the tool.

**Reading the stills is not reading the behaviour.** Twelve frames off a contact sheet led to two wrong calls: first that the second hand never engages (it was on ~20% of steps), then that run 4 looked no better than run 3 (both-hands contact had more than doubled). With episodes ending at ~85 frames, a 1000-frame render holds about twelve episodes, so a sheet sampled across it mixes good starts with post-drop frames. Watch the video, or read `eef_contact_both`.

**Stage 9 addition: a canvas in the scene**

Renders showed the robot painting empty air, because nothing in the scene represented the wall. There is now a canvas prim.

```jsx
$PY_HDMI $SB/HDMI/pipeline/scripts/make_canvas_usd.py
// authors a wall standing on the ground: 2.0 m wide, tall enough to cover the
// highest randomized target plus headroom -> 2.0 x 2.349 m for wall_painting2
// --width, --max-z-offset, --headroom to change it
```

Registered as `"canvas"` in the OBJECTS registry and spawned automatically by `locomotion.py` for any task that sets `target_region_path`, so every painting render gets it with no config change. `show_canvas: false` in the command config turns it off.

**It has no collision on purpose.** The roller passes through, so the physics is identical to a run without it and every result above stays comparable. Giving it collision is a real experiment rather than a rendering change: the human in the demo was pressing against a wall while the robot pushes against nothing, so a surface to react against might help exactly the contact-loss failure that still caps these runs. Worth doing against a converged baseline, not layered onto a policy that is still improving.

The wall is fixed and the target moves on it. A real wall does not slide when the job does, so `_place_canvas()` stands it on the ground under the nominal target: the prim origin is authored at the wall's bottom edge, so grounding it is just placing the origin at z = 0, with no height bookkeeping in the command. Target randomization moves the painted region across this static wall. The rotation whose columns are `[axis_u, axis_v, normal]` maps the plate's local X/Y/Z onto the wall's horizontal, vertical and outward directions; verified orthonormal with det +1, and all four target corners land in the wall plane at |z| = 6e-17.

(An earlier version sized the canvas to the target rectangle plus 0.15 m and re-placed it on the target every reset, so it floated 0.65 m above the floor and would have slid with the target. Replaced 2026-09-29.)

**Debug overlay, GUI only**

`_debug_draw_paint()` outlines the target rectangle in amber and dots every painted cell in blue, subsampled 4x and limited to the first 4 envs. It only runs when `sim.has_gui()` is true (`envs/base.py`), so it appears in an interactive `play.py` and **never in a headless render**. That is why the canvas had to be real geometry.

```jsx
$PY_HDMI scripts/play.py algo=ppo_roa_train task=G1/hdmi/wall_painting2_both \
  checkpoint_path=<ckpt> headless=false task.num_envs=4
```

If the blue dots do not land under the roller head and inside the amber rectangle, the wall fit or the hardcoded head geometry is wrong and every coverage number is measuring the wrong surface.

**Multi-motion support is in and is backward compatible**

Both changes the motion set needs are implemented. `object_contact` concatenates across `motion_paths` in dataset order, with an assert against `dataset.ends[-1]`. `target_region_path` accepts a string or a list; rectangles load as `[num_motions, ...]` and `_sample_target_region` indexes them by `motion_ids`, so each env paints the rectangle of the clip it drew. The raster is sized to the largest rectangle across motions, and `target_region_size` normalizes per-motion.

Single-clip behaviour is unchanged, verified exactly: per-env width, height, centre and the `target_region_size` observation all `max|old-new| = 0`, raster grid identical at (151, 214), and `np.concatenate([x])` array-equal to `x`. A single-clip replay loads clean.

`axis_u`/`axis_v` stay shared across motions rather than per-motion, because `axis_v` is `[0,0,1]` by construction and `axis_u` is the in-plane horizontal, so clips shot against the same wall from the same standoff agree. Rectangles that disagree by more than 1e-3 raise at load with an explicit message. **Same wall, same standoff is now enforced, not just advised.**

**Artefacts**

```jsx
// best policy of the day
HDMI/outputs/2026-09-29/19-09-19-G1WallPainting2Both-ppo_roa/
  wandb/run-20260929_190923-ajlaed22/files/checkpoint_final.pt

pipeline/vidoes/wall_painting2_policy_05_canvas.mp4   // run 4 policy, floating canvas
pipeline/vidoes/wall_painting2_policy_04_both.mp4     // run 4 policy, no canvas
pipeline/vidoes/wall_painting2_policy_03_cont.mp4     // run 3
pipeline/vidoes/wall_painting2_policy_02_fixed.mp4    // run 2
pipeline/vidoes/wall_painting2_policy_01_broken.mp4   // swapped targets, roller loose
```

**Open**

Stage 5 has never been run for painting. Every checkpoint so far is a teacher; the student distillation step (`algo=ppo_roa_finetune`) is what shovel_dirt did and what deployment needs.

The canvas-with-collision experiment, against a converged baseline.

Stage 10, once the policy holds the roller for a meaningful fraction of the clip. At 111 of 342 frames it is closer than it was, but a z-sweep still partly measures drop rate rather than painting quality.

**Run log — 2026-09-30: converged, measured properly, and the baseline**

**A metric misread, corrected.** During training every episode starts at a random frame of the clip (`start_t = rand() * (motion_len - 32)` in `_sample_motions`), and `success` means reaching the end of the clip from wherever it started (`t >= motion_len - 1`). So a training-time `episode_len` of 155 is not "155 of 342 frames into the stroke", and a training `success` of 0.795 means finishing from a random start -- about half the clip on average -- not finishing the whole stroke. Evaluation (`play.py`, `render.py`, the sweep) starts every episode at frame 0 (`if not self.env.training: start_t.fill_(0)`), so only evaluation measures the full stroke. Earlier sections that divided `episode_len` by 342 are marked as corrected where they appear.

**The overnight run converged.** Warm-started from run 5, 450M frames, 3433 iterations, 3.17 h at 43.5k fps, and this time it genuinely plateaued: episode_len by quarter 134.6 -> 147.3 -> 152.3 -> 155.2, and the last eight samples inside a 1.5-step band.

| overnight policy | training (random start) | evaluation (frame 0, target pinned at z=0) |
| --- | --- | --- |
| success | 0.795 | **0.758** -- completes the full 342-frame stroke |
| coverage | -- | **0.742** -- the replayed demo gets 0.73; 0.922 over the episodes that finish |
| lost-contact terminations | 12.6% | 13% |

*Correction, 2026-09-30 evening:* this table and the sweep below first read success 0.683 and coverage 0.690. Those came from averaging `play.py`'s stats blocks, which undercounts long successful episodes; see "the sweep was biased" in the evening run log. Re-measured one episode per env.

```jsx
// best fixed-target policy
HDMI/outputs/2026-09-29/23-25-54-G1WallPainting2Both-ppo_roa/
  wandb/run-20260929_232558-nvq3kptp/files/checkpoint_final.pt
```

**The baseline sweep: the fixed-target policy ignores the target.** 128 episodes per offset, one per env, from frame 0, target pinned at each vertical offset. `cov|success` is coverage over the episodes that finish the stroke:

| z offset | coverage | vs z=0 | overlap-only prediction | success | cov\|success |
| --- | --- | --- | --- | --- | --- |
| -0.50 | 0.350 | 47% | 0.307 | 0.73 | 0.440 |
| -0.40 | 0.459 | 62% | 0.394 | 0.75 | 0.563 |
| -0.30 | 0.550 | 74% | 0.481 | 0.75 | 0.685 |
| -0.20 | 0.649 | 87% | 0.568 | 0.79 | 0.789 |
| -0.10 | 0.662 | 89% | 0.655 | 0.72 | 0.878 |
| 0.00 | 0.742 | 100% | 0.742 | 0.76 | 0.922 |
| +0.10 | 0.661 | 89% | 0.655 | 0.73 | 0.862 |
| +0.20 | 0.612 | 82% | 0.568 | 0.78 | 0.762 |
| +0.30 | 0.458 | 62% | 0.481 | 0.68 | 0.646 |
| +0.40 | 0.390 | 53% | 0.394 | 0.71 | 0.531 |
| +0.50 | 0.306 | 41% | 0.307 | 0.72 | 0.414 |

The overlap-only column is what a policy that paints the same spot every time would score: z=0 coverage times the fraction of the moved 0.853 m target that still overlaps the painted region. Measured coverage tracks it to within 0.95-1.16x across the whole range, and `cov|success` tracks the same line drawn from its own z=0 value at 1.06-1.15x. Success is flat. The robot drops the roller at the same rate wherever the target is, and coverage falls only because the target walks away from where it paints. A ">=80% of z=0 coverage over +-0.2 m" band is overlap, not generalization.

That makes the goal-conditioned comparison sharp: overlap predicts 0.31 at +-0.5 m, so a policy that follows the target has to hold clearly above that at the edges.

*Correction, 2026-09-30 evening:* this table first came from the biased block-averaging sweep (z=0: coverage 0.690, success 0.68; success 0.60-0.74 overall). The shape and the conclusion are unchanged. The old logs are in `zsweep_blocks_biased/` next to the checkpoint.

```jsx
HDMI/outputs/2026-09-29/23-25-54-G1WallPainting2Both-ppo_roa/wandb/run-*/files/zsweep/results.json
```

**Three ways the sweep failed before it worked.** Worth knowing because each one returned a plausible-looking empty table rather than an error.

`algo=ppo_roa` does not exist. The algo configs come from a structured config store, not `cfg/algo/*.yaml`, so they are not visible by listing that directory; the valid name is `ppo_roa_train`. Hydra failed instantly and every offset parsed as None.

A checkpoint only loads into a task with the same observation space. Sweeping the fixed-target policy on `wall_painting2_zsweep` failed because that config adds `target_center_b` (object obs 30 -> 33) and vecnorm refused to load. The sweep now takes `--task` and defaults to `wall_painting2_both`; the z offset is applied by command-line override, so sweep the task the checkpoint was trained on.

`play.py` never exits. Its rollout is `for i in itertools.count()`, printing a stats block each time `num_envs` episodes finish, forever. The first working sweep streamed `play.py`'s output, stopped after a few stats blocks and killed the process group. That worked, and it was wrong -- see the next run log. The sweep no longer uses `play.py` at all.

**Goal-conditioned training started.** `wall_painting_goal`, from scratch, 450M frames, launched 11:50 at ~38.7k fps (finishes ~15:05). At iteration 172 (10 min): episode_len 27.1, lost-contact 0.908, success 0.002 -- off the 25-step floor and essentially on the same curve as run 2 at the same point (26.9, 0.918, 0.001), the only other from-scratch run. Both-hands contact is 0.010, which matters more here than it did for run 2: the product contact reward pays only when both hands are on, so near-zero both-hands means the contact reward is contributing almost nothing yet. If episode_len keeps rising while both-hands stays near 0.01 by iteration ~600, the policy is learning to survive one-handed and the fallback is two-stage -- mean contact reward first to establish the grip, then product.

```jsx
HDMI/outputs/2026-09-30/11-50-08-G1WallPaintingGoal-ppo_roa/
```

**Housekeeping**

`pipeline/` now lives only at `HDMI/pipeline/`, in the fork `github.com/wkdalswns0427/HDMI` on branch `wall-painting`. Scripts locate themselves through `hdmi_pipeline/paths.py`; set `SIMBENCH_ROOT` and `CONDA_ROOT` on a machine with a different layout. `target_region_path` in the task configs is repo-relative.

Policy videos are numbered in the order they were rendered, in `pipeline/vidoes/`: `_01_broken`, `_02_fixed`, `_03_cont`, `_04_both`, `_05_canvas`, `_06_wall`, `_07_converged`.

**Open**

The goal-conditioned result: run the same sweep on its checkpoint with `--task wall_painting_goal` and compare against the table above.

Stage 5, student distillation, has never been run for painting.

Canvas with collision, against the converged fixed-target baseline.

The bend and squat clips. Whether they are needed depends on the goal-conditioned result: if it holds coverage to +-0.3 m or so, one standing clip already reaches most of what they would add.

**Run log — 2026-09-30 evening: the goal-conditioned result, and a biased sweep**

**Training.** `wall_painting_goal` from scratch, 450M frames, 3424 iterations, finished 15:17. Slower off the mark than run 2, the only other from-scratch run: at iteration 1140 it had episode_len 34.0, success 0.029, lost-contact 0.83 against run 2's 47.7, 0.114, 0.74. Both-hands contact sat near 0.02 per step until iteration ~1200 -- the product reward's cold start -- then climbed to 0.36. It plateaued from about iteration 2500 (episode_len 57.5 -> 58.9 -> 58.8, success 0.193 -> 0.200 -> 0.201), roughly where run 3 ended (60.5, 0.206). These are random-start training figures.

```jsx
// goal-conditioned policy
HDMI/outputs/2026-09-30/11-50-08-G1WallPaintingGoal-ppo_roa/
  wandb/run-20260930_115012-1mfissu4/files/checkpoint_final.pt
```

**The sweep was biased, and it mattered.** `play.py` prints the mean of the first `num_envs` episodes to finish, and the sweep stopped after four such blocks. Short failed episodes finish, restart at frame 0 and finish again many times before a long successful one finishes once, so stopping after 128 finished episodes drops exactly the long ones. On the goal-conditioned policy it read success 0.000 and episode_len 28 at every offset; the true figures at z=0 are 0.27 and 115. On the fixed-target policy, whose failures are rarer, it understated z=0 success 0.758 as 0.683 and coverage 0.742 as 0.690. The earlier tables in this log are corrected in place.

The sweep now runs `pipeline/scripts/eval_episodes.py`, which is HDMI's own `scripts.helpers.evaluate` underneath: every env runs one complete episode from frame 0 and only that first episode counts, so 128 envs is 128 unbiased episodes. It exits on its own, so the streaming and block counting are gone. It also reports `cov|success`, coverage over the episodes that finish the stroke, which separates painting quality from drop rate. Do not quote `play.py`'s printed stats as a measurement for any policy whose episode lengths vary.

**Result.** 128 episodes per offset, target pinned. Standard error on `cov|success` is 0.003-0.010 everywhere, on coverage 0.012-0.031.

| z offset | fixed-target coverage | success | cov\|success | goal-conditioned coverage | success | cov\|success |
| --- | --- | --- | --- | --- | --- | --- |
| -0.50 | 0.350 | 0.73 | 0.440 | 0.321 | 0.27 | **0.639** |
| -0.40 | 0.459 | 0.75 | 0.563 | 0.327 | 0.27 | **0.717** |
| -0.30 | 0.550 | 0.75 | 0.685 | 0.345 | 0.28 | **0.800** |
| -0.20 | 0.649 | 0.79 | 0.789 | 0.368 | 0.30 | **0.872** |
| -0.10 | 0.662 | 0.72 | 0.878 | 0.360 | 0.29 | 0.888 |
| 0.00 | 0.742 | 0.76 | 0.922 | 0.301 | 0.27 | 0.817 |
| +0.10 | 0.661 | 0.73 | 0.862 | 0.264 | 0.28 | 0.718 |
| +0.20 | 0.612 | 0.78 | 0.762 | 0.209 | 0.28 | 0.610 |
| +0.30 | 0.458 | 0.68 | 0.646 | 0.166 | 0.30 | 0.510 |
| +0.40 | 0.390 | 0.71 | 0.531 | 0.116 | 0.27 | 0.393 |
| +0.50 | 0.306 | 0.72 | 0.414 | 0.091 | 0.28 | 0.293 |

```jsx
HDMI/outputs/2026-09-30/11-50-08-G1WallPaintingGoal-ppo_roa/wandb/run-*/files/zsweep/results.json
// per-episode coverage and success are in there too
```

**It drops the roller.** The goal-conditioned policy finishes the stroke in 27-30% of episodes at every offset, the fixed-target one in 68-79%. 62% of its episodes end on lost contact against 13%, and at z=0 it has both hands on the roller 23% of steps against 62%. The rate is flat across offsets, so this is grip, not the target. It is why its mean coverage is below the baseline at every offset, and by the test written down beforehand -- mean coverage clearly above the baseline at the edges -- it fails.

**When it holds on, it follows the target down.** At -0.5 m it covers 0.639 against the baseline's 0.440, and against the 0.338 it would score by painting its own z=0 spot (1.89x). The margin over the baseline grows with the offset: +0.08, +0.11, +0.15, +0.20 at -0.2, -0.3, -0.4, -0.5. The fixed-target policy tracks its overlap-only line at 1.06-1.15x in the same column. This is the first measurement in this pipeline of a policy painting a region its demonstration did not.

**It does not follow up, and it paints lower than the demo.** Above z=0 it tracks its own overlap-only line (0.87-1.00x) and sits 0.10-0.15 below the baseline at every offset, including 0.817 against 0.922 at z=0. Its painted band is lower than the demonstrated stroke: its best offset is -0.1, and 0.293 at +0.5 is 0.25 m of the target, so it paints up to about 1.55 m where the demo reaches 1.649 m. The baseline's 0.414 at +0.5 is exactly the demo's top. The demonstrated stroke is near the top of standing reach, so following targets above it was never on offer, but a goal-conditioned policy should at least reach the demo's top when the target is high, and this one does not. Whether that is the same weak grip -- stretching up is where it would fail -- is not measured.

**What it means.** The effect the thesis needs exists and is measurable, in the reachable direction, and it is hidden in mean coverage by a grip that fails three times as often. The drop rate is the binding problem. The fixed-target policy got its grip from 1.05B frames and a mean-then-product contact curriculum; this one has 450M with the product reward from a cold start.

**Open**

More training on the goal-conditioned policy from its own `checkpoint_final.pt`, which keeps it a from-scratch lineage. Success from frame 0 is the number to move, toward the baseline's ~0.75, while `cov|success` at -0.5 holds near 0.64 and z=0 recovers toward 0.92. Then the same sweep.

```jsx
$PY_HDMI scripts/train.py algo=ppo_roa_train task=G1/hdmi/wall_painting_goal \
  checkpoint_path=<goal checkpoint_final.pt> total_frames=450_000_000 save_interval=50
```

If it stays on the plateau, the two-stage fallback from scratch: mean contact reward to establish the grip, then product.

The bend and squat clips extend downward reach. This result says downward is where one standing clip already follows the target, to about -0.5 m when it holds on.

Stage 5 distillation and canvas collision, as before.
