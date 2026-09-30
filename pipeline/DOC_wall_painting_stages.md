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
    export WORK=$SB/pipeline/work/$TASK
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
    PYTHONNOUSERSITE=1 $PY_GVHMR $SB/pipeline/scripts/render_smpl.py \
      "outputs/demo/wall_painting2/hmr4d_results.pt" \
      -o $SB/pipeline/wall_painting2_human.mp4
    ```

- **Stage 2: Retarget joints**

    ```jsx
    cd $SB/pipeline
    PYTHONNOUSERSITE=1 $PY_GMR scripts/gvhmr_to_gmr.py \
      --gvhmr_pred "$PRED" --robot $ROBOT --video "$CLIP" \
      --save_path "$WORK/$TASK.pkl"
    ```

- **Stage 3: Forward Kinematic solve validate**

    ```jsx
    PYTHONNOUSERSITE=1 $PY_GMR $SB/pipeline/scripts/gmr_to_hdmi.py \
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
    $PY_HDMI $SB/pipeline/scripts/make_roller_usd.py
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
      --robot unitree_g1 -o $SB/pipeline/wall_painting2_gmr.mp4

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
    PYTHONNOUSERSITE=1 $PY_GMR $SB/pipeline/task_info/scripts/visualize_roller_swept_area.py
    // re-derives the two_hand roller pose from both wrist links and writes the
    // head centreline endpoints per frame -> roller_head_trajectory.npz  (342 frames)

    PYTHONNOUSERSITE=1 $PY_GMR $SB/pipeline/task_info/scripts/fit_roller_plane.py
    // SVD plane through 25 samples per roller line, normal_z forced to 0 (vertical
    // wall), then pushed one roller radius AWAY from the human so it is the wall
    // surface rather than the roller axis -> roller_fitted_plane.npz
    //   rmse 0.0307 m   p95 0.0600 m   normal [0.8787, 0.4774, 0]

    PYTHONNOUSERSITE=1 $PY_GMR $SB/pipeline/task_info/scripts/target_area.py
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

**Move the goal.** `target_region_pos_range.z` and `target_region_scale_range.height` are already config values pinned at zero and 1.0. Unpinning them moves the target while the reference stays put — goal-conditioned training for the price of a config edit. `cfg/task/G1/hdmi/wall_painting2_zsweep.yaml` does this with z ±0.15 m and height 0.80–1.25, and switches on `target_center_b`, which the policy needs once the target is no longer where the demo left it.

```jsx
$PY_HDMI scripts/train.py algo=ppo_roa_train task=G1/hdmi/wall_painting2_zsweep \
  save_interval=50

python pipeline/scripts/zsweep_eval.py --checkpoint <ckpt> \
  --offsets -0.20 -0.15 -0.10 -0.05 0.0 0.05 0.10 0.15 0.20
// pins the target at each offset, reads stats/debug/paint_coverage,
// reports the band where coverage holds >=80% of the pinned baseline
```

**Add postures.** Film the same task with knees bent ~45° and in a squat, retarget each, and train on all three as a motion set. Projected bands: stand 0.80–1.65 (measured), bend ~0.58–1.43, squat ~0.44–1.29, union 0.44–1.65 — 42% more wall for two more shoots. Needs two code changes first, both described in `DOC_multi_demo_same_task.md`: concatenating `object_contact` across motions (four lines, `command.py:619`) and making `target_region_path` per-motion so coverage 1.0 stays reachable in every episode.

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

**It holds the roller one-handed.** Visible in `pipeline/wall_painting2_policy.mp4`: one hand on the shaft, the other parked on the hip for most of the clip. This caps the contact metric at 0.5, because `eef_contact_all` is a mean over the two end-effectors — the final 0.230 is roughly one hand engaged half the time. The policy found a local optimum where half the contact reward is cheap to collect. Worth trying before more frames: make the contact reward multiplicative across hands instead of a mean, so half credit stops being available.

**The rasterizer was 19.6x slower than it needed to be.** Covered in Stage 9 above; the numbers came from here. Batched, it was verified bitwise-identical to the per-env version offline (40 steps x 24 envs, mixed rectangle sizes, zero difference in mask, ratio and grid) and across four live replays whose final coverage values (0.7307, 0.7378, 0.7364, 0.7367) are indistinguishable between versions.

**Artefacts from this run**

```jsx
// the only checkpoint worth using -- corrected contact targets
HDMI/outputs/2026-09-29/11-34-19-G1WallPainting2-ppo_roa/
  wandb/run-20260929_113423-6bvobnzi/files/checkpoint_final.pt

// LOOKS complete, is worthless -- trained with the swapped targets
HDMI/outputs/2026-09-28/19-11-47-G1WallPainting2-ppo_roa/   // 23 ckpts + final

pipeline/wall_painting2_policy.mp4          // working policy, one-handed
pipeline/wall_painting2_policy_broken.mp4   // swapped targets, roller loose
```

**Open, in order**

Make the contact reward multiplicative and retrain, to find out whether the one-handed grip is reward shaping or a physical limit of `eef_box` palms on a 0.503 m two-handed span with the roller's centre of mass (local z +0.05) above both hands.

Then continue from `checkpoint_final.pt` with `total_frames=300_000_000`, since the last run had not plateaued.

The z-sweep is premature until the policy holds the roller for a meaningful fraction of the 342-frame clip. At 48.9 steps it holds for about a second, so a sweep now would mostly measure how quickly it drops the roller at each height.

**Run log — 2026-09-29 continued: does more training fix the one-handed grip?**

No. Warm-started from the corrected run's `checkpoint_final.pt` for another 150M frames (`checkpoint_path` loads weights only; `total_frames` is a fresh budget, so 150M means 1144 more iterations, not a continuation of the old count).

| | run 2 corrected | run 3 continued |
| --- | --- | --- |
| episode_len | 48.9 | 60.7 |
| success | 0.123 | 0.207 |
| contact/step | 0.230 | 0.384 |
| wall time | 57.8 min | 64.6 min |

It converged. Over the final 320 iterations episode_len moved 60.0 -> 60.7, success 0.202 -> 0.207, contact 0.375 -> 0.384. Almost all the gain landed in the first 200 iterations, which is the warm start settling rather than new learning. The previous run's curve was still bending upward when it stopped; this one is flat.

The video (`pipeline/wall_painting2_policy_cont.mp4`) shows the same grip as before: one hand on the shaft, the other bent at the waist, roller dropped partway through. Longer hold, same strategy.

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
| % of the 342-frame clip | 7% | 14% | 18% | 25% | **32%** |

Run 3 (more frames on the mean reward) converged: its last 320 iterations moved episode_len by 0.7. Runs 4 and 5 (product reward) were still climbing when each stopped, run 5 steeper at the end than the middle. The ceiling has not been found.

**What each change bought.** Fixing the swapped contact targets took the task from not-learning to learning at all. Switching `eef_contact_exp` from a mean over the two hands to a product took it from a one-handed local optimum to a genuine two-handed grip. Frames alone, on the mean reward, plateaued.

**A metric trap worth repeating.** `eef_contact_all` is a mean over the two end-effectors, so a perfect one-handed grip reads 0.5 and looks like "halfway there". It is not — it is the ceiling of the wrong strategy. `eef_contact_both` (added in `rewards.py`) is the binary both-hands indicator and reads 0 for a one-handed grip. Use that one when the question is whether the robot has actually got two hands on the tool.

**Reading the stills is not reading the behaviour.** Twelve frames off a contact sheet led to two wrong calls: first that the second hand never engages (it was on ~20% of steps), then that run 4 looked no better than run 3 (both-hands contact had more than doubled). With episodes ending at ~85 frames, a 1000-frame render holds about twelve episodes, so a sheet sampled across it mixes good starts with post-drop frames. Watch the video, or read `eef_contact_both`.

**Stage 9 addition: a canvas in the scene**

Renders showed the robot painting empty air, because nothing in the scene represented the wall. There is now a canvas prim.

```jsx
$PY_HDMI $SB/pipeline/scripts/make_canvas_usd.py
// reads the task's paint_target_rectangle.npz, authors a plate of that size
// plus a margin -> 0.9025 x 1.1529 m for wall_painting2 (0.15 m each side)
```

Registered as `"canvas"` in the OBJECTS registry and spawned automatically by `locomotion.py` for any task that sets `target_region_path`, so every painting render gets it with no config change. `show_canvas: false` in the command config turns it off.

**It has no collision on purpose.** The roller passes through, so the physics is identical to a run without it and every result above stays comparable. Giving it collision is a real experiment rather than a rendering change: the human in the demo was pressing against a wall while the robot pushes against nothing, so a surface to react against might help exactly the contact-loss failure that still caps these runs. Worth doing against a converged baseline, not layered onto a policy that is still improving.

`_place_canvas()` sits it on the wall plane at each env's target centre, re-placed on reset so it follows target randomization in the z-sweep. The rotation whose columns are `[axis_u, axis_v, normal]` maps the plate's local X/Y/Z onto the wall's horizontal, vertical and outward directions; verified orthonormal with det +1, and all four target corners land in the canvas plane at |z| = 6e-17.

Known cosmetic issue: the canvas spans only the target rectangle plus margin, so its bottom edge sits at z 0.646 m and it floats above the floor. A larger vertical margin fixes it.

**Debug overlay, GUI only**

`_debug_draw_paint()` outlines the target rectangle in amber and dots every painted cell in blue, subsampled 4x and limited to the first 4 envs. It only runs when `sim.has_gui()` is true (`envs/base.py`), so it appears in an interactive `play.py` and **never in a headless render**. That is why the canvas had to be real geometry.

```jsx
$PY_HDMI scripts/play.py algo=ppo_roa_train task=G1/hdmi/wall_painting2_both \
  checkpoint_path=<ckpt> headless=false task.num_envs=4
```

If the blue dots do not land under the roller head and inside the amber rectangle, the wall fit or the hardcoded head geometry is wrong and every coverage number is measuring the wrong surface.

**Multi-motion support is in and is backward compatible**

Both changes from `DOC_multi_demo_same_task.md` are implemented. `object_contact` concatenates across `motion_paths` in dataset order, with an assert against `dataset.ends[-1]`. `target_region_path` accepts a string or a list; rectangles load as `[num_motions, ...]` and `_sample_target_region` indexes them by `motion_ids`, so each env paints the rectangle of the clip it drew. The raster is sized to the largest rectangle across motions, and `target_region_size` normalizes per-motion.

Single-clip behaviour is unchanged, verified exactly: per-env width, height, centre and the `target_region_size` observation all `max|old-new| = 0`, raster grid identical at (151, 214), and `np.concatenate([x])` array-equal to `x`. A single-clip replay loads clean.

`axis_u`/`axis_v` stay shared across motions rather than per-motion, because `axis_v` is `[0,0,1]` by construction and `axis_u` is the in-plane horizontal, so clips shot against the same wall from the same standoff agree. Rectangles that disagree by more than 1e-3 raise at load with an explicit message. **Same wall, same standoff is now enforced, not just advised.**

**Artefacts**

```jsx
// best policy of the day
HDMI/outputs/2026-09-29/19-09-19-G1WallPainting2Both-ppo_roa/
  wandb/run-20260929_190923-ajlaed22/files/checkpoint_final.pt

pipeline/wall_painting2_policy_canvas.mp4   // run 4 policy, canvas in scene
pipeline/wall_painting2_policy_both.mp4     // run 4 policy, no canvas
pipeline/wall_painting2_policy_cont.mp4     // run 3
pipeline/wall_painting2_policy.mp4          // run 2
pipeline/wall_painting2_policy_broken.mp4   // swapped targets, roller loose
```

**Open**

Stage 5 has never been run for painting. Every checkpoint so far is a teacher; the student distillation step (`algo=ppo_roa_finetune`) is what shovel_dirt did and what deployment needs.

The canvas-with-collision experiment, against a converged baseline.

Stage 10, once the policy holds the roller for a meaningful fraction of the clip. At 111 of 342 frames it is closer than it was, but a z-sweep still partly measures drop rate rather than painting quality.
