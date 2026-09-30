# HDMI pipeline — usage manual

Turns one monocular RGB video of a person interacting with an object into the
`motion.npz` + `meta.json` pair that [HDMI](https://github.com/LeCAR-Lab/HDMI)
trains on, plus the Hydra task config.

```
video.mp4
   │  [1] GVHMR              env: gvhmr   → hmr4d_results.pt  (SMPL-X, world-grounded)
   │  [2] GMR retarget       env: gmr     → <task>.pkl        (root pose + joint angles)
   │  [3] MuJoCo FK          env: gmr     → link states + velocities
   │  [4] object annotation  MANUAL       → annotations/<task>.json
   │  [5] pack               env: gmr     → motion.npz + meta.json
   │  [6] task config        env: gmr     → HDMI/cfg/task/<R>/hdmi/<task>.yaml
   ▼
   [7] train                 env: hdmi    → teacher / student policy
```

Supported robots: `unitree_g1` (29 dof), `unitree_h1_2` (27 dof).

---

# 1. Setup

One-time. Both scripts are idempotent — safe to re-run.

```bash
cd ~/mj_ws/simbench/pipeline

./setup_gvhmr_env.sh            # builds the `gvhmr` conda env (~30 min: pytorch3d compiles)
./fetch_gvhmr_checkpoints.sh    # ~5 GB of weights
./setup_locomujoco_env.sh       # OPTIONAL — alternative retargeter, not used by default
```

Check it worked:

```bash
PYTHONNOUSERSITE=1 ~/miniconda3/envs/gvhmr/bin/python -c \
  "import torch,pytorch3d;print(torch.__version__, torch.cuda.get_device_name(0))"
# expect: 2.8.0+cu128 NVIDIA GeForce RTX 5090
```

## The `PYTHONNOUSERSITE` rule

`~/.local/lib/python3.10/site-packages` holds ~4 GB of packages that shadow
*every* Python 3.10 conda env on this machine. It cuts both ways:

- **`gvhmr` and `gmr` → always `PYTHONNOUSERSITE=1`.** Without it you silently
  get wrong versions (a half-broken `ultralytics` missing `yaml`, wrong `numpy`).
- **`hdmi` → never set it.** That env is *not* self-contained; it resolves
  `typing_extensions` from `~/.local` and the guard breaks `import
  active_adaptation`.

Every command below follows this rule. `run_pipeline.sh` never invokes the hdmi
interpreter, so the two never collide.

---

# 2. Shell pipeline

Two ways to run the whole thing. Both stop at stage 4, because the object
annotation is manual by design.

## 2a. The driver

```bash
cd ~/mj_ws/simbench/pipeline

./run_pipeline.sh /abs/path/clip.mp4 move_suitcase unitree_g1
# ... runs stages 1-3, then STOPS with an annotation template ...

nano annotations/move_suitcase.json

STAGE=post ./run_pipeline.sh /abs/path/clip.mp4 move_suitcase unitree_g1
```

Arguments: `<video> <task_name> [robot] [object_name]`. `object_name` defaults
to the last word of the task name (`move_suitcase` → `suitcase`).

Env switches:

| var | default | meaning |
|---|---|---|
| `STATIC_CAM` | `1` | passes `-s` to GVHMR (skip visual odometry). Set `0` for handheld/panning footage. |
| `STAGE` | `all` | `post` resumes at stage 5 after you edit the annotation. |

```bash
STATIC_CAM=0 ./run_pipeline.sh /abs/path/clip.mp4 my_task unitree_g1   # handheld
```

## 2b. The same thing as one paste-able block

Identical work, nothing hidden — use this when you want to see or edit each
command. Paste part 1, edit the JSON, paste part 2. Deliberately no `set -e`:
pasted into an interactive shell it would close your terminal on the first
failure. Read each stage's output before pasting the next part.

```bash
# ---------- config ----------
export SB=~/mj_ws/simbench
export PY_GVHMR=~/miniconda3/envs/gvhmr/bin/python
export PY_GMR=~/miniconda3/envs/gmr/bin/python
export PY_HDMI=~/miniconda3/envs/hdmi/bin/python

export CLIP=/abs/path/to/clip.mp4     # MUST be absolute
export TASK=push_box                  # snake_case
export ROBOT=unitree_g1
export OBJECT=box                     # must be a registry asset (see §6)
export NAME=PushBox                   # CamelCase

export MOTION=$SB/HDMI/data/motion/data_for_sim/$TASK
export WORK=$SB/pipeline/work/$TASK
export ANN=$SB/pipeline/annotations/$TASK.json
mkdir -p "$WORK"

# ---------- part 1: stages 1-4 ----------
# [1] video -> SMPL-X   (ends in a render_incam traceback; that is expected)
#     MUST run from the GVHMR dir: vitpose.py hardcodes a relative ckpt path.
( cd $SB/GVHMR && PYTHONNOUSERSITE=1 $PY_GVHMR tools/demo/demo.py --video="$CLIP" -s ) || true
export PRED=$SB/GVHMR/outputs/demo/$(basename "${CLIP%.*}")/hmr4d_results.pt
test -f "$PRED" && echo "OK: $PRED"

# [2] SMPL-X -> robot qpos   (--video detects true fps)
PYTHONNOUSERSITE=1 $PY_GMR $SB/pipeline/scripts/gvhmr_to_gmr.py \
  --gvhmr_pred "$PRED" --robot $ROBOT --video "$CLIP" \
  --save_path $WORK/$TASK.pkl

# [3] FK -> HDMI motion (robot only) + validate
PYTHONNOUSERSITE=1 $PY_GMR $SB/pipeline/scripts/gmr_to_hdmi.py \
  --gmr_pkl $WORK/$TASK.pkl --robot $ROBOT \
  --out_dir $MOTION --target_fps 50

PYTHONNOUSERSITE=1 $PY_GMR $SB/pipeline/scripts/validate_motion.py $MOTION \
  --reference $SB/HDMI/data/motion/data_for_sim/carry_and_place_bread_box-0829

# [4] annotation template
PYTHONNOUSERSITE=1 $PY_GMR $SB/pipeline/scripts/make_annotation.py $MOTION \
  --object $OBJECT --out "$ANN"

echo; echo ">>> Now edit $ANN against the video, then paste part 2."
```

```bash
# ---------- part 2: stages 5-7 ----------
# [5] repack with object + contact
PYTHONNOUSERSITE=1 $PY_GMR $SB/pipeline/scripts/gmr_to_hdmi.py \
  --gmr_pkl $WORK/$TASK.pkl --robot $ROBOT \
  --out_dir $MOTION --annotation "$ANN" --target_fps 50

PYTHONNOUSERSITE=1 $PY_GMR $SB/pipeline/scripts/validate_motion.py $MOTION

# [6] preview — object must be green exactly while held
PYTHONNOUSERSITE=1 $PY_GMR $SB/pipeline/scripts/render_motion.py $MOTION \
  --robot $ROBOT -o $WORK/preview.mp4

# [7] task config
PYTHONNOUSERSITE=1 $PY_GMR $SB/pipeline/scripts/make_task_cfg.py \
  --motion_dir $MOTION --annotation "$ANN" --name $NAME --robot g1
```

```bash
# ---------- part 3: train (NO PYTHONNOUSERSITE here) ----------
cd $SB/HDMI
$PY_HDMI scripts/play.py  algo=ppo_roa_train task=G1/hdmi/$TASK +task.command.replay_motion=true
$PY_HDMI scripts/train.py algo=ppo_roa_train task=G1/hdmi/$TASK
```

---

# 3. Running the stages manually

Same commands as §2b, one at a time, with what each is doing and what to check
before moving on. Export the config block from §2b first.

Full interpreter paths mean you never `conda activate` and never mix envs.

## Stage 1 — GVHMR: video → SMPL-X

```bash
cd $SB/GVHMR
PYTHONNOUSERSITE=1 $PY_GVHMR tools/demo/demo.py --video="$CLIP" -s
cd -
```

- **You must `cd` into `$SB/GVHMR` first.** GVHMR is inconsistent about paths:
  `tracker.py` resolves the YOLO weights via `PROJ_ROOT` (absolute), but
  `vitpose.py` hardcodes a *relative* `"inputs/checkpoints/vitpose/..."`. Run it
  from anywhere else and YOLO tracking succeeds, then ViTPose dies with
  `FileNotFoundError`. `hmr4d/__init__.py` defines `os_chdir_to_proj_root()` but
  the demo never calls it.
- **`--video` must therefore be an absolute path** — you've changed directory.
- `-s` skips visual odometry. Use it whenever the camera is locked off. **Drop
  it for a moving camera** (slower; solves camera motion with SimpleVO). To
  check whether a clip is static, phase-correlate the background across frames —
  cumulative drift over ~10 px at 270×480 means it pans.
- **It ends in a traceback.** `render_incam` fails on the missing SMPL body
  model. That is the optional overlay video only — the real output is already
  saved. See Troubleshooting.

```bash
export PRED=$SB/GVHMR/outputs/demo/$(basename "${CLIP%.*}")/hmr4d_results.pt
ls -lh "$PRED"
```

## Stage 2 — retarget to the robot

```bash
PYTHONNOUSERSITE=1 $PY_GMR $SB/pipeline/scripts/gvhmr_to_gmr.py \
  --gvhmr_pred "$PRED" \
  --robot $ROBOT \
  --video "$CLIP" \
  --save_path $SB/pipeline/work/$TASK/$TASK.pkl
```

Pass `--video` so the true frame rate is detected. GMR's loader hardcodes
`mocap_frame_rate = 30`, and GVHMR emits one pose per video frame — so a 60 fps
clip would otherwise retarget at **half speed with wrong velocities**. If the
`[fps] overriding` line prints, that just saved you. Override manually with
`--src_fps 59.94` if you have no video file to hand.

## Stage 3 — FK → HDMI motion (robot only)

```bash
PYTHONNOUSERSITE=1 $PY_GMR $SB/pipeline/scripts/gmr_to_hdmi.py \
  --gmr_pkl $SB/pipeline/work/$TASK/$TASK.pkl \
  --robot $ROBOT \
  --out_dir $SB/HDMI/data/motion/data_for_sim/$TASK \
  --target_fps 50

PYTHONNOUSERSITE=1 $PY_GMR $SB/pipeline/scripts/validate_motion.py \
  $SB/HDMI/data/motion/data_for_sim/$TASK \
  --reference $SB/HDMI/data/motion/data_for_sim/carry_and_place_bread_box-0829
```

Keep motions under `HDMI/data/motion/` — Hydra resolves `data_path` relative to
the HDMI repo root.

## Stage 4 — annotate the object (manual)

```bash
PYTHONNOUSERSITE=1 $PY_GMR $SB/pipeline/scripts/make_annotation.py \
  $SB/HDMI/data/motion/data_for_sim/$TASK \
  --object box \
  --out $SB/pipeline/annotations/$TASK.json

nano $SB/pipeline/annotations/$TASK.json
```

It prints per-wrist height/speed sparklines and guesses a carry window. **Always
check the guessed frames against the video.** `--object` must name an asset in
HDMI's registry (see §6). Worked examples: `annotations/EXAMPLE_*.json`.

For G1 you may set `parent` to `left_rubber_hand` / `right_rubber_hand` even
though the template suggests `*_wrist_yaw_link` — the resolver knows both,
because FK runs over a superset of the bodies actually written to the file.

## Stage 5 — repack with object + contact

```bash
PYTHONNOUSERSITE=1 $PY_GMR $SB/pipeline/scripts/gmr_to_hdmi.py \
  --gmr_pkl $SB/pipeline/work/$TASK/$TASK.pkl \
  --robot $ROBOT \
  --out_dir $SB/HDMI/data/motion/data_for_sim/$TASK \
  --annotation $SB/pipeline/annotations/$TASK.json \
  --target_fps 50

PYTHONNOUSERSITE=1 $PY_GMR $SB/pipeline/scripts/validate_motion.py \
  $SB/HDMI/data/motion/data_for_sim/$TASK
```

Same command as stage 3 plus `--annotation`. Iterate 4↔5 until it's right.

## Stage 6 — look at it

```bash
PYTHONNOUSERSITE=1 $PY_GMR $SB/pipeline/scripts/render_motion.py \
  $SB/HDMI/data/motion/data_for_sim/$TASK \
  --robot $ROBOT \
  -o $SB/pipeline/work/$TASK/preview.mp4
```

The object must turn **green** exactly while the hand is on it. That is the
annotation verified. Do this before spending GPU hours.

## Stage 7 — task config

```bash
PYTHONNOUSERSITE=1 $PY_GMR $SB/pipeline/scripts/make_task_cfg.py \
  --motion_dir $SB/HDMI/data/motion/data_for_sim/$TASK \
  --annotation $SB/pipeline/annotations/$TASK.json \
  --name PushBox \
  --robot g1
```

`--name` is CamelCase; the script prints the exact train command back. Writes to
`HDMI/cfg/task/G1/hdmi/push_box.yaml`. Tune the contact offsets in that file
before training (see §5).

### Objectless clips → tracking tasks

If the clip has no object, or its object has no simulatable asset, there is
still a useful task: pure whole-body motion tracking. `--mode auto` detects this
and emits a `tracking` config instead:

```bash
PYTHONNOUSERSITE=1 $PY_GMR $SB/pipeline/scripts/make_task_cfg.py \
  --motion_dir $MOTION --name ShovelDirt --robot g1
# [mode] auto -> tracking (no object bodies)
# writes HDMI/cfg/task/G1/tracking/shovel_dirt.yaml
```

The difference: it inherits `base/tracking-base` (command
`RobotTracking`, not `RobotObjectTracking`), carries no object or contact
fields, and uses the no-hand collision variant `g1_29dof_nohand-feet_sphere` —
matching HDMI's shipped `cfg/task/G1/tracking/*.yaml`. Train it the same way,
with `task=G1/tracking/<snake>`.

Note `cfg/task/G1/tracking/dance.yaml` ships broken upstream: it references
`base/tracking-base-wbt`, which does not exist in this checkout.

## Stage 8 — replay, then train

```bash
cd $SB/HDMI

# replay the reference first
$PY_HDMI scripts/play.py algo=ppo_roa_train task=G1/hdmi/$TASK \
  +task.command.replay_motion=true

# teacher
$PY_HDMI scripts/train.py algo=ppo_roa_train task=G1/hdmi/$TASK
$PY_HDMI scripts/play.py  algo=ppo_roa_train task=G1/hdmi/$TASK \
  checkpoint_path='run:<teacher-wandb-run-path>'

# student
$PY_HDMI scripts/train.py algo=ppo_roa_finetune task=G1/hdmi/$TASK \
  checkpoint_path='run:<teacher-wandb-run-path>'
```

No `PYTHONNOUSERSITE` on these. `cd $SB/HDMI` matters — Hydra resolves paths
from the repo root. Add `export_policy=true` to `play.py` to export.

## Dry run

Prove every env works before using your own footage — a couple of minutes:

```bash
export CLIP=$SB/GVHMR/docs/example_video/tennis.mp4
export TASK=tennis_test
```

Then run stages 1, 2, 3, 6. Skip 4–5; that clip has no object.

---

# 4. Command reference

All paths may be absolute or relative to `pipeline/`.

### `scripts/gvhmr_to_gmr.py` — stage 2
| flag | default | meaning |
|---|---|---|
| `--gvhmr_pred` | required | `hmr4d_results.pt` from GVHMR |
| `--robot` | `unitree_g1` | target robot |
| `--save_path` | required | output `.pkl` |
| `--video` | — | source video, used to auto-detect fps |
| `--src_fps` | — | override the detected/hardcoded source fps |
| `--tgt_fps` | `30` | retargeting fps |

### `scripts/gmr_to_hdmi.py` — stages 3 & 5
| flag | default | meaning |
|---|---|---|
| `--gmr_pkl` | required | output of stage 2 |
| `--out_dir` | required | motion folder to create |
| `--robot` | `unitree_g1` | target robot |
| `--annotation` | — | object annotation JSON; omit for robot-only |
| `--target_fps` | `50` | HDMI trains at 50 |
| `--trim` | — | `START:END` frame range, in **source** fps, applied before resampling |
| `--z_offset` | `0.0` | raise/lower the whole motion (metres) |

### `scripts/make_annotation.py` — stage 4
| flag | default | meaning |
|---|---|---|
| `motion_dir` | required | positional |
| `--object` | required | object name (must be a registry asset) |
| `--asset` | `--object` | HDMI asset key if it differs from the name |
| `--out` | required | JSON to write |
| `--parent` | busiest wrist | carrying link |

### `scripts/validate_motion.py`
| flag | default | meaning |
|---|---|---|
| `motion_dir` | required | positional |
| `--reference` | — | a known-good motion to print alongside |

### `scripts/render_motion.py` — stage 6
| flag | default | meaning |
|---|---|---|
| `motion_dir` | required | positional |
| `-o, --out` | `<motion_dir>/preview.mp4` | output mp4 |
| `--robot` | `unitree_g1` | target robot |
| `--width` / `--height` | `960` / `720` | frame size |
| `--distance` / `--azimuth` / `--elevation` | `3.2` / `135` / `-15` | camera |
| `--object_size` | `0.15 0.15 0.15` | drawn box half-extents |
| `--stride` | `1` | render every Nth frame |
| `--max_frames` | — | cap frame count |

### `scripts/make_task_cfg.py` — stage 7
| flag | default | meaning |
|---|---|---|
| `--motion_dir` | required | |
| `--annotation` | — | reads object/articulation names from it |
| `--name` | required | CamelCase task name |
| `--robot` | `g1` | `g1` or `h1_2` |
| `--mode` | `auto` | `hdmi` (object interaction) or `tracking` (whole-body only). `auto` picks `tracking` when the motion has no object bodies. |
| `--out` | `HDMI/cfg/task/<R>/hdmi/<snake>.yaml` | |
| `--mass_range` | `1.0 2.0` | randomization |
| `--half_width` / `--contact_height` | `0.17` / `0.14` | default contact points |

---

# 5. Annotation format

`annotations/*.json`. Frame indices are on the **resampled** timeline (50 fps),
*not* the source video.

| mode | meaning |
|---|---|
| `static` | object holds a fixed pose |
| `lerp` | object moves between two poses (slerp on rotation) |
| `attach` | object rides a robot link rigidly |

```json
{
  "object": {"name": "suitcase", "asset": "suitcase"},
  "extra_objects": [{"name": "support0", "pos": [0.6,0,0.35], "quat": [1,0,0,0]}],
  "articulation": {"joint": "door_joint", "keyframes": [[0,0.0],[430,-1.40]]},
  "segments": [
    {"start": 0,   "end": 209, "mode": "static", "pos": [0.62,-0.18,0.16], "quat": [1,0,0,0]},
    {"start": 210, "end": 640, "mode": "attach", "parent": "right_rubber_hand"},
    {"start": 641, "end": 900, "mode": "static", "pos": [2.10,0.35,0.16], "quat": [1,0,0,0]}
  ],
  "contact": [[210, 640]],
  "smoothing": {"window": 9}
}
```

`attach` derives the grasp transform from the pose the object held at the end of
the *previous* segment, so the handoff is continuous. Override with
`offset_pos`/`offset_quat`. It warns if the carrying link is >0.6 m from the
object at pickup — that means your frames are wrong.

Articulated objects (doors, fold-chairs) add the `articulation` block; the joint
is appended to `joint_names`/`joint_pos`, matching HDMI's shipped door data.

## Contact points live in the task YAML, not here

Only the binary `object_contact` flag is per-frame. The *desired contact points*
are static offsets in `HDMI/cfg/task/<R>/hdmi/<task>.yaml`:

- `contact_target_pos_offset` — where to touch, in the **object's** local frame
- `contact_eef_pos_offset` — which point on the hand touches, in the **EEF's** local frame

---

# 6. Objects available in sim

RL runs against a simulatable USD — the policy never sees your object, only its
sim twin. Registered in `HDMI/active_adaptation/assets/__init__.py`:

`door` (articulated) · `foldchair` (articulated) · `box` · `box_small` ·
`bread_box` · `suitcase` · `stool` · `stool_low` · `ball` · `foam` ·
`wood_board` · `stair` · `platform0` · `platform1` · `wall0` · `support0` ·
`support1`

Anything else means authoring a USD asset first.

---

# 7. Output format

`motion.npz`

| key | shape | notes |
|---|---|---|
| `body_pos_w` | `[T,B,3]` | world positions |
| `body_quat_w` | `[T,B,4]` | **wxyz** |
| `body_lin_vel_w` | `[T,B,3]` | |
| `body_ang_vel_w` | `[T,B,3]` | |
| `joint_pos` | `[T,J]` | |
| `joint_vel` | `[T,J]` | |
| `object_contact` | `[T,1]` | bool |

`B` = robot links **+ object bodies**; `J` = robot joints **+ object joints**.
`meta.json` carries `body_names`, `joint_names`, `fps`.

Quaternion conventions: MuJoCo, Isaac and HDMI all use **wxyz**. GMR's `.pkl` is
the sole exception — it stores root rotation as **xyzw**.

---

# 8. Troubleshooting

**Stage 1 ends in a traceback but I got `hmr4d_results.pt`.**
Expected. `render_incam` needs the **SMPL** body model; we only have SMPL-X.
Pose inference is unaffected. To get the overlay, download SMPL from
smpl.is.tue.mpg.de into `GVHMR/inputs/checkpoints/body_models/smpl/`.

**Stage 1: `FileNotFoundError: 'inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth'`**
YOLO tracking succeeded, then ViTPose failed. You ran the demo from outside the
GVHMR directory. `vitpose.py` hardcodes a relative checkpoint path while
`tracker.py` uses an absolute one, so the run gets partway before dying.
`cd $SB/GVHMR` first and keep `--video` absolute.

**Stage 1: `UnpicklingError: Unsupported global: numpy.core.multiarray._reconstruct`**
torch 2.6 flipped `torch.load`'s `weights_only` default to `True`; GVHMR reads
`.pt` files it wrote itself (here `slam_results.pt`, a numpy array) plus the
official checkpoints. Only appears on the **non-static-camera path**, since
`slam_results.pt` exists only when you omit `-s`. Fix once with:

```bash
./patch_gvhmr_torch26.sh
```

It appends a `torch.load` shim to `GVHMR/hmr4d/__init__.py` (backup at
`__init__.py.orig`), is idempotent, and is re-applied by `setup_gvhmr_env.sh`.
Preprocessing is cached, so re-running the demo resumes at HMR4D inference.

**The retargeted motion looks half/double speed.**
The source clip isn't ~30 fps. Re-run stage 2 with `--video` (auto-detect) or
`--src_fps`.

**`ModuleNotFoundError: No module named 'yaml'` / wrong numpy in gvhmr or gmr.**
You dropped `PYTHONNOUSERSITE=1`. See §1.

**`ModuleNotFoundError: No module named 'active_adaptation'` running HDMI scripts.**
The editable install points at a path that no longer exists — the repos were
moved into `simbench/` after the envs were built. Check `pip show
active_adaptation | grep -i editable`; if it names a dead path:

```bash
cd $SB/HDMI
~/miniconda3/envs/hdmi/bin/pip install -e . --no-deps
```

**Keep `--no-deps`** — `setup.py` pins `torch==2.7.0`, and a plain
`pip install -e .` can try to reinstall torch and wreck the IsaacSim stack.
There is also a hand-written `site-packages/hdmi-repo-root.pth` that had the old
path and needed updating separately. The `gmr` env had the identical problem
(it pointed at `~/mj_ws/GMR_HuroMi`).

Note this only shows up via `python scripts/foo.py`, where `sys.path[0]` is
`scripts/`, not the repo root — a `-c` or heredoc run from the repo root puts
cwd on the path and hides it.

**`ModuleNotFoundError: typing_extensions` in the hdmi env.**
You *added* `PYTHONNOUSERSITE=1`. Remove it for hdmi commands.

**Validator says the robot floats — read `min-z` correctly first.**
It reports the *mean* lowest-body height, which conflates a systematic vertical
offset (bad) with genuinely dynamic footwork (fine). Compare the **minimum**
against a planted-foot reference: HDMI's bread_box sits at 0.035 m for ~100% of
frames, so 0.035 m is what a planted G1 `ankle_roll_link` looks like. The tennis
clip's mean is 0.110 m but its minimum is 0.046 m — only ~1 cm of real bias, the
rest is lunges. Apply `--z_offset -(min_ankle_z - 0.035)` only when the
*minimum* is clearly off; an offset based on the mean will bury the feet.

**"object is N m from `<link>` at pickup".**
Your carry-window frames are wrong, or `parent` names the wrong link. Re-check
against the video.

**Object never turns green in the preview.**
`contact` intervals are empty or outside the clip. Remember frame indices are on
the 50 fps timeline.

---

# 9. Design notes

**Why stage 3 exists.** GMR emits only `root_pos`, `root_rot`, `dof_pos`. HDMI
wants **every link's** world pose *and* velocities. `gmr_to_hdmi.py` pushes qpos
through the robot's MuJoCo model (`mj_kinematics`), reads `xpos`/`xquat` back,
resamples 30→50 fps, and finite-differences velocities with the same quaternion
formula as `HDMI/active_adaptation/utils/motion.py`, so the baked-in values match
what the trainer would compute. No upstream repo does this.

**Why stage 4 is manual.** Inherent to HDMI, not a shortcut here — see the
upstream FAQ. The video gives you the *human*; nothing in it recovers the
object's 6-DoF trajectory.

**Environments.**

| env | purpose | notes |
|---|---|---|
| `gvhmr` | stage 1 | torch 2.8.0+**cu128** — upstream pins cu121, which has no sm_120 kernels and cannot run on this RTX 5090. `pytorch3d` compiled from source for arch 12.0. `chumpy` dropped (unused; its setup.py imports `pip` and fails under build isolation). `yacs` added (imported by `hmr4d/network/hmr2`, missing from requirements). `ultralytics>=8.3.0` required because torch 2.6 flipped `torch.load`'s `weights_only` default. |
| `gmr` | stages 2–7 | `general_motion_retargeting` editable |
| `locomujoco` | optional | alternative retargeter, AMASS/SMPL-fitting based — better for AMASS/OMOMO source data. Isolated: needs numpy 2.2 / torch 2.14. |
| `hdmi` | training | IsaacSim 4.5 + IsaacLab 2.2. Verified: HDMI's own `MotionDataset` loads motions from this pipeline, and names match its canonical G1 lists. |

**Scale expectation.** HDMI trains **one specialist policy per task from
essentially one reference motion** — 6 tasks on hardware, 14 in sim. There is no
dataset in the usual sense; this pipeline is for careful single-clip curation,
not bulk throughput. Reference clips run 11.5–29.3 s, median 16.7 s.

**Deployment caveat.** The trained policy is not vision-conditioned. Upstream
gets global pose from mocap markers on the robot pelvis and the object, and
lists that as a limitation. The video buys you the *skill specification*, not
runtime state estimation.
