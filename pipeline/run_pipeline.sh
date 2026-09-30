#!/usr/bin/env bash
# HDMI end-to-end: monocular RGB video -> trainable reference motion.
#
#   ./run_pipeline.sh <video.mp4> <task_name> [robot] [object_name]
#
# Stages 1-3 are automatic. Stage 4 (object trajectory + contact) is manual by
# design -- the script stops there and hands you a template to edit, exactly as
# the HDMI FAQ describes. Re-run with STAGE=post to finish once it's edited.
#
# Envs: gvhmr (stage 1) | gmr (stages 2-3) | hdmi (training). They are kept
# separate on purpose -- GVHMR and IsaacSim do not share a dependency set.
set -euo pipefail

ROOT=/home/mchang344/mj_ws/simbench
PIPE=$ROOT/pipeline
export PYTHONNOUSERSITE=1   # ~/.local holds 4GB of py310 pkgs that shadow envs

PY_GVHMR=/home/mchang344/miniconda3/envs/gvhmr/bin/python
PY_GMR=/home/mchang344/miniconda3/envs/gmr/bin/python

VIDEO=${1:?usage: run_pipeline.sh <video.mp4> <task_name> [robot] [object_name]}
TASK=${2:?usage: run_pipeline.sh <video.mp4> <task_name> [robot] [object_name]}
ROBOT=${3:-unitree_g1}
# Object name for the annotation template; defaults to the last word of the
# task name ("move_suitcase" -> "suitcase"), which is right more often than not.
OBJECT=${4:-${TASK##*_}}
STAGE=${STAGE:-all}
# STATIC_CAM=1 (default) passes -s to GVHMR, skipping visual odometry.
# Set STATIC_CAM=0 for handheld/panning footage so camera motion is solved.
STATIC_CAM=${STATIC_CAM:-1}

WORK=$PIPE/work/$TASK
MOTION=$ROOT/HDMI/data/motion/data_for_sim/$TASK
ANN=$PIPE/annotations/$TASK.json
mkdir -p "$WORK"

if [ "$STAGE" != "post" ]; then
  echo "################ [1/6] GVHMR: video -> SMPL-X ################"
  # -s skips visual odometry; omitted when STATIC_CAM=0 so VO solves the pan.
  SFLAG=(); [ "$STATIC_CAM" = "1" ] && SFLAG=(-s)
  ( cd "$ROOT/GVHMR" && $PY_GVHMR tools/demo/demo.py --video="$VIDEO" \
      --output_root "$WORK/gvhmr" "${SFLAG[@]}" )
  PRED=$(find "$WORK/gvhmr" -name hmr4d_results.pt | head -1)
  echo "  -> $PRED"

  echo "################ [2/6] GMR: SMPL-X -> robot qpos ################"
  # --video lets stage 2 detect the true fps and override GMR's hardcoded 30.
  $PY_GMR "$PIPE/scripts/gvhmr_to_gmr.py" \
      --gvhmr_pred "$PRED" --robot "$ROBOT" --save_path "$WORK/$TASK.pkl" \
      --video "$VIDEO"

  echo "################ [3/6] FK -> HDMI motion (robot only) ################"
  $PY_GMR "$PIPE/scripts/gmr_to_hdmi.py" \
      --gmr_pkl "$WORK/$TASK.pkl" --robot "$ROBOT" \
      --out_dir "$MOTION" --target_fps 50

  $PY_GMR "$PIPE/scripts/validate_motion.py" "$MOTION"

  if [ ! -f "$ANN" ]; then
    echo "################ [4/6] object annotation (MANUAL) ################"
    $PY_GMR "$PIPE/scripts/make_annotation.py" "$MOTION" \
        --object "$OBJECT" --out "$ANN"
    cat <<MSG

  ------------------------------------------------------------------
  Stopping here. Edit the object trajectory and contact frames:

      $ANN

  Check the guessed pickup/drop-off frames against $VIDEO, set the
  object's resting poses, and name the carrying link. Then run:

      STAGE=post STATIC_CAM=$STATIC_CAM $0 "$VIDEO" "$TASK" "$ROBOT"
  ------------------------------------------------------------------
MSG
    exit 0
  fi
fi

echo "################ [5/6] re-pack with object + contact ################"
$PY_GMR "$PIPE/scripts/gmr_to_hdmi.py" \
    --gmr_pkl "$WORK/$TASK.pkl" --robot "$ROBOT" \
    --out_dir "$MOTION" --annotation "$ANN" --target_fps 50

$PY_GMR "$PIPE/scripts/validate_motion.py" "$MOTION" \
    --reference "$ROOT/HDMI/data/motion/data_for_sim/carry_and_place_bread_box-0829"

echo "################ [6/6] task config ################"
CAMEL=$($PY_GMR -c "import sys;print(''.join(w.capitalize() for w in sys.argv[1].split('_')))" "$TASK")
SHORT_ROBOT=$([ "$ROBOT" = "unitree_g1" ] && echo g1 || echo h1_2)
$PY_GMR "$PIPE/scripts/make_task_cfg.py" \
    --motion_dir "$MOTION" --annotation "$ANN" --name "$CAMEL" --robot "$SHORT_ROBOT"

cat <<MSG

Done. Before burning GPU hours, replay the reference in Isaac Sim:

  cd $ROOT/HDMI
  conda activate hdmi
  python scripts/play.py algo=ppo_roa_train task=<R>/hdmi/$TASK +task.command.replay_motion=true

MSG
