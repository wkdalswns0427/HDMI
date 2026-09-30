#!/usr/bin/env bash
# Stage-1 weights for GVHMR.
#
# The SMPL/SMPLX body models are licensed and must be obtained by signing up at
# smpl.is.tue.mpg.de / smpl-x.is.tue.mpg.de. They are already present in this
# workspace under GMR/assets/body_models, so we link rather than re-download.
# Everything else comes from the authors' Google Drive folder.
set -euo pipefail

PIPE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # this script's dir
HDMI_ROOT="$(dirname "$PIPE")"                             # the HDMI checkout
SIMBENCH="${SIMBENCH_ROOT:-$(dirname "$HDMI_ROOT")}"       # holds GVHMR/, GMR/, loco-mujoco/
CONDA="${CONDA_ROOT:-$HOME/miniconda3}"
GVHMR="${GVHMR_ROOT:-$SIMBENCH/GVHMR}"
GMR_BODY="${GMR_ROOT:-$SIMBENCH/GMR}/assets/body_models"
PY=$CONDA/envs/gvhmr/bin/python
PIP=$CONDA/envs/gvhmr/bin/pip
export PYTHONNOUSERSITE=1

CKPT=$GVHMR/inputs/checkpoints
mkdir -p "$CKPT/body_models" "$GVHMR/outputs" "$GVHMR/inputs/demo"

echo "=== SMPLX body models (link existing licensed copies) ==="
if [ -d "$GMR_BODY/smplx" ]; then
  ln -sfn "$GMR_BODY/smplx" "$CKPT/body_models/smplx"
  ls -l "$CKPT/body_models/smplx/" | head -5
else
  echo "!! $GMR_BODY/smplx not found - download SMPLX yourself and place it there"
fi

if [ ! -d "$CKPT/body_models/smpl" ]; then
  echo "note: SMPL (not SMPLX) is only needed for rendering/eval; skipping."
fi

echo
echo "=== pretrained weights from the authors' Google Drive ==="
$PIP install -q gdown
$PY -m gdown --folder \
  "https://drive.google.com/drive/folders/1eebJ13FUEXrKBawHpJroW0sNSxLjh9xD" \
  -O "$CKPT" || {
    echo
    echo "!! gdown failed (Drive rate-limits bulk folder pulls)."
    echo "   Download the folder by hand and unpack into: $CKPT"
    echo "   Expected: gvhmr/gvhmr_siga24_release.ckpt, hmr2/epoch=10-step=25000.ckpt,"
    echo "             vitpose/vitpose-h-multi-coco.pth, yolo/yolov8x.pt, dpvo/dpvo.pth"
    exit 1
  }

echo
echo "=== result ==="
find "$CKPT" -maxdepth 2 -type f -o -maxdepth 2 -type l | sort
