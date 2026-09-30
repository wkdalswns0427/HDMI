#!/usr/bin/env bash
# Optional stage-2 alternative: LocoMuJoCo instead of GMR.
#
# HDMI's README allows either ("GVHMR -> GMR/LocoMujoco"). GMR is the wired-in
# default here because it ships a GVHMR bridge and is already working. Keep
# LocoMuJoCo in its own env: it needs numpy>2.0 + jax + mjx, which conflicts
# with the gmr env's numpy 1.26.4.
#
# Its retargeting path is AMASS/SMPL-fitting based (optimization per clip), so
# it is the better choice for AMASS/OMOMO source data -- e.g. HDMI's suitcase
# task, which came from OMOMO rather than from video.
set -euo pipefail

CONDA=/home/mchang344/miniconda3
ENV=locomujoco
PIP=$CONDA/envs/$ENV/bin/pip
PY=$CONDA/envs/$ENV/bin/python
REPO=/home/mchang344/mj_ws/simbench/loco-mujoco
export PYTHONNOUSERSITE=1

if [ ! -x "$PY" ]; then
  $CONDA/bin/conda create -y -n $ENV python=3.10
fi

echo "=== loco-mujoco (+ smpl extra) ==="
$PIP install --no-cache-dir -e "$REPO[smpl]"

echo "=== verify ==="
$PY - <<'PYEOF'
import loco_mujoco, numpy, mujoco
print("loco_mujoco", getattr(loco_mujoco, "__version__", "?"))
print("numpy", numpy.__version__, "| mujoco", mujoco.__version__)
from loco_mujoco.smpl import retargeting
print("retargeting API OK:", [n for n in ("fit_smpl_motion","fit_smpl_shape")
                              if hasattr(retargeting, n)])
PYEOF

cat <<'MSG'

Note: LocoMuJoCo expects SMPL(-H) models under its own path. Reuse the licensed
copies already in this workspace:
    GMR/assets/body_models/smplx/
See loco_mujoco/smpl/retargeting.py:get_smpl_model_path() for the expected layout.
MSG
