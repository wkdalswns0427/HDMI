#!/usr/bin/env bash
# Build a `gvhmr` conda env that works on Blackwell (sm_120 / RTX 5090).
# GVHMR upstream pins torch==2.3.0+cu121, which has no sm_120 kernels, so we
# move to cu128 wheels and compile pytorch3d from source for arch 12.0.
set -euo pipefail

CONDA=/home/mchang344/miniconda3
ENV=gvhmr
PY=$CONDA/envs/$ENV/bin/python
PIP=$CONDA/envs/$ENV/bin/pip
GVHMR=/home/mchang344/mj_ws/simbench/GVHMR

export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export TORCH_CUDA_ARCH_LIST="12.0"
export MAX_JOBS=16
export FORCE_CUDA=1
export PYTHONNOUSERSITE=1   # ~/.local has 4GB of py310 pkgs that shadow the env

echo "=== [1/6] create env ==="
if [ ! -x "$PY" ]; then
  $CONDA/bin/conda create -y -n $ENV python=3.10
fi

echo "=== [2/6] torch cu128 (sm_120 capable) ==="
$PIP install --no-cache-dir "torch==2.8.0" "torchvision==0.23.0" \
  --index-url https://download.pytorch.org/whl/cu128

$PY -c "import torch;print('torch',torch.__version__);print('arch',torch.cuda.get_arch_list())"

echo "=== [3/6] GVHMR deps (torch/torchvision/pytorch3d stripped) ==="
# Drop the pinned torch lines, the cu121 index, and the prebuilt pytorch3d wheel.
grep -vE '^(--extra-index-url|torch==|torchvision==|pytorch3d @|chumpy|numpy==)' $GVHMR/requirements.txt \
  | grep -vE '^\s*#' | grep -vE '^\s*$' > /tmp/gvhmr_reqs.txt
echo 'numpy==1.26.4' >> /tmp/gvhmr_reqs.txt   # matches gmr/hdmi envs; 1.23.5 pin existed only for chumpy
cat /tmp/gvhmr_reqs.txt

$PIP install --no-cache-dir -r /tmp/gvhmr_reqs.txt

# yacs is imported by hmr4d/network/hmr2 but missing from requirements.txt.
$PIP install --no-cache-dir yacs

# torch>=2.6 flipped torch.load's weights_only default to True; the pinned
# ultralytics==8.2.42 predates that and cannot load its own yolov8x.pt.
$PIP install --no-cache-dir -U "ultralytics>=8.3.0"

echo "=== [4/6] pytorch3d from source (arch 12.0) ==="
$PIP install --no-cache-dir --no-build-isolation \
  "git+https://github.com/facebookresearch/pytorch3d.git@V0.7.8"

echo "=== [5/6] install GVHMR editable ==="
$PIP install --no-cache-dir -e $GVHMR --no-deps

# torch>=2.6 weights_only shim (GVHMR reads its own .pt artifacts + official
# ckpts containing numpy arrays). Only bites on the non-static-camera path.
bash /home/mchang344/mj_ws/simbench/pipeline/patch_gvhmr_torch26.sh

echo "=== [6/6] verify ==="
$PY - <<'PYEOF'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0))
import pytorch3d, pytorch3d.transforms as T
print("pytorch3d", pytorch3d.__version__)
x = torch.randn(2,3).cuda()
print("so3_exp_map on cuda:", T.so3_exp_map(x).shape)
from pytorch3d import _C
print("pytorch3d._C OK")
import numpy, smplx, ultralytics
print("numpy", numpy.__version__)
print("ALL OK")
PYEOF
