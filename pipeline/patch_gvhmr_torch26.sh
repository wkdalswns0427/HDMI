#!/usr/bin/env bash
# torch >= 2.6 flipped torch.load's `weights_only` default to True. GVHMR calls
# torch.load in ~60 places to read (a) .pt artifacts it just wrote itself and
# (b) the official checkpoints from its own Google Drive. Several of those hold
# pickled numpy arrays, which the new default rejects:
#
#   UnpicklingError: Unsupported global: numpy.core.multiarray._reconstruct
#
# Rather than edit every call site, restore the old default once at the package
# entry point -- every GVHMR entry path imports hmr4d. Scope is this repo only.
# Idempotent: safe to re-run.
set -euo pipefail

INIT=/home/mchang344/mj_ws/simbench/GVHMR/hmr4d/__init__.py
MARK="# --- torch>=2.6 weights_only shim"

if grep -q "$MARK" "$INIT"; then
  echo "already patched: $INIT"
  exit 0
fi

cp "$INIT" "$INIT.orig"
cat >> "$INIT" <<'PYEOF'


# --- torch>=2.6 weights_only shim -------------------------------------------
# torch 2.6 changed torch.load's `weights_only` default from False to True.
# GVHMR reads .pt files it produced itself (bbx, vitpose, slam, vit_features,
# hmr4d_results) plus the official released checkpoints; several contain
# pickled numpy arrays, which the strict loader refuses. These are all local,
# trusted files, so restore the previous behaviour for this package.
# Applied by pipeline/patch_gvhmr_torch26.sh -- original at __init__.py.orig
import torch as _torch

if not getattr(_torch.load, "_gvhmr_weights_only_shim", False):
    _gvhmr_orig_torch_load = _torch.load

    def _gvhmr_torch_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _gvhmr_orig_torch_load(*args, **kwargs)

    _gvhmr_torch_load._gvhmr_weights_only_shim = True
    _torch.load = _gvhmr_torch_load
PYEOF

echo "patched: $INIT (backup at $INIT.orig)"
