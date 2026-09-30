"""Where things are, derived from this file's location.

The pipeline lives at <repo>/pipeline, so everything inside the repo is found
relative to here and works from any clone.

The sibling repos (GVHMR, GMR, loco-mujoco) and the conda envs live outside
the repo. They default to the layout this was built in -- the siblings next to
the HDMI checkout, conda in ~/miniconda3 -- and each can be overridden by an
environment variable for a different machine:

    SIMBENCH_ROOT   directory holding GVHMR/, GMR/, loco-mujoco/
    GVHMR_ROOT      GVHMR checkout, if not under SIMBENCH_ROOT
    GMR_ROOT        GMR checkout, if not under SIMBENCH_ROOT
    CONDA_ROOT      conda install holding envs/<name>/bin/python

Pure pathlib/os, no heavy imports, because it is loaded from the gvhmr, gmr
and hdmi envs alike.
"""

import os
from pathlib import Path

PIPELINE = Path(__file__).resolve().parents[1]          # <repo>/pipeline
HDMI_ROOT = PIPELINE.parent                              # <repo>

TASK_INFO = PIPELINE / "task_info"
ANNOTATIONS = PIPELINE / "annotations"
ASSETS = HDMI_ROOT / "active_adaptation" / "assets"
TASK_CFG = HDMI_ROOT / "cfg" / "task"
MOTION_DATA = HDMI_ROOT / "data" / "motion" / "data_for_sim"

SIMBENCH = Path(os.environ.get("SIMBENCH_ROOT", HDMI_ROOT.parent))
GVHMR_ROOT = Path(os.environ.get("GVHMR_ROOT", SIMBENCH / "GVHMR"))
GMR_ROOT = Path(os.environ.get("GMR_ROOT", SIMBENCH / "GMR"))

CONDA_ROOT = Path(os.environ.get("CONDA_ROOT", Path.home() / "miniconda3"))


def conda_python(env: str) -> str:
    """The python interpreter of a named conda env."""
    return str(CONDA_ROOT / "envs" / env / "bin" / "python")
