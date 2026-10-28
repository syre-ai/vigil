"""Central path resolution for the Vigil ML pipeline (backend/vigil/*).

Mirrors backend/app/config.py's env vars so the API layer and the offline
pipeline modules (vigil.pipeline, vigil.embed, vigil.data, vigil.evaluate, ...)
always agree on the same directories -- whichever one starts up first --
whether pointed at the real, growing dataset under `data/` or at the small
synthetic fixture under `data/scores_fixture/` used by the test suite (see
backend/app/test_api.py).

Env vars (same names/defaults as backend/app/config.py):
    VIGIL_DATA_ROOT  -- root directory holding rareplanes/, chips/, emb/
                        (default: <repo>/data)
    VIGIL_SCORES_DIR -- directory holding {version}.parquet and friends
                        (default: <VIGIL_DATA_ROOT>/scores)

Deliberately dependency-free (stdlib only), so importing this module alone
never pulls in numpy/pandas/torch/etc.
"""

from __future__ import annotations

import os
from pathlib import Path

# backend/vigil/paths.py -> parents[0]=vigil, [1]=backend, [2]=<repo root>
REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = Path(os.environ.get("VIGIL_DATA_ROOT", str(REPO_ROOT / "data"))).resolve()
SCORES_DIR = Path(os.environ.get("VIGIL_SCORES_DIR", str(DATA_ROOT / "scores"))).resolve()

RP_DIR = DATA_ROOT / "rareplanes"
EMB_DIR = DATA_ROOT / "emb"
CHIPS_DIR = DATA_ROOT / "chips"
CHIPS_DET_DIR = DATA_ROOT / "chips_det"  # detector-output chips (field mode) -- see scripts/extract_det_chips.py
