"""Central configuration for the Vigil backend.

All paths are derived from a handful of environment variables so the same
code can run against the real, growing dataset under `data/` or against the
small synthetic fixture under `data/scores_fixture/` (used by the test
suite). See backend/app/test_api.py for how the fixture wires these up.

DATA_ROOT/SCORES_DIR are resolved by backend/vigil/paths.py (the offline ML
pipeline's own path module) and re-exported here under these names, so the
API layer and vigil.pipeline/vigil.embed/vigil.data/vigil.evaluate always
agree on the same directories regardless of which one imports first.

Env vars:
    VIGIL_DATA_ROOT  -- root directory holding rareplanes/ and chips/
                        (default: <repo>/data)
    VIGIL_SCORES_DIR -- directory holding {version}.parquet and friends
                        (default: <VIGIL_DATA_ROOT>/scores)
    VIGIL_DB_PATH    -- sqlite file for mutable review state
                        (default: <VIGIL_DATA_ROOT>/vigil.db)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# backend/app/config.py -> parents[0]=app, [1]=backend, [2]=<repo root>
REPO_ROOT = Path(__file__).resolve().parents[2]

# sys.path needs backend/ on it so `import vigil...` works (here, and from /adapt).
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from vigil.paths import CHIPS_DET_DIR as _VIGIL_CHIPS_DET_DIR  # noqa: E402
from vigil.paths import CHIPS_DIR as _VIGIL_CHIPS_DIR  # noqa: E402
from vigil.paths import DATA_ROOT, RP_DIR, SCORES_DIR  # noqa: E402

DB_PATH = Path(os.environ.get("VIGIL_DB_PATH", str(DATA_ROOT / "vigil.db"))).resolve()

RAREPLANES_DIR = RP_DIR
TILES_PARQUET = RAREPLANES_DIR / "tiles.parquet"
ANNOTATIONS_PARQUET = RAREPLANES_DIR / "annotations.parquet"

CHIPS_DIR = _VIGIL_CHIPS_DIR
CHIPS_INDEX_PARQUET = CHIPS_DIR / "index.parquet"

# Field mode (research/08-field-mode-proposal.md): detector-output chips, see scripts/extract_det_chips.py.
CHIPS_DET_DIR = _VIGIL_CHIPS_DET_DIR
CHIPS_DET_INDEX_PARQUET = CHIPS_DET_DIR / "index.parquet"

CORS_ORIGINS = [
    o.strip()
    for o in os.environ.get("VIGIL_CORS_ORIGINS", "http://localhost:5173").split(",")
    if o.strip()
]
