#!/bin/bash
# Vigil ML core, end to end: extract chips -> embed chips -> embed tiles -> extract det chips ->
# embed det chips -> build v1 (which scores det chips + evaluates v1 internally, writing
# eval_v1.json -- see vigil.pipeline.build_version). Det-chip steps are field mode (research/
# 08-field-mode-proposal.md); they need data/scores/detections.parquet (scripts/run_detector.py) to
# already exist -- if it's missing, extract_det_chips.py exits with an error, so run
# scripts/run_detector.py first (or skip these two steps -- build_version degrades gracefully if
# data/chips_det/index.parquet doesn't exist yet).
# Idempotent -- safe to re-run at any point while the RarePlanes download is still filling in;
# each step only does the work that's missing (new chips, new embeddings, a fresh version build).
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python

echo "== 1/6 extract_chips =="
"$PY" scripts/extract_chips.py

echo "== 2/6 embed chips =="
"$PY" -m backend.vigil.embed

echo "== 3/6 embed tiles (scene-level drift) =="
"$PY" -m backend.vigil.embed tiles

echo "== 4/6 extract_det_chips (field mode) =="
"$PY" scripts/extract_det_chips.py

echo "== 5/6 embed det chips (field mode) =="
"$PY" -m backend.vigil.embed --det

echo "== 6/6 build_version v1 (scores det chips + evaluates v1 internally) =="
"$PY" -m backend.vigil.pipeline
