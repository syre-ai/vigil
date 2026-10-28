#!/bin/bash
# Undo the most recently built adapted version (v2+) so it can be rebuilt with a different
# adaptation rule/parameter (e.g. a new `assign_min_conf`) WITHOUT redoing the analyst's reviews:
# deletes that version's own scores/batches/model/eval files, drops it from versions.json, and resets
# every review that was applied to it back to unapplied (version_applied -> NULL) -- so the next
# POST /adapt (the frontend's "Apply") picks up exactly the same feedback again, through
# vigil.pipeline.build_version's current code path. Refuses to touch v1 (the no-feedback baseline
# build) or run when fewer than two versions exist. Never touches reviews applied to an EARLIER
# version (only the top one) -- if there are more than two versions, run this repeatedly to peel
# them off one at a time.
set -e
cd "$(dirname "$0")/.."
.venv/bin/python - <<'PY'
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, "backend")
from app import config          # noqa: E402  -- VIGIL_DB_PATH-aware DB_PATH
from vigil.paths import SCORES_DIR  # noqa: E402  -- VIGIL_DATA_ROOT/VIGIL_SCORES_DIR-aware

versions_path = SCORES_DIR / "versions.json"
if not versions_path.exists():
    print(f"no {versions_path} -- nothing to do")
    raise SystemExit(0)

versions = json.loads(versions_path.read_text())
if len(versions) < 2:
    print(f"only {len(versions)} version(s) recorded ({[v['id'] for v in versions]}) -- nothing to rebuild")
    raise SystemExit(1)

top = versions[-1]
top_id = top["id"]
if top_id == "v1":
    print("top version is v1 (the no-feedback baseline build) -- nothing to rebuild")
    raise SystemExit(1)

removed = []
for f in sorted(SCORES_DIR.glob(f"{top_id}*")):
    f.unlink()
    removed.append(str(f))
eval_path = SCORES_DIR / f"eval_{top_id}.json"
if eval_path.exists():
    eval_path.unlink()
    removed.append(str(eval_path))

kept = [v for v in versions if v["id"] != top_id]
versions_path.write_text(json.dumps(kept, indent=2))

n_reset = 0
if config.DB_PATH.exists():
    con = sqlite3.connect(config.DB_PATH)
    cur = con.execute("UPDATE reviews SET version_applied=NULL WHERE version_applied=?", (top_id,))
    n_reset = cur.rowcount
    con.commit()
    con.close()

print(f"removed version {top_id}: {len(removed)} file(s)")
for f in removed:
    print("  -", f)
print(f"versions.json now: {[v['id'] for v in kept]}")
print(f"reset version_applied -> NULL for {n_reset} review(s) previously applied to {top_id}")
print("press Apply again (POST /adapt) to rebuild those same reviews under the current rule/parameters")
PY
