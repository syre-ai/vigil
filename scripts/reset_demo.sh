#!/bin/bash
# Reset the demo to a clean v1 state: drop analyst reviews and any adapted versions (v2+).
set -e
cd "$(dirname "$0")/.."
.venv/bin/python - <<'PY'
import json, sqlite3, pathlib
db = pathlib.Path("data/vigil.db")
if db.exists():
    con = sqlite3.connect(db); con.execute("DELETE FROM reviews"); con.commit(); con.close(); print("reviews cleared")
scores = pathlib.Path("data/scores"); vf = scores / "versions.json"
if vf.exists():
    versions = json.load(open(vf)); keep = [v for v in versions if v["id"] == "v1"]
    for v in versions:
        if v["id"] != "v1":
            for f in scores.glob(f"{v['id']}*"): f.unlink(); print("removed", f)
            ef = scores / f"eval_{v['id']}.json"
            if ef.exists(): ef.unlink(); print("removed", ef)
    json.dump(keep, open(vf, "w"), indent=2); print("versions:", [v["id"] for v in keep])
PY
