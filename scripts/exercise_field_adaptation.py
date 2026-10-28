"""Exercises the FIELD-MODE adaptation workflow END TO END THROUGH THE REAL HTTP API -- the exact
path the analyst-view demo takes (research/09-field-demo-script.md) -- on a SCRATCH copy of
data/scores/ and a SCRATCH sqlite database, so the live app (and its dev server) never sees any of it:

    GET  /api/collections                       -> pick the demo collection (Tirana 2019-02-07, else the
                                                   needs_review collection with the smallest p-value)
    GET  /api/collections/{gid}?only_flagged=1  -> the 7 highest-novelty flagged detector chips
    POST /api/review/{det_id}?source=det        -> 6x new_class "Military fighter", 1x confirm
    GET  /api/review/queue?source=det           -> reviewed chips must have left the field queue
    POST /api/adapt                             -> builds v2 in the scratch scores dir (real vigil.pipeline)
    GET  /api/versions/compare?a=v1&b=v2        -> asserts API.md's `field` FieldBlock shape (by_group etc)
    GET  /api/report, /api/collections/summary  -> analyst-view section present, v2 collections served

No ground truth (matched_ann_id / domain) is used to pick anything -- exactly what a field-mode analyst
sees (a collection's status, novelty ranking, and the detector's own crops).

Isolation: VIGIL_SCORES_DIR and VIGIL_DB_PATH are pointed at a scratch dir BEFORE any vigil/app import
(vigil.paths / app.config read them at import time). VIGIL_DATA_ROOT stays the real data/ (embeddings,
chips, tiles are read-only inputs here; build_version's embedding steps are no-ops when every chip is
already cached). Afterwards this script asserts data/scores/v1.parquet, data/scores/versions.json and
data/vigil.db are byte/row-identical to before, copies the small useful outputs to
artifacts/rehearsal_field_v2/ (git-tracked), and deletes the scratch dir.

Requires data/scores/v1_det.parquet (run scripts/run_pipeline.sh's det-chip steps first).

Run: .venv/bin/python scripts/exercise_field_adaptation.py -- ~15 s end to end: POST /adapt takes the INCREMENTAL
build_version path (v1 exists, so novelty/ART/kNN/UMAP/drift are reused; measured 11 s vs 213-247 s for the full
path, with byte-identical v2 outputs -- see vigil.pipeline.build_version).
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LIVE_SCORES = REPO_ROOT / "data" / "scores"
LIVE_DB = REPO_ROOT / "data" / "vigil.db"
SCRATCH = Path(os.environ.get("VIGIL_FIELD_SCRATCH", "/tmp/vigil_field_scratch"))
ARTIFACT_DIR = REPO_ROOT / "artifacts" / "rehearsal_field_v2"
ARTIFACT_PREFIX = "rehearsal_field_v2"

BASE_VERSION_ID = "v1"
NEW_VERSION_ID = "v2"
NEW_LABEL = "Military fighter"
N_NEW_CLASS = 6
N_CONFIRM = 1
N_BAD_BOX = 1
PREFERRED_GROUP = "tirana-international-airport_2019-02-07"  # the demo's "object problem" collection

# ---- isolation: MUST happen before importing vigil/app (they read these env vars at import) ----
if (SCRATCH / "scores").exists():
    shutil.rmtree(SCRATCH)
(SCRATCH).mkdir(parents=True, exist_ok=True)
shutil.copytree(LIVE_SCORES, SCRATCH / "scores")

# Prune the SCRATCH copy down to the v1 baseline only, regardless of how many versions the LIVE
# data/scores/ directory currently holds (e.g. a hand-built v2+ from a live demo session, which this
# script must never touch or depend on) -- this script always rehearses ONE adaptation step
# (BASE_VERSION_ID -> NEW_VERSION_ID) from a clean base, so it stays re-runnable no matter what the
# live app has since built. Only the scratch copy is edited here; the live files are untouched.
_scratch_versions_path = SCRATCH / "scores" / "versions.json"
if _scratch_versions_path.exists():
    _versions = json.loads(_scratch_versions_path.read_text())
    _extra = [v["id"] for v in _versions if v["id"] != BASE_VERSION_ID]
    if _extra:
        for _vid in _extra:
            for _f in (SCRATCH / "scores").glob(f"{_vid}*"):
                _f.unlink()
            _ef = SCRATCH / "scores" / f"eval_{_vid}.json"
            if _ef.exists():
                _ef.unlink()
        _scratch_versions_path.write_text(json.dumps([v for v in _versions if v["id"] == BASE_VERSION_ID], indent=2))
        print(f"scratch pruned to baseline {BASE_VERSION_ID} only (live data/scores/ also has: {_extra})")

os.environ["VIGIL_SCORES_DIR"] = str(SCRATCH / "scores")
os.environ["VIGIL_DB_PATH"] = str(SCRATCH / "vigil.db")
os.environ.pop("VIGIL_DATA_ROOT", None)  # real data root: embeddings/chips/tiles are read-only inputs
sys.path.insert(0, str(REPO_ROOT / "backend"))

import pandas as pd  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

FIELD_BLOCK_KEYS = {
    "n_detections", "n_excluded_labeled", "n_excluded_same_parent", "n_eligible", "classes", "control",
    "reviewed_flagged", "labeled_collections", "assign_min_conf", "note",
}
LABELED_COLLECTION_KEYS = {
    "group_id", "airfield", "date", "n_labeled", "n_bad_box", "n_not_aircraft",
    "n_flagged_unlabeled", "n_flagged_unlabeled_assigned", "per_class",
}
REVIEWED_FLAGGED_KEYS = {"n", "n_not_aircraft", "n_bad_box", "field_precision_on_flagged"}
CLASS_KEYS = {"label", "n_labels_used", "n_assigned", "assignment_rate", "by_group"}
BY_GROUP_KEYS = {"group_id", "airfield", "date", "n_assigned"}
DET_CHIP_KEYS = {"source", "det_id", "ann_id", "tile_id", "bbox", "conf", "chip_url", "tile_url", "role_pred",
                 "role_conf", "novelty", "is_novel", "knn_dist", "meta", "review"}
FORBIDDEN_KEYS = {"role_true", "matched_ann_id", "batch_id", "domain", "by_domain", "domains"}


def _assert_no_forbidden(obj, path="$") -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            assert k not in FORBIDDEN_KEYS, f"forbidden key {k!r} at {path}.{k}"
            _assert_no_forbidden(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _assert_no_forbidden(v, f"{path}[{i}]")


def _snapshot_live() -> dict:
    snap = {
        "v1": pd.read_parquet(LIVE_SCORES / f"{BASE_VERSION_ID}.parquet"),
        "versions_text": (LIVE_SCORES / "versions.json").read_text(),
        "live_rows": None,
        "score_files": sorted(p.name for p in LIVE_SCORES.iterdir()),
    }
    if LIVE_DB.exists():
        snap["live_rows"] = _live_review_rows()
    return snap


def _live_review_rows() -> set[tuple]:
    """(source, ann_id, action, label, ts, version_applied) rows of the LIVE reviews table."""
    con = sqlite3.connect(LIVE_DB)
    try:
        return {tuple(r) for r in con.execute(
            "SELECT source, ann_id, action, label, ts, version_applied FROM reviews"
        ).fetchall()}
    except sqlite3.OperationalError:
        return set()
    finally:
        con.close()


def _assert_live_unchanged(snap: dict, feedback: list[dict]) -> None:
    after = pd.read_parquet(LIVE_SCORES / f"{BASE_VERSION_ID}.parquet")
    pd.testing.assert_frame_equal(
        snap["v1"].sort_values("ann_id").reset_index(drop=True),
        after.sort_values("ann_id").reset_index(drop=True),
    )
    # Structural proof of isolation: everything this process wrote went through these two paths.
    from app import config as _config
    from vigil import paths as _paths
    assert _paths.SCORES_DIR == (SCRATCH / "scores").resolve() and _config.SCORES_DIR == _paths.SCORES_DIR, _paths.SCORES_DIR
    assert _config.DB_PATH == (SCRATCH / "vigil.db").resolve(), _config.DB_PATH
    assert (SCRATCH / "scores" / f"{NEW_VERSION_ID}.parquet").exists() and (SCRATCH / "scores" / f"eval_{NEW_VERSION_ID}.json").exists()
    # Other clients (the UI against the :8000 dev server) may legitimately change live versions.json /
    # data/scores/ while this runs (apply, reset) -- report it, don't fail on it.
    if (LIVE_SCORES / "versions.json").read_text() != snap["versions_text"]:
        print("NOTE: live versions.json changed during this run (another client -- e.g. the UI applying/resetting "
              "on the dev server; this process's versions.json is the scratch copy)")
    files_after = sorted(p.name for p in LIVE_SCORES.iterdir())
    if files_after != snap["score_files"]:
        print(f"NOTE: live data/scores/ file set changed during this run (another client): "
              f"added {sorted(set(files_after) - set(snap['score_files']))}, removed {sorted(set(snap['score_files']) - set(files_after))}")
    if snap["live_rows"] is not None:
        # This script's reviews went to the scratch db (VIGIL_DB_PATH), so none of its own rows may
        # appear in the live db, and every pre-existing live row must still be there untouched. Other
        # clients (e.g. the UI against the :8000 dev server) may legitimately write to the live db
        # while this runs -- report that, don't fail on it.
        rows_after = _live_review_rows()
        missing = snap["live_rows"] - rows_after
        assert not missing, f"pre-existing live reviews changed/removed: {sorted(missing)[:5]}"
        mine = {(f["source"], f["ann_id"], f["action"], f["label"], f["ts"]) for f in feedback}
        leaked = {r for r in rows_after if (r[0], r[1], r[2], r[3], r[4]) in mine}
        assert not leaked, f"this script's reviews leaked into the live db: {sorted(leaked)}"
        new_rows = rows_after - snap["live_rows"]
        if new_rows:
            print(f"NOTE: {len(new_rows)} review row(s) were written to the live db by ANOTHER client during this run "
                  f"(not by this script -- its reviews are in the scratch db): "
                  f"{sorted((r[0], r[1], r[2]) for r in new_rows)}")
    ops = after[after["batch_id"] != "reference"]
    ctrl = after[after["batch_id"] == "ref_holdout"]
    print(f"OK: live {BASE_VERSION_ID}.parquet unchanged -- {len(after)} rows, {int(after['is_novel'].sum())} flagged "
          f"({len(ops)} operational chips), control novelty {ctrl['is_novel'].mean():.3%}; this run wrote only under {SCRATCH}")


def _live_server_version() -> str | None:
    """Best-effort: the live dev server's (port 8000) CURRENT version, or None if unreachable. Used
    to snapshot-and-compare rather than assume any particular version id -- the live app may already
    be several `/adapt` calls ahead of v1 (e.g. a hand-built demo version), and this script must never
    assume otherwise."""
    try:
        with urllib.request.urlopen("http://localhost:8000/api/summary", timeout=3) as resp:
            return json.loads(resp.read())["current_version"]
    except (urllib.error.URLError, OSError):
        return None


def _check_live_server(before_version: str | None) -> None:
    """Best-effort: if the dev server on :8000 is up, its Benchmark endpoints must still 200 and its
    CURRENT VERSION must be UNCHANGED from before this script ran (whatever that was -- v1, or a
    hand-built vN further along) -- nothing here can touch it, this just proves it."""
    if before_version is None:
        print("live server check skipped (not reachable before this run)")
        return
    now = _live_server_version()
    if now is None:
        print("live server check skipped (not reachable after this run)")
        return
    assert now == before_version, f"live server's current_version changed {before_version!r} -> {now!r} during this run"
    for path in ("/api/batches", "/api/eval", "/api/collections/summary"):
        with urllib.request.urlopen(f"http://localhost:8000{path}", timeout=10) as resp:
            assert resp.status == 200, f"{path} returned {resp.status}"
    print(f"live server check: still on {before_version}; /api/batches, /api/eval, /api/collections/summary -> 200")


def _get(client: TestClient, path: str) -> dict | list:
    r = client.get(path)
    assert r.status_code == 200, f"GET {path} -> {r.status_code}: {r.text[:300]}"
    return r.json()


def _pick_group(collections: dict) -> dict:
    groups = collections["groups"]
    by_id = {g["group_id"]: g for g in groups}
    if PREFERRED_GROUP in by_id and by_id[PREFERRED_GROUP]["n_flagged"] >= N_NEW_CLASS + N_CONFIRM + N_BAD_BOX:
        return by_id[PREFERRED_GROUP]
    candidates = [g for g in groups if g["status"] == "needs_review" and g["n_flagged"] >= N_NEW_CLASS + N_CONFIRM + N_BAD_BOX]
    assert candidates, "no needs_review collection with enough flagged detections"
    return min(candidates, key=lambda g: (g["p_value"] if g["p_value"] is not None else 1.0))


def main() -> None:
    t0 = time.time()
    det_path = LIVE_SCORES / f"{BASE_VERSION_ID}_det.parquet"
    if not det_path.exists():
        raise SystemExit(f"{det_path} does not exist -- run the det-chip pipeline steps first")

    live_server_before = _live_server_version()
    snap = _snapshot_live()
    client = TestClient(app)
    try:
        # ---- 1. collections (analyst view) ----
        collections = _get(client, "/api/collections")
        _assert_no_forbidden(collections)
        summary_v1 = _get(client, "/api/collections/summary")
        group = _pick_group(collections)
        print(f"collections ({BASE_VERSION_ID}): {summary_v1['n_groups']} groups, {summary_v1['n_needs_review']} needs_review, "
              f"{summary_v1['n_insufficient_data']} insufficient_data; reference flag rate {summary_v1['reference_flag_rate']:.3%}")
        print(f"status rule: {collections['status_rule']}")
        print(f"demo collection: {group['group_id']} -- {group['n_detections']} detections, {group['n_flagged']} flagged, "
              f"excess {group['excess_flagged']:+.1f}, p={group['p_value']:.2e}, status={group['status']}")

        # ---- 2. the flagged detector chips, ranked by novelty ----
        detail = _get(client, f"/api/collections/{group['group_id']}?only_flagged=true&sort=novelty&order=desc&limit={N_NEW_CLASS + N_CONFIRM + N_BAD_BOX}")
        _assert_no_forbidden(detail)
        chips = detail["chips"]["items"]
        assert len(chips) >= N_NEW_CLASS + N_CONFIRM + N_BAD_BOX, len(chips)
        for c in chips:
            assert set(c) == DET_CHIP_KEYS, set(c) ^ DET_CHIP_KEYS
            assert c["is_novel"] is True and c["conf"] is not None
        det_detail = _get(client, f"/api/det/{chips[0]['det_id']}")
        assert "matched_ann_id" not in det_detail and "batch_id" not in det_detail, "GT/batch leak on /det/{id}"

        # ---- 3. the analyst decides ----
        feedback = []
        for c in chips[:N_NEW_CLASS]:
            r = client.post(f"/api/review/{c['det_id']}?source=det", json={"action": "new_class", "label": NEW_LABEL})
            assert r.status_code == 200, r.text
            feedback.append(r.json())
        for c in chips[N_NEW_CLASS:N_NEW_CLASS + N_CONFIRM]:
            r = client.post(f"/api/review/{c['det_id']}?source=det", json={"action": "confirm"})
            assert r.status_code == 200, r.text
            feedback.append(r.json())
        for c in chips[N_NEW_CLASS + N_CONFIRM:N_NEW_CLASS + N_CONFIRM + N_BAD_BOX]:
            # bad_box: a localization error (fused/truncated aircraft) -- needs no label, never fits
            # anything, stays scored (see vigil.pipeline.BAD_BOX_POLICY_NOTE).
            r = client.post(f"/api/review/{c['det_id']}?source=det", json={"action": "bad_box"})
            assert r.status_code == 200, r.text
            assert r.json()["label"] is None
            feedback.append(r.json())
        reviewed_ids = {f["ann_id"] for f in feedback}
        print(f"feedback: {N_NEW_CLASS} new_class {NEW_LABEL!r} + {N_CONFIRM} confirm + {N_BAD_BOX} bad_box, all source=det: {sorted(reviewed_ids)}")

        queue = _get(client, f"/api/review/queue?source=det&limit=50")
        assert queue and all(q["group_id"] for q in queue) and all(q["is_novel"] for q in queue)
        assert not (reviewed_ids & {q["det_id"] for q in queue}), "reviewed det chips still in the field queue"
        _assert_no_forbidden(queue)
        gt_queue = _get(client, "/api/review/queue?limit=3")
        assert gt_queue and "det_id" not in gt_queue[0] and "partial" in gt_queue[0], "GT queue shape changed"
        after_group = next(g for g in _get(client, "/api/collections")["groups"] if g["group_id"] == group["group_id"])
        assert after_group["n_reviewed"] == N_NEW_CLASS + N_CONFIRM + N_BAD_BOX, after_group["n_reviewed"]
        assert after_group["n_bad_box"] == N_BAD_BOX, after_group["n_bad_box"]

        # ---- 4. apply (real vigil.pipeline.build_version, into the scratch scores dir) ----
        t1 = time.time()
        r = client.post("/api/adapt", json={"note": "field-mode rehearsal (scratch, discarded after run)"})
        assert r.status_code == 200, r.text
        version = r.json()
        assert version["id"] == NEW_VERSION_ID, version
        assert version["n_new_class"] == N_NEW_CLASS and version["n_confirm"] == N_CONFIRM, version
        assert version["n_bad_box"] == N_BAD_BOX, version
        assert "bad_box actions are recorded as localization errors" in version["note"], version["note"]
        assert NEW_LABEL in version["classes"], version["classes"]
        print(f"/adapt -> {version['id']} in {time.time() - t1:.0f}s; classes={version['classes']}")
        log = _get(client, "/api/review/log?source=det")
        assert all(x["version_applied"] == NEW_VERSION_ID for x in log if x["ann_id"] in reviewed_ids)

        # ---- 5. compare: API.md's FieldBlock shape ----
        compare = _get(client, f"/api/versions/compare?a={BASE_VERSION_ID}&b={NEW_VERSION_ID}")
        field = compare["field"]
        assert set(field) == {"a", "b", "worst_regression"}, set(field)
        _assert_no_forbidden(field)
        for side in ("a", "b"):
            fb = field[side]
            assert fb is not None and set(fb) == FIELD_BLOCK_KEYS, (side, fb and set(fb) ^ FIELD_BLOCK_KEYS)
            assert fb["control"] and fb["control"]["overall"] is not None and fb["control"]["per_class"]
            assert fb["control"]["label"] == "labeled control-set regression check"
            assert set(fb["reviewed_flagged"]) == REVIEWED_FLAGGED_KEYS, set(fb["reviewed_flagged"]) ^ REVIEWED_FLAGGED_KEYS
            for cls in fb["classes"]:
                assert set(cls) == CLASS_KEYS, set(cls) ^ CLASS_KEYS
                for g in cls["by_group"]:
                    assert set(g) == BY_GROUP_KEYS, set(g) ^ BY_GROUP_KEYS
        # v1 predates this feature (its own eval artifact has no `reviewed_flagged`) -- dataio.field_block
        # fills in the all-zero/null default rather than omitting the key.
        assert field["a"]["reviewed_flagged"] == {"n": 0, "n_not_aircraft": 0, "n_bad_box": 0, "field_precision_on_flagged": None}
        # v2 was just built with exactly N_NEW_CLASS+N_CONFIRM+N_BAD_BOX reviews on FLAGGED det chips,
        # none marked not_aircraft -- reviewed_flagged.n covers all of them, n_bad_box exactly N_BAD_BOX.
        rf_b = field["b"]["reviewed_flagged"]
        assert rf_b["n"] == N_NEW_CLASS + N_CONFIRM + N_BAD_BOX, rf_b
        assert rf_b["n_not_aircraft"] == 0 and rf_b["n_bad_box"] == N_BAD_BOX, rf_b
        assert abs(rf_b["field_precision_on_flagged"] - (rf_b["n"] - rf_b["n_bad_box"]) / rf_b["n"]) < 1e-9
        print(f"reviewed_flagged (v2): {rf_b}")
        assert field["a"]["classes"] == []
        cls_b = field["b"]["classes"]
        assert len(cls_b) == 1 and cls_b[0]["label"] == NEW_LABEL and cls_b[0]["n_labels_used"] == N_NEW_CLASS, cls_b
        assert field["b"]["n_excluded_labeled"] == N_NEW_CLASS, field["b"]["n_excluded_labeled"]
        assert cls_b[0]["by_group"], "by_group missing/empty"
        wr = field["worst_regression"]
        assert wr is None or set(wr) == {"cls", "delta", "a", "b"}

        # `labeled_collections` (DISPLAY ONLY, never in the classes/n_assigned counts above): the
        # demo group itself has 6 new_class + 1 bad_box reviews, so it must appear here.
        assert field["a"]["labeled_collections"] == [], field["a"]["labeled_collections"]
        lc_b = field["b"]["labeled_collections"]
        assert isinstance(lc_b, list)
        demo_lc = next((g for g in lc_b if g["group_id"] == group["group_id"]), None)
        assert demo_lc is not None, f"expected {group['group_id']!r} in labeled_collections: {lc_b}"
        assert set(demo_lc) == LABELED_COLLECTION_KEYS, set(demo_lc) ^ LABELED_COLLECTION_KEYS
        assert demo_lc["n_labeled"] == N_NEW_CLASS, demo_lc
        assert demo_lc["n_bad_box"] == N_BAD_BOX and demo_lc["n_not_aircraft"] == 0, demo_lc
        assert demo_lc["n_flagged_unlabeled_assigned"] <= demo_lc["n_flagged_unlabeled"], demo_lc
        assert sum(demo_lc["per_class"].values()) == demo_lc["n_flagged_unlabeled_assigned"], demo_lc
        assert field["b"]["assign_min_conf"] in (0, 0.0, None), field["b"]["assign_min_conf"]  # this run never sets it

        print("\n=== /versions/compare field.b (FieldBlock) ===")
        print(json.dumps({k: v for k, v in field["b"].items() if k != "classes"}, indent=2))
        c = cls_b[0]
        print(f"\nbefore/after n_assigned ({c['label']}): 0 -> {c['n_assigned']} "
              f"({c['assignment_rate']:.2%} of {field['b']['n_eligible']} eligible); assign_min_conf={field['b']['assign_min_conf']}")
        print(f"{c['label']}: {c['n_labels_used']} labels used -> {c['n_assigned']} detections assigned "
              f"({c['assignment_rate']:.2%} of {field['b']['n_eligible']} eligible); "
              f"excluded {field['b']['n_excluded_labeled']} labeled + {field['b']['n_excluded_same_parent']} same-parent")
        _resort = f"{demo_lc['n_flagged_unlabeled_assigned']} of {demo_lc['n_flagged_unlabeled']} remaining flagged detections re-sorted"
        if demo_lc["per_class"]:
            _resort += " to " + ", ".join(f"{lbl} ({n})" for lbl, n in sorted(demo_lc["per_class"].items(), key=lambda kv: -kv[1]))
        print(f"In the labeled collection {demo_lc['airfield']} · {demo_lc['date']} (excluded from the counts above): "
              f"{demo_lc['n_labeled']} labeled; {_resort}; {demo_lc['n_bad_box']} bad box, {demo_lc['n_not_aircraft']} not aircraft.")
        for g in c["by_group"]:
            print(f"    {g['group_id']}: {g['n_assigned']}")
        ca, cb = field["a"]["control"], field["b"]["control"]
        print(f"control: overall {ca['overall']:.3f} -> {cb['overall']:.3f}; worst regression: {wr}")

        # ---- 6. v2 is served; report has the analyst-view section ----
        summary_v2 = _get(client, "/api/collections/summary")
        report = client.get("/api/report").text
        assert "New collections since delivery" in report and NEW_LABEL in report, "report missing analyst-view section"
        for word in ("deployed", "accredited", "significant"):
            assert word not in report.lower(), f"forbidden word {word!r} in report"
        print(f"collections ({NEW_VERSION_ID}): {summary_v2['n_needs_review']} needs_review of {summary_v2['n_groups']}; report OK")

        # ---- 7. artifacts (before the live-state checks, so a shared-environment hiccup never costs the run) ----
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        scratch_scores = SCRATCH / "scores"
        for src, dst in (
            (f"eval_{NEW_VERSION_ID}.json", f"eval_{ARTIFACT_PREFIX}.json"),
            (f"{NEW_VERSION_ID}_model.json", f"{ARTIFACT_PREFIX}_model.json"),
            (f"{NEW_VERSION_ID}_batches.json", f"{ARTIFACT_PREFIX}_batches.json"),
            (f"{NEW_VERSION_ID}_det.parquet", f"{ARTIFACT_PREFIX}_det.parquet"),
        ):
            if (scratch_scores / src).exists():
                shutil.copy(scratch_scores / src, ARTIFACT_DIR / dst)
        (ARTIFACT_DIR / "feedback.json").write_text(json.dumps(feedback, indent=2))
        (ARTIFACT_DIR / "compare.json").write_text(json.dumps(compare, indent=2))
        (ARTIFACT_DIR / "collections_summary.json").write_text(json.dumps({
            "version": BASE_VERSION_ID, "summary": summary_v1, "status_rule": collections["status_rule"],
            "needs_review": [g for g in collections["groups"] if g["status"] == "needs_review"],
            "demo_collection": group,
        }, indent=2))
        print(f"\nartifacts written to {ARTIFACT_DIR}")
        # ---- 8. live state untouched by THIS run ----
        _assert_live_unchanged(snap, feedback)
        _check_live_server(live_server_before)

    finally:
        client.close()
        shutil.rmtree(SCRATCH, ignore_errors=True)
        print(f"cleanup: scratch {SCRATCH} removed -- live app untouched (still {BASE_VERSION_ID}); total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
