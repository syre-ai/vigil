"""Tests for backend/app/changes.py -- GET /api/versions/changes* ("Changed in this version"):
detector chips whose role_pred differs between v1 and v2's own `{version}_det.parquet`
(data/scores_fixture builds both -- see build_fixture.py's `_det_scores_for_version`, which
resolves a handful of previously-generic-civil det chips to the new analyst class in v2).

Reuses backend/app/test_api.py's already-isolated fixture copy + FastAPI TestClient (importing it
first pins VIGIL_DATA_ROOT/VIGIL_SCORES_DIR/VIGIL_DB_PATH before any vigil/app module resolves its
paths -- same pattern as test_field_pipeline.py). Because this file runs in the same process AFTER
test_api.py's own tests -- including the adapt tests deliberately placed last in that file, which
leave a stubbed 'v3' (no real score files) as `current_version()` for the rest of the session --
every test below passes explicit `a=v1&b=v2` query params rather than relying on the current
version, and any review seeded to test `predicted_version`-scoped summary counting is written
directly via `app.db.upsert_review` (never the live POST endpoint, whose `predicted_version`
snapshot follows whatever `current_version()` is at POST time -- already 'v3' here).

Run with: .venv/bin/python -m pytest backend/app -q
"""

from __future__ import annotations

import pandas as pd

from app import test_api as _api  # noqa: F401  -- sets the fixture env vars at import; must come first
from app import db as db_mod
from app.test_api import FORBIDDEN_KEYS, _assert_no_forbidden_keys, _get, client  # noqa: F401


def test_changes_basic_shape_and_diff_against_fixture():
    body = _get("/api/versions/changes?a=v1&b=v2").json()
    assert body["a"] == "v1" and body["b"] == "v2"
    assert body["total"] >= 1, "fixture v1/v2 det scores must differ for at least one chip"
    assert isinstance(body["by_group"], list) and body["by_group"]
    assert isinstance(body["by_class"], list) and body["by_class"]
    assert len(body["items"]) == min(body["total"], 48)
    _assert_no_forbidden_keys(body)

    for item in body["items"]:
        assert item["source"] == "det"
        assert item["det_id"] == item["ann_id"]
        assert item["role_pred_a"] != item["role_pred_b"]
        assert item["role_pred"] == item["role_pred_b"]
        assert "analyst_sim" in item
        assert item["group_id"]
        assert item["airfield"] and item["date"]
        assert isinstance(item["same_parent"], bool)
        assert "review" in item  # present (possibly null) on every item

    by_group_total = sum(g["n_changed"] for g in body["by_group"])
    assert by_group_total == body["total"]
    by_class_total = sum(c["n"] for c in body["by_class"])
    assert by_class_total == body["total"]


def test_changes_sort_default_is_analyst_sim_desc():
    body = _get("/api/versions/changes?a=v1&b=v2").json()
    sims = [item["analyst_sim"] for item in body["items"]]
    assert sims == sorted(sims, reverse=True)


def test_changes_sort_novelty_and_conf():
    for sort_key, field in (("novelty", "novelty"), ("conf", "conf")):
        body = _get(f"/api/versions/changes?a=v1&b=v2&sort={sort_key}").json()
        values = [item[field] for item in body["items"] if item[field] is not None]
        assert values == sorted(values, reverse=True), f"sort={sort_key} not applied"


def test_changes_group_id_filter():
    all_body = _get("/api/versions/changes?a=v1&b=v2").json()
    top_group = all_body["by_group"][0]
    filtered = _get(f"/api/versions/changes?a=v1&b=v2&group_id={top_group['group_id']}").json()
    assert filtered["total"] == top_group["n_changed"]
    assert all(item["group_id"] == top_group["group_id"] for item in filtered["items"])
    # facets stay computed over the FULL changed set, independent of the group_id filter
    assert filtered["by_group"] == all_body["by_group"]


def test_changes_pagination_is_disjoint():
    page1 = _get("/api/versions/changes?a=v1&b=v2&limit=2&offset=0").json()
    page2 = _get("/api/versions/changes?a=v1&b=v2&limit=2&offset=2").json()
    ids1 = {i["det_id"] for i in page1["items"]}
    ids2 = {i["det_id"] for i in page2["items"]}
    assert len(page1["items"]) <= 2
    assert ids1.isdisjoint(ids2)


def test_changes_no_diff_when_a_equals_b():
    body = _get("/api/versions/changes?a=v2&b=v2").json()
    assert body == {"a": "v2", "b": "v2", "total": 0, "by_group": [], "by_class": [], "items": []}


def test_changes_unknown_version_degrades_to_empty_not_error():
    body = _get("/api/versions/changes?a=v1&b=does-not-exist").json()
    assert body["total"] == 0
    assert body["items"] == []
    summary = _get("/api/versions/changes/summary?a=v1&b=does-not-exist").json()
    assert summary["total"] == 0
    assert summary["verified_precision"] is None


def test_changes_only_unreviewed_filter():
    body = _get("/api/versions/changes?a=v1&b=v2").json()
    target = next(i for i in body["items"] if i["review"] is None)
    det_id = target["det_id"]

    r = client.post(f"/api/review/{det_id}?source=det", json={"action": "confirm"})
    assert r.status_code == 200, r.text

    unreviewed = _get("/api/versions/changes?a=v1&b=v2&only_unreviewed=true").json()
    assert det_id not in {i["det_id"] for i in unreviewed["items"]}
    assert unreviewed["total"] == body["total"] - 1

    all_again = _get("/api/versions/changes?a=v1&b=v2").json()
    reviewed_item = next(i for i in all_again["items"] if i["det_id"] == det_id)
    assert reviewed_item["review"] is not None and reviewed_item["review"]["action"] == "confirm"


def test_changes_summary_counts_predicted_version_scoped_reviews():
    body = _get("/api/versions/changes?a=v1&b=v2").json()
    unreviewed_ids = [i["det_id"] for i in body["items"] if i["review"] is None]
    assert len(unreviewed_ids) >= 4, "fixture must leave >=4 unreviewed changed det chips for this test"
    ids = unreviewed_ids[:4]

    before = _get("/api/versions/changes/summary?a=v1&b=v2").json()
    assert before["total"] == body["total"]

    # Seeded directly (bypassing the live POST endpoint, whose predicted_version snapshot follows
    # current_version() -- already 'v3' by the time this file runs; see module docstring) against
    # KNOWN, previously-unreviewed changed det_ids, one per action the summary must count.
    seed = [(ids[0], "confirm", None), (ids[1], "correct", "Military Fighter/Interceptor"),
            (ids[2], "not_aircraft", None), (ids[3], "bad_box", None)]
    for det_id, action, label in seed:
        db_mod.upsert_review(
            det_id, action, label, None,
            predicted_role="Small Civil Transport/Utility", predicted_version="v2", source="det",
        )

    after = _get("/api/versions/changes/summary?a=v1&b=v2").json()
    assert after["n_confirmed"] == before["n_confirmed"] + 1
    assert after["n_corrected"] == before["n_corrected"] + 1
    assert after["n_not_aircraft"] == before["n_not_aircraft"] + 1
    assert after["n_bad_box"] == before["n_bad_box"] + 1
    assert after["n_reviewed"] == before["n_reviewed"] + 4
    assert after["total"] == before["total"]
    denom = after["n_confirmed"] + after["n_corrected"] + after["n_not_aircraft"] + after["n_bad_box"]
    assert after["verified_precision"] == after["n_confirmed"] / denom


def test_changes_excludes_used_in_fit_and_flags_same_parent(monkeypatch):
    """The fixture never marks a det chip `used_in_fit=True` (build_fixture.py's
    `_det_scores_for_version` hardcodes it False), so the exclusion/same-parent-flag logic can't be
    exercised against real fixture data -- verified here with fully controlled synthetic tables
    instead, monkeypatching only `collections_api._new_collections_table` (the one real-I/O call
    `_changed_table` makes) and `_labeled_parent_ids`.
    """
    from app import changes as changes_mod

    def fake_new_collections_table(version: str):
        base = dict(
            det_id=[1, 2, 3, 4], tile_id=[10, 10, 20, 20], parent_id=[100, 100, 200, 200],
            role_conf=[0.9] * 4, novelty=[0.1] * 4, is_novel=[False] * 4, knn_dist=[0.1] * 4,
            analyst_sim=[0.5] * 4, conf=[0.8] * 4, x1=[0.0] * 4, y1=[0.0] * 4, w=[10.0] * 4, h=[10.0] * 4,
            airfield=["Test Field"] * 4, country=["USA"] * 4, weather=["Clear Skies"] * 4,
            off_nadir=[5.0] * 4, gsd=[0.3] * 4, date=["2026-01-01"] * 4,
            group_id=["test-field_2026-01-01"] * 4,
        )
        if version == "fake_a":
            df = pd.DataFrame({**base, "role_pred": [1.0, 1.0, 1.0, 1.0], "used_in_fit": [False] * 4})
        else:
            # det_id=1 changed AND is the chip `b` used to fit the new class -> must be excluded.
            # det_id=2 changed, shares det_id=1's parent (100) -> kept, flagged same_parent.
            # det_id=3,4 unchanged -> never enter the diff.
            df = pd.DataFrame({
                **base,
                "role_pred": [6.0, 6.0, 1.0, 1.0],
                "used_in_fit": [True, False, False, False],
            })
        return df, None

    monkeypatch.setattr(changes_mod.collections_api, "_new_collections_table", fake_new_collections_table)
    monkeypatch.setattr(changes_mod, "_scored_det_ids", lambda version: {1, 2, 3, 4})
    monkeypatch.setattr(changes_mod, "_labeled_parent_ids", lambda version: {100})

    df = changes_mod._changed_table("fake_a", "fake_b")
    assert set(df["det_id"]) == {2}
    assert bool(df.iloc[0]["_same_parent"]) is True
