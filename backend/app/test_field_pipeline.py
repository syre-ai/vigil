"""Unit tests for the field-mode PIPELINE side (backend/vigil/{pipeline,field,evaluate,grouping}.py) --
the parts /adapt drives -- run against the same isolated fixture copy backend/app/test_api.py sets up
(importing it first pins VIGIL_DATA_ROOT/VIGIL_SCORES_DIR/VIGIL_DB_PATH to the temp copy before any
vigil module resolves its paths). Kept in its own module because importing vigil.pipeline pulls in
umap/torch (~10 s) -- the API tests stay fast.

Run with: .venv/bin/python -m pytest backend/app -q
"""

from __future__ import annotations

import pandas as pd
import pytest

from app import test_api as _api  # noqa: F401  -- sets the fixture env vars at import; must come first

from vigil import evaluate as evaluate_mod  # noqa: E402
from vigil import field as field_mod  # noqa: E402
from vigil import grouping  # noqa: E402
from vigil import pipeline  # noqa: E402


def test_grouping_slug_and_group_id_match_collections_api():
    """One group_id implementation for both the API (/collections) and the pipeline (by_group)."""
    assert grouping.slug("Some Airfield, TX") == "some-airfield-tx"
    assert grouping.slug("  ") == "unknown" and grouping.slug(None) == "unknown"
    assert grouping.group_id("Tirana International Airport", "2019-02-07") == "tirana-international-airport_2019-02-07"
    body = _api._get("/api/collections").json()
    for g in body["groups"][:20]:
        assert g["group_id"].startswith(grouping.group_id(g["airfield"], g["date"]))  # -2/-3 suffix only on slug collisions
    assert body["grouping"] == grouping.GROUPING


def test_apply_feedback_is_source_aware_and_not_aircraft_is_never_a_training_negative():
    """vigil.pipeline._apply_feedback: `new_class` ids go to used_in_fit_{gt,det} by source (both feed
    one analyst mean); `not_aircraft` on a det chip lands in exclude_det ONLY (dropped from det
    scoring on the next build) and never in new_classes/used_in_fit; `not_aircraft` on a GT chip drops
    it from GT scoring; confirm/correct/bad_box only count. `bad_box` (localization error -- fused or
    truncated aircraft) is like confirm/correct: counted, but never in new_classes/used_in_fit/
    exclude_gt/exclude_det -- its chip stays scored (never dropped), unlike not_aircraft's."""
    chips = pd.DataFrame({"ann_id": [1, 2, 3, 4]})
    fb = pipeline._apply_feedback(chips, [
        {"source": "det", "ann_id": 5, "action": "not_aircraft"},
        {"source": "gt", "ann_id": 2, "action": "not_aircraft"},
        {"source": "det", "ann_id": 7, "action": "new_class", "label": "Military fighter"},
        {"source": "gt", "ann_id": 3, "action": "new_class", "label": "Military fighter"},
        {"source": "det", "ann_id": 8, "action": "confirm"},
        {"ann_id": 1, "action": "correct", "label": "Small Civil"},  # source omitted -> gt
        {"source": "det", "ann_id": 9, "action": "skip"},
        {"source": "det", "ann_id": 10, "action": "bad_box"},
        {"ann_id": 4, "action": "bad_box"},  # source omitted -> gt; chip 4 must stay in fb["chips"]
    ])
    assert fb["new_classes"] == {"Military fighter": [("det", 7), ("gt", 3)]}
    assert fb["used_in_fit_det"] == {7} and fb["used_in_fit_gt"] == {3}
    assert fb["exclude_det"] == {5}
    assert 5 not in fb["used_in_fit_det"] and 5 not in {i for ids in fb["new_classes"].values() for _, i in ids}
    assert 10 not in fb["used_in_fit_det"] and 4 not in fb["used_in_fit_gt"]  # bad_box never fits/excludes
    assert fb["chips"]["ann_id"].tolist() == [1, 3, 4]  # GT not_aircraft (2) dropped; GT bad_box (4) stays scored
    assert fb["counts"] == {"n_confirm": 1, "n_new_class": 2, "n_not_aircraft": 2, "n_correct": 1, "n_bad_box": 2}


def test_used_in_fit_parent_lookups_tolerate_older_parquet_shapes(tmp_path, monkeypatch):
    """Regression: gt_used_in_fit_parents/det_used_in_fit_parents used `read_parquet(columns=[...])`,
    which raises ArrowInvalid on a parquet without `used_in_fit` (any pre-adaptation version). They
    must return an empty set instead, so vigil.field never fails a build over an old artifact."""
    monkeypatch.setattr(evaluate_mod, "SCORES_DIR", tmp_path)
    pd.DataFrame({"ann_id": [1], "tile_id": [1]}).to_parquet(tmp_path / "old.parquet", index=False)
    pd.DataFrame({"det_id": [1], "parent_id": ["p"]}).to_parquet(tmp_path / "old_det.parquet", index=False)
    assert evaluate_mod.gt_used_in_fit_parents("old") == set()
    assert evaluate_mod.det_used_in_fit_parents("old") == set()
    assert evaluate_mod.labeled_parent_ids("missing_version") == set()
    pd.DataFrame({"det_id": [1, 2], "parent_id": ["p1", "p2"], "used_in_fit": [True, False]}).to_parquet(
        tmp_path / "new_det.parquet", index=False
    )
    assert evaluate_mod.det_used_in_fit_parents("new") == {"p1"}


def test_compute_field_shape_and_exclusions_on_fixture():
    """vigil.field.compute_field on the fixture's v2: `field` carries n_detections/n_excluded_labeled/
    n_excluded_same_parent/n_eligible/classes(+by_group)/by_domain/note over NEW-collection det chips
    only (never ref_holdout, never the reference domains); `control` restates the role-head numbers
    under the "labeled control-set regression check" label."""
    control_in = {"accuracy_ref_holdout": 0.9, "per_class": {"Small Civil": {"accuracy": 0.9, "n": 10}}}
    result = field_mod.compute_field("v2", control_role_head=control_in)
    f, c = result["field"], result["control"]
    assert set(f) == {
        "domains", "n_detections", "n_excluded_labeled", "n_excluded_same_parent", "n_eligible",
        "classes", "by_domain", "reviewed_flagged", "labeled_collections", "assign_min_conf", "note",
    }
    # fixture v2 has no det-chip `new_class` reviews recorded against it -> no labeled collection.
    assert f["labeled_collections"] == []
    assert f["assign_min_conf"] is None  # fixture's hand-built model.json predates assign_min_conf
    assert "ref_holdout" not in f["domains"] and set(f["by_domain"]) <= set(f["domains"])
    assert f["n_detections"] == sum(b["n_detections"] for b in f["by_domain"].values())
    assert f["n_eligible"] == f["n_detections"] - f["n_excluded_labeled"] - f["n_excluded_same_parent"]
    det = pd.read_parquet(_api._TEST_DATA_ROOT / "scores" / "v2_det.parquet")
    assert f["n_detections"] == int(det["batch_id"].isin(f["domains"]).sum())
    for cls in f["classes"]:
        assert set(cls) == {"label", "n_labels_used", "n_assigned", "assignment_rate", "by_group"}
        assert isinstance(cls["by_group"], list)
    assert f["note"] == field_mod.NOTE_DESCRIPTIVE
    assert c == {"label": "labeled control-set regression check", "accuracy_ref_holdout": 0.9, "per_class": control_in["per_class"]}

    # `reviewed_flagged` -- built from the LIVE reviews table (test_api.py's suite has, by this
    # point, recorded real source='det' reviews, incl. at least one bad_box) -- shape + internal
    # consistency only; exact counts are covered in isolation by
    # test_reviewed_flagged_block_current_and_historical_lookup below.
    rf = f["reviewed_flagged"]
    assert set(rf) == {"n", "n_not_aircraft", "n_bad_box", "field_precision_on_flagged"}
    assert rf["n"] >= rf["n_not_aircraft"] + rf["n_bad_box"] >= 0
    if rf["n"]:
        assert rf["field_precision_on_flagged"] == pytest.approx(
            (rf["n"] - rf["n_not_aircraft"] - rf["n_bad_box"]) / rf["n"]
        )
    else:
        assert rf["field_precision_on_flagged"] is None


def test_field_by_group_uses_collections_group_ids_and_same_parent_exclusion():
    """_by_group counts eligible det chips per airfield x date with /collections' own group_id, top
    BY_GROUP_TOP_N by n_assigned; _domain_block excludes labeled chips AND every other chip from their
    parent images (parents of labeled det AND gt chips, pooled by the caller) before counting."""
    tiles = pd.read_parquet(_api._TEST_DATA_ROOT / "rareplanes" / "tiles.parquet")
    t = tiles.iloc[:3]
    eligible = pd.DataFrame({
        "tile_id": [t.tile_id.iloc[0]] * 3 + [t.tile_id.iloc[1]] * 2 + [t.tile_id.iloc[2]],
        "role_pred": ["X", "X", "Y", "X", "X", "X"],
    })
    groups = field_mod._by_group(eligible, "X")
    assert [g["n_assigned"] for g in groups] == sorted([g["n_assigned"] for g in groups], reverse=True)
    assert sum(g["n_assigned"] for g in groups) == 5
    for g in groups:
        assert set(g) == {"group_id", "airfield", "date", "n_assigned"}
        assert g["group_id"] == grouping.group_id(g["airfield"], g["date"])
    assert field_mod._by_group(eligible, "Z") == [] and field_mod._by_group(eligible.iloc[0:0], "X") == []

    df = pd.DataFrame({
        "tile_id": [1, 1, 2, 3, 4],
        "parent_id": ["pA", "pA", "pA", "pB", "pC"],
        "used_in_fit": [True, False, False, False, False],
        "role_pred": ["X", "X", "X", "X", "X"],
    })
    block = field_mod._domain_block(df, labeled_parents={"pA", "pB"}, analyst_labels=["X"])
    assert (block["n_detections"], block["n_excluded_labeled"], block["n_excluded_same_parent"], block["n_eligible"]) == (5, 1, 3, 1)
    assert block["classes"] == [{"label": "X", "n_assigned": 1, "assignment_rate": 1.0}]
    assert block.pop("eligible")["parent_id"].tolist() == ["pC"]


@pytest.mark.parametrize("n,k,p_ref,expected", [
    (510, 21, 0.0314, "nominal"),        # big civil airport: 4.1% vs 3.1% is not an excess
    (66, 18, 0.0314, "needs_review"),    # small collection with a real excess
    (19, 19, 0.0314, "insufficient_data"),
])
def test_status_rule_examples(n, k, p_ref, expected):
    from app.collections import _status_for

    assert _status_for(n, k, p_ref)["status"] == expected


def test_assign_roles_gates_analyst_classes_by_novelty_and_nearest_exemplar_margin():
    """vigil.pipeline._assign_roles (shared by the full and incremental builds), nearest-exemplar rule
    (ADAPTATION_RULE): a chip may only join an analyst class if (a) it is flagged novel AND (b) its
    closest labeled exemplar of that class is nearer (cosine) than its own 1-NN distance to the
    labeled reference set (`ref_1nn_dist`) -- never a class MEAN. Everything else keeps the civil
    prototype head's prediction. role_conf stays a probability on both sides; analyst_sim/
    analyst_margin are 0 for every unclaimed chip."""
    import numpy as np
    from vigil.proto import ProtoHead

    rng = np.random.default_rng(0)
    civ = rng.normal(size=(3, 8)); civ /= np.linalg.norm(civ, axis=1, keepdims=True)
    proto = ProtoHead()
    proto.fit(np.repeat(civ, 4, axis=0), [c for c in ("Small Civil", "Medium Civil", "Large Civil") for _ in range(4)])
    analyst_dir = rng.normal(size=8); analyst_dir /= np.linalg.norm(analyst_dir)
    exemplars = {"Military fighter": analyst_dir[None, :]}  # ONE labeled exemplar, not a class mean
    X = np.stack([analyst_dir, analyst_dir, civ[0]])  # two chips identical to the exemplar, one civil (itself a reference chip)
    is_novel = np.array([True, False, True])
    # X[0]/X[1]: distance-to-exemplar is 0 (identical direction) -- any positive ref_1nn_dist clears
    # the margin test for X[0] (novel). X[2] is itself a reference vector: its distance to the
    # (unrelated, random) analyst exemplar is close to 1, while its own ref_1nn_dist is ~0 -- so it
    # must NOT be claimed even though it's flagged novel here.
    ref_1nn_dist = np.array([0.5, 0.5, 0.0])

    out = pipeline._assign_roles(X, is_novel, proto, exemplars, ref_1nn_dist)
    assert list(out["role_pred"]) == ["Military fighter", proto.predict(X[1:2])[0][0], proto.predict(X[2:3])[0][0]]
    assert 0.0 < out["role_conf"][0] <= 1.0
    assert out["analyst_sim"][0] == pytest.approx(1.0, abs=1e-5)
    assert out["analyst_margin"][0] == pytest.approx(ref_1nn_dist[0], abs=1e-5)  # exemplar dist ~0 -> margin ~= ref_1nn_dist
    assert out["analyst_sim"][1] == 0.0 and out["analyst_sim"][2] == 0.0
    assert out["analyst_margin"][1] == 0.0 and out["analyst_margin"][2] == 0.0

    none = pipeline._assign_roles(X, np.array([True, True, True]), proto, {}, ref_1nn_dist)
    assert set(none["role_pred"]) <= {"Small Civil", "Medium Civil", "Large Civil"}
    assert (none["analyst_sim"] == 0).all() and (none["analyst_margin"] == 0).all()


def test_assign_roles_optional_conf_gate_is_det_only_and_off_by_default():
    """`assign_min_conf` (vigil.pipeline.ASSIGN_MIN_CONF_DEFAULT = 0, off) is an OPTIONAL additional
    gate on top of the nearest-exemplar rule, meaningful for detector chips only: a chip that already
    passes the exemplar rule is claimed regardless of confidence when the gate is off (0) or `det_conf`
    is None (GT chips), but is refused when its own `det_conf` falls below a positive `assign_min_conf`
    -- e.g. a low-confidence (0.3) novel detector chip nearest an analyst exemplar is assigned at
    assign_min_conf=0 but not at 0.5."""
    import numpy as np
    from vigil.proto import ProtoHead

    rng = np.random.default_rng(1)
    civ = rng.normal(size=(3, 8)); civ /= np.linalg.norm(civ, axis=1, keepdims=True)
    proto = ProtoHead()
    proto.fit(np.repeat(civ, 4, axis=0), [c for c in ("Small Civil", "Medium Civil", "Large Civil") for _ in range(4)])
    analyst_dir = rng.normal(size=8); analyst_dir /= np.linalg.norm(analyst_dir)
    exemplars = {"Military fighter": analyst_dir[None, :]}
    X = analyst_dir[None, :]  # one novel det chip, identical to the labeled exemplar
    is_novel = np.array([True])
    ref_1nn_dist = np.array([0.5])
    low_conf = np.array([0.3])

    off = pipeline._assign_roles(X, is_novel, proto, exemplars, ref_1nn_dist, det_conf=low_conf, assign_min_conf=0.0)
    assert off["role_pred"][0] == "Military fighter"

    gated = pipeline._assign_roles(X, is_novel, proto, exemplars, ref_1nn_dist, det_conf=low_conf, assign_min_conf=0.5)
    assert gated["role_pred"][0] != "Military fighter"
    assert gated["analyst_sim"][0] == 0.0

    # a GT chip (det_conf=None) is never gated on assign_min_conf, even when it's set high.
    gt = pipeline._assign_roles(X, is_novel, proto, exemplars, ref_1nn_dist, det_conf=None, assign_min_conf=0.5)
    assert gt["role_pred"][0] == "Military fighter"

    # the gate only kicks in when a det chip's confidence clears it.
    high_conf = np.array([0.6])
    passes = pipeline._assign_roles(X, is_novel, proto, exemplars, ref_1nn_dist, det_conf=high_conf, assign_min_conf=0.5)
    assert passes["role_pred"][0] == "Military fighter"


def test_find_base_version_and_det_block_on_fixture():
    """The incremental build's base = the most recent built version whose scores/batches/model files exist
    (fixture: v2, or v1 when building 'v2' itself); a never-built id has none of the required files."""
    assert pipeline._find_base_version("v3") == "v2"
    assert pipeline._find_base_version("v2") == "v1"
    det = pd.read_parquet(_api._TEST_DATA_ROOT / "scores" / "v2_det.parquet")
    block = pipeline._det_block(det)
    assert "reference" in block and block["reference"]["n_det_chips_ref_holdout"] == block["ref_holdout"]["n_det_chips"]
    assert block["ref_holdout"]["flag_rate"] == pytest.approx(det[det["batch_id"] == "ref_holdout"]["is_novel"].mean())
    assert set(pipeline.REUSED_SCORE_COLS) <= set(det.columns)


def test_reviewed_flagged_block_empty_when_no_det_reviews(monkeypatch):
    monkeypatch.setattr(field_mod, "_det_reviews", lambda: [])
    assert field_mod._reviewed_flagged_block(None) == {
        "n": 0, "n_not_aircraft": 0, "n_bad_box": 0, "field_precision_on_flagged": None,
    }


def test_reviewed_flagged_block_current_and_historical_lookup(tmp_path, monkeypatch):
    """`_reviewed_flagged_block`: `n` counts det reviews (any action) on chips that WERE flagged
    (is_novel) -- looked up in the CURRENT det scores first, falling back to the review's own
    `predicted_version`'s det parquet for a chip since excluded from the current one (not_aircraft
    drops a chip from det scoring going forward -- see vigil.pipeline's `exclude_det`). A chip
    determined NOT flagged, or with no determinable flagged status at all (no historical parquet
    either), is excluded from `n`. `field_precision_on_flagged` = (n - n_not_aircraft - n_bad_box) / n.
    """
    monkeypatch.setattr(field_mod, "SCORES_DIR", tmp_path)
    # v_old's det parquet: det_id 1 was flagged when reviewed there (now excluded from "current");
    # det_id 2 was NOT flagged there.
    pd.DataFrame({"det_id": [1, 2], "is_novel": [True, False]}).to_parquet(tmp_path / "v_old_det.parquet", index=False)

    reviews = [
        {"ann_id": 10, "action": "confirm", "predicted_version": "v_new"},       # flagged, neither
        {"ann_id": 11, "action": "not_aircraft", "predicted_version": "v_new"},  # flagged
        {"ann_id": 12, "action": "bad_box", "predicted_version": "v_new"},       # flagged
        {"ann_id": 13, "action": "not_aircraft", "predicted_version": "v_new"},  # NOT flagged -> excluded from n
        {"ann_id": 1, "action": "not_aircraft", "predicted_version": "v_old"},   # historical lookup: flagged
        {"ann_id": 2, "action": "bad_box", "predicted_version": "v_old"},        # historical lookup: not flagged -> excluded
        {"ann_id": 999, "action": "confirm", "predicted_version": "v_missing"},  # no det parquet at all -> excluded
        {"ann_id": 998, "action": "skip"},                                      # no predicted_version at all -> excluded
    ]
    monkeypatch.setattr(field_mod, "_det_reviews", lambda: reviews)

    det_current = pd.DataFrame({"det_id": [10, 11, 12, 13], "is_novel": [True, True, True, False]})
    result = field_mod._reviewed_flagged_block(det_current)
    assert result == {
        "n": 4,               # 10, 11, 12 (current) + 1 (historical)
        "n_not_aircraft": 2,  # 11, and historical 1
        "n_bad_box": 1,       # 12
        "field_precision_on_flagged": pytest.approx((4 - 2 - 1) / 4),
    }
