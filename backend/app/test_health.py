"""Tests for backend/app/health.py -- GET /api/health ("Model health · retrain readiness").

Shape + forbidden-key checks run against the isolated fixture through the live route; every status
branch of every indicator is exercised through the pure functions with hand-built inputs (no fixture
rebuild needed). Imports test_api first so the fixture env vars are pinned before any app module
resolves its paths (same pattern as test_changes.py).

Run with: .venv/bin/python -m pytest backend/app -q
"""

from __future__ import annotations

import pytest

from app import test_api as _api  # noqa: F401  -- sets the fixture env vars at import; must come first
from app import health
from app.test_api import _assert_no_forbidden_keys, _get

INDICATOR_KEYS = {"key", "label", "value", "display", "status", "rule", "evidence"}
FORBIDDEN_WORDS = ("deployed", "accredited", "significant", "recall")


def test_health_route_shape_and_forbidden_keys():
    body = _get("/api/health").json()
    assert set(body) == {"version", "recommendation", "reasons", "indicators", "rules", "note"}
    assert body["recommendation"] in {"keep", "patch", "schedule_retrain"}
    assert body["note"] == "indicative; computed from field-available evidence and the labeled control set"
    assert isinstance(body["reasons"], list)
    assert [i["key"] for i in body["indicators"]] == ["novelty_pressure", "new_class_demand", "detector_quality_field", "patch_safety"]
    for ind in body["indicators"]:
        assert set(ind) == INDICATOR_KEYS
        assert ind["status"] in {"nominal", "watch", "retrain"}
        assert ind["rule"] and ind["evidence"] and ind["display"]
    assert len(body["rules"]) == 5
    _assert_no_forbidden_keys(body)
    # vocabulary: stated thresholds, indicative, never the banned words
    text = " ".join(body["rules"] + body["reasons"] + [i["evidence"] for i in body["indicators"]]).lower()
    for word in FORBIDDEN_WORDS:
        assert word not in text, word
    assert "10%" in body["rules"][0] and "25%" in body["rules"][0]
    assert "20 labels" in body["rules"][1] and "60%" in body["rules"][1]
    assert "40%" in body["rules"][2] and "25%" in body["rules"][2]
    assert "1pp" in body["rules"][3] and "3pp" in body["rules"][3]
    # recommendation is the stated function of the indicator statuses
    statuses = {i["status"] for i in body["indicators"]}
    expected = "schedule_retrain" if "retrain" in statuses else "patch" if "watch" in statuses else "keep"
    assert body["recommendation"] == expected
    assert len(body["reasons"]) == sum(1 for i in body["indicators"] if i["status"] != "nominal")


def _groups(n_review: int, n_total: int, recent_review: int = 0):
    """n_total groups dated so the LAST `RECENT_WINDOW` are the most recent; `n_review` of the older ones
    and `recent_review` of the recent ones are needs_review."""
    out = []
    for i in range(n_total):
        recent = i >= n_total - health.RECENT_WINDOW
        if recent:
            status = "needs_review" if (i - (n_total - health.RECENT_WINDOW)) < recent_review else "nominal"
        else:
            status = "needs_review" if i < n_review else "nominal"
        out.append({"status": status, "date": f"2020-01-{1 + i // 31:02d}" if i < 62 else f"2021-{1 + (i // 28) % 12:02d}-{1 + i % 28:02d}"})
    return out


def test_novelty_pressure_branches():
    assert health.novelty_pressure([])["status"] == "nominal"
    assert health.novelty_pressure(_groups(2, 100))["status"] == "nominal"        # 2% overall, 0% recent
    assert health.novelty_pressure(_groups(12, 100))["status"] == "watch"         # 12% overall
    assert health.novelty_pressure(_groups(30, 100))["status"] == "retrain"       # 30% overall
    # the recent window alone can raise it: 0 old, 4 of the last 30 = 13.3%
    ind = health.novelty_pressure(_groups(0, 100, recent_review=4))
    assert ind["status"] == "watch" and "most recent" in ind["evidence"]
    assert ind["value"] == pytest.approx(4 / 30)
    ind = health.novelty_pressure(_groups(0, 100, recent_review=8))
    assert ind["status"] == "retrain"


def _cls(**kw):
    base = {"label": "Military fighter", "n_labels": 3, "n_collections_labeled": 1, "n_assigned": 5,
            "n_verified": 0, "n_confirmed": 0, "n_confirmed_collections": 0, "verified_precision": None}
    base.update(kw)
    return base


def test_new_class_demand_branches():
    assert health.new_class_demand([])["status"] == "nominal"
    emerging = health.new_class_demand([_cls()])
    assert emerging["status"] == "nominal" and "emerging" in emerging["evidence"] and emerging["value"] == 0
    recurring = health.new_class_demand([_cls(n_collections_labeled=2)])
    assert recurring["status"] == "watch" and recurring["value"] == 1
    # recurring via verified-confirmed assignments across >= 2 collections
    via_conf = health.new_class_demand([_cls(n_verified=6, n_confirmed=5, n_confirmed_collections=2, verified_precision=5 / 6)])
    assert via_conf["status"] == "watch"
    # 5 confirmations in a single collection is NOT recurring
    one_coll = health.new_class_demand([_cls(n_verified=6, n_confirmed=5, n_confirmed_collections=1, verified_precision=5 / 6)])
    assert one_coll["status"] == "nominal"
    # retrain: recurring + labels >= 20
    assert health.new_class_demand([_cls(n_collections_labeled=3, n_labels=20)])["status"] == "retrain"
    # retrain: recurring + verified precision < 0.6 with >= 10 verified
    low = health.new_class_demand([_cls(n_collections_labeled=2, n_verified=10, n_confirmed=5, verified_precision=0.5)])
    assert low["status"] == "retrain"
    # same precision with < 10 verified stays watch
    few = health.new_class_demand([_cls(n_collections_labeled=2, n_verified=4, n_confirmed=2, verified_precision=0.5)])
    assert few["status"] == "watch"
    # emerging class alone never triggers retrain even with many labels
    assert health.new_class_demand([_cls(n_labels=50)])["status"] == "nominal"
    # worst class wins
    mixed = health.new_class_demand([_cls(), _cls(label="Tanker", n_collections_labeled=2, n_labels=25)])
    assert mixed["status"] == "retrain" and mixed["value"] == 1 and "Tanker" in mixed["evidence"]


def test_detector_quality_field_branches():
    insufficient = health.detector_quality_field({"n": 9, "n_not_aircraft": 9, "n_bad_box": 0, "field_precision_on_flagged": 0.0})
    assert insufficient["status"] == "nominal" and "insufficient" in insufficient["evidence"] and insufficient["value"] is None
    assert health.detector_quality_field(None)["status"] == "nominal"
    good = health.detector_quality_field({"n": 20, "n_not_aircraft": 2, "n_bad_box": 2, "field_precision_on_flagged": 0.8})
    assert good["status"] == "nominal" and good["value"] == pytest.approx(0.8)
    low_precision = health.detector_quality_field({"n": 20, "n_not_aircraft": 9, "n_bad_box": 0, "field_precision_on_flagged": 0.55})
    assert low_precision["status"] == "watch"
    bad_boxes = health.detector_quality_field({"n": 20, "n_not_aircraft": 0, "n_bad_box": 5, "field_precision_on_flagged": 0.75})
    assert bad_boxes["status"] == "watch" and "bad-box rate" in bad_boxes["evidence"]
    retrain = health.detector_quality_field({"n": 20, "n_not_aircraft": 10, "n_bad_box": 3, "field_precision_on_flagged": 0.35})
    assert retrain["status"] == "retrain"
    # precision recomputed when the block predates the key
    derived = health.detector_quality_field({"n": 10, "n_not_aircraft": 7, "n_bad_box": 0})
    assert derived["status"] == "retrain" and derived["value"] == pytest.approx(0.3)


def test_patch_safety_branches():
    assert health.patch_safety([])["status"] == "nominal"
    assert health.patch_safety([("v1", {"A": 0.9})])["status"] == "nominal"
    assert health.patch_safety([("v1", {"A": 0.9}), ("v2", {})])["status"] == "nominal"  # v2 has no control yet
    same = health.patch_safety([("v1", {"A": 0.9, "B": 0.8}), ("v2", {"A": 0.9, "B": 0.8})])
    assert same["status"] == "nominal" and same["value"] == 0.0
    small = health.patch_safety([("v1", {"A": 0.9, "B": 0.8}), ("v2", {"A": 0.895, "B": 0.81})])
    assert small["status"] == "nominal" and small["value"] == pytest.approx(0.5)
    watch = health.patch_safety([("v1", {"A": 0.9, "B": 0.8}), ("v2", {"A": 0.9, "B": 0.78})])
    assert watch["status"] == "watch" and "B" in watch["display"]
    retrain = health.patch_safety([("v1", {"A": 0.9, "B": 0.8}), ("v2", {"A": 0.85, "B": 0.8})])
    assert retrain["status"] == "retrain" and retrain["value"] == pytest.approx(5.0)
    # a step regression that v1 -> latest would hide (v2 dropped, v3 recovered) is still caught
    step = health.patch_safety([("v1", {"A": 0.9}), ("v2", {"A": 0.86}), ("v3", {"A": 0.9})])
    assert step["status"] == "retrain" and "v1 -> v2" in step["evidence"]
    # an improvement is never a regression
    up = health.patch_safety([("v1", {"A": 0.8}), ("v2", {"A": 0.9})])
    assert up["status"] == "nominal" and up["value"] == 0.0


def test_compute_health_recommendation_rules():
    nominal = health.compute_health("v2", _groups(0, 50), [], None, [("v1", {"A": 0.9}), ("v2", {"A": 0.9})])
    assert nominal["recommendation"] == "keep" and nominal["reasons"] == []
    assert all(i["status"] == "nominal" for i in nominal["indicators"])
    for ind in nominal["indicators"]:
        assert "_reason" not in ind

    patch = health.compute_health("v2", _groups(0, 50), [_cls(n_collections_labeled=2)], None, [])
    assert patch["recommendation"] == "patch" and len(patch["reasons"]) == 1 and "New-class demand" in patch["reasons"][0]

    retrain = health.compute_health("v2", _groups(20, 50), [_cls(n_collections_labeled=2)],
                                    {"n": 20, "n_not_aircraft": 15, "n_bad_box": 0}, [])
    assert retrain["recommendation"] == "schedule_retrain"
    assert len(retrain["reasons"]) == 3
    assert retrain["version"] == "v2"


def test_health_inputs_from_fixture():
    inputs = health.gather_inputs()
    assert set(inputs) == {"version", "groups", "classes", "reviewed_flagged", "controls"}
    assert inputs["groups"]  # the fixture builds det chips across every non-reference domain
    versions = [v for v, _ in inputs["controls"]]
    assert versions[:2] == ["v1", "v2"]
    assert inputs["controls"][0][1]  # v1's control per-class accuracies are read from eval_v1.json
    for c in inputs["classes"]:
        assert set(c) == {"label", "n_labels", "n_collections_labeled", "n_assigned", "n_verified", "n_confirmed",
                          "n_confirmed_collections", "verified_precision"}
        assert c["n_collections_labeled"] <= c["n_labels"]


def test_report_has_model_health_section():
    # Built directly for v2: test_api's adapt tests (run earlier in the session) leave a stubbed 'v3'
    # with no artifacts as current_version(), so the /api/report route itself can't be used here.
    from app import report
    text = report.build_report("v2")
    assert "### Model health · retrain readiness (indicative)" in text
    idx_health = text.index("Model health · retrain readiness")
    idx_analyst = text.index("## New collections since delivery (analyst view)")
    assert idx_health > idx_analyst  # top of the analyst section
    assert "Rules (as applied):" in text
    assert "Novelty pressure" in text and "Patch safety" in text
