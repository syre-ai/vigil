"""Smoke tests for the Vigil API against the synthetic fixture.

Run with: .venv/bin/python -m pytest backend/app -q

Every GET endpoint in backend/API.md is hit at least once, plus one
review POST (and one /adapt POST, expected to 501 since backend/vigil
does not ship a `pipeline.build_version` yet). All state (data + sqlite)
is isolated under a temp copy of data/scores_fixture/ so running the
suite never touches the real data/ directory.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_SRC = REPO_ROOT / "data" / "scores_fixture"

_TMP_ROOT = Path(tempfile.mkdtemp(prefix="vigil_test_"))
_TEST_DATA_ROOT = _TMP_ROOT / "data"
shutil.copytree(FIXTURE_SRC, _TEST_DATA_ROOT)

os.environ["VIGIL_DATA_ROOT"] = str(_TEST_DATA_ROOT)
os.environ["VIGIL_SCORES_DIR"] = str(_TEST_DATA_ROOT / "scores")
os.environ["VIGIL_DB_PATH"] = str(_TMP_ROOT / "vigil_test.db")

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

client = TestClient(app)


def _get(path: str, **kw):
    r = client.get(path, **kw)
    assert r.status_code == 200, f"GET {path} -> {r.status_code}: {r.text}"
    return r


def test_summary():
    r = _get("/api/summary")
    body = r.json()
    assert body["current_version"] == "v2"
    assert isinstance(body["batches"], list) and len(body["batches"]) == 7
    assert set(body["totals"]) == {"chips", "novel", "reviewed"}
    assert len(body["versions"]) == 2


def test_summary_detector_card():
    r = _get("/api/summary")
    model = r.json()["model"]
    assert "oracle localization" in model["localization"]
    assert "IoU" in model["localization"]
    card = model["detector_card"]
    assert card is not None
    for key in ("name", "trained_on", "val_split", "val_map50", "val_map50_95",
                "val_precision", "val_recall", "conf_threshold", "iou_threshold", "note"):
        assert key in card
    assert model["detector"].startswith(card["name"])


def test_summary_model_selection_passthrough():
    r = _get("/api/summary")
    model = r.json()["model"]
    for key in ("calibration_check", "model_selection_note", "reference_definition"):
        assert key in model
    # fixture's v1/v2 model.json sets all three -- exercised end-to-end, not just null-degraded.
    assert model["calibration_check"]["n_parents"] == 3
    assert "AUROC" in model["model_selection_note"]
    assert "ref_train" in model["reference_definition"]
    assert "trained on" in model["detector"]


def test_batches_list():
    r = _get("/api/batches")
    batches = r.json()
    ids = [b["id"] for b in batches]
    assert ids == ["ref_holdout", "off_nadir", "foreign", "haze", "snow", "military", "synthetic"]
    for b in batches:
        assert "drift" in b and "statistic" in b["drift"]


def test_batches_detection():
    r = _get("/api/batches")
    by_id = {b["id"]: b for b in r.json()}
    # the fixture's detection_metrics.json (built alongside data/scores_fixture)
    # deliberately omits "synthetic" -- mirrors the real pipeline, which hasn't
    # evaluated the detector against the (not-yet-built) synthetic batch either.
    assert by_id["synthetic"]["detection"] is None
    for bid in ("ref_holdout", "off_nadir", "foreign", "haze", "snow", "military"):
        det = by_id[bid]["detection"]
        assert det is not None, bid
        assert set(det) == {"recall", "precision", "n_gt", "n_det"}
        assert 0.0 <= det["recall"] <= 1.0
        assert 0.0 <= det["precision"] <= 1.0
        assert det["n_gt"] > 0


def test_batch_detail():
    r = _get("/api/batches/military")
    body = r.json()
    assert body["id"] == "military"
    assert "novelty_hist" in body and "bins" in body["novelty_hist"]
    assert "ref_novelty_hist" in body
    assert len(body["top_novel"]) <= 10
    for chip in body["top_novel"]:
        assert chip["batch_id"] == "military"


def test_batch_detail_novelty_histograms_share_edges():
    """novelty_hist/ref_novelty_hist must use the SAME 30 equal-width bins,
    spanning [min, max] of novelty pooled over (reference U batch) with a
    small margin -- not the old fixed [0,1] 10-bin scheme, which made the
    chart useless once real novelty scores turned out to cluster in a
    narrow band (see API.md).
    """
    r = _get("/api/batches/military")
    body = r.json()
    nov, ref = body["novelty_hist"], body["ref_novelty_hist"]
    for hist in (nov, ref):
        assert len(hist["bins"]) == 31
        assert len(hist["counts"]) == 30
        assert len(hist["density"]) == 30
        assert hist["bins"] == sorted(hist["bins"])
        total = sum(hist["counts"])
        if total:
            assert sum(hist["density"]) == pytest.approx(1.0, abs=1e-6)
            assert all(d == pytest.approx(c / total, abs=1e-5) for c, d in zip(hist["counts"], hist["density"]))
    assert nov["bins"] == ref["bins"]  # shared edges

    # Fixture batches don't all sit in [0, 1] narrowly, but the edges must
    # still be derived from the actual pooled data, not hard-coded to [0, 1].
    chips = _get("/api/batches/military/chips?limit=500&min_partial=0").json()["items"]
    ref_pts = _get("/api/embedding/points?batch=military").json()["reference"]
    batch_novelties = [c["novelty"] for c in chips if c["novelty"] is not None]
    assert min(batch_novelties) >= nov["bins"][0]
    assert max(batch_novelties) <= nov["bins"][-1]
    assert ref_pts  # reference chips exist in the fixture
    assert not (nov["bins"][0] == 0.0 and nov["bins"][-1] == 1.0)


def test_batch_detail_vigilance_novelty_threshold():
    summary = _get("/api/summary").json()
    vigilance = summary["model"]["vigilance"]
    assert vigilance is not None

    body = _get("/api/batches/military").json()
    assert body["vigilance_novelty_threshold"] == pytest.approx(1.0 - vigilance)


def test_batch_detail_unknown_id():
    r = client.get("/api/batches/does_not_exist")
    assert r.status_code == 404


def test_batch_chips_pagination_and_filters():
    r = _get("/api/batches/foreign/chips?sort=novelty&order=desc&limit=5&offset=0")
    body = r.json()
    assert body["total"] >= 5
    assert len(body["items"]) == 5
    novelties = [c["novelty"] for c in body["items"]]
    assert novelties == sorted(novelties, reverse=True)

    r2 = _get("/api/batches/military/chips?only_novel=true")
    for chip in r2.json()["items"]:
        assert chip["is_novel"] is True

    r3 = _get("/api/batches/military/chips?reviewed=no")
    assert r3.status_code == 200


def test_batch_chips_have_detected_field():
    r = _get("/api/batches/military/chips?limit=48")
    items = r.json()["items"]
    assert items  # fixture builds detections for the military batch
    assert all("detected" in c and (c["detected"] is None or isinstance(c["detected"], bool)) for c in items)
    # at least one chip actually has a resolved (non-null) detected value
    assert any(c["detected"] is not None for c in items)


def test_batch_chips_detected_filter():
    all_items = _get("/api/batches/military/chips?limit=500").json()["items"]
    n_yes = sum(1 for c in all_items if c["detected"] is True)
    n_no = sum(1 for c in all_items if c["detected"] is False)

    yes = _get("/api/batches/military/chips?limit=500&detected=yes").json()
    assert yes["total"] == n_yes
    assert all(c["detected"] is True for c in yes["items"])

    no = _get("/api/batches/military/chips?limit=500&detected=no").json()
    assert no["total"] == n_no
    assert all(c["detected"] is False for c in no["items"])


def test_batch_chips_have_partial_and_truncated_fields():
    r = _get("/api/batches/military/chips?limit=500&min_partial=0")
    items = r.json()["items"]
    assert items
    for c in items:
        assert isinstance(c["partial"], (int, float))
        assert 0.0 <= c["partial"] <= 1.0
        assert isinstance(c["truncated"], bool)
        # fixture invariant (mirrors the real data): not truncated => partial == 1.0
        if not c["truncated"]:
            assert c["partial"] == 1.0
    assert any(c["truncated"] for c in items)  # fixture builds some truncated chips
    assert any(c["partial"] < 0.75 for c in items)  # ... some below the default threshold


def test_batch_chips_min_partial_filter():
    """min_partial is a display/triage filter: excludes chips with
    partial < min_partial; min_partial=0 must include everything; the
    paginated `total` must reflect the filter (see API.md).
    """
    unfiltered = _get("/api/batches/military/chips?limit=500&min_partial=0").json()
    default = _get("/api/batches/military/chips?limit=500").json()  # default min_partial=0.75
    explicit = _get("/api/batches/military/chips?limit=500&min_partial=0.75").json()

    assert default["total"] == explicit["total"]
    assert unfiltered["total"] > default["total"]  # the fixture has some partial < 0.75 chips
    assert all(c["partial"] >= 0.75 for c in default["items"])
    assert any(c["partial"] < 0.75 for c in unfiltered["items"])

    strict = _get("/api/batches/military/chips?limit=500&min_partial=1.0").json()
    assert all(c["partial"] >= 1.0 for c in strict["items"])
    assert strict["total"] <= default["total"]


def test_batch_detail_top_novel_min_partial():
    """min_partial only trims /batches/{id}'s top_novel -- novelty_hist,
    ref_novelty_hist and the batch metrics themselves must be unaffected
    (this is a display/triage filter only, per API.md).
    """
    default = _get("/api/batches/military").json()
    unfiltered = _get("/api/batches/military?min_partial=0").json()

    assert default["novelty_hist"] == unfiltered["novelty_hist"]
    assert default["ref_novelty_hist"] == unfiltered["ref_novelty_hist"]
    assert default["n_chips"] == unfiltered["n_chips"]
    assert default["novelty_rate"] == unfiltered["novelty_rate"]
    assert all(c["partial"] >= 0.75 for c in default["top_novel"])


CHIP_ANN_ID = None


def test_chip_detail_and_neighbors():
    global CHIP_ANN_ID
    batch = _get("/api/batches/military/chips?limit=1").json()
    ann_id = batch["items"][0]["ann_id"]
    CHIP_ANN_ID = ann_id
    r = _get(f"/api/chips/{ann_id}")
    body = r.json()
    assert body["ann_id"] == ann_id
    assert "neighbors" in body
    assert "art_match" in body
    assert body["role_true"] is not None


def test_chip_not_found():
    r = client.get("/api/chips/999999999")
    assert r.status_code == 404


def test_chip_image():
    batch = _get("/api/batches/military/chips?limit=1").json()
    ann_id = batch["items"][0]["ann_id"]
    r = client.get(f"/api/chips/{ann_id}/image")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert "cache-control" in {k.lower() for k in r.headers.keys()}


def test_tile_image_and_detail():
    batch = _get("/api/batches/military/chips?limit=1").json()
    tile_id = batch["items"][0]["tile_id"]
    r = client.get(f"/api/tiles/{tile_id}/image")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"

    r2 = _get(f"/api/tiles/{tile_id}")
    body = r2.json()
    assert body["tile_id"] == tile_id
    assert len(body["chips"]) > 0
    assert all(c["tile_id"] == tile_id for c in body["chips"])
    assert "detections" in body
    assert isinstance(body["detections"], list)  # this tile is in a detection-evaluated domain
    for d in body["detections"]:
        assert set(d) == {"x1", "y1", "x2", "y2", "conf", "tp"}


def test_tile_not_found():
    r = client.get("/api/tiles/999999999")
    assert r.status_code == 404


def test_embedding_points():
    r = _get("/api/embedding/points?batch=military")
    body = r.json()
    assert "reference" in body and "batch" in body
    assert len(body["reference"]) <= 1500
    assert len(body["batch"]) > 0
    for pt in body["reference"]:
        assert set(pt) == {"ann_id", "x", "y", "role_true"}


def test_embedding_points_unknown_batch():
    r = client.get("/api/embedding/points?batch=nope")
    assert r.status_code == 404


def test_drift():
    r = _get("/api/drift")
    body = r.json()
    assert body["statistic"]
    assert len(body["series"]) == 7
    orders = [s["order"] for s in body["series"]]
    assert orders == sorted(orders)


def test_envelope():
    r = _get("/api/envelope")
    body = r.json()
    dim_names = {d["name"] for d in body["dimensions"]}
    assert dim_names == {"weather", "off_nadir", "country", "gsd", "aircraft_size"}
    assert body["thresholds"]["min_n"] == 30
    assert body["thresholds"]["novelty_multiplier"] == 3.0
    off_nadir = next(d for d in body["dimensions"] if d["name"] == "off_nadir")
    statuses = {b["label"]: b["status"] for b in off_nadir["buckets"]}
    assert statuses  # non-empty
    # "reference" is no longer a status -- status comes from evidence only
    # (see in_reference_definition for the separate reference-definition flag).
    for dim in body["dimensions"]:
        for b in dim["buckets"]:
            assert b["status"] in {"tested", "untested", "degraded"}
            assert isinstance(b["in_reference_definition"], bool)
            assert isinstance(b["n_parent_images"], int)
            assert set(b["control_delta"]) == {"role_accuracy", "detection_recall"}


def test_envelope_control_thresholds():
    r = _get("/api/envelope")
    body = r.json()
    control = body["thresholds"]["control"]
    assert set(control) == {"role_accuracy", "novelty_rate", "detection_recall", "n_chips", "n_parent_images"}
    # the fixture's ref_holdout batch has real detection + role-accuracy data
    assert control["detection_recall"] is not None
    assert control["role_accuracy"] is not None
    assert control["n_chips"] > 0


def test_envelope_reference_label_can_be_degraded():
    """Regression for the tested-envelope bug: a bucket whose label matches
    the reference definition (Clear Skies / USA / off-nadir <=15 or 15-25)
    must never be exempted from the degradation checks -- status is computed
    purely from evidence against the ref_holdout control (see
    dataio._dimension); in_reference_definition is a separate annotation
    that must not force any particular status.
    """
    r = _get("/api/envelope")
    body = r.json()
    dims = {d["name"]: d for d in body["dimensions"]}

    weather = {b["label"]: b for b in dims["weather"]["buckets"]}
    assert weather["Clear Skies"]["in_reference_definition"] is True
    assert weather["Clear Skies"]["status"] == "degraded"

    country = {b["label"]: b for b in dims["country"]["buckets"]}
    assert country["USA"]["in_reference_definition"] is True
    assert country["USA"]["status"] == "degraded"

    off_nadir = {b["label"]: b for b in dims["off_nadir"]["buckets"]}
    assert off_nadir["15-25deg"]["in_reference_definition"] is True
    assert off_nadir["15-25deg"]["status"] == "degraded"
    # a reference-matching bucket that ISN'T degraded proves the flag doesn't
    # force "degraded" either -- it's purely informational.
    assert off_nadir["<=15deg"]["in_reference_definition"] is True
    assert off_nadir["<=15deg"]["status"] != "degraded"

    # gsd/aircraft_size aren't part of the reference definition at all.
    for dim_name in ("gsd", "aircraft_size"):
        for b in dims[dim_name]["buckets"]:
            assert b["in_reference_definition"] is False


def test_envelope_detection_recall():
    r = _get("/api/envelope")
    body = r.json()
    assert body["thresholds"]["control"]["detection_recall"] is not None

    dims = {d["name"]: d for d in body["dimensions"]}
    for name in ("weather", "off_nadir", "country", "gsd"):
        buckets = dims[name]["buckets"]
        assert buckets
        assert any(b["detection_recall"] is not None for b in buckets), name
        for b in buckets:
            if b["detection_recall"] is not None:
                assert 0.0 <= b["detection_recall"] <= 1.0

    # aircraft_size has no detection breakdown in detection_metrics.json
    for b in dims["aircraft_size"]["buckets"]:
        assert b["detection_recall"] is None


def test_eval():
    r = _get("/api/eval")
    body = r.json()
    assert body["version"] == "v2"
    assert "note" in body and "evaluation only" in body["note"]
    assert "auroc_military_vs_civil" in body["novelty"]
    assert "accuracy_ref_holdout" in body["role_head"]

    det = body["detection"]
    assert det is not None
    for group in ("civil", "military"):
        assert 0.0 <= det["by_role_group"][group]["recall"] <= 1.0
        assert det["by_role_group"][group]["n"] > 0
    # by_role_group is scoped to within the military batch; by_role_group_pooled
    # pools civil recall across every detection-evaluated batch instead -- both
    # carry a `scope` string describing what population they're computed over
    # (see dataio._normalize_role_group).
    assert det["by_role_group"]["scope"]
    assert det["by_role_group_pooled"] is not None
    for group in ("civil", "military"):
        assert 0.0 <= det["by_role_group_pooled"][group]["recall"] <= 1.0
        assert det["by_role_group_pooled"][group]["n"] > 0
    assert det["by_role_group_pooled"]["scope"]
    assert det["by_role_group"]["scope"] != det["by_role_group_pooled"]["scope"]
    assert "military" in det["by_batch"] and "snow" in det["by_batch"]
    for b in det["by_batch"].values():
        assert set(b) == {"recall", "precision", "n_gt"}
    assert det["val_map50"] is not None
    assert det["conf_threshold"] is not None


def test_eval_detection_role_group_falls_back_to_old_shape():
    """detection_metrics.json is written by vigil (rewritten concurrently --
    its output shape may change under us). Older files carry a flat
    `recall_by_role_group` with no `scope` key and no
    `recall_by_role_group_pooled` at all -- dataio.detection_eval_summary()
    must degrade gracefully to that shape rather than erroring.
    """
    from app import dataio

    old_shape = {
        "conf_threshold": 0.25,
        "iou_threshold": 0.5,
        "by_domain": {},
        "by_condition": {},
        "recall_by_role_group": {
            "civil": 0.9, "military": 0.6, "n_civil": 100, "n_military": 10,
        },
    }
    original = dataio.detection_metrics
    dataio.detection_metrics = lambda: old_shape
    try:
        summary = dataio.detection_eval_summary()
    finally:
        dataio.detection_metrics = original

    assert summary is not None
    assert summary["by_role_group"]["civil"] == {"recall": 0.9, "n": 100}
    assert summary["by_role_group"]["military"] == {"recall": 0.6, "n": 10}
    assert summary["by_role_group"]["scope"] is None
    assert summary["by_role_group_pooled"] is None


def test_review_queue_strategies():
    for strategy in ("novelty", "uncertainty", "random"):
        r = _get(f"/api/review/queue?limit=5&strategy={strategy}")
        assert isinstance(r.json(), list)


def test_review_queue_det_source():
    """GET /review/queue?source=det (field mode): flagged detector chips across NEW collections only,
    unreviewed, ranked -- DetChip dicts (same as /collections/{id}'s chips) plus `group_id`. Without
    `source` the queue is the GT-chip queue exactly as before.
    """
    import pandas as pd

    q = _get("/api/review/queue?source=det&limit=500").json()
    assert q
    collections = _get("/api/collections").json()
    groups = {g["group_id"]: g for g in collections["groups"]}
    idx = pd.read_parquet(_TEST_DATA_ROOT / "chips_det" / "index.parquet")
    domain_by_det = dict(zip(idx["det_id"], idx["domain"]))
    base_keys = {
        "source", "det_id", "ann_id", "tile_id", "bbox", "conf", "chip_url", "tile_url",
        "role_pred", "role_conf", "novelty", "is_novel", "knn_dist", "meta", "review", "group_id",
    }
    # default strategy for det is "collection": items also carry their collection's airfield/date/status
    group_keys = {"airfield", "date", "group_status", "group_n_flagged"}
    for c in q:
        assert set(c) in (base_keys, base_keys | group_keys)
        assert c["source"] == "det" and c["is_novel"] is True and c["review"] is None
        assert (c.get("conf") or 0.0) >= 0.5  # default min_conf
        assert c["group_id"] in groups
        assert domain_by_det[c["det_id"]] not in {"ref_train", "ref_calib", "ref_holdout"}
    _assert_no_forbidden_keys(q)
    # explicit strategies still work; min_conf=0 disables the confidence gate
    q_nov = _get("/api/review/queue?source=det&limit=500&strategy=novelty&min_conf=0").json()
    assert len(q_nov) >= len(q)
    # collection strategy: needs_review collections come first
    statuses = [c.get("group_status") for c in q if c.get("group_status")]
    if "needs_review" in statuses:
        first_other = next((i for i, s_ in enumerate(statuses) if s_ != "needs_review"), len(statuses))
        assert all(s_ == "needs_review" for s_ in statuses[:first_other])
    # the uncapped queue = every flagged det chip in a new collection minus those already reviewed
    # (source='det' reviews recorded earlier in this suite)
    reviewed_det = {c["ann_id"] for c in _get("/api/review/log?source=det").json()}
    assert not (reviewed_det & {c["det_id"] for c in q})
    flagged_total = sum(g["n_flagged"] for g in groups.values())
    assert flagged_total - len(reviewed_det) <= len(q_nov) <= flagged_total  # ungated novelty queue covers every unreviewed flagged chip

    novelties = [c["novelty"] for c in _get("/api/review/queue?source=det&strategy=novelty&limit=50").json()]
    assert novelties == sorted(novelties, reverse=True)
    confs = [c["role_conf"] for c in _get("/api/review/queue?source=det&strategy=uncertainty&limit=50").json()]
    assert confs == sorted(confs)
    assert len(_get("/api/review/queue?source=det&strategy=random&limit=5").json()) == 5
    assert len(_get("/api/review/queue?source=det&limit=3").json()) == 3

    # reviewing a queued det chip removes it from the det queue (and only from the det queue)
    target = q[0]["det_id"]
    r = client.post(f"/api/review/{target}?source=det", json={"action": "skip"})
    assert r.status_code == 200, r.text
    assert target not in {c["det_id"] for c in _get("/api/review/queue?source=det&limit=500").json()}

    # source omitted -> the GT-chip queue, unchanged shape (Chip, no det_id/group_id, has partial/role_true)
    gt = _get("/api/review/queue?limit=5").json()
    assert gt and all("det_id" not in c and "group_id" not in c and "partial" in c and "role_true" in c for c in gt)
    assert client.get("/api/review/queue?source=bogus").status_code == 422


def test_review_queue_min_partial_filter():
    default = _get("/api/review/queue?limit=500").json()  # default min_partial=0.75
    assert all(c["partial"] >= 0.75 for c in default)

    unfiltered = _get("/api/review/queue?limit=500&min_partial=0").json()
    assert len(unfiltered) >= len(default)
    assert any(c["partial"] < 0.75 for c in unfiltered)


def test_review_post_and_log():
    batch = _get("/api/batches/foreign/chips?limit=1").json()
    ann_id = batch["items"][0]["ann_id"]
    predicted_role = batch["items"][0]["role_pred"]

    r = client.post(f"/api/review/{ann_id}", json={"action": "confirm", "note": "looks right"})
    assert r.status_code == 200, r.text
    review = r.json()
    assert review["ann_id"] == ann_id
    assert review["action"] == "confirm"
    assert review["version_applied"] is None
    # every POST snapshots the chip's current role_pred + current version (see db.upsert_review)
    assert review["predicted_role"] == predicted_role
    assert review["predicted_version"] == _get("/api/summary").json()["current_version"]

    r_log = _get("/api/review/log")
    log = r_log.json()
    assert any(x["ann_id"] == ann_id for x in log)
    logged = next(x for x in log if x["ann_id"] == ann_id)
    assert logged["predicted_role"] == predicted_role

    # invalid action rejected
    r_bad = client.post(f"/api/review/{ann_id}", json={"action": "bogus"})
    assert r_bad.status_code == 422

    # new_class without label rejected
    r_bad2 = client.post(f"/api/review/{ann_id}", json={"action": "new_class"})
    assert r_bad2.status_code == 422


def test_review_classes():
    r = _get("/api/review/classes")
    body = r.json()
    assert "classes" in body
    summary = _get("/api/summary").json()
    current = next(v for v in summary["versions"] if v["id"] == summary["current_version"])
    assert body["classes"] == current["classes"]
    assert len(body["classes"]) >= 3  # fixture v2 has the 3 civil classes + 1 analyst class


def test_review_correct_action():
    """`action='correct'` records the analyst's DISAGREEMENT with the predicted role and
    assigns a known class label -- never alters the model, exactly like `confirm` (see
    vigil.pipeline.CORRECT_POLICY_NOTE / API.md).
    """
    batch = _get("/api/batches/off_nadir/chips?limit=1").json()
    chip = batch["items"][0]
    ann_id = chip["ann_id"]
    predicted_role = chip["role_pred"]
    current_version = _get("/api/summary").json()["current_version"]

    allowed = _get("/api/review/classes").json()["classes"]
    other_label = next(c for c in allowed if c != predicted_role)

    # invalid label rejected, with a helpful detail listing the allowed classes
    r_bad = client.post(f"/api/review/{ann_id}", json={"action": "correct", "label": "Not A Real Class"})
    assert r_bad.status_code == 422
    detail = r_bad.json()["detail"]
    assert "Not A Real Class" in detail
    for cls in allowed:
        assert cls in detail

    # missing label rejected
    r_missing = client.post(f"/api/review/{ann_id}", json={"action": "correct"})
    assert r_missing.status_code == 422

    # valid correction
    r = client.post(f"/api/review/{ann_id}", json={"action": "correct", "label": other_label})
    assert r.status_code == 200, r.text
    review = r.json()
    assert review["ann_id"] == ann_id
    assert review["action"] == "correct"
    assert review["label"] == other_label
    assert review["predicted_role"] == predicted_role
    assert review["predicted_version"] == current_version
    assert review["version_applied"] is None

    r_log = _get("/api/review/log")
    logged = next(x for x in r_log.json() if x["ann_id"] == ann_id)
    assert logged["action"] == "correct" and logged["label"] == other_label


def test_review_labels():
    batch = _get("/api/batches/haze/chips?limit=1").json()
    ann_id = batch["items"][0]["ann_id"]
    client.post(f"/api/review/{ann_id}", json={"action": "new_class", "label": "Test Label"})

    r = _get("/api/review/labels")
    body = r.json()
    assert "labels" in body
    assert "Test Label" in body["labels"]
    assert body["labels"][0] == "Test Label"  # most-recently-used first
    assert all(lbl is not None for lbl in body["labels"])


def test_det_chip_detail_and_image():
    """GET /det/{det_id} (field mode): the same Chip shape as /chips/{ann_id} where it applies, with
    source='det'/det_id/conf always shown and matched_ann_id (benchmark-only) hidden unless
    ?benchmark=1 -- see backend/API.md.
    """
    import pandas as pd

    idx = pd.read_parquet(_TEST_DATA_ROOT / "chips_det" / "index.parquet")
    matched_row = idx[idx["matched_ann_id"].notna()].iloc[0]
    det_id = int(matched_row["det_id"])

    r = _get(f"/api/det/{det_id}")
    body = r.json()
    assert body["source"] == "det"
    assert body["det_id"] == det_id
    assert body["conf"] is not None
    assert "matched_ann_id" not in body  # benchmark-only, hidden by default
    assert "neighbors" in body and "art_match" in body
    assert body["chip_url"] == f"/api/det/{det_id}/image"
    assert body["role_true"] is None if "role_true" in body else True  # field mode: no GT on a det chip

    assert body["ann_id"] == det_id  # alias, so shared chip-drawer/review components work unchanged
    assert "batch_id" not in body and "domain" not in body  # benchmark batch identity, hidden by default
    # Recursive forbidden-key check, with the one deliberate exception: `neighbors` are labeled
    # REFERENCE chips (the delivered validation set), so their role_true is legitimately known.
    _assert_no_forbidden_keys({k: v for k, v in body.items() if k != "neighbors"})
    assert all(set(n) == {"ann_id", "chip_url", "role_true", "dist"} for n in body["neighbors"])  # fixture: may be []

    r_bench = _get(f"/api/det/{det_id}?benchmark=1")
    body_bench = r_bench.json()
    assert body_bench["matched_ann_id"] == int(matched_row["matched_ann_id"])
    assert body_bench["batch_id"] == matched_row["domain"]

    r_img = client.get(f"/api/det/{det_id}/image")
    assert r_img.status_code == 200
    assert r_img.headers["content-type"] == "image/png"


def test_det_chip_not_found():
    r = client.get("/api/det/999999999")
    assert r.status_code == 404
    r_img = client.get("/api/det/999999999/image")
    assert r_img.status_code == 404


def test_det_review_post_and_source_isolation():
    """POST /review/{det_id}?source=det reviews a detector chip -- kept isolated from a source='gt'
    review that happens to share the same numeric id (db.py's composite (source, ann_id) primary
    key). GET /review/log?source=gt|det|all is source-aware (see API.md).
    """
    import pandas as pd

    idx = pd.read_parquet(_TEST_DATA_ROOT / "chips_det" / "index.parquet")
    det_id = int(idx.iloc[0]["det_id"])

    predicted_role = _get(f"/api/det/{det_id}").json()["role_pred"]

    r = client.post(f"/api/review/{det_id}?source=det", json={"action": "confirm", "note": "field mode"})
    assert r.status_code == 200, r.text
    review = r.json()
    assert review["source"] == "det"
    assert review["ann_id"] == det_id
    assert review["action"] == "confirm"
    assert review["predicted_role"] == predicted_role

    det_chip = _get(f"/api/det/{det_id}").json()
    assert det_chip["review"] is not None
    assert det_chip["review"]["action"] == "confirm"
    assert det_chip["review"]["source"] == "det"

    log_det = _get("/api/review/log?source=det").json()
    log_gt = _get("/api/review/log?source=gt").json()
    log_all = _get("/api/review/log?source=all").json()
    assert any(r["ann_id"] == det_id and r["source"] == "det" for r in log_det)
    assert not any(r["ann_id"] == det_id and r["source"] == "det" for r in log_gt)
    assert any(r["ann_id"] == det_id and r["source"] == "det" for r in log_all)
    assert len(log_all) >= len(log_det)
    assert len(log_all) >= len(log_gt)

    # A source='gt' review sharing the SAME numeric id must not be conflated with the det one.
    r_gt = client.post(f"/api/review/{det_id}", json={"action": "skip"})  # source defaults to 'gt'
    assert r_gt.status_code == 200, r_gt.text
    log_det_after = _get("/api/review/log?source=det").json()
    det_entry = next(x for x in log_det_after if x["ann_id"] == det_id)
    assert det_entry["action"] == "confirm"  # unaffected by the source='gt' POST above


def test_review_bad_box_action():
    """`action='bad_box'` (localization error -- the detector's box fuses two aircraft or truncates
    one) needs no label, marks the chip reviewed, and is recorded as evidence exactly like
    confirm/correct/not_aircraft -- never a training negative, never fitted (see backend/API.md's
    Field mode section / vigil.pipeline.BAD_BOX_POLICY_NOTE). Allowed for both sources; exercised here
    on a det chip (source='det'), the case it's most useful for.
    """
    # a chip in a new collection (never ref_train/ref_calib/ref_holdout), unreviewed so far --
    # the flagged det queue is exactly that population, see flagged_det_queue().
    queued = _get("/api/review/queue?source=det&limit=1&min_conf=0").json()
    assert queued, "expected at least one unreviewed flagged det chip left in the queue"
    det_id = int(queued[0]["det_id"])
    predicted_role = _get(f"/api/det/{det_id}").json()["role_pred"]

    groups_before = {g["group_id"]: g["n_bad_box"] for g in _get("/api/collections").json()["groups"]}

    r = client.post(f"/api/review/{det_id}?source=det", json={"action": "bad_box"})
    assert r.status_code == 200, r.text
    review = r.json()
    assert review["source"] == "det"
    assert review["ann_id"] == det_id
    assert review["action"] == "bad_box"
    assert review["label"] is None
    assert review["predicted_role"] == predicted_role

    det_chip = _get(f"/api/det/{det_id}").json()
    assert det_chip["review"]["action"] == "bad_box"

    log_det = _get("/api/review/log?source=det").json()
    assert any(x["ann_id"] == det_id and x["action"] == "bad_box" for x in log_det)

    # cheap per-group count on /collections -- bumped by exactly this one review
    groups_after = _get("/api/collections").json()["groups"]
    total_before = sum(groups_before.values())
    total_after = sum(g["n_bad_box"] for g in groups_after)
    assert total_after == total_before + 1

    # a GT-chip bad_box review is accepted too (allowed for both sources) -- pulled from the GT
    # review queue (unreviewed chips only) so this never collides with a chip another test reviewed.
    gt_ann_id = _get("/api/review/queue?limit=1").json()[0]["ann_id"]
    r_gt = client.post(f"/api/review/{gt_ann_id}", json={"action": "bad_box"})
    assert r_gt.status_code == 200, r_gt.text
    assert r_gt.json()["action"] == "bad_box" and r_gt.json()["label"] is None


def test_eval_analyst_disagreement():
    """/eval.analyst_disagreement -- computed from the reviews table, scoped to the
    CURRENT version (unlike /versions/compare's, which is deliberately all-time -- see
    dataio.analyst_disagreement). By this point in the suite, test_review_post_and_log has
    recorded one `confirm` and test_review_correct_action one `correct`, both against the
    current version (v2), so this must show a real (non-null) disagreement rate and a
    non-empty by_pair/by_predicted_role breakdown. test_review_bad_box_action recorded at least
    one `bad_box` (both sources) against v2 too.
    """
    r = _get("/api/eval")
    body = r.json()
    ad = body["analyst_disagreement"]
    assert set(ad) == {
        "n_reviewed", "n_confirm", "n_correct", "n_new_class", "n_not_aircraft", "n_skip", "n_bad_box",
        "disagreement_rate", "by_pair", "by_predicted_role",
    }
    assert ad["n_confirm"] >= 1
    assert ad["n_correct"] >= 1
    assert ad["n_new_class"] >= 1  # test_review_labels
    assert ad["n_bad_box"] >= 2  # test_review_bad_box_action: one det, one gt
    assert ad["n_reviewed"] >= ad["n_confirm"] + ad["n_correct"] + ad["n_new_class"] + ad["n_bad_box"]
    assert ad["disagreement_rate"] is not None
    assert 0.0 <= ad["disagreement_rate"] <= 1.0

    assert ad["by_pair"]
    for p in ad["by_pair"]:
        assert set(p) == {"from_role", "to_role", "n"}
        assert p["n"] >= 1
    ns = [p["n"] for p in ad["by_pair"]]
    assert ns == sorted(ns, reverse=True)  # sorted desc, per API.md

    assert ad["by_predicted_role"]
    for row in ad["by_predicted_role"]:
        assert set(row) == {"role", "n_confirm", "n_correct", "disagreement_rate"}


def test_review_classes_empty_denominator():
    """disagreement_rate is null (not a ZeroDivisionError) when n_confirm + n_correct == 0 --
    exercised directly against dataio since every review test above has, by now, recorded at
    least one confirm/correct against the fixture's shared reviews table.
    """
    from app import dataio

    result = dataio.analyst_disagreement(version="v_never_used")
    assert result["n_confirm"] == 0 and result["n_correct"] == 0
    assert result["n_bad_box"] == 0
    assert result["disagreement_rate"] is None
    assert result["by_pair"] == []
    assert result["by_predicted_role"] == []


def test_versions_compare():
    r = _get("/api/versions/compare?a=v1&b=v2")
    body = r.json()
    assert body["a"] == "v1" and body["b"] == "v2"
    assert "forgetting" in body and body["forgetting"] >= 0
    assert len(body["ref_holdout"]["per_class"]) >= 3
    assert len(body["novelty_rate_by_batch"]) == 7

    # all-time, not per version a/b -- see API.md
    ad = body["analyst_disagreement"]
    assert set(ad) == {
        "n_reviewed", "n_confirm", "n_correct", "n_new_class", "n_not_aircraft", "n_skip", "n_bad_box",
        "disagreement_rate", "by_pair", "by_predicted_role",
    }
    assert ad["n_correct"] >= 1
    assert ad["n_confirm"] >= 1
    assert ad["n_bad_box"] >= 2  # test_review_bad_box_action
    assert "recall_new_class_a" in body["military"]
    assert "recall_new_class_b" in body["military"]


FIELD_BLOCK_KEYS = {
    "n_detections", "n_excluded_labeled", "n_excluded_same_parent", "n_eligible", "classes", "control",
    "reviewed_flagged", "labeled_collections", "assign_min_conf", "note",
}
FIELD_REVIEWED_FLAGGED_KEYS = {"n", "n_not_aircraft", "n_bad_box", "field_precision_on_flagged"}
FIELD_CLASS_KEYS = {"label", "n_labels_used", "n_assigned", "assignment_rate", "by_group"}
FIELD_BY_GROUP_KEYS = {"group_id", "airfield", "date", "n_assigned"}
FIELD_LABELED_COLLECTION_KEYS = {
    "group_id", "airfield", "date", "n_labeled", "n_bad_box", "n_not_aircraft",
    "n_flagged_unlabeled", "n_flagged_unlabeled_assigned", "per_class",
}


def test_versions_compare_field_shape():
    """/versions/compare's `field` = {a: FieldBlock|null, b: FieldBlock|null, worst_regression} per
    API.md's Field mode section: each side composed by dataio.field_block from eval_{version}.json's
    `field` + `control` blocks (vigil.field) -- descriptive assignment counts with per-collection
    `by_group`, plus the labeled control-set regression check nested as `control`. Benchmark batch
    names (`domains`/`by_domain`) must NOT pass through to this analyst-view shape.
    """
    body = _get("/api/versions/compare?a=v1&b=v2").json()
    field = body["field"]
    assert set(field) == {"a", "b", "worst_regression"}
    _assert_no_forbidden_keys(field)
    for side in ("a", "b"):
        f = field[side]
        assert f is not None
        assert set(f) == FIELD_BLOCK_KEYS, set(f) ^ FIELD_BLOCK_KEYS
        assert f["n_detections"] >= f["n_eligible"] >= 0
        assert "descriptive" in f["note"]
        # the fixture's hand-built eval jsons predate `reviewed_flagged` -- dataio.field_block()
        # fills in the all-zero/null default rather than omitting the key (see FIELD_BLOCK_KEYS).
        assert set(f["reviewed_flagged"]) == FIELD_REVIEWED_FLAGGED_KEYS
        assert f["reviewed_flagged"] == {"n": 0, "n_not_aircraft": 0, "n_bad_box": 0, "field_precision_on_flagged": None}
        c = f["control"]
        assert c["label"] == "labeled control-set regression check"
        assert c["overall"] is not None
        assert c["per_class"] and all(set(p) == {"cls", "accuracy", "n"} for p in c["per_class"])
        for cls in f["classes"]:
            assert set(cls) == FIELD_CLASS_KEYS, set(cls) ^ FIELD_CLASS_KEYS
            for g in cls["by_group"]:
                assert set(g) == FIELD_BY_GROUP_KEYS
                assert g["group_id"].endswith(g["date"])
        # the fixture's hand-built eval jsons predate `labeled_collections`/`assign_min_conf` -- both
        # default rather than being omitted (see dataio.field_block).
        assert f["labeled_collections"] == []
        assert f["assign_min_conf"] is None

    # v1 has no analyst class yet; v2 (fixture: "adaptation" added Military Fighter/...) does, with
    # its assignments broken out by collection (top 10, desc by n_assigned).
    assert field["a"]["classes"] == []
    cls_b = field["b"]["classes"]
    assert cls_b and cls_b[0]["label"] == "Military Fighter/Interceptor/Attack"
    assert cls_b[0]["n_labels_used"] >= 1
    assert cls_b[0]["by_group"], "fixture v2 assigns the analyst class in the military collections"
    counts = [g["n_assigned"] for g in cls_b[0]["by_group"]]
    assert counts == sorted(counts, reverse=True) and len(counts) <= 10
    assert sum(counts) <= cls_b[0]["n_assigned"]
    # by_group ids are real /collections group ids
    known = {g["group_id"] for g in _get("/api/collections").json()["groups"]}
    assert all(g["group_id"] in known for g in cls_b[0]["by_group"])

    wr = field["worst_regression"]
    assert wr is None or set(wr) == {"cls", "delta", "a", "b"}


def test_field_block_none_for_unknown_version():
    from app import dataio

    assert dataio.field_block("v_never_built") is None
    fc = dataio.field_compare("v1", "v_never_built")
    assert fc["a"] is not None and fc["b"] is None and fc["worst_regression"] is None


def test_map():
    r = _get("/api/map")
    points = r.json()["points"]
    assert len(points) > 0
    for p in points:
        assert "parent_id" in p and "lat" in p and "lon" in p


# Ground truth (role_true, matched_ann_id) and benchmark batch/domain identity (batch_id, domain,
# domains, by_domain) must never appear anywhere in an analyst-view payload -- see API.md's Field
# mode section. `role_true` on /det/{id}'s `neighbors` is the ONE deliberate exception (those are
# labeled reference chips -- the delivered validation set's own labels), tested separately.
FORBIDDEN_KEYS = {"role_true", "matched_ann_id", "batch_id", "domain", "domains", "by_domain"}


def _assert_no_forbidden_keys(obj, path="$"):
    """Recursively assert no forbidden key shows up anywhere in an analyst-view payload."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            assert k not in FORBIDDEN_KEYS, f"forbidden key '{k}' found at {path}.{k}"
            _assert_no_forbidden_keys(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            _assert_no_forbidden_keys(item, f"{path}[{i}]")


def test_collections_list_shape():
    r = _get("/api/collections")
    body = r.json()
    assert set(body) == {"groups", "grouping", "min_denominator", "p_threshold", "reference", "reference_flag_rate", "status_rule"}
    assert body["grouping"] == "airfield × collection date (UTC day)"
    assert body["min_denominator"] == 20
    assert body["p_threshold"] == 0.01
    assert "insufficient_data" in body["status_rule"] and "needs_review" in body["status_rule"]
    assert "binomial" in body["status_rule"] and "correlated" in body["status_rule"]  # precise + indicative
    assert body["reference"]["n_parents"] > 0
    assert body["reference"]["n_tiles"] > 0
    assert "only labeled data" in body["reference"]["description"]
    assert body["reference_flag_rate"] is not None
    assert 0.0 <= body["reference_flag_rate"] <= 1.0

    groups = body["groups"]
    assert groups  # the fixture builds det chips across every non-reference domain
    dates = [g["date"] for g in groups]
    assert dates == sorted(dates)  # sorted by date asc, per API.md

    for g in groups:
        assert set(g) == {
            "group_id", "airfield", "country", "state", "date", "weather", "off_nadir", "gsd",
            "lat", "lon", "n_parents", "n_tiles", "n_detections", "n_flagged", "flag_rate",
            "excess_flagged", "p_value", "n_reviewed", "n_bad_box", "status",
        }
        assert g["n_bad_box"] <= g["n_detections"]
        assert g["status"] in {"needs_review", "nominal", "insufficient_data"}
        if g["n_detections"] > 0:
            assert 0.0 <= g["p_value"] <= 1.0
            assert g["excess_flagged"] == pytest.approx(g["n_flagged"] - g["n_detections"] * body["reference_flag_rate"])
        assert g["n_parents"] >= 1
        assert g["n_tiles"] >= 1
        assert g["n_flagged"] <= g["n_detections"]
        if g["n_detections"] > 0:
            assert g["flag_rate"] == pytest.approx(g["n_flagged"] / g["n_detections"])
        # group_id is a url-safe slug(airfield) + "_" + date
        assert g["group_id"].endswith(g["date"])
        assert " " not in g["group_id"]


def test_collections_status_rule_all_three_outcomes():
    """The fixture is built so all three status outcomes are actually exercised (see
    build_fixture.py's DET_FLAG_FRACTION comment): one small domain stays under min_denominator,
    foreign/haze sit near the ref_holdout control's own flag rate (nominal), and snow/military/
    synthetic are flagged far above it (needs_review). Status = one-sided binomial tail test of
    n_flagged given n_detections against reference_flag_rate, p < p_threshold (see collections.STATUS_RULE).
    """
    from scipy.stats import binom

    body = _get("/api/collections").json()
    statuses = {g["status"] for g in body["groups"]}
    assert statuses == {"insufficient_data", "nominal", "needs_review"}
    p_ref = body["reference_flag_rate"]
    thr = body["p_threshold"]

    for g in body["groups"]:
        expected_p = float(binom.sf(g["n_flagged"] - 1, g["n_detections"], p_ref))
        assert g["p_value"] == pytest.approx(expected_p)
        if g["n_detections"] < body["min_denominator"]:
            assert g["status"] == "insufficient_data"
        elif g["p_value"] < thr:
            assert g["status"] == "needs_review"
        else:
            assert g["status"] == "nominal"
    assert any(g["status"] == "nominal" for g in body["groups"])


def test_collections_status_rule_no_size_bias():
    """The old absolute `n_flagged >= 10` rule marked every big civil airport needs_review (21 flagged
    of 510 = 4.1% vs a 3.1% reference). The binomial rule has no such size bias, while a small
    collection with a real excess (18 of 66) is caught, and a k=0 collection gets p=1.
    """
    from app.collections import MIN_DENOMINATOR, P_THRESHOLD, _status_for

    p_ref = 0.0314
    big = _status_for(n_detections=510, n_flagged=21, reference_flag_rate=p_ref)
    assert big["status"] == "nominal" and big["p_value"] > P_THRESHOLD
    assert big["excess_flagged"] == pytest.approx(21 - 510 * p_ref)

    small = _status_for(n_detections=66, n_flagged=18, reference_flag_rate=p_ref)
    assert small["status"] == "needs_review" and small["p_value"] < 1e-9
    assert small["excess_flagged"] > 15

    none_flagged = _status_for(n_detections=100, n_flagged=0, reference_flag_rate=p_ref)
    assert none_flagged["status"] == "nominal" and none_flagged["p_value"] == pytest.approx(1.0)
    assert none_flagged["excess_flagged"] < 0

    # below the minimum denominator is always insufficient_data, but p/excess are still reported
    tiny = _status_for(n_detections=MIN_DENOMINATOR - 1, n_flagged=19, reference_flag_rate=p_ref)
    assert tiny["status"] == "insufficient_data" and tiny["p_value"] is not None


def test_collections_status_zero_or_null_reference_rate_guarded():
    """Regression: with a null OR ZERO reference_flag_rate there is no tail to test against (binom.sf
    at p=0 is 0 for any n_flagged >= 1, which would mark every collection with one flagged chip
    needs_review) -- status falls back to n_flagged >= FALLBACK_MIN_FLAGGED, p_value/excess_flagged are
    null, and the served status_rule string says so.
    """
    from app.collections import (
        FALLBACK_MIN_FLAGGED,
        MIN_DENOMINATOR,
        STATUS_RULE,
        STATUS_RULE_FALLBACK,
        _status_for,
        status_rule_for,
    )

    for p_ref in (0.0, None):
        one = _status_for(n_detections=MIN_DENOMINATOR, n_flagged=1, reference_flag_rate=p_ref)
        assert one == {"status": "nominal", "p_value": None, "excess_flagged": None}
        many = _status_for(n_detections=100, n_flagged=FALLBACK_MIN_FLAGGED, reference_flag_rate=p_ref)
        assert many["status"] == "needs_review" and many["p_value"] is None
        assert _status_for(n_detections=MIN_DENOMINATOR - 1, n_flagged=0, reference_flag_rate=p_ref)["status"] == "insufficient_data"
        assert status_rule_for(p_ref) == STATUS_RULE_FALLBACK
    assert status_rule_for(0.03) == STATUS_RULE
    assert "Fallback" in STATUS_RULE_FALLBACK and "binomial" in STATUS_RULE


def test_collections_summary():
    body = _get("/api/collections").json()
    groups = body["groups"]
    r = _get("/api/collections/summary")
    summary = r.json()
    assert set(summary) == {"n_groups", "n_needs_review", "n_insufficient_data", "n_detections", "n_flagged", "reference_flag_rate"}
    assert summary["n_groups"] == len(groups)
    assert summary["n_needs_review"] == sum(1 for g in groups if g["status"] == "needs_review")
    assert summary["n_insufficient_data"] == sum(1 for g in groups if g["status"] == "insufficient_data")
    assert summary["n_detections"] == sum(g["n_detections"] for g in groups)
    assert summary["n_flagged"] == sum(g["n_flagged"] for g in groups)
    assert summary["reference_flag_rate"] == body["reference_flag_rate"]


def test_collections_detail_shape_and_chip_fields():
    body = _get("/api/collections").json()
    group = next(g for g in body["groups"] if g["status"] == "needs_review")
    r = _get(f"/api/collections/{group['group_id']}")
    detail = r.json()

    assert detail["group_id"] == group["group_id"]
    assert detail["n_detections"] == group["n_detections"]
    assert "tiles" in detail and isinstance(detail["tiles"], list) and detail["tiles"]
    for t in detail["tiles"]:
        assert set(t) == {"tile_id", "n_detections", "n_flagged"}
        assert t["n_flagged"] <= t["n_detections"]
    assert sum(t["n_detections"] for t in detail["tiles"]) == detail["n_detections"]

    assert set(detail["chips"]) == {"total", "items"}
    assert detail["chips"]["total"] == detail["n_detections"]
    assert detail["chips"]["items"]
    for chip in detail["chips"]["items"]:
        assert set(chip) == {
            "source", "det_id", "ann_id", "tile_id", "bbox", "conf", "chip_url", "tile_url",
            "role_pred", "role_conf", "novelty", "is_novel", "knn_dist", "meta", "review",
        }
        assert chip["source"] == "det"
        assert chip["ann_id"] == chip["det_id"]
        assert chip["chip_url"] == f"/api/det/{chip['det_id']}/image"
        assert len(chip["bbox"]) == 4
        assert set(chip["meta"]) == {"airfield", "country", "weather", "off_nadir", "gsd"}


def test_collections_detail_unknown_group_404():
    r = client.get("/api/collections/does-not-exist_2020-01-01")
    assert r.status_code == 404


def test_collections_detail_pagination_sort_and_filters():
    body = _get("/api/collections").json()
    group = next(g for g in body["groups"] if g["status"] == "needs_review")
    gid = group["group_id"]

    r = _get(f"/api/collections/{gid}?sort=conf&order=asc&limit=500")
    confs = [c["conf"] for c in r.json()["chips"]["items"]]
    assert confs == sorted(confs)

    r_desc = _get(f"/api/collections/{gid}?sort=novelty&order=desc&limit=500")
    novelties = [c["novelty"] for c in r_desc.json()["chips"]["items"]]
    assert novelties == sorted(novelties, reverse=True)

    r_flagged = _get(f"/api/collections/{gid}?only_flagged=true&limit=500")
    flagged_items = r_flagged.json()["chips"]["items"]
    assert flagged_items
    assert all(c["is_novel"] for c in flagged_items)
    assert r_flagged.json()["chips"]["total"] == group["n_flagged"]

    r_page = _get(f"/api/collections/{gid}?limit=2&offset=0")
    assert len(r_page.json()["chips"]["items"]) == 2
    assert r_page.json()["chips"]["total"] == group["n_detections"]


def test_collections_no_forbidden_ground_truth_keys():
    """Field mode must never surface ground truth or batch/domain identity
    (API.md: no role_true, no matched_ann_id, no batch names) -- checked
    recursively over both /collections and a /collections/{id} detail.
    """
    body = _get("/api/collections").json()
    _assert_no_forbidden_keys(body)

    group = body["groups"][0]
    detail = _get(f"/api/collections/{group['group_id']}?limit=500").json()
    _assert_no_forbidden_keys(detail)


def test_collections_reviews_count_det_source_and_id_space_isolation():
    """n_reviewed on a collection group counts det-chip reviews (the shared
    `reviews` table's `source='det'` rows, keyed by det_id -- see
    backend/app/db.py, landed by a concurrently-developed source-aware-review
    milestone, and collections._det_reviews_by_det_id, which reads it through
    db.reviews_by_ann_id(source="det")). Posting a det-chip review through
    HTTP is that same milestone's own route (out of scope here) -- recorded
    directly via db.upsert_review(..., source="det") instead. Also confirms
    the det_id and GT ann_id spaces are independent (API.md / db.py: a
    source='gt' row with the same numeric id must never be conflated with a
    source='det' one).
    """
    from app import db

    body = _get("/api/collections").json()
    # a needs_review group nobody has reviewed yet (test_review_queue_det_source skips one det chip earlier)
    group = next(g for g in body["groups"] if g["status"] == "needs_review" and g["n_reviewed"] == 0)
    det_id = _get(f"/api/collections/{group['group_id']}?reviewed=no&limit=1").json()["chips"]["items"][0]["det_id"]
    assert group["n_reviewed"] == 0

    # A source='gt' review sharing the same numeric id must not be counted.
    db.upsert_review(det_id, "skip", None, None, source="gt")
    unchanged = _get("/api/collections").json()
    same_group = next(g for g in unchanged["groups"] if g["group_id"] == group["group_id"])
    assert same_group["n_reviewed"] == 0

    db.upsert_review(det_id, "confirm", None, "field-mode test", source="det")

    after = _get("/api/collections").json()
    after_group = next(g for g in after["groups"] if g["group_id"] == group["group_id"])
    assert after_group["n_reviewed"] == 1

    detail = _get(f"/api/collections/{group['group_id']}?limit=500").json()
    chip = next(c for c in detail["chips"]["items"] if c["det_id"] == det_id)
    assert chip["review"] is not None
    assert chip["review"]["action"] == "confirm"

    reviewed_yes = _get(f"/api/collections/{group['group_id']}?reviewed=yes&limit=500").json()
    assert any(c["det_id"] == det_id for c in reviewed_yes["chips"]["items"])
    reviewed_no = _get(f"/api/collections/{group['group_id']}?reviewed=no&limit=500").json()
    assert all(c["det_id"] != det_id for c in reviewed_no["chips"]["items"])


def test_reference_shape():
    r = _get("/api/reference")
    body = r.json()
    assert set(body) == {
        "name", "n_images", "n_tiles", "n_chips", "military_aircraft", "splits",
        "classes", "conditions", "example_tiles", "detector", "flag_rate", "definition",
    }
    assert body["name"] == "labeled validation set (reference)"
    assert set(body["splits"]) == {"fit", "control"}
    assert set(body["conditions"]) == {"weather", "countries", "off_nadir", "gsd", "dates"}
    for key in ("off_nadir", "gsd", "dates"):
        assert set(body["conditions"][key]) == {"min", "max"}
    assert set(body["detector"]) == {"name", "val_map50", "val_map50_95", "trained_on"}


def test_reference_counts_consistent_with_tiles_parquet():
    """n_images/n_tiles/n_chips/splits must equal what a direct query of
    chips/index.parquet + tiles.parquet + annotations.parquet gives for the
    ref_train ∪ ref_calib ∪ ref_holdout domains -- the same source-of-truth
    files the rest of the backend reads (see backend/app/reference.py)."""
    from app import dataio

    chips = dataio.chips_index_df()
    tiles = dataio.tiles_df()
    ref_domains = {"ref_train", "ref_calib", "ref_holdout"}
    ref = chips[chips["domain"].isin(ref_domains)].merge(tiles[["tile_id", "parent_id"]], on="tile_id", how="left")

    body = _get("/api/reference").json()
    assert body["n_chips"] == len(ref)
    assert body["n_tiles"] == ref["tile_id"].nunique()
    assert body["n_images"] == ref["parent_id"].nunique()
    fit_ref = ref[chips.loc[ref.index, "domain"].isin({"ref_train", "ref_calib"})]
    control_ref = ref[chips.loc[ref.index, "domain"] == "ref_holdout"]
    assert body["splits"]["fit"] == fit_ref["parent_id"].nunique()
    assert body["splits"]["control"] == control_ref["parent_id"].nunique()
    # every reference image is in exactly one of the two splits
    assert body["splits"]["fit"] + body["splits"]["control"] == body["n_images"]
    # the fixture's total chip count across the three reference domains sums per-class n_chips
    assert sum(c["n_chips"] for c in body["classes"]) == body["n_chips"]


def test_reference_classes_civil_only():
    body = _get("/api/reference").json()
    classes = body["classes"]
    assert len(classes) == 3
    civil_names = {
        "Small Civil Transport/Utility", "Medium Civil Transport/Utility", "Large Civil Transport/Utility",
    }
    assert {c["label"] for c in classes} == civil_names
    assert body["military_aircraft"] == 0
    for c in classes:
        assert c["n_chips"] > 0
        assert 1 <= len(c["examples"]) <= 8
        seen_ann_ids = set()
        for ex in c["examples"]:
            assert set(ex) == {"ann_id", "chip_url"}
            assert ex["ann_id"] not in seen_ann_ids
            seen_ann_ids.add(ex["ann_id"])
            assert ex["chip_url"] == f"/api/chips/{ex['ann_id']}/image"
            # example urls resolve
            img = client.get(ex["chip_url"])
            assert img.status_code == 200


def test_reference_examples_spread_across_parent_images():
    """At least one class's 8 examples should be drawn from more than one
    parent image -- the fixture's reference domain spans 5 distinct parent
    images, so a single-image draw would defeat the point of the page."""
    from app import dataio

    chips = dataio.chips_index_df()
    tiles = dataio.tiles_df()
    ann_to_parent = chips.merge(tiles[["tile_id", "parent_id"]], on="tile_id", how="left").set_index("ann_id")["parent_id"]

    body = _get("/api/reference").json()
    spreads = []
    for c in body["classes"]:
        parents = {ann_to_parent.get(ex["ann_id"]) for ex in c["examples"]}
        spreads.append(len(parents))
    assert max(spreads) > 1


def test_reference_example_tiles():
    body = _get("/api/reference").json()
    tiles_out = body["example_tiles"]
    assert 1 <= len(tiles_out) <= 12
    for t in tiles_out:
        assert set(t) == {"tile_id", "tile_url", "airfield", "n_aircraft"}
        assert t["tile_url"] == f"/api/tiles/{t['tile_id']}/image"
        img = client.get(t["tile_url"])
        assert img.status_code == 200
    # distinct parent images -- no two example tiles share a tile_id, and
    # aircraft counts are sorted descending (most aircraft first)
    assert len({t["tile_id"] for t in tiles_out}) == len(tiles_out)
    counts = [t["n_aircraft"] for t in tiles_out]
    assert counts == sorted(counts, reverse=True)


def test_reference_flag_rate_matches_collections():
    """flag_rate must be the exact same number /api/collections reports as
    reference_flag_rate -- both computed from ref_holdout's own det chips
    (see collections._new_collections_table), so the two pages can never
    disagree."""
    ref_body = _get("/api/reference").json()
    coll_body = _get("/api/collections").json()
    assert ref_body["flag_rate"] == coll_body["reference_flag_rate"]
    assert ref_body["flag_rate"] is not None
    assert 0.0 <= ref_body["flag_rate"] <= 1.0


def test_reference_detector_and_definition():
    body = _get("/api/reference").json()
    detector = body["detector"]
    assert detector["name"] is not None
    assert detector["val_map50"] is not None
    assert detector["trained_on"] is not None
    assert body["definition"] is not None
    assert "ref_train" in body["definition"]


def test_reference_conditions_reflect_fixture():
    """Every fixture reference tile is Clear Skies / USA (see
    build_fixture.py's PLAN), so conditions should collapse to single-label
    counts equal to n_images -- and off_nadir/gsd/dates ranges must be real
    (min < max, computed, never hard-coded)."""
    body = _get("/api/reference").json()
    cond = body["conditions"]
    assert cond["weather"] == {"Clear Skies": body["n_images"]}
    assert cond["countries"] == {"USA": body["n_images"]}
    assert cond["off_nadir"]["min"] < cond["off_nadir"]["max"]
    assert cond["gsd"]["min"] <= cond["gsd"]["max"]
    assert cond["dates"]["min"] <= cond["dates"]["max"]


def test_report():
    r = client.get("/api/report")
    assert r.status_code == 200
    assert "markdown" in r.headers["content-type"]
    text = r.text
    for heading in ("Purpose & scope", "Model & reference domain", "Data card", "Method",
                    "Operational batches", "Tested envelope", "evaluation", "Detector performance",
                    "Analyst feedback", "Version history", "Limitations"):
        assert heading.lower() in text.lower(), f"missing section: {heading}"
    assert "assurance evidence prototype" in text
    # The report may explain what it is NOT ("not an accreditation decision")
    # but must never claim to itself be accredited/an accreditation.
    assert "is accredited" not in text.lower()
    assert "not an accreditation decision" in text.lower() or "not accredited" in text.lower()
    # oracle-localization / IoU-threshold sentence must appear (Purpose & Limitations)
    assert "oracle localization" in text
    assert "IoU" in text
    # Detector performance section: model card + civil-vs-military sentence with numbers
    assert "Val mAP50" in text
    assert "detected less often than civil" in text.lower()
    assert "snow batch" in text.lower()
    # by_role_group_pooled (fixture now writes both the military-batch-scoped
    # and pooled shapes) surfaces as its own sentence, distinct from the
    # military-batch-scoped one above.
    assert "pooled across every detection-evaluated batch" in text.lower()

    # Method section: read from model.json / batches.json, not hard-coded
    # (fixture's v1/v2 model.json sets all of these).
    assert "fuzzy_art (fixture stub)" in text
    assert "Calibration check" in text
    assert "n_parents" not in text  # rendered as prose, not the raw key name
    assert "AUROC" in text  # from model_selection_note
    assert "ref_train" in text  # from reference_definition
    assert "block bootstrap over ref_calib (fixture stub)" in text  # drift null_method
    assert "descriptive" in text  # drift interpretation

    # Limitations: the three new bullets required by the report redesign.
    assert "not independent confirmation" in text.lower()
    assert "descriptive/indicative" in text.lower()
    assert "fixture stub: chips used to fit new classes" in text  # exclusion_method passthrough

    # Analyst feedback section: by-pair correction table + the never-alters-the-model
    # sentence, both backed by real reviews recorded earlier in the suite (test_review_post_
    # and_log's confirm, test_review_correct_action's correct, test_review_labels' new_class).
    assert "disagreement rate" in text.lower()
    assert "corrections, predicted role" in text.lower()
    assert "neither alters the model" in text.lower()

    # Analyst-view section ("New collections since delivery"): field-mode evidence -- collections
    # table with status/p, det-chip decisions, and the field compare (fixture: v1 -> v2) using the
    # descriptive-assignment + labeled-control-set-regression-check vocabulary. Placed before Method.
    assert "## New collections since delivery (analyst view)" in text
    assert text.index("New collections since delivery") < text.index("## Method")
    assert "Status rule:" in text and "binomial" in text
    assert "Collections needing review" in text
    assert "p (indicative)" in text and "Excess vs reference" in text
    assert "Analyst decisions on detector chips" in text
    assert "Field compare, `v1` → `v2`" in text
    assert "Detections assigned (descriptive)" in text
    assert "Military Fighter/Interceptor/Attack" in text
    assert "labeled control-set regression check" in text.lower()
    assert "not an accreditation decision" in text.lower()
    assert "Top collections" in text and "military_0 field 2020-01-26" in text  # by_group rendered
    # never batch/domain names in the analyst-view section
    section = text[text.index("## New collections since delivery"):text.index("## Method")]
    for word in ("ref_holdout", "off_nadir", "military_batch", "snow batch", "recall", "validation set accuracy"):
        assert word not in section.lower().replace("ref_holdout detector chips", "").replace("ref_holdout ground-truth chips", ""), word

    # Words the report must never use.
    assert "deployed" not in text.lower()
    assert "significant" not in text.lower()
    assert " is accredited" not in text.lower()


def test_detector_files_absent_degrades_gracefully():
    """Every detector-derived field must degrade to null/empty (never a 5xx)
    when detector_model.json / detection_metrics.json / detections.parquet are
    missing -- these files aren't written by the same milestone as the rest of
    data/scores/*, so the backend has to keep working without them too.
    """
    scores_dir = Path(os.environ["VIGIL_SCORES_DIR"])
    names = ["detector_model.json", "detection_metrics.json", "detections.parquet"]
    moved = []
    for name in names:
        src = scores_dir / name
        assert src.exists(), f"fixture should have built {name}"
        dst = scores_dir / (name + ".bak")
        src.rename(dst)
        moved.append((src, dst))
    try:
        summary = _get("/api/summary").json()
        assert summary["model"]["detector_card"] is None
        # localization sentence itself is static copy, unaffected by file absence
        assert "oracle localization" in summary["model"]["localization"]

        batches = _get("/api/batches").json()
        assert batches and all(b["detection"] is None for b in batches)

        batch_detail = _get("/api/batches/military").json()
        assert batch_detail["detection"] is None
        assert batch_detail["top_novel"]
        assert all(c["detected"] is None for c in batch_detail["top_novel"])

        page = _get("/api/batches/military/chips?limit=5").json()
        assert page["items"]
        assert all(c["detected"] is None for c in page["items"])
        ann_id = page["items"][0]["ann_id"]
        tile_id = page["items"][0]["tile_id"]

        # with no detector data at all, both "yes" and "no" come back empty (not an error)
        assert _get("/api/batches/military/chips?detected=yes").json()["total"] == 0
        assert _get("/api/batches/military/chips?detected=no").json()["total"] == 0

        chip = _get(f"/api/chips/{ann_id}").json()
        assert chip["detected"] is None

        tile = _get(f"/api/tiles/{tile_id}").json()
        assert tile["detections"] is None
        assert all(c["detected"] is None for c in tile["chips"])

        env = _get("/api/envelope").json()
        assert env["thresholds"]["control"]["detection_recall"] is None
        for dim in env["dimensions"]:
            for b in dim["buckets"]:
                assert b["detection_recall"] is None

        ev = _get("/api/eval").json()
        assert ev["detection"] is None

        queue = _get("/api/review/queue?limit=5").json()
        assert all(c["detected"] is None for c in queue)

        report_text = _get("/api/report").text
        assert "Detector metrics not computed yet" in report_text
    finally:
        for src, dst in moved:
            dst.rename(src)


# --------------------------------------------------------------------------
# /adapt: exercised last, via sys.modules stubbing of vigil.pipeline, since
# (a) backend/vigil/* is owned by another agent and must not be edited, and
# (b) the real build_version() is a slow, side-effecting ML job that writes
# under the *real* data/scores/ directory regardless of VIGIL_SCORES_DIR --
# actually invoking it from this test suite would be unsafe. These tests
# stub sys.modules["vigil.pipeline"] instead (standard dependency
# injection, not a file edit) to exercise our own /adapt route logic
# (import-error handling, marking feedback applied, building the Version
# response) safely and quickly. They run last because a successful /adapt
# call appends a new "current" version with no real score files, which
# would break every other test that relies on v2 being current.
# --------------------------------------------------------------------------

_MISSING = object()


def test_adapt_501_when_pipeline_missing():
    import sys

    previous = sys.modules.get("vigil.pipeline", _MISSING)
    sys.modules["vigil.pipeline"] = None  # forces ImportError on `from vigil.pipeline import ...`
    try:
        r = client.post("/api/adapt", json={"note": "should fail"})
        assert r.status_code == 501
        assert "detail" in r.json()
    finally:
        if previous is _MISSING:
            del sys.modules["vigil.pipeline"]
        else:
            sys.modules["vigil.pipeline"] = previous


def test_adapt_success_with_stubbed_pipeline():
    import json as _json
    import sys
    import types

    scores_dir = Path(os.environ["VIGIL_SCORES_DIR"])
    captured = {}

    def _fake_build_version(version_id, feedback=None, note="", assign_min_conf=0.0):
        captured["version_id"] = version_id
        captured["feedback"] = feedback
        captured["assign_min_conf"] = assign_min_conf
        versions = _json.loads((scores_dir / "versions.json").read_text())
        entry = {
            "id": version_id,
            "created": "2026-09-05T23:00:00Z",
            "classes": [
                "Small Civil Transport/Utility",
                "Medium Civil Transport/Utility",
                "Large Civil Transport/Utility",
            ],
            "n_feedback": len(feedback or []),
            "note": note or "",
        }
        versions = [v for v in versions if v["id"] != version_id] + [entry]
        (scores_dir / "versions.json").write_text(_json.dumps(versions))
        return {"version_id": version_id}

    fake_module = types.ModuleType("vigil.pipeline")
    fake_module.build_version = _fake_build_version
    previous = sys.modules.get("vigil.pipeline", _MISSING)
    sys.modules["vigil.pipeline"] = fake_module

    batch = _get("/api/batches/haze/chips?limit=1").json()
    ann_id = batch["items"][0]["ann_id"]
    client.post(f"/api/review/{ann_id}", json={"action": "confirm"})

    try:
        r = client.post("/api/adapt", json={"note": "fixture adapt", "assign_min_conf": 0.5})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["id"] == "v3"
        assert body["note"] == "fixture adapt"
        assert captured["version_id"] == "v3"
        assert captured["assign_min_conf"] == 0.5  # POST /adapt's assign_min_conf reaches build_version
        assert any(f["ann_id"] == ann_id for f in captured["feedback"])
    finally:
        if previous is _MISSING:
            del sys.modules["vigil.pipeline"]
        else:
            sys.modules["vigil.pipeline"] = previous

    log = _get("/api/review/log").json()
    entry = next(x for x in log if x["ann_id"] == ann_id)
    assert entry["version_applied"] == "v3"
