"""Read-only data access: parquet/json -> pandas -> plain dict/list.

Everything the API returns is computed here from data/*.parquet +
data/scores/*.json at request time (per CLAUDE.md: "no hard-coded
results"). Files are cached in-process, keyed by (path, mtime), so a
running dev server picks up a freshly-built version (e.g. after /adapt)
without a restart.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from . import config
from .constants import (
    BATCH_META,
    CIVIL_ROLE_IDS,
    GSD_BUCKETS,
    OFF_NADIR_BUCKETS,
    REFERENCE_DOMAINS,
    ROLE_ALIASES,
    ROLE_NAMES,
    WINGSPAN_BUCKETS,
    DEGRADED_NOVELTY_MULTIPLIER,
    DEGRADED_ROLE_ACC_MARGIN,
    MIN_N_CHIPS,
)

# config's own import already puts backend/ on sys.path (so `vigil...` is importable), but this
# guards dataio.py against being imported before that in some future refactor.
if str(config.BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(config.BACKEND_DIR))
from vigil.jsonutil import json_safe as _sanitize_nan_core  # noqa: E402

CIVIL_ROLE_NAMES = {ROLE_NAMES[i] for i in CIVIL_ROLE_IDS}


class DataUnavailable(Exception):
    """Raised when a required data file doesn't exist (yet).

    Callers turn this into a 404/503 with a clear detail message -- the ML
    and data-layer milestones run concurrently with this backend, so
    "not built yet" is an expected, not exceptional, condition during
    development.
    """


_FILE_CACHE: dict[Path, tuple[float, Any]] = {}


def _load_cached(path: Path, loader: Callable[[Path], Any]) -> Any:
    if not path.exists():
        raise DataUnavailable(f"missing data file: {path}")
    mtime = path.stat().st_mtime
    hit = _FILE_CACHE.get(path)
    if hit is not None and hit[0] == mtime:
        return hit[1]
    obj = loader(path)
    _FILE_CACHE[path] = (mtime, obj)
    return obj


def _read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def _sanitize_nan(obj: Any) -> Any:
    """Recursively replaces float('nan')/inf with None so every JSON file we
    read round-trips through FastAPI's JSON encoder as valid, spec-compliant
    JSON (Python's json module happily reads/writes literal NaN/Infinity,
    but that isn't valid JSON and most frontend JSON.parse implementations
    reject it). Thin wrapper around vigil.jsonutil.json_safe (shared with
    vigil.pipeline/vigil.evaluate) so there is exactly one implementation.
    """
    return _sanitize_nan_core(obj)


def _read_json(path: Path) -> Any:
    with open(path) as f:
        return _sanitize_nan(json.load(f))


# ---------------------------------------------------------------- loaders --

def tiles_df() -> pd.DataFrame:
    return _load_cached(config.TILES_PARQUET, _read_parquet)


def annotations_df() -> pd.DataFrame:
    return _load_cached(config.ANNOTATIONS_PARQUET, _read_parquet)


def chips_index_df() -> pd.DataFrame:
    return _load_cached(config.CHIPS_INDEX_PARQUET, _read_parquet)


def versions_list() -> list[dict]:
    path = config.SCORES_DIR / "versions.json"
    return list(_load_cached(path, _read_json))


def current_version() -> str:
    versions = versions_list()
    if not versions:
        raise DataUnavailable("no versions recorded in scores/versions.json")
    return versions[-1]["id"]


def scores_df(version: str) -> pd.DataFrame:
    path = config.SCORES_DIR / f"{version}.parquet"
    return _load_cached(path, _read_parquet)


def batches_json(version: str) -> dict:
    path = config.SCORES_DIR / f"{version}_batches.json"
    return _load_cached(path, _read_json)


def model_json(version: str) -> dict:
    path = config.SCORES_DIR / f"{version}_model.json"
    return _load_cached(path, _read_json)


def eval_json(version: str) -> dict:
    path = config.SCORES_DIR / f"eval_{version}.json"
    return _load_cached(path, _read_json)


def model_vigilance(model: dict) -> float | None:
    """vigil.pipeline writes the calibrated vigilance as `vigilance_rho_star`
    (API.md's contract calls it `vigilance`) -- accept either key name.
    """
    v = model.get("vigilance")
    return v if v is not None else model.get("vigilance_rho_star")


def version_classes(version: str) -> list[str]:
    """The current version's known classes (civil classes + analyst classes),
    read straight off `{version}_model.json`'s `classes` field -- the same
    list `/review/classes` hands the analyst's "Correct role" picker and that
    `action="correct"` labels are validated against (see main.py's
    post_review). Empty list (never an error) if the model file is missing.
    """
    try:
        model = model_json(version)
    except DataUnavailable:
        return []
    return list(model.get("classes") or [])


def chip_role_pred(ann_id: int, version: str) -> str | None:
    """The chip's CURRENT predicted role, read straight off
    scores_df(version) (no annotations/tiles join needed) -- used to snapshot
    a review's `predicted_role` at POST time (see db.upsert_review). None if
    the chip isn't in this version's scores.
    """
    try:
        df = scores_df(version)
    except DataUnavailable:
        return None
    match = df[df["ann_id"] == ann_id]
    if match.empty:
        return None
    return role_label(match.iloc[0].get("role_pred"))


def analyst_disagreement(version: str | None = None) -> dict:
    """Analyst-feedback evidence computed from the SQLite `reviews` table
    (backend/app/db.py): how often the analyst agreed (`confirm`) vs.
    disagreed (`correct`) with the predicted role, broken down by predicted
    role and by (predicted -> corrected) pair. `confirm` and `correct` both
    just mark a chip reviewed and NEVER alter the model (no prototype/ART
    update, exactly like each other -- see vigil.pipeline.ADAPTATION_RULE);
    `correct` additionally assigns a known class label as evidence of
    disagreement, and a growing labeled set for a future retrain.

    `version` restricts to reviews whose stored `predicted_version` matches
    it (see db.upsert_review); the default (None) pools every review ever
    recorded, regardless of version -- used by `/versions/compare`, which is
    explicitly all-time, not per version (see API.md). `/eval` instead passes
    the current version, scoping the evidence to what the analyst thinks of
    THIS version's predictions.
    """
    from . import db  # local import: mirrors report.py's own pattern, avoids a hard import-time dependency

    reviews = db.all_reviews()
    if version is not None:
        reviews = [r for r in reviews if r.get("predicted_version") == version]

    n_confirm = sum(1 for r in reviews if r["action"] == "confirm")
    n_correct = sum(1 for r in reviews if r["action"] == "correct")
    n_new_class = sum(1 for r in reviews if r["action"] == "new_class")
    n_not_aircraft = sum(1 for r in reviews if r["action"] == "not_aircraft")
    n_skip = sum(1 for r in reviews if r["action"] == "skip")
    n_bad_box = sum(1 for r in reviews if r["action"] == "bad_box")
    n_reviewed = len(reviews)

    denom = n_confirm + n_correct
    disagreement_rate = (n_correct / denom) if denom else None

    pair_counts: dict[tuple[str, str], int] = {}
    role_stats: dict[str, dict[str, int]] = {}
    for r in reviews:
        action = r["action"]
        if action not in ("confirm", "correct"):
            continue
        from_role = r.get("predicted_role")
        if not from_role:
            continue  # pre-migration reviews (or a chip missing from scores) carry no predicted_role
        stats = role_stats.setdefault(from_role, {"n_confirm": 0, "n_correct": 0})
        if action == "confirm":
            stats["n_confirm"] += 1
        else:
            stats["n_correct"] += 1
            to_role = r.get("label")
            if to_role:
                key = (from_role, to_role)
                pair_counts[key] = pair_counts.get(key, 0) + 1

    by_pair = [
        {"from_role": frm, "to_role": to, "n": n}
        for (frm, to), n in sorted(pair_counts.items(), key=lambda kv: kv[1], reverse=True)
    ]
    by_predicted_role = [
        {
            "role": role,
            "n_confirm": stats["n_confirm"],
            "n_correct": stats["n_correct"],
            "disagreement_rate": (
                stats["n_correct"] / (stats["n_confirm"] + stats["n_correct"])
                if (stats["n_confirm"] + stats["n_correct"]) else None
            ),
        }
        for role, stats in role_stats.items()
    ]
    by_predicted_role.sort(key=lambda d: d["n_confirm"] + d["n_correct"], reverse=True)

    return {
        "n_reviewed": n_reviewed,
        "n_confirm": n_confirm,
        "n_correct": n_correct,
        "n_new_class": n_new_class,
        "n_not_aircraft": n_not_aircraft,
        "n_skip": n_skip,
        "n_bad_box": n_bad_box,
        "disagreement_rate": disagreement_rate,
        "by_pair": by_pair,
        "by_predicted_role": by_predicted_role,
    }


# ---------------------------------------------------------------- detector --
# The detector milestone's three files (detector_model.json, detection_metrics.json,
# detections.parquet) live directly under SCORES_DIR, not per-version -- the
# detector is trained once, not re-fit by /adapt. Per API.md, every value derived
# from them must degrade to null/empty (never a 503) if the files are absent, since
# the demo must keep working before/without the detector milestone.

def _load_optional(path: Path, loader: Callable[[Path], Any]) -> Any | None:
    if not path.exists():
        return None
    return _load_cached(path, loader)


def detector_card() -> dict | None:
    """detector_model.json as-is: name, trained_on, val_split, val_map50,
    val_map50_95, val_precision, val_recall, conf_threshold, iou_threshold, note
    -- all fields at the top level of the dict (no nested "thresholds" object).
    """
    return _load_optional(config.SCORES_DIR / "detector_model.json", _read_json)


def detection_metrics() -> dict | None:
    """detection_metrics.json as-is: {conf_threshold, iou_threshold, by_domain,
    by_condition, recall_by_role_group}.
    """
    return _load_optional(config.SCORES_DIR / "detection_metrics.json", _read_json)


def detections_df() -> pd.DataFrame | None:
    """detections.parquet as-is: kind="gt" rows (ann_id, tile_id, domain, tp) and
    kind="det" rows (x1,y1,x2,y2,conf,tp,ann_id if matched) -- None if absent.
    """
    return _load_optional(config.SCORES_DIR / "detections.parquet", _read_parquet)


def detector_summary_string(card: dict | None) -> str | None:
    """Renders detector_card() into the one-line summary used as
    `/summary.model.detector`, e.g. "YOLO11n single-class aircraft detector,
    trained on ref_train tiles only (val mAP50 0.981 on ref_calib)" -- built
    from the card's own fields (trained_on / val_split's text up to their
    first parenthetical, plus val_map50), never hard-coded.
    """
    if not card:
        return None
    name = card.get("name") or "detector"
    trained_on = (card.get("trained_on") or "").split("(")[0].strip()
    val_split = (card.get("val_split") or "").split("(")[0].strip()
    val_map50 = card.get("val_map50")
    if not trained_on:
        return name
    detail = f"trained on {trained_on}"
    if val_map50 is not None:
        detail += f" (val mAP50 {val_map50}" + (f" on {val_split})" if val_split else ")")
    return f"{name}, {detail}"


def detected_by_ann_id() -> dict[int, bool]:
    """ann_id -> tp (detected at IoU>=0.5), from detections.parquet's kind="gt"
    rows. Empty dict (never an error) if detections.parquet is absent -- callers
    then leave `Chip.detected` as None.
    """
    df = detections_df()
    if df is None or "kind" not in df.columns or "ann_id" not in df.columns or "tp" not in df.columns:
        return {}
    gt = df[df["kind"] == "gt"].dropna(subset=["ann_id"])
    if gt.empty:
        return {}
    return dict(zip(gt["ann_id"].astype("int64"), gt["tp"].astype(bool)))


def tile_detections(tile_id: int) -> list[dict] | None:
    """kind="det" rows for one tile, as {x1,y1,x2,y2,conf,tp}[] -- None (not an
    empty list) if detections.parquet itself is absent, so the frontend can tell
    "no detector data at all" apart from "detector found nothing on this tile".
    """
    df = detections_df()
    if df is None:
        return None
    if "kind" not in df.columns or "tile_id" not in df.columns:
        return []
    sub = df[(df["tile_id"] == tile_id) & (df["kind"] == "det")]
    return [
        {"x1": _f(r, "x1"), "y1": _f(r, "y1"), "x2": _f(r, "x2"), "y2": _f(r, "y2"),
         "conf": _f(r, "conf"), "tp": _b(r, "tp")}
        for _, r in sub.iterrows()
    ]


def _normalize_role_group(raw: dict | None) -> dict | None:
    """Normalizes one `recall_by_role_group*` entry from detection_metrics.json
    into `{civil: {recall, n}, military: {recall, n}, scope: str|None}`.

    The vigil pipeline (rewritten concurrently -- its output shape may change
    under us) writes `{civil, military, n_civil, n_military}` plus, once the
    military-vs-civil recall was rescoped to compare within the same batch, a
    `scope` string describing what population the two recalls were computed
    over (e.g. "within the military batch (same collections for both
    groups)" for `recall_by_role_group`, "pooled over all operational
    batches ..." for `recall_by_role_group_pooled`). Returns None if `raw` is
    falsy (key absent -- old shape / not yet provided).
    """
    if not raw:
        return None
    return {
        "civil": {"recall": raw.get("civil"), "n": raw.get("n_civil")},
        "military": {"recall": raw.get("military"), "n": raw.get("n_military")},
        "scope": raw.get("scope"),
    }


def detection_eval_summary() -> dict | None:
    """`/eval.detection`: {by_role_group, by_role_group_pooled, by_batch,
    val_map50, val_map50_95, conf_threshold, iou_threshold, note}. None if
    neither detector_card() nor detection_metrics() is available.

    `by_role_group` = detection_metrics.json's `recall_by_role_group`
    (military-batch-scoped: civil vs military recall computed only among GT
    boxes in the military batch, so both groups share the same collections).
    `by_role_group_pooled` = its `recall_by_role_group_pooled` (civil recall
    pooled across every detection-evaluated batch, acquisition-domain
    confounded) -- None if that key isn't present yet (older detection_metrics.json
    shape, before the pipeline split pooled from military-batch-scoped).
    Both carry their own `scope` string (None for the old, pre-split shape).
    """
    card = detector_card()
    metrics = detection_metrics()
    if card is None and metrics is None:
        return None
    metrics = metrics or {}
    by_domain = metrics.get("by_domain", {}) or {}
    by_role_group = _normalize_role_group(metrics.get("recall_by_role_group")) or {
        "civil": {"recall": None, "n": None}, "military": {"recall": None, "n": None}, "scope": None,
    }
    return {
        "by_role_group": by_role_group,
        "by_role_group_pooled": _normalize_role_group(metrics.get("recall_by_role_group_pooled")),
        "by_batch": {
            bid: {"recall": v.get("recall"), "precision": v.get("precision"), "n_gt": v.get("n_gt")}
            for bid, v in by_domain.items()
        },
        "val_map50": (card or {}).get("val_map50"),
        "val_map50_95": (card or {}).get("val_map50_95"),
        "conf_threshold": (card or {}).get("conf_threshold", metrics.get("conf_threshold")),
        "iou_threshold": (card or {}).get("iou_threshold", metrics.get("iou_threshold")),
        "note": (card or {}).get("note"),
    }


# ------------------------------------------------------------- field mode --
# Detector-CHIP milestone (STATE.md "Field mode" step 1-2, built concurrently
# with this backend by another agent): data/chips_det/index.parquet (one row
# per detector-box chip: det_id, tile_id, parent_id, domain, conf,
# x1,y1,x2,y2,w,h, matched_ann_id, path) and data/scores/{version}_det.parquet
# (that chip scored by the same fitted role/novelty models as the GT-chip
# scores parquet -- never used for fitting). Both degrade to an empty,
# correctly-columned DataFrame (never DataUnavailable) if the files don't
# exist yet -- see backend/app/collections.py ("/api/collections/*"), which
# must keep working before/without them.

DET_INDEX_COLUMNS = [
    "det_id", "tile_id", "parent_id", "domain", "conf",
    "x1", "y1", "x2", "y2", "w", "h", "matched_ann_id", "path",
]
DET_SCORE_COLUMNS = [
    "det_id", "tile_id", "parent_id", "batch_id", "role_pred", "role_conf",
    "novelty", "is_novel", "art_category", "knn_dist", "neighbor_ids",
    "neighbor_dists", "umap_x", "umap_y", "used_in_fit", "analyst_sim", "conf",
]


def det_chips_index_df() -> pd.DataFrame:
    """data/chips_det/index.parquet as-is -- empty (typed) frame if the
    detector-chip extraction milestone hasn't landed yet.
    """
    path = getattr(config, "CHIPS_DET_INDEX_PARQUET", config.DATA_ROOT / "chips_det" / "index.parquet")
    if not path.exists():
        return pd.DataFrame(columns=DET_INDEX_COLUMNS)
    return _load_cached(path, _read_parquet)


def det_scores_df(version: str) -> pd.DataFrame:
    """data/scores/{version}_det.parquet as-is -- empty (typed) frame if this
    version's detector-chip scores haven't been written yet.
    """
    path = config.SCORES_DIR / f"{version}_det.parquet"
    if not path.exists():
        return pd.DataFrame(columns=DET_SCORE_COLUMNS)
    return _load_cached(path, _read_parquet)


def full_det_chip_table(version: str) -> pd.DataFrame:
    """One row per DETECTOR chip (field mode): data/chips_det/index.parquet (box/conf/domain/
    parent_id) LEFT-joined with {version}_det.parquet's own score columns (role_pred/novelty/etc --
    a chip not yet scored for this version still appears, with those columns null) + tile metadata.
    The det-chip analogue of full_chip_table(), but with NO ground-truth annotation join (a detector
    chip has no GT box of its own -- `matched_ann_id` is benchmark-only, see det_chip_dict). Empty
    (never DataUnavailable) if data/chips_det/index.parquet doesn't exist yet -- see
    det_chips_index_df/det_scores_df.
    """
    idx = det_chips_index_df()
    if idx.empty:
        return idx
    scores = det_scores_df(version)
    overlap = (set(idx.columns) & set(scores.columns)) - {"det_id"}
    score_cols = [c for c in scores.columns if c not in overlap]
    df = idx.merge(scores[score_cols], on="det_id", how="left") if len(scores) else idx.copy()

    tiles = tiles_df()
    tiles_to_merge = tiles.drop(
        columns=[c for c in ("file_name", "parent_id", "domain", "batch_order") if c in tiles.columns]
    )
    df = df.merge(tiles_to_merge, on="tile_id", how="left")
    return df


def det_chip_role_pred(det_id: int, version: str) -> str | None:
    """The det chip's CURRENT predicted role, read straight off det_scores_df(version) -- used to
    snapshot a det-chip review's `predicted_role` at POST time (see main.py's post_review,
    source='det'). None if the chip isn't in this version's det scores.
    """
    scores = det_scores_df(version)
    if scores.empty:
        return None
    match = scores[scores["det_id"] == det_id]
    if match.empty:
        return None
    return role_label(match.iloc[0].get("role_pred"))


def det_chip_image_path(det_id: int) -> Path | None:
    df = det_chips_index_df()
    if df.empty:
        return None
    row = df.loc[df["det_id"] == det_id]
    if row.empty:
        return None
    return _resolve_path(str(row.iloc[0]["path"]))


def det_chip_dict(row: Any, version: str, review: dict | None = None, benchmark: bool = False) -> dict:
    """Chip-shaped dict for ONE detector chip (field mode `/det/{det_id}`), mirroring chip_dict()'s
    shape where it makes sense. Differences the field-mode story requires: `source='det'`, `det_id`,
    `conf` (the detector's own confidence) always shown; `matched_ann_id` (benchmark-only -- which GT
    box, if any, this detection was matched to at IoU>=0.5) hidden unless `benchmark=True`; no
    `role_true`/`partial`/`truncated`/`wingspan_m`/`detected` (a detector chip has no ground truth of
    its own in field mode -- see backend/API.md).
    """
    det_id = int(row["det_id"])
    tile_id = int(row["tile_id"])
    umap_x, umap_y = _f(row, "umap_x"), _f(row, "umap_y")
    umap = [umap_x, umap_y] if umap_x is not None and umap_y is not None else None
    x1, y1, x2, y2 = _f(row, "x1"), _f(row, "y1"), _f(row, "x2"), _f(row, "y2")
    w = _f(row, "w")
    h = _f(row, "h")
    if w is None and x1 is not None and x2 is not None:
        w = x2 - x1
    if h is None and y1 is not None and y2 is not None:
        h = y2 - y1
    out = {
        "source": "det",
        "det_id": det_id,
        "ann_id": det_id,  # alias == det_id, so the shared chip-drawer/review components work unchanged
        "tile_id": tile_id,
        "version": version,
        "bbox": [x1, y1, w, h],
        "conf": _f(row, "conf"),
        "chip_url": f"/api/det/{det_id}/image",
        "tile_url": f"/api/tiles/{tile_id}/image",
        "role_pred": role_label(row.get("role_pred")),
        "role_conf": _f(row, "role_conf"),
        "novelty": _f(row, "novelty"),
        "is_novel": _b(row, "is_novel"),
        "art_category": _i(row, "art_category"),
        "knn_dist": _f(row, "knn_dist"),
        "umap": umap,
        "used_in_fit": _b(row, "used_in_fit") or False,
        "meta": {
            "airfield": row.get("airfield"),
            "country": row.get("country"),
            "weather": row.get("weather"),
            "off_nadir": _f(row, "off_nadir"),
            "gsd": _f(row, "gsd"),
        },
        "review": review,
    }
    if benchmark:
        # Benchmark-only evidence: which GT box this detection matched (IoU>=0.5) and which
        # hand-built benchmark batch/domain its parent image belongs to. Both are omitted (not
        # null) by default -- a field-mode analyst has neither.
        m = row.get("matched_ann_id")
        out["matched_ann_id"] = None if _isna(m) else int(m)
        dom = row.get("domain")
        if _isna(dom):
            dom = row.get("batch_id")
        out["batch_id"] = None if _isna(dom) else str(dom)
    return out


def field_block(version: str) -> dict | None:
    """API.md's `FieldBlock` for one version -- the analyst-view compare/report shape, composed from
    eval_{version}.json's `field` + `control` blocks (written by vigil.field via vigil.evaluate):
    `{n_detections, n_excluded_labeled, n_excluded_same_parent, n_eligible, classes: [{label,
    n_labels_used, n_assigned, assignment_rate, by_group: [{group_id, airfield, date, n_assigned}]}],
    control: {label, overall, per_class: [{cls, accuracy, n}]} | None, reviewed_flagged: {n,
    n_not_aircraft, n_bad_box, field_precision_on_flagged}, note}`.

    Deliberately DROPS `field.domains`/`field.by_domain` (benchmark batch names -- an analyst never
    sees those; they stay in the eval artifact for the Benchmark view). None if the version has no
    eval json or predates field mode (no `field` block).
    """
    try:
        ej = eval_json(version)
    except DataUnavailable:
        return None
    field = ej.get("field")
    if not isinstance(field, dict):
        return None
    classes = []
    for cls in field.get("classes") or []:
        if not isinstance(cls, dict):
            continue
        classes.append({
            "label": cls.get("label"),
            "n_labels_used": cls.get("n_labels_used"),
            "n_assigned": cls.get("n_assigned"),
            "assignment_rate": cls.get("assignment_rate"),
            "by_group": [
                {"group_id": g.get("group_id"), "airfield": g.get("airfield"), "date": g.get("date"),
                 "n_assigned": g.get("n_assigned")}
                for g in (cls.get("by_group") or []) if isinstance(g, dict)
            ],
        })
    control_raw = ej.get("control")
    control = None
    if isinstance(control_raw, dict):
        control = {
            "label": control_raw.get("label") or "labeled control-set regression check",
            "overall": control_raw.get("accuracy_ref_holdout"),
            "per_class": _normalize_per_class(control_raw.get("per_class", {})),
        }
    rf_raw = field.get("reviewed_flagged")
    reviewed_flagged = {
        "n": (rf_raw or {}).get("n", 0) if isinstance(rf_raw, dict) else 0,
        "n_not_aircraft": (rf_raw or {}).get("n_not_aircraft", 0) if isinstance(rf_raw, dict) else 0,
        "n_bad_box": (rf_raw or {}).get("n_bad_box", 0) if isinstance(rf_raw, dict) else 0,
        "field_precision_on_flagged": (rf_raw or {}).get("field_precision_on_flagged") if isinstance(rf_raw, dict) else None,
    }
    labeled_collections = [
        {
            "group_id": lc.get("group_id"), "airfield": lc.get("airfield"), "date": lc.get("date"),
            "n_labeled": lc.get("n_labeled"), "n_bad_box": lc.get("n_bad_box"),
            "n_bad_box_assigned": lc.get("n_bad_box_assigned"),
            "n_not_aircraft": lc.get("n_not_aircraft"), "n_flagged_unlabeled": lc.get("n_flagged_unlabeled"),
            "n_flagged_unlabeled_assigned": lc.get("n_flagged_unlabeled_assigned"),
            "per_class": lc.get("per_class") or {},
        }
        for lc in (field.get("labeled_collections") or []) if isinstance(lc, dict)
    ]
    return {
        "n_detections": field.get("n_detections"),
        "n_excluded_labeled": field.get("n_excluded_labeled"),
        "n_excluded_same_parent": field.get("n_excluded_same_parent"),
        "n_eligible": field.get("n_eligible"),
        "classes": classes,
        "control": control,
        "reviewed_flagged": reviewed_flagged,
        # DISPLAY ONLY -- see vigil.field._labeled_collections_block; never folded into `classes` above.
        "labeled_collections": labeled_collections,
        # This version's optional additional detector-confidence gate on top of ADAPTATION_RULE's
        # nearest-exemplar rule (vigil.pipeline.build_version's assign_min_conf; 0/None = off).
        "assign_min_conf": field.get("assign_min_conf"),
        "note": field.get("note") or "assignments are descriptive; not ground-truth accuracy",
    }


def field_compare(a: str, b: str) -> dict:
    """`/versions/compare`'s `field` key: `{a: FieldBlock|None, b: FieldBlock|None, worst_regression:
    {cls, delta, a, b}|None}` -- worst_regression is the class whose labeled control-set accuracy
    dropped most a -> b (delta = a - b; positive = forgetting)."""
    fa = field_block(a)
    fb = field_block(b)

    def _pc(fx: dict | None) -> dict:
        if not fx or not fx.get("control"):
            return {}
        return {p["cls"]: {"accuracy": p.get("accuracy"), "n": p.get("n")} for p in fx["control"]["per_class"]}

    return {"a": fa, "b": fb, "worst_regression": worst_control_regression(_pc(fa), _pc(fb))}


def worst_control_regression(pc_a: dict, pc_b: dict) -> dict | None:
    """Field-mode `/versions/compare` helper: the single class with the largest drop, a -> b, in the
    labeled control-set regression check (`control.per_class`, ref_holdout GT chips -- see
    vigil.field). `pc_a`/`pc_b` are {cls: {accuracy, n}} dicts (vigil.evaluate._role_head_eval's own
    shape). Returns {cls, delta (a - b; positive = forgetting), a, b}, or None if there's no common,
    non-null class to compare (e.g. neither version has a `control` block yet).
    """
    diffs = []
    for cls in set(pc_a) & set(pc_b):
        av = (pc_a.get(cls) or {}).get("accuracy")
        bv = (pc_b.get(cls) or {}).get("accuracy")
        if av is None or bv is None:
            continue
        diffs.append((cls, av - bv, av, bv))
    if not diffs:
        return None
    cls, delta, av, bv = max(diffs, key=lambda t: t[1])
    return {"cls": cls, "delta": delta, "a": av, "b": bv}


def _normalize_per_role(raw: Any) -> list[dict]:
    """Accepts either a role-name/role_id-keyed dict or a list of
    {role|role_id, n, auroc} objects (the latter is what vigil's eval
    script actually writes) and returns API.md's `{role, n, auroc}[]`.
    """
    out: list[dict] = []
    if isinstance(raw, dict):
        pairs = list(raw.items())
    elif isinstance(raw, list):
        pairs = [(item.get("role", item.get("role_id")), item) for item in raw if isinstance(item, dict)]
    else:
        return out
    for key, v in pairs:
        if not isinstance(v, dict):
            continue
        role = v.get("role")
        if role is None:
            role = role_label(v.get("role_id", key))
        out.append({"role": role, "n": v.get("n"), "auroc": v.get("auroc")})
    return out


def _normalize_per_class(raw: Any) -> list[dict]:
    """Accepts either a class-name-keyed dict (what vigil's eval script
    actually writes) or a list of {cls, n, accuracy} objects and returns
    a normalized `{cls, n, accuracy}[]`, canonicalizing class names through
    role_label() (see ROLE_ALIASES).
    """
    out: list[dict] = []
    if isinstance(raw, dict):
        for k, v in raw.items():
            cls = role_label(k) or str(k)
            if isinstance(v, dict):
                out.append({"cls": cls, "n": v.get("n"), "accuracy": v.get("accuracy")})
            else:
                out.append({"cls": cls, "n": None, "accuracy": v})
    elif isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            cls = item.get("cls") or item.get("role") or item.get("class")
            out.append({"cls": role_label(cls) or cls, "n": item.get("n"), "accuracy": item.get("accuracy")})
    return out


def eval_summary(version: str) -> dict:
    """Normalizes eval_{version}.json into API.md's /eval contract shape:
    {novelty: {auroc_military_vs_civil, auprc, auroc_size_matched,
    auroc_wingspan_stratified, wingspan_stratified_method, per_role,
    baseline_knn_auroc}, role_head: {accuracy_ref_holdout, per_class},
    military: {rare_target_recall, recall_new_class, miss_rate_civil, n_eval}}.
    Every metric here already excludes chips used to fit a new prototype/ART
    category from analyst feedback (see vigil.evaluate.evaluate).
    """
    ej = eval_json(version)
    nov = ej.get("novelty", {}) or {}
    rh = ej.get("role_head", {}) or {}
    mil = ej.get("military", {}) or {}
    return {
        "novelty": {
            "auroc_military_vs_civil": nov.get("auroc_military_vs_civil"),
            "auprc": nov.get("auprc"),
            "auroc_size_matched": nov.get("auroc_size_matched"),
            "auroc_wingspan_stratified": nov.get("auroc_wingspan_stratified", nov.get("auroc_size_matched")),
            "wingspan_stratified_method": nov.get("wingspan_stratified_method"),
            "per_role": _normalize_per_role(nov.get("per_role", [])),
            "baseline_knn_auroc": nov.get("baseline_knn_auroc"),
            "size_matched_bins": nov.get("size_matched_bins", []),
            "novelty_flag_rate": nov.get("novelty_flag_rate") or ej.get("novelty_flag_rate"),
        },
        "role_head": {
            "accuracy_ref_holdout": rh.get("accuracy_ref_holdout"),
            "per_class": _normalize_per_class(rh.get("per_class", {})),
        },
        "military": {
            **{k: mil.get(k) for k in ("analyst_class_assignment_rate", "civil_false_assignment_rate", "adaptation_rule",
                                       "alias_note", "n_eval_civil", "assignment_rate_by_role") if k in mil},
            # exclusion disclosure lives at the eval-json top level (vigil.evaluate); military block may mirror it
            **{k: mil.get(k, ej.get(k)) for k in ("n_excluded_used_in_fit", "n_excluded_same_parent", "exclusion_method")
               if k in mil or k in ej},
            "rare_target_recall": mil.get("rare_target_recall", mil.get("analyst_class_assignment_rate")),
            "recall_new_class": mil.get("recall_new_class", mil.get("rare_target_recall")),
            "miss_rate_civil": mil.get("miss_rate_civil"),
            "n_eval": mil.get("n_eval"),
        },
    }


# ------------------------------------------------------------- join logic --

def full_chip_table(version: str) -> pd.DataFrame:
    """One row per scored chip: scores + annotations (role_true) + chip
    index (domain) + tile metadata. `domain` is read from chips/index.parquet
    (present per-chip today) in preference to tiles.parquet's own `domain`
    column (a milestone still landing at the time this was written) so the
    backend works either way.
    """
    scores = scores_df(version)
    ann = annotations_df()
    tiles = tiles_df()

    df = scores.merge(ann, on=["ann_id", "tile_id"], how="inner", suffixes=("", "_ann"))

    try:
        chips = chips_index_df()
        if "domain" in chips.columns:
            df = df.merge(chips[["ann_id", "domain"]], on="ann_id", how="left")
    except DataUnavailable:
        pass

    drop_cols = [c for c in ("file_name", "domain", "batch_order") if c in tiles.columns and c in df.columns]
    tiles_to_merge = tiles.drop(columns=[c for c in ("file_name",) if c in tiles.columns])
    if "domain" in df.columns and "domain" in tiles_to_merge.columns:
        tiles_to_merge = tiles_to_merge.drop(columns=["domain"])
    if "batch_order" in tiles_to_merge.columns:
        tiles_to_merge = tiles_to_merge.drop(columns=["batch_order"])
    df = df.merge(tiles_to_merge, on="tile_id", how="left")

    if "domain" not in df.columns and "batch_id" in df.columns:
        df["domain"] = df["batch_id"]

    return df


def reference_chip_table(version: str) -> pd.DataFrame:
    """Rows for the reference domain (ref_train ∪ ref_calib), if the scores
    file carries them. Returns an empty frame (not an error) if it doesn't
    -- callers (e.g. /embedding/points, /batches/{id}'s ref_novelty_hist)
    fall back gracefully. NOT used by /envelope, which must never pool this
    fit-set domain into a bucket's evidence -- see operational_chip_table().
    """
    try:
        df = full_chip_table(version)
    except DataUnavailable:
        return pd.DataFrame()
    if "domain" not in df.columns:
        return df.iloc[0:0]
    return df[df["domain"].isin(REFERENCE_DOMAINS)]


def operational_chip_table(version: str) -> pd.DataFrame:
    df = full_chip_table(version)
    if "domain" in df.columns:
        return df[~df["domain"].isin(REFERENCE_DOMAINS)]
    return df


# ------------------------------------------------------------ conversions --

def role_label(value: Any) -> str | None:
    """role_pred / eval per_class-key values may be a role_id (int), a full
    RarePlanes role name, or (as vigil.proto.ROLE_NAMES actually emits) a
    short alias like "Small Civil" -- accept any of these and always return
    the canonical full name used by annotations.parquet's `role` column, so
    role-accuracy comparisons match. A genuinely new class label (added via
    analyst feedback) has no alias and passes through unchanged.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (int, np.integer)):
        return ROLE_NAMES.get(int(value), str(value))
    if isinstance(value, str):
        try:
            return ROLE_NAMES.get(int(value), ROLE_ALIASES.get(value, value))
        except ValueError:
            return ROLE_ALIASES.get(value, value)
    return str(value)


def _isna(v: Any) -> bool:
    """pd.isna() that never raises on scalars it doesn't like (e.g. plain
    Python containers) -- used everywhere we pull a value out of a pandas
    row, since those may be None, float('nan'), or pandas' own NA/NaT
    sentinels (nullable Int64 columns like `art_category` use pd.NA).
    """
    try:
        r = pd.isna(v)
        return bool(r) if not hasattr(r, "__len__") else False
    except (TypeError, ValueError):
        return False


def _f(row: Any, key: str) -> float | None:
    v = row.get(key)
    if _isna(v):
        return None
    return float(v)


def _i(row: Any, key: str) -> int | None:
    v = row.get(key)
    if _isna(v):
        return None
    return int(v)


def _b(row: Any, key: str) -> bool | None:
    v = row.get(key)
    if _isna(v):
        return None
    return bool(v)


def chip_dict(row: Any, version: str, review: dict | None = None, detected: bool | None = None) -> dict:
    ann_id = int(row["ann_id"])
    tile_id = int(row["tile_id"])
    umap_x, umap_y = _f(row, "umap_x"), _f(row, "umap_y")
    umap = [umap_x, umap_y] if umap_x is not None and umap_y is not None else None
    return {
        "ann_id": ann_id,
        "tile_id": tile_id,
        "batch_id": row.get("domain") or row.get("batch_id"),
        "version": version,
        "bbox": [_f(row, "x"), _f(row, "y"), _f(row, "w"), _f(row, "h")],
        "chip_url": f"/api/chips/{ann_id}/image",
        "tile_url": f"/api/tiles/{tile_id}/image",
        "role_pred": role_label(row.get("role_pred")),
        "role_conf": _f(row, "role_conf"),
        "novelty": _f(row, "novelty"),
        "is_novel": _b(row, "is_novel"),
        "art_category": _i(row, "art_category"),
        "knn_dist": _f(row, "knn_dist"),
        "umap": umap,
        "partial": _f(row, "partial"),
        "truncated": _b(row, "truncated") or False,
        "used_in_fit": _b(row, "used_in_fit") or False,
        "detected": detected,
        "meta": {
            "airfield": row.get("airfield"),
            "country": row.get("country"),
            "weather": row.get("weather"),
            "off_nadir": _f(row, "off_nadir"),
            "gsd": _f(row, "gsd"),
            "wingspan_m": _f(row, "wingspan"),
        },
        "role_true": row.get("role"),
        "review": review,
    }


def histogram(values: pd.Series, n_bins: int = 10, lo: float = 0.0, hi: float = 1.0) -> dict:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        edges = np.linspace(lo, hi, n_bins + 1)
        return {"bins": [round(float(e), 4) for e in edges], "counts": [0] * n_bins}
    counts, edges = np.histogram(clean.to_numpy(), bins=n_bins, range=(lo, hi))
    return {"bins": [round(float(e), 4) for e in edges], "counts": [int(c) for c in counts]}


def filter_min_partial(df: pd.DataFrame, min_partial: float) -> pd.DataFrame:
    """Display/triage filter only (see API.md's `min_partial` query param on
    `/batches/{id}/chips`, `/review/queue`, and `/batches/{id}`'s
    `top_novel`): drops chips whose annotation is mostly cut off at the tile
    edge -- RarePlanes' `partial` is the fraction of the aircraft's true
    extent that falls inside the tile (1.0 = fully inside; `truncated` just
    marks partial < 1). `min_partial=0` is a no-op (every partial value is
    >= 0). Never applied to scoring/batch metrics/envelope/eval, which must
    keep every chip -- see full_chip_table/operational_chip_table callers
    that do NOT route through this.
    """
    if min_partial <= 0 or "partial" not in df.columns:
        return df
    partial = pd.to_numeric(df["partial"], errors="coerce")
    return df[partial.fillna(1.0) >= min_partial]


def _novelty_bin_edges(combined: pd.Series, n_bins: int = 30) -> list[float]:
    """`n_bins` equal-width edges spanning [min, max] of `combined` plus a
    small margin (2% of the range, or a flat +/-0.05 if every value is
    identical) -- real novelty scores can cluster in a narrow band (e.g.
    ~0.37-0.45 across the whole reference+batch pool), so fixed [0,1] bins
    put everything into 1-2 bins and make the histogram unreadable.
    """
    clean = pd.to_numeric(combined, errors="coerce").dropna()
    if clean.empty:
        lo, hi = 0.0, 1.0
    else:
        lo, hi = float(clean.min()), float(clean.max())
        if lo == hi:
            pad = abs(lo) * 0.05 or 0.05
            lo, hi = lo - pad, hi + pad
        else:
            pad = (hi - lo) * 0.02
            lo, hi = lo - pad, hi + pad
    return np.linspace(lo, hi, n_bins + 1).tolist()


def _histogram_on_edges(values: pd.Series, edges: list[float]) -> dict:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    counts, _ = np.histogram(clean.to_numpy(), bins=np.asarray(edges))
    total = int(counts.sum())
    return {
        "bins": [round(float(e), 6) for e in edges],
        "counts": [int(c) for c in counts],
        "density": [round(float(c) / total, 6) if total else 0.0 for c in counts],
    }


def novelty_histograms(batch_values: pd.Series, ref_values: pd.Series, n_bins: int = 30) -> tuple[dict, dict]:
    """`/batches/{id}`'s `novelty_hist` + `ref_novelty_hist`: 30 equal-width
    bins spanning [min, max] of novelty pooled over (reference U this
    batch), both histograms sharing the SAME edges so they're directly
    comparable on one chart -- see _novelty_bin_edges. Each histogram is
    `{bins: edges[] (n_bins+1), counts: number[] (n_bins), density: counts/total}`.
    """
    batch_values = batch_values if batch_values is not None else pd.Series(dtype=float)
    ref_values = ref_values if ref_values is not None else pd.Series(dtype=float)
    combined = pd.concat([
        pd.to_numeric(batch_values, errors="coerce"),
        pd.to_numeric(ref_values, errors="coerce"),
    ])
    edges = _novelty_bin_edges(combined, n_bins)
    return _histogram_on_edges(batch_values, edges), _histogram_on_edges(ref_values, edges)


def neighbors_for_chip(row: Any, version: str, k: int = 5) -> list[dict]:
    """5 nearest ref_train chips to `row`, as {ann_id, chip_url, role_true,
    dist}. True embedding-space (cosine) neighbors, precomputed once per
    version by vigil.pipeline (sklearn NearestNeighbors over DINOv2
    embeddings) and stored on the scores parquet as `neighbor_ids` /
    `neighbor_dists` -- not a UMAP-distance proxy.
    """
    neighbor_ids = row.get("neighbor_ids")
    if neighbor_ids is None or _isna(neighbor_ids):
        return []
    try:
        ids = list(neighbor_ids)
    except TypeError:
        return []
    if not ids:
        return []
    neighbor_dists = row.get("neighbor_dists")
    try:
        dists = list(neighbor_dists) if neighbor_dists is not None else [None] * len(ids)
    except TypeError:
        dists = [None] * len(ids)

    lookup = full_chip_table(version).set_index("ann_id")
    out = []
    for nid, dist in list(zip(ids, dists))[:k]:
        nid = int(nid)
        if nid not in lookup.index:
            continue
        r = lookup.loc[nid]
        out.append({
            "ann_id": nid,
            "chip_url": f"/api/chips/{nid}/image",
            "role_true": r.get("role"),
            "dist": float(dist) if dist is not None and not _isna(dist) else None,
        })
    return out


# --------------------------------------------------------------- batches --

def batch_list(version: str) -> list[dict]:
    try:
        bj = batches_json(version)
    except DataUnavailable:
        bj = {}
    det_by_domain = (detection_metrics() or {}).get("by_domain", {}) or {}
    out = []
    for bid, meta in BATCH_META.items():
        if bid not in bj:
            continue
        b = bj[bid]
        drift = b.get("drift", {}) or {}
        det = det_by_domain.get(bid)
        detection = None
        if det is not None:
            detection = {
                "recall": det.get("recall"),
                "precision": det.get("precision"),
                "n_gt": det.get("n_gt"),
                "n_det": det.get("n_det"),
            }
        out.append({
            "id": bid,
            "order": meta["order"],
            "name": meta["name"],
            "description": meta["description"],
            "condition": meta["condition"],
            "n_tiles": b.get("n_tiles", 0),
            "n_chips": b.get("n_chips", 0),
            "n_parent_images": b.get("n_parent_images", 0),
            "drift": {
                "score": drift.get("score"),
                "null_p95": drift.get("null_p95"),
                "p_value": drift.get("p_value"),
                "n_boot": drift.get("n_boot"),
                "statistic": drift.get("statistic"),
                "null_method": drift.get("null_method"),
                "calib_mean": drift.get("calib_mean"),
                "ratio_to_calib": drift.get("ratio_to_calib"),
            },
            "novelty_rate": b.get("novelty_rate"),
            "role_accuracy": b.get("role_accuracy"),
            "detection": detection,
            "conf_shift": b.get("conf_shift"),
        })
    out.sort(key=lambda d: d["order"])
    return out


def batch_meta_or_404(batch_id: str) -> dict:
    if batch_id not in BATCH_META:
        raise KeyError(batch_id)
    return BATCH_META[batch_id]


# --------------------------------------------------------------- envelope --

def _bucket_numeric(value: float | None, buckets: list[tuple[str, Callable]]) -> str | None:
    if value is None:
        return None
    try:
        if np.isnan(value):
            return None
    except TypeError:
        return None
    for label, fn in buckets:
        if fn(value):
            return label
    return None


def _control_metrics(version: str) -> dict:
    """The evidence baseline every /envelope bucket's status is judged
    against: ref_holdout's OWN batch-level metrics (API.md's "control batch"
    -- same reference domain as ref_train/ref_calib, but never used for
    fitting or calibration). Read from the same precomputed files /batches
    shows for the ref_holdout row, so /envelope and /batches never disagree
    about what ref_holdout's numbers are.
    """
    try:
        bj = batches_json(version)
    except DataUnavailable:
        bj = {}
    rh = bj.get("ref_holdout") or {}
    dm = detection_metrics() or {}
    det_rh = (dm.get("by_domain") or {}).get("ref_holdout") or {}
    return {
        "role_accuracy": rh.get("role_accuracy"),
        "novelty_rate": rh.get("novelty_rate"),
        "detection_recall": det_rh.get("recall"),
        "n_chips": rh.get("n_chips"),
        "n_parent_images": rh.get("n_parent_images"),
    }


def _dimension(name: str, df: pd.DataFrame, bucket_fn: Callable, control: dict,
               reference_labels: frozenset = frozenset(), order: list[str] | None = None,
               detection_by_label: dict | None = None) -> dict:
    """`df` must already be restricted to OPERATIONAL batches only (ref_holdout,
    off_nadir, foreign, haze, snow, military -- never ref_train/ref_calib; see
    operational_chip_table()), grouped here by a physical-condition bucket
    (weather/off-nadir/country/gsd/wingspan), not by batch id -- a bucket
    legitimately pools rows from several batches that happen to share that
    condition (e.g. the military batch's own tiles are mostly "Clear Skies").

    Status is computed purely from evidence against `control` (ref_holdout's
    own metrics, from _control_metrics -- never bypassed just because a
    bucket's label happens to match the reference definition; that's flagged
    separately via `in_reference_definition`, see `reference_labels`).
    """
    if df.empty:
        return {"name": name, "buckets": []}
    labels = df.apply(bucket_fn, axis=1)
    work = df.assign(_bucket=labels)
    work = work[work["_bucket"].notna()]
    ctrl_role_acc = control.get("role_accuracy")
    ctrl_detection_recall = control.get("detection_recall")
    ctrl_novelty_rate = control.get("novelty_rate")
    buckets_out = []
    for label, g in work.groupby("_bucket"):
        n_chips = int(len(g))
        n_images = int(g["tile_id"].nunique()) if "tile_id" in g.columns else 0
        n_parent_images = int(g["parent_id"].nunique()) if "parent_id" in g.columns else 0
        novelty_rate = None
        if "is_novel" in g.columns and g["is_novel"].notna().any():
            novelty_rate = float(pd.to_numeric(g["is_novel"], errors="coerce").mean())
        role_accuracy = None
        if "role" in g.columns and "role_pred" in g.columns:
            civil = g[g["role"].isin(CIVIL_ROLE_NAMES)]
            if len(civil) > 0:
                preds = civil["role_pred"].apply(role_label)
                role_accuracy = float((preds == civil["role"]).mean())
        detection_recall = None
        if detection_by_label:
            det = detection_by_label.get(str(label))
            if det is not None:
                detection_recall = det.get("recall")

        if n_chips < MIN_N_CHIPS:
            status = "untested"
        elif (
            (role_accuracy is not None and ctrl_role_acc is not None
             and role_accuracy < ctrl_role_acc - DEGRADED_ROLE_ACC_MARGIN)
            or (detection_recall is not None and ctrl_detection_recall is not None
                and detection_recall < ctrl_detection_recall - DEGRADED_ROLE_ACC_MARGIN)
            or (novelty_rate is not None and ctrl_novelty_rate is not None
                and novelty_rate > DEGRADED_NOVELTY_MULTIPLIER * ctrl_novelty_rate)
        ):
            status = "degraded"
        else:
            status = "tested"

        control_delta = {
            "role_accuracy": (
                role_accuracy - ctrl_role_acc if role_accuracy is not None and ctrl_role_acc is not None else None
            ),
            "detection_recall": (
                detection_recall - ctrl_detection_recall
                if detection_recall is not None and ctrl_detection_recall is not None else None
            ),
        }

        buckets_out.append({
            "label": str(label),
            "n_chips": n_chips,
            "n_images": n_images,
            "n_parent_images": n_parent_images,
            "novelty_rate": novelty_rate,
            "role_accuracy": role_accuracy,
            "detection_recall": detection_recall,
            "control_delta": control_delta,
            "in_reference_definition": label in reference_labels,
            "status": status,
        })
    if order:
        rank = {lbl: i for i, lbl in enumerate(order)}
        buckets_out.sort(key=lambda b: rank.get(b["label"], len(order)))
    else:
        buckets_out.sort(key=lambda b: b["label"])
    return {"name": name, "buckets": buckets_out}


def envelope(version: str) -> dict:
    # Operational batches only (ref_holdout, off_nadir, foreign, haze, snow,
    # military) -- never ref_train/ref_calib, which are the model's own fit
    # set and must never be pooled into a bucket's evidence (see _dimension).
    df_ops = operational_chip_table(version)
    control = _control_metrics(version)

    # detection_recall per bucket comes straight from detection_metrics.json's
    # by_condition breakdown (precomputed over detector output vs GT boxes),
    # not derived from these chip rows -- see detection_eval_summary(). Its bucket
    # keys don't carry the "deg"/"m" unit suffixes envelope's own bucket labels do
    # (e.g. "<=15" vs "<=15deg"), so re-key onto envelope's labels before lookup.
    dm = detection_metrics() or {}
    by_condition = dm.get("by_condition", {}) or {}

    def _resuffix(raw: dict, suffix: str) -> dict:
        return {f"{k}{suffix}": v for k, v in raw.items()}

    dims = [
        _dimension("weather", df_ops, lambda r: r.get("weather"), control,
                   reference_labels=frozenset({"Clear Skies"}),
                   detection_by_label=by_condition.get("weather")),
        _dimension("off_nadir", df_ops, lambda r: _bucket_numeric(r.get("off_nadir"), OFF_NADIR_BUCKETS),
                   control, reference_labels=frozenset({"<=15deg", "15-25deg"}),
                   order=[b[0] for b in OFF_NADIR_BUCKETS],
                   detection_by_label=_resuffix(by_condition.get("off_nadir_b", {}), "deg")),
        _dimension("country", df_ops, lambda r: "USA" if r.get("country") == "USA" else "other",
                   control, reference_labels=frozenset({"USA"}), order=["USA", "other"],
                   detection_by_label=by_condition.get("country_g")),
        _dimension("gsd", df_ops, lambda r: _bucket_numeric(r.get("gsd"), GSD_BUCKETS),
                   control, order=[b[0] for b in GSD_BUCKETS],
                   detection_by_label=_resuffix(by_condition.get("gsd_b", {}), "m")),
        # aircraft size (wingspan) has no detection breakdown in detection_metrics.json
        # -- detection_recall stays null for every bucket of this dimension.
        _dimension("aircraft_size", df_ops, lambda r: _bucket_numeric(r.get("wingspan"), WINGSPAN_BUCKETS),
                   control, order=[b[0] for b in WINGSPAN_BUCKETS]),
    ]
    thresholds = {
        "min_n": MIN_N_CHIPS,
        "degraded_margin": DEGRADED_ROLE_ACC_MARGIN,
        "novelty_multiplier": DEGRADED_NOVELTY_MULTIPLIER,
        "control": control,
    }
    return {"dimensions": dims, "thresholds": thresholds}


# -------------------------------------------------------------------- map --

def map_points(version: str) -> list[dict]:
    tiles = tiles_df()
    ann = annotations_df()

    tile_counts = ann.groupby("tile_id").size().rename("n_chips_calc")
    t = tiles.merge(tile_counts, on="tile_id", how="left")
    t["n_chips_calc"] = t["n_chips_calc"].fillna(0).astype(int)

    try:
        chips = chips_index_df()
    except DataUnavailable:
        chips = pd.DataFrame(columns=["ann_id", "tile_id", "domain"])

    domain_by_tile: dict[int, str] = {}
    if "domain" in chips.columns and "tile_id" in chips.columns:
        domain_by_tile = chips.groupby("tile_id")["domain"].first().to_dict()

    novelty_by_tile: dict[int, float] = {}
    try:
        scores = scores_df(version)
        merged = scores.copy()
        if "is_novel" in merged.columns:
            novelty_by_tile = merged.groupby("tile_id")["is_novel"].mean().to_dict()
    except DataUnavailable:
        pass

    try:
        bj = batches_json(version)
    except DataUnavailable:
        bj = {}

    rows = []
    for parent_id, g in t.groupby("parent_id"):
        lat = g["lat"].dropna()
        lon = g["lon"].dropna()
        airfield = g["airfield"].dropna()
        country = g["country"].dropna()
        tile_ids = g["tile_id"].tolist()
        domains = [domain_by_tile[tid] for tid in tile_ids if tid in domain_by_tile]
        domain = domains[0] if domains else None
        nov_vals = [novelty_by_tile[tid] for tid in tile_ids if tid in novelty_by_tile]
        drift_score = None
        if domain and domain in bj:
            drift_score = (bj[domain].get("drift") or {}).get("score")
        rows.append({
            "parent_id": parent_id,
            "lat": float(lat.iloc[0]) if len(lat) else None,
            "lon": float(lon.iloc[0]) if len(lon) else None,
            "airfield": airfield.iloc[0] if len(airfield) else None,
            "country": country.iloc[0] if len(country) else None,
            "domain": domain,
            "n_chips": int(g["n_chips_calc"].sum()),
            "novelty_rate": float(np.mean(nov_vals)) if nov_vals else None,
            "drift_score": drift_score,
        })
    return rows


# ---------------------------------------------------------- image lookups --

def _resolve_path(value: str) -> Path:
    p = Path(value)
    if not p.is_absolute():
        p = config.REPO_ROOT / p
    return p


def chip_image_path(ann_id: int) -> Path | None:
    df = chips_index_df()
    row = df.loc[df["ann_id"] == ann_id]
    if row.empty:
        return None
    return _resolve_path(str(row.iloc[0]["path"]))


def tile_image_path(tile_id: int) -> Path | None:
    df = tiles_df()
    row = df.loc[df["tile_id"] == tile_id]
    if row.empty:
        return None
    return _resolve_path(str(row.iloc[0]["image_path"]))
