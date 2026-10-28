"""Vigil FastAPI backend. Implements backend/API.md exactly.

Run with `backend/run_dev.sh` (uvicorn on :8000, reload). Every number
returned here is computed from data/*.parquet + data/scores/*.{parquet,json}
at request time, plus the SQLite `reviews` table -- no hard-coded results.
"""

from __future__ import annotations

import sys
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.routing import APIRouter
from pydantic import BaseModel, Field

from . import changes as changes_api
from . import collections as collections_api
from . import health as health_api
from . import reference as reference_api
from . import config, constants, dataio, db, report

# `/adapt` does `from vigil.pipeline import build_version` -- make sure
# backend/ (which holds the `vigil` package, owned by other agents) is on
# sys.path regardless of cwd.
if str(config.BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(config.BACKEND_DIR))

app = FastAPI(title="Vigil API", description="Assurance evidence prototype for an overhead-imagery aircraft role classifier.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

router = APIRouter(prefix="/api")

IMAGE_CACHE_HEADERS = {"Cache-Control": "public, max-age=86400, immutable"}


def _unavailable(e: dataio.DataUnavailable) -> HTTPException:
    return HTTPException(status_code=503, detail=str(e))


def _current_version() -> str:
    try:
        return dataio.current_version()
    except dataio.DataUnavailable as e:
        raise _unavailable(e)


# --------------------------------------------------------------- /summary --

@router.get("/summary")
def get_summary():
    version = _current_version()
    try:
        model = dataio.model_json(version)
    except dataio.DataUnavailable:
        model = {}
    try:
        versions = dataio.versions_list()
    except dataio.DataUnavailable:
        versions = []
    batches = dataio.batch_list(version)
    df_ops = dataio.operational_chip_table(version)
    chips_total = int(len(df_ops))
    novel_total = 0
    if chips_total and "is_novel" in df_ops.columns:
        novel_total = int(pd.to_numeric(df_ops["is_novel"], errors="coerce").fillna(0).sum())
    reviewed_total = len(db.all_reviews())
    detector_card = dataio.detector_card()
    return {
        "model": {
            "name": model.get("name", "Vigil role/novelty model"),
            "detector": dataio.detector_summary_string(detector_card) or model.get("detector"),
            "embedder": model.get("embedder"),
            "novelty_method": model.get("novelty_method"),
            "reference_domain": model.get("reference_domain", "ref_train ∪ ref_calib"),
            "n_ref_chips": model.get("n_ref_chips"),
            "n_ref_images": model.get("n_ref_images"),
            "vigilance": dataio.model_vigilance(model),
            "calibrated_on": model.get("calibrated_on"),
            "calibration_check": model.get("calibration_check"),
            "model_selection_note": model.get("model_selection_note"),
            "reference_definition": model.get("reference_definition"),
            "localization": constants.NOTE_LOCALIZATION,
            "detector_card": detector_card,
        },
        "versions": versions,
        "current_version": version,
        "batches": batches,
        "totals": {"chips": chips_total, "novel": novel_total, "reviewed": reviewed_total},
    }


# --------------------------------------------------------------- /batches --

@router.get("/batches")
def get_batches():
    version = _current_version()
    return dataio.batch_list(version)


@router.get("/batches/{batch_id}")
def get_batch(batch_id: str, min_partial: float = Query(0.75, ge=0.0, le=1.0)):
    if batch_id not in constants.BATCH_META:
        raise HTTPException(404, f"unknown batch id: {batch_id}")
    version = _current_version()
    batches = {b["id"]: b for b in dataio.batch_list(version)}
    if batch_id not in batches:
        raise HTTPException(404, f"batch '{batch_id}' has no computed data for version {version}")
    base = batches[batch_id]

    df = dataio.full_chip_table(version)
    bdf = df[df["domain"] == batch_id] if "domain" in df.columns else df.iloc[0:0]
    ref = dataio.reference_chip_table(version)
    novelty_hist, ref_novelty_hist = dataio.novelty_histograms(
        bdf["novelty"] if "novelty" in bdf.columns else pd.Series(dtype=float),
        ref["novelty"] if "novelty" in ref.columns else pd.Series(dtype=float),
    )

    reviews = db.reviews_by_ann_id()
    detected_map = dataio.detected_by_ann_id()
    # min_partial is a display/triage filter on top_novel only -- novelty_hist/
    # ref_novelty_hist and `base` (batch metrics) above are computed over every
    # chip, unfiltered (see API.md).
    top_source = dataio.filter_min_partial(bdf, min_partial)
    top = top_source.sort_values("novelty", ascending=False).head(10) if "novelty" in top_source.columns else top_source.iloc[0:0]
    top_novel = [
        dataio.chip_dict(r, version, reviews.get(int(r["ann_id"])), detected_map.get(int(r["ann_id"])))
        for _, r in top.iterrows()
    ]

    try:
        model = dataio.model_json(version)
    except dataio.DataUnavailable:
        model = {}
    vigilance = dataio.model_vigilance(model)
    vigilance_novelty_threshold = (1.0 - vigilance) if vigilance is not None else None

    return {
        **base,
        "novelty_hist": novelty_hist,
        "ref_novelty_hist": ref_novelty_hist,
        "top_novel": top_novel,
        "vigilance_novelty_threshold": vigilance_novelty_threshold,
    }


@router.get("/batches/{batch_id}/chips")
def get_batch_chips(
    batch_id: str,
    sort: str = Query("novelty", pattern="^(novelty|role_conf|knn_dist)$"),
    order: str = Query("desc", pattern="^(asc|desc)$"),
    limit: int = Query(48, ge=1, le=500),
    offset: int = Query(0, ge=0),
    only_novel: bool = Query(False),
    reviewed: str = Query("all", pattern="^(all|yes|no)$"),
    detected: str = Query("all", pattern="^(all|yes|no)$"),
    min_partial: float = Query(0.75, ge=0.0, le=1.0),
):
    if batch_id not in constants.BATCH_META:
        raise HTTPException(404, f"unknown batch id: {batch_id}")
    version = _current_version()
    df = dataio.full_chip_table(version)
    df = df[df["domain"] == batch_id] if "domain" in df.columns else df.iloc[0:0]
    # Display/triage filter only -- excludes chips whose GT box is mostly cut
    # off at the tile edge (partial < min_partial); scoring/eval never see this.
    df = dataio.filter_min_partial(df, min_partial)

    if only_novel and "is_novel" in df.columns:
        df = df[df["is_novel"].astype(bool)]

    reviews = db.reviews_by_ann_id()
    if reviewed != "all":
        reviewed_ids = set(reviews.keys())
        mask = df["ann_id"].astype(int).isin(reviewed_ids)
        df = df[mask] if reviewed == "yes" else df[~mask]

    detected_map = dataio.detected_by_ann_id()
    if detected != "all":
        det_series = df["ann_id"].astype(int).map(detected_map)
        df = df[det_series == True] if detected == "yes" else df[det_series == False]  # noqa: E712

    if sort in df.columns:
        df = df.sort_values(sort, ascending=(order == "asc"), na_position="last")

    total = int(len(df))
    page = df.iloc[offset:offset + limit]
    items = [
        dataio.chip_dict(r, version, reviews.get(int(r["ann_id"])), detected_map.get(int(r["ann_id"])))
        for _, r in page.iterrows()
    ]
    return {"total": total, "items": items}


# ----------------------------------------------------------------- /chips --

@router.get("/chips/{ann_id}")
def get_chip(ann_id: int):
    version = _current_version()
    df = dataio.full_chip_table(version)
    match = df[df["ann_id"] == ann_id]
    if match.empty:
        raise HTTPException(404, f"chip {ann_id} not found in version {version}")
    row = match.iloc[0]
    reviews = db.reviews_by_ann_id()
    detected_map = dataio.detected_by_ann_id()
    chip = dataio.chip_dict(row, version, reviews.get(ann_id), detected_map.get(ann_id))
    neighbors = dataio.neighbors_for_chip(row, version)
    novelty = row.get("novelty")
    art_match = None
    if novelty is not None:
        try:
            if not np.isnan(novelty):
                art_match = 1.0 - float(novelty)
        except TypeError:
            pass
    return {**chip, "neighbors": neighbors, "art_match": art_match}


@router.get("/chips/{ann_id}/image")
def get_chip_image(ann_id: int):
    path = dataio.chip_image_path(ann_id)
    if path is None or not path.exists():
        raise HTTPException(404, f"chip image {ann_id} not found")
    return FileResponse(path, media_type="image/png", headers=IMAGE_CACHE_HEADERS)


# ------------------------------------------------------------------- /det --
# Field mode: single detector-chip detail, in the same Chip shape as /chips/{ann_id} where it makes
# sense, with `source: "det"`, `det_id`, `conf` always shown, and `matched_ann_id` (benchmark-only)
# hidden unless `?benchmark=1`. Distinct from backend/app/collections.py's own listing endpoints
# (/api/collections/*), which build their own lighter chip dict for the collection-detail grid.

@router.get("/det/{det_id}")
def get_det_chip(det_id: int, benchmark: int = Query(0)):
    version = _current_version()
    df = dataio.full_det_chip_table(version)
    if df.empty:
        raise HTTPException(404, f"det chip {det_id} not found in version {version}")
    match = df[df["det_id"] == det_id]
    if match.empty:
        raise HTTPException(404, f"det chip {det_id} not found in version {version}")
    row = match.iloc[0]
    reviews = db.reviews_by_ann_id(source="det")
    chip = dataio.det_chip_dict(row, version, reviews.get(det_id), benchmark=bool(benchmark))
    neighbors = dataio.neighbors_for_chip(row, version)
    novelty = row.get("novelty")
    art_match = None
    if novelty is not None:
        try:
            if not np.isnan(novelty):
                art_match = 1.0 - float(novelty)
        except TypeError:
            pass
    return {**chip, "neighbors": neighbors, "art_match": art_match}


@router.get("/det/{det_id}/image")
def get_det_chip_image(det_id: int):
    path = dataio.det_chip_image_path(det_id)
    if path is None or not path.exists():
        raise HTTPException(404, f"det chip image {det_id} not found")
    return FileResponse(path, media_type="image/png", headers=IMAGE_CACHE_HEADERS)


# ----------------------------------------------------------------- /tiles --

@router.get("/tiles/{tile_id}/image")
def get_tile_image(tile_id: int):
    path = dataio.tile_image_path(tile_id)
    if path is None or not path.exists():
        raise HTTPException(404, f"tile image {tile_id} not found")
    return FileResponse(path, media_type="image/png", headers=IMAGE_CACHE_HEADERS)


@router.get("/tiles/{tile_id}")
def get_tile(tile_id: int):
    tiles = dataio.tiles_df()
    trow = tiles[tiles["tile_id"] == tile_id]
    if trow.empty:
        raise HTTPException(404, f"tile {tile_id} not found")
    trow = trow.iloc[0]
    version = _current_version()
    df = dataio.full_chip_table(version)
    tdf = df[df["tile_id"] == tile_id]
    reviews = db.reviews_by_ann_id()
    detected_map = dataio.detected_by_ann_id()
    chips = [
        dataio.chip_dict(r, version, reviews.get(int(r["ann_id"])), detected_map.get(int(r["ann_id"])))
        for _, r in tdf.iterrows()
    ]
    return {
        "tile_id": tile_id,
        "file_name": trow.get("file_name"),
        "meta": {
            "airfield": trow.get("airfield"),
            "country": trow.get("country"),
            "weather": trow.get("weather"),
            "off_nadir": float(trow["off_nadir"]) if pd.notna(trow.get("off_nadir")) else None,
            "gsd": float(trow["gsd"]) if pd.notna(trow.get("gsd")) else None,
        },
        "chips": chips,
        "detections": dataio.tile_detections(tile_id),
    }


# ------------------------------------------------------------- /embedding --

@router.get("/embedding/points")
def get_embedding_points(batch: str = Query(...)):
    if batch not in constants.BATCH_META:
        raise HTTPException(404, f"unknown batch id: {batch}")
    version = _current_version()
    df = dataio.full_chip_table(version)
    ref = dataio.reference_chip_table(version)

    ref_pts = []
    if "umap_x" in ref.columns and "umap_y" in ref.columns:
        ref_valid = ref.dropna(subset=["umap_x", "umap_y"])
        if len(ref_valid) > 1500:
            rng = np.random.RandomState(0)
            idx = rng.choice(len(ref_valid), size=1500, replace=False)
            ref_valid = ref_valid.iloc[idx]
        for _, r in ref_valid.iterrows():
            ref_pts.append({
                "ann_id": int(r["ann_id"]), "x": float(r["umap_x"]), "y": float(r["umap_y"]),
                "role_true": r.get("role"),
            })

    bdf = df[df["domain"] == batch] if "domain" in df.columns else df.iloc[0:0]
    batch_pts = []
    if "umap_x" in bdf.columns and "umap_y" in bdf.columns:
        bvalid = bdf.dropna(subset=["umap_x", "umap_y"])
        for _, r in bvalid.iterrows():
            batch_pts.append({
                "ann_id": int(r["ann_id"]), "x": float(r["umap_x"]), "y": float(r["umap_y"]),
                "novelty": float(r["novelty"]) if pd.notna(r.get("novelty")) else None,
                "is_novel": bool(r["is_novel"]) if pd.notna(r.get("is_novel")) else None,
                "art_category": int(r["art_category"]) if pd.notna(r.get("art_category")) else None,
            })

    return {"reference": ref_pts, "batch": batch_pts}


# ------------------------------------------------------------------ /drift --

@router.get("/drift")
def get_drift():
    version = _current_version()
    try:
        bj = dataio.batches_json(version)
    except dataio.DataUnavailable as e:
        raise _unavailable(e)
    statistic = None
    null_method = None
    series = []
    for bid, meta in constants.BATCH_META.items():
        if bid not in bj:
            continue
        drift = bj[bid].get("drift") or {}
        if statistic is None:
            statistic = drift.get("statistic")
        if null_method is None:
            null_method = drift.get("null_method")
        series.append({
            "batch_id": bid, "order": meta["order"], "name": meta["name"],
            "score": drift.get("score"), "null_p95": drift.get("null_p95"),
            "p_value": drift.get("p_value"), "n": drift.get("n"),
            "calib_mean": drift.get("calib_mean"), "ratio_to_calib": drift.get("ratio_to_calib"),
        })
    series.sort(key=lambda s: s["order"])
    return {
        "statistic": statistic or "mean 5-NN cosine distance to reference",
        "null_method": null_method,
        "series": series,
    }


# --------------------------------------------------------------- /envelope --

@router.get("/envelope")
def get_envelope():
    version = _current_version()
    return dataio.envelope(version)


# ------------------------------------------------------------------ /eval --

@router.get("/eval")
def get_eval():
    version = _current_version()
    try:
        summary = dataio.eval_summary(version)
    except dataio.DataUnavailable as e:
        raise _unavailable(e)
    return {
        "version": version,
        "novelty": summary["novelty"],
        "role_head": summary["role_head"],
        "military": summary["military"],
        "detection": dataio.detection_eval_summary(),
        "analyst_disagreement": dataio.analyst_disagreement(version),
        "note": constants.NOTE_EVAL,
    }


# ---------------------------------------------------------------- /review --

@router.get("/review/queue")
def get_review_queue(
    limit: int = Query(24, ge=1, le=500),
    strategy: str | None = Query(None, pattern="^(novelty|uncertainty|random|collection)$"),
    min_partial: float = Query(0.75, ge=0.0, le=1.0),
    source: str = Query("gt", pattern="^(gt|det)$"),
    min_conf: float = Query(0.5, ge=0.0, le=1.0),
):
    version = _current_version()
    if source == "det":
        # Field mode: flagged detector chips across NEW collections only, unreviewed, ranked --
        # DetChip dicts (same as /collections/{group_id}'s chips) plus `group_id`. `min_partial` is a
        # GT-box concept and does not apply. See collections.flagged_det_queue.
        # Default strategy differs by source: det queues are collection-first (see flagged_det_queue).
        return collections_api.flagged_det_queue(version, strategy=strategy or "collection", limit=limit, min_conf=min_conf)
    df = dataio.operational_chip_table(version)
    strategy = strategy or "novelty"
    # Display/triage filter only -- see filter_min_partial's docstring.
    df = dataio.filter_min_partial(df, min_partial)
    reviews = db.reviews_by_ann_id()
    unreviewed = df[~df["ann_id"].astype(int).isin(reviews.keys())]
    if strategy == "novelty" and "novelty" in unreviewed.columns:
        unreviewed = unreviewed.sort_values("novelty", ascending=False)
    elif strategy == "uncertainty" and "role_conf" in unreviewed.columns:
        unreviewed = unreviewed.sort_values("role_conf", ascending=True)
    elif strategy == "random" and len(unreviewed):
        unreviewed = unreviewed.sample(frac=1)
    page = unreviewed.head(limit)
    detected_map = dataio.detected_by_ann_id()
    return [dataio.chip_dict(r, version, None, detected_map.get(int(r["ann_id"]))) for _, r in page.iterrows()]


class ReviewBody(BaseModel):
    action: str
    label: Optional[str] = None
    note: Optional[str] = None


@router.post("/review/{ann_id}")
def post_review(ann_id: int, body: ReviewBody, source: str = Query("gt", pattern="^(gt|det)$")):
    """`source='gt'` (default) reviews a ground-truth chip (`ann_id` is a real annotations.parquet
    ann_id); `source='det'` reviews a detector chip (field mode -- `ann_id` here actually carries a
    det_id, see backend/API.md's "keep the API's ann_id naming" convention). The two id spaces are
    independent (db.py's reviews table keys on (source, ann_id)), so a gt review and a det review
    that happen to share a numeric id never collide.
    """
    if body.action not in constants.REVIEW_ACTIONS:
        raise HTTPException(422, f"invalid action '{body.action}', expected one of {sorted(constants.REVIEW_ACTIONS)}")
    if body.action == "new_class" and not body.label:
        raise HTTPException(422, "label is required when action='new_class'")
    version = _current_version()
    if body.action == "correct":
        # A correction records the analyst's DISAGREEMENT with the predicted role and assigns
        # the chip a KNOWN class label -- unlike new_class, it never fits a prototype/ART
        # category, so the label must already be one of this version's classes (civil + analyst).
        if not body.label:
            raise HTTPException(422, "label is required when action='correct'")
        allowed = dataio.version_classes(version)
        if body.label not in allowed:
            raise HTTPException(
                422,
                f"invalid label '{body.label}' for action='correct'; expected one of {sorted(allowed)}",
            )
    if source == "det":
        predicted_role = dataio.det_chip_role_pred(ann_id, version)
    else:
        predicted_role = dataio.chip_role_pred(ann_id, version)
    return db.upsert_review(ann_id, body.action, body.label, body.note, predicted_role, version, source=source)


@router.get("/review/log")
def get_review_log(source: str = Query("all", pattern="^(all|gt|det)$")):
    return db.all_reviews(source=source)


@router.get("/review/labels")
def get_review_labels():
    """Distinct labels ever used in a review, most-recently-used first -- pools BOTH sources (gt and
    det chip reviews), see db.distinct_labels."""
    return {"labels": db.distinct_labels()}


@router.get("/review/classes")
def get_review_classes():
    """The current version's known classes (civil + analyst-created) -- the
    same allowed list `action='correct'` labels are validated against, so the
    frontend's "Correct role" picker always offers exactly what the backend
    will accept.
    """
    return {"classes": dataio.version_classes(_current_version())}


# ----------------------------------------------------------------- /adapt --

class AdaptBody(BaseModel):
    note: Optional[str] = None
    # Optional additional detector-confidence gate on top of ADAPTATION_RULE's nearest-exemplar rule
    # (vigil.pipeline; det chips only -- GT chips are never gated on it). 0 = off, the default: the
    # nearest-exemplar rule already removes most of the junk on its own (see
    # vigil.pipeline.ASSIGN_MIN_CONF_DEFAULT's comment for the measured trade-off).
    assign_min_conf: float = Field(default=0.0, ge=0.0, le=1.0)


@router.post("/adapt")
def post_adapt(body: AdaptBody):
    try:
        from vigil.pipeline import build_version  # type: ignore
    except ImportError as e:
        raise HTTPException(
            status_code=501,
            detail=f"vigil.pipeline.build_version is not available yet ({e}); adaptation cannot run.",
        )

    try:
        versions = dataio.versions_list()
    except dataio.DataUnavailable:
        versions = []
    new_id = f"v{len(versions) + 1}"
    feedback = db.unapplied_reviews()

    try:
        try:
            result = build_version(new_id, feedback, note=body.note, assign_min_conf=body.assign_min_conf)
        except TypeError:
            result = build_version(new_id, feedback)
    except HTTPException:
        raise
    except Exception as e:  # pragma: no cover - depends on vigil.pipeline internals
        raise HTTPException(500, f"vigil.pipeline.build_version failed: {e}")

    db.mark_applied([(r.get("source") or "gt", r["ann_id"]) for r in feedback], new_id)
    dataio._FILE_CACHE.clear()

    try:
        for v in dataio.versions_list():
            if v["id"] == new_id:
                return v
    except dataio.DataUnavailable:
        pass
    if isinstance(result, dict) and "id" in result:
        return result
    return {"id": new_id, "created": db.now_iso(), "classes": [], "n_feedback": len(feedback), "note": body.note or ""}


# ------------------------------------------------------------- /versions/compare --

@router.get("/versions/compare")
def get_versions_compare(a: str = Query(...), b: str = Query(...)):
    try:
        eval_a = dataio.eval_summary(a)
        eval_b = dataio.eval_summary(b)
    except dataio.DataUnavailable as e:
        raise HTTPException(404, str(e))

    rh_a = eval_a["role_head"]
    rh_b = eval_b["role_head"]
    pc_a = {d.get("cls"): d.get("accuracy") for d in rh_a.get("per_class", [])}
    pc_b = {d.get("cls"): d.get("accuracy") for d in rh_b.get("per_class", [])}
    known = set(pc_a) & set(pc_b)
    forgetting = 0.0
    diffs = [pc_a[c] - pc_b[c] for c in known if pc_a[c] is not None and pc_b[c] is not None]
    if diffs:
        forgetting = max(0.0, max(diffs))
    per_class = [{"cls": c, "a": pc_a.get(c), "b": pc_b.get(c)} for c in sorted(set(pc_a) | set(pc_b))]

    # Honest, label-agnostic rare-target recall (see vigil.evaluate._rare_target_recall): among
    # held-out military chips NOT used to fit any prototype/ART category, the fraction predicted
    # as any analyst-created class. Computed once in vigil.evaluate.evaluate() and read back here,
    # rather than re-derived from label strings (which used to leak analyst-labeled chips into
    # their own recall and required the predicted label to string-match the analyst's choice).
    mil_a = eval_a["military"]
    mil_b = eval_b["military"]

    try:
        bj_a = dataio.batches_json(a)
    except dataio.DataUnavailable:
        bj_a = {}
    try:
        bj_b = dataio.batches_json(b)
    except dataio.DataUnavailable:
        bj_b = {}
    novelty_rate_by_batch = []
    for bid in constants.BATCH_META:
        if bid in bj_a or bid in bj_b:
            novelty_rate_by_batch.append({
                "batch_id": bid,
                "a": (bj_a.get(bid) or {}).get("novelty_rate"),
                "b": (bj_b.get(bid) or {}).get("novelty_rate"),
            })

    return {
        "a": a, "b": b,
        "ref_holdout": {
            "role_accuracy_a": rh_a.get("accuracy_ref_holdout"),
            "role_accuracy_b": rh_b.get("accuracy_ref_holdout"),
            "per_class": per_class,
        },
        "military": {
            "analyst_class_assignment_rate_a": mil_a.get("analyst_class_assignment_rate", mil_a.get("rare_target_recall")),
            "analyst_class_assignment_rate_b": mil_b.get("analyst_class_assignment_rate", mil_b.get("rare_target_recall")),
            "civil_false_assignment_rate_a": mil_a.get("civil_false_assignment_rate"),
            "civil_false_assignment_rate_b": mil_b.get("civil_false_assignment_rate"),
            "n_eval_civil_a": mil_a.get("n_eval_civil"),
            "n_eval_civil_b": mil_b.get("n_eval_civil"),
            "adaptation_rule": mil_b.get("adaptation_rule") or mil_a.get("adaptation_rule"),
            "exclusion_method": mil_b.get("exclusion_method") or mil_a.get("exclusion_method"),
            "rare_target_recall_a": mil_a.get("rare_target_recall"),
            "rare_target_recall_b": mil_b.get("rare_target_recall"),
            "recall_new_class_a": mil_a.get("recall_new_class"),  # alias, kept for frontend compatibility
            "recall_new_class_b": mil_b.get("recall_new_class"),
            "miss_rate_civil_a": mil_a.get("miss_rate_civil"),
            "miss_rate_civil_b": mil_b.get("miss_rate_civil"),
            "n_eval_a": mil_a.get("n_eval"),
            "n_eval_b": mil_b.get("n_eval"),
        },
        "novelty_rate_by_batch": novelty_rate_by_batch,
        "forgetting": forgetting,
        # All-time, NOT per version a/b -- analyst confirm/correct feedback isn't naturally
        # split by which of two versions it was recorded against for a compare view; see
        # dataio.analyst_disagreement and API.md.
        "analyst_disagreement": dataio.analyst_disagreement(),
        # Field mode: {a: FieldBlock|null, b: FieldBlock|null, worst_regression} -- each side's
        # descriptive detector-chip assignment counts over new collections (with per-collection
        # by_group) + the labeled control-set regression check, composed from eval_{version}.json by
        # dataio.field_block (which drops benchmark batch names); see API.md's Field mode section.
        "field": dataio.field_compare(a, b),
    }


# ------------------------------------------------------------------- /map --

@router.get("/map")
def get_map():
    version = _current_version()
    return {"points": dataio.map_points(version)}


# ---------------------------------------------------------------- /report --

@router.get("/report", response_class=PlainTextResponse)
def get_report():
    version = _current_version()
    md = report.build_report(version)
    return PlainTextResponse(md, media_type="text/markdown")


app.include_router(router)
app.include_router(collections_api.router)
app.include_router(changes_api.router)
app.include_router(health_api.router)
app.include_router(reference_api.router)
