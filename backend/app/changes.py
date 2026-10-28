"""Field mode: `/api/versions/changes*` -- "Changed in this version": detector chips whose
predicted role differs between two built versions' own `{version}_det.parquet`, i.e. detections
`/adapt` re-sorted into a different (usually a newly-created analyst) class. An analyst-facing
spot-check surface, separate from `/versions/compare`'s aggregate `field` block: this exposes the
actual chips that changed, so an analyst can verify individual re-classifications rather than only
reading a rate.

Design notes (why this reuses collections.py instead of re-deriving grouping):
  - `backend/app/collections.py` is stable -- not part of the concurrently-edited
    pipeline/field-mode-eval lane (that's `backend/vigil/{pipeline,field}.py` and
    `backend/app/dataio.py`'s `field_block`) -- so importing its private, well-tested
    `_new_collections_table(version)` (domain filtering to NEW collections only, airfield x date
    grouping, `group_id` assignment) is safe and keeps "what group_id means" in exactly one place:
    a chip's `group_id` here always matches `/collections`' group_id for the same collection.
  - The same-parent-as-labeled-chips lookup is a small, self-contained re-read of
    `{version}.parquet` / `{version}_det.parquet`'s own `used_in_fit` columns (mirrors
    `vigil.evaluate.labeled_parent_ids`'s logic) rather than importing `vigil.evaluate` (which in
    turn imports `vigil.pipeline` -- both under concurrent, in-flight redesign) -- avoids any
    transient breakage from that unrelated import chain.

Ground-truth discipline: per backend/API.md's field-mode contract, no `role_true` / `matched_ann_id`
/ `domain` / `batch_id` / `domains` / `by_domain` key may appear anywhere in this module's
responses -- see backend/app/test_changes.py's forbidden-key check (shared with test_api.py's).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi import APIRouter, Query

from . import collections as collections_api
from . import dataio, db

router = APIRouter(prefix="/api/versions", tags=["versions"])

SORT_COLUMNS = {"sim": "analyst_sim", "novelty": "novelty", "conf": "conf"}


# ------------------------------------------------------------------ utils --

def _num(v: object) -> float | None:
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _bool(v: object) -> bool | None:
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return bool(v)


def _role_key(v: object) -> str:
    """Comparable identity for a raw role_pred value (a role_id int/float, an analyst class
    string, or NaN/None for "not scored" or "novel, no role") -- the SAME sentinel string for the
    latter two, so "still novel/unscored in both versions" reads as unchanged. Deliberately always
    returns a string (never None/NaN): mapping a float64 Series through a function that returns
    Python `None` silently coerces back to `float('nan')` (pandas.Series.map's dtype-preservation),
    and `float('nan') != float('nan')` is True by IEEE 754 -- which would misreport every
    still-novel-in-both-versions chip as "changed". A string result keeps the mapped Series
    object-dtype, where ordinary (and correct) `==`/`!=` string comparison applies.
    """
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except (TypeError, ValueError):
        pass
    return v if isinstance(v, str) else str(int(v))


def _scored_det_ids(version: str) -> set[int]:
    """det_ids that actually have a score ROW in `{version}_det.parquet` (regardless of whether
    role_pred is null because the chip is novel) -- as opposed to a chip absent from that file
    entirely (never scored yet, or dropped by a `not_aircraft` review on a later /adapt). Only
    det chips scored in BOTH `a` and `b` are eligible for a "changed" comparison; anything else
    (e.g. `b` is a not-yet-built or unknown version) can't be honestly diffed and is excluded, not
    reported as "changed to None".
    """
    scores = dataio.det_scores_df(version)
    if scores.empty or "det_id" not in scores.columns:
        return set()
    return set(scores["det_id"].dropna().astype(int).tolist())


def _labeled_parent_ids(version: str) -> set:
    """Parent ids of every chip -- GT or detector -- used to fit an analyst class for `version`,
    read straight from that version's own `{version}.parquet` / `{version}_det.parquet`
    `used_in_fit` columns (self-contained equivalent of `vigil.evaluate.labeled_parent_ids`; see
    this module's docstring for why it isn't imported directly).
    """
    parents: set = set()

    det = dataio.det_scores_df(version)
    if not det.empty and "used_in_fit" in det.columns and "parent_id" in det.columns:
        used = det[pd.to_numeric(det["used_in_fit"], errors="coerce").fillna(0).astype(bool)]
        parents |= set(used["parent_id"].dropna().unique().tolist())

    try:
        scores = dataio.scores_df(version)
    except dataio.DataUnavailable:
        scores = pd.DataFrame()
    if not scores.empty and "used_in_fit" in scores.columns and "tile_id" in scores.columns:
        used_mask = pd.to_numeric(scores["used_in_fit"], errors="coerce").fillna(0).astype(bool)
        if used_mask.any():
            try:
                tiles = dataio.tiles_df()
            except dataio.DataUnavailable:
                tiles = pd.DataFrame(columns=["tile_id", "parent_id"])
            if "parent_id" in tiles.columns:
                merged = scores.loc[used_mask, ["tile_id"]].merge(tiles[["tile_id", "parent_id"]], on="tile_id", how="left")
                parents |= set(merged["parent_id"].dropna().unique().tolist())
    return parents


def _review_map() -> dict[int, dict]:
    try:
        return db.reviews_by_ann_id(source="det")
    except TypeError:
        return {}


# --------------------------------------------------------------- diff table --

def _changed_table(a: str, b: str) -> pd.DataFrame:
    """One row per det chip, across NEW collections only (never ref_train/ref_calib/ref_holdout --
    same universe as /collections), whose role_pred differs between version `a` and `b`'s own det
    scores. EXCLUDES chips `b` itself used to fit an analyst class (`used_in_fit` in `b` -- reviewing
    the exact example that seeded a class mean is circular). Chips from the same parent image as any
    labeled chip (either source, in `b`) are kept but flagged `_same_parent` rather than excluded, so
    the caller can show them muted without hiding real evidence.
    """
    df_a, _ = collections_api._new_collections_table(a)
    df_b, _ = collections_api._new_collections_table(b)
    if df_a.empty or df_b.empty:
        return df_b.iloc[0:0]

    eligible = _scored_det_ids(a) & _scored_det_ids(b)
    if not eligible:
        return df_b.iloc[0:0]

    role_a = df_a.set_index("det_id")["role_pred"] if "role_pred" in df_a.columns else pd.Series(dtype=float)
    df = df_b[df_b["det_id"].astype(int).isin(eligible)].copy()
    df["role_pred_a_raw"] = df["det_id"].map(role_a)
    df["role_pred_b_raw"] = df["role_pred"] if "role_pred" in df.columns else np.nan

    key_a = df["role_pred_a_raw"].map(_role_key)
    key_b = df["role_pred_b_raw"].map(_role_key)
    df = df[key_a != key_b]
    if df.empty:
        return df

    if "used_in_fit" in df.columns:
        used = pd.to_numeric(df["used_in_fit"], errors="coerce").fillna(0).astype(bool)
        df = df[~used]
    if df.empty:
        return df

    labeled_parents = _labeled_parent_ids(b)
    df = df.assign(_same_parent=df["parent_id"].isin(labeled_parents) if labeled_parents else False)
    return df


def _item_dict(row: pd.Series, review: dict | None) -> dict:
    det_id = int(row["det_id"])
    tile_id = int(row["tile_id"])
    return {
        "source": "det",
        "det_id": det_id,
        "ann_id": det_id,  # alias == det_id, so the shared chip-drawer/review components work unchanged
        "tile_id": tile_id,
        "bbox": [_num(row.get("x1")), _num(row.get("y1")), _num(row.get("w")), _num(row.get("h"))],
        "conf": _num(row.get("conf")),
        "chip_url": f"/api/det/{det_id}/image",
        "tile_url": f"/api/tiles/{tile_id}/image",
        "role_pred": dataio.role_label(row.get("role_pred_b_raw")),  # current = b's own role_pred
        "role_pred_a": dataio.role_label(row.get("role_pred_a_raw")),
        "role_pred_b": dataio.role_label(row.get("role_pred_b_raw")),
        "role_conf": _num(row.get("role_conf")),
        "novelty": _num(row.get("novelty")),
        "is_novel": _bool(row.get("is_novel")),
        "knn_dist": _num(row.get("knn_dist")),
        "analyst_sim": _num(row.get("analyst_sim")),
        "meta": {
            "airfield": row.get("airfield"),
            "country": row.get("country"),
            "weather": row.get("weather"),
            "off_nadir": _num(row.get("off_nadir")),
            "gsd": _num(row.get("gsd")),
        },
        "review": review,
        "group_id": row.get("group_id"),
        "airfield": row.get("airfield"),
        "date": row.get("date"),
        # Evidence-only tag (never an exclusion): this chip's parent image also produced a chip
        # used to fit an analyst class in `b` -- visually/positionally correlated with that label,
        # so the UI mutes it rather than treating it as independent confirmation.
        "same_parent": bool(row.get("_same_parent", False)),
    }


# ------------------------------------------------------------------ routes --

@router.get("/changes")
def get_changes(
    a: str = Query(...),
    b: str = Query(...),
    source: str = Query("det", pattern="^(det)$"),
    limit: int = Query(48, ge=1, le=500),
    offset: int = Query(0, ge=0),
    group_id: str | None = Query(None),
    only_unreviewed: bool = Query(False),
    sort: str = Query("sim", pattern="^(sim|novelty|conf)$"),
):
    df = _changed_table(a, b)

    review_map = _review_map()
    if only_unreviewed and review_map:
        reviewed_ids = set(review_map.keys())
        df = df[~df["det_id"].astype(int).isin(reviewed_ids)]

    # by_group / by_class facets over the full only_unreviewed-filtered (but not yet group_id-
    # filtered or paginated) changed set, so the UI can use them as pickers alongside `items`.
    by_group: list[dict] = []
    by_class: list[dict] = []
    if not df.empty:
        for gid, gg in df.groupby("group_id", sort=False):
            by_group.append({
                "group_id": gid,
                "airfield": gg["airfield"].iloc[0],
                "date": gg["date"].iloc[0],
                "n_changed": int(len(gg)),
            })
        by_group.sort(key=lambda r: r["n_changed"], reverse=True)

        labels = df["role_pred_b_raw"].map(lambda v: dataio.role_label(v) or "Still novel")
        for label, n in labels.value_counts(dropna=False).items():
            by_class.append({"label": label, "n": int(n)})
        by_class.sort(key=lambda r: r["n"], reverse=True)

    if group_id:
        df = df[df["group_id"] == group_id]

    sort_col = SORT_COLUMNS.get(sort, "analyst_sim")
    if sort_col in df.columns:
        df = df.sort_values(sort_col, ascending=False, na_position="last")

    total = int(len(df))
    page = df.iloc[offset:offset + limit]
    items = [_item_dict(r, review_map.get(int(r["det_id"]))) for _, r in page.iterrows()]

    return {"a": a, "b": b, "total": total, "by_group": by_group, "by_class": by_class, "items": items}


@router.get("/changes/summary")
def get_changes_summary(a: str = Query(...), b: str = Query(...)):
    df = _changed_table(a, b)
    total = int(len(df))
    changed_ids = set(df["det_id"].astype(int).tolist()) if total else set()

    # A review counts here only if it was recorded while `b` was the current version (its
    # `predicted_version` snapshot, see db.upsert_review) -- otherwise it's evidence about some
    # other version's prediction on this chip, not this change.
    relevant = [
        r for r in db.all_reviews(source="det")
        if r.get("ann_id") in changed_ids and r.get("predicted_version") == b
    ]
    n_reviewed = len(relevant)
    n_confirmed = sum(1 for r in relevant if r["action"] == "confirm")
    n_corrected = sum(1 for r in relevant if r["action"] == "correct")
    n_not_aircraft = sum(1 for r in relevant if r["action"] == "not_aircraft")
    n_bad_box = sum(1 for r in relevant if r["action"] == "bad_box")
    denom = n_confirmed + n_corrected + n_not_aircraft + n_bad_box
    verified_precision = (n_confirmed / denom) if denom else None

    return {
        "total": total,
        "n_reviewed": n_reviewed,
        "n_confirmed": n_confirmed,
        "n_corrected": n_corrected,
        "n_not_aircraft": n_not_aircraft,
        "n_bad_box": n_bad_box,
        "verified_precision": verified_precision,
    }
