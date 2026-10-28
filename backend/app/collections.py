"""Field mode: `/api/collections*` -- an analyst-facing view of new,
UNLABELED RarePlanes parent images grouped by airfield x collection date,
scored on the DETECTOR's own boxes (never ground truth). Implements
backend/API.md's "Field mode" section.

Per STATE.md's Field mode consensus: collections are grouped at
airfield x date, never reported per-parent; no drift statistics here (that
stays Benchmark-only, computed against a KNOWN batch/domain, which field
mode must never expose); status comes purely from flagged-detection
evidence with a stated minimum denominator; ground truth (role_true,
matched_ann_id, batch/domain names) must never appear in any response this
module returns.

Data model (built by a concurrently-developed det-chip milestone; every
loader degrades to empty results, never a 5xx, if its file doesn't exist
yet):
    data/chips_det/index.parquet      -- backend/app/dataio.det_chips_index_df()
    data/scores/{version}_det.parquet -- backend/app/dataio.det_scores_df()
    data/rareplanes/tiles.parquet     -- existing loader, dataio.tiles_df()

Reviews on a det chip are recorded in the SAME `reviews` sqlite table as
GT-chip reviews (backend/app/db.py), keyed by the det chip's `det_id`
(reused as the review's `ann_id` -- see API.md's Chip.ann_id note), and
tagged `source='det'` -- see db.reviews_by_ann_id(source=...), landed by a
concurrently-developed source-aware-review milestone. This module reads
reviews only through that shared accessor (never its own SQL), falling back
to "no det reviews yet" if an older db.py doesn't support `source`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from scipy.stats import binom

from . import dataio, db
from .constants import REFERENCE_DOMAINS

# config's own import already puts backend/ on sys.path (so `vigil...` is importable)
from vigil.grouping import GROUPING, group_id as _group_id  # noqa: E402  (stdlib-only module)

router = APIRouter(prefix="/api/collections", tags=["collections"])

# "New collections" = every parent image NOT in the labeled reference set
# (ref_train ∪ ref_calib, the model's fit set, PLUS ref_holdout, the labeled
# control/validation set) -- per API.md's Field mode contract.
REFERENCE_ALL_DOMAINS = frozenset(REFERENCE_DOMAINS) | {"ref_holdout"}

MIN_DENOMINATOR = 20
P_THRESHOLD = 0.01
# Fallback ONLY when there is no usable reference flag rate (ref_holdout has no det chips, or none of
# them are flagged): an absolute count, since a rate of 0 has no tail to test against.
FALLBACK_MIN_FLAGGED = 10
STATUS_RULE = (
    f"insufficient_data if n_detections < {MIN_DENOMINATOR}; else needs_review if the one-sided "
    f"binomial tail probability of seeing >= n_flagged flagged detections out of n_detections at the "
    f"reference flag rate (ref_holdout detector chips) is < {P_THRESHOLD}; else nominal. Indicative "
    f"only: tiles within a collection are correlated, so the binomial p-value understates uncertainty."
)
STATUS_RULE_FALLBACK = (
    f"insufficient_data if n_detections < {MIN_DENOMINATOR}; else needs_review if n_flagged >= "
    f"{FALLBACK_MIN_FLAGGED}; else nominal. (Fallback: no usable reference flag rate -- ref_holdout has "
    f"no flagged detector chips -- so there is no rate to test against.)"
)
REFERENCE_DESCRIPTION = "labeled validation set (reference), the only labeled data"

TILE_META_COLS = [
    "tile_id", "airfield", "country", "state", "weather",
    "off_nadir", "gsd", "lat", "lon", "collection_date",
]


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


def _first(g: pd.DataFrame, col: str):
    if col not in g.columns:
        return None
    s = g[col].dropna()
    return s.iloc[0] if len(s) else None


def _mean(g: pd.DataFrame, col: str) -> float | None:
    if col not in g.columns:
        return None
    s = pd.to_numeric(g[col], errors="coerce").dropna()
    return float(s.mean()) if len(s) else None


def _safe_version() -> str | None:
    try:
        return dataio.current_version()
    except dataio.DataUnavailable:
        return None


def _domain_by_parent() -> dict:
    """Fallback domain-per-parent lookup from the (already-built) GT-chip
    index, for det chips whose own `domain` column is missing/null -- domain
    is assigned at parent-image granularity (STATE.md), and every parent
    image already has GT-chip rows from the milestone-1 pipeline, so this is
    available even before the det-chip milestone writes its own domain.
    """
    try:
        chips = dataio.chips_index_df()
    except dataio.DataUnavailable:
        return {}
    if "domain" not in chips.columns or "tile_id" not in chips.columns:
        return {}
    try:
        tiles = dataio.tiles_df()
    except dataio.DataUnavailable:
        return {}
    if "parent_id" not in tiles.columns:
        return {}
    merged = (
        chips[["tile_id", "domain"]]
        .drop_duplicates()
        .merge(tiles[["tile_id", "parent_id"]], on="tile_id", how="left")
        .dropna(subset=["parent_id"])
    )
    if merged.empty:
        return {}
    return merged.groupby("parent_id")["domain"].first().to_dict()


def _det_reviews_by_det_id() -> dict[int, dict]:
    """Reviews recorded against det chips, keyed by det_id (stored as the
    review's `ann_id` -- see API.md's Chip.ann_id note). Delegates to the
    shared, source-aware `db.reviews_by_ann_id` (backend/app/db.py, landed by
    a concurrently-developed source-aware-review milestone -- its
    `reviews` table carries a `source` column, auto-migrating older,
    single-source databases to `source='gt'` for every pre-existing row).
    Falls back to "no det reviews recorded yet" (empty dict) if that db
    module doesn't support `source` yet, rather than guessing.
    """
    try:
        return db.reviews_by_ann_id(source="det")
    except TypeError:
        return {}


def _reference_block() -> dict:
    """/api/collections' `reference` object: how big the ONLY labeled set
    in this story is (ref_train ∪ ref_calib ∪ ref_holdout combined) -- never
    score-dependent, so this is available even before any version/det
    scores exist.
    """
    dbp = _domain_by_parent()
    n_parents = 0
    n_tiles = 0
    if dbp:
        try:
            tiles = dataio.tiles_df()
        except dataio.DataUnavailable:
            tiles = pd.DataFrame(columns=["tile_id", "parent_id"])
        if "parent_id" in tiles.columns:
            domain = tiles["parent_id"].map(dbp)
            ref = tiles[domain.isin(REFERENCE_ALL_DOMAINS)]
            n_parents = int(ref["parent_id"].nunique())
            n_tiles = int(ref["tile_id"].nunique())
    return {"n_parents": n_parents, "n_tiles": n_tiles, "description": REFERENCE_DESCRIPTION}


def _flag_stats(frame: pd.DataFrame) -> tuple[int, int, float | None]:
    """(n_detections, n_flagged, flag_rate) for one det-chip frame.
    n_detections counts every det chip (scored or not -- from the det chip
    *index*, per API.md's `n_detections`); a chip this version hasn't
    scored yet (no `is_novel`) counts toward n_detections but never toward
    n_flagged. flag_rate is None only when there are no det chips at all
    (nothing to compute a rate over) -- an all-unscored, nonempty frame is a
    real (if pessimistic) 0.0, not "no data".
    """
    n_detections = int(len(frame))
    if n_detections == 0:
        return 0, 0, None
    if "is_novel" not in frame.columns:
        return n_detections, 0, 0.0
    is_novel = pd.to_numeric(frame["is_novel"], errors="coerce").fillna(0).astype(bool)
    n_flagged = int(is_novel.sum())
    return n_detections, n_flagged, n_flagged / n_detections


def _reference_rate_usable(reference_flag_rate: float | None) -> bool:
    return reference_flag_rate is not None and 0.0 < float(reference_flag_rate) <= 1.0


def _status_for(n_detections: int, n_flagged: int, reference_flag_rate: float | None) -> dict:
    """Evidence-only status for one collection (see STATUS_RULE): returns
    {status, p_value, excess_flagged}.

    - `insufficient_data` if n_detections < MIN_DENOMINATOR (too few detections for any rate to mean
      anything; p_value/excess are still reported for transparency, when computable).
    - otherwise a ONE-SIDED BINOMIAL TAIL TEST of n_flagged given n_detections against
      reference_flag_rate: p_value = P[X >= n_flagged], X ~ Binomial(n_detections, reference_flag_rate)
      (scipy.stats.binom.sf(n_flagged - 1, n, p)); `needs_review` if p_value < P_THRESHOLD, else
      `nominal`. Unlike a fixed n_flagged >= k rule this has no size bias: a big civil airport with
      21 flagged of 510 (4.1% vs a 3.1% reference) is nominal, a small collection with 18 of 66 is not.
    - `excess_flagged` = n_flagged - n_detections * reference_flag_rate (how many more flagged
      detections than the reference rate predicts; negative = fewer).
    - Guard: with a null OR ZERO reference_flag_rate there is no tail to test against (binom.sf at p=0
      is 0 for any n_flagged >= 1, which would mark every collection with one flagged chip
      needs_review), so status falls back to n_flagged >= FALLBACK_MIN_FLAGGED, with p_value and
      excess_flagged null and `status_rule` (on the /collections response) saying so.

    The p-value is INDICATIVE, not inferential: detector chips within a collection share tiles and a
    parent image, so they are correlated and the binomial understates the real uncertainty.
    """
    n = int(n_detections)
    k = int(n_flagged)
    p_value: float | None = None
    excess: float | None = None
    if _reference_rate_usable(reference_flag_rate) and n > 0:
        p_ref = float(reference_flag_rate)
        p_value = float(binom.sf(k - 1, n, p_ref))  # P[X >= k]; k=0 -> 1.0
        excess = float(k - n * p_ref)
    if n < MIN_DENOMINATOR:
        status = "insufficient_data"
    elif p_value is not None:
        status = "needs_review" if p_value < P_THRESHOLD else "nominal"
    else:
        status = "needs_review" if k >= FALLBACK_MIN_FLAGGED else "nominal"
    return {"status": status, "p_value": p_value, "excess_flagged": excess}


def status_rule_for(reference_flag_rate: float | None) -> str:
    """The rule string actually applied for this reference rate (see _status_for's guard)."""
    return STATUS_RULE if _reference_rate_usable(reference_flag_rate) else STATUS_RULE_FALLBACK


# --------------------------------------------------------------- det table --

def _det_full_table(version: str | None) -> pd.DataFrame:
    """One row per det chip: det chip index (bbox, domain, parent_id) +
    this version's det scores (role/novelty/etc, LEFT-joined -- a det chip
    with no score yet still counts toward n_detections, just not
    n_flagged) + tile metadata (airfield/country/state/weather/off_nadir/
    gsd/lat/lon/collection_date) needed for the airfield x date grouping key
    and Chip.meta. Empty (never an error) if the det chip index hasn't been
    built yet.
    """
    idx = dataio.det_chips_index_df()
    if idx.empty:
        cols = list(idx.columns) + [
            "role_pred", "role_conf", "novelty", "is_novel", "art_category", "knn_dist",
            "airfield", "country", "state", "weather", "off_nadir", "gsd", "lat", "lon", "date",
        ]
        return pd.DataFrame(columns=cols)

    if version is not None:
        scores = dataio.det_scores_df(version)
    else:
        scores = pd.DataFrame(columns=dataio.DET_SCORE_COLUMNS)
    score_cols = [
        c for c in ("det_id", "role_pred", "role_conf", "novelty", "is_novel",
                    "art_category", "knn_dist", "used_in_fit", "analyst_sim")
        if c in scores.columns
    ]
    if score_cols and "det_id" in score_cols:
        df = idx.merge(scores[score_cols], on="det_id", how="left")
    else:
        df = idx.copy()
    for c in ("role_pred", "role_conf", "novelty", "is_novel", "art_category", "knn_dist"):
        if c not in df.columns:
            df[c] = np.nan

    try:
        tiles = dataio.tiles_df()
    except dataio.DataUnavailable:
        tiles = pd.DataFrame(columns=TILE_META_COLS)
    tile_meta_cols = [c for c in TILE_META_COLS if c in tiles.columns and c != "tile_id"]
    if "tile_id" in tiles.columns:
        df = df.merge(tiles[["tile_id", *tile_meta_cols]], on="tile_id", how="left")

    # domain: prefer the det index's own column; fall back to tiles.parquet's
    # own `domain` (present in the test fixture, not in real data yet), then
    # to the GT-chip-derived per-parent domain (real data today).
    if "domain" not in df.columns:
        df["domain"] = None
    missing = df["domain"].isna()
    if missing.any() and "domain" in tiles.columns and "tile_id" in tiles.columns:
        tile_domain = dict(zip(tiles["tile_id"], tiles["domain"]))
        df.loc[missing, "domain"] = df.loc[missing, "tile_id"].map(tile_domain)
    missing = df["domain"].isna()
    if missing.any():
        dbp = _domain_by_parent()
        if dbp:
            df.loc[missing, "domain"] = df.loc[missing, "parent_id"].map(dbp)

    if "collection_date" in df.columns:
        df["date"] = pd.to_datetime(df["collection_date"], errors="coerce", utc=True).dt.strftime("%Y-%m-%d")
    else:
        df["date"] = None

    return df


def _det_chip_dict(row: pd.Series, review: dict | None) -> dict:
    det_id = int(row["det_id"])
    tile_id = int(row["tile_id"])
    return {
        "source": "det",
        "det_id": det_id,
        "ann_id": det_id,  # for UI reuse of the shared review/chip-drawer components
        "tile_id": tile_id,
        "bbox": [_num(row.get("x1")), _num(row.get("y1")), _num(row.get("w")), _num(row.get("h"))],
        "conf": _num(row.get("conf")),
        "chip_url": f"/api/det/{det_id}/image",
        "tile_url": f"/api/tiles/{tile_id}/image",
        "role_pred": dataio.role_label(row.get("role_pred")),
        "role_conf": _num(row.get("role_conf")),
        "novelty": _num(row.get("novelty")),
        "is_novel": _bool(row.get("is_novel")),
        "knn_dist": _num(row.get("knn_dist")),
        "meta": {
            "airfield": row.get("airfield"),
            "country": row.get("country"),
            "weather": row.get("weather"),
            "off_nadir": _num(row.get("off_nadir")),
            "gsd": _num(row.get("gsd")),
        },
        "review": review,
    }


# ---------------------------------------------------------------- grouping --

def _first_non_null(series: pd.Series):
    s = series.dropna()
    return s.iloc[0] if len(s) else None


def _new_collections_table(version: str | None) -> tuple[pd.DataFrame, float | None]:
    """(det chips in NEW collections with a `group_id` and boolean `_flagged` column, reference_flag_rate).

    Grouping key = (airfield, collection date UTC day) -- never per-parent (STATE.md). `reference_flag_rate`
    is computed from ref_holdout's OWN det chips (the labeled control set) -- the only "evidence baseline"
    this module uses; ref_holdout itself never appears as a collection. Empty frame if det chips don't
    exist yet.
    """
    df = _det_full_table(version)
    if df.empty:
        return df.iloc[0:0], None

    _, _, reference_flag_rate = _flag_stats(df[df["domain"] == "ref_holdout"])

    df_new = df[df["domain"].notna() & ~df["domain"].isin(REFERENCE_ALL_DOMAINS)]
    df_new = df_new[df_new["airfield"].notna() & df_new["date"].notna()].copy()
    if df_new.empty:
        return df_new, reference_flag_rate

    if "is_novel" in df_new.columns:
        df_new["_flagged"] = pd.to_numeric(df_new["is_novel"], errors="coerce").fillna(0).astype(bool)
    else:
        df_new["_flagged"] = False

    # group_id per distinct (airfield, date); airfield names that slugify to the same string on the
    # same day (rare, but possible with punctuation-only differences) get a stable -2, -3, ... suffix
    # rather than silently merging.
    keys = df_new[["airfield", "date"]].drop_duplicates().itertuples(index=False)
    gid_by_key: dict[tuple, str] = {}
    seen: dict[str, int] = {}
    for airfield, date in keys:
        gid = _group_id(airfield, date)
        seen[gid] = seen.get(gid, 0) + 1
        gid_by_key[(airfield, date)] = gid if seen[gid] == 1 else f"{gid}-{seen[gid]}"
    df_new["group_id"] = [gid_by_key[(a, d)] for a, d in zip(df_new["airfield"], df_new["date"])]
    return df_new, reference_flag_rate


def _build_collections(version: str | None) -> tuple[list[dict], pd.DataFrame, float | None]:
    """Returns (groups sorted by date, the new-collections det table with `group_id`, reference_flag_rate).
    One vectorized groupby (not a per-group Python loop) -- /collections is hit on every list load."""
    df_new, reference_flag_rate = _new_collections_table(version)
    if df_new.empty:
        return [], df_new, reference_flag_rate

    det_review_map = _det_reviews_by_det_id()
    df_new = df_new.assign(_reviewed=df_new["det_id"].astype(int).isin(set(det_review_map.keys())))
    bad_box_ids = {det_id for det_id, r in det_review_map.items() if r.get("action") == "bad_box"}
    df_new = df_new.assign(_bad_box=df_new["det_id"].astype(int).isin(bad_box_ids))

    by = df_new.groupby("group_id", sort=False)
    agg = by.agg(
        airfield=("airfield", "first"), date=("date", "first"),
        country=("country", _first_non_null), state=("state", _first_non_null), weather=("weather", _first_non_null),
        n_parents=("parent_id", "nunique"), n_tiles=("tile_id", "nunique"),
        n_detections=("det_id", "size"), n_flagged=("_flagged", "sum"), n_reviewed=("_reviewed", "sum"),
        n_bad_box=("_bad_box", "sum"),
    )
    # per-tile means (each tile counted once, not once per det chip)
    tiles_once = df_new.drop_duplicates(subset=["tile_id"])
    means = {}
    for col in ("off_nadir", "gsd", "lat", "lon"):
        if col in tiles_once.columns:
            means[col] = pd.to_numeric(tiles_once[col], errors="coerce").groupby(tiles_once["group_id"]).mean()
    agg = agg.sort_values("date", kind="stable")

    groups: list[dict] = []
    for gid, r in agg.iterrows():
        n_detections = int(r["n_detections"])
        n_flagged = int(r["n_flagged"])
        verdict = _status_for(n_detections, n_flagged, reference_flag_rate)
        groups.append({
            "group_id": gid,
            "airfield": r["airfield"],
            "country": r["country"],
            "state": r["state"],
            "date": r["date"],
            "weather": r["weather"],
            "off_nadir": _num(means["off_nadir"].get(gid)) if "off_nadir" in means else None,
            "gsd": _num(means["gsd"].get(gid)) if "gsd" in means else None,
            "lat": _num(means["lat"].get(gid)) if "lat" in means else None,
            "lon": _num(means["lon"].get(gid)) if "lon" in means else None,
            "n_parents": int(r["n_parents"]),
            "n_tiles": int(r["n_tiles"]),
            "n_detections": n_detections,
            "n_flagged": n_flagged,
            "flag_rate": (n_flagged / n_detections) if n_detections else None,
            "excess_flagged": verdict["excess_flagged"],
            "p_value": verdict["p_value"],
            "n_reviewed": int(r["n_reviewed"]),
            "n_bad_box": int(r["n_bad_box"]),
            "status": verdict["status"],
        })
    return groups, df_new, reference_flag_rate


# --------------------------------------------------------------- det queue --

def flagged_det_queue(version: str | None, strategy: str = "collection", limit: int = 24, min_conf: float = 0.5) -> list[dict]:
    """Field-mode review queue (GET /review/queue?source=det): FLAGGED (is_novel) detector chips
    across NEW collections only (never ref_train/ref_calib/ref_holdout -- exactly the groups
    /collections serves), not yet reviewed (no source='det' review), with detector confidence
    >= `min_conf` (detector false positives are novel by definition and cluster at low confidence;
    0.5 default, 0 disables), ranked by `strategy`: collection (default: needs_review collections
    first, by excess flagged desc, then novelty desc within a collection) | novelty desc |
    uncertainty (role_conf asc) | random. Each item is the same DetChip dict as
    /collections/{group_id}'s chips plus `group_id`, so the UI can link back to its collection.
    Empty list (never an error) if det chips/scores don't exist yet.
    """
    df, _ref = _new_collections_table(version)
    if df.empty:
        return []
    df = df[df["_flagged"]]
    det_review_map = _det_reviews_by_det_id()
    if det_review_map:
        df = df[~df["det_id"].astype(int).isin(set(det_review_map.keys()))]
    if df.empty:
        return []
    if min_conf > 0 and "conf" in df.columns:
        df = df[df["conf"].fillna(0.0) >= min_conf]
        if df.empty:
            return []
    if strategy == "uncertainty" and "role_conf" in df.columns:
        df = df.sort_values("role_conf", ascending=True, na_position="last")
    elif strategy == "random":
        df = df.sample(frac=1)
    elif strategy == "collection":
        groups, _t, _r = _build_collections(version)
        prio = {g["group_id"]: (0 if g["status"] == "needs_review" else 1, -(g.get("excess_flagged") or 0.0)) for g in groups}
        df = df.assign(_p0=[prio.get(g, (2, 0.0))[0] for g in df["group_id"]],
                       _p1=[prio.get(g, (2, 0.0))[1] for g in df["group_id"]])
        df = df.sort_values(["_p0", "_p1", "group_id", "novelty"], ascending=[True, True, True, False], na_position="last")
    elif "novelty" in df.columns:
        df = df.sort_values("novelty", ascending=False, na_position="last")
    page = df.head(limit)
    ginfo = {g["group_id"]: g for g in _build_collections(version)[0]} if strategy == "collection" else {}
    out = []
    for _, r in page.iterrows():
        item = {**_det_chip_dict(r, None), "group_id": r["group_id"]}
        g = ginfo.get(r["group_id"])
        if g:
            item.update({"airfield": g["airfield"], "date": g["date"], "group_status": g["status"], "group_n_flagged": g["n_flagged"]})
        out.append(item)
    return out


# ------------------------------------------------------------------ routes --

@router.get("")
def get_collections():
    groups, _table, reference_flag_rate = _build_collections(_safe_version())
    return {
        "groups": groups,
        "grouping": GROUPING,
        "min_denominator": MIN_DENOMINATOR,
        "p_threshold": P_THRESHOLD,
        "reference": _reference_block(),
        "reference_flag_rate": reference_flag_rate,
        "status_rule": status_rule_for(reference_flag_rate),
    }


@router.get("/summary")
def get_collections_summary():
    groups, _table, reference_flag_rate = _build_collections(_safe_version())
    return {
        "n_groups": len(groups),
        "n_needs_review": sum(1 for g in groups if g["status"] == "needs_review"),
        "n_insufficient_data": sum(1 for g in groups if g["status"] == "insufficient_data"),
        "n_detections": sum(g["n_detections"] for g in groups),
        "n_flagged": sum(g["n_flagged"] for g in groups),
        "reference_flag_rate": reference_flag_rate,
    }


@router.get("/{group_id}")
def get_collection_detail(
    group_id: str,
    sort: str = Query("novelty", pattern="^(novelty|conf)$"),
    order: str = Query("desc", pattern="^(asc|desc)$"),
    limit: int = Query(48, ge=1, le=500),
    offset: int = Query(0, ge=0),
    only_flagged: bool = Query(False),
    reviewed: str = Query("all", pattern="^(all|yes|no)$"),
):
    version = _safe_version()
    groups, table, _reference_flag_rate = _build_collections(version)
    group = next((g for g in groups if g["group_id"] == group_id), None)
    if group is None:
        raise HTTPException(404, f"unknown collection group id: {group_id}")

    g = table[table["group_id"] == group_id]

    tiles_out = []
    for tile_id, tg in g.groupby("tile_id"):
        n_det, n_flag, _ = _flag_stats(tg)
        tiles_out.append({"tile_id": int(tile_id), "n_detections": n_det, "n_flagged": n_flag})
    tiles_out.sort(key=lambda t: t["tile_id"])

    if only_flagged:
        g = g[g["_flagged"]]

    det_review_map = _det_reviews_by_det_id()
    if reviewed != "all":
        reviewed_ids = set(det_review_map.keys())
        mask = g["det_id"].astype(int).isin(reviewed_ids)
        g = g[mask] if reviewed == "yes" else g[~mask]

    if sort in g.columns:
        g = g.sort_values(sort, ascending=(order == "asc"), na_position="last")

    total = int(len(g))
    page = g.iloc[offset:offset + limit]
    chips = [_det_chip_dict(r, det_review_map.get(int(r["det_id"]))) for _, r in page.iterrows()]

    return {**group, "tiles": tiles_out, "chips": {"total": total, "items": chips}}
