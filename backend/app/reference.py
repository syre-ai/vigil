"""Validation set / reference -- `GET /api/reference`.

Backs the analyst view's landing page: how big and how varied the ONLY
labeled data in this system is, what classes the detector/role-head shipped
knowing, and the flag rate its own novelty layer produces on that same
labeled set.

Reference = the union of the three RarePlanes domains split out at
parent-image granularity by `vigil.domains.assign_domains` (see
backend/API.md's vocabulary section): `ref_train` (60%, fits the detector +
prototype classifier), `ref_calib` (20%, calibrates vigilance/thresholds --
never used for fitting), and `ref_holdout` (20%, the "control" batch never
used for fitting or calibration). All three are GROUND-TRUTH LABELED -- the
only labeled imagery in the whole system; every image collected since lives
in `/api/collections` instead, entirely unlabeled.

No forbidden-ground-truth-key concern here (contrast backend/API.md's
field-mode "no ground truth anywhere in these payloads" rule for
`/collections*`): this endpoint IS the labeled set, so role/class
information is legitimate and the whole point of the page.

`flag_rate` reuses `collections._new_collections_table`'s own
`reference_flag_rate` computation (the `ref_holdout` control set's det-chip
flag rate) so this page and `/api/collections` can never disagree about that
number.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi import APIRouter

from . import dataio
from . import collections as collections_api
from .constants import CIVIL_ROLE_IDS, REFERENCE_DOMAINS

router = APIRouter(prefix="/api/reference", tags=["reference"])

NAME = "labeled validation set (reference)"
N_EXAMPLES_PER_CLASS = 8
N_EXAMPLE_TILES = 12
EXAMPLE_SEED = 0

TILE_COLS = [
    "tile_id", "parent_id", "airfield", "country", "weather",
    "off_nadir", "gsd", "collection_date", "n_aircraft",
]


# ---------------------------------------------------------------- table --

def _reference_chip_table() -> pd.DataFrame:
    """One row per reference-domain (ref_train ∪ ref_calib ∪ ref_holdout) GT
    chip: ann_id, tile_id, domain, role_id (chip index's own), role (GT name,
    from annotations), parent_id, airfield, country, weather, off_nadir,
    gsd, collection_date, n_aircraft. Empty (never an error) if the chip
    index / annotations / tiles files aren't built yet.
    """
    try:
        chips = dataio.chips_index_df()
    except dataio.DataUnavailable:
        return pd.DataFrame()
    if "domain" not in chips.columns:
        return pd.DataFrame()
    ref = chips[chips["domain"].isin(collections_api.REFERENCE_ALL_DOMAINS)]
    if ref.empty:
        return ref

    try:
        ann = dataio.annotations_df()
    except dataio.DataUnavailable:
        return ref
    if "role" in ann.columns:
        ref = ref.merge(ann[["ann_id", "role"]], on="ann_id", how="left")

    try:
        tiles = dataio.tiles_df()
    except dataio.DataUnavailable:
        return ref
    tile_cols = [c for c in TILE_COLS if c in tiles.columns and c != "tile_id"]
    if "tile_id" in tiles.columns:
        ref = ref.merge(tiles[["tile_id", *tile_cols]], on="tile_id", how="left")
    return ref


# ----------------------------------------------------------------- picks --

def _pick_examples(df_role: pd.DataFrame, k: int = N_EXAMPLES_PER_CLASS, seed: int = EXAMPLE_SEED) -> list[dict]:
    """Up to `k` example chips for one class, spread across as many distinct
    parent images as possible -- round-robin over parent_id groups (each
    shuffled under `seed`) rather than letting one image's chips dominate.
    Deterministic under `seed` so the page is stable across reloads.
    """
    if df_role.empty or "parent_id" not in df_role.columns:
        return []
    sub = df_role.dropna(subset=["parent_id"])
    if sub.empty:
        return []
    rng = np.random.RandomState(seed)
    buckets: dict[str, list[dict]] = {}
    for pid, g in sub.groupby("parent_id"):
        order = rng.permutation(len(g))
        buckets[str(pid)] = g.iloc[order].to_dict("records")
    pids = sorted(buckets)
    picks: list[dict] = []
    i = 0
    while len(picks) < k and any(buckets[p] for p in pids):
        pid = pids[i % len(pids)]
        i += 1
        if buckets[pid]:
            picks.append(buckets[pid].pop(0))
    return [{"ann_id": int(r["ann_id"]), "chip_url": f"/api/chips/{int(r['ann_id'])}/image"} for r in picks[:k]]


def _classes_block(df: pd.DataFrame) -> list[dict]:
    if df.empty or "role" not in df.columns:
        return []
    out = []
    for role, g in df.groupby("role"):
        if pd.isna(role):
            continue
        out.append({"label": str(role), "n_chips": int(len(g)), "examples": _pick_examples(g)})
    out.sort(key=lambda c: c["label"])
    return out


# ------------------------------------------------------------- conditions --

def _image_counts_by(df: pd.DataFrame, col: str) -> dict[str, int]:
    if df.empty or col not in df.columns or "parent_id" not in df.columns:
        return {}
    sub = df.dropna(subset=[col, "parent_id"])
    if sub.empty:
        return {}
    counts = sub.drop_duplicates(subset=["parent_id"]).groupby(col)["parent_id"].nunique()
    return {str(k): int(v) for k, v in counts.items()}


def _minmax(series: pd.Series) -> dict[str, float | None]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return {"min": None, "max": None}
    return {"min": float(clean.min()), "max": float(clean.max())}


def _date_minmax(series: pd.Series) -> dict[str, str | None]:
    dt = pd.to_datetime(series, errors="coerce", utc=True).dropna()
    if dt.empty:
        return {"min": None, "max": None}
    return {"min": dt.min().strftime("%Y-%m-%d"), "max": dt.max().strftime("%Y-%m-%d")}


def _conditions_block(df: pd.DataFrame) -> dict:
    tiles_once = df.drop_duplicates(subset=["tile_id"]) if "tile_id" in df.columns else df
    return {
        "weather": _image_counts_by(df, "weather"),
        "countries": _image_counts_by(df, "country"),
        "off_nadir": _minmax(tiles_once["off_nadir"]) if "off_nadir" in tiles_once.columns else {"min": None, "max": None},
        "gsd": _minmax(tiles_once["gsd"]) if "gsd" in tiles_once.columns else {"min": None, "max": None},
        "dates": _date_minmax(tiles_once["collection_date"]) if "collection_date" in tiles_once.columns else {"min": None, "max": None},
    }


# ---------------------------------------------------------- example tiles --

def _example_tiles(df: pd.DataFrame, k: int = N_EXAMPLE_TILES) -> list[dict]:
    needed = {"tile_id", "parent_id", "n_aircraft", "airfield"}
    if df.empty or not needed.issubset(df.columns):
        return []
    tiles = df.drop_duplicates(subset=["tile_id"])[["tile_id", "parent_id", "n_aircraft", "airfield"]]
    tiles = tiles.dropna(subset=["n_aircraft", "parent_id"])
    if tiles.empty:
        return []
    # One tile per distinct parent image -- its own highest-aircraft-count tile.
    idx = tiles.groupby("parent_id")["n_aircraft"].idxmax()
    best = tiles.loc[idx].sort_values(["n_aircraft", "tile_id"], ascending=[False, True]).head(k)
    return [
        {
            "tile_id": int(r.tile_id),
            "tile_url": f"/api/tiles/{int(r.tile_id)}/image",
            "airfield": r.airfield,
            "n_aircraft": int(r.n_aircraft),
        }
        for r in best.itertuples(index=False)
    ]


# --------------------------------------------------------------- helpers --

def _detector_block() -> dict:
    card = dataio.detector_card() or {}
    return {
        "name": card.get("name"),
        "val_map50": card.get("val_map50"),
        "val_map50_95": card.get("val_map50_95"),
        "trained_on": card.get("trained_on"),
    }


def _reference_definition(version: str | None) -> str | None:
    if not version:
        return None
    try:
        model = dataio.model_json(version)
    except dataio.DataUnavailable:
        return None
    return model.get("reference_definition")


def _flag_rate(version: str | None) -> float | None:
    try:
        _table, reference_flag_rate = collections_api._new_collections_table(version)
    except Exception:
        return None
    return reference_flag_rate


# ------------------------------------------------------------------ route --

@router.get("")
def get_reference():
    version = collections_api._safe_version()
    df = _reference_chip_table()

    n_chips = int(len(df))
    has_tiles = n_chips and "tile_id" in df.columns
    has_images = n_chips and "parent_id" in df.columns
    n_tiles = int(df["tile_id"].nunique()) if has_tiles else 0
    n_images = int(df["parent_id"].nunique()) if has_images else 0

    fit_images = 0
    control_images = 0
    if has_images and "domain" in df.columns:
        fit_images = int(df.loc[df["domain"].isin(REFERENCE_DOMAINS), "parent_id"].nunique())
        control_images = int(df.loc[df["domain"] == "ref_holdout", "parent_id"].nunique())

    military_aircraft = 0
    if n_chips and "role_id" in df.columns:
        military_aircraft = int((~df["role_id"].isin(CIVIL_ROLE_IDS)).fillna(True).sum())

    return {
        "name": NAME,
        "n_images": n_images,
        "n_tiles": n_tiles,
        "n_chips": n_chips,
        "military_aircraft": military_aircraft,
        "splits": {"fit": fit_images, "control": control_images},
        "classes": _classes_block(df),
        "conditions": _conditions_block(df),
        "example_tiles": _example_tiles(df),
        "detector": _detector_block(),
        "flag_rate": _flag_rate(version),
        "definition": _reference_definition(version),
    }
