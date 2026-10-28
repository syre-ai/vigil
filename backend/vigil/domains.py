"""Domain assignment for Vigil's operational stream.

Every tile's domain is decided at PARENT-IMAGE granularity, so tiles that
are spatial neighbours (they all come from tiling the same parent image)
never straddle the reference/operational boundary. Priority order (first
match wins), evaluated on a per-parent-image basis:

    (a) parent image contains >=1 military aircraft (role_id 4-7) -> 'military'
    (b) Weather == 'Snow'                                          -> 'snow'
    (c) Weather == 'Cloud Cover or Haze'                           -> 'haze'
    (d) Country != 'USA'                                           -> 'foreign'
    (e) off_nadir > 25                                             -> 'off_nadir'
    (f) else                                                       -> 'reference'

'reference' parent images are then split deterministically (seed 0) three
ways:
    ref_train   (60%) -- fits the detector + prototype classifier
    ref_calib   (20%) -- calibrates vigilance/thresholds/the drift null;
                          never used for fitting anything
    ref_holdout (20%) -- the "control" batch in the operational stream

ref_calib is NOT part of the operational stream (batch_order = -1).
Everything else gets a `batch_order` for the operational playback:
    ref_holdout=0, off_nadir=1, foreign=2, haze=3, snow=4, military=5
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MILITARY_ROLE_MIN = 4

REF_TRAIN_FRAC = 0.60
REF_CALIB_FRAC = 0.20
# ref_holdout gets the remainder (~0.20)

SPLIT_SEED = 0

BATCH_ORDER = {
    "ref_holdout": 0,
    "off_nadir": 1,
    "foreign": 2,
    "haze": 3,
    "snow": 4,
    "military": 5,
}
NOT_IN_STREAM = -1  # ref_train, ref_calib

DOMAIN_PRINT_ORDER = [
    "ref_train", "ref_calib", "ref_holdout",
    "off_nadir", "foreign", "haze", "snow", "military",
]

MIN_OPERATIONAL_TILES = 40
MIN_REF_TRAIN_TILES = 2000


def _base_domain(row: pd.Series) -> str:
    if row["_parent_has_military"]:
        return "military"
    if row["weather"] == "Snow":
        return "snow"
    if row["weather"] == "Cloud Cover or Haze":
        return "haze"
    if row["country"] != "USA":
        return "foreign"
    if pd.notna(row["off_nadir"]) and row["off_nadir"] > 25:
        return "off_nadir"
    return "reference"


def assign_domains(tiles: pd.DataFrame) -> pd.DataFrame:
    """Add `domain` (str) and `batch_order` (int) columns to `tiles`,
    assigned at parent-image granularity. Returns a new DataFrame."""
    tiles = tiles.copy()

    parent_has_mil = tiles.groupby("parent_id")["n_military"].sum() > 0
    parent_info = (
        tiles.groupby("parent_id")
        .agg(weather=("weather", "first"), country=("country", "first"),
             off_nadir=("off_nadir", "first"))
        .reset_index()
    )
    parent_info["_parent_has_military"] = parent_info["parent_id"].map(parent_has_mil).fillna(False)
    parent_info["_base_domain"] = parent_info.apply(_base_domain, axis=1)

    # Deterministic 60/20/20 split of the 'reference' parents.
    ref_parents = sorted(parent_info.loc[parent_info["_base_domain"] == "reference", "parent_id"])
    n = len(ref_parents)
    rng = np.random.RandomState(SPLIT_SEED)
    perm = rng.permutation(n)
    n_train = int(round(n * REF_TRAIN_FRAC))
    n_calib = int(round(n * REF_CALIB_FRAC))
    ref_arr = np.array(ref_parents, dtype=object)
    train_ids = set(ref_arr[perm[:n_train]])
    calib_ids = set(ref_arr[perm[n_train:n_train + n_calib]])
    # holdout_ids = everything else (implicit)

    def _finalize(pid: str, base: str) -> str:
        if base != "reference":
            return base
        if pid in train_ids:
            return "ref_train"
        if pid in calib_ids:
            return "ref_calib"
        return "ref_holdout"

    parent_info["domain"] = [
        _finalize(pid, base)
        for pid, base in zip(parent_info["parent_id"], parent_info["_base_domain"])
    ]
    parent_info["batch_order"] = parent_info["domain"].map(BATCH_ORDER).fillna(NOT_IN_STREAM).astype(int)

    tiles = tiles.merge(parent_info[["parent_id", "domain", "batch_order"]], on="parent_id", how="left")
    return tiles


def summarize_domains(tiles: pd.DataFrame) -> pd.DataFrame:
    """One row per domain: n parent images, n tiles, n aircraft, n military
    aircraft, and the set of countries present. `tiles` must already have
    a `domain` column (see assign_domains)."""
    rows = []
    for dom, g in tiles.groupby("domain"):
        rows.append({
            "domain": dom,
            "batch_order": int(g["batch_order"].iloc[0]),
            "n_parent_images": g["parent_id"].nunique(),
            "n_tiles": len(g),
            "n_aircraft": int(g["n_aircraft"].sum()),
            "n_military_aircraft": int(g["n_military"].sum()),
            "countries": ", ".join(sorted(g["country"].dropna().unique().tolist())),
        })
    df = pd.DataFrame(rows)
    order_key = {d: i for i, d in enumerate(DOMAIN_PRINT_ORDER)}
    df["_ord"] = df["domain"].map(order_key).fillna(999)
    df = df.sort_values("_ord").drop(columns="_ord").reset_index(drop=True)
    return df


def _check_thresholds(summary: pd.DataFrame) -> list[str]:
    warnings = []
    operational = {"off_nadir", "foreign", "haze", "snow", "military", "ref_holdout"}
    for _, row in summary.iterrows():
        if row["domain"] in operational and row["n_tiles"] < MIN_OPERATIONAL_TILES:
            warnings.append(
                f"domain '{row['domain']}' has only {row['n_tiles']} tiles "
                f"(< {MIN_OPERATIONAL_TILES} target for an operational batch)"
            )
        if row["domain"] == "ref_train" and row["n_tiles"] < MIN_REF_TRAIN_TILES:
            warnings.append(
                f"'ref_train' has only {row['n_tiles']} tiles "
                f"(< {MIN_REF_TRAIN_TILES} target for detector/prototype training)"
            )
    return warnings


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from vigil.data import load_tiles

    tiles = assign_domains(load_tiles())
    summary = summarize_domains(tiles)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 80)
    print(summary.to_string(index=False))

    warnings = _check_thresholds(summary)
    if warnings:
        print("\nDOMAIN-SIZE WARNINGS (rules kept as specified, flagging only):")
        for w in warnings:
            print(f"  - {w}")
    else:
        print("\nAll domains meet the minimum size targets.")
