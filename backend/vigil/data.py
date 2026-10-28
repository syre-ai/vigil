"""Data layer for Vigil: loads RarePlanes tiles + annotations into tidy,
cached DataFrames.

Two public entry points:

    load_tiles()       -> one row per 512x512 tile (parent-image metadata,
                           geo centroid, and an aircraft-count summary
                           attached to each tile).
    load_annotations() -> one row per aircraft bounding box.

Both are built from the COCO-tiled annotation files, the public metadata
CSV, and the public GeoJSON footprints. Nothing here touches the tile PNGs
themselves, so this module works whether or not the image download has
finished -- `image_path` is always computed, but existence is never checked
here (callers that need pixels, e.g. scripts/build_yolo_dataset.py and
scripts/extract_chips.py, are responsible for skipping missing files).

Results are cached to data/rareplanes/tiles.parquet and
data/rareplanes/annotations.parquet. Both parquet files are always built
together (they share a `tile_id` / `ann_id` space), and are rebuilt
whenever either is missing or `force=True` is passed.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from vigil.paths import RP_DIR  # honors VIGIL_DATA_ROOT, same resolution as backend/app/config.py

REPO_ROOT = Path(__file__).resolve().parents[2]
ANN_DIR = RP_DIR / "ann"

TRAIN_COCO_PATH = ANN_DIR / "RarePlanes_Train_Coco_Annotations_tiled.json"
TEST_COCO_PATH = ANN_DIR / "RarePlanes_Test_Coco_Annotations_tiled.json"
METADATA_CSV_PATH = ANN_DIR / "RarePlanes_Public_Metadata.csv"
GEOJSON_PATH = ANN_DIR / "RarePlanes_Public_All_Annotations.geojson"

TILES_PARQUET = RP_DIR / "tiles.parquet"
ANNOTATIONS_PARQUET = RP_DIR / "annotations.parquet"

MILITARY_ROLE_MIN = 4  # role_id 4-7 are the military roles

_PARENT_RE = re.compile(r"^(\d+_[0-9A-Fa-f]+)_tile_\d+\.png$")

_TILE_COLUMNS = [
    "tile_id", "file_name", "image_path", "coco_split", "parent_id",
    "loc_id", "cat_id", "country", "state", "airfield", "weather",
    "off_nadir", "gsd", "sun_elev", "cloud_cover", "collection_date",
    "lat", "lon", "n_aircraft", "n_military", "roles", "has_military",
]

_ANN_COLUMNS = [
    "ann_id", "tile_id", "file_name", "x", "y", "w", "h", "role",
    "role_id", "wingspan", "length", "truncated", "partial",
]


def _parent_id_from_filename(file_name: str) -> str:
    m = _PARENT_RE.match(file_name)
    if not m:
        raise ValueError(f"could not parse parent id out of file_name {file_name!r}")
    return m.group(1)


def _load_coco(path: Path, coco_split: str, image_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    with open(path) as f:
        d = json.load(f)

    images = pd.DataFrame(d["images"])[["id", "file_name"]].rename(
        columns={"id": "_orig_image_id"}
    )
    images["coco_split"] = coco_split
    images["parent_id"] = images["file_name"].apply(_parent_id_from_filename)
    images["image_path"] = images["file_name"].apply(lambda fn: str(image_dir / fn))

    anns = pd.DataFrame(d["annotations"]).rename(
        columns={"id": "_orig_ann_id", "image_id": "_orig_image_id"}
    )
    anns["coco_split"] = coco_split
    return images, anns


def _match_metadata(parent_ids: list[str], meta: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Match parent_ids against the metadata CSV.

    Primary key: exact string match on `image_id` (== our parent_id, since
    both are "<loc_id>_<cat_id>"). Some `cat_id` values in the CSV were
    mangled by Excel into floats (e.g. '1.04001E+15'), so as a fallback for
    any parent_id that doesn't match directly, we match on the CSV's
    (reliable, numeric) `loc_id` column, disambiguating by a cat_id prefix
    match if a loc_id has more than one row.

    Returns (matched metadata rows indexed by parent_id, n_failed_to_match).
    """
    meta = meta.copy()
    meta["image_id"] = meta["image_id"].astype(str)
    by_image_id = meta.set_index("image_id", drop=False)

    rows = {}
    n_fallback = 0
    n_failed = 0
    for pid in parent_ids:
        if pid in by_image_id.index:
            rows[pid] = by_image_id.loc[pid]
            continue

        loc_str, _, cat_str = pid.partition("_")
        try:
            loc_num = int(loc_str)
        except ValueError:
            loc_num = None

        candidates = meta[meta["loc_id"] == loc_num] if loc_num is not None else meta.iloc[0:0]
        if len(candidates) == 1:
            rows[pid] = candidates.iloc[0]
            n_fallback += 1
        elif len(candidates) > 1:
            prefix = cat_str[:6]
            narrowed = candidates[candidates["cat_id"].astype(str).str.startswith(prefix)]
            if len(narrowed) == 1:
                rows[pid] = narrowed.iloc[0]
                n_fallback += 1
            else:
                n_failed += 1
        else:
            n_failed += 1

    if n_fallback:
        print(f"[vigil.data] matched {n_fallback} parent image(s) to metadata via loc_id fallback")
    if n_failed:
        print(f"[vigil.data] WARNING: {n_failed} parent image(s) have NO metadata match "
              f"(direct or fallback) -- their tiles will carry null condition metadata")

    matched = pd.DataFrame(rows.values())
    matched.insert(0, "parent_id", list(rows.keys()))
    return matched, n_failed


def _load_metadata() -> pd.DataFrame:
    meta = pd.read_csv(METADATA_CSV_PATH)
    meta["collection_date"] = pd.to_datetime(
        meta["collection_time_start"], format="%m/%d/%y %H:%M", errors="coerce"
    ).dt.normalize()
    meta["gsd"] = (meta["pan_resolution_minimum"] + meta["pan_resolution_maximum"]) / 2.0
    return meta


def _compute_centroids() -> pd.DataFrame:
    """Per-parent-image (loc_id + cat_id) lat/lon centroid, averaged over
    every aircraft footprint polygon that belongs to that parent image."""
    with open(GEOJSON_PATH) as f:
        gj = json.load(f)

    rows = []
    for feat in gj["features"]:
        props = feat["properties"]
        ring = feat["geometry"]["coordinates"][0]
        lon = sum(c[0] for c in ring) / len(ring)
        lat = sum(c[1] for c in ring) / len(ring)
        rows.append({
            "parent_id": f"{props['loc_id']}_{props['cat_id']}",
            "lon": lon,
            "lat": lat,
        })
    df = pd.DataFrame(rows)
    return df.groupby("parent_id")[["lat", "lon"]].mean().reset_index()


def _build() -> tuple[pd.DataFrame, pd.DataFrame]:
    images_tr, anns_tr = _load_coco(TRAIN_COCO_PATH, "train", RP_DIR / "train")
    images_te, anns_te = _load_coco(TEST_COCO_PATH, "test", RP_DIR / "test")

    # tile_id: fresh, globally-unique ids (COCO 'id' collides between the
    # two files, since each is numbered independently starting at 1).
    images_tr = images_tr.reset_index(drop=True)
    images_te = images_te.reset_index(drop=True)
    images_tr["tile_id"] = range(0, len(images_tr))
    images_te["tile_id"] = range(len(images_tr), len(images_tr) + len(images_te))

    tile_id_map_tr = dict(zip(images_tr["_orig_image_id"], images_tr["tile_id"]))
    tile_id_map_te = dict(zip(images_te["_orig_image_id"], images_te["tile_id"]))
    anns_tr["tile_id"] = anns_tr["_orig_image_id"].map(tile_id_map_tr)
    anns_te["tile_id"] = anns_te["_orig_image_id"].map(tile_id_map_te)

    images = pd.concat([images_tr, images_te], ignore_index=True)
    anns = pd.concat([anns_tr, anns_te], ignore_index=True)

    if anns["tile_id"].isna().any():
        raise ValueError("some annotations did not map to a tile_id -- COCO image_id mismatch")

    anns = anns.reset_index(drop=True)
    anns["ann_id"] = range(len(anns))
    anns["role_id"] = anns["role_id"].astype(int)

    # ---- attach parent-image metadata (country/state/weather/etc.) ----
    parent_ids = sorted(images["parent_id"].unique())
    meta, n_failed = _match_metadata(parent_ids, _load_metadata())
    meta = meta.rename(columns={
        "Country": "country", "State_Province": "state", "Air_Field": "airfield",
        "Weather": "weather", "off_nadir_max": "off_nadir",
        "avg_sun_elevation_angle": "sun_elev", "cloud_cover_percentage": "cloud_cover",
    })
    meta_cols = ["parent_id", "country", "state", "airfield", "weather",
                 "off_nadir", "gsd", "sun_elev", "cloud_cover", "collection_date"]
    images = images.merge(meta[meta_cols], on="parent_id", how="left")

    # ---- attach lat/lon centroids ----
    centroids = _compute_centroids()
    images = images.merge(centroids, on="parent_id", how="left")

    # ---- attach loc_id / cat_id (parsed straight from the filename --
    # these are reliable, unlike the CSV's Excel-mangled cat_id column) ----
    split_ids = images["parent_id"].str.split("_", n=1, expand=True)
    images["loc_id"] = split_ids[0].astype(int)
    images["cat_id"] = split_ids[1]

    # ---- per-tile aircraft summary, from the annotations we just built ----
    def _roles_list(s: pd.Series) -> list:
        return sorted(int(v) for v in s.tolist())

    agg = anns.groupby("tile_id").agg(
        n_aircraft=("ann_id", "count"),
        n_military=("role_id", lambda s: int((s >= MILITARY_ROLE_MIN).sum())),
        roles=("role_id", _roles_list),
    ).reset_index()
    images = images.merge(agg, on="tile_id", how="left")
    images["n_aircraft"] = images["n_aircraft"].fillna(0).astype(int)
    images["n_military"] = images["n_military"].fillna(0).astype(int)
    images["roles"] = images["roles"].apply(lambda v: v if isinstance(v, list) else [])
    images["has_military"] = images["n_military"] > 0

    tiles = images[_TILE_COLUMNS].sort_values("tile_id").reset_index(drop=True)

    # ---- final annotations table ----
    bbox = pd.DataFrame(anns["bbox"].tolist(), columns=["x", "y", "w", "h"], index=anns.index)
    file_name_by_tile = dict(zip(images["tile_id"], images["file_name"]))
    anns_out = pd.DataFrame({
        "ann_id": anns["ann_id"],
        "tile_id": anns["tile_id"].astype(int),
        "file_name": anns["tile_id"].map(file_name_by_tile),
        "x": bbox["x"],
        "y": bbox["y"],
        "w": bbox["w"],
        "h": bbox["h"],
        "role": anns["role"],
        "role_id": anns["role_id"],
        "wingspan": anns["wingspan"],
        "length": anns["length"],
        "truncated": anns["truncated"].astype(bool),
        "partial": anns["partialDec"],
    })
    annotations = anns_out[_ANN_COLUMNS].sort_values("ann_id").reset_index(drop=True)

    return tiles, annotations


def _ensure_built(force: bool = False) -> None:
    if force or not TILES_PARQUET.exists() or not ANNOTATIONS_PARQUET.exists():
        tiles, annotations = _build()
        RP_DIR.mkdir(parents=True, exist_ok=True)
        tiles.to_parquet(TILES_PARQUET, index=False)
        annotations.to_parquet(ANNOTATIONS_PARQUET, index=False)
        print(f"[vigil.data] built {len(tiles)} tiles, {len(annotations)} annotations "
              f"-> {TILES_PARQUET.name}, {ANNOTATIONS_PARQUET.name}")


def load_tiles(force: bool = False) -> pd.DataFrame:
    """One row per 512x512 tile. See module docstring / _TILE_COLUMNS."""
    _ensure_built(force=force)
    return pd.read_parquet(TILES_PARQUET)


def load_annotations(force: bool = False) -> pd.DataFrame:
    """One row per aircraft bounding box. See module docstring / _ANN_COLUMNS."""
    _ensure_built(force=force)
    return pd.read_parquet(ANNOTATIONS_PARQUET)


if __name__ == "__main__":
    t = load_tiles(force=True)
    a = load_annotations()
    print(t.head())
    print(a.head())
    print(f"tiles: {len(t)}, annotations: {len(a)}, unique parents: {t['parent_id'].nunique()}")
