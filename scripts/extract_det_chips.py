#!/usr/bin/env python
"""Build data/chips_det/: a 224x224 PNG crop per DETECTOR-output box (data/scores/detections.parquet's
kind="det" rows), plus data/chips_det/index.parquet (det_id, tile_id, parent_id, domain, conf,
x1,y1,x2,y2,w,h, matched_ann_id, path).

These are field mode's chips (research/08-field-mode-proposal.md): an analyst reviews what the
DETECTOR found, never a ground-truth annotation box -- unlike scripts/extract_chips.py's chips
(cropped from RarePlanes' own annotations, oracle localization). Same crop recipe as extract_chips.py
(15% padding on each side, clamped to the image bounds, resized to 224x224).

det_id is the STABLE row order of kind="det" rows within detections.parquet (that file's own row
order never changes once scripts/run_detector.py has produced it -- it isn't touched by anything
downstream), so chip filenames/index rows stay stable across reruns of this script.

`domain` and `conf` are read straight off detections.parquet (already computed by run_detector.py at
CONF=0.25 -- see its own module docstring: "all confidences >= 0.25, which is how it was produced" --
so no additional confidence filtering happens here). `matched_ann_id` is detections.parquet's own
`ann_id` column for kind="det" rows (null unless this detection was matched to a ground-truth box at
IoU>=0.5) -- BENCHMARK-ONLY: it exists so the benchmark view can check the detector's own precision,
and is never used to fit or score anything in field mode (see backend/vigil/pipeline.py).

Idempotent: chip filenames are keyed by the (stable) det_id, so re-running just overwrites them with
the same content; index.parquet is rebuilt from scratch each run to reflect exactly what's on disk
right now (same convention as extract_chips.py). Annotations/detections whose tile PNG is missing
(the background download may still be running) are skipped and counted -- safe to re-run as more
images land.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from vigil.data import load_tiles  # noqa: E402
from vigil.paths import CHIPS_DET_DIR, SCORES_DIR  # noqa: E402  -- honors VIGIL_DATA_ROOT/VIGIL_SCORES_DIR

CHIP_SIZE = 224
PAD_FRAC = 0.15


def _crop_with_padding(img: Image.Image, x1: float, y1: float, x2: float, y2: float) -> Image.Image:
    w = x2 - x1
    h = y2 - y1
    pad_x = w * PAD_FRAC
    pad_y = h * PAD_FRAC
    left = max(0, x1 - pad_x)
    top = max(0, y1 - pad_y)
    right = min(img.width, x2 + pad_x)
    bottom = min(img.height, y2 + pad_y)
    return img.crop((left, top, right, bottom))


def main() -> None:
    det_path = SCORES_DIR / "detections.parquet"
    if not det_path.exists():
        print(f"no {det_path} -- run scripts/run_detector.py first", file=sys.stderr)
        sys.exit(1)

    dets = pd.read_parquet(det_path)
    det_rows = dets[dets["kind"] == "det"].reset_index(drop=True)
    det_rows = det_rows.assign(det_id=det_rows.index.astype("int64"))

    tiles = load_tiles()[["tile_id", "image_path", "parent_id"]]
    path_by_tile = dict(zip(tiles["tile_id"], tiles["image_path"]))
    parent_by_tile = dict(zip(tiles["tile_id"], tiles["parent_id"]))

    CHIPS_DET_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    n_written = 0
    n_skipped_missing = 0
    open_tile_cache: dict[int, Image.Image | None] = {}

    for det in det_rows.itertuples():
        tile_id = int(det.tile_id)
        tile_path = path_by_tile.get(tile_id)
        if tile_path is None:
            n_skipped_missing += 1
            continue

        if tile_id not in open_tile_cache:
            if len(open_tile_cache) > 64:  # bounded cache: dets are grouped by tile, so a small LRU is enough
                open_tile_cache.clear()
            p = Path(tile_path)
            if not p.exists():
                open_tile_cache[tile_id] = None
            else:
                try:
                    open_tile_cache[tile_id] = Image.open(p).convert("RGB")
                except Exception as e:  # corrupt/partial download in progress
                    print(f"  could not open {p}: {e}")
                    open_tile_cache[tile_id] = None

        img = open_tile_cache[tile_id]
        if img is None:
            n_skipped_missing += 1
            continue

        chip = _crop_with_padding(img, float(det.x1), float(det.y1), float(det.x2), float(det.y2))
        chip = chip.resize((CHIP_SIZE, CHIP_SIZE), Image.BILINEAR)
        out_path = CHIPS_DET_DIR / f"{int(det.det_id)}.png"
        chip.save(out_path)

        matched_ann_id = None if pd.isna(det.ann_id) else int(det.ann_id)
        rows.append({
            "det_id": int(det.det_id),
            "tile_id": tile_id,
            "parent_id": parent_by_tile.get(tile_id),
            "domain": det.domain,
            "conf": float(det.conf),
            "x1": float(det.x1), "y1": float(det.y1), "x2": float(det.x2), "y2": float(det.y2),
            "w": float(det.x2) - float(det.x1), "h": float(det.y2) - float(det.y1),
            "matched_ann_id": matched_ann_id,
            "path": str(out_path),
        })
        n_written += 1

    index = pd.DataFrame(rows, columns=[
        "det_id", "tile_id", "parent_id", "domain", "conf",
        "x1", "y1", "x2", "y2", "w", "h", "matched_ann_id", "path",
    ])
    if len(index):
        index["det_id"] = index["det_id"].astype("int64")
        index["matched_ann_id"] = index["matched_ann_id"].astype("Int64")  # nullable int -- benchmark-only
    index_path = CHIPS_DET_DIR / "index.parquet"
    index.to_parquet(index_path, index=False)

    print(f"wrote {n_written} det chips to {CHIPS_DET_DIR}")
    print(f"skipped {n_skipped_missing} detections (tile PNG not on disk yet)")
    print(f"wrote {index_path} ({len(index)} rows)")


if __name__ == "__main__":
    main()
