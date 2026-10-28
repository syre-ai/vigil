#!/usr/bin/env python
"""Build data/chips/: a 224x224 PNG crop per aircraft annotation, plus
data/chips/index.parquet (ann_id, tile_id, domain, role_id, path).

For every annotation whose tile PNG exists on disk, crops the bbox with
15% padding on each side (clamped to the image bounds), resizes to
224x224, and writes data/chips/{ann_id}.png. Annotations whose tile PNG is
missing (the background download may still be running) are skipped and
counted -- safe to re-run as more images land.

Idempotent: chip filenames are keyed by the (stable) ann_id, so re-running
just overwrites them with the same content; index.parquet is rebuilt from
scratch each run to reflect exactly what's on disk right now.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from vigil.data import load_annotations, load_tiles  # noqa: E402
from vigil.domains import assign_domains  # noqa: E402

CHIPS_DIR = REPO_ROOT / "data" / "chips"
CHIP_SIZE = 224
PAD_FRAC = 0.15


def _crop_with_padding(img: Image.Image, x: float, y: float, w: float, h: float) -> Image.Image:
    pad_x = w * PAD_FRAC
    pad_y = h * PAD_FRAC
    left = max(0, x - pad_x)
    top = max(0, y - pad_y)
    right = min(img.width, x + w + pad_x)
    bottom = min(img.height, y + h + pad_y)
    return img.crop((left, top, right, bottom))


def main() -> None:
    tiles = assign_domains(load_tiles())
    anns = load_annotations()

    domain_by_tile = dict(zip(tiles["tile_id"], tiles["domain"]))
    path_by_tile = dict(zip(tiles["tile_id"], tiles["image_path"]))

    CHIPS_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    n_written = 0
    n_skipped_missing = 0
    n_skipped_other = 0
    open_tile_cache: dict[int, Image.Image | None] = {}

    for ann in anns.itertuples():
        tile_path = path_by_tile.get(ann.tile_id)
        if tile_path is None:
            n_skipped_other += 1
            continue

        if ann.tile_id not in open_tile_cache:
            if len(open_tile_cache) > 64:  # bounded cache: annotations are grouped by tile, so a small LRU is enough
                open_tile_cache.clear()
            p = Path(tile_path)
            if not p.exists():
                open_tile_cache[ann.tile_id] = None
            else:
                try:
                    open_tile_cache[ann.tile_id] = Image.open(p).convert("RGB")
                except Exception as e:  # corrupt/partial download in progress
                    print(f"  could not open {p}: {e}")
                    open_tile_cache[ann.tile_id] = None

        img = open_tile_cache[ann.tile_id]
        if img is None:
            n_skipped_missing += 1
            continue

        chip = _crop_with_padding(img, ann.x, ann.y, ann.w, ann.h)
        chip = chip.resize((CHIP_SIZE, CHIP_SIZE), Image.BILINEAR)
        out_path = CHIPS_DIR / f"{ann.ann_id}.png"
        chip.save(out_path)

        rows.append({
            "ann_id": ann.ann_id,
            "tile_id": ann.tile_id,
            "domain": domain_by_tile.get(ann.tile_id),
            "role_id": ann.role_id,
            "path": str(out_path),
        })
        n_written += 1

    index = pd.DataFrame(rows, columns=["ann_id", "tile_id", "domain", "role_id", "path"])
    index_path = CHIPS_DIR / "index.parquet"
    index.to_parquet(index_path, index=False)

    print(f"wrote {n_written} chips to {CHIPS_DIR}")
    print(f"skipped {n_skipped_missing} annotations (tile PNG not on disk yet)")
    if n_skipped_other:
        print(f"skipped {n_skipped_other} annotations (no tile path found -- shouldn't happen)")
    print(f"wrote {index_path} ({len(index)} rows)")


if __name__ == "__main__":
    main()
