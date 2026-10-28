#!/usr/bin/env python
"""Build data/yolo/: a YOLO-format single-class 'aircraft' detection dataset.

train = domain 'ref_train' (detector fitting)
val   = domain 'ref_calib' (calibration split, held out of training)

Images are symlinked (not copied) from the RarePlanes tile PNGs. Tiles
whose PNG isn't on disk yet (the background download may still be running)
are skipped -- this script is safe to re-run once more images land.

Idempotent: data/yolo/ is wiped and rebuilt from scratch on every run, so
output never depends on what was there before.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from vigil.data import load_tiles  # noqa: E402
from vigil.domains import assign_domains  # noqa: E402

YOLO_DIR = REPO_ROOT / "data" / "yolo"
TILE_SIZE = 512  # all RarePlanes tiles are 512x512

SPLIT_DOMAINS = {"train": "ref_train", "val": "ref_calib"}


def _write_yaml(path: Path) -> None:
    path.write_text(
        f"path: {YOLO_DIR}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n"
        f"  0: aircraft\n"
    )


def main() -> None:
    tiles = assign_domains(load_tiles())
    annotations_by_tile = _load_annotations_by_tile()

    if YOLO_DIR.exists():
        shutil.rmtree(YOLO_DIR)

    totals = {}
    for split, domain in SPLIT_DOMAINS.items():
        img_dir = YOLO_DIR / "images" / split
        lbl_dir = YOLO_DIR / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        split_tiles = tiles[tiles["domain"] == domain]
        n_written = 0
        n_skipped_missing = 0

        for row in split_tiles.itertuples():
            src = Path(row.image_path)
            if not src.exists():
                n_skipped_missing += 1
                continue

            dst = img_dir / row.file_name
            if not dst.exists():
                dst.symlink_to(src)

            stem = Path(row.file_name).stem
            lbl_path = lbl_dir / f"{stem}.txt"
            lines = []
            for ann in annotations_by_tile.get(row.tile_id, []):
                x, y, w, h = ann
                cx = (x + w / 2.0) / TILE_SIZE
                cy = (y + h / 2.0) / TILE_SIZE
                nw = w / TILE_SIZE
                nh = h / TILE_SIZE
                lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            lbl_path.write_text("\n".join(lines) + ("\n" if lines else ""))
            n_written += 1

        totals[split] = (n_written, n_skipped_missing, len(split_tiles))
        print(f"[{split}] domain={domain}: wrote {n_written} images "
              f"(skipped {n_skipped_missing} missing PNGs, of {len(split_tiles)} tiles in domain)")

    _write_yaml(YOLO_DIR / "aircraft.yaml")
    print(f"wrote {YOLO_DIR / 'aircraft.yaml'}")

    for split, (written, skipped, total) in totals.items():
        if total == 0:
            print(f"WARNING: domain {SPLIT_DOMAINS[split]!r} ({split}) has zero tiles")
        elif written == 0:
            print(f"WARNING: {split} has 0 images on disk yet -- download likely still running")


def _load_annotations_by_tile() -> dict[int, list[tuple[float, float, float, float]]]:
    from vigil.data import load_annotations
    anns = load_annotations()
    out: dict[int, list[tuple[float, float, float, float]]] = {}
    for row in anns.itertuples():
        out.setdefault(row.tile_id, []).append((row.x, row.y, row.w, row.h))
    return out


if __name__ == "__main__":
    main()
