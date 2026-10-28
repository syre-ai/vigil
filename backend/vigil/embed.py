"""DINOv2-small chip embeddings, cached to data/emb/embeddings.npz.

Each chip's embedding is the CLS token concatenated with the mean patch
token (384 + 384 = 768 dims), L2-normalized. Only ann_ids missing from the
cache are computed on each call, so this is safe to re-run as more chips
land from the (partial) background download.

Also provides `embed_tiles()`: the same embedding recipe applied to whole
512x512 tiles (not chips), cached to data/emb/tile_embeddings.npz, for
scene-level drift (vigil.drift) -- one row per tile in tiles.parquet.

Also provides `embed_det_chips()` (field mode): the same recipe applied to detector-output chips
(scripts/extract_det_chips.py's data/chips_det/), cached to data/emb/det_embeddings.npz, keyed by
det_id instead of ann_id.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))  # so `import vigil.data` works when run as -m backend.vigil.embed

from vigil.paths import CHIPS_DET_DIR, CHIPS_DIR, EMB_DIR  # noqa: E402  -- honors VIGIL_DATA_ROOT, same as backend/app/config.py

CHIP_INDEX_PATH = CHIPS_DIR / "index.parquet"
EMB_PATH = EMB_DIR / "embeddings.npz"
TILE_EMB_PATH = EMB_DIR / "tile_embeddings.npz"
DET_CHIP_INDEX_PATH = CHIPS_DET_DIR / "index.parquet"
DET_EMB_PATH = EMB_DIR / "det_embeddings.npz"

MODEL_NAME = "facebook/dinov2-small"
EMBED_DIM = 768  # CLS (384) ++ mean patch (384)
BATCH_SIZE = 64

_model = None
_processor = None


def load_chip_index() -> pd.DataFrame:
    """Reads the chip index (ann_id, tile_id, domain, role_id, path) written by extract_chips.py."""
    return pd.read_parquet(CHIP_INDEX_PATH)


def load_cache() -> tuple[np.ndarray, np.ndarray]:
    """Loads cached (ann_id, X) arrays from data/emb/embeddings.npz, or empty arrays if no cache exists yet."""
    if EMB_PATH.exists():
        with np.load(EMB_PATH) as data:
            return data["ann_id"].astype(np.int64), data["X"].astype(np.float32)
    return np.zeros(0, dtype=np.int64), np.zeros((0, EMBED_DIM), dtype=np.float32)


def save_cache(ann_id: np.ndarray, X: np.ndarray) -> None:
    """Atomically writes (ann_id, X) arrays to data/emb/embeddings.npz."""
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    tmp = EMB_DIR / "embeddings.tmp.npz"  # np.savez appends .npz if the name lacks it, so keep it here
    np.savez(tmp, ann_id=ann_id.astype(np.int64), X=X.astype(np.float32))
    tmp.replace(EMB_PATH)


def _get_model():
    """Lazily loads and caches the DINOv2-small model + image processor, moving the model to GPU if available."""
    global _model, _processor
    if _model is None:
        _processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
        _model = AutoModel.from_pretrained(MODEL_NAME)
        _model.eval()
        if torch.cuda.is_available():
            _model = _model.to("cuda")
    return _model, _processor


@torch.no_grad()
def embed_paths(paths: list[str], batch_size: int = BATCH_SIZE) -> np.ndarray:
    """Computes L2-normalized 768-d (CLS ++ mean patch token) DINOv2-small embeddings for a list of image paths."""
    model, processor = _get_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = np.zeros((len(paths), EMBED_DIM), dtype=np.float32)
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start:start + batch_size]
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        inputs = processor(images=images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
            outputs = model(**inputs)
        hidden = outputs.last_hidden_state.float()  # (B, 1+P, 384)
        cls = hidden[:, 0, :]
        patch_mean = hidden[:, 1:, :].mean(dim=1)
        feat = torch.cat([cls, patch_mean], dim=1)  # (B, 768)
        feat = torch.nn.functional.normalize(feat, dim=1)
        out[start:start + len(batch_paths)] = feat.cpu().numpy()
    return out


def compute_missing(index: pd.DataFrame | None = None) -> dict:
    """Embeds every ann_id in the chip index that is missing from the cache, merges it in, and saves; returns stats.

    Chips whose PNG file doesn't exist yet (chip extraction/download still partial) are skipped, not
    raised on -- they stay missing from the cache and are picked up on a later call once their file
    lands. `n_skipped_missing` in the returned stats counts them."""
    if index is None:
        index = load_chip_index()
    cached_ids, cached_X = load_cache()
    cached_set = set(cached_ids.tolist())
    missing = index[~index["ann_id"].isin(cached_set)].drop_duplicates("ann_id")

    n_missing = len(missing)
    missing = missing[[Path(p).exists() for p in missing["path"]]]
    n_skipped = n_missing - len(missing)
    if n_skipped:
        print(f"embed.compute_missing: skipping {n_skipped} chip(s) with missing PNG file(s)", file=sys.stderr)

    t0 = time.time()
    if len(missing) == 0:
        return {"n_total": int(len(cached_ids)), "n_new": 0, "seconds": 0.0, "n_skipped_missing": n_skipped}

    new_X = embed_paths(missing["path"].tolist())
    new_ids = missing["ann_id"].to_numpy(dtype=np.int64)
    all_ids = np.concatenate([cached_ids, new_ids])
    all_X = np.concatenate([cached_X, new_X], axis=0) if len(cached_X) else new_X
    save_cache(all_ids, all_X)
    return {
        "n_total": int(len(all_ids)), "n_new": int(len(new_ids)), "seconds": time.time() - t0,
        "n_skipped_missing": n_skipped,
    }


def get_embeddings(ann_ids: np.ndarray) -> np.ndarray:
    """Looks up cached embeddings for the given ann_ids, in the same order (raises KeyError if any are missing)."""
    cached_ids, cached_X = load_cache()
    pos = {int(a): i for i, a in enumerate(cached_ids.tolist())}
    idx = np.array([pos[int(a)] for a in ann_ids], dtype=np.int64)
    return cached_X[idx]


# ---------------------------------------------------------------------- tile (scene-level) embeddings

def load_tile_cache() -> tuple[np.ndarray, np.ndarray]:
    """Loads cached (tile_id, X) arrays from data/emb/tile_embeddings.npz, or empty arrays if none exist yet."""
    if TILE_EMB_PATH.exists():
        with np.load(TILE_EMB_PATH) as data:
            return data["tile_id"].astype(np.int64), data["X"].astype(np.float32)
    return np.zeros(0, dtype=np.int64), np.zeros((0, EMBED_DIM), dtype=np.float32)


def save_tile_cache(tile_id: np.ndarray, X: np.ndarray) -> None:
    """Atomically writes (tile_id, X) arrays to data/emb/tile_embeddings.npz."""
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    tmp = EMB_DIR / "tile_embeddings.tmp.npz"
    np.savez(tmp, tile_id=tile_id.astype(np.int64), X=X.astype(np.float32))
    tmp.replace(TILE_EMB_PATH)


def embed_tiles(tiles: pd.DataFrame | None = None) -> dict:
    """Embeds every tile in tiles.parquet that is missing from data/emb/tile_embeddings.npz (whole-tile
    DINOv2 embedding, same recipe as embed_paths -- CLS ++ mean patch token, L2-normalized), merges it
    in, and saves. Reuses the existing cache untouched if every tile_id is already present.

    Tiles whose PNG file doesn't exist yet (the background tile download is partial) are skipped, not
    raised on -- they stay missing from the cache and are picked up on a later call once their file
    lands. `n_skipped_missing` in the returned stats counts them."""
    if tiles is None:
        from vigil.data import load_tiles
        tiles = load_tiles()
    cached_ids, cached_X = load_tile_cache()
    cached_set = set(cached_ids.tolist())
    missing = tiles[~tiles["tile_id"].isin(cached_set)].drop_duplicates("tile_id")

    n_missing = len(missing)
    missing = missing[[Path(p).exists() for p in missing["image_path"]]]
    n_skipped = n_missing - len(missing)
    if n_skipped:
        print(f"embed.embed_tiles: skipping {n_skipped} tile(s) with missing PNG file(s)", file=sys.stderr)

    t0 = time.time()
    if len(missing) == 0:
        return {"n_total": int(len(cached_ids)), "n_new": 0, "seconds": 0.0, "n_skipped_missing": n_skipped}

    new_X = embed_paths(missing["image_path"].tolist())
    new_ids = missing["tile_id"].to_numpy(dtype=np.int64)
    all_ids = np.concatenate([cached_ids, new_ids])
    all_X = np.concatenate([cached_X, new_X], axis=0) if len(cached_X) else new_X
    save_tile_cache(all_ids, all_X)
    return {
        "n_total": int(len(all_ids)), "n_new": int(len(new_ids)), "seconds": time.time() - t0,
        "n_skipped_missing": n_skipped,
    }


def get_tile_embeddings(tile_ids: np.ndarray) -> np.ndarray:
    """Looks up cached tile embeddings for the given tile_ids, in the same order (raises KeyError if any are missing)."""
    cached_ids, cached_X = load_tile_cache()
    pos = {int(t): i for i, t in enumerate(cached_ids.tolist())}
    idx = np.array([pos[int(t)] for t in tile_ids], dtype=np.int64)
    return cached_X[idx]


def load_det_chip_index() -> pd.DataFrame:
    """Reads the detector-chip index (det_id, tile_id, parent_id, domain, conf, x1,y1,x2,y2,w,h,
    matched_ann_id, path) written by scripts/extract_det_chips.py."""
    return pd.read_parquet(DET_CHIP_INDEX_PATH)


def load_det_cache() -> tuple[np.ndarray, np.ndarray]:
    """Loads cached (det_id, X) arrays from data/emb/det_embeddings.npz, or empty arrays if none exist yet."""
    if DET_EMB_PATH.exists():
        with np.load(DET_EMB_PATH) as data:
            return data["det_id"].astype(np.int64), data["X"].astype(np.float32)
    return np.zeros(0, dtype=np.int64), np.zeros((0, EMBED_DIM), dtype=np.float32)


def save_det_cache(det_id: np.ndarray, X: np.ndarray) -> None:
    """Atomically writes (det_id, X) arrays to data/emb/det_embeddings.npz."""
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    tmp = EMB_DIR / "det_embeddings.tmp.npz"
    np.savez(tmp, det_id=det_id.astype(np.int64), X=X.astype(np.float32))
    tmp.replace(DET_EMB_PATH)


def embed_det_chips(index: pd.DataFrame | None = None) -> dict:
    """Embeds every det_id in the detector-chip index (scripts/extract_det_chips.py's
    data/chips_det/index.parquet) missing from data/emb/det_embeddings.npz, using the SAME DINOv2
    recipe as embed_paths (CLS ++ mean patch token, L2-normalized) -- these embeddings are only ever
    TRANSFORMED through the already-fitted PCA/ART/prototype/UMAP/kNN models (vigil.pipeline), never
    used to fit anything.

    Chips whose PNG file doesn't exist yet (extract_det_chips.py still catching up) are skipped, not
    raised on -- `n_skipped_missing` counts them, same convention as compute_missing/embed_tiles."""
    if index is None:
        index = load_det_chip_index()
    cached_ids, cached_X = load_det_cache()
    cached_set = set(cached_ids.tolist())
    missing = index[~index["det_id"].isin(cached_set)].drop_duplicates("det_id")

    n_missing = len(missing)
    missing = missing[[Path(p).exists() for p in missing["path"]]]
    n_skipped = n_missing - len(missing)
    if n_skipped:
        print(f"embed.embed_det_chips: skipping {n_skipped} det chip(s) with missing PNG file(s)", file=sys.stderr)

    t0 = time.time()
    if len(missing) == 0:
        return {"n_total": int(len(cached_ids)), "n_new": 0, "seconds": 0.0, "n_skipped_missing": n_skipped}

    new_X = embed_paths(missing["path"].tolist())
    new_ids = missing["det_id"].to_numpy(dtype=np.int64)
    all_ids = np.concatenate([cached_ids, new_ids])
    all_X = np.concatenate([cached_X, new_X], axis=0) if len(cached_X) else new_X
    save_det_cache(all_ids, all_X)
    return {
        "n_total": int(len(all_ids)), "n_new": int(len(new_ids)), "seconds": time.time() - t0,
        "n_skipped_missing": n_skipped,
    }


def get_det_embeddings(det_ids: np.ndarray) -> np.ndarray:
    """Looks up cached detector-chip embeddings for the given det_ids, in the same order (raises
    KeyError if any are missing)."""
    cached_ids, cached_X = load_det_cache()
    pos = {int(a): i for i, a in enumerate(cached_ids.tolist())}
    idx = np.array([pos[int(a)] for a in det_ids], dtype=np.int64)
    return cached_X[idx]


def main() -> None:
    """CLI entry point: embeds every chip in the index missing from the cache and prints a summary.
    Pass 'tiles' as argv[1] to embed whole tiles (data/emb/tile_embeddings.npz), or '--det' to embed
    detector chips (data/emb/det_embeddings.npz, field mode) instead."""
    if len(sys.argv) > 1 and sys.argv[1] == "tiles":
        stats = embed_tiles()
        rate = stats["n_new"] / stats["seconds"] if stats["seconds"] > 0 else 0.0
        print(
            f"tile embeddings cache: {stats['n_total']} total, {stats['n_new']} newly computed "
            f"in {stats['seconds']:.1f}s ({rate:.0f} tiles/s), {stats.get('n_skipped_missing', 0)} skipped (missing file)"
        )
        return
    if len(sys.argv) > 1 and sys.argv[1] == "--det":
        stats = embed_det_chips()
        rate = stats["n_new"] / stats["seconds"] if stats["seconds"] > 0 else 0.0
        print(
            f"det-chip embeddings cache: {stats['n_total']} total, {stats['n_new']} newly computed "
            f"in {stats['seconds']:.1f}s ({rate:.0f} chips/s), {stats.get('n_skipped_missing', 0)} skipped (missing file)"
        )
        return
    stats = compute_missing()
    rate = stats["n_new"] / stats["seconds"] if stats["seconds"] > 0 else 0.0
    print(
        f"embeddings cache: {stats['n_total']} total, {stats['n_new']} newly computed "
        f"in {stats['seconds']:.1f}s ({rate:.0f} chips/s), {stats.get('n_skipped_missing', 0)} skipped (missing file)"
    )


if __name__ == "__main__":
    sys.exit(main())
