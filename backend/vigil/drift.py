"""Scene-level embedding drift: mean 5-NN cosine distance of a batch's whole-tile DINOv2 embeddings to
the reference tile set, with a parent-image block-bootstrap null.

Chip-level embedding drift (the original design) does not separate haze/off-nadir/foreign from the
control batch -- validated experimentally, chip-level drift put the control batch ABOVE its own null
p95. Whole-TILE embeddings (one DINOv2 embedding per 512x512 tile, see vigil.embed.embed_tiles) carry
scene-level context (illumination, texture, whole-image statistics) that chip crops don't, and DO
separate control from haze/snow/military (validated: control p=0.11, haze p=0.002, snow p<0.001,
military p<0.001; off_nadir/foreign genuinely don't shift scene features, p=0.08/0.14).

The null is a PARENT-IMAGE block bootstrap over OUT-OF-FOLD reference-tile distances (8-fold
cross-fitting by parent image, see `cross_fit_reference_tile_distances`): each reference tile's
"distance to reference" is computed with its own parent image's fold held out, so the null isn't
contaminated by a tile matching itself (or a spatial neighbour from the same parent) in the reference
index. The bootstrap then resamples reference PARENT IMAGES with replacement and appends every one of
that parent's out-of-fold tile distances, until the running total reaches (at least) the target sample
size n; whichever parent's block would push the running total past n has its OWN tiles kept by random
subsampling WITHOUT replacement (`rng.choice`), not by positionally keeping its first tiles and
dropping the rest, so every tile in that final block has an equal chance of being the one truncated
away -- tiles from the same parent image are still correlated, so this makes the p-value indicative
rather than a rigorous inferential test (see `NULL_METHOD`).

The p-value itself is a finite-simulation estimate, `(exceedances + 1) / (n_boot + 1)` rather than the
naive `mean(null >= score)`: the naive form can report exactly 0 (impossible/misleadingly precise for a
finite `n_boot`-draw simulation), whereas the +1/+1 (Davison & Hinkley 1997) form is a proper, strictly
positive estimate consistent with `n_boot` finite bootstrap draws (see `P_VALUE_METHOD`).

The per-chip `knn_dist` column (chip-level distance to reference CHIPS) is unaffected by this module --
it remains the novelty baseline and the source of true embedding-space neighbors (see vigil.novelty).
"""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors

N_BOOT = 500
K_FOLDS = 8
KNN_K = 5
STATISTIC_NAME = (
    "mean 5-NN cosine distance of whole-tile DINOv2 embeddings to reference tiles (scene level)"
)
NULL_METHOD = (
    "parent-image block bootstrap over out-of-fold reference tiles (8-fold by parent image); "
    "the final truncated parent block's tiles are kept by random subsampling without replacement "
    "(rng.choice), not positional truncation; p-value indicative"
)
P_VALUE_METHOD = (
    "finite-simulation estimate: (exceedances + 1) / (n_boot + 1) (Davison & Hinkley 1997), "
    "never exactly 0 for a finite n_boot"
)


def cross_fit_reference_tile_distances(
    ref_tile_X: np.ndarray,
    ref_tile_parent_ids: np.ndarray,
    k_folds: int = K_FOLDS,
    seed: int = 0,
    k: int = KNN_K,
) -> np.ndarray:
    """Out-of-fold mean k-NN cosine distance for every reference tile.

    Parent-image k-fold cross-fitting (folds = unique parent ids, shuffled with `seed`): for each
    fold, fits a NearestNeighbors index on the OTHER folds' reference tiles and scores the held-out
    fold's tiles against it, so no reference tile's distance is computed against an index that
    contains it (or another tile from the same parent image). Returns one distance per row of
    ref_tile_X, in the same order.
    """
    parents = np.unique(ref_tile_parent_ids)
    rng = np.random.default_rng(seed)
    shuffled = parents.copy()
    rng.shuffle(shuffled)
    folds = np.array_split(shuffled, k_folds)

    oof = np.zeros(len(ref_tile_X))
    for fold_parents in folds:
        held_out = np.isin(ref_tile_parent_ids, fold_parents)
        if held_out.sum() == 0 or (~held_out).sum() == 0:
            continue
        nn = NearestNeighbors(n_neighbors=k, metric="cosine").fit(ref_tile_X[~held_out])
        d, _ = nn.kneighbors(ref_tile_X[held_out])
        oof[held_out] = d.mean(axis=1)
    return oof


def bootstrap_null(
    oof_ref: np.ndarray, ref_parent_ids: np.ndarray, n: int, n_boot: int = N_BOOT, seed: int = 0
) -> np.ndarray:
    """Draws n_boot bootstrap means of exactly n reference tiles by resampling PARENT IMAGES with
    replacement (not tiles): each draw repeatedly samples one parent image at a time and appends
    all of that parent's out-of-fold tile distances, until the accumulated count reaches n. Whichever
    parent's block would push the running total past n has only `n - total_so_far` of its own tiles
    kept -- chosen by `rng.choice` WITHOUT replacement, not by positionally keeping its first tiles
    and dropping the rest, so truncation doesn't systematically favor tiles by their position within
    that parent's array (see module docstring / NULL_METHOD).
    """
    rng = np.random.default_rng(seed)
    parents = np.unique(ref_parent_ids)
    parent_vals = {p: oof_ref[ref_parent_ids == p] for p in parents}
    n_parents = len(parents)
    means = np.empty(n_boot)
    for b in range(n_boot):
        chunks = []
        total = 0
        while total < n:
            p = parents[rng.integers(0, n_parents)]
            v = parent_vals[p]
            remaining = n - total
            if len(v) > remaining:
                v = rng.choice(v, size=remaining, replace=False)
            chunks.append(v)
            total += len(v)
        sample = np.concatenate(chunks)
        means[b] = sample.mean()
    return means


def batch_scene_drift(
    batch_tile_X: np.ndarray,
    ref_tile_X: np.ndarray,
    oof_ref: np.ndarray,
    ref_parent_ids: np.ndarray,
    n_boot: int = N_BOOT,
    seed: int = 0,
    k: int = KNN_K,
) -> dict:
    """Computes a batch's scene-level drift score (mean k-NN cosine distance of the batch's whole-tile
    embeddings to the FULL reference tile set) and its parent-image block-bootstrap null (drawn from
    out-of-fold reference-tile distances, resampled to the batch's OWN tile count -- not capped by the
    reference set's size, so a batch bigger than the reference tile pool still gets a null whose
    variance matches its own sample size), plus a plain descriptive ratio to the reference's own
    out-of-fold mean."""
    n = max(1, len(batch_tile_X))
    nn = NearestNeighbors(n_neighbors=k, metric="cosine").fit(ref_tile_X)
    d, _ = nn.kneighbors(batch_tile_X)
    score = float(d.mean(axis=1).mean())
    null = bootstrap_null(oof_ref, ref_parent_ids, n=n, n_boot=n_boot, seed=seed)
    null_p95 = float(np.percentile(null, 95))
    exceedances = int(np.sum(null >= score))
    p_value = float((exceedances + 1) / (n_boot + 1))  # finite-simulation estimate, see P_VALUE_METHOD
    calib_mean = float(np.mean(oof_ref)) if len(oof_ref) else float("nan")
    ratio_to_calib = float(score / calib_mean) if calib_mean and not np.isnan(calib_mean) else float("nan")
    return {
        "score": score,
        "null_p95": null_p95,
        "p_value": p_value,
        "p_value_method": P_VALUE_METHOD,
        "n_boot": n_boot,
        "n": int(n),  # tiles
        "statistic": STATISTIC_NAME,
        "null_method": NULL_METHOD,
        "calib_mean": calib_mean,
        "ratio_to_calib": ratio_to_calib,
        "level": "scene",
        "interpretation": "descriptive",  # p-value is indicative, not an inferential/significance test
    }
