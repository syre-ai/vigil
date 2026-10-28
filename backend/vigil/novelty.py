"""Fuzzy ART novelty detection (NOTE: add_category / the adapted-category choice gate are retained but UNUSED by the pipeline as of 2026-09-05; analyst classes are handled in vigil.pipeline per ADAPTATION_RULE) over PCA-reduced, complement-coded embeddings, plus an honest kNN-distance baseline.

Preprocessing: PCA(embeddings) -> 256 dims (32 was at chance for rare-target AUROC; see STATE.md) (fit on
the reference set, ref_train ∪ ref_calib) -> min-max scale to [0,1] using the reference set's per-dim
range (clipped) -> complement code to 64 dims ([x, 1-x]). Standard Fuzzy ART then runs on those 64-d
vectors.

Vigilance (rho*) is calibrated by PARENT-IMAGE 8-FOLD CROSS-FITTING over the whole reference set (see
`cross_fit_vigilance`): calibrating on the small ref_calib split alone was unstable (control-batch
novelty rate 15-18% vs a 5% target, validated experimentally) -- cross-fitting over all 43 reference
parent images/3,942 chips gives a stable rho* (~0.588) and a ~4.5% control novelty rate.

adaptation_rule: an analyst-created category (added via `add_category`, after analyst feedback) can
claim a chip only if it passes the vigilance test (the winning category's raw match >= rho*, i.e. the
chip is not novel) AND its Fuzzy-ART CHOICE value exceeds the BEST choice value among trained
(fit_art) categories -- max(choice[:, trained_idx]), not the choice value of whichever trained
category happens to win the raw-match argmax. Otherwise the chip is scored against trained categories
only (plain raw-match argmax, as if no analyst category existed). See `_match_all` for why comparing
against the match-argmax trained category's choice value (instead of the best trained choice value)
was wrong: it let an analyst category win merely because it beat a possibly weak best-match trained
category, rather than beating the strongest true competitor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

PCA_DIMS = 256
ALPHA = 0.001       # choice parameter
BETA = 1.0          # learning rate (1.0 = fast learning)
CAP = 300           # soft cap on the number of ART categories
RHO_TRAIN_START = 0.9
RHO_TRAIN_MIN = 0.3
RHO_TRAIN_STEP = 0.05
KNN_K = 5
CROSS_FIT_K_FOLDS = 8
CROSS_FIT_SEED = 0


def complement_code(Xs: np.ndarray) -> np.ndarray:
    """Complement-codes a [0,1]-scaled (N, M) matrix into (N, 2M) as [x, 1-x]."""
    return np.concatenate([Xs, 1.0 - Xs], axis=1)


class NoveltyModel:
    """PCA -> min-max -> complement-coded Fuzzy ART novelty detector, vigilance calibrated on ref_calib."""

    def __init__(self, alpha: float = ALPHA, beta: float = BETA, pca_dims: int = PCA_DIMS, cap: int = CAP):
        """Initializes an untrained novelty model with the given Fuzzy ART hyperparameters."""
        self.alpha = alpha
        self.beta = beta
        self.pca_dims = pca_dims
        self.cap = cap
        self.pca: PCA | None = None
        self.lo: np.ndarray | None = None
        self.hi: np.ndarray | None = None
        self.categories: list[np.ndarray] = []
        # Parallel to self.categories: True for a category created by add_category (analyst feedback),
        # False for one created during fit_art's one-epoch training. See _match_all for why this
        # distinction matters: an add_category category is a raw, never-shrunk single-point fuzzy set
        # (w_sum == 2*pca_dims, by construction of complement coding), categorically different in scale
        # from a fit_art-trained category (empirically w_sum shrinks to a much smaller, tightly-clustered
        # equilibrium after even one or two min-update absorptions) -- so it must not compete for the
        # RAW-match argmax on equal footing, or it wins against almost every query regardless of true
        # similarity (validated: it swamped 100% of a 25k-chip dataset in one adaptation-exercise test).
        self._adapted: list[bool] = []
        # Parallel to self.categories/self._adapted: the analyst-supplied label for an add_category
        # category, None for a fit_art-trained category (or an add_category call with no label). Lets
        # callers (vigil.pipeline) map a winning ART category index straight back to the analyst class
        # name it was created for, without keeping a second lookup table in sync themselves.
        self.category_labels: list[str | None] = []
        self.rho_train: float | None = None
        self.rho_star: float | None = None
        self._nn: NearestNeighbors | None = None

    # ---------------------------------------------------------------- preprocessing
    def fit_preprocessing(self, X_ref_train: np.ndarray) -> None:
        """Fits the PCA (to pca_dims) and the min-max range used for scaling, both on ref_train embeddings."""
        n_components = min(self.pca_dims, X_ref_train.shape[0], X_ref_train.shape[1])
        self.pca = PCA(n_components=n_components, random_state=0)
        Z = self.pca.fit_transform(X_ref_train)
        self.lo = Z.min(axis=0)
        self.hi = Z.max(axis=0)

    def preprocess(self, X: np.ndarray) -> np.ndarray:
        """Transforms raw embeddings through the fitted PCA, clipped min-max scaling, then complement coding."""
        Z = self.pca.transform(X)
        span = np.where(self.hi - self.lo < 1e-8, 1.0, self.hi - self.lo)
        Zs = np.clip((Z - self.lo) / span, 0.0, 1.0)
        return complement_code(Zs)

    # ---------------------------------------------------------------- Fuzzy ART core
    def _present(self, I: np.ndarray, rho: float, learn: bool) -> int | None:
        """Presents one complement-coded vector at vigilance rho; on learn=True creates/updates the winner category."""
        if self.categories:
            W = np.stack(self.categories, axis=0)
            AND = np.minimum(I[None, :], W)
            size_and = AND.sum(axis=1)
            match = size_and / (I.sum() + 1e-12)
            choice = size_and / (self.alpha + W.sum(axis=1))
            for j in np.argsort(-choice):
                if match[j] >= rho:
                    if learn:
                        self.categories[j] = self.beta * np.minimum(I, self.categories[j]) + (1 - self.beta) * self.categories[j]
                    return int(j)
        if learn and len(self.categories) < self.cap:
            self.categories.append(I.copy())
            self._adapted.append(False)
            self.category_labels.append(None)
            return len(self.categories) - 1
        return None

    def _train_epoch(self, I: np.ndarray, rho: float, seed: int) -> int:
        """Resets categories and runs one presentation epoch (fixed shuffled order) at vigilance rho."""
        self.categories = []
        self._adapted = []
        self.category_labels = []
        order = np.random.default_rng(seed).permutation(len(I))
        for i in order:
            self._present(I[i], rho, learn=True)
            if len(self.categories) >= self.cap:
                break
        return len(self.categories)

    def fit_art(
        self,
        X_ref_train_civil: np.ndarray,
        seed: int = 0,
        rho_start: float = RHO_TRAIN_START,
        rho_min: float = RHO_TRAIN_MIN,
        rho_step: float = RHO_TRAIN_STEP,
    ) -> dict:
        """Trains Fuzzy ART for one epoch on ref_train civil chips, backing off rho_train if categories exceed cap."""
        I = self.preprocess(X_ref_train_civil)
        rho = rho_start
        n_cat = self._train_epoch(I, rho, seed)
        exploded = n_cat >= self.cap
        while n_cat >= self.cap and rho - rho_step >= rho_min:
            rho -= rho_step
            n_cat = self._train_epoch(I, rho, seed)
        self.rho_train = rho
        return {"n_categories": n_cat, "rho_train": rho, "n_presented": len(I), "hit_cap": exploded and n_cat >= self.cap}

    def _match_all(self, X: np.ndarray, chunk: int = 256) -> tuple[np.ndarray, np.ndarray]:
        """For every row of X, finds the best-matching category's match ratio and index (vectorized).

        Among fit_art-TRAINED categories this is the plain raw-match argmax (unchanged from the
        pre-adaptation design; this is what the validated cross-fit calibration experiment used).

        An add_category (analyst-feedback) category is never allowed to compete for that argmax on
        equal footing: it is a raw, never-shrunk single-point fuzzy set (w_sum == 2*pca_dims exactly,
        by construction of complement coding), whereas a fit_art-trained category's w_sum empirically
        shrinks to a much smaller, tightly-clustered equilibrium after even one or two min-update
        absorptions (measured: ~60% of a trained category's typical w_sum). Because raw match is NOT
        normalized by category size, an unshrunk category's inherent "generality" lets it spuriously
        out-match every trained category for almost ANY query, regardless of true similarity --
        validated: one single add_category call made 2 new categories win the argmax for 100% of a
        25k-chip dataset, collapsing novelty detection entirely. Instead, an add_category category may
        only win if its Fuzzy-ART CHOICE value (size_and/(alpha+w_sum), the same size-normalized rule
        `_present` uses to rank candidates during training) beats the BEST Fuzzy-ART choice value among
        ALL trained categories -- max(choice[:, trained_idx]) -- not merely the choice value of
        whichever trained category happens to win the raw-match argmax (that trained match-argmax
        winner need not be the trained category with the best choice value, since match and choice
        rank candidates differently; comparing against it let an adapted category win by beating a
        possibly-weak match-argmax competitor instead of the strongest true trained competitor). This
        is the module's `adaptation_rule` (see module docstring): the adapted category wins only when
        no trained category -- not just the match-argmax one -- explains the query at all well, never
        when an ordinary, already-known chip has a moderate-to-good trained match. Whichever category
        wins this comparison is also still subject to the vigilance test in `score()` (its raw match
        must clear rho*, or the chip is novel regardless of which category "won" here)."""
        I = self.preprocess(X)
        W = np.stack(self.categories, axis=0)
        adapted = np.asarray(self._adapted, dtype=bool)
        n = len(I)
        best = np.zeros(n)
        best_idx = np.zeros(n, dtype=np.int64)

        if not adapted.any() or adapted.all():
            # fast path: no adapted categories to gate (or nothing else to compare them against) --
            # plain raw-match argmax, identical to the pre-adaptation behavior.
            for s in range(0, n, chunk):
                Ic = I[s:s + chunk]
                AND = np.minimum(Ic[:, None, :], W[None, :, :])
                match = AND.sum(axis=2) / (Ic.sum(axis=1, keepdims=True) + 1e-12)
                j = match.argmax(axis=1)
                rows = np.arange(len(Ic))
                best[s:s + chunk] = match[rows, j]
                best_idx[s:s + chunk] = j
            return best, best_idx

        w_sum = W.sum(axis=1)
        trained_idx = np.where(~adapted)[0]
        adapted_idx = np.where(adapted)[0]
        for s in range(0, n, chunk):
            Ic = I[s:s + chunk]
            AND = np.minimum(Ic[:, None, :], W[None, :, :])
            size_and = AND.sum(axis=2)
            match = size_and / (Ic.sum(axis=1, keepdims=True) + 1e-12)
            choice = size_and / (self.alpha + w_sum[None, :])
            rows = np.arange(len(Ic))

            j_trained_local = match[:, trained_idx].argmax(axis=1)
            j_trained = trained_idx[j_trained_local]

            # The gate compares against the BEST choice value among ALL trained categories, not the
            # choice value of whichever trained category happens to win the raw-match argmax (j_trained
            # above) -- those are generally different categories. See `adaptation_rule` / this method's
            # docstring for why the latter was wrong.
            choice_trained_best = choice[:, trained_idx].max(axis=1)

            j_adapted_local = choice[:, adapted_idx].argmax(axis=1)
            j_adapted = adapted_idx[j_adapted_local]
            choice_adapted = choice[rows, j_adapted]

            use_adapted = choice_adapted > choice_trained_best
            j = np.where(use_adapted, j_adapted, j_trained)
            best[s:s + chunk] = match[rows, j]
            best_idx[s:s + chunk] = j
        return best, best_idx

    def calibrate(self, X_ref_calib: np.ndarray, percentile: float = 5.0) -> float:
        """Sets rho_star = the given percentile of max-match over ref_calib (uses no novel/military labels)."""
        best, _ = self._match_all(X_ref_calib)
        self.rho_star = float(np.percentile(best, percentile))
        return self.rho_star

    def score(self, X: np.ndarray) -> dict:
        """Computes novelty=1-max_match, is_novel (max_match < rho_star), and art_category (-1 if novel) for X."""
        best, idx = self._match_all(X)
        is_novel = best < self.rho_star
        art_category = np.where(is_novel, -1, idx)
        return {"novelty": 1.0 - best, "is_novel": is_novel, "art_category": art_category, "max_match": best}

    def add_category(self, embeddings: np.ndarray, label: str | None = None) -> int:
        """Adds one new ART category as the complement-coded mean of `embeddings` (the adaptation step).
        Marked in self._adapted so _match_all gates it against raw-match dominance (see there), and
        `label` is recorded in self.category_labels so callers can map a winning category index back
        to the analyst class it was created for (see vigil.pipeline's `adaptation_rule`-gated role_pred)."""
        I = self.preprocess(embeddings)
        self.categories.append(I.mean(axis=0))
        self._adapted.append(True)
        self.category_labels.append(label)
        return len(self.categories) - 1

    # ---------------------------------------------------------------- kNN baseline
    def fit_knn(self, X_ref_train: np.ndarray, k: int = KNN_K) -> None:
        """Fits a cosine NearestNeighbors index on raw ref_train embeddings, for the knn_dist baseline."""
        self._nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(X_ref_train)
        self._k = k

    def knn_dist(self, X: np.ndarray) -> np.ndarray:
        """Mean cosine distance to the k nearest ref_train embeddings, dropping an exact self-match if present."""
        return knn_dist(self._nn, X, self._k)

    def kneighbors(self, X: np.ndarray, k: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Returns (dists, idx): the k nearest ref_train embeddings (cosine) for each row of X, nearest-first,
        dropping an exact self-match if present. `idx` indexes the ref_train array passed to fit_knn -- callers
        map it back to ann_ids/ids of their own reference table."""
        k = k or self._k
        dist, idx = self._nn.kneighbors(X, n_neighbors=k + 1)
        self_match = dist[:, 0] < 1e-6
        out_dist = np.empty((len(X), k))
        out_idx = np.empty((len(X), k), dtype=np.int64)
        out_dist[self_match] = dist[self_match, 1:k + 1]
        out_idx[self_match] = idx[self_match, 1:k + 1]
        out_dist[~self_match] = dist[~self_match, 0:k]
        out_idx[~self_match] = idx[~self_match, 0:k]
        return out_dist, out_idx


def cross_fit_vigilance(
    X_ref: np.ndarray,
    parent_ids: np.ndarray,
    percentile: float = 5.0,
    k_folds: int = CROSS_FIT_K_FOLDS,
    seed: int = CROSS_FIT_SEED,
    alpha: float = ALPHA,
    beta: float = BETA,
    pca_dims: int = PCA_DIMS,
    cap: int = CAP,
) -> dict:
    """Parent-image k-fold cross-fitted vigilance calibration over the whole reference set.

    Folds are unique parent images (shuffled with `seed`, split into `k_folds` roughly-equal
    groups). For each fold, fits PCA/min-max/Fuzzy-ART on the OTHER folds' reference chips
    (fit_preprocessing + fit_art, one epoch, same recipe as the production model) and scores the
    held-out fold's chips for their out-of-fold max-match. rho* is the `percentile`th percentile
    of the pooled out-of-fold max-match over every reference chip -- no operational, held-out, or
    military data is used anywhere in this computation.

    Returns {rho_star, oof_match (per-chip out-of-fold max-match, same order as X_ref/parent_ids),
    oof_is_novel, calibration_check (per-reference-parent out-of-fold novelty rate: min/median/max/
    n_parents), k_folds}.
    """
    parents = np.unique(parent_ids)
    rng = np.random.default_rng(seed)
    shuffled = parents.copy()
    rng.shuffle(shuffled)
    folds = np.array_split(shuffled, k_folds)

    oof_match = np.zeros(len(X_ref))
    for fold_parents in folds:
        held_out = np.isin(parent_ids, fold_parents)
        if held_out.sum() == 0 or (~held_out).sum() == 0:
            continue
        fold_model = NoveltyModel(alpha=alpha, beta=beta, pca_dims=pca_dims, cap=cap)
        fold_model.fit_preprocessing(X_ref[~held_out])
        fold_model.fit_art(X_ref[~held_out], seed=seed)
        match, _ = fold_model._match_all(X_ref[held_out])
        oof_match[held_out] = match

    rho_star = float(np.percentile(oof_match, percentile))
    oof_is_novel = oof_match < rho_star
    per_parent_rate = pd.Series(oof_is_novel).groupby(parent_ids).mean()
    calibration_check = {
        "min": float(per_parent_rate.min()),
        "median": float(per_parent_rate.median()),
        "max": float(per_parent_rate.max()),
        "n_parents": int(len(per_parent_rate)),
    }
    return {
        "rho_star": rho_star,
        "oof_match": oof_match,
        "oof_is_novel": oof_is_novel,
        "calibration_check": calibration_check,
        "k_folds": k_folds,
    }


def knn_dist(nn: NearestNeighbors, X: np.ndarray, k: int = KNN_K) -> np.ndarray:
    """Mean cosine distance from each row of X to its k nearest neighbors in a fitted index, dropping self-matches."""
    dist, _ = nn.kneighbors(X, n_neighbors=k + 1)
    self_match = dist[:, 0] < 1e-6
    out = np.empty(len(X))
    out[self_match] = dist[self_match, 1:k + 1].mean(axis=1)
    out[~self_match] = dist[~self_match, 0:k].mean(axis=1)
    return out
