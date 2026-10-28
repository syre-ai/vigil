"""Nearest-class-mean civil-role head over L2-normalized DINOv2 embeddings."""

from __future__ import annotations

import numpy as np

# role_id -> known-class name (role_id 4-7 are military, never fit or predicted here)
ROLE_NAMES = {1: "Small Civil", 2: "Medium Civil", 3: "Large Civil"}
KNOWN_ROLE_IDS = sorted(ROLE_NAMES)

DEFAULT_TAU = 0.05


class ProtoHead:
    """Nearest-class-mean classifier: argmax cosine similarity to per-class prototypes, softmax(cos/tau) confidence."""

    def __init__(self, tau: float = DEFAULT_TAU):
        """Initializes an empty prototype head with softmax temperature tau."""
        self.tau = tau
        self.tau_method = "default"  # updated by calibrate_tau / fit_tau_cross_fitted
        self.class_names: list[str] = []
        self.means: np.ndarray | None = None  # (C, D), each row L2-normalized

    def fit(self, X: np.ndarray, labels: list[str]) -> None:
        """Fits one L2-normalized class-mean prototype per unique label seen in `labels` (skips empty classes)."""
        names = sorted(set(labels))
        labels_arr = np.asarray(labels)
        means = []
        for name in names:
            m = X[labels_arr == name].mean(axis=0)
            means.append(m / (np.linalg.norm(m) + 1e-8))
        self.class_names = names
        self.means = np.stack(means, axis=0) if means else np.zeros((0, X.shape[1]), dtype=X.dtype)

    def add_class(self, name: str, embeddings: np.ndarray) -> None:
        """Adds one new prototype class as the L2-normalized mean of `embeddings`, without touching existing means."""
        m = embeddings.mean(axis=0)
        m = (m / (np.linalg.norm(m) + 1e-8))[None, :]
        if self.means is None or len(self.class_names) == 0:
            self.means = m
        else:
            self.means = np.concatenate([self.means, m], axis=0)
        self.class_names = self.class_names + [name]

    def cosine(self, X: np.ndarray) -> np.ndarray:
        """Computes cosine similarity of each row of X to each class mean -> (N, C)."""
        return X @ self.means.T

    def _softmax_conf(self, cos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Computes the argmax class index and its softmax(cos/tau) probability for each row of cosine similarities."""
        logits = cos / self.tau
        logits = logits - logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        p = p / p.sum(axis=1, keepdims=True)
        idx = p.argmax(axis=1)
        conf = p[np.arange(len(idx)), idx]
        return idx, conf

    def calibrate_tau(self, X: np.ndarray, labels: list[str], grid: np.ndarray | None = None) -> float:
        """Fits temperature tau on the given (X, labels) by grid search minimizing softmax-cosine NLL."""
        if len(labels) == 0 or self.means is None or len(self.class_names) == 0:
            return self.tau  # not enough calibration data: keep the default
        cos = self.cosine(X)
        name_to_idx = {n: i for i, n in enumerate(self.class_names)}
        true_idx = np.array([name_to_idx[l] for l in labels if l in name_to_idx])
        cos = cos[[i for i, l in enumerate(labels) if l in name_to_idx]]
        if len(true_idx) == 0:
            return self.tau
        self.tau = self._best_tau(cos, true_idx, grid=grid)
        self.tau_method = "calibrated_plain"
        return self.tau

    def _best_tau(self, cos: np.ndarray, true_idx: np.ndarray, grid: np.ndarray | None = None) -> float:
        """Grid-searches tau to minimize softmax-cosine negative log-likelihood on (cos, true_idx) rows."""
        if grid is None:
            grid = np.geomspace(0.01, 0.5, 30)
        best_tau, best_nll = self.tau, np.inf
        for tau in grid:
            logits = cos / tau
            logits = logits - logits.max(axis=1, keepdims=True)
            logp = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
            nll = -logp[np.arange(len(true_idx)), true_idx].mean()
            if nll < best_nll:
                best_nll, best_tau = nll, float(tau)
        return best_tau

    def fit_tau_cross_fitted(
        self,
        X_civil: np.ndarray,
        labels_civil: list[str],
        parent_ids_civil: np.ndarray,
        k_folds: int = 8,
        seed: int = 0,
        grid: np.ndarray | None = None,
    ) -> float:
        """Cross-fitted temperature calibration, mirroring the parent-image k-fold cross-fitting used
        to calibrate ART vigilance (see vigil.novelty.cross_fit_vigilance) -- calibrating tau on the
        small ref_calib split alone was as unstable as vigilance, for the same reason (few parent
        images). Requires self.means to already be fit on the FULL civil reference set (fit() called
        first); this only recalibrates self.tau, it never touches self.means.

        For each fold, fits a temporary class-mean prototype head on the OTHER folds' civil reference
        chips and records the held-out fold's cosine row (skipping a fold if its complement is missing
        a known class, which would make cosine columns incomparable). Falls back to a plain (non-cross-
        fitted) calibrate_tau on the full civil reference set if cross-fitting is degenerate (e.g. too
        few parent images, or every fold gets skipped).
        """
        if len(labels_civil) == 0 or self.means is None or len(self.class_names) == 0:
            return self.tau
        parents = np.unique(parent_ids_civil)
        k_folds = min(k_folds, len(parents))
        if k_folds < 2:
            return self.calibrate_tau(X_civil, labels_civil, grid=grid)

        rng = np.random.default_rng(seed)
        shuffled = parents.copy()
        rng.shuffle(shuffled)
        folds = np.array_split(shuffled, k_folds)
        labels_arr = np.asarray(labels_civil)

        cos_chunks, idx_chunks = [], []
        for fold_parents in folds:
            held_out = np.isin(parent_ids_civil, fold_parents)
            if held_out.sum() == 0 or (~held_out).sum() == 0:
                continue
            fold_head = ProtoHead(tau=self.tau)
            fold_head.fit(X_civil[~held_out], labels_arr[~held_out].tolist())
            if fold_head.class_names != self.class_names:
                continue  # fold's complement is missing a known class -- cosine columns wouldn't line up
            keep = np.isin(labels_arr[held_out], fold_head.class_names)
            if keep.sum() == 0:
                continue
            name_to_idx = {n: i for i, n in enumerate(fold_head.class_names)}
            cos_chunks.append(fold_head.cosine(X_civil[held_out][keep]))
            idx_chunks.append(np.array([name_to_idx[l] for l in labels_arr[held_out][keep]]))

        if not cos_chunks:
            tau = self.calibrate_tau(X_civil, labels_civil, grid=grid)
            self.tau_method = "cross_fit_degenerate_plain_fallback"
            return tau

        cos_all = np.concatenate(cos_chunks, axis=0)
        idx_all = np.concatenate(idx_chunks, axis=0)
        self.tau = self._best_tau(cos_all, idx_all, grid=grid)
        self.tau_method = "cross_fitted"
        return self.tau

    def predict(self, X: np.ndarray) -> tuple[list[str], np.ndarray]:
        """Predicts the nearest-prototype role name and its softmax(cos/tau) confidence for each row of X."""
        cos = self.cosine(X)
        idx, conf = self._softmax_conf(cos)
        names = [self.class_names[i] for i in idx]
        return names, conf

    def accuracy(self, X: np.ndarray, labels: list[str]) -> float:
        """Computes top-1 accuracy of predict() against ground-truth labels."""
        if len(labels) == 0:
            return float("nan")
        preds, _ = self.predict(X)
        return float(np.mean([p == l for p, l in zip(preds, labels)]))

    def per_class_accuracy(self, X: np.ndarray, labels: list[str]) -> dict[str, dict]:
        """Computes per-class recall (fraction of each true class correctly predicted) and support n."""
        preds, _ = self.predict(X)
        out = {}
        labels_arr = np.asarray(labels)
        preds_arr = np.asarray(preds)
        for name in sorted(set(labels)):
            mask = labels_arr == name
            n = int(mask.sum())
            acc = float((preds_arr[mask] == name).mean()) if n else float("nan")
            out[name] = {"accuracy": acc, "n": n}
        return out
