"""Builds one scored version: cached embeddings -> role head + Fuzzy ART -> per-chip scores + per-batch summaries.

The reference set for ALL fitting (PCA, min-max scaling, Fuzzy ART categories, role prototypes,
role-head temperature, the UMAP reducer, and the kNN index) is ref_train UNION ref_calib. Vigilance
(rho*) and the role head's temperature (tau) are both calibrated by parent-image 8-fold
cross-fitting over that whole reference set -- calibrating on ref_calib alone was unstable (few
parent images; see STATE.md's "calibration + drift redesign"). Per-batch drift is a SCENE-level
statistic computed from whole-tile embeddings (vigil.embed.embed_tiles / data/emb/tile_embeddings.npz),
not chip-level embeddings (see vigil.drift for why).

Analyst-created classes (from `new_class` feedback) are NOT ART categories and NOT prototype-head
classes. They are separate L2-normalized EXEMPLAR sets (one row per labeled chip, never averaged for
the assignment decision) consulted only for chips that the trained ART categories reject as novel
(ADAPTATION_RULE, "Rule B", nearest-exemplar rule): a novel chip joins the analyst class whose closest
labeled exemplar (cosine distance) is nearer than both the chip's own 1-NN distance to the labeled
reference set and every other analyst class's closest exemplar; every other chip gets the prototype
head's civil-only prediction. An optional additional gate (`assign_min_conf`, det chips only, off by
default) can also require the detector's own confidence to clear a threshold. Trained ART categories,
prototypes, and novelty scores are never modified by adaptation, so `analyst_class_assignment_rate`
(vigil.evaluate) is bounded by the novelty flag rate and comes with a `civil_false_assignment_rate`
cost measure.

FIELD MODE (research/08-field-mode-proposal.md): after fitting everything on reference GT chips
exactly as always, this module ALSO scores every DETECTOR chip (scripts/extract_det_chips.py's
data/chips_det/, cropped from the detector's own boxes, not ground-truth annotations) with the SAME
fitted PCA/min-max/ART/proto/kNN/UMAP models -- TRANSFORM ONLY, never re-fits anything; fitting NEVER
uses detector chips, in the initial build or after adaptation. Analyst feedback (`new_class`) may
reference either a GT chip (`source='gt'`) or a detector chip (`source='det'`) -- see _apply_feedback
-- and `used_in_fit` is marked in BOTH scores parquets where applicable (a detector chip's own
used_in_fit in `{version}_det.parquet`; a GT chip's own used_in_fit in `{version}.parquet`, only if a
GT chip itself was labeled). A detector chip reviewed `not_aircraft` is dropped from DET SCORING ONLY
(excluded from `{version}_det.parquet` entirely) -- it is NEVER a training negative for anything,
which holds trivially here since fitting never touches detector chips at all, regardless of this
exclusion (see API.md's POST /review).

Writes, under data/scores/:
    {version_id}.parquet          one row per scored GT chip (still carries the chip-level knn_dist
                                   baseline + true embedding-space neighbors, both reference-set-wide)
    {version_id}_det.parquet      one row per scored DETECTOR chip (field mode) -- absent if
                                   data/chips_det/index.parquet doesn't exist yet
    {version_id}_batches.json     per-batch scene-level drift/novelty/accuracy summary, plus a
                                   "reference" entry (out-of-fold calibration stats, not a batch) and
                                   a "det" entry (field mode: per-domain {n_det_chips, flag_rate} +
                                   a "reference" sub-key naming the ref_holdout numbers again as the
                                   labeled control set -- descriptive only)
    {version_id}_model.json       model metadata (vigilance, classes, calibration_check, ...,
                                   analyst_class_label_counts)
    versions.json                 running list of built versions
    umap.joblib                   cached 2-D UMAP reducer (fit once, reused across versions)

vigil.evaluate.evaluate() (called at the end of build_version) additionally writes a `field` block
(descriptive detector-chip assignment counts over new collections) and a `control` block (labeled
control-set regression check on ref_holdout GT chips) into eval_{version_id}.json -- see vigil.field.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from umap import UMAP

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from vigil import drift as drift_mod  # noqa: E402
from vigil import embed, novelty  # noqa: E402
from vigil.data import MILITARY_ROLE_MIN, load_tiles  # noqa: E402
from vigil.domains import assign_domains  # noqa: E402
from vigil.jsonutil import json_safe  # noqa: E402
from vigil.paths import SCORES_DIR  # noqa: E402  -- honors VIGIL_DATA_ROOT/VIGIL_SCORES_DIR, same as backend/app/config.py
from vigil.proto import KNOWN_ROLE_IDS, ROLE_NAMES, ProtoHead  # noqa: E402

UMAP_PATH = SCORES_DIR / "umap.joblib"
VERSIONS_PATH = SCORES_DIR / "versions.json"

UMAP_SAMPLE_N = 3000
CALIB_PERCENTILE = 5.0
CROSS_FIT_K_FOLDS = 8
CROSS_FIT_SEED = 0
OPERATIONAL_BATCHES = ["ref_holdout", "off_nadir", "foreign", "haze", "snow", "military"]

# The reference set for ALL fitting (PCA, min-max, Fuzzy ART categories, role prototypes, tau,
# UMAP, kNN index) is ref_train UNION ref_calib -- see STATE.md "calibration + drift redesign":
# calibrating on ref_calib alone (5-11 parent images) was unstable (control novelty 15-18% vs a 5%
# target); cross-fitting vigilance over the whole 43-parent-image reference set fixes this.
REFERENCE_DOMAINS = ["ref_train", "ref_calib"]

EMBEDDER_DESC = "facebook/dinov2-small, CLS ++ mean patch token, 768-d, L2-normalized"
NOVELTY_METHOD_DESC = (
    f"Fuzzy ART over PCA-{novelty.PCA_DIMS} + min-max + complement-coded embeddings "
    f"(alpha={novelty.ALPHA}, beta={novelty.BETA}); "
    f"vigilance rho* = {CALIB_PERCENTILE}th percentile of OUT-OF-FOLD max-match, parent-image "
    f"{CROSS_FIT_K_FOLDS}-fold cross-fitting over the whole reference set (ref_train ∪ ref_calib)"
)
ADAPTATION_RULE = (
    "analyst-created classes can only claim chips that fail the vigilance test against every trained "
    "ART category; a chip joins a class only if it is closer to one of that class's analyst-labeled "
    "exemplars than to any chip in the labeled validation set (nearest-exemplar rule). Trained "
    "categories, prototypes and novelty scores are never modified."
)

# Optional ADDITIONAL gate on top of ADAPTATION_RULE, det chips only (det_conf is None for GT chips --
# see _assign_roles): a detector chip may be claimed only if its OWN detection confidence also clears
# `assign_min_conf`. OFF by default (0) -- the nearest-exemplar rule above already removes most of the
# junk on its own: replacing the earlier mean-prototype rule with nearest-exemplar cut a live 8-label
# run's stray assignments elsewhere from 129 (4 military/31 civil/94 junk) to 9 (2/3/4) while keeping
# all 6 of the labeled collection's fused-fighter re-sorts; a mean-prototype + conf>=0.4 gate had only
# gotten to 65 (35 junk) and a conf>=0.5 gate to 21 junk while dropping the labeled collection's own
# re-sort from 14 to 4 (fused two-fighter boxes have detector confidence 0.28-0.46, below 0.5).
# Recorded per version in model.json (`assign_min_conf`) and in versions.json.
ASSIGN_MIN_CONF_DEFAULT = 0.0


def _load_master_chip_table() -> pd.DataFrame:
    """Joins the chip index with tile metadata (domain, parent_id) into one chip-level table, in the
    SAME row order as the cached embeddings (data/emb/embeddings.npz).

    This ordering matters: Fuzzy ART's one-epoch fast learning (beta=1.0) is presentation-order
    sensitive -- fitting the identical reference chips in a different row order can produce a
    materially different set of categories (validated: `index.parquet`'s natural row order produced
    a final model whose match scores never dipped below the cross-fitted rho*, flagging ~0% novel
    everywhere, vs. the embeddings-cache order -- used by the validated reference experiments -- which
    reproduces the expected ~4.5% control novelty rate). Pinning this order keeps `build_version`
    deterministic and consistent with those experiments. Callers must call
    `embed.compute_missing(...)` BEFORE this function so every chip in the current index is already
    present in the cache (any cached ann_id no longer in the current index is dropped defensively).
    """
    cached_ids, _ = embed.load_cache()
    index = embed.load_chip_index().drop(columns=["domain"]).set_index("ann_id")
    cached_ids = np.array([a for a in cached_ids if a in index.index], dtype=np.int64)
    chips = index.loc[cached_ids].reset_index()
    tiles = assign_domains(load_tiles())[["tile_id", "parent_id", "domain"]]
    return chips.merge(tiles, on="tile_id", how="left")


def _role_name(role_id: int) -> str:
    """Maps a ground-truth role_id to its display name ('military' for role_id 4-7)."""
    return ROLE_NAMES.get(int(role_id), "military")


CONFIRM_POLICY_NOTE = "confirm actions are recorded as agreement and do not alter the model"
CORRECT_POLICY_NOTE = "correct actions are recorded as disagreement and do not alter the model"
BAD_BOX_POLICY_NOTE = "bad_box actions are recorded as localization errors and do not alter the model"


def _compose_note(user_note: str | None) -> str:
    """Appends the fixed confirm/correct/bad_box-policy disclaimer to whatever note the caller supplied (or uses it alone)."""
    user_note = (user_note or "").strip()
    policy = f"{CONFIRM_POLICY_NOTE}; {CORRECT_POLICY_NOTE}; {BAD_BOX_POLICY_NOTE}"
    if user_note:
        return f"{user_note} ({policy})"
    return policy


def _apply_feedback(chips: pd.DataFrame, feedback: list[dict] | None) -> dict:
    """Drops 'not_aircraft' GT chips from GT scoring, groups 'new_class' feedback ids (either source)
    by label, and returns everything build_version needs to fit/mark/exclude on both scores parquets.

    Feedback items are `{source: 'gt'|'det' (default 'gt'), ann_id: int, action: str, label: str|None}`
    -- for a detector-chip item, `ann_id` actually carries the det_id (see backend/API.md's "keep the
    API's ann_id naming" convention for POST /review). 'confirm' (agreement) and 'correct'
    (disagreement -- assigns a known class label as evidence) are both recorded but NEVER used to fit
    or exclude anything, exactly alike -- see CONFIRM_POLICY_NOTE / CORRECT_POLICY_NOTE. 'bad_box'
    (the detector's box fuses two aircraft or truncates one -- a localization error, not a
    classification one) is recorded and counted but NEVER fits anything, NEVER excludes the chip from
    scoring (it stays scored, exactly like 'confirm'/'correct'), and is deliberately absent from
    `new_classes`/`used_in_fit_*` -- see BAD_BOX_POLICY_NOTE. Only 'new_class' feeds the analyst class
    exemplar sets (see _analyst_exemplars_from). A detector chip reviewed 'not_aircraft' is dropped from DET SCORING
    ONLY (`exclude_det`) -- never a training negative (fitting never touches detector chips at all,
    regardless of this exclusion).

    Returns {chips (GT chips, 'not_aircraft'(gt) rows dropped), new_classes (label -> [(source, id),
    ...]), used_in_fit_gt (set of GT ann_ids), used_in_fit_det (set of det_ids), exclude_det (set of
    det_ids to drop from det scoring), counts (feedback-action counts, pooled over both sources)}.
    """
    new_classes: dict[str, list[tuple[str, int]]] = {}
    counts = {"n_confirm": 0, "n_new_class": 0, "n_not_aircraft": 0, "n_correct": 0, "n_bad_box": 0}
    exclude_gt: set[int] = set()
    exclude_det: set[int] = set()
    if feedback:
        for f in feedback:
            action = f.get("action")
            source = f.get("source") or "gt"
            fid = int(f["ann_id"])
            if action == "not_aircraft":
                (exclude_det if source == "det" else exclude_gt).add(fid)
                counts["n_not_aircraft"] += 1
            elif action == "confirm":
                counts["n_confirm"] += 1
            elif action == "correct":
                counts["n_correct"] += 1
            elif action == "bad_box":
                counts["n_bad_box"] += 1
            elif action == "new_class" and f.get("label"):
                new_classes.setdefault(f["label"], []).append((source, fid))
                counts["n_new_class"] += 1

    used_in_fit_gt = {fid for ids in new_classes.values() for (src, fid) in ids if src == "gt"}
    used_in_fit_det = {fid for ids in new_classes.values() for (src, fid) in ids if src == "det"}
    chips = chips[~chips["ann_id"].isin(exclude_gt)].reset_index(drop=True)
    return {
        "chips": chips,
        "new_classes": new_classes,
        "used_in_fit_gt": used_in_fit_gt,
        "used_in_fit_det": used_in_fit_det,
        "exclude_det": exclude_det,
        "counts": counts,
    }


def _analyst_exemplars_from(x_for, new_classes: dict[str, list[tuple[str, int]]]) -> dict[str, np.ndarray]:
    """Builds one L2-normalized EXEMPLAR matrix per analyst label -- one row per labeled chip, never
    averaged (see ADAPTATION_RULE's nearest-exemplar rule) -- from whichever embedding space each
    labeled chip belongs to: GT chips via `x_for` (the master GT chip table's cached embeddings,
    ref_train ∪ ref_calib ∪ every operational GT chip), detector chips via
    vigil.embed.get_det_embeddings (field mode). A label may mix chips from both sources. A
    (source, id) pair whose embedding can't be found (e.g. stale feedback referencing a chip that no
    longer exists) is skipped with a warning rather than crashing the whole build."""
    exemplars: dict[str, np.ndarray] = {}
    for label, ids in new_classes.items():
        gt_ids = [i for s, i in ids if s == "gt"]
        det_ids = [i for s, i in ids if s == "det"]
        parts = []
        if gt_ids:
            try:
                parts.append(x_for(np.array(gt_ids, dtype=np.int64)))
            except KeyError as e:
                print(f"vigil.pipeline: skipping unknown GT chip(s) for analyst class {label!r}: {e}", file=sys.stderr)
        if det_ids:
            try:
                parts.append(embed.get_det_embeddings(np.array(det_ids, dtype=np.int64)))
            except KeyError as e:
                print(f"vigil.pipeline: skipping unknown det chip(s) for analyst class {label!r}: {e}", file=sys.stderr)
        if not parts:
            continue
        Xc = np.concatenate(parts, axis=0)
        exemplars[label] = Xc / (np.linalg.norm(Xc, axis=1, keepdims=True) + 1e-12)
    return exemplars


def _assign_roles(
    X: np.ndarray,
    is_novel: np.ndarray,
    proto: "ProtoHead",
    analyst_exemplars: dict[str, np.ndarray],
    ref_1nn_dist: np.ndarray,
    det_conf: np.ndarray | None = None,
    assign_min_conf: float = ASSIGN_MIN_CONF_DEFAULT,
) -> dict:
    """role_pred/role_conf/analyst_sim/analyst_margin for embeddings X given their (already computed)
    ART novelty flags: the civil prototype head decides for every chip (softmax(cos/tau)); analyst
    classes may ONLY claim chips flagged novel, via the NEAREST-EXEMPLAR rule (ADAPTATION_RULE): a
    novel chip joins the analyst class whose closest labeled exemplar (cosine distance) is BOTH (a)
    nearer than the chip's own 1-NN cosine distance to the labeled reference set (`ref_1nn_dist` --
    already computed by art.kneighbors, the first column of `neighbor_dists`) and (b) nearer than
    every other analyst class's closest exemplar. `det_conf`/`assign_min_conf` are an OPTIONAL
    additional gate, det chips only (det_conf is None for GT chips, which are never gated on it): when
    assign_min_conf > 0 a detector chip may only be claimed if its own detection confidence also
    clears assign_min_conf -- off (0) by default, since the exemplar rule already removes most of the
    junk on its own (see ASSIGN_MIN_CONF_DEFAULT's comment for the measured trade-off). Shared by the
    full build (_score_all, is_novel/ref_1nn_dist fresh from the fitted ART/kNN index) and the
    incremental build (both reused from the base version -- the reference set, hence ART and the kNN
    index, is unchanged)."""
    role_pred_civil, role_conf_civil = proto.predict(X)
    role_pred_civil = np.asarray(role_pred_civil, dtype=object)

    n = len(X)
    analyst_label = np.full(n, None, dtype=object)
    analyst_sim = np.zeros(n, dtype=np.float32)     # 1 - nearest-exemplar cosine distance (winning class); 0 if unclaimed
    analyst_margin = np.zeros(n, dtype=np.float32)  # ref_1nn_dist - nearest-exemplar distance (winning class); 0 if unclaimed
    analyst_conf = np.zeros(n, dtype=np.float32)    # comparable-scale role_conf for claimed chips (see below)
    if analyst_exemplars:
        is_novel = np.asarray(is_novel, dtype=bool)
        ref_1nn_dist = np.asarray(ref_1nn_dist, dtype=np.float64)
        rows = np.arange(n)
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        a_names = list(analyst_exemplars.keys())
        # per-class nearest-exemplar cosine distance: 1 - max cosine similarity to that class's own
        # labeled exemplars (never a class mean -- ADAPTATION_RULE's nearest-exemplar rule).
        class_min_dist = np.stack(
            [1.0 - (Xn @ analyst_exemplars[name].T).max(axis=1) for name in a_names], axis=1,
        )
        best_a = class_min_dist.argmin(axis=1)
        best_dist = class_min_dist[rows, best_a]
        margin = ref_1nn_dist - best_dist
        claim = is_novel & (best_dist < ref_1nn_dist)
        if det_conf is not None and assign_min_conf > 0:
            claim = claim & (np.asarray(det_conf, dtype=np.float64) >= assign_min_conf)
        analyst_label[claim] = np.array(a_names, dtype=object)[best_a[claim]]
        analyst_sim[claim] = (1.0 - best_dist[claim]).astype(np.float32)
        analyst_margin[claim] = margin[claim].astype(np.float32)

        # role_conf for analyst-claimed chips: softmax(cos/tau) over the UNION of civil class means +
        # each analyst class's own exemplar MEAN (display-scale confidence only -- the nearest-exemplar
        # rule above, not this mean, decides which chips are claimed and by which class), using the
        # role head's own fitted tau (proto.tau) -- the SAME scale as role_conf_civil (softmax(cos/tau)
        # over civil means only), so `role_conf` is one comparable probability column across civil and
        # analyst-claimed predictions regardless of which side wins. The raw nearest-exemplar
        # similarity/margin are kept separately as `analyst_sim`/`analyst_margin`.
        civ_means = np.asarray(proto.means); civ_means = civ_means / (np.linalg.norm(civ_means, axis=1, keepdims=True) + 1e-12)
        a_means = np.stack([
            analyst_exemplars[name].mean(axis=0) / (np.linalg.norm(analyst_exemplars[name].mean(axis=0)) + 1e-12)
            for name in a_names
        ])
        union_means = np.concatenate([civ_means, a_means], axis=0)
        n_civil = civ_means.shape[0]
        logits = (Xn @ union_means.T) / proto.tau
        logits = logits - logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        p = p / p.sum(axis=1, keepdims=True)
        union_idx = n_civil + best_a
        analyst_conf[claim] = p[rows, union_idx][claim]

    has_analyst_label = analyst_label != None  # noqa: E711 -- elementwise object-array None check
    return {
        "role_pred": np.where(has_analyst_label, analyst_label, role_pred_civil),
        "role_conf": np.where(has_analyst_label, analyst_conf, role_conf_civil),
        "analyst_sim": analyst_sim,
        "analyst_margin": analyst_margin,
    }


def _score_all(
    X: np.ndarray,
    art: "novelty.NoveltyModel",
    proto: "ProtoHead",
    analyst_exemplars: dict[str, np.ndarray],
    det_conf: np.ndarray | None = None,
    assign_min_conf: float = ASSIGN_MIN_CONF_DEFAULT,
) -> dict:
    """Scores an arbitrary set of chip embeddings X (GT chips OR detector chips -- the only
    difference between them is which embeddings are passed in) with the ALREADY-FITTED
    art/proto/analyst_exemplars models -- TRANSFORM ONLY, never re-fits anything: novelty/is_novel/
    art_category (Fuzzy ART), knn_dist (chip-level baseline distance to the reference set), the 1-NN
    reference distance + neighbor matrices (art.kneighbors, computed once here and reused by callers
    instead of a second kNN pass), and role_pred/role_conf/analyst_sim/analyst_margin via
    _assign_roles. Shared by build_version's GT-chip scoring (all_X -> {version}.parquet) and its
    detector-chip scoring (field mode -> {version}_det.parquet, which also passes det_conf/
    assign_min_conf) so both use IDENTICAL logic.
    """
    scored = art.score(X)
    knn_dist = art.knn_dist(X)
    neighbor_dist, neighbor_idx = art.kneighbors(X)  # nearest-first; column 0 = 1-NN cosine distance to the reference set
    roles = _assign_roles(X, scored["is_novel"], proto, analyst_exemplars, neighbor_dist[:, 0], det_conf, assign_min_conf)
    return {
        **roles,
        "novelty": scored["novelty"],
        "is_novel": scored["is_novel"],
        "art_category": scored["art_category"],
        "knn_dist": knn_dist,
        "neighbor_dist": neighbor_dist,
        "neighbor_idx": neighbor_idx,
    }


def _get_or_fit_umap(ref_ann_ids: np.ndarray, x_for) -> UMAP:
    """Loads the cached UMAP reducer if present, else fits one on <=3000 sampled reference chips
    (ref_train ∪ ref_calib, seed 0) and caches it."""
    if UMAP_PATH.exists():
        return joblib.load(UMAP_PATH)
    if len(ref_ann_ids) <= UMAP_SAMPLE_N:
        sample_ids = ref_ann_ids
    else:
        sample_ids = np.random.default_rng(0).choice(ref_ann_ids, size=UMAP_SAMPLE_N, replace=False)
    reducer = UMAP(n_components=2, random_state=0)
    reducer.fit(x_for(sample_ids))
    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(reducer, UMAP_PATH)
    return reducer


def _load_versions() -> list[dict]:
    """Reads data/scores/versions.json, or an empty list if it doesn't exist yet."""
    if VERSIONS_PATH.exists():
        return json.loads(VERSIONS_PATH.read_text())
    return []


def _save_version_entry(entry: dict) -> None:
    """Appends (or replaces, if version_id already exists) one entry in data/scores/versions.json."""
    versions = [v for v in _load_versions() if v["id"] != entry["id"]]
    versions.append(entry)
    VERSIONS_PATH.write_text(json.dumps(versions, indent=2))


def _score_det_chips(
    version_id: str, art: "novelty.NoveltyModel", proto: "ProtoHead", analyst_exemplars: dict[str, np.ndarray],
    ref_fit_ann_ids: np.ndarray, reducer: UMAP, used_in_fit_det: set[int], exclude_det: set[int],
    assign_min_conf: float = ASSIGN_MIN_CONF_DEFAULT,
) -> dict | None:
    """Field mode: scores every detector chip currently in data/chips_det/index.parquet with the
    SAME already-fitted models (transform only) and writes {version_id}_det.parquet. Returns the
    "det" block for {version_id}_batches.json ({domain: {n_det_chips, flag_rate}, "reference":
    {n_det_chips_ref_holdout, flag_rate_ref_holdout}}), or None if data/chips_det/index.parquet
    doesn't exist yet (a repo/version built before the det-chip milestone -- degrades gracefully,
    never an error).
    """
    if not embed.DET_CHIP_INDEX_PATH.exists():
        return None

    det_cached_ids, _ = embed.load_det_cache()
    det_cached_set = set(det_cached_ids.tolist())
    det_index = embed.load_det_chip_index()
    det_df = det_index[det_index["det_id"].isin(det_cached_set)].drop_duplicates("det_id").reset_index(drop=True)
    # not_aircraft(det) drops the chip from DET SCORING ONLY -- fitting never touched detector chips
    # in the first place, so this can never become a training negative either way.
    det_df = det_df[~det_df["det_id"].isin(exclude_det)].reset_index(drop=True)

    if len(det_df) == 0:
        (SCORES_DIR / f"{version_id}_det.parquet").unlink(missing_ok=True)
        return {}

    det_X = embed.get_det_embeddings(det_df["det_id"].to_numpy())
    det_scored = _score_all(
        det_X, art, proto, analyst_exemplars, det_conf=det_df["conf"].to_numpy(), assign_min_conf=assign_min_conf,
    )
    det_neighbor_ids_mat = ref_fit_ann_ids[det_scored["neighbor_idx"]]
    det_umap_xy = reducer.transform(det_X)

    det_scores = pd.DataFrame({
        "det_id": det_df["det_id"].to_numpy(),
        "tile_id": det_df["tile_id"].to_numpy(),
        "parent_id": det_df["parent_id"].to_numpy(),
        "batch_id": det_df["domain"].to_numpy(),
        "role_pred": det_scored["role_pred"],
        "role_conf": det_scored["role_conf"],
        "analyst_sim": det_scored["analyst_sim"],
        "analyst_margin": det_scored["analyst_margin"],
        "novelty": det_scored["novelty"],
        "is_novel": det_scored["is_novel"],
        "art_category": pd.array(
            [None if c < 0 else int(c) for c in det_scored["art_category"]], dtype="Int64"
        ),
        "knn_dist": det_scored["knn_dist"],
        "neighbor_ids": [row.tolist() for row in det_neighbor_ids_mat],
        "neighbor_dists": [row.tolist() for row in det_scored["neighbor_dist"]],
        "umap_x": det_umap_xy[:, 0],
        "umap_y": det_umap_xy[:, 1],
        "used_in_fit": det_df["det_id"].isin(used_in_fit_det).to_numpy(),
        "conf": det_df["conf"].to_numpy(),
    })
    det_scores.to_parquet(SCORES_DIR / f"{version_id}_det.parquet", index=False)
    return _det_block(det_scores)


def _det_block(det_scores: pd.DataFrame) -> dict:
    """{version}_batches.json's "det" block: per-domain {n_det_chips, flag_rate} + a "reference" sub-key
    restating ref_holdout's numbers under field-mode key names (descriptive only)."""
    domain_arr = det_scores["batch_id"].to_numpy()
    is_novel_arr = det_scores["is_novel"].to_numpy()
    det_block: dict = {}
    for dom in sorted(set(domain_arr.tolist())):
        mask = domain_arr == dom
        n_det_chips = int(mask.sum())
        flag_rate = float(is_novel_arr[mask].mean()) if n_det_chips else float("nan")
        det_block[dom] = {"n_det_chips": n_det_chips, "flag_rate": flag_rate}
    if "ref_holdout" in det_block:
        det_block["reference"] = {
            "n_det_chips_ref_holdout": det_block["ref_holdout"]["n_det_chips"],
            "flag_rate_ref_holdout": det_block["ref_holdout"]["flag_rate"],
        }
    return det_block


# Columns that depend ONLY on the reference set + the fitted ART/kNN/UMAP models -- identical across every
# version (adaptation never modifies trained categories, prototypes' novelty gate, or the reducer), so an
# incremental build reuses them from the base version instead of recomputing (see build_version).
REUSED_SCORE_COLS = ["novelty", "is_novel", "art_category", "knn_dist", "umap_x", "umap_y", "neighbor_ids", "neighbor_dists"]


def _score_det_chips_incremental(
    version_id: str, base_version_id: str, proto: "ProtoHead", analyst_exemplars: dict[str, np.ndarray],
    used_in_fit_det: set[int], exclude_det: set[int], assign_min_conf: float = ASSIGN_MIN_CONF_DEFAULT,
) -> dict | None:
    """Incremental twin of _score_det_chips: reuses REUSED_SCORE_COLS from {base}_det.parquet (aligned by
    det_id) and recomputes only role_pred/role_conf/analyst_sim/analyst_margin/used_in_fit. Returns None if
    the base has no det scores or doesn't cover every det chip in the index (caller then falls back to the
    full path)."""
    if not embed.DET_CHIP_INDEX_PATH.exists():
        return {}  # nothing to score (same as the full path's "no det index" -> batches["det"] = None below)
    base_path = SCORES_DIR / f"{base_version_id}_det.parquet"
    if not base_path.exists():
        return None
    base = pd.read_parquet(base_path)
    if not set(REUSED_SCORE_COLS) <= set(base.columns):
        return None
    det_cached_ids, _ = embed.load_det_cache()
    det_index = embed.load_det_chip_index()
    det_df = det_index[det_index["det_id"].isin(set(det_cached_ids.tolist()))].drop_duplicates("det_id").reset_index(drop=True)
    det_df = det_df[~det_df["det_id"].isin(exclude_det)].reset_index(drop=True)
    if len(det_df) == 0:
        (SCORES_DIR / f"{version_id}_det.parquet").unlink(missing_ok=True)
        return {}
    base_by_id = base.set_index("det_id")
    if not det_df["det_id"].isin(base_by_id.index).all():
        return None  # new det chips since the base version -> full rescoring needed
    reused = base_by_id.loc[det_df["det_id"].to_numpy(), REUSED_SCORE_COLS]
    det_X = embed.get_det_embeddings(det_df["det_id"].to_numpy())
    ref_1nn_dist = np.array([d[0] for d in reused["neighbor_dists"]], dtype=np.float64)
    roles = _assign_roles(
        det_X, reused["is_novel"].to_numpy(dtype=bool), proto, analyst_exemplars, ref_1nn_dist,
        det_conf=det_df["conf"].to_numpy(), assign_min_conf=assign_min_conf,
    )
    det_scores = pd.DataFrame({
        "det_id": det_df["det_id"].to_numpy(),
        "tile_id": det_df["tile_id"].to_numpy(),
        "parent_id": det_df["parent_id"].to_numpy(),
        "batch_id": det_df["domain"].to_numpy(),
        "role_pred": roles["role_pred"],
        "role_conf": roles["role_conf"],
        "analyst_sim": roles["analyst_sim"],
        "analyst_margin": roles["analyst_margin"],
        "novelty": reused["novelty"].to_numpy(),
        "is_novel": reused["is_novel"].to_numpy(dtype=bool),
        "art_category": reused["art_category"].to_numpy(),
        "knn_dist": reused["knn_dist"].to_numpy(),
        "neighbor_ids": reused["neighbor_ids"].tolist(),
        "neighbor_dists": reused["neighbor_dists"].tolist(),
        "umap_x": reused["umap_x"].to_numpy(),
        "umap_y": reused["umap_y"].to_numpy(),
        "used_in_fit": det_df["det_id"].isin(used_in_fit_det).to_numpy(),
        "conf": det_df["conf"].to_numpy(),
    })
    det_scores["art_category"] = det_scores["art_category"].astype("Int64")
    det_scores.to_parquet(SCORES_DIR / f"{version_id}_det.parquet", index=False)
    return _det_block(det_scores)


def _empty_scene_drift() -> dict:
    """Fallback drift dict (same schema as drift_mod.batch_scene_drift) for an empty batch."""
    return {
        "score": float("nan"), "null_p95": float("nan"), "p_value": float("nan"),
        "p_value_method": drift_mod.P_VALUE_METHOD,
        "n_boot": drift_mod.N_BOOT, "n": 0, "statistic": drift_mod.STATISTIC_NAME,
        "null_method": drift_mod.NULL_METHOD, "calib_mean": float("nan"),
        "ratio_to_calib": float("nan"), "level": "scene", "interpretation": "descriptive",
    }


def _batch_summaries(chips: pd.DataFrame, scores: pd.DataFrame, tiles_dom: pd.DataFrame, drift_by_batch: dict) -> dict:
    """{batch_id: {drift, novelty_rate, novelty_rate_military, novelty_rate_civil, role_accuracy, conf_shift,
    n_chips, n_tiles, n_parent_images}} for every OPERATIONAL batch. `drift_by_batch` is computed by the
    full build (scene-level tile embeddings) or copied from the base version by the incremental build (the
    reference set, hence drift, is unchanged). Everything else is recomputed from `chips`/`scores`."""
    domain_arr = chips["domain"].to_numpy()
    ref_holdout_conf = scores.loc[domain_arr == "ref_holdout", "role_conf"].to_numpy()
    batches = {}
    for batch_id in OPERATIONAL_BATCHES:
        mask = domain_arr == batch_id
        n = int(mask.sum())
        batch_chips = chips[mask]
        sub_scores = scores[mask]
        tile_mask = (tiles_dom["domain"] == batch_id).to_numpy()
        novelty_rate = float(sub_scores["is_novel"].mean()) if n > 0 else float("nan")

        known_mask = batch_chips["role_id"].isin(KNOWN_ROLE_IDS).to_numpy()
        if known_mask.sum() > 0:
            true_names = [_role_name(r) for r in batch_chips.loc[known_mask, "role_id"]]
            pred_names = sub_scores.loc[known_mask, "role_pred"].tolist()
            role_accuracy = float(np.mean([p == t for p, t in zip(pred_names, true_names)]))
        else:
            role_accuracy = None

        if n > 1 and len(ref_holdout_conf) > 1:
            conf_shift = {
                "ref_mean": float(np.mean(ref_holdout_conf)),
                "batch_mean": float(sub_scores["role_conf"].mean()),
                "ks_p": float(ks_2samp(ref_holdout_conf, sub_scores["role_conf"].to_numpy()).pvalue),
            }
        else:
            conf_shift = None

        # rare-target flag rate (evaluation-only, GT role_id) -- only meaningful where the batch mixes
        # military and civil aircraft at the same airfields (the "military" batch); null elsewhere.
        if batch_id == "military" and n > 0:
            role_id_arr = batch_chips["role_id"].to_numpy()
            is_novel_arr = sub_scores["is_novel"].to_numpy()
            mil_mask = role_id_arr >= MILITARY_ROLE_MIN
            civ_mask = ~mil_mask
            novelty_rate_military = float(is_novel_arr[mil_mask].mean()) if mil_mask.sum() else float("nan")
            novelty_rate_civil = float(is_novel_arr[civ_mask].mean()) if civ_mask.sum() else float("nan")
        else:
            novelty_rate_military = None
            novelty_rate_civil = None

        batches[batch_id] = {
            "drift": drift_by_batch.get(batch_id, _empty_scene_drift()),
            "novelty_rate": novelty_rate,
            "novelty_rate_military": novelty_rate_military,
            "novelty_rate_civil": novelty_rate_civil,
            "role_accuracy": role_accuracy,
            "conf_shift": conf_shift,
            "n_chips": n,
            "n_tiles": int(tile_mask.sum()),
            "n_parent_images": int(tiles_dom.loc[tile_mask, "parent_id"].nunique()),
        }
    return batches


def _find_base_version(version_id: str) -> str | None:
    """Most recent already-built version (other than version_id) whose scores/batches/model files all exist
    -- the incremental build's source for REUSED_SCORE_COLS. None if there is none (first build)."""
    for v in reversed(_load_versions()):
        vid = v["id"]
        if vid == version_id:
            continue
        if all((SCORES_DIR / f).exists() for f in (f"{vid}.parquet", f"{vid}_batches.json", f"{vid}_model.json")):
            return vid
    return None


def _version_entry(
    version_id: str, all_class_names: list[str], feedback, feedback_counts: dict, note: str,
    assign_min_conf: float = ASSIGN_MIN_CONF_DEFAULT,
) -> dict:
    return {
        "id": version_id,
        "created": datetime.now(timezone.utc).isoformat(),
        "classes": all_class_names,
        "n_feedback": len(feedback) if feedback else 0,
        "n_confirm": feedback_counts["n_confirm"],
        "n_new_class": feedback_counts["n_new_class"],
        "n_not_aircraft": feedback_counts["n_not_aircraft"],
        "n_correct": feedback_counts["n_correct"],
        "n_bad_box": feedback_counts["n_bad_box"],
        "assign_min_conf": assign_min_conf,
        # The confirm/correct-policy disclaimer only makes sense once feedback exists to apply it to
        # (i.e. an /adapt-built version) -- the initial no-feedback build keeps its plain note.
        "note": _compose_note(note) if feedback else (note or ""),
    }


def _analyst_class_label_counts(new_classes: dict[str, list[tuple[str, int]]]) -> dict:
    """Field mode: how many feedback items, per source, fed each analyst class -- lets vigil.field report
    n_labels_used per class without re-deriving it from role_pred."""
    return {
        label: {
            "n_total": len(ids),
            "n_gt": sum(1 for s, _ in ids if s == "gt"),
            "n_det": sum(1 for s, _ in ids if s == "det"),
        }
        for label, ids in new_classes.items()
    }


def build_version(
    version_id: str, feedback: list[dict] | None = None, note: str = "", full: bool | None = None,
    assign_min_conf: float = ASSIGN_MIN_CONF_DEFAULT,
) -> dict:
    """Builds one scored version and writes scores/batches/model/eval/versions for version_id.

    `full=None` (default) picks the path automatically: the FULL build (fit PCA/ART/prototypes/tau/UMAP/kNN
    on the reference set, cross-fit vigilance, scene drift, score everything -- minutes) when no earlier
    version exists, else the INCREMENTAL build (_build_version_incremental: the reference set -- hence the
    trained ART categories, novelty scores, kNN baseline, neighbors, UMAP coordinates and scene drift -- is
    identical across versions, so those are reused from the base version and only the role head's
    outputs, the analyst class means, used_in_fit, the batch summaries and the evaluation are recomputed;
    seconds). `full=True` forces the full path (the CLI's v1 build); `full=False` forces incremental and
    raises if no base exists. The incremental path falls back to the full one if the base doesn't cover
    every chip currently in the index (new chips since the base was built). `assign_min_conf` is the
    OPTIONAL additional detector-confidence gate on top of ADAPTATION_RULE's nearest-exemplar rule (det
    chips only; 0 = off, the default -- see ASSIGN_MIN_CONF_DEFAULT), recorded in model.json/versions.json.
    """
    if full is not True:
        base = _find_base_version(version_id)
        if base is not None:
            result = _build_version_incremental(version_id, feedback, note, base, assign_min_conf)
            if result is not None:
                return result
            print(f"vigil.pipeline: base {base} does not cover every chip in the current index -- full build", file=sys.stderr)
        elif full is False:
            raise RuntimeError("build_version(full=False): no earlier version to build incrementally from")
    return _build_version_full(version_id, feedback, note, assign_min_conf)


def _build_version_incremental(
    version_id: str, feedback: list[dict] | None, note: str, base: str,
    assign_min_conf: float = ASSIGN_MIN_CONF_DEFAULT,
) -> dict | None:
    """See build_version. Reuses REUSED_SCORE_COLS (+ drift, reference stats, ART metadata) from `base`;
    refits ONLY the civil prototype means (tau reused from the base model -- same data, same seed, same
    value) and the analyst class means; recomputes role_pred/role_conf/analyst_sim/used_in_fit for GT and
    det chips, the per-batch summaries, model.json, versions.json and eval_{version}.json. Returns None if
    the base doesn't cover every chip in the current index."""
    t0 = time.time()
    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    base_scores = pd.read_parquet(SCORES_DIR / f"{base}.parquet")
    if not set(REUSED_SCORE_COLS) <= set(base_scores.columns):
        return None
    base_model = json.loads((SCORES_DIR / f"{base}_model.json").read_text())
    base_batches = json.loads((SCORES_DIR / f"{base}_batches.json").read_text())

    chips = _load_master_chip_table()
    fb = _apply_feedback(chips, feedback)
    chips = fb["chips"]
    new_classes = fb["new_classes"]
    base_by_id = base_scores.set_index("ann_id")
    if not chips["ann_id"].isin(base_by_id.index).all():
        return None

    all_X = embed.get_embeddings(chips["ann_id"].to_numpy())
    ann_id_to_pos = {a: i for i, a in enumerate(chips["ann_id"].to_numpy())}

    def x_for(ann_ids: np.ndarray) -> np.ndarray:
        return all_X[[ann_id_to_pos[a] for a in ann_ids]]

    ref_fit = chips[chips["domain"].isin(REFERENCE_DOMAINS)]
    proto = ProtoHead()
    ref_civil = ref_fit[ref_fit["role_id"].isin(KNOWN_ROLE_IDS)]
    proto.fit(x_for(ref_civil["ann_id"].to_numpy()), [_role_name(r) for r in ref_civil["role_id"]])
    if base_model.get("tau") is not None:
        proto.tau = float(base_model["tau"])
        proto.tau_method = base_model.get("tau_method") or proto.tau_method
    else:  # pragma: no cover - a base model.json without tau (never written by this module)
        proto.fit_tau_cross_fitted(
            x_for(ref_civil["ann_id"].to_numpy()), [_role_name(r) for r in ref_civil["role_id"]],
            ref_civil["parent_id"].to_numpy(), k_folds=CROSS_FIT_K_FOLDS, seed=CROSS_FIT_SEED,
        )
    analyst_exemplars = _analyst_exemplars_from(x_for, new_classes)
    analyst_class_names = sorted(analyst_exemplars.keys())
    all_class_names = proto.class_names + analyst_class_names

    reused = base_by_id.loc[chips["ann_id"].to_numpy(), REUSED_SCORE_COLS]
    ref_1nn_dist = np.array([d[0] for d in reused["neighbor_dists"]], dtype=np.float64)
    roles = _assign_roles(
        all_X, reused["is_novel"].to_numpy(dtype=bool), proto, analyst_exemplars, ref_1nn_dist,
    )
    scores = pd.DataFrame({
        "ann_id": chips["ann_id"].to_numpy(),
        "tile_id": chips["tile_id"].to_numpy(),
        "batch_id": np.where(chips["domain"].isin(REFERENCE_DOMAINS), "reference", chips["domain"]),
        "role_pred": roles["role_pred"],
        "role_conf": roles["role_conf"],
        "analyst_sim": roles["analyst_sim"],
        "analyst_margin": roles["analyst_margin"],
        "novelty": reused["novelty"].to_numpy(),
        "is_novel": reused["is_novel"].to_numpy(dtype=bool),
        "art_category": reused["art_category"].to_numpy(),
        "knn_dist": reused["knn_dist"].to_numpy(),
        "umap_x": reused["umap_x"].to_numpy(),
        "umap_y": reused["umap_y"].to_numpy(),
        "used_in_fit": chips["ann_id"].isin(fb["used_in_fit_gt"]).to_numpy(),
        "neighbor_ids": reused["neighbor_ids"].tolist(),
        "neighbor_dists": reused["neighbor_dists"].tolist(),
    })
    scores["art_category"] = scores["art_category"].astype("Int64")
    scores.to_parquet(SCORES_DIR / f"{version_id}.parquet", index=False)

    tiles_dom = assign_domains(load_tiles())[["tile_id", "parent_id", "domain"]]
    drift_by_batch = {b: (base_batches.get(b) or {}).get("drift") or _empty_scene_drift() for b in OPERATIONAL_BATCHES}
    batches = _batch_summaries(chips, scores, tiles_dom, drift_by_batch)
    batches["reference"] = base_batches.get("reference", {})
    det_block = _score_det_chips_incremental(
        version_id, base, proto, analyst_exemplars, fb["used_in_fit_det"], fb["exclude_det"], assign_min_conf,
    )
    if det_block is None:
        return None  # base has no det scores (or doesn't cover the index): full rescoring needed
    batches["det"] = det_block if embed.DET_CHIP_INDEX_PATH.exists() else None
    (SCORES_DIR / f"{version_id}_batches.json").write_text(json.dumps(json_safe(batches), indent=2))

    model_info = {
        **base_model,
        "classes": all_class_names,
        "tau": proto.tau,
        "tau_method": proto.tau_method,
        "adaptation_rule": ADAPTATION_RULE,
        "assign_min_conf": assign_min_conf,
        "analyst_class_label_counts": _analyst_class_label_counts(new_classes),
        "built_from": base,
        "build_mode": (
            f"incremental: reference set unchanged, so Fuzzy ART categories/vigilance, novelty scores, kNN "
            f"baseline, neighbors, UMAP coordinates and scene drift are reused from {base}; role head means "
            f"refit (tau reused), analyst class means, role predictions, used_in_fit, batch summaries and "
            f"evaluation recomputed"
        ),
    }
    (SCORES_DIR / f"{version_id}_model.json").write_text(json.dumps(json_safe(model_info), indent=2))

    version_entry = _version_entry(version_id, all_class_names, feedback, fb["counts"], note, assign_min_conf)
    from vigil import evaluate as evaluate_mod  # local import: evaluate.py imports from this module at load time

    eval_result = evaluate_mod.evaluate(version_id)
    _save_version_entry(version_entry)
    return {
        "version_id": version_id,
        "n_chips": len(chips),
        "model": model_info,
        "batches": batches,
        "eval": eval_result,
        "seconds": time.time() - t0,
        "build_mode": "incremental",
        "built_from": base,
    }


def _build_version_full(version_id: str, feedback: list[dict] | None, note: str, assign_min_conf: float = ASSIGN_MIN_CONF_DEFAULT) -> dict:
    """The full build -- see build_version. Fits the role head + Fuzzy ART on cached embeddings and writes
    scores/batches/model/eval/versions for version_id."""
    t0 = time.time()
    SCORES_DIR.mkdir(parents=True, exist_ok=True)

    embed.compute_missing(embed.load_chip_index())  # ensure every chip currently in the index is embedded
    if embed.DET_CHIP_INDEX_PATH.exists():
        # Field mode: ensure every detector chip currently in the index is embedded too (transform-only
        # downstream -- see _score_det_chips; never used to fit anything above).
        embed.embed_det_chips()

    chips = _load_master_chip_table()  # ordered to match the (now-complete) embeddings cache -- see its docstring
    fb = _apply_feedback(chips, feedback)
    chips = fb["chips"]
    new_classes = fb["new_classes"]
    used_in_fit_gt = fb["used_in_fit_gt"]
    used_in_fit_det = fb["used_in_fit_det"]
    exclude_det = fb["exclude_det"]
    feedback_counts = fb["counts"]

    all_X = embed.get_embeddings(chips["ann_id"].to_numpy())
    ann_id_to_pos = {a: i for i, a in enumerate(chips["ann_id"].to_numpy())}

    def x_for(ann_ids: np.ndarray) -> np.ndarray:
        """Looks up rows of all_X for a list of ann_ids, in order."""
        return all_X[[ann_id_to_pos[a] for a in ann_ids]]

    # ---- the reference set for ALL fitting is ref_train UNION ref_calib (see REFERENCE_DOMAINS) ----
    ref_fit = chips[chips["domain"].isin(REFERENCE_DOMAINS)]
    ref_fit_X = x_for(ref_fit["ann_id"].to_numpy())
    ref_fit_parent_ids = ref_fit["parent_id"].to_numpy()

    # ---- role head: nearest-class-mean over reference civil chips, tau cross-fitted (parent-image 8-fold) ----
    # Analyst-created classes are NEVER added to the prototype head. They are assigned ONLY via the
    # ART gate below (see ADAPTATION_RULE / novelty.NoveltyModel's module docstring) -- the head is
    # fit, and stays fit, on the three original civil classes for the life of every version, so its
    # own predictions can never claim an analyst class independently of ART's vigilance/choice gate.
    proto = ProtoHead()
    ref_civil = ref_fit[ref_fit["role_id"].isin(KNOWN_ROLE_IDS)]
    ref_civil_X = x_for(ref_civil["ann_id"].to_numpy())
    ref_civil_labels = [_role_name(r) for r in ref_civil["role_id"]]
    proto.fit(ref_civil_X, ref_civil_labels)
    proto.fit_tau_cross_fitted(
        ref_civil_X, ref_civil_labels, ref_civil["parent_id"].to_numpy(),
        k_folds=CROSS_FIT_K_FOLDS, seed=CROSS_FIT_SEED,
    )

    # ---- novelty: Fuzzy ART fit on the reference set, vigilance cross-fitted (parent-image 8-fold), + kNN baseline ----
    art = novelty.NoveltyModel()
    art.fit_preprocessing(ref_fit_X)
    art_stats = art.fit_art(ref_fit_X, seed=CROSS_FIT_SEED)
    cross_fit = novelty.cross_fit_vigilance(
        ref_fit_X, ref_fit_parent_ids, percentile=CALIB_PERCENTILE,
        k_folds=CROSS_FIT_K_FOLDS, seed=CROSS_FIT_SEED,
    )
    art.rho_star = cross_fit["rho_star"]
    calibration_check = cross_fit["calibration_check"]
    reference_oof_novelty_rate = float(np.mean(cross_fit["oof_is_novel"]))
    # Analyst-created classes are NOT ART categories and NOT prototype-head classes: they are separate
    # L2-normalized EXEMPLAR sets (one row per labeled chip, never averaged for the assignment decision
    # itself) consulted only for chips the trained ART categories reject (ADAPTATION_RULE's
    # nearest-exemplar rule). Fitting NEVER uses detector chips -- see _analyst_exemplars_from's own
    # docstring for how a det-sourced label's exemplars come from vigil.embed.get_det_embeddings
    # instead of a GT embedding lookup.
    analyst_exemplars = _analyst_exemplars_from(x_for, new_classes)
    art.fit_knn(ref_fit_X)

    # ---- GT-chip scoring (unchanged since before field mode: see _score_all) ----
    scored_all = _score_all(all_X, art, proto, analyst_exemplars)

    # Analyst-created class names aren't in proto.class_names (they were never added to the
    # prototype head -- see above), but are still surfaced in `classes` for transparency/UI display.
    analyst_class_names = sorted(analyst_exemplars.keys())
    all_class_names = proto.class_names + analyst_class_names

    # ---- true embedding-space neighbors: 5 nearest reference chips (cosine) for every scored chip --
    # (computed once inside _score_all -- column 0 also doubles as ADAPTATION_RULE's ref_1nn_dist) ----
    ref_fit_ann_ids = ref_fit["ann_id"].to_numpy()
    neighbor_ids_mat = ref_fit_ann_ids[scored_all["neighbor_idx"]]
    neighbor_dist_mat = scored_all["neighbor_dist"]

    # ---- UMAP (cached reducer, fit once on the reference set) ----
    reducer = _get_or_fit_umap(ref_fit_ann_ids, x_for)
    umap_xy = reducer.transform(all_X)

    scores = pd.DataFrame({
        "ann_id": chips["ann_id"].to_numpy(),
        "tile_id": chips["tile_id"].to_numpy(),
        "batch_id": np.where(chips["domain"].isin(REFERENCE_DOMAINS), "reference", chips["domain"]),
        "role_pred": scored_all["role_pred"],
        "role_conf": scored_all["role_conf"],
        "analyst_sim": scored_all["analyst_sim"],  # 1 - nearest-exemplar cosine distance (winning class); 0 if not analyst-claimed
        "analyst_margin": scored_all["analyst_margin"],  # ref_1nn_dist - nearest-exemplar distance (winning class); 0 if not analyst-claimed
        "novelty": scored_all["novelty"],
        "is_novel": scored_all["is_novel"],
        "art_category": pd.array(
            [None if c < 0 else int(c) for c in scored_all["art_category"]], dtype="Int64"
        ),
        "knn_dist": scored_all["knn_dist"],
        "umap_x": umap_xy[:, 0],
        "umap_y": umap_xy[:, 1],
        "used_in_fit": chips["ann_id"].isin(used_in_fit_gt).to_numpy(),
        "neighbor_ids": [row.tolist() for row in neighbor_ids_mat],
        "neighbor_dists": [row.tolist() for row in neighbor_dist_mat],
    })
    scores.to_parquet(SCORES_DIR / f"{version_id}.parquet", index=False)

    # ---- scene-level drift: whole-TILE embeddings, parent-image cross-fitted null (see vigil.drift) ----
    embed.embed_tiles()  # ensure every tile in tiles.parquet is embedded (no-op if the cache already covers all of them)
    tiles_full = load_tiles()
    tiles_dom = assign_domains(tiles_full)[["tile_id", "parent_id", "domain"]]
    ref_tile_mask = tiles_dom["domain"].isin(REFERENCE_DOMAINS).to_numpy()
    ref_tile_ids = tiles_dom.loc[ref_tile_mask, "tile_id"].to_numpy()
    ref_tile_parent_ids = tiles_dom.loc[ref_tile_mask, "parent_id"].to_numpy()
    ref_tile_X = embed.get_tile_embeddings(ref_tile_ids)
    oof_ref_tile_dist = drift_mod.cross_fit_reference_tile_distances(
        ref_tile_X, ref_tile_parent_ids, k_folds=CROSS_FIT_K_FOLDS, seed=CROSS_FIT_SEED,
    )

    # ---- per-batch summaries ----
    drift_by_batch = {}
    for batch_id in OPERATIONAL_BATCHES:
        tile_mask = (tiles_dom["domain"] == batch_id).to_numpy()
        if tile_mask.sum() > 0 and len(ref_tile_X) > 0:
            batch_tile_X = embed.get_tile_embeddings(tiles_dom.loc[tile_mask, "tile_id"].to_numpy())
            drift_by_batch[batch_id] = drift_mod.batch_scene_drift(
                batch_tile_X, ref_tile_X, oof_ref_tile_dist, ref_tile_parent_ids, seed=CROSS_FIT_SEED,
            )
        else:
            drift_by_batch[batch_id] = _empty_scene_drift()
    batches = _batch_summaries(chips, scores, tiles_dom, drift_by_batch)

    # ---- reference-set stats (not an operational batch; additive, for the UI's calibration story) ----
    batches["reference"] = {
        "reference_oof_novelty_rate": reference_oof_novelty_rate,
        "calibration_check": calibration_check,
        "reference_scene_oof_mean": float(np.mean(oof_ref_tile_dist)) if len(oof_ref_tile_dist) else float("nan"),
        "n_chips": int(len(ref_fit)),
        "n_parent_images": int(ref_fit["parent_id"].nunique()),
        "n_tiles": int(ref_tile_mask.sum()),
    }

    # ---- field mode: score every detector chip with the SAME fitted models (transform only) ----
    batches["det"] = _score_det_chips(
        version_id, art, proto, analyst_exemplars, ref_fit_ann_ids, reducer, used_in_fit_det, exclude_det,
        assign_min_conf,
    )

    (SCORES_DIR / f"{version_id}_batches.json").write_text(json.dumps(json_safe(batches), indent=2))

    analyst_class_label_counts = _analyst_class_label_counts(new_classes)

    # ---- model metadata ----
    model_info = {
        "vigilance_rho_star": art.rho_star,
        "rho_train": art.rho_train,
        "n_categories": len(art.categories),
        "classes": all_class_names,
        "tau": proto.tau,
        "tau_method": getattr(proto, "tau_method", None),
        "calibrated_on": (
            "parent-image 8-fold cross-fitting within the reference set (43 images, 3,942 chips); "
            "no operational, held-out, or military data used"
        ),
        "calibration_check": calibration_check,
        "reference_definition": (
            "ref_train ∪ ref_calib -- the fit set for PCA, min-max scaling, Fuzzy ART categories, "
            "role prototypes, role-head temperature (tau), the 2-D UMAP reducer, and the kNN index "
            "(chip-level baseline distance + true embedding-space neighbors)"
        ),
        "n_ref_chips": int(len(ref_fit)),
        "n_ref_images": int(ref_fit["parent_id"].nunique()),
        "embedder": EMBEDDER_DESC,
        "novelty_method": NOVELTY_METHOD_DESC,
        "adaptation_rule": ADAPTATION_RULE,
        "assign_min_conf": assign_min_conf,
        "art_train_stats": art_stats,
        "model_selection_note": (
            "PCA dimensionality (256) was chosen using military-vs-civil AUROC on operational "
            "batches, and the cross-fitted calibration was adopted after the control batch exposed "
            "split instability; treat those as model-selection decisions, not independent confirmation."
        ),
        "analyst_class_label_counts": analyst_class_label_counts,
        "build_mode": "full: PCA/min-max/Fuzzy ART/prototypes/tau/UMAP/kNN fit on the reference set, vigilance cross-fitted, scene drift computed",
    }
    (SCORES_DIR / f"{version_id}_model.json").write_text(json.dumps(json_safe(model_info), indent=2))

    version_entry = _version_entry(version_id, all_class_names, feedback, feedback_counts, note, assign_min_conf)

    # ---- evaluation: run BEFORE the version becomes "current" (_save_version_entry, below), so
    # versions.json never advances to a version_id whose eval_{version_id}.json doesn't exist yet --
    # a live request racing this build (current_version() just reads the last id in versions.json)
    # can no longer observe a current_version with a missing eval file. ----
    from vigil import evaluate as evaluate_mod  # local import: evaluate.py imports from this module at load time

    eval_result = evaluate_mod.evaluate(version_id)

    _save_version_entry(version_entry)

    return {
        "version_id": version_id,
        "n_chips": len(chips),
        "model": model_info,
        "batches": batches,
        "eval": eval_result,
        "seconds": time.time() - t0,
        "build_mode": "full",
    }


def main() -> None:
    """CLI entry point: builds version v1 with no feedback (always the full path)."""
    result = build_version("v1", feedback=None, note="initial build", full=True)
    print(f"built version v1: {result['n_chips']} chips scored in {result['seconds']:.1f}s")
    print(f"  classes: {result['model']['classes']}")
    print(f"  ART categories: {result['model']['n_categories']}, rho_train={result['model']['rho_train']}, "
          f"rho*={result['model']['vigilance_rho_star']}")
    det_block = result["batches"].get("det")
    if det_block:
        print(f"  det chips scored: {sum(v.get('n_det_chips', 0) for k, v in det_block.items() if k != 'reference')}")


if __name__ == "__main__":
    main()
