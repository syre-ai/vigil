"""Evaluation: AUROC/AUPRC of the novelty score (and a kNN-distance baseline) for military vs civil chips.

Also reports a wingspan-stratified control (`auroc_wingspan_stratified`, alias `auroc_size_matched`;
see WINGSPAN_STRATIFIED_METHOD for what it does and does not rule out), per-military-role AUROC,
role-head accuracy on ref_holdout, an honest, label-agnostic `military.analyst_class_assignment_rate`
(among held-out true-military chips, the fraction assigned an analyst-created class; `rare_target_recall`
is kept as an alias -- see `alias_note` -- and `recall_new_class` as a second alias, for compatibility),
a rare-target novelty FLAG rate (`novelty_flag_rate`), and the ref_holdout novelty rate (which should
land near 5% by construction of rho*, calibrated over the whole cross-fitted reference set). Every
metric here excludes chips used to fit a new prototype/ART category from analyst feedback
(`used_in_fit`) AND every other chip from the same parent image (see EXCLUSION_METHOD /
_adaptation_exclusion_mask) -- a parent-image-GROUP exclusion, not just a chip exclusion, since chips
from the same collect as a labeled one are visually correlated with it and would otherwise leak
information about what the model was just adapted to.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from vigil.data import load_annotations, load_tiles  # noqa: E402
from vigil.domains import assign_domains  # noqa: E402
from vigil.jsonutil import json_safe  # noqa: E402
from vigil.pipeline import ADAPTATION_RULE, OPERATIONAL_BATCHES, SCORES_DIR  # noqa: E402
from vigil.proto import KNOWN_ROLE_IDS, ROLE_NAMES  # noqa: E402

MILITARY_ROLE_MIN = 4
MILITARY_ROLE_IDS = [4, 5, 6, 7]
MILITARY_ROLE_NAMES = {
    4: "Military Transport/Utility/AWAC",
    5: "Military Bomber",
    6: "Military Fighter/Interceptor/Attack",
    7: "Military Trainer",
}


def _load_eval_table(version_id: str) -> pd.DataFrame:
    """Joins scores/{version}.parquet with ground-truth role_id/wingspan and each chip's parent_id
    (needed for the parent-group adaptation exclusion, see _adaptation_exclusion_mask), restricted to
    operational batches."""
    scores = pd.read_parquet(SCORES_DIR / f"{version_id}.parquet")
    anns = load_annotations()[["ann_id", "role_id", "wingspan"]]
    df = scores.merge(anns, on="ann_id", how="left")
    tiles = assign_domains(load_tiles())[["tile_id", "parent_id"]]
    df = df.merge(tiles, on="tile_id", how="left")
    df = df[df["batch_id"].isin(OPERATIONAL_BATCHES)].reset_index(drop=True)
    if "used_in_fit" not in df.columns:
        df["used_in_fit"] = False
    df["used_in_fit"] = df["used_in_fit"].fillna(False).astype(bool)
    return df


EXCLUSION_METHOD = "chips used to fit new classes and all chips from the same parent images are excluded from evaluation"


def det_used_in_fit_parents(version_id: str) -> set:
    """Parent ids of detector chips used to fit an analyst class for this version (field mode; see
    vigil.pipeline's `used_in_fit_det`) -- read straight from {version_id}_det.parquet's own
    `parent_id`/`used_in_fit` columns. Empty set if that file doesn't exist (a version built before,
    or without, det-chip scoring)."""
    path = SCORES_DIR / f"{version_id}_det.parquet"
    if not path.exists():
        return set()
    det = pd.read_parquet(path)
    if "used_in_fit" not in det.columns or "parent_id" not in det.columns:
        return set()  # older shape (pre-adaptation det scores): nothing was ever used_in_fit
    used = det[det["used_in_fit"].fillna(False).astype(bool)]
    return set(used["parent_id"].dropna().unique().tolist())


def gt_used_in_fit_parents(version_id: str) -> set:
    """Parent ids of GT chips used to fit an analyst class for this version -- read straight from
    {version_id}.parquet's own `used_in_fit` column (joined to tiles for parent_id). Self-contained
    (does its own small tiles join) so it works independent of _load_eval_table's operational-batch
    filter -- irrelevant here since only operational chips could ever be used_in_fit, but this avoids
    a hidden dependency on that filter for callers (e.g. vigil.field) that don't otherwise build that
    table."""
    path = SCORES_DIR / f"{version_id}.parquet"
    if not path.exists():
        return set()
    scores = pd.read_parquet(path)
    if "used_in_fit" not in scores.columns or "tile_id" not in scores.columns:
        return set()  # older shape (no adaptation columns yet): nothing was ever used_in_fit
    if not scores["used_in_fit"].fillna(False).astype(bool).any():
        return set()  # common case (no GT labels): skip the tiles join entirely
    scores = scores[["tile_id", "used_in_fit"]]
    tiles = assign_domains(load_tiles())[["tile_id", "parent_id"]]
    scores = scores.merge(tiles, on="tile_id", how="left")
    used = scores[scores["used_in_fit"].fillna(False).astype(bool)]
    return set(used["parent_id"].dropna().unique().tolist())


def labeled_parent_ids(version_id: str) -> set:
    """Parent ids of every chip -- GT or detector -- used to fit an analyst class for this version,
    pooling BOTH sources (field mode: an analyst labels detector chips; benchmark: an analyst labels
    GT chips; either can happen against the same version). This is the "same-parent exclusion ...
    parents of ALL labeled chips (either source)" guard shared by this module's own GT-eval exclusion
    (_adaptation_exclusion_mask) and vigil.field's field-mode assignment counts."""
    return gt_used_in_fit_parents(version_id) | det_used_in_fit_parents(version_id)


def _adaptation_exclusion_mask(
    df: pd.DataFrame, version_id: str | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parent-image-GROUP adaptation exclusion: analyst feedback (`used_in_fit`) can only ever label
    a handful of chips from a handful of parent images, but every OTHER chip from those same parent
    images is visually/acquisition-correlated with the labeled ones (same collect, same conditions) --
    evaluating the model on them would leak information about what it was just adapted to, even though
    their own embedding was never used to fit anything. So the exclusion is by PARENT IMAGE, not just
    by chip: excludes used_in_fit chips themselves, plus every chip (used_in_fit or not) that shares a
    parent_id with at least one used_in_fit chip -- GT OR DETECTOR (field mode: an analyst can label a
    detector chip whose parent image also has unlabeled GT chips in this same eval table; those GT
    chips would otherwise leak information about what the model was just adapted to, even though no GT
    chip in that parent image was itself used_in_fit). `version_id` (when given) pulls in detector-chip
    used_in_fit parents via det_used_in_fit_parents; omitted (None) reproduces the pre-field-mode,
    GT-only behavior (e.g. for callers with no det scoring yet).

    Returns (exclude_mask, used_in_fit_mask, same_parent_only_mask) -- the last is the additional chips
    excluded purely for sharing a parent image (not used_in_fit themselves), for reporting.
    """
    used_in_fit_mask = df["used_in_fit"].to_numpy()
    used_in_fit_parents = set(df.loc[used_in_fit_mask, "parent_id"].dropna().unique().tolist())
    if version_id is not None:
        used_in_fit_parents |= det_used_in_fit_parents(version_id)
    in_used_parent = df["parent_id"].isin(used_in_fit_parents).to_numpy()
    exclude_mask = used_in_fit_mask | in_used_parent
    same_parent_only_mask = in_used_parent & ~used_in_fit_mask
    return exclude_mask, used_in_fit_mask, same_parent_only_mask


def _auroc_auprc(y: np.ndarray, score: np.ndarray) -> dict:
    """Computes AUROC and AUPRC of `score` for predicting y=1, or nan if only one class is present."""
    if len(y) == 0 or len(np.unique(y)) < 2:
        return {"auroc": float("nan"), "auprc": float("nan"), "n": int(len(y))}
    return {"auroc": float(roc_auc_score(y, score)), "auprc": float(average_precision_score(y, score)), "n": int(len(y))}


WINGSPAN_STRATIFIED_METHOD = (
    "support-weighted mean of within-wingspan-quartile AUROCs (quartiles from civil chips); "
    "military positives come only from the military batch, so acquisition-domain confounding is "
    "not removed"
)


def _size_matched_auroc(df: pd.DataFrame, y: np.ndarray) -> dict:
    """AUROC of military-vs-civil novelty restricted to wingspan quartile bins (bins from civil chips), weighted mean.

    Also reported as `auroc_wingspan_stratified` (same value, see WINGSPAN_STRATIFIED_METHOD for the
    caveat this control does and does not address)."""
    civil_wingspan = df.loc[y == 0, "wingspan"].dropna()
    if len(civil_wingspan) < 8:
        return {"auroc": float("nan"), "bins": [], "note": "not enough civil wingspan data yet for quartile bins"}

    q = np.quantile(civil_wingspan, [0.25, 0.5, 0.75])
    # None stands in for an open-ended edge (JSON has no Infinity literal, unlike Python's json module).
    edges = [None, float(q[0]), float(q[1]), float(q[2]), None]
    wingspan = df["wingspan"].to_numpy()
    has_wingspan = df["wingspan"].notna().to_numpy()
    bin_idx = np.digitize(wingspan, q, right=True)  # 0..3

    bins = []
    for b in range(4):
        mask = has_wingspan & (bin_idx == b)
        yb = y[mask]
        n = int(mask.sum())
        if n == 0 or len(np.unique(yb)) < 2:
            bins.append({"bin": b, "range": [edges[b], edges[b + 1]], "n": n, "auroc": None})
            continue
        auroc = float(roc_auc_score(yb, df.loc[mask, "novelty"].to_numpy()))
        bins.append({"bin": b, "range": [edges[b], edges[b + 1]], "n": n, "auroc": auroc})

    valid = [b for b in bins if b["auroc"] is not None]
    total_n = sum(b["n"] for b in valid)
    weighted = float(sum(b["auroc"] * b["n"] for b in valid) / total_n) if valid and total_n else float("nan")
    return {"auroc": weighted, "bins": bins}


def _per_role_auroc(df: pd.DataFrame, y: np.ndarray) -> list[dict]:
    """AUROC of novelty for each individual military role_id (4-7) vs all civil chips, with support n."""
    civil_mask = y == 0
    role_id_arr = df["role_id"].to_numpy()
    out = []
    for role_id in MILITARY_ROLE_IDS:
        role_mask = role_id_arr == role_id
        n = int(role_mask.sum())
        name = MILITARY_ROLE_NAMES[role_id]
        if n == 0:
            out.append({"role_id": role_id, "role_name": name, "n": 0, "auroc": None})
            continue
        sub = role_mask | civil_mask
        yb = role_mask[sub].astype(int)
        res = _auroc_auprc(yb, df.loc[sub, "novelty"].to_numpy())
        out.append({
            "role_id": role_id, "role_name": name, "n": n,
            "auroc": res["auroc"] if not np.isnan(res["auroc"]) else None,
        })
    return out


CIVIL_ROLE_NAMES = set(ROLE_NAMES.values())  # {"Small Civil", "Medium Civil", "Large Civil"}


ALIAS_NOTE = "label-agnostic: counts assignment to any analyst-created class"


def _analyst_class_assignment_rate(df: pd.DataFrame) -> dict:
    """Primary metric (formerly `rare_target_recall`, kept as an alias -- see ALIAS_NOTE): among chips
    whose ground truth is military (role_id 4-7) and that were NOT used to fit any prototype/ART
    category (`used_in_fit`, already excluded from `df` by the caller), the fraction whose role_pred
    is an analyst-created class -- i.e. not one of the three original civil classes. Analyst classes
    are assigned ONLY via the ART gate (vigil.pipeline's ADAPTATION_RULE; the prototype head never has
    an analyst class added to it -- see vigil.pipeline / vigil.novelty), so "role_pred not civil" is
    exactly "the ART gate assigned this chip an analyst-created class"; there is no third option. Does
    not require the predicted label string to match the analyst's chosen label -- this counts
    assignment to ANY analyst-created class, not just the one that chip's true role would suggest.
    Also reports the complementary miss rate (predicted civil)."""
    mil_mask = df["role_id"].to_numpy() >= MILITARY_ROLE_MIN
    eligible = df[mil_mask]
    civil_eligible = df[~mil_mask]
    n_eval = int(len(eligible))
    n_eval_civil = int(len(civil_eligible))
    if n_eval == 0:
        return {"analyst_class_assignment_rate": float("nan"), "miss_rate_civil": float("nan"), "n_eval": 0,
                "civil_false_assignment_rate": float("nan"), "n_eval_civil": n_eval_civil}
    predicted_civil = eligible["role_pred"].isin(CIVIL_ROLE_NAMES)
    miss_rate_civil = float(predicted_civil.mean())
    analyst_class_assignment_rate = float(1.0 - miss_rate_civil)
    civil_false = float((~civil_eligible["role_pred"].isin(CIVIL_ROLE_NAMES)).mean()) if n_eval_civil else float("nan")
    by_role = []
    for rid, grp in eligible.groupby("role_id"):
        by_role.append({"role_id": int(rid), "role_name": MILITARY_ROLE_NAMES.get(int(rid), ROLE_NAMES.get(int(rid), str(rid))), "n": int(len(grp)),
                        "assignment_rate": float((~grp["role_pred"].isin(CIVIL_ROLE_NAMES)).mean())})
    return {
        "assignment_rate_by_role": by_role,
        "analyst_class_assignment_rate": analyst_class_assignment_rate,
        "miss_rate_civil": miss_rate_civil,
        "n_eval": n_eval,
        "civil_false_assignment_rate": civil_false,
        "n_eval_civil": n_eval_civil,
    }


def _novelty_flag_rate(df: pd.DataFrame) -> dict:
    """Fraction of chips flagged novel (is_novel, at the calibrated vigilance rho*) among true-military
    vs true-known-civil-class chips, over operational batches, excluding used_in_fit -- the "rare-target
    flag rate" the UI can show alongside `military.analyst_class_assignment_rate`. Also breaks it out per
    individual military role (role_id 4-7)."""
    role_id_arr = df["role_id"].to_numpy()
    is_novel = df["is_novel"].to_numpy()
    mil_mask = role_id_arr >= MILITARY_ROLE_MIN
    civ_mask = df["role_id"].isin(KNOWN_ROLE_IDS).to_numpy()
    n_military = int(mil_mask.sum())
    n_civil = int(civ_mask.sum())
    military_rate = float(is_novel[mil_mask].mean()) if n_military else float("nan")
    civil_rate = float(is_novel[civ_mask].mean()) if n_civil else float("nan")
    per_role = []
    for role_id in MILITARY_ROLE_IDS:
        rmask = role_id_arr == role_id
        n = int(rmask.sum())
        per_role.append({
            "role_id": role_id, "role_name": MILITARY_ROLE_NAMES[role_id], "n": n,
            "novelty_flag_rate": float(is_novel[rmask].mean()) if n else None,
        })
    return {
        "military": military_rate, "civil": civil_rate,
        "n_military": n_military, "n_civil": n_civil,
        "per_role": per_role,
    }


def _role_head_eval(df: pd.DataFrame) -> dict:
    """Role-head accuracy (overall + per class) on ref_holdout chips whose true role is a known civil class."""
    holdout = df[(df["batch_id"] == "ref_holdout") & df["role_id"].isin(KNOWN_ROLE_IDS)]
    if len(holdout) == 0:
        return {"accuracy_ref_holdout": float("nan"), "per_class": {}}
    true_names = holdout["role_id"].map(ROLE_NAMES)
    accuracy = float((holdout["role_pred"].to_numpy() == true_names.to_numpy()).mean())
    per_class = {}
    for name in ROLE_NAMES.values():
        mask = (true_names == name).to_numpy()
        n = int(mask.sum())
        acc = float((holdout.loc[mask, "role_pred"] == name).mean()) if n else float("nan")
        per_class[name] = {"accuracy": acc, "n": n}
    return {"accuracy_ref_holdout": accuracy, "per_class": per_class}


def evaluate(version_id: str) -> dict:
    """Computes novelty AUROC/AUPRC (military vs civil, + wingspan-stratified control + kNN baseline),
    role-head accuracy, and rare-target recall -- all EXCLUDING chips used to fit a new prototype/ART
    category from analyst feedback (`used_in_fit`) AND every other chip from the same parent image
    (see _adaptation_exclusion_mask), so analyst feedback can never leak into its own evaluation, even
    via a visually-correlated chip from the same collect that wasn't itself labeled."""
    df_all = _load_eval_table(version_id)
    exclude_mask, used_in_fit_mask, same_parent_only_mask = _adaptation_exclusion_mask(df_all, version_id)
    n_excluded_used_in_fit = int(used_in_fit_mask.sum())
    n_excluded_same_parent = int(same_parent_only_mask.sum())
    df = df_all[~exclude_mask].reset_index(drop=True)
    y = (df["role_id"].to_numpy() >= MILITARY_ROLE_MIN).astype(int)

    novelty_eval = _auroc_auprc(y, df["novelty"].to_numpy())
    knn_eval = _auroc_auprc(y, df["knn_dist"].to_numpy())
    size_matched = _size_matched_auroc(df, y)
    per_role = _per_role_auroc(df, y)
    role_head = _role_head_eval(df)
    analyst_assignment = _analyst_class_assignment_rate(df)
    novelty_flag_rate = _novelty_flag_rate(df)

    holdout = df[df["batch_id"] == "ref_holdout"]
    novelty_rate_holdout = float(holdout["is_novel"].mean()) if len(holdout) else float("nan")

    result = {
        "version": version_id,
        "n_chips_evaluated": int(len(df)),
        "n_excluded_used_in_fit": n_excluded_used_in_fit,
        "n_excluded_same_parent": n_excluded_same_parent,
        "exclusion_method": EXCLUSION_METHOD,
        "novelty": {
            "auroc_military_vs_civil": novelty_eval["auroc"],
            "auprc": novelty_eval["auprc"],
            "n": novelty_eval["n"],
            "auroc_size_matched": size_matched["auroc"],
            "auroc_wingspan_stratified": size_matched["auroc"],
            "wingspan_stratified_method": WINGSPAN_STRATIFIED_METHOD,
            "size_matched_bins": size_matched["bins"],
            "per_role": per_role,
            "baseline_knn_auroc": knn_eval["auroc"],
            "baseline_knn_auprc": knn_eval["auprc"],
        },
        "role_head": role_head,
        "military": {
            "analyst_class_assignment_rate": analyst_assignment["analyst_class_assignment_rate"],
            "civil_false_assignment_rate": analyst_assignment.get("civil_false_assignment_rate"),
            "n_eval_civil": analyst_assignment.get("n_eval_civil"),
            "assignment_rate_by_role": analyst_assignment.get("assignment_rate_by_role", []),
            "n_excluded_used_in_fit": n_excluded_used_in_fit,
            "n_excluded_same_parent": n_excluded_same_parent,
            "exclusion_method": EXCLUSION_METHOD,
            "adaptation_rule": ADAPTATION_RULE,  # see vigil.pipeline -- kept as one string, not duplicated here
            "rare_target_recall": analyst_assignment["analyst_class_assignment_rate"],  # alias
            "alias_note": ALIAS_NOTE,
            "recall_new_class": analyst_assignment["analyst_class_assignment_rate"],  # alias, kept for frontend compatibility
            "miss_rate_civil": analyst_assignment["miss_rate_civil"],
            "n_eval": analyst_assignment["n_eval"],
        },
        "novelty_flag_rate": novelty_flag_rate,
        "novelty_rate_ref_holdout": novelty_rate_holdout,
        "novelty_rate_ref_holdout_note": (
            "rho* is calibrated so ~5% of the cross-fitted reference set (ref_train ∪ ref_calib, "
            "out-of-fold) is flagged novel; ref_holdout is drawn from the same reference domain, so "
            "this rate should land near 5% too (by construction, not by evaluation)"
        ),
        "note": (
            "ground truth (role_id) used for evaluation only -- never for fitting or calibration; "
            f"{EXCLUSION_METHOD} (see used_in_fit / n_excluded_used_in_fit / n_excluded_same_parent / "
            "exclusion_method)"
        ),
    }

    # ---- field mode: descriptive assignment counts over detector chips from NEW collections, plus a
    # labeled control-set regression check (ref_holdout GT chips) -- see vigil.field. Local import: field
    # imports THIS module at its own top level (it reuses _load_eval_table/_adaptation_exclusion_mask/
    # _role_head_eval), so importing it up front here would be circular; by the time evaluate() runs,
    # this module has already finished loading, so the local import resolves cleanly. ----
    from vigil import field as field_mod  # noqa: E402 (local import: see comment above)

    field_result = field_mod.compute_field(version_id, control_role_head=role_head)
    result["field"] = field_result["field"]
    result["control"] = field_result["control"]

    (SCORES_DIR / f"eval_{version_id}.json").write_text(json.dumps(json_safe(result), indent=2))
    return result


def _fmt(x) -> str:
    """Formats a float as 3 decimals, or 'nan' for None/NaN."""
    return "nan" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def _print_summary(result: dict) -> None:
    """Prints a compact human-readable summary of the evaluation result."""
    n = result["novelty"]
    print(f"=== eval: {result['version']} (n={result['n_chips_evaluated']} chips in operational batches, "
          f"{result.get('n_excluded_used_in_fit', 0)} excluded as used_in_fit, "
          f"{result.get('n_excluded_same_parent', 0)} more excluded as same-parent-image) ===")
    print(f"novelty AUROC military-vs-civil: {_fmt(n['auroc_military_vs_civil'])}  AUPRC: {_fmt(n['auprc'])}  (n={n['n']})")
    print(f"  wingspan-stratified control AUROC: {_fmt(n['auroc_wingspan_stratified'])}")
    print(f"  baseline kNN-dist AUROC:       {_fmt(n['baseline_knn_auroc'])}  AUPRC: {_fmt(n['baseline_knn_auprc'])}")
    for r in n["per_role"]:
        print(f"    role_id={r['role_id']} ({r['role_name']}): n={r['n']} auroc={_fmt(r['auroc'])}")
    rh = result["role_head"]
    print(f"role-head accuracy on ref_holdout: {_fmt(rh['accuracy_ref_holdout'])}")
    mil = result["military"]
    print(f"analyst_class_assignment_rate: {_fmt(mil['analyst_class_assignment_rate'])}  "
          f"miss_rate_civil: {_fmt(mil['miss_rate_civil'])}  n_eval={mil['n_eval']}")
    nfr = result["novelty_flag_rate"]
    print(f"novelty_flag_rate: military={_fmt(nfr['military'])} (n={nfr['n_military']})  "
          f"civil={_fmt(nfr['civil'])} (n={nfr['n_civil']})")
    for cls, v in rh["per_class"].items():
        print(f"    {cls}: acc={_fmt(v['accuracy'])} n={v['n']}")
    print(f"novelty_rate on ref_holdout: {_fmt(result['novelty_rate_ref_holdout'])} (should be ~0.05 by construction)")


def main() -> None:
    """CLI entry point: evaluates v1 and prints a summary."""
    result = evaluate("v1")
    _print_summary(result)


if __name__ == "__main__":
    main()
