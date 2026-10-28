"""Field-mode compare numbers: descriptive assignment counts over detector chips from NEW
collections, plus a labeled control-set regression check on ref_holdout GT chips. See
research/08-field-mode-proposal.md and STATE.md's "Field mode" section.

"New collections" = detector chips whose domain is NOT ref_train/ref_calib/ref_holdout (i.e.
off_nadir, foreign, haze, snow, military) -- these are UNLABELED in the field-mode story (an analyst
reviews what the detector found on imagery collected after the labeled validation set, per
vigil.pipeline's module docstring). ref_holdout is the one domain with ground truth (the delivered
validation/control set), so it is reported separately as `control`, never mixed into the field
assignment counts.

Every number under `field` is DESCRIPTIVE: `n_assigned`/`assignment_rate` count detector chips whose
role_pred (after the field-mode-reviewed analyst classes were fit) IS a given analyst class -- there
is no ground truth for detector chips in field mode, so this is never called "accuracy" or "recall"
(see NOTE_DESCRIPTIVE). `control` is the one place with real ground truth (ref_holdout GT chips),
labeled "labeled control-set regression check", never "validation" -- a regression CHECK against a
known-good set, not an accreditation decision (see backend/API.md's vocabulary section).

Both blocks exclude the same two kinds of chip vigil.evaluate already excludes from its own GT-chip
evaluation: chips used to fit an analyst class (`used_in_fit`, either source) and every OTHER chip
from the same parent image (see vigil.evaluate.labeled_parent_ids) -- so an analyst's own labeled
detector chips (and anything visually correlated with them) never inflate their own assignment count,
and the control check never grades the model on chips it was just adapted to.

`field` also carries `reviewed_flagged` (see _reviewed_flagged_block): among det-chip reviews (any
action) on chips that WERE flagged (is_novel), how many the analyst marked `not_aircraft` (a false
positive) or `bad_box` (a localization error -- the box fuses two aircraft or truncates one) --
`field_precision_on_flagged` = the remaining fraction, a field estimate of detector precision on
flagged detections. Neither action ever fits or excludes anything from scoring (bad_box's chip stays
scored; only not_aircraft's is excluded, from det scoring only, never a training negative).

`field.labeled_collections` (see _labeled_collections_block) is a DISPLAY-ONLY list, one entry per
airfield x date collection that contains at least one analyst-labeled det chip: among that
collection's own FLAGGED det chips, `n_labeled` were the analyst's training exemplars, `n_bad_box`/
`n_not_aircraft` were reviewed as such, and of the rest (`n_flagged_unlabeled` -- confirm/skip/
unreviewed chips all count as "not labeled" here), `n_flagged_unlabeled_assigned` (broken down in
`per_class`) now carry an analyst-created class. These counts are NEVER folded into `classes`/
`n_assigned`/`assignment_rate` above -- a labeled collection's own re-sorted detections are excluded
from every other number in this module the same way `used_in_fit`/same-parent chips are.

`field.assign_min_conf` is this version's OPTIONAL additional detector-confidence gate on top of
ADAPTATION_RULE's nearest-exemplar rule (vigil.pipeline; 0 = off, the default), read straight from
model.json so the UI can show "assignment gate: conf >= X" next to these numbers.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from vigil import evaluate as evaluate_mod  # noqa: E402
from vigil.data import load_tiles  # noqa: E402
from vigil.grouping import group_id as _group_id  # noqa: E402
from vigil.pipeline import OPERATIONAL_BATCHES, SCORES_DIR  # noqa: E402

NEW_COLLECTION_DOMAINS = [b for b in OPERATIONAL_BATCHES if b != "ref_holdout"]
CONTROL_DOMAIN = "ref_holdout"
BY_GROUP_TOP_N = 10

NOTE_DESCRIPTIVE = "assignments are descriptive; not ground-truth accuracy"
CONTROL_LABEL = "labeled control-set regression check"


def _model_json(version_id: str) -> dict:
    path = SCORES_DIR / f"{version_id}_model.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _det_scores(version_id: str) -> pd.DataFrame | None:
    path = SCORES_DIR / f"{version_id}_det.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


def _det_reviews() -> list[dict]:
    """All source='det' reviews from the SQLite `reviews` table (backend/app/db.py). This is the one
    place vigil/field.py depends on backend/app -- needed to compute build-time evidence
    (`reviewed_flagged`, see _reviewed_flagged_block) about what the analyst found among flagged
    detector chips. `backend/` is already on sys.path (see this module's header). Degrades to []
    (never raises) if the db module is unavailable or the reviews table can't be read -- the field
    block must never fail a build over this."""
    try:
        from app import db  # local import: avoids a hard import-time dependency, mirrors dataio.py's own pattern
        return db.all_reviews(source="det")
    except Exception:
        return []


def _reviewed_flagged_block(det_current: pd.DataFrame | None) -> dict:
    """`reviewed_flagged`: {n, n_not_aircraft, n_bad_box, field_precision_on_flagged} -- a field
    estimate of detector precision on FLAGGED (is_novel) detections, from det-chip reviews of any
    action. `not_aircraft` marks a false-positive detection; `bad_box` marks a localization error
    (fused/truncated aircraft); everything else (confirm/correct/new_class/skip) counts as neither.
    `field_precision_on_flagged = (n - n_not_aircraft - n_bad_box) / n`, or None if n == 0.

    Flagged status is looked up in `det_current` (this version's own {version}_det.parquet, ALL
    domains) first; a det chip reviewed `not_aircraft` is EXCLUDED from det scoring starting with the
    version that feedback was applied to (see vigil.pipeline's `exclude_det`), so it can disappear
    from the current parquet entirely -- for those, this falls back to the review's own
    `predicted_version` (the version the analyst was actually looking at when they reviewed it,
    snapshotted by db.upsert_review), whose det parquet still has the chip's real is_novel. A review
    with no determinable flagged status either way (very old review, or its det parquet is gone) is
    excluded from `n` rather than guessed at.
    """
    reviews = _det_reviews()
    if not reviews:
        return {"n": 0, "n_not_aircraft": 0, "n_bad_box": 0, "field_precision_on_flagged": None}

    current_is_novel: dict[int, bool] = {}
    if det_current is not None and len(det_current) and {"det_id", "is_novel"} <= set(det_current.columns):
        current_is_novel = {
            int(k): bool(v) for k, v in zip(det_current["det_id"], det_current["is_novel"].fillna(False))
        }

    hist_cache: dict[str, dict[int, bool]] = {}

    def _hist_is_novel(pred_version: str) -> dict[int, bool]:
        if pred_version not in hist_cache:
            path = SCORES_DIR / f"{pred_version}_det.parquet"
            if path.exists():
                hdf = pd.read_parquet(path, columns=["det_id", "is_novel"])
                hist_cache[pred_version] = {
                    int(k): bool(v) for k, v in zip(hdf["det_id"], hdf["is_novel"].fillna(False))
                }
            else:
                hist_cache[pred_version] = {}
        return hist_cache[pred_version]

    n = n_not_aircraft = n_bad_box = 0
    for r in reviews:
        det_id = r.get("ann_id")
        if det_id is None:
            continue
        det_id = int(det_id)
        flagged = current_is_novel.get(det_id)
        if flagged is None:
            pred_version = r.get("predicted_version")
            if pred_version:
                flagged = _hist_is_novel(pred_version).get(det_id)
        if not flagged:
            continue
        n += 1
        action = r.get("action")
        if action == "not_aircraft":
            n_not_aircraft += 1
        elif action == "bad_box":
            n_bad_box += 1

    precision = ((n - n_not_aircraft - n_bad_box) / n) if n else None
    return {"n": n, "n_not_aircraft": n_not_aircraft, "n_bad_box": n_bad_box, "field_precision_on_flagged": precision}


def _det_chip_group_ids() -> dict[int, tuple[str, str, str]]:
    """det_id -> (group_id, airfield, date) for every detector chip ever EXTRACTED (the permanent
    data/chips_det/index.parquet + tiles.parquet -- never a version's own det scores), so a chip
    already excluded from the current version's det parquet (a reviewed `not_aircraft` chip -- see
    vigil.pipeline's `exclude_det`) still resolves to its collection. Local import of vigil.embed
    (mirrors _det_reviews's own local-import pattern) -- degrades to {} rather than raising if the
    det-chip index or tiles.parquet isn't available."""
    try:
        from vigil import embed  # local import: keeps this module's own import cheap for callers that never need it
        idx = embed.load_det_chip_index()[["det_id", "tile_id"]]
        tiles = load_tiles()[["tile_id", "airfield", "collection_date"]]
    except Exception:
        return {}
    merged = idx.merge(tiles, on="tile_id", how="left")
    merged["date"] = pd.to_datetime(merged["collection_date"], errors="coerce", utc=True).dt.strftime("%Y-%m-%d")
    out: dict[int, tuple[str, str, str]] = {}
    for r in merged.itertuples(index=False):
        if pd.isna(r.airfield) or pd.isna(r.date):
            continue
        out[int(r.det_id)] = (_group_id(r.airfield, r.date), str(r.airfield), str(r.date))
    return out


def _labeled_collections_block(det_current: pd.DataFrame | None, analyst_labels: list[str]) -> list[dict]:
    """DISPLAY-ONLY block (never folded into `classes`/`n_assigned`/`assignment_rate`): for every
    airfield x date collection (vigil.grouping) that contains at least one analyst-labeled
    (`new_class`) det chip, partitions that collection's FLAGGED (is_novel) det chips into four
    buckets -- labeled (`n_labeled`, the analyst's own training exemplars), `n_bad_box`,
    `n_not_aircraft`, and everything else still unlabeled (`n_flagged_unlabeled`: confirm/skip/
    unreviewed chips count as "not labeled" here, same as anywhere else in this module) -- then
    reports, among that last bucket, how many now carry ANY analyst-created class
    (`n_flagged_unlabeled_assigned`, broken down by label in `per_class`). Flagged status uses the
    same current-parquet-first/historical-fallback lookup as `_reviewed_flagged_block` (a
    `not_aircraft` chip is dropped from the CURRENT det parquet entirely -- see vigil.pipeline's
    `exclude_det` -- so its flagged-ness at review time has to come from its own
    `predicted_version`'s det parquet instead). Returns [] if there are no det reviews, no det-chip
    index to resolve collections from, or no collection has an analyst-labeled det chip at all.
    """
    reviews = _det_reviews()
    if not reviews:
        return []
    group_by_det = _det_chip_group_ids()
    if not group_by_det:
        return []
    analyst_label_set = set(analyst_labels)

    current_is_novel: dict[int, bool] = {}
    current_role_pred: dict[int, str] = {}
    if det_current is not None and len(det_current) and {"det_id", "is_novel"} <= set(det_current.columns):
        current_is_novel = {
            int(k): bool(v) for k, v in zip(det_current["det_id"], det_current["is_novel"].fillna(False))
        }
    if det_current is not None and len(det_current) and {"det_id", "role_pred"} <= set(det_current.columns):
        current_role_pred = {
            int(k): v for k, v in zip(det_current["det_id"], det_current["role_pred"]) if pd.notna(v)
        }

    hist_cache: dict[str, dict[int, bool]] = {}

    def _hist_is_novel(pred_version: str) -> dict[int, bool]:
        if pred_version not in hist_cache:
            path = SCORES_DIR / f"{pred_version}_det.parquet"
            if path.exists():
                hdf = pd.read_parquet(path, columns=["det_id", "is_novel"])
                hist_cache[pred_version] = {
                    int(k): bool(v) for k, v in zip(hdf["det_id"], hdf["is_novel"].fillna(False))
                }
            else:
                hist_cache[pred_version] = {}
        return hist_cache[pred_version]

    groups: dict[str, dict] = {}
    explained_det_ids: set[int] = set()  # chips already counted as labeled/bad_box/not_aircraft

    for r in reviews:
        det_id = r.get("ann_id")
        if det_id is None:
            continue
        det_id = int(det_id)
        gid = group_by_det.get(det_id)
        if gid is None:
            continue
        group_id, airfield, date = gid
        flagged = current_is_novel.get(det_id)
        if flagged is None:
            pred_version = r.get("predicted_version")
            if pred_version:
                flagged = _hist_is_novel(pred_version).get(det_id)
        if not flagged:
            continue
        g = groups.setdefault(group_id, {
            "group_id": group_id, "airfield": airfield, "date": date,
            "n_labeled": 0, "n_bad_box": 0, "n_not_aircraft": 0, "_labeled": False,
        })
        action = r.get("action")
        if action == "new_class" and r.get("label"):
            g["n_labeled"] += 1
            g["_labeled"] = True
            explained_det_ids.add(det_id)
        elif action == "bad_box":
            g["n_bad_box"] += 1
            g.setdefault("_bad_box_ids", set()).add(det_id)
            explained_det_ids.add(det_id)
        elif action == "not_aircraft":
            g["n_not_aircraft"] += 1
            explained_det_ids.add(det_id)
        # confirm/correct/skip: deliberately NOT added to explained_det_ids -- they count as "not
        # labeled" and fall into n_flagged_unlabeled below, same as a never-reviewed flagged chip.

    if not any(g["_labeled"] for g in groups.values()):
        return []

    remaining_by_group: dict[str, set[int]] = {gid: set() for gid, g in groups.items() if g["_labeled"]}
    if det_current is not None and len(det_current) and {"det_id", "is_novel"} <= set(det_current.columns):
        flagged_ids = det_current.loc[det_current["is_novel"].fillna(False), "det_id"].astype(int).tolist()
        for det_id in flagged_ids:
            if det_id in explained_det_ids:
                continue
            gid = group_by_det.get(det_id)
            if gid is None or gid[0] not in remaining_by_group:
                continue
            remaining_by_group[gid[0]].add(det_id)

    out = []
    for group_id, g in groups.items():
        if not g["_labeled"]:
            continue
        remaining = remaining_by_group.get(group_id, set())
        per_class: dict[str, int] = {}
        for det_id in remaining:
            label = current_role_pred.get(det_id)
            if label in analyst_label_set:
                per_class[label] = per_class.get(label, 0) + 1
        # bad-box detections stay scored (they are aircraft with a poor box); report how many of them
        # now carry an analyst class -- display only, like everything else in this block
        n_bad_box_assigned = sum(1 for det_id in g.get("_bad_box_ids", set()) if current_role_pred.get(det_id) in analyst_label_set)
        out.append({
            "group_id": group_id,
            "airfield": g["airfield"],
            "date": g["date"],
            "n_labeled": g["n_labeled"],
            "n_bad_box": g["n_bad_box"],
            "n_bad_box_assigned": n_bad_box_assigned,
            "n_not_aircraft": g["n_not_aircraft"],
            "n_flagged_unlabeled": len(remaining),
            "n_flagged_unlabeled_assigned": sum(per_class.values()),
            "per_class": per_class,
        })
    out.sort(key=lambda g: (g["airfield"], g["date"]))
    return out


def _class_counts(df: pd.DataFrame, labels: list[str]) -> dict[str, int]:
    """n_assigned per analyst-class label, over an already-eligible (unlabeled, non-same-parent)
    detector-chip subset `df`."""
    if df.empty:
        return {label: 0 for label in labels}
    return {label: int((df["role_pred"] == label).sum()) for label in labels}


def _by_group(eligible: pd.DataFrame, label: str) -> list[dict]:
    """Top BY_GROUP_TOP_N collections (airfield x collection date, the SAME group_id /api/collections
    serves -- vigil.grouping) by how many eligible detector chips were assigned `label`. Descriptive
    counts, same caveat as everything else under `field`. Empty if nothing was assigned or tiles
    metadata is unavailable."""
    if eligible.empty or "tile_id" not in eligible.columns:
        return []
    sub = eligible[eligible["role_pred"] == label]
    if sub.empty:
        return []
    try:
        tiles = load_tiles()[["tile_id", "airfield", "collection_date"]]
    except Exception:  # tiles.parquet missing/unbuildable: degrade, never fail the build
        return []
    sub = sub[["tile_id"]].merge(tiles, on="tile_id", how="left")
    sub["date"] = pd.to_datetime(sub["collection_date"], errors="coerce", utc=True).dt.strftime("%Y-%m-%d")
    sub = sub.dropna(subset=["airfield", "date"])
    if sub.empty:
        return []
    counts = sub.groupby(["airfield", "date"]).size().reset_index(name="n_assigned")
    counts = counts.sort_values(["n_assigned", "airfield", "date"], ascending=[False, True, True])
    return [
        {
            "group_id": _group_id(r.airfield, r.date),
            "airfield": str(r.airfield),
            "date": str(r.date),
            "n_assigned": int(r.n_assigned),
        }
        for r in counts.head(BY_GROUP_TOP_N).itertuples(index=False)
    ]


def _domain_block(df: pd.DataFrame, labeled_parents: set, analyst_labels: list[str]) -> dict:
    """One {n_detections, n_excluded_labeled, n_excluded_same_parent, n_eligible, classes, eligible} block, over
    detector-chip rows `df` (already restricted to one domain, or pooled over every new-collection
    domain -- see compute_field)."""
    n_detections = int(len(df))
    if n_detections == 0:
        used_in_fit = pd.Series(dtype=bool)
        in_labeled_parent = pd.Series(dtype=bool)
    else:
        used_in_fit = df["used_in_fit"].fillna(False).astype(bool)
        in_labeled_parent = df["parent_id"].isin(labeled_parents)
    n_excluded_labeled = int(used_in_fit.sum())
    same_parent_only = in_labeled_parent & ~used_in_fit
    n_excluded_same_parent = int(same_parent_only.sum())
    eligible = df[~(used_in_fit | in_labeled_parent)] if n_detections else df
    n_eligible = int(len(eligible))
    counts = _class_counts(eligible, analyst_labels)
    classes = [
        {
            "label": label,
            "n_assigned": n,
            "assignment_rate": (n / n_eligible) if n_eligible else None,
        }
        for label, n in counts.items()
    ]
    return {
        "n_detections": n_detections,
        "n_excluded_labeled": n_excluded_labeled,
        "n_excluded_same_parent": n_excluded_same_parent,
        "n_eligible": n_eligible,
        "classes": classes,
        "eligible": eligible,  # the eligible frame itself (for by_group); compute_field pops it, never serialized
    }


def compute_field(version_id: str, control_role_head: dict | None = None) -> dict:
    """Builds the `field` + `control` blocks folded into eval_{version_id}.json by
    vigil.evaluate.evaluate(). `control_role_head` lets evaluate() pass in its own already-computed
    ref_holdout role-head result (same exclusion-safe computation, avoids redoing it); when None
    (standalone use, e.g. from a script), this recomputes it itself from disk state alone, same as
    every other vigil.evaluate metric.

    Degrades to an all-zero-but-present `field` block (never an error) if {version_id}_det.parquet
    doesn't exist yet (a version built before det-chip scoring landed).
    """
    model = _model_json(version_id)
    analyst_label_counts = model.get("analyst_class_label_counts") or {}
    analyst_labels = sorted(analyst_label_counts.keys())

    det = _det_scores(version_id)
    labeled_parents = evaluate_mod.labeled_parent_ids(version_id)

    if det is not None and len(det):
        new_coll = det[det["batch_id"].isin(NEW_COLLECTION_DOMAINS)].reset_index(drop=True)
        present_domains = set(new_coll["batch_id"].unique().tolist())
    else:
        new_coll = pd.DataFrame(columns=["batch_id", "parent_id", "used_in_fit", "role_pred"])
        present_domains = set()

    overall = _domain_block(new_coll, labeled_parents, analyst_labels)
    eligible_overall = overall.pop("eligible")
    classes_with_labels_used = [
        {
            **cls,
            "n_labels_used": int((analyst_label_counts.get(cls["label"]) or {}).get("n_total", 0)),
            "by_group": _by_group(eligible_overall, cls["label"]),
        }
        for cls in overall["classes"]
    ]

    by_domain = {}
    for dom in NEW_COLLECTION_DOMAINS:
        if dom not in present_domains:
            continue
        sub = new_coll[new_coll["batch_id"] == dom]
        block = _domain_block(sub, labeled_parents, analyst_labels)
        block.pop("eligible")
        block["classes"] = [
            {**cls, "n_labels_used": int((analyst_label_counts.get(cls["label"]) or {}).get("n_total", 0))}
            for cls in block["classes"]
        ]
        by_domain[dom] = block

    field = {
        # `classes[].by_group` = top collections (airfield x date, /api/collections' own group_id) by
        # n_assigned -- the analyst-view compare shows THIS, never `domains`/`by_domain` (benchmark
        # batch names an analyst never sees; kept here for the eval artifact / Benchmark view only).
        "domains": NEW_COLLECTION_DOMAINS,
        "n_detections": overall["n_detections"],
        "n_excluded_labeled": overall["n_excluded_labeled"],
        "n_excluded_same_parent": overall["n_excluded_same_parent"],
        "n_eligible": overall["n_eligible"],
        "classes": classes_with_labels_used,
        "by_domain": by_domain,
        "reviewed_flagged": _reviewed_flagged_block(det),
        # DISPLAY ONLY -- see _labeled_collections_block; never folded into `classes`/n_assigned above.
        "labeled_collections": _labeled_collections_block(det, analyst_labels),
        "assign_min_conf": model.get("assign_min_conf"),
        "note": NOTE_DESCRIPTIVE,
    }

    # ---- control: labeled control-set regression check (ref_holdout GT chips, existing role-head numbers) ----
    if control_role_head is None:
        df_all = evaluate_mod._load_eval_table(version_id)
        exclude_mask, _, _ = evaluate_mod._adaptation_exclusion_mask(df_all, version_id)
        df = df_all[~exclude_mask].reset_index(drop=True)
        control_role_head = evaluate_mod._role_head_eval(df)

    control = {
        "label": CONTROL_LABEL,
        "accuracy_ref_holdout": control_role_head.get("accuracy_ref_holdout"),
        "per_class": control_role_head.get("per_class", {}),
    }

    return {"field": field, "control": control}
