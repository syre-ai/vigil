"""Model health · retrain readiness -- `GET /api/health`.

An INDICATIVE, analyst-facing answer to "keep the current version, keep patching it with analyst
feedback, or schedule a retrain?", computed only from field-available evidence (the collections
list, det-chip reviews, the latest version's field block, the changes verification) plus the one
labeled set this story has (the ref_holdout control set's per-class accuracy in each version's
eval artifact). Never ground truth on new collections, never a benchmark batch name, never a
drift statistic -- same contract as the rest of the analyst view (see backend/API.md, Field mode).

Four indicators, each with its rule stated verbatim (`rule`) and the numbers behind it
(`evidence`); the recommendation is a fixed function of their statuses (RULE_RECOMMENDATION).
Every threshold is a module constant below and appears in the rule strings the UI shows.

Structure: `gather_inputs()` reads disk/sqlite state through the existing accessors;
`compute_health()` is a pure function of those inputs so every status branch can be tested with
hand-built inputs (backend/app/test_health.py) without rebuilding a fixture.
"""

from __future__ import annotations

import pandas as pd
from fastapi import APIRouter

from . import changes as changes_api
from . import collections as collections_api
from . import dataio, db

# config's own import already puts backend/ on sys.path (so `vigil...` is importable)
from vigil.grouping import group_id as _group_id  # noqa: E402  (stdlib-only module)

router = APIRouter(prefix="/api/health", tags=["health"])

NOTE = "indicative; computed from field-available evidence and the labeled control set"

# ---- thresholds (every one of these is restated in RULES below, which the UI shows verbatim) ----
RECENT_WINDOW = 30            # "the 30 most recent collections by date"
NOVELTY_WATCH = 0.10          # share of collections needing review
NOVELTY_RETRAIN = 0.25

DEMAND_RECURRING_MIN_COLLECTIONS = 2   # labeled in >= 2 collections -> recurring
DEMAND_RECURRING_MIN_CONFIRMED = 5     # or >= 5 verified-confirmed assignments across >= 2 collections
DEMAND_RETRAIN_MIN_LABELS = 20
DEMAND_RETRAIN_MAX_PRECISION = 0.60
DEMAND_MIN_VERIFIED = 10

DETQ_MIN_REVIEWED = 10
DETQ_WATCH_PRECISION = 0.60
DETQ_WATCH_BAD_BOX = 0.25
DETQ_RETRAIN_PRECISION = 0.40

PATCH_NOMINAL_PP = 1.0
PATCH_WATCH_PP = 3.0

STATUS_ORDER = {"nominal": 0, "watch": 1, "retrain": 2}

RULES = {
    "novelty_pressure": (
        f"Novelty pressure = share of new collections whose status is needs_review, over all collections "
        f"and over the {RECENT_WINDOW} most recent by collection date; status from the larger share: "
        f"watch if >= {NOVELTY_WATCH:.0%}, retrain if >= {NOVELTY_RETRAIN:.0%}."
    ),
    "new_class_demand": (
        f"New-class demand, per analyst class: emerging (labeled in 1 collection) is nominal; recurring "
        f"(labeled in >= {DEMAND_RECURRING_MIN_COLLECTIONS} collections, or >= {DEMAND_RECURRING_MIN_CONFIRMED} "
        f"verified-confirmed assignments across >= {DEMAND_RECURRING_MIN_COLLECTIONS} collections) is watch -- the "
        f"class exists in the field; the detector cannot learn it without retraining; retrain if recurring and "
        f"(>= {DEMAND_RETRAIN_MIN_LABELS} labels, or verified precision < {DEMAND_RETRAIN_MAX_PRECISION:.0%} with "
        f">= {DEMAND_MIN_VERIFIED} verified) -- labels growing faster than assignments means the representation "
        f"cannot carry the class."
    ),
    "detector_quality_field": (
        f"Detector quality in the field, from reviewed flagged detections in the latest version's field block: "
        f"field precision on flagged = (reviewed - not aircraft - bad box) / reviewed; bad-box rate = bad box / "
        f"reviewed. Fewer than {DETQ_MIN_REVIEWED} reviewed = insufficient evidence (nominal). Watch if precision "
        f"< {DETQ_WATCH_PRECISION:.0%} or bad-box rate >= {DETQ_WATCH_BAD_BOX:.0%}; retrain if precision < "
        f"{DETQ_RETRAIN_PRECISION:.0%}."
    ),
    "patch_safety": (
        f"Patch safety = largest per-class accuracy drop on the labeled control set, v1 -> latest and across any "
        f"single consecutive version step: nominal if <= {PATCH_NOMINAL_PP:.0f}pp, watch if <= {PATCH_WATCH_PP:.0f}pp, "
        f"retrain if > {PATCH_WATCH_PP:.0f}pp (patches are costing validated performance)."
    ),
}
RULE_RECOMMENDATION = (
    "Recommendation: keep if every indicator is nominal; patch if any indicator is watch and none is retrain; "
    "schedule retrain if any indicator is retrain. Indicative only -- an input to a retraining decision, not the decision."
)
LABELS = {
    "novelty_pressure": "Novelty pressure",
    "new_class_demand": "New-class demand",
    "detector_quality_field": "Detector quality (field)",
    "patch_safety": "Patch safety",
}


# ------------------------------------------------------------------ utils --

def _pct(v: float | None) -> str:
    return "n/a" if v is None else f"{100 * v:.1f}%"


def _worse(a: str, b: str) -> str:
    return a if STATUS_ORDER[a] >= STATUS_ORDER[b] else b


def _indicator(key: str, value, display: str, status: str, evidence: str, reason: str | None) -> dict:
    return {
        "key": key,
        "label": LABELS[key],
        "value": value,
        "display": display,
        "status": status,
        "rule": RULES[key],
        "evidence": evidence,
        "_reason": reason,  # popped by compute_health; one sentence, only for non-nominal indicators
    }


# ------------------------------------------------------------ indicators --

def novelty_pressure(groups: list[dict]) -> dict:
    """Share of new collections with status needs_review -- overall and among the RECENT_WINDOW most
    recent by date. Status from the larger of the two shares."""
    n = len(groups)
    if n == 0:
        return _indicator("novelty_pressure", None, "no collections", "nominal",
                          "No new collections yet, so there is no novelty pressure to measure.", None)
    n_review = sum(1 for g in groups if g.get("status") == "needs_review")
    share = n_review / n
    recent = sorted(groups, key=lambda g: g.get("date") or "", reverse=True)[:RECENT_WINDOW]
    n_recent = len(recent)
    n_recent_review = sum(1 for g in recent if g.get("status") == "needs_review")
    share_recent = n_recent_review / n_recent if n_recent else 0.0
    worst = max(share, share_recent)
    if worst >= NOVELTY_RETRAIN:
        status = "retrain"
    elif worst >= NOVELTY_WATCH:
        status = "watch"
    else:
        status = "nominal"
    which = "overall" if share >= share_recent else f"among the {n_recent} most recent"
    evidence = (
        f"{n_review} of {n} new collections need review ({_pct(share)}); {n_recent_review} of the {n_recent} most "
        f"recent by date ({_pct(share_recent)}). Status from the larger share ({_pct(worst)}, {which})."
    )
    reason = None
    if status != "nominal":
        bound = NOVELTY_RETRAIN if status == "retrain" else NOVELTY_WATCH
        reason = f"Novelty pressure: {_pct(worst)} of new collections {which} need review ({status} at >= {bound:.0%})."
    display = f"{_pct(share)} · last {n_recent}: {_pct(share_recent)}"
    return _indicator("novelty_pressure", worst, display, status, evidence, reason)


def new_class_demand(classes: list[dict]) -> dict:
    """Per analyst class: {label, n_labels, n_collections_labeled, n_assigned, n_verified, n_confirmed,
    n_confirmed_collections, verified_precision}. A class is emerging (1 collection) or recurring
    (labeled in >= 2 collections, or >= 5 verified-confirmed assignments across >= 2 collections)."""
    if not classes:
        return _indicator("new_class_demand", 0, "no analyst classes", "nominal",
                          "No analyst class has been created yet (only new-class labels create one).", None)
    status = "nominal"
    parts: list[str] = []
    n_recurring = 0
    reasons: list[str] = []
    for c in sorted(classes, key=lambda c: c["label"]):
        n_labels = int(c.get("n_labels") or 0)
        n_coll = int(c.get("n_collections_labeled") or 0)
        n_assigned = c.get("n_assigned")
        n_verified = int(c.get("n_verified") or 0)
        n_confirmed = int(c.get("n_confirmed") or 0)
        n_conf_coll = int(c.get("n_confirmed_collections") or 0)
        vp = c.get("verified_precision")
        recurring = (
            n_coll >= DEMAND_RECURRING_MIN_COLLECTIONS
            or (n_confirmed >= DEMAND_RECURRING_MIN_CONFIRMED and n_conf_coll >= DEMAND_RECURRING_MIN_COLLECTIONS)
        )
        cls_status = "nominal"
        why = ""
        if recurring:
            n_recurring += 1
            cls_status = "watch"
            why = "recurring"
            if n_labels >= DEMAND_RETRAIN_MIN_LABELS:
                cls_status = "retrain"
                why = f"recurring, {n_labels} labels (>= {DEMAND_RETRAIN_MIN_LABELS})"
            elif vp is not None and n_verified >= DEMAND_MIN_VERIFIED and vp < DEMAND_RETRAIN_MAX_PRECISION:
                cls_status = "retrain"
                why = f"recurring, verified precision {_pct(vp)} (< {DEMAND_RETRAIN_MAX_PRECISION:.0%}) on {n_verified} verified"
        else:
            why = "emerging"
        status = _worse(status, cls_status)
        assigned_txt = "n/a" if n_assigned is None else str(int(n_assigned))
        verified_txt = (
            f"{n_confirmed} of {n_verified} verified confirmed ({_pct(vp)}) across {n_conf_coll} collection{'s' if n_conf_coll != 1 else ''}"
            if n_verified else "none verified yet"
        )
        parts.append(
            f"{c['label']}: {n_labels} label{'s' if n_labels != 1 else ''} in {n_coll} collection{'s' if n_coll != 1 else ''}, "
            f"{assigned_txt} assigned, {verified_txt} -> {why}"
        )
        if cls_status != "nominal":
            reasons.append(f"{c['label']} is {why}")
    n = len(classes)
    display = f"{n} class{'es' if n != 1 else ''} · {n_recurring} recurring" if n_recurring else f"{n} class{'es' if n != 1 else ''} · emerging"
    reason = None
    if status != "nominal":
        tail = "the detector cannot learn it without retraining" if status == "watch" else "labels are outrunning what the representation can carry"
        reason = f"New-class demand: {'; '.join(reasons)} -- {tail}."
    return _indicator("new_class_demand", n_recurring, display, status, ". ".join(parts) + ".", reason)


def detector_quality_field(reviewed_flagged: dict | None) -> dict:
    """Field precision on flagged + bad-box rate from the latest field block's `reviewed_flagged`."""
    rf = reviewed_flagged or {}
    n = int(rf.get("n") or 0)
    if n < DETQ_MIN_REVIEWED:
        evidence = (
            f"{n} reviewed flagged detection{'s' if n != 1 else ''} in the latest version's field block -- fewer than "
            f"{DETQ_MIN_REVIEWED}, so insufficient evidence for a status (nominal by default)."
        )
        return _indicator("detector_quality_field", None, f"n={n} · insufficient evidence", "nominal", evidence, None)
    n_na = int(rf.get("n_not_aircraft") or 0)
    n_bb = int(rf.get("n_bad_box") or 0)
    precision = rf.get("field_precision_on_flagged")
    if precision is None:
        precision = (n - n_na - n_bb) / n
    bad_box_rate = n_bb / n
    if precision < DETQ_RETRAIN_PRECISION:
        status = "retrain"
    elif precision < DETQ_WATCH_PRECISION or bad_box_rate >= DETQ_WATCH_BAD_BOX:
        status = "watch"
    else:
        status = "nominal"
    evidence = (
        f"Of {n} reviewed flagged detections: {n_na} not aircraft, {n_bb} bad box -> field precision on flagged "
        f"{_pct(precision)}, bad-box rate {_pct(bad_box_rate)}. A field estimate from analyst reviews, refreshed when feedback is applied."
    )
    reason = None
    if status == "retrain":
        reason = f"Detector quality: field precision on flagged is {_pct(precision)} (retrain below {DETQ_RETRAIN_PRECISION:.0%})."
    elif status == "watch":
        bits = []
        if precision < DETQ_WATCH_PRECISION:
            bits.append(f"field precision on flagged {_pct(precision)} (watch below {DETQ_WATCH_PRECISION:.0%})")
        if bad_box_rate >= DETQ_WATCH_BAD_BOX:
            bits.append(f"bad-box rate {_pct(bad_box_rate)} (watch at >= {DETQ_WATCH_BAD_BOX:.0%})")
        reason = f"Detector quality: {' and '.join(bits)}."
    display = f"precision {_pct(precision)} · bad box {_pct(bad_box_rate)} (n={n})"
    return _indicator("detector_quality_field", precision, display, status, evidence, reason)


def patch_safety(controls: list[tuple[str, dict]]) -> dict:
    """`controls` = [(version, {cls: accuracy}), ...] in version order (v1 first). Worst per-class drop,
    v1 -> latest and across every consecutive step."""
    usable = [(v, pc) for v, pc in controls if pc]
    if len(usable) < 2:
        evidence = "Only one version has a labeled control-set check, so no patch has been applied to compare yet."
        return _indicator("patch_safety", None, "no patch yet", "nominal", evidence, None)
    pairs = [(usable[0], usable[-1])] + [(usable[i], usable[i + 1]) for i in range(len(usable) - 1)]
    worst = None  # (drop_pp, cls, a_version, b_version, a_acc, b_acc)
    for (va, pa), (vb, pb) in pairs:
        wr = dataio.worst_control_regression(
            {k: {"accuracy": v} for k, v in pa.items()}, {k: {"accuracy": v} for k, v in pb.items()}
        )
        if wr is None:
            continue
        drop_pp = 100.0 * wr["delta"]
        if worst is None or drop_pp > worst[0]:
            worst = (drop_pp, wr["cls"], va, vb, wr["a"], wr["b"])
    if worst is None:
        evidence = "No class is common to two versions' labeled control-set checks, so no regression can be measured."
        return _indicator("patch_safety", None, "no comparable classes", "nominal", evidence, None)
    drop_pp, cls, va, vb, a_acc, b_acc = worst
    drop_pp = max(drop_pp, 0.0)
    if drop_pp > PATCH_WATCH_PP:
        status = "retrain"
    elif drop_pp > PATCH_NOMINAL_PP:
        status = "watch"
    else:
        status = "nominal"
    first, last = usable[0][0], usable[-1][0]
    evidence = (
        f"Largest per-class drop on the labeled control set across {first} -> {last}: {cls} {_pct(a_acc)} -> {_pct(b_acc)} "
        f"({va} -> {vb}, -{drop_pp:.1f}pp)." if drop_pp > 0 else
        f"No per-class drop on the labeled control set across {first} -> {last} (worst change: {cls} {_pct(a_acc)} -> {_pct(b_acc)})."
    )
    reason = None
    if status != "nominal":
        reason = f"Patch safety: {cls} lost {drop_pp:.1f}pp on the labeled control set ({va} -> {vb}); patches are costing validated performance."
    display = f"worst -{drop_pp:.1f}pp ({cls})" if drop_pp > 0 else "no regression"
    return _indicator("patch_safety", drop_pp, display, status, evidence, reason)


# ------------------------------------------------------------- assembly --

def compute_health(version: str | None, groups: list[dict], classes: list[dict],
                   reviewed_flagged: dict | None, controls: list[tuple[str, dict]]) -> dict:
    """Pure: the `/api/health` response from already-gathered inputs (see gather_inputs)."""
    indicators = [
        novelty_pressure(groups),
        new_class_demand(classes),
        detector_quality_field(reviewed_flagged),
        patch_safety(controls),
    ]
    reasons = [ind.pop("_reason") for ind in indicators]
    reasons = [r for r in reasons if r]
    statuses = {ind["status"] for ind in indicators}
    if "retrain" in statuses:
        recommendation = "schedule_retrain"
    elif "watch" in statuses:
        recommendation = "patch"
    else:
        recommendation = "keep"
    return {
        "version": version,
        "recommendation": recommendation,
        "reasons": reasons,
        "indicators": indicators,
        "rules": [RULES[k] for k in ("novelty_pressure", "new_class_demand", "detector_quality_field", "patch_safety")] + [RULE_RECOMMENDATION],
        "note": NOTE,
    }


# --------------------------------------------------------------- inputs --

def _gt_chip_group_ids() -> dict[int, str]:
    """ann_id -> group_id for GT chips (a GT-chip new_class label counts toward the class's labeled
    collections too). Empty if the chip index / tiles aren't available."""
    try:
        chips = dataio.chips_index_df()[["ann_id", "tile_id"]]
        tiles = dataio.tiles_df()[["tile_id", "airfield", "collection_date"]]
    except (dataio.DataUnavailable, KeyError):
        return {}
    merged = chips.merge(tiles, on="tile_id", how="left")
    merged["date"] = pd.to_datetime(merged["collection_date"], errors="coerce", utc=True).dt.strftime("%Y-%m-%d")
    merged = merged.dropna(subset=["airfield", "date"])
    return {int(r.ann_id): _group_id(str(r.airfield), str(r.date)) for r in merged.itertuples(index=False)}


def _class_inputs(version: str | None, versions: list[str], field: dict | None) -> list[dict]:
    """Per analyst class: labels (both review sources), distinct labeled collections, latest field-block
    assignments, and the verification of the changed detections (latest vs previous version -- the same
    predicted_version-scoped counting as /versions/changes/summary, broken out per class)."""
    try:
        reviews_all = db.all_reviews()
    except Exception:
        reviews_all = []
    labels_from_reviews = {r["label"] for r in reviews_all if r.get("action") == "new_class" and r.get("label")}
    field_classes = {c["label"]: c for c in (field or {}).get("classes", []) if c.get("label")}
    labels = sorted(labels_from_reviews | set(field_classes))
    if not labels:
        return []

    det_groups: dict[int, str] = {}
    try:
        table, _ = collections_api._new_collections_table(version)
        if not table.empty:
            det_groups = dict(zip(table["det_id"].astype(int), table["group_id"]))
    except Exception:
        det_groups = {}
    gt_groups = _gt_chip_group_ids() if any(r.get("source") == "gt" and r.get("action") == "new_class" for r in reviews_all) else {}

    per: dict[str, dict] = {
        lab: {"label": lab, "n_labels": 0, "_coll": set(), "n_assigned": (field_classes.get(lab) or {}).get("n_assigned"),
              "n_verified": 0, "n_confirmed": 0, "_conf_coll": set(), "verified_precision": None}
        for lab in labels
    }
    for r in reviews_all:
        if r.get("action") != "new_class" or r.get("label") not in per:
            continue
        p = per[r["label"]]
        p["n_labels"] += 1
        gid = None
        try:
            if r.get("source") == "det":
                gid = det_groups.get(int(r["ann_id"]))
            else:
                gid = gt_groups.get(int(r["ann_id"]))
        except (TypeError, ValueError):
            gid = None
        if gid:
            p["_coll"].add(gid)

    # verification of changed detections, previous -> latest version, per class
    if version and len(versions) >= 2 and versions[-1] == version:
        prev = versions[-2]
        try:
            changed = changes_api._changed_table(prev, version)
        except Exception:
            changed = pd.DataFrame()
        if not changed.empty:
            changed = changed.assign(label_b=changed["role_pred_b_raw"].map(lambda v: dataio.role_label(v)))
            by_det = {int(r.det_id): (r.label_b, r.group_id) for r in changed[["det_id", "label_b", "group_id"]].itertuples(index=False)}
            try:
                det_reviews = db.all_reviews(source="det")
            except Exception:
                det_reviews = []
            for r in det_reviews:
                if r.get("predicted_version") != version:
                    continue
                hit = by_det.get(r.get("ann_id"))
                if not hit or hit[0] not in per:
                    continue
                p = per[hit[0]]
                action = r.get("action")
                if action in ("confirm", "correct", "not_aircraft", "bad_box"):
                    p["n_verified"] += 1
                    if action == "confirm":
                        p["n_confirmed"] += 1
                        if hit[1]:
                            p["_conf_coll"].add(hit[1])
    out = []
    for lab in labels:
        p = per[lab]
        out.append({
            "label": lab,
            "n_labels": p["n_labels"],
            "n_collections_labeled": len(p["_coll"]),
            "n_assigned": p["n_assigned"],
            "n_verified": p["n_verified"],
            "n_confirmed": p["n_confirmed"],
            "n_confirmed_collections": len(p["_conf_coll"]),
            "verified_precision": (p["n_confirmed"] / p["n_verified"]) if p["n_verified"] else None,
        })
    return out


def gather_inputs() -> dict:
    """Everything compute_health needs, read through the existing accessors; degrades (never raises)
    when a file is absent -- e.g. a version whose eval artifact isn't built yet."""
    try:
        versions = [v["id"] for v in dataio.versions_list()]
    except dataio.DataUnavailable:
        versions = []
    version = versions[-1] if versions else None
    try:
        groups, _table, _ref = collections_api._build_collections(version)
    except Exception:
        groups = []
    field = None
    if version:
        try:
            field = dataio.field_block(version)
        except Exception:
            field = None
    controls: list[tuple[str, dict]] = []
    for v in versions:
        try:
            fb = dataio.field_block(v)
        except Exception:
            fb = None
        ctl = (fb or {}).get("control") if fb else None
        pc = {p["cls"]: p["accuracy"] for p in (ctl or {}).get("per_class", []) if p.get("accuracy") is not None} if ctl else {}
        controls.append((v, pc))
    return {
        "version": version,
        "groups": groups,
        "classes": _class_inputs(version, versions, field),
        "reviewed_flagged": (field or {}).get("reviewed_flagged") if field else None,
        "controls": controls,
    }


def health() -> dict:
    return compute_health(**gather_inputs())


@router.get("")
def get_health():
    return health()
