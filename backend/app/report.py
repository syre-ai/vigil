"""Markdown assurance-evidence report for the current version.

Per API.md / CLAUDE.md vocabulary: this is an "assurance evidence
prototype", never "accredited"/"accreditation decision". Section tags use
NIST AI RMF function names (Map/Measure/Manage) where natural.
"""

from __future__ import annotations

from datetime import datetime, timezone

from . import collections as collections_api
from . import dataio
from .constants import BATCH_META, NOTE_EVAL, NOTE_LOCALIZATION


def _fmt(v, digits=3, pct=False):
    if v is None:
        return "n/a"
    try:
        if pct:
            return f"{100 * float(v):.1f}%"
        return f"{float(v):.{digits}f}"
    except (TypeError, ValueError):
        return str(v)


def _table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(lines)


def _pct_or_na(v) -> str:
    return _fmt(v, pct=True) if v is not None else "n/a"


def _p_str(p) -> str:
    if p is None:
        return "n/a"
    try:
        p = float(p)
    except (TypeError, ValueError):
        return "n/a"
    return "<0.001" if p < 0.001 else f"{p:.3f}"



def _model_health_section() -> list[str]:
    """"Model health · retrain readiness" -- the indicative keep / patch / schedule-retrain indicator
    (backend/app/health.py, GET /api/health): four indicators from field-available evidence plus the
    labeled control set, each with its rule and evidence. Never a decision; wording per API.md."""
    from . import health as health_api
    try:
        h = health_api.health()
    except Exception:
        return ["_Model health could not be computed for this version._"]
    label = {"keep": "Keep", "patch": "Patch", "schedule_retrain": "Schedule retrain"}[h["recommendation"]]
    out = ["### Model health · retrain readiness (indicative)"]
    out.append(f"**{label}** (version `{h['version']}`). " + (" ".join(h["reasons"]) if h["reasons"] else "Every indicator is nominal.") + f" _{h['note']}._")
    out.append(_table(
        ["Indicator", "Value", "Status", "Evidence"],
        [[i["label"], i["display"], i["status"], i["evidence"]] for i in h["indicators"]],
    ))
    out.append("Rules (as applied):")
    out.extend(f"- {r}" for r in h["rules"])
    return out


def _analyst_view_sections(version: str, versions: list[dict]) -> list[str]:
    """"New collections since delivery" -- the analyst-view (field mode) evidence: detector-chip
    collections grouped by airfield x collection date with their evidence-only status, what the
    analyst decided on detector chips, and (once an adapted version exists) the field compare --
    descriptive assignment counts plus the labeled control-set regression check. Wording rules
    (STATE.md / API.md): assignments are DESCRIPTIVE (never accuracy/recall), the control check is a
    "labeled control-set regression check" (never validation/accreditation), p-values are indicative.
    Never surfaces ground truth or benchmark batch names. Degrades to a one-line note if detector
    chips haven't been extracted/scored yet.
    """
    out: list[str] = []
    out.append("## New collections since delivery (analyst view)")
    out.extend(_model_health_section())
    try:
        groups, _frames, reference_flag_rate = collections_api._build_collections(version)
    except Exception:
        groups, reference_flag_rate = [], None
    if not groups:
        out.append(
            "_No detector-chip collections computed yet for this version (data/chips_det/index.parquet "
            "or the version's detector-chip scores are absent)._"
        )
        return out

    ref = collections_api._reference_block()
    n_groups = len(groups)
    n_det = sum(g["n_detections"] for g in groups)
    n_flag = sum(g["n_flagged"] for g in groups)
    by_status = {k: sum(1 for g in groups if g["status"] == k) for k in ("needs_review", "nominal", "insufficient_data")}
    out.append(
        "Field mode: what the detector found on imagery collected after delivery, scored on the "
        "detector's own boxes and grouped by airfield × collection date (UTC day). The only labeled "
        f"data is the delivered validation set ({ref['n_parents']} parent images / {ref['n_tiles']} "
        "tiles); no ground truth is used anywhere in this section. \"Flagged\" = a detection that "
        "fails the Fuzzy ART vigilance test against every category learned from that validation set. "
        f"Reference flag rate (ref_holdout detector chips): {_pct_or_na(reference_flag_rate)}."
    )
    out.append(f"Status rule: {collections_api.status_rule_for(reference_flag_rate)}")
    out.append(
        f"{n_groups} collections, {n_det} detections, {n_flag} flagged "
        f"({_pct_or_na(n_flag / n_det if n_det else None)}): **{by_status['needs_review']} need review**, "
        f"{by_status['nominal']} nominal, {by_status['insufficient_data']} with fewer than "
        f"{collections_api.MIN_DENOMINATOR} detections (insufficient data)."
    )
    review = sorted(
        [g for g in groups if g["status"] == "needs_review"],
        key=lambda g: (g["p_value"] if g["p_value"] is not None else 1.0, -g["n_flagged"]),
    )
    out.append("**Collections needing review** (sorted by indicative p-value)")
    if review:
        out.append(_table(
            ["Airfield", "Date", "Weather", "Detections", "Flagged", "Flag rate", "Excess vs reference",
             "p (indicative)", "Reviewed"],
            [[g["airfield"], g["date"], g.get("weather") or "n/a", g["n_detections"], g["n_flagged"],
              _pct_or_na(g["flag_rate"]),
              (f"{g['excess_flagged']:+.1f}" if g.get("excess_flagged") is not None else "n/a"),
              _p_str(g.get("p_value")), g["n_reviewed"]]
             for g in review],
        ))
    else:
        out.append("_No collection exceeds the reference flag rate at the stated threshold._")

    # --- analyst decisions on detector chips ---
    try:
        from . import db
        det_reviews = db.all_reviews(source="det")
    except Exception:
        det_reviews = []
    out.append("**Analyst decisions on detector chips**")
    if det_reviews:
        counts = {a: sum(1 for r in det_reviews if r["action"] == a)
                  for a in ("confirm", "new_class", "not_aircraft", "correct", "bad_box", "skip")}
        labels = sorted({r["label"] for r in det_reviews if r.get("label")})
        out.append(
            f"{len(det_reviews)} detector chips reviewed -- confirm {counts['confirm']}, new class "
            f"{counts['new_class']}, not aircraft {counts['not_aircraft']}, correct role {counts['correct']}, "
            f"bad box (localization error) {counts['bad_box']}, skip {counts['skip']}." +
            (f" Labels used: {', '.join(labels)}." if labels else "") +
            " `confirm`/`correct` are recorded as evidence and never alter the model; `not_aircraft` drops "
            "that detection from detector-chip scoring only (never a training negative); `bad_box` marks the "
            "detector's box as a localization error (fused or truncated aircraft) -- it stays scored but is "
            "never used to fit anything; only `new_class` fits an analyst class mean, consulted solely for "
            "detections the trained categories reject as novel."
        )
        try:
            rf = dataio.field_block(version)
        except Exception:
            rf = None
        reviewed_flagged = (rf or {}).get("reviewed_flagged") if rf else None
        if reviewed_flagged and reviewed_flagged.get("n"):
            out.append(
                f"Of {reviewed_flagged['n']} reviewed flagged detections: {reviewed_flagged['n_not_aircraft']} "
                f"not aircraft, {reviewed_flagged['n_bad_box']} bad box → field precision on flagged ≈ "
                f"{_pct_or_na(reviewed_flagged.get('field_precision_on_flagged'))}. A field estimate of "
                "detector precision on flagged detections, not an accreditation number."
            )
    else:
        out.append("_No detector-chip reviews recorded yet._")

    # --- field compare: first version -> this version ---
    base = versions[0]["id"] if versions else None
    if base and base != version:
        fc = dataio.field_compare(base, version)
        fa, fb = fc.get("a"), fc.get("b")
        out.append(f"**Field compare, `{base}` → `{version}`**")
        if fa and fb:
            out.append(
                f"Detector chips across new collections: {fb['n_detections']}; excluded from the counts: "
                f"{fb['n_excluded_labeled']} analyst-labeled detections and {fb['n_excluded_same_parent']} "
                f"other detections from the same parent images ({fb['n_eligible']} counted). "
                f"_{fb['note']}_ -- these are counts of detections an analyst class claimed, not accuracy; "
                "the field has no labels to grade against."
            )
            if fb["classes"]:
                out.append(_table(
                    ["Analyst class", "Labels used", "Detections assigned (descriptive)", "Assignment rate",
                     "Top collections"],
                    [[c["label"], c["n_labels_used"], c["n_assigned"], _pct_or_na(c["assignment_rate"]),
                      "; ".join(f"{g['airfield']} {g['date']} ({g['n_assigned']})" for g in c["by_group"][:5]) or "n/a"]
                     for c in fb["classes"]],
                ))
            else:
                out.append("_No analyst class in this version._")
            ca, cb = fa.get("control"), fb.get("control")
            if ca and cb:
                pa = {p["cls"]: p for p in ca["per_class"]}
                pb = {p["cls"]: p for p in cb["per_class"]}
                rows = []
                for cls in sorted(set(pa) | set(pb)):
                    a_acc = (pa.get(cls) or {}).get("accuracy")
                    b_acc = (pb.get(cls) or {}).get("accuracy")
                    delta = (b_acc - a_acc) if a_acc is not None and b_acc is not None else None
                    rows.append([cls, (pb.get(cls) or pa.get(cls) or {}).get("n", "n/a"), _pct_or_na(a_acc),
                                 _pct_or_na(b_acc), (f"{100 * delta:+.1f} pp" if delta is not None else "n/a")])
                wr = fc.get("worst_regression")
                worst = (f" Worst regression: {wr['cls']} ({-100 * wr['delta']:+.1f} pp)." if wr and wr.get("delta") is not None else "")
                out.append(
                    f"{cb['label'].capitalize()} (ref_holdout ground-truth chips, the delivered labeled set): "
                    f"overall accuracy {_pct_or_na(ca['overall'])} → {_pct_or_na(cb['overall'])}.{worst} "
                    "A regression check against the delivered labeled set -- not an accreditation decision."
                )
                out.append(_table(["Class", "n", f"`{base}`", f"`{version}`", "Δ"], rows))
        else:
            out.append("_One of the two versions predates field mode (no `field` block in its evaluation artifact)._")
    return out


def build_report(version: str) -> str:
    model = dataio.model_json(version)
    batches = dataio.batch_list(version)
    env = dataio.envelope(version)
    try:
        ev = dataio.eval_summary(version)
    except dataio.DataUnavailable:
        ev = {}
    try:
        ej_raw = dataio.eval_json(version)
    except dataio.DataUnavailable:
        ej_raw = {}
    try:
        bj_raw = dataio.batches_json(version)
    except dataio.DataUnavailable:
        bj_raw = {}
    reviews = None
    try:
        from . import db
        reviews = db.all_reviews()
    except Exception:
        reviews = []
    try:
        versions = dataio.versions_list()
    except dataio.DataUnavailable:
        versions = []

    generated = datetime.now(timezone.utc).isoformat(timespec="seconds")

    parts = []
    parts.append(f"# Vigil assurance evidence report -- version `{version}`")
    parts.append(f"_Generated {generated}. This is an **assurance evidence prototype**, "
                 f"not an accreditation decision._")

    # --- Purpose & scope --------------------------------------------------
    parts.append("## Purpose & scope (Map)")
    parts.append(
        "Vigil is a post-delivery assurance evidence prototype for an overhead-imagery "
        "aircraft role classifier. It reports embedding drift, novelty (Fuzzy ART "
        "vigilance), and role-head accuracy for a stream of operational batches formed "
        "from real RarePlanes collection metadata, and lets an analyst fold feedback "
        "back in without retraining the embedder. It is **not** an accreditation "
        "decision and does not certify fitness for any operational use; it is evidence "
        "an accreditation process could consume. " + NOTE_LOCALIZATION
    )

    # --- Model & reference domain ------------------------------------------
    parts.append("## Model & reference domain (Map)")
    model_rows = [
        ["Embedder", model.get("embedder", "n/a")],
        ["Novelty method", model.get("novelty_method", "n/a")],
        ["Vigilance (rho)", _fmt(dataio.model_vigilance(model))],
        ["Known categories", str(model.get("n_categories", "n/a"))],
        ["Classes", ", ".join(model.get("classes", []) or []) or "n/a"],
        ["Calibrated on", model.get("calibrated_on", "n/a")],
        ["Reference chips (n)", str(model.get("n_ref_chips", "n/a"))],
        ["Reference parent images (n)", str(model.get("n_ref_images", "n/a"))],
    ]
    parts.append(_table(["Field", "Value"], model_rows))
    parts.append(
        "Reference domain = ref_train ∪ ref_calib: clear skies, USA, off-nadir <=25 "
        "degrees, no military aircraft, WorldView-3 sensor only."
    )

    # --- Data card -----------------------------------------------------
    parts.append("## Data card")
    parts.append(
        "Source: RarePlanes (public), CC BY-SA 4.0, WorldView-3 real collects (this "
        "report's synthetic batch, if present, uses the RarePlanes synthetic renders "
        "separately -- see Limitations). Domain/batch assignment is decided at "
        "**parent-image granularity** (a whole WorldView-3 collect over one airfield), "
        "so tiles that are spatial neighbours never straddle a domain boundary."
    )
    batch_rows = [
        [b["id"], b["name"], b["n_parent_images"], b["n_tiles"], b["n_chips"]]
        for b in batches
    ]
    parts.append(_table(
        ["Batch", "Name", "Parent images", "Tiles", "Chips"], batch_rows
    ) if batch_rows else "_No operational batches computed yet for this version._")

    # --- New collections since delivery (analyst view / field mode) -------------
    parts.extend(_analyst_view_sections(version, versions))

    # --- Method ----------------------------------------------------------
    # Every line here is read from the version's own artifacts (model.json /
    # batches.json), not hard-coded -- vigil.pipeline is under concurrent
    # redesign, so keys are read defensively (.get, with a fallback line
    # when the key is missing) rather than assumed present.
    parts.append("## Method (Measure)")
    method_lines = [
        f"- **Embedder:** {model.get('embedder', 'n/a')}",
        "- **Role head:** nearest-prototype classifier over civil roles (cosine-softmax confidence)",
    ]
    novelty_method = model.get("novelty_method")
    calibrated_on = model.get("calibrated_on")
    if novelty_method:
        line = f"- **Novelty method:** {novelty_method}"
        if calibrated_on:
            line += f" -- calibrated on {calibrated_on}"
        method_lines.append(line + " -- a chip is 'novel' if it fails resonance against every known ART category")
    else:
        line = f"- **Novelty:** Fuzzy ART vigilance = {_fmt(dataio.model_vigilance(model))}"
        if calibrated_on:
            line += f", calibrated on `{calibrated_on}`"
        method_lines.append(line + " -- a chip is 'novel' if it fails resonance against every known ART category")

    cal_check = model.get("calibration_check")
    if cal_check:
        method_lines.append(
            "- **Calibration check** (out-of-fold resonance rate over the reference set, should "
            f"land near the target vigilance quantile): min {_fmt(cal_check.get('min'))}, "
            f"median {_fmt(cal_check.get('median'))}, max {_fmt(cal_check.get('max'))} across "
            f"{cal_check.get('n_parents', 'n/a')} parent images"
        )

    reference_definition = model.get("reference_definition")
    if reference_definition:
        method_lines.append(f"- **Reference definition:** {reference_definition}")

    model_selection_note = model.get("model_selection_note")
    if model_selection_note:
        method_lines.append(f"- **Model selection note:** {model_selection_note}")

    # Drift statistic/null_method/level/interpretation are the same across
    # every batch (a batch's own entry only differs in score/p_value/n), so
    # read them off whichever operational batch has them, in BATCH_META order.
    drift_meta: dict = {}
    for bid in sorted(BATCH_META, key=lambda k: BATCH_META[k]["order"]):
        d = (bj_raw.get(bid) or {}).get("drift") or {}
        if d.get("statistic"):
            drift_meta = d
            break
    if drift_meta:
        line = f"- **Drift statistic:** {drift_meta['statistic']}"
        if drift_meta.get("level"):
            line += f" (level: {drift_meta['level']})"
        method_lines.append(line)
        if drift_meta.get("null_method"):
            method_lines.append(f"- **Drift null method:** {drift_meta['null_method']}")
        if drift_meta.get("interpretation"):
            method_lines.append(f"- **Drift interpretation:** {drift_meta['interpretation']}")
    else:
        method_lines.append(
            f"- **Drift statistic:** mean 5-NN cosine distance to the reference set, with a "
            f"bootstrap null distribution (n_boot draws from `{calibrated_on or 'ref_calib'}`) "
            f"giving a p95 threshold and a p-value per batch"
        )
    parts.append("\n".join(method_lines))

    # --- Operational batches table -----------------------------------------
    parts.append("## Operational batches (Measure)")
    if batches:
        rows = []
        for b in batches:
            d = b["drift"]
            rows.append([
                b["id"], b["name"],
                _fmt(d.get("score")), _fmt(d.get("null_p95")), _fmt(d.get("p_value")),
                _fmt(b.get("novelty_rate"), pct=True),
                _fmt(b.get("role_accuracy"), pct=True),
            ])
        parts.append(_table(
            ["Batch", "Name", "Drift score", "Null p95", "p-value", "Novelty rate", "Role accuracy"],
            rows,
        ))
    else:
        parts.append("_No operational batches computed yet for this version._")

    # --- Tested envelope tables --------------------------------------------
    # Every bucket's status is computed purely from evidence against the
    # ref_holdout control batch -- a bucket whose label happens to match the
    # reference definition (Clear Skies / USA / off-nadir <=25) is never
    # exempted from the degradation checks; that's flagged separately below
    # as "In ref. definition?" so it can be read next to (not instead of) status.
    parts.append("## Tested envelope (Measure)")
    th = env.get("thresholds", {})
    control = th.get("control", {}) or {}
    parts.append(
        f"Status thresholds (evidence against the ref_holdout control batch, never bypassed by a "
        f"bucket's label matching the reference definition): untested if n_chips < {th.get('min_n')}; "
        f"degraded if role accuracy or detection recall falls more than "
        f"{_fmt(th.get('degraded_margin'), pct=True)} below the control's, or novelty rate exceeds "
        f"{th.get('novelty_multiplier', 'n/a')}x the control's; else tested. Control (ref_holdout, "
        f"n={control.get('n_chips', 'n/a')} chips / {control.get('n_parent_images', 'n/a')} parent images): "
        f"role accuracy {_fmt(control.get('role_accuracy'), pct=True)}, detection recall "
        f"{_fmt(control.get('detection_recall'), pct=True)}, novelty rate "
        f"{_fmt(control.get('novelty_rate'), pct=True)}."
    )
    for dim in env.get("dimensions", []):
        rows = [
            [b["label"], b["n_chips"], b["n_images"], b.get("n_parent_images", "n/a"),
             _fmt(b["novelty_rate"], pct=True), _fmt(b["role_accuracy"], pct=True),
             _fmt((b.get("control_delta") or {}).get("role_accuracy"), pct=True),
             _fmt(b.get("detection_recall"), pct=True),
             _fmt((b.get("control_delta") or {}).get("detection_recall"), pct=True),
             "yes" if b.get("in_reference_definition") else "no", b["status"]]
            for b in dim["buckets"]
        ]
        parts.append(f"**{dim['name']}**")
        parts.append(_table(
            ["Bucket", "Chips", "Images", "Parent images", "Novelty rate", "Role accuracy",
             "Δ role acc (control)", "Det. recall", "Δ det. recall (control)", "In ref. definition?", "Status"],
            rows,
        ) if rows else "_no data_")

    # --- Novelty evaluation -------------------------------------------------
    parts.append("## Novelty & role-head evaluation (Measure)")
    parts.append(f"_Note: {ev.get('note', NOTE_EVAL)}._")
    nov = ev.get("novelty", {})
    rh = ev.get("role_head", {})
    mil = ev.get("military", {})
    if nov or rh:
        eval_rows = [
            ["AUROC military vs civil", _fmt(nov.get("auroc_military_vs_civil"))],
            ["AUPRC", _fmt(nov.get("auprc"))],
            ["AUROC (wingspan-stratified control)", _fmt(nov.get("auroc_wingspan_stratified"))],
            ["Baseline kNN AUROC", _fmt(nov.get("baseline_knn_auroc"))],
            ["Role-head accuracy (ref_holdout)", _fmt(rh.get("accuracy_ref_holdout"), pct=True)],
            ["Analyst-class assignment rate (held-out military chips; label-agnostic, never 'recall')",
             _fmt(mil.get("analyst_class_assignment_rate", mil.get("rare_target_recall")), pct=True)],
            ["Civil chips wrongly assigned an analyst class (held-out)", _fmt(mil.get("civil_false_assignment_rate"), pct=True)],
            ["Miss rate (military predicted civil)", _fmt(mil.get("miss_rate_civil"), pct=True)],
        ]
        parts.append(_table(["Metric", "Value"], eval_rows))
        method = nov.get("wingspan_stratified_method")
        if method:
            parts.append(f"_Wingspan-stratified AUROC method: {method}._")
    else:
        parts.append("_Evaluation not computed yet for this version._")

    # --- Detector performance ----------------------------------------------
    parts.append("## Detector performance (Measure)")
    card = dataio.detector_card()
    metrics = dataio.detection_metrics()
    if card is None and metrics is None:
        parts.append("_Detector metrics not computed yet._")
    else:
        if card:
            card_rows = [
                ["Name", card.get("name", "n/a")],
                ["Trained on", card.get("trained_on", "n/a")],
                ["Validation split", card.get("val_split", "n/a")],
                ["Val mAP50", _fmt(card.get("val_map50"))],
                ["Val mAP50-95", _fmt(card.get("val_map50_95"))],
                ["Val precision", _fmt(card.get("val_precision"))],
                ["Val recall", _fmt(card.get("val_recall"))],
                ["Confidence threshold", _fmt(card.get("conf_threshold"))],
                ["IoU threshold", _fmt(card.get("iou_threshold"))],
            ]
            parts.append(_table(["Field", "Value"], card_rows))
            if card.get("note"):
                parts.append(f"_{card['note']}_")

        by_domain = (metrics or {}).get("by_domain", {}) or {}
        if by_domain:
            batch_rows = []
            for bid, bmeta in BATCH_META.items():
                d = by_domain.get(bid)
                if d is None:
                    continue
                batch_rows.append([
                    bid, bmeta["name"], _fmt(d.get("recall"), pct=True), _fmt(d.get("precision"), pct=True),
                    d.get("n_gt", "n/a"), d.get("n_det", "n/a"),
                ])
            parts.append(_table(["Batch", "Name", "Recall", "Precision", "n_gt", "n_det"], batch_rows))
        else:
            parts.append("_No per-batch detection metrics computed yet._")

        # by_role_group is military-batch-scoped (civil vs military recall within the
        # same batch, same collections for both groups); by_role_group_pooled (may be
        # absent -- older detection_metrics.json shape) pools civil recall across every
        # detection-evaluated batch instead, which confounds acquisition domain with role.
        det_eval = dataio.detection_eval_summary() or {}
        role_group = det_eval.get("by_role_group") or {}
        civil, mil = role_group.get("civil") or {}, role_group.get("military") or {}
        if civil.get("recall") is not None and mil.get("recall") is not None:
            line = (
                f"Military aircraft are detected less often than civil ones: recall "
                f"{_fmt(mil.get('recall'), pct=True)} on military aircraft (n={mil.get('n')}) vs. "
                f"{_fmt(civil.get('recall'), pct=True)} on civil aircraft (n={civil.get('n')})"
            )
            if role_group.get("scope"):
                line += f" ({role_group['scope']})"
            parts.append(line + ".")
            pooled = det_eval.get("by_role_group_pooled")
            pooled_civil = (pooled or {}).get("civil") or {}
            if pooled and pooled_civil.get("recall") is not None:
                pooled_line = (
                    "Pooled across every detection-evaluated batch instead (civil aircraft drawn "
                    f"from every batch, not just the military one), civil recall is "
                    f"{_fmt(pooled_civil.get('recall'), pct=True)} (n={pooled_civil.get('n')})"
                )
                if pooled.get("scope"):
                    pooled_line += f" -- {pooled['scope']}"
                parts.append(pooled_line + ".")
        snow = by_domain.get("snow")
        if snow is not None:
            ref_recall = (by_domain.get("ref_holdout") or {}).get("recall")
            snow_line = f"Detector recall on the snow batch is {_fmt(snow.get('recall'), pct=True)}"
            if ref_recall is not None:
                snow_line += f" (vs. {_fmt(ref_recall, pct=True)} on the reference-domain control)"
            parts.append(snow_line + " -- weather is the sharpest degradation this detector shows.")

    # --- Analyst feedback ----------------------------------------------------
    # Both `confirm` (agreement) and `correct` (disagreement -- assigns a known
    # class label) just mark a chip reviewed and NEVER alter the model, exactly
    # alike; only `new_class` (Rule B, gated by the vigilance test) fits
    # anything -- see vigil.pipeline.CONFIRM_POLICY_NOTE / CORRECT_POLICY_NOTE.
    parts.append("## Analyst feedback (Manage)")
    try:
        disagreement = dataio.analyst_disagreement(version)
    except Exception:
        disagreement = None
    if disagreement:
        parts.append(
            f"n reviewed: {disagreement['n_reviewed']} -- confirm {disagreement['n_confirm']}, "
            f"correct {disagreement['n_correct']}, new class {disagreement['n_new_class']}, "
            f"not aircraft {disagreement['n_not_aircraft']}, bad box {disagreement.get('n_bad_box', 0)}, "
            f"skip {disagreement['n_skip']}. "
            f"Disagreement rate (correct / (confirm + correct)): "
            f"{_fmt(disagreement.get('disagreement_rate'), pct=True)}. Both **confirm** and "
            "**correct** mark a chip reviewed and record the analyst's agreement or disagreement "
            "with the predicted role as evidence -- neither alters the model (no prototype/ART "
            "update); **bad_box** marks a localization error (the detector's box fuses two aircraft "
            "or truncates one) and likewise never alters the model or fits anything; only "
            "**new_class** feedback, gated by the vigilance test against every trained ART category, "
            "fits anything."
        )
        if disagreement["by_pair"]:
            parts.append("**Corrections, predicted role -> corrected to**")
            parts.append(_table(
                ["Predicted role", "Corrected to", "n"],
                [[p["from_role"], p["to_role"], p["n"]] for p in disagreement["by_pair"]],
            ))
        else:
            parts.append("_No corrections recorded yet._")
    else:
        parts.append("_No analyst feedback evidence computed yet for this version._")

    parts.append("**Feedback log** (most recent 50)")
    if reviews:
        rows = [[r["ann_id"], r["action"], r.get("label") or "", r["ts"], r.get("version_applied") or ""] for r in reviews[:50]]
        parts.append(_table(["ann_id", "action", "label", "timestamp", "applied in"], rows))
        if len(reviews) > 50:
            parts.append(f"_...and {len(reviews) - 50} more._")
    else:
        parts.append("_No analyst reviews logged yet._")

    # --- Version history -------------------------------------------------
    parts.append("## Version history (Manage)")
    if versions:
        rows = [
            [v["id"], v["created"], v.get("n_feedback", 0),
             v.get("n_confirm", 0), v.get("n_correct", 0), v.get("n_new_class", 0), v.get("n_not_aircraft", 0),
             v.get("n_bad_box", 0), v.get("note", "")]
            for v in versions
        ]
        parts.append(_table(
            ["Version", "Created", "n_feedback", "n_confirm", "n_correct", "n_new_class", "n_not_aircraft",
             "n_bad_box", "Note"], rows
        ))
    else:
        parts.append("_No versions recorded._")

    # --- Limitations -------------------------------------------------------
    parts.append("## Limitations (Manage)")
    exclusion_method = ej_raw.get("exclusion_method")
    parts.append(
        "- Military-role sample sizes are small (single digits to low hundreds across "
        "the whole public RarePlanes set); AUROC/recall estimates on that batch carry "
        "wide uncertainty.\n"
        "- All imagery is single-sensor (WorldView-3, pan-sharpened). No cross-sensor "
        "generalization evidence exists here.\n"
        "- The synthetic-render batch (if present) is an **external-domain stress "
        "test**, not evidence of temporal drift on an operational sensor -- it demonstrates "
        "novelty/drift detection against a domain gap, not a forecast of real sensor "
        "aging.\n"
        f"- {NOTE_LOCALIZATION}\n"
        "- The wingspan-stratified AUROC control only rules out trivial size "
        "discrimination; military positives still come only from the military batch, "
        "so acquisition-domain confounding (sensor look angle, airfield, processing "
        "chain) is not removed by this control.\n"
        "- ref_holdout's control behavior and the military-vs-civil AUROC were used "
        "during model selection (PCA dimensionality, calibration scheme); they are not "
        "independent confirmation of this prototype model's performance.\n"
        "- Drift p-values are descriptive/indicative, not a rigorous inferential test "
        "(chips/tiles within a parent image are correlated, and the bootstrap null is "
        "resampled at parent-image granularity to reflect that).\n"
        + (
            f"- {exclusion_method} (see `used_in_fit` / `exclusion_method` in the evaluation "
            "artifact).\n"
            if exclusion_method else
            "- Adaptation evaluation excludes chips used to fit a new prototype/ART category "
            "(`used_in_fit`); this version's evaluation artifact does not yet report an "
            "`exclusion_method`, so whether same-parent-image chips are also excluded is not "
            "stated here.\n"
        )
        + "- This report is generated evidence for a human accreditation process; it is "
        "not itself an accreditation and does not certify operational fitness."
    )

    return "\n\n".join(parts) + "\n"
