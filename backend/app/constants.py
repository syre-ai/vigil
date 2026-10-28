"""Static vocabulary that is *display metadata*, not a computed result.

Everything a user sees as a number (counts, rates, scores) is computed from
data/scores/*.parquet + data/*.parquet at request time -- see dataio.py.
This file only holds the names/descriptions/thresholds that make those
numbers legible, per API.md and CLAUDE.md ("no hard-coded results", but
labels/thresholds are fine).
"""

from __future__ import annotations

ROLE_NAMES: dict[int, str] = {
    1: "Small Civil Transport/Utility",
    2: "Medium Civil Transport/Utility",
    3: "Large Civil Transport/Utility",
    4: "Military Transport/Utility/AWAC",
    5: "Military Bomber",
    6: "Military Fighter/Interceptor/Attack",
    7: "Military Trainer",
}
ROLE_NAME_TO_ID: dict[str, int] = {v: k for k, v in ROLE_NAMES.items()}
CIVIL_ROLE_IDS = {1, 2, 3}
MILITARY_ROLE_IDS = {4, 5, 6, 7}

# backend/vigil/proto.py fits its civil role head on short labels
# ("Small Civil", "Medium Civil", "Large Civil"), while RarePlanes ground
# truth (annotations.parquet's `role` column) uses the full names above.
# Canonicalize role_pred / eval per_class keys through this alias map so
# role-accuracy comparisons actually match; unrecognized strings (e.g. a
# genuinely new class label an analyst typed in) pass through unchanged.
ROLE_ALIASES: dict[str, str] = {
    "Small Civil": "Small Civil Transport/Utility",
    "Medium Civil": "Medium Civil Transport/Utility",
    "Large Civil": "Large Civil Transport/Utility",
}

# Reference domain, per research/04-decision.md + backend/vigil/domains.py:
# ref_train ∪ ref_calib -- clear skies, USA, off-nadir <= 25deg, no military.
REFERENCE_DOMAINS = ("ref_train", "ref_calib")

# The 7 "operational batch" ids, in stream order. This is the only place in
# the backend that defines batch display copy; every number attached to a
# batch (counts, drift, novelty_rate, ...) is read from
# data/scores/{version}_batches.json, never hard-coded here.
BATCH_META: dict[str, dict] = {
    "ref_holdout": {
        "order": 0,
        "name": "Control (same domain as reference)",
        "description": (
            "Held-out parent images from the same real RarePlanes collection "
            "metadata as the reference domain (clear skies, USA, off-nadir "
            "<=25deg, no military aircraft), never used for fitting or "
            "calibration -- formed at parent-image granularity."
        ),
        "condition": {"weather": "Clear Skies", "country": "USA", "off_nadir": "<=25deg"},
    },
    "off_nadir": {
        "order": 1,
        "name": "High off-nadir (>25deg)",
        "description": (
            "Parent images whose real RarePlanes collection metadata records "
            "an off-nadir angle above 25 degrees, held out at parent-image "
            "granularity."
        ),
        "condition": {"off_nadir": ">25deg"},
    },
    "foreign": {
        "order": 2,
        "name": "New geography (outside USA)",
        "description": (
            "Parent images collected outside the USA, per real RarePlanes "
            "collection metadata, held out at parent-image granularity."
        ),
        "condition": {"country": "non-USA"},
    },
    "haze": {
        "order": 3,
        "name": "Cloud cover / haze",
        "description": (
            "Parent images whose real RarePlanes collection metadata records "
            "cloud cover or haze, held out at parent-image granularity."
        ),
        "condition": {"weather": "Cloud Cover or Haze"},
    },
    "snow": {
        "order": 4,
        "name": "Snow",
        "description": (
            "Parent images whose real RarePlanes collection metadata records "
            "snow cover, held out at parent-image granularity."
        ),
        "condition": {"weather": "Snow"},
    },
    "military": {
        "order": 5,
        "name": "Rare targets present (military airfields)",
        "description": (
            "Parent images containing at least one military aircraft "
            "(role_id 4-7) per real RarePlanes ground truth, held out at "
            "parent-image granularity -- the rare-target stress test."
        ),
        "condition": {"has_military": "true"},
    },
    "synthetic": {
        "order": 6,
        "name": "Synthetic renders (external-domain stress test)",
        "description": (
            "Synthetic RarePlanes renders standing in for a real sensor/"
            "domain gap. Formed from the same public RarePlanes collection "
            "at parent-image granularity, but the imagery itself is "
            "computer-generated, not a real WorldView-3 collect -- treat as "
            "an external-domain stress test, not evidence of temporal drift."
        ),
        "condition": {"source": "synthetic render"},
    },
}
BATCH_ORDER = {k: v["order"] for k, v in BATCH_META.items()}

# /envelope, /versions/compare thresholds. Status is computed purely from
# evidence against the ref_holdout "control batch" (never bypassed by a
# bucket's label matching the reference definition -- see
# dataio._dimension / Chip.in_reference_definition):
#   untested  if n_chips < MIN_N_CHIPS
#   degraded  if role_accuracy < control.role_accuracy - DEGRADED_ROLE_ACC_MARGIN
#          or detection_recall < control.detection_recall - DEGRADED_ROLE_ACC_MARGIN
#          or novelty_rate > DEGRADED_NOVELTY_MULTIPLIER * control.novelty_rate
#   tested    otherwise
MIN_N_CHIPS = 30
DEGRADED_ROLE_ACC_MARGIN = 0.10
DEGRADED_NOVELTY_MULTIPLIER = 3.0

# Aircraft-size buckets, meters wingspan (our own convention -- RarePlanes'
# own faa_wingspan_class isn't carried into annotations.parquet).
WINGSPAN_BUCKETS = [
    ("<15m", lambda w: w < 15),
    ("15-30m", lambda w: 15 <= w < 30),
    (">=30m", lambda w: w >= 30),
]

OFF_NADIR_BUCKETS = [
    ("<=15deg", lambda v: v <= 15),
    ("15-25deg", lambda v: 15 < v <= 25),
    (">25deg", lambda v: v > 25),
]

GSD_BUCKETS = [
    ("<0.35m", lambda v: v < 0.35),
    (">=0.35m", lambda v: v >= 0.35),
]

REVIEW_ACTIONS = {"confirm", "new_class", "not_aircraft", "skip", "correct", "bad_box"}

NOTE_EVAL = "ground truth used for evaluation only"

# GT-chip limitation (Codex Mode A blocker #3): role/novelty scores in this demo are computed on
# ground-truth RarePlanes aircraft boxes (oracle localization), never a detector's own output.
# Now that the detector milestone has landed, detector recall/precision are reported separately
# (see /eval.detection, /envelope's detection_recall, Batch.detection) rather than folded into
# the role/novelty numbers above. Surfaced on /summary.model.localization and in the report's
# Purpose & scope and Limitations sections (see report.py).
NOTE_LOCALIZATION = (
    "Role and novelty scores are computed on ground-truth aircraft boxes (oracle localization). "
    "Detector recall/precision are scored separately against those boxes at IoU ≥ 0.5."
)
