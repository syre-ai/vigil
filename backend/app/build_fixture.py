"""Builds a small, fully self-contained synthetic fixture under
data/scores_fixture/ so backend/app/test_api.py can exercise every endpoint
without depending on the (concurrently-being-built) real data pipeline.

Layout produced (mirrors the real schema in backend/API.md):
    <root>/rareplanes/tiles.parquet
    <root>/rareplanes/tiles_png/*.png        (512x512 tile images)
    <root>/chips/index.parquet
    <root>/chips/images/*.png                (224x224 chip images)
    <root>/rareplanes/annotations.parquet
    <root>/scores/versions.json
    <root>/scores/v1.parquet, v1_batches.json, v1_model.json, eval_v1.json
    <root>/scores/v2.parquet, v2_batches.json, v2_model.json, eval_v2.json

Run directly to (re)build: `.venv/bin/python -m app.build_fixture`
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROLE_NAMES = {
    1: "Small Civil Transport/Utility",
    2: "Medium Civil Transport/Utility",
    3: "Large Civil Transport/Utility",
    4: "Military Transport/Utility/AWAC",
    5: "Military Bomber",
    6: "Military Fighter/Interceptor/Attack",
    7: "Military Trainer",
}
CIVIL_ROLES = [1, 2, 3]
BATCH_ORDER = {
    "ref_holdout": 0, "off_nadir": 1, "foreign": 2, "haze": 3,
    "snow": 4, "military": 5, "synthetic": 6,
}
REFERENCE_DOMAINS = {"ref_train", "ref_calib"}
OPERATIONAL_DOMAINS = list(BATCH_ORDER.keys())

WINGSPAN_M = {1: 12.0, 2: 20.0, 3: 35.0, 4: 28.0, 5: 45.0, 6: 14.0, 7: 11.0}

# domain -> generation config
PLAN = {
    "ref_train":   dict(n_parents=2, tiles_per_parent=3, chips_per_tile=4, roles=CIVIL_ROLES, weather="Clear Skies", country="USA", off_nadir=10.0, gsd=0.30),
    "ref_calib":   dict(n_parents=1, tiles_per_parent=3, chips_per_tile=4, roles=CIVIL_ROLES, weather="Clear Skies", country="USA", off_nadir=12.0, gsd=0.31),
    "ref_holdout": dict(n_parents=2, tiles_per_parent=3, chips_per_tile=6, roles=CIVIL_ROLES, weather="Clear Skies", country="USA", off_nadir=14.0, gsd=0.32),
    "off_nadir":   dict(n_parents=1, tiles_per_parent=2, chips_per_tile=5, roles=CIVIL_ROLES, weather="Clear Skies", country="USA", off_nadir=32.0, gsd=0.33),
    "foreign":     dict(n_parents=2, tiles_per_parent=2, chips_per_tile=8, roles=CIVIL_ROLES, weather="Clear Skies", country="Germany", off_nadir=15.0, gsd=0.34),
    "haze":        dict(n_parents=2, tiles_per_parent=2, chips_per_tile=8, roles=CIVIL_ROLES, weather="Cloud Cover or Haze", country="USA", off_nadir=16.0, gsd=0.35),
    "snow":        dict(n_parents=2, tiles_per_parent=2, chips_per_tile=8, roles=CIVIL_ROLES, weather="Snow", country="USA", off_nadir=18.0, gsd=0.36),
    "military":    dict(n_parents=2, tiles_per_parent=2, chips_per_tile=8, roles=[1, 2, 6], weather="Clear Skies", country="USA", off_nadir=17.0, gsd=0.37),
    "synthetic":   dict(n_parents=2, tiles_per_parent=2, chips_per_tile=8, roles=CIVIL_ROLES, weather="Clear Skies", country="USA", off_nadir=19.0, gsd=0.38),
}

# Domain-level base "novelty" level (higher = more out-of-reference), used
# only to synthesize plausible-looking scores -- not a claim about the real
# model's behaviour.
NOVELTY_BASE = {
    "ref_train": 0.05, "ref_calib": 0.08, "ref_holdout": 0.10,
    "off_nadir": 0.35, "foreign": 0.40, "haze": 0.30, "snow": 0.28,
    "military": 0.75, "synthetic": 0.60,
}
NOVELTY_THRESHOLD = 0.5

# Detector-milestone fixture: mirrors the real data/scores/{detector_model.json,
# detection_metrics.json, detections.parquet} (all *not* per-version -- the
# detector is trained once, not re-fit by /adapt). Deliberately excludes
# "synthetic" (that milestone hasn't landed on the real detector either) and
# the reference domains (the detector isn't evaluated against them) -- this
# exercises the "detection data absent for this batch" null-degradation path
# for real, using the same batch id the real pipeline currently omits.
DETECTION_DOMAINS = ["ref_holdout", "off_nadir", "foreign", "haze", "snow", "military"]


def _detect_prob(role_id: int, domain: str) -> float:
    """Fixture-only stand-in for "probability the detector finds this GT box"
    -- civil aircraft detected more often than military (mirrors the real
    detector's recall_by_role_group), and snow depresses recall further
    (mirrors the real detector's worst weather condition).
    """
    base = 0.92 if role_id in CIVIL_ROLES else 0.65
    if domain == "snow":
        base *= 0.5
    return base


def _off_nadir_bucket(v: float) -> str:
    if v <= 15:
        return "<=15"
    if v <= 25:
        return "15-25"
    return ">25"


def _gsd_bucket(v: float) -> str:
    return "<0.35" if v < 0.35 else ">=0.35"


def _country_bucket(c: str) -> str:
    return "USA" if c == "USA" else "other"


def _solid_png(path: Path, size: int, color: tuple[int, int, int]) -> None:
    Image.new("RGB", (size, size), color=color).save(path)


def _build_tables(rng: np.random.RandomState, tiles_img_dir: Path, chips_img_dir: Path):
    tile_rows, ann_rows, chip_rows = [], [], []
    tile_id = 0
    ann_id = 0

    for domain, cfg in PLAN.items():
        for p in range(cfg["n_parents"]):
            parent_id = f"{domain}_{p}"
            lat = 35.0 + rng.uniform(-4, 4)
            lon = -100.0 + rng.uniform(-4, 4)
            for t in range(cfg["tiles_per_parent"]):
                file_name = f"{parent_id}_tile_{t}.png"
                img_path = tiles_img_dir / file_name
                color = tuple(int(v) for v in rng.randint(0, 255, size=3))
                _solid_png(img_path, 512, color)

                chip_role_ids = [int(rng.choice(cfg["roles"])) for _ in range(cfg["chips_per_tile"])]
                n_military = sum(1 for r in chip_role_ids if r >= 4)

                tile_rows.append(dict(
                    tile_id=tile_id, file_name=file_name, image_path=str(img_path),
                    coco_split="train", parent_id=parent_id,
                    loc_id=abs(hash(parent_id)) % 100000, cat_id=parent_id,
                    country=cfg["country"], state="NA", airfield=f"{parent_id} field",
                    weather=cfg["weather"], off_nadir=float(cfg["off_nadir"] + rng.uniform(-1, 1)),
                    gsd=float(cfg["gsd"]), sun_elev=45.0,
                    cloud_cover=(60 if "Haze" in cfg["weather"] else 5),
                    # Field-mode fixture (backend/app/collections.py): distinct
                    # dates per domain/parent so /api/collections has more than
                    # one group to sort -- reference domains (not in
                    # BATCH_ORDER) collapse to the same base date, which is
                    # fine since they're excluded from collections grouping.
                    collection_date=pd.Timestamp("2020-01-01") + pd.Timedelta(days=BATCH_ORDER.get(domain, -1) * 5 + p),
                    lat=lat, lon=lon,
                    n_aircraft=cfg["chips_per_tile"], n_military=n_military,
                    roles=list(chip_role_ids), has_military=(n_military > 0),
                    domain=domain, batch_order=BATCH_ORDER.get(domain, -1),
                ))

                for role_id in chip_role_ids:
                    x, y = float(rng.uniform(0, 400)), float(rng.uniform(0, 400))
                    w, h = 80.0, 60.0
                    wingspan = WINGSPAN_M[role_id] * float(rng.uniform(0.9, 1.1))
                    # Mirrors the real annotations.parquet: ~28% truncated (an
                    # aircraft cut off by the tile edge), and partial (fraction
                    # of the aircraft's true extent inside the tile) is always
                    # 1.0 when not truncated, else a value below 1 -- some of
                    # which fall under 0.75 so /batches/{id}/chips's,
                    # /review/queue's and /batches/{id}'s top_novel's default
                    # min_partial=0.75 filter has something real to exclude.
                    is_truncated = bool(rng.uniform() < 0.28)
                    partial = float(rng.uniform(0.30, 0.95)) if is_truncated else 1.0
                    ann_rows.append(dict(
                        ann_id=ann_id, tile_id=tile_id, file_name=file_name,
                        x=x, y=y, w=w, h=h,
                        role=ROLE_NAMES[role_id], role_id=role_id,
                        wingspan=wingspan, length=wingspan * 1.1,
                        truncated=is_truncated, partial=partial,
                    ))
                    chip_path = chips_img_dir / f"{ann_id}.png"
                    ccolor = tuple(int(v) for v in rng.randint(0, 255, size=3))
                    _solid_png(chip_path, 224, ccolor)
                    chip_rows.append(dict(
                        ann_id=ann_id, tile_id=tile_id, domain=domain,
                        role_id=role_id, path=str(chip_path),
                    ))
                    ann_id += 1
                tile_id += 1

    tiles = pd.DataFrame(tile_rows)
    ann = pd.DataFrame(ann_rows)
    chips = pd.DataFrame(chip_rows)
    return tiles, ann, chips


def _score_row(rng: np.random.RandomState, ann_row: dict, domain: str, extra_class_role_id: int | None) -> dict:
    role_id = ann_row["role_id"]
    novelty_base = NOVELTY_BASE[domain]
    novelty = float(np.clip(rng.normal(novelty_base, 0.08), 0.0, 1.0))

    known_roles = set(CIVIL_ROLES)
    if extra_class_role_id is not None:
        known_roles.add(extra_class_role_id)

    # Bias: unknown roles (rare targets not yet in the prototype set) are
    # usually flagged novel; known roles are novel only in the score's tail.
    if role_id not in known_roles:
        is_novel = bool(rng.uniform() < 0.85)
    else:
        is_novel = bool(novelty > NOVELTY_THRESHOLD)

    if role_id in known_roles:
        correct = rng.uniform() < (0.92 if domain != "ref_holdout" else 0.90)
        role_pred = role_id if correct else int(rng.choice(list(known_roles - {role_id}) or [role_id]))
        art_category = role_id
    else:
        role_pred = int(rng.choice(CIVIL_ROLES)) if not is_novel else None
        art_category = None if is_novel else role_pred

    role_conf = float(np.clip(rng.normal(0.85 if not is_novel else 0.55, 0.08), 0.05, 0.99))
    knn_dist = float(np.clip(rng.normal(0.05 + 0.5 * novelty, 0.03), 0.0, 1.5))

    # Cluster reference/operational domains apart in a fake 2D UMAP space.
    cluster_x = {
        "ref_train": 0, "ref_calib": 0.3, "ref_holdout": 0.6,
        "off_nadir": 4, "foreign": -4, "haze": 2, "snow": -2,
        "military": 6, "synthetic": -6,
    }[domain]
    umap_x = float(cluster_x + rng.normal(0, 0.8))
    umap_y = float(rng.normal(0, 0.8))

    return dict(
        ann_id=ann_row["ann_id"], tile_id=ann_row["tile_id"], batch_id=domain,
        role_pred=role_pred, role_conf=role_conf, novelty=novelty, is_novel=is_novel,
        art_category=art_category, knn_dist=knn_dist, umap_x=umap_x, umap_y=umap_y,
    )


def _compute_batches_json(scores: pd.DataFrame, ann: pd.DataFrame, tiles: pd.DataFrame) -> dict:
    merged = scores.merge(ann[["ann_id", "role"]], on="ann_id", how="left")
    merged = merged.merge(tiles[["tile_id", "parent_id"]], on="tile_id", how="left")

    out = {}
    for bid in OPERATIONAL_DOMAINS:
        g = merged[merged["batch_id"] == bid]
        if g.empty:
            continue
        civil = g[g["role"].isin([ROLE_NAMES[r] for r in CIVIL_ROLES])]
        role_acc = None
        if len(civil):
            pred_name = civil["role_pred"].map(lambda v: ROLE_NAMES.get(v) if pd.notna(v) else None)
            role_acc = float((pred_name == civil["role"]).mean())
        novelty_rate = float(g["is_novel"].mean())
        knn = g["knn_dist"]
        out[bid] = {
            "drift": {
                "score": round(float(knn.mean()), 4),
                "null_p95": 0.15,
                "p_value": round(float(max(0.001, 1.0 - knn.mean())), 4) if knn.mean() < 1 else 0.001,
                "n_boot": 200,
                "n": int(len(g)),
                "statistic": "mean 5-NN cosine distance to reference (fixture stub)",
                "null_method": "block bootstrap over ref_calib (fixture stub); p-value indicative",
                "level": "chip",
                "interpretation": "descriptive",
            },
            "novelty_rate": round(novelty_rate, 4),
            "role_accuracy": round(role_acc, 4) if role_acc is not None else None,
            "conf_shift": {
                "ref_mean": 0.9,
                "batch_mean": round(float(g["role_conf"].mean()), 4),
                "ks_p": 0.05,
            },
            "n_chips": int(len(g)),
            "n_tiles": int(g["tile_id"].nunique()),
            "n_parent_images": int(g["parent_id"].nunique()),
        }
    return out


def _compute_eval_json(scores: pd.DataFrame, ann: pd.DataFrame, extra_class_role_id: int | None) -> dict:
    merged = scores.merge(ann[["ann_id", "role", "role_id"]], on="ann_id", how="left")
    is_military = merged["role_id"] >= 4
    y_true = is_military.astype(int).to_numpy()
    y_score = merged["novelty"].to_numpy()
    auroc = _simple_auroc(y_true, y_score)

    holdout = merged  # fixture doesn't need a real held-out split for this synthetic eval
    per_class = []
    known_roles = list(CIVIL_ROLES) + ([extra_class_role_id] if extra_class_role_id else [])
    for rid in known_roles:
        name = ROLE_NAMES[rid]
        g = holdout[holdout["role_id"] == rid]
        if g.empty:
            continue
        pred_name = g["role_pred"].map(lambda v: ROLE_NAMES.get(v) if pd.notna(v) else None)
        acc = float((pred_name == name).mean())
        per_class.append({"cls": name, "n": int(len(g)), "accuracy": round(acc, 4)})
    overall_acc = float(np.mean([p["accuracy"] for p in per_class])) if per_class else None

    return {
        "exclusion_method": (
            "fixture stub: chips used to fit new classes and all chips from the same parent "
            "images are excluded from evaluation"
        ),
        "novelty": {
            "auroc_military_vs_civil": round(auroc, 4),
            "auprc": round(min(1.0, auroc), 4),
            "auroc_size_matched": round(max(0.5, auroc - 0.05), 4),
            "per_role": [
                {"role": ROLE_NAMES[r], "n": int((merged["role_id"] == r).sum()), "auroc": round(auroc, 4)}
                for r in sorted(merged["role_id"].unique())
            ],
            "baseline_knn_auroc": round(max(0.5, auroc - 0.1), 4),
        },
        "role_head": {
            "accuracy_ref_holdout": round(overall_acc, 4) if overall_acc is not None else None,
            "per_class": per_class,
        },
    }


def _simple_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    # Mann-Whitney U statistic -> AUROC, O(n*m) is fine at fixture scale.
    wins = sum((p > n) for p in pos for n in neg)
    ties = sum((p == n) for p in pos for n in neg)
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


def _build_detections(rng: np.random.RandomState, ann: pd.DataFrame, chips: pd.DataFrame) -> pd.DataFrame:
    """Synthesizes data/scores/detections.parquet's schema: one kind="gt" row
    per ground-truth box in a detection-evaluated domain (ann_id, tile_id,
    domain, tp), one kind="det" row per matched detection (same ann_id) plus
    a handful of unmatched false positives (ann_id=NaN).
    """
    domain_by_ann = dict(zip(chips["ann_id"], chips["domain"]))
    rows: list[dict] = []
    for _, a in ann.iterrows():
        domain = domain_by_ann.get(a["ann_id"])
        if domain not in DETECTION_DOMAINS:
            continue
        role_id = int(a["role_id"])
        x1, y1, x2, y2 = float(a["x"]), float(a["y"]), float(a["x"] + a["w"]), float(a["y"] + a["h"])
        tp = bool(rng.uniform() < _detect_prob(role_id, domain))
        rows.append(dict(tile_id=int(a["tile_id"]), domain=domain, kind="gt",
                          conf=np.nan, tp=tp, x1=x1, y1=y1, x2=x2, y2=y2, ann_id=int(a["ann_id"])))
        if tp:
            conf = float(np.clip(rng.normal(0.75, 0.15), 0.26, 0.99))
            jitter = 3.0
            rows.append(dict(
                tile_id=int(a["tile_id"]), domain=domain, kind="det", conf=conf, tp=True,
                x1=x1 + rng.uniform(-jitter, jitter), y1=y1 + rng.uniform(-jitter, jitter),
                x2=x2 + rng.uniform(-jitter, jitter), y2=y2 + rng.uniform(-jitter, jitter),
                ann_id=int(a["ann_id"]),
            ))

    # A handful of false-positive detections (no matching GT box) per tile, so
    # precision < 1 like the real data.
    tiles_by_domain = chips.groupby("domain")["tile_id"].unique()
    for domain in DETECTION_DOMAINS:
        for tile_id in tiles_by_domain.get(domain, []):
            if rng.uniform() < 0.3:
                conf = float(np.clip(rng.normal(0.4, 0.1), 0.25, 0.9))
                rows.append(dict(
                    tile_id=int(tile_id), domain=domain, kind="det", conf=conf, tp=False,
                    x1=10.0, y1=10.0, x2=50.0, y2=50.0, ann_id=np.nan,
                ))
    return pd.DataFrame(rows)


def _compute_detection_metrics(det_df: pd.DataFrame, ann: pd.DataFrame, tiles: pd.DataFrame) -> dict:
    """Mirrors data/scores/detection_metrics.json's schema, computed from the
    fixture's own synthesized detections.parquet (see _build_detections).
    """
    gt = det_df[det_df["kind"] == "gt"].merge(
        tiles[["tile_id", "off_nadir", "gsd", "weather", "country"]], on="tile_id", how="left"
    )
    det = det_df[det_df["kind"] == "det"]

    by_domain = {}
    for domain in DETECTION_DOMAINS:
        g_gt = gt[gt["domain"] == domain]
        if g_gt.empty:
            continue
        g_det = det[det["domain"] == domain]
        by_domain[domain] = {
            "n_gt": int(len(g_gt)),
            "n_det": int(len(g_det)),
            "recall": round(float(g_gt["tp"].mean()), 4),
            "precision": round(float(g_det["tp"].mean()), 4) if len(g_det) else None,
        }

    def _by_condition(col: str, bucket_fn) -> dict:
        labels = gt[col].map(bucket_fn) if bucket_fn else gt[col]
        out = {}
        for label in sorted(set(labels.dropna())):
            g_gt_b = gt[labels == label]
            tile_ids_b = set(g_gt_b["tile_id"])
            g_det_b = det[det["tile_id"].isin(tile_ids_b)]
            out[str(label)] = {
                "n_gt": int(len(g_gt_b)),
                "n_det": int(len(g_det_b)),
                "recall": round(float(g_gt_b["tp"].mean()), 4),
                "precision": round(float(g_det_b["tp"].mean()), 4) if len(g_det_b) else None,
            }
        return out

    by_condition = {
        "weather": _by_condition("weather", None),
        "off_nadir_b": _by_condition("off_nadir", _off_nadir_bucket),
        "country_g": _by_condition("country", _country_bucket),
        "gsd_b": _by_condition("gsd", _gsd_bucket),
    }

    role_by_ann = dict(zip(ann["ann_id"], ann["role_id"]))
    gt = gt.assign(role_id=gt["ann_id"].map(role_by_ann))
    civil_gt = gt[gt["role_id"].isin(CIVIL_ROLES)]
    mil_gt = gt[~gt["role_id"].isin(CIVIL_ROLES)]

    # military-batch-scoped: civil vs military recall computed only among GT
    # boxes within the "military" batch itself, so both groups share the same
    # collections (mirrors the real pipeline's rescoped recall_by_role_group --
    # see backend/app/dataio.py's detection_eval_summary / _normalize_role_group).
    mil_batch_gt = gt[gt["domain"] == "military"]
    mil_batch_civil = mil_batch_gt[mil_batch_gt["role_id"].isin(CIVIL_ROLES)]
    mil_batch_mil = mil_batch_gt[~mil_batch_gt["role_id"].isin(CIVIL_ROLES)]

    return {
        "conf_threshold": 0.25,
        "iou_threshold": 0.5,
        "by_domain": by_domain,
        "by_condition": by_condition,
        "recall_by_role_group": {
            "civil": round(float(mil_batch_civil["tp"].mean()), 4) if len(mil_batch_civil) else None,
            "military": round(float(mil_batch_mil["tp"].mean()), 4) if len(mil_batch_mil) else None,
            "n_civil": int(len(mil_batch_civil)),
            "n_military": int(len(mil_batch_mil)),
            "scope": "within the military batch (same collections for both groups)",
        },
        "recall_by_role_group_pooled": {
            "civil": round(float(civil_gt["tp"].mean()), 4) if len(civil_gt) else None,
            "military": round(float(mil_gt["tp"].mean()), 4) if len(mil_gt) else None,
            "n_civil": int(len(civil_gt)),
            "n_military": int(len(mil_gt)),
            "scope": "pooled over all operational batches (civil from every batch; acquisition-domain confounded)",
        },
    }


# ---------------------------------------------------------------------------
# Field mode fixture (STATE.md "Field mode" steps 1-3; backend/app/collections.py):
# data/chips_det/index.parquet (one row per detector-box chip, matched to a GT
# annotation or a false positive) + data/scores/{version}_det.parquet (that chip
# scored by the same role/novelty pipeline as the GT-chip scores, never used for
# fitting). Deterministic (no probability thresholds) so /api/collections' status
# rule -- insufficient_data / needs_review / nominal -- lands on all three
# outcomes reliably under the fixed seed: off_nadir's single small parent stays
# under min_denominator (insufficient_data); foreign/haze sit near the ref_holdout
# control's own flag rate (nominal); snow/military/synthetic are flagged far
# above it (needs_review) -- snow and military deliberately echo the real
# pipeline's own worst-case findings (STATE.md's snow/military novelty spikes).
DET_FLAG_FRACTION = {
    "ref_train": 0.03, "ref_calib": 0.03, "ref_holdout": 0.04,
    "off_nadir": 0.05, "foreign": 0.05, "haze": 0.05,
    "snow": 0.60, "military": 0.65, "synthetic": 0.55,
}


def _build_det_chips(rng: np.random.RandomState, ann: pd.DataFrame, chips: pd.DataFrame,
                      tiles: pd.DataFrame, det_img_dir: Path) -> pd.DataFrame:
    """data/chips_det/index.parquet fixture: one "matched" det chip per GT
    annotation (a stand-in for "the detector found this aircraft", jittered a
    couple pixels from the GT box) plus a handful of guaranteed (not merely
    probabilistic) false-positive det chips per tile, scaled to each tile's own
    GT density -- across EVERY domain, including the reference ones, so
    ref_holdout has enough det chips for a real reference_flag_rate.
    """
    domain_by_ann = dict(zip(chips["ann_id"], chips["domain"]))
    tile_domain = dict(zip(tiles["tile_id"], tiles["domain"])) if "domain" in tiles.columns else {}
    parent_by_tile = dict(zip(tiles["tile_id"], tiles["parent_id"]))
    gt_count_per_tile = ann.groupby("tile_id").size().to_dict()

    rows: list[dict] = []
    det_id = 0
    jitter = 2.0

    for _, a in ann.iterrows():
        tile_id = int(a["tile_id"])
        domain = domain_by_ann.get(a["ann_id"], tile_domain.get(tile_id))
        x1, y1 = float(a["x"]), float(a["y"])
        w, h = float(a["w"]), float(a["h"])
        jx1 = x1 + float(rng.uniform(-jitter, jitter))
        jy1 = y1 + float(rng.uniform(-jitter, jitter))
        path = det_img_dir / f"{det_id}.png"
        _solid_png(path, 224, tuple(int(v) for v in rng.randint(0, 255, size=3)))
        rows.append(dict(
            det_id=det_id, tile_id=tile_id, parent_id=parent_by_tile.get(tile_id), domain=domain,
            conf=float(np.clip(rng.normal(0.75, 0.15), 0.25, 0.99)),
            x1=jx1, y1=jy1, x2=jx1 + w, y2=jy1 + h, w=w, h=h,
            matched_ann_id=int(a["ann_id"]), path=str(path),
        ))
        det_id += 1

    for tile_id, n_gt in gt_count_per_tile.items():
        tile_id = int(tile_id)
        domain = tile_domain.get(tile_id)
        n_fp = max(1, int(n_gt) // 3 + 1)
        for _ in range(n_fp):
            path = det_img_dir / f"{det_id}.png"
            _solid_png(path, 224, tuple(int(v) for v in rng.randint(0, 255, size=3)))
            rows.append(dict(
                det_id=det_id, tile_id=tile_id, parent_id=parent_by_tile.get(tile_id), domain=domain,
                conf=float(np.clip(rng.normal(0.35, 0.08), 0.25, 0.9)),
                x1=10.0, y1=10.0, x2=50.0, y2=40.0, w=40.0, h=30.0,
                matched_ann_id=None, path=str(path),
            ))
            det_id += 1

    return pd.DataFrame(rows)


def _det_scores_for_version(det_index: pd.DataFrame, ann_role_by_id: dict,
                             extra_class_role_id: int | None) -> pd.DataFrame:
    """data/scores/{version}_det.parquet fixture: scores det_index's chips
    deterministically by DET_FLAG_FRACTION, grouped by parent_id (a
    collections group is, in this fixture, exactly one parent image -- see
    build_fixture's per-domain PLAN) so each group's n_flagged/flag_rate is
    reproducible under the fixed seed rather than left to chance.
    """
    rows: list[dict] = []
    for parent_id, g in det_index.groupby("parent_id", sort=False):
        domain = g["domain"].iloc[0]
        frac = DET_FLAG_FRACTION.get(domain, 0.05)
        acc = 0.0
        for _, d in g.iterrows():
            acc += frac
            is_novel = acc >= 1.0
            if is_novel:
                acc -= 1.0
            novelty = 0.75 if is_novel else 0.20

            matched = d.get("matched_ann_id")
            role_id = None
            if matched is not None and not (isinstance(matched, float) and np.isnan(matched)):
                role_id = ann_role_by_id.get(int(matched))
            known_roles = set(CIVIL_ROLES) | ({extra_class_role_id} if extra_class_role_id else set())

            if is_novel:
                role_pred = None
                art_category = None
            elif role_id is not None and role_id in known_roles:
                role_pred = role_id
                art_category = role_id
            else:
                role_pred = CIVIL_ROLES[0]
                art_category = role_pred

            rows.append(dict(
                det_id=int(d["det_id"]), tile_id=int(d["tile_id"]), parent_id=parent_id,
                batch_id=domain, role_pred=role_pred, role_conf=(0.5 if is_novel else 0.85),
                novelty=novelty, is_novel=bool(is_novel), art_category=art_category,
                knn_dist=(0.4 if is_novel else 0.08), neighbor_ids=None, neighbor_dists=None,
                umap_x=0.0, umap_y=0.0, used_in_fit=False, analyst_sim=0.3, conf=d.get("conf"),
            ))
    return pd.DataFrame(rows)


# Field-mode `field`/`control` blocks (backend/vigil/field.py, folded into eval_{version}.json by
# vigil.evaluate.evaluate() on the real pipeline) -- computed here from this fixture's OWN synthetic
# det scores / eval numbers, since vigil.field.compute_field only ever runs against the real ML
# pipeline (this fixture hand-crafts eval_v*.json, same as the rest of build_fixture.py). Just needs
# a real, self-consistent shape for the API layer's /versions/compare field-pass-through test.
FIELD_DOMAINS = [d for d in BATCH_ORDER if d != "ref_holdout"]  # BATCH_ORDER already excludes ref_train/ref_calib


def _compute_field_and_control(det_scores: pd.DataFrame, eval_json: dict, extra_class_role_id: int | None,
                               tiles: pd.DataFrame | None = None) -> dict:
    from vigil.grouping import group_id as _group_id  # same group_id /api/collections serves

    label = ROLE_NAMES[extra_class_role_id] if extra_class_role_id else None

    def _by_group(df: pd.DataFrame) -> list[dict]:
        """Top collections (airfield x date) by n_assigned -- mirrors vigil.field._by_group."""
        if not label or tiles is None or df.empty:
            return []
        sub = df[df["role_pred"] == extra_class_role_id][["tile_id"]]
        if sub.empty:
            return []
        sub = sub.merge(tiles[["tile_id", "airfield", "collection_date"]], on="tile_id", how="left")
        sub["date"] = pd.to_datetime(sub["collection_date"], errors="coerce", utc=True).dt.strftime("%Y-%m-%d")
        counts = sub.groupby(["airfield", "date"]).size().reset_index(name="n_assigned")
        counts = counts.sort_values(["n_assigned", "airfield", "date"], ascending=[False, True, True]).head(10)
        return [{"group_id": _group_id(r.airfield, r.date), "airfield": r.airfield, "date": r.date,
                 "n_assigned": int(r.n_assigned)} for r in counts.itertuples(index=False)]
    new_coll = det_scores[det_scores["batch_id"].isin(FIELD_DOMAINS)]
    n_detections = int(len(new_coll))
    used_in_fit = (
        new_coll["used_in_fit"].fillna(False).astype(bool)
        if "used_in_fit" in new_coll.columns else pd.Series([False] * n_detections, dtype=bool)
    )
    n_excluded_labeled = int(used_in_fit.sum())
    eligible = new_coll[~used_in_fit.to_numpy()] if n_detections else new_coll
    n_eligible = int(len(eligible))

    def _classes_for(df: pd.DataFrame, n_elig: int, with_groups: bool = False) -> list[dict]:
        if not label:
            return []
        n_assigned = int((df["role_pred"] == extra_class_role_id).sum()) if len(df) else 0
        cls = {
            "label": label, "n_labels_used": 1, "n_assigned": n_assigned,
            "assignment_rate": (n_assigned / n_elig) if n_elig else None,
        }
        if with_groups:
            cls["by_group"] = _by_group(df)
        return [cls]

    by_domain = {}
    for dom in FIELD_DOMAINS:
        sub = new_coll[new_coll["batch_id"] == dom]
        if sub.empty:
            continue
        sub_used = (
            sub["used_in_fit"].fillna(False).astype(bool)
            if "used_in_fit" in sub.columns else pd.Series([False] * len(sub), dtype=bool)
        )
        sub_eligible = sub[~sub_used.to_numpy()]
        by_domain[dom] = {
            "n_detections": int(len(sub)), "n_excluded_labeled": int(sub_used.sum()),
            "n_excluded_same_parent": 0, "n_eligible": int(len(sub_eligible)),
            "classes": _classes_for(sub_eligible, len(sub_eligible)),
        }

    field = {
        "domains": FIELD_DOMAINS,
        "n_detections": n_detections,
        "n_excluded_labeled": n_excluded_labeled,
        "n_excluded_same_parent": 0,
        "n_eligible": n_eligible,
        "classes": _classes_for(eligible, n_eligible, with_groups=True),
        "by_domain": by_domain,
        "note": "assignments are descriptive; not ground-truth accuracy",
    }

    role_head = eval_json.get("role_head", {}) or {}
    control = {
        "label": "labeled control-set regression check",
        "accuracy_ref_holdout": role_head.get("accuracy_ref_holdout"),
        "per_class": {p["cls"]: {"accuracy": p["accuracy"], "n": p["n"]} for p in role_head.get("per_class", [])},
    }
    return {"field": field, "control": control}


DETECTOR_MODEL_CARD = {
    "name": "YOLO11n single-class aircraft detector (fixture stub)",
    "trained_on": "ref_train tiles only (fixture), 1 epoch, imgsz 512, seed 0",
    "val_split": "ref_calib (fixture)",
    "val_map50": 0.95,
    "val_map50_95": 0.7,
    "val_precision": 0.9,
    "val_recall": 0.9,
    "conf_threshold": 0.25,
    "iou_threshold": 0.5,
    "note": (
        "Detector performance is scored against ground-truth boxes at IoU >= 0.5 and "
        "reported separately from role/novelty scores, which are computed on "
        "ground-truth chips (oracle localization)."
    ),
}


def build_fixture(root: str | Path, seed: int = 0) -> None:
    root = Path(root)
    if root.exists():
        shutil.rmtree(root)
    rp_dir = root / "rareplanes"
    tiles_img_dir = rp_dir / "tiles_png"
    chips_dir = root / "chips"
    chips_img_dir = chips_dir / "images"
    chips_det_dir = root / "chips_det"
    chips_det_img_dir = chips_det_dir / "images"
    scores_dir = root / "scores"
    for d in (rp_dir, tiles_img_dir, chips_dir, chips_img_dir, chips_det_dir, chips_det_img_dir, scores_dir):
        d.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(seed)
    tiles, ann, chips = _build_tables(rng, tiles_img_dir, chips_img_dir)
    tiles.to_parquet(rp_dir / "tiles.parquet", index=False)
    ann.to_parquet(rp_dir / "annotations.parquet", index=False)
    chips.to_parquet(chips_dir / "index.parquet", index=False)

    domain_by_ann = dict(zip(chips["ann_id"], chips["domain"]))
    ann_by_id = ann.set_index("ann_id", drop=False)
    ann_role_by_id = dict(zip(ann["ann_id"], ann["role_id"]))

    # ---- field mode: chips_det/index.parquet + {version}_det.parquet ----
    # See backend/app/collections.py -- not per-version at the index level
    # (bboxes/conf don't change), but scores are, exactly like GT chips.
    # Uses its OWN RandomState (not `rng`) so this addition doesn't perturb
    # the existing GT-chip/detection/score draw sequence every other test in
    # this suite was tuned against.
    rng_det = np.random.RandomState(seed + 1)
    det_index = _build_det_chips(rng_det, ann, chips, tiles, chips_det_img_dir)
    det_index.to_parquet(chips_det_dir / "index.parquet", index=False)

    det_scores_v1 = _det_scores_for_version(det_index, ann_role_by_id, extra_class_role_id=None)
    det_scores_v1.to_parquet(scores_dir / "v1_det.parquet", index=False)

    # ---- detector milestone: detections.parquet + detection_metrics.json + detector_model.json ----
    # Not per-version (mirrors the real data -- the detector isn't re-fit by /adapt).
    detections = _build_detections(rng, ann, chips)
    detections.to_parquet(scores_dir / "detections.parquet", index=False)
    detection_metrics = _compute_detection_metrics(detections, ann, tiles)
    (scores_dir / "detection_metrics.json").write_text(json.dumps(detection_metrics, indent=2))
    (scores_dir / "detector_model.json").write_text(json.dumps(DETECTOR_MODEL_CARD, indent=2))

    # ---- v1: no adaptation yet, military roles other than the "extra" one are novel ----
    v1_rows = [
        _score_row(rng, ann_by_id.loc[aid].to_dict(), domain_by_ann[aid], extra_class_role_id=None)
        for aid in ann["ann_id"]
    ]
    scores_v1 = pd.DataFrame(v1_rows)
    scores_v1.to_parquet(scores_dir / "v1.parquet", index=False)

    batches_v1 = _compute_batches_json(scores_v1, ann, tiles)
    (scores_dir / "v1_batches.json").write_text(json.dumps(batches_v1, indent=2))

    model_v1 = {
        "vigilance": 0.72,
        "n_categories": 3,
        "classes": [ROLE_NAMES[r] for r in CIVIL_ROLES],
        "calibrated_on": "ref_calib",
        "n_ref_chips": int(len(ann[ann["tile_id"].isin(tiles[tiles["domain"].isin(REFERENCE_DOMAINS)]["tile_id"])])),
        "n_ref_images": int(tiles[tiles["domain"].isin(REFERENCE_DOMAINS)]["parent_id"].nunique()),
        "embedder": "dinov2_vits14 (fixture stub)",
        "novelty_method": "fuzzy_art (fixture stub)",
        "name": "Vigil role/novelty model (fixture)",
        "detector": "yolo11n (fixture stub)",
        "reference_domain": "ref_train ∪ ref_calib",
        "calibration_check": {"min": 0.0, "median": 0.02, "max": 0.4, "n_parents": 3},
        "reference_definition": "ref_train ∪ ref_calib (fixture stub reference definition)",
        "model_selection_note": (
            "fixture stub: PCA dims and calibration scheme were chosen using ref_holdout/AUROC "
            "signals, not independent confirmation."
        ),
        "analyst_class_label_counts": {},
    }
    (scores_dir / "v1_model.json").write_text(json.dumps(model_v1, indent=2))

    eval_v1 = _compute_eval_json(scores_v1, ann, extra_class_role_id=None)
    eval_v1.update(_compute_field_and_control(det_scores_v1, eval_v1, extra_class_role_id=None, tiles=tiles))
    (scores_dir / "eval_v1.json").write_text(json.dumps(eval_v1, indent=2))

    # ---- v2: simulate /adapt having added the "Military Fighter/..." class ----
    extra_role_id = 6
    v2_rows = [
        _score_row(rng, ann_by_id.loc[aid].to_dict(), domain_by_ann[aid], extra_class_role_id=extra_role_id)
        for aid in ann["ann_id"]
    ]
    scores_v2 = pd.DataFrame(v2_rows)
    # Simulate mild forgetting: flip a few ref_holdout civil predictions to wrong.
    holdout_ann_ids = set(chips.loc[chips["domain"] == "ref_holdout", "ann_id"])
    idxs = scores_v2.index[scores_v2["ann_id"].isin(holdout_ann_ids)].tolist()
    flip = rng.choice(idxs, size=max(1, len(idxs) // 6), replace=False)
    for i in flip:
        true_role = int(ann_by_id.loc[scores_v2.at[i, "ann_id"], "role_id"])
        others = [r for r in CIVIL_ROLES if r != true_role]
        scores_v2.at[i, "role_pred"] = int(rng.choice(others))
    scores_v2.to_parquet(scores_dir / "v2.parquet", index=False)

    batches_v2 = _compute_batches_json(scores_v2, ann, tiles)
    (scores_dir / "v2_batches.json").write_text(json.dumps(batches_v2, indent=2))

    model_v2 = dict(model_v1)
    model_v2["vigilance"] = 0.70
    model_v2["n_categories"] = 4
    model_v2["classes"] = [ROLE_NAMES[r] for r in CIVIL_ROLES] + [ROLE_NAMES[extra_role_id]]
    model_v2["analyst_class_label_counts"] = {ROLE_NAMES[extra_role_id]: {"n_total": 1, "n_gt": 0, "n_det": 1}}
    (scores_dir / "v2_model.json").write_text(json.dumps(model_v2, indent=2))

    det_scores_v2 = _det_scores_for_version(det_index, ann_role_by_id, extra_class_role_id=extra_role_id)
    det_scores_v2.to_parquet(scores_dir / "v2_det.parquet", index=False)

    eval_v2 = _compute_eval_json(scores_v2, ann, extra_class_role_id=extra_role_id)
    eval_v2.update(_compute_field_and_control(det_scores_v2, eval_v2, extra_class_role_id=extra_role_id, tiles=tiles))
    (scores_dir / "eval_v2.json").write_text(json.dumps(eval_v2, indent=2))

    versions = [
        {"id": "v1", "created": "2026-09-05T12:00:00Z", "classes": model_v1["classes"], "n_feedback": 0, "note": "initial fixture build"},
        {"id": "v2", "created": "2026-09-05T13:00:00Z", "classes": model_v2["classes"], "n_feedback": 3, "note": "fixture: added military fighter class from analyst feedback"},
    ]
    (scores_dir / "versions.json").write_text(json.dumps(versions, indent=2))


if __name__ == "__main__":
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parents[2] / "data" / "scores_fixture")
    build_fixture(target)
    print(f"fixture built at {target}")
