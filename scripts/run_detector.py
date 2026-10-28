"""Run the trained aircraft detector over every tile outside ref_train/ref_calib and score detections against
ground-truth boxes (IoU >= 0.5, greedy matching). Writes data/scores/detections.parquet (one row per detection or
unmatched GT) and data/scores/detection_metrics.json (recall/precision per domain and per tile-level condition).
Ground truth is used for scoring only."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from vigil.data import load_tiles, load_annotations  # noqa: E402
from vigil.domains import assign_domains  # noqa: E402

CONF = 0.25
IOU_T = 0.5

def iou_matrix(a, b):
    """IoU between boxes a (N,4) and b (M,4) in xyxy."""
    ax1, ay1, ax2, ay2 = [a[:, i][:, None] for i in range(4)]
    bx1, by1, bx2, by2 = [b[:, i][None, :] for i in range(4)]
    iw = np.clip(np.minimum(ax2, bx2) - np.maximum(ax1, bx1), 0, None)
    ih = np.clip(np.minimum(ay2, by2) - np.maximum(ay1, by1), 0, None)
    inter = iw * ih
    return inter / ((ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter + 1e-9)

def main():
    model = YOLO(str(ROOT / "models/detector/best.pt"))
    tiles = assign_domains(load_tiles())
    anns = load_annotations()
    tiles = tiles[~tiles.domain.isin(["ref_train", "ref_calib"])]
    tiles = tiles[[Path(p).exists() for p in tiles.image_path]]
    rows = []
    paths = tiles.image_path.tolist()
    for i in range(0, len(paths), 32):
        batch = paths[i:i + 32]
        res = model.predict(batch, conf=CONF, imgsz=512, verbose=False, device=0)
        for r, (_, t) in zip(res, tiles.iloc[i:i + 32].iterrows()):
            det = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0, 4))
            conf = r.boxes.conf.cpu().numpy() if r.boxes is not None else np.zeros(0)
            g = anns[anns.tile_id == t.tile_id]
            gt = np.stack([g.x, g.y, g.x + g.w, g.y + g.h], 1) if len(g) else np.zeros((0, 4))
            # greedy matching: detections in descending confidence, each takes the best REMAINING GT with IoU >= threshold
            det_to_gt: dict[int, int] = {}
            if len(det) and len(gt):
                M = iou_matrix(det, gt)
                free = np.ones(len(gt), dtype=bool)
                for d in np.argsort(-conf):
                    cand = np.where(free & (M[d] >= IOU_T))[0]
                    if len(cand):
                        j = int(cand[np.argmax(M[d, cand])])
                        det_to_gt[int(d)] = j; free[j] = False
            gt_to_det = {j: d for d, j in det_to_gt.items()}
            for d in range(len(det)):
                j = det_to_gt.get(d)
                rows.append(dict(tile_id=t.tile_id, domain=t.domain, kind="det", conf=float(conf[d]), tp=j is not None,
                                 x1=float(det[d, 0]), y1=float(det[d, 1]), x2=float(det[d, 2]), y2=float(det[d, 3]),
                                 ann_id=int(g.iloc[j].ann_id) if j is not None else None))
            for j in range(len(gt)):
                rows.append(dict(tile_id=t.tile_id, domain=t.domain, kind="gt", conf=None, tp=j in gt_to_det,
                                 x1=float(gt[j, 0]), y1=float(gt[j, 1]), x2=float(gt[j, 2]), y2=float(gt[j, 3]),
                                 ann_id=int(g.iloc[j].ann_id)))
        print(f"{i + len(batch)}/{len(paths)} tiles", flush=True)
    df = pd.DataFrame(rows)
    out = ROOT / "data/scores"; out.mkdir(exist_ok=True)
    df.to_parquet(out / "detections.parquet", index=False)
    # per-domain and per-condition metrics (recall over GT, precision over detections)
    tmeta = tiles.set_index("tile_id")
    df["weather"] = df.tile_id.map(tmeta.weather); df["country"] = df.tile_id.map(tmeta.country)
    # NaN off_nadir/gsd (missing tile metadata) bucket to "unknown", not a stringified "nan" or a
    # silently-wrong real bucket (np.where(nan < 0.35, ...) is False, which used to mislabel unknown
    # gsd as ">=0.35") -- "unknown" rows are then excluded from by_condition below (never silently
    # folded into a real bucket's recall/precision). by_domain is unaffected (grouped on `domain`, not
    # these condition buckets).
    off_nadir_raw = df.tile_id.map(tmeta.off_nadir)
    off_nadir_cut = pd.cut(off_nadir_raw, [-1, 15, 25, 90], labels=["<=15", "15-25", ">25"])
    df["off_nadir_b"] = off_nadir_cut.astype(object).fillna("unknown")
    gsd_raw = df.tile_id.map(tmeta.gsd)
    df["gsd_b"] = np.where(gsd_raw.isna(), "unknown", np.where(gsd_raw < 0.35, "<0.35", ">=0.35"))
    df["country_g"] = np.where(df.country == "USA", "USA", "other")
    def metrics(sub):
        gt = sub[sub.kind == "gt"]; de = sub[sub.kind == "det"]
        return dict(n_gt=int(len(gt)), n_det=int(len(de)), recall=float(gt.tp.mean()) if len(gt) else None,
                    precision=float(de.tp.mean()) if len(de) else None)
    def condition_metrics(dim):
        # exclude the "unknown" bucket (missing off_nadir/gsd metadata) from the report entirely
        return {str(k): metrics(s) for k, s in df.groupby(dim) if str(k) != "unknown"}
    res = {"conf_threshold": CONF, "iou_threshold": IOU_T, "by_domain": {d: metrics(s) for d, s in df.groupby("domain")},
           "by_condition": {dim: condition_metrics(dim) for dim in ["weather", "off_nadir_b", "country_g", "gsd_b"]}}
    # military vs civil recall (ground truth used for scoring only). Two views: pooled over all operational batches
    # (civil comes from every batch, military only from the military batch -> acquisition-domain confounded) and
    # WITHIN the military batch only (same airfields/collections for both groups -> the fair comparison).
    gtm = df[(df.kind == "gt")].merge(anns[["ann_id", "role_id"]], on="ann_id", how="left")
    def by_group(sub):
        return {"civil": float(sub[sub.role_id < 4].tp.mean()), "military": float(sub[sub.role_id >= 4].tp.mean()),
                "n_civil": int((sub.role_id < 4).sum()), "n_military": int((sub.role_id >= 4).sum())}
    res["recall_by_role_group"] = {**by_group(gtm[gtm.domain == "military"]), "scope": "within the military batch (same collections for both groups)"}
    res["recall_by_role_group_pooled"] = {**by_group(gtm), "scope": "pooled over all operational batches (civil from every batch; acquisition-domain confounded)"}
    json.dump(res, open(out / "detection_metrics.json", "w"), indent=2)
    print(json.dumps(res["by_domain"], indent=1)); print(res["recall_by_role_group"])

if __name__ == "__main__":
    main()
