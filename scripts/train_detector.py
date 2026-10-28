"""Train a single-class YOLO11n aircraft detector on ref_train tiles (val = ref_calib). Writes models/detector/best.pt."""
import argparse, shutil
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--model", default=str(ROOT / "models/yolo11n.pt"))
    a = ap.parse_args()
    model = YOLO(a.model)
    model.train(data=str(ROOT / "data/yolo/aircraft.yaml"), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch,
                project=str(ROOT / "data/runs"), name="aircraft_yolo11n", exist_ok=True, seed=0,
                workers=4, patience=10, plots=False, verbose=False)
    best = ROOT / "data/runs/aircraft_yolo11n/weights/best.pt"
    out = ROOT / "models/detector"; out.mkdir(parents=True, exist_ok=True)
    shutil.copy(best, out / "best.pt")
    print("saved", out / "best.pt")

if __name__ == "__main__":
    main()
