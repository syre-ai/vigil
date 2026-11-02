"""Fetches the images listed in snap/images.txt from the backend and saves them as WebP.

    python scripts/snapshot_images.py snap http://localhost:8000
"""
import io
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

out = Path(sys.argv[1])
backend = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8000"
(out / "img").mkdir(parents=True, exist_ok=True)
items = [line.strip() for line in (out / "images.txt").read_text().splitlines() if line.strip()]


def convert(item: str) -> str | None:
    kind, ident = item.split("/")
    dest = out / "img" / f"{kind}_{ident}.webp"
    if dest.exists():
        return None
    try:
        raw = urllib.request.urlopen(f"{backend}/api/{item}/image", timeout=60).read()
    except Exception as exc:  # missing image: the snapshot shows a broken thumbnail, nothing else
        return f"{item}: {exc}"
    Image.open(io.BytesIO(raw)).convert("RGB").save(dest, "WEBP", quality=75, method=6)
    return None


with ThreadPoolExecutor(8) as pool:
    errors = [e for e in pool.map(convert, items) if e]
print(f"{len(items)} images, {len(errors)} failed")
for e in errors[:10]:
    print(" ", e)
