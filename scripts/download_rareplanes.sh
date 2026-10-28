#!/bin/bash
# Download full RarePlanes real tiled train/test PNGs + COCO annotations (anonymous S3 over HTTPS).
set -euo pipefail
BASE="https://rareplanes-public.s3.amazonaws.com"
OUT="${VIGIL_DATA_ROOT:-$(cd "$(dirname "$0")/.." && pwd)/data}/rareplanes"
mkdir -p "$OUT/ann" "$OUT/train" "$OUT/test"
list_all() { # $1 prefix
  local token=""; local url
  while :; do
    url="$BASE/?list-type=2&prefix=$1&max-keys=1000${token:+&continuation-token=$(python3 -c "import urllib.parse,sys;print(urllib.parse.quote(sys.argv[1],safe=''))" "$token")}"
    xml=$(curl -s "$url" || true)
    echo "$xml" | grep -oP '(?<=<Key>)[^<]+\.png' || true
    token=$(echo "$xml" | grep -oP '(?<=<NextContinuationToken>)[^<]+' || true)
    [ -z "$token" ] && break
  done
}
for f in RarePlanes_Public_Metadata.csv RarePlanes_Public_Role_Counts.csv RarePlanes_Train_Coco_Annotations_tiled.json RarePlanes_Test_Coco_Annotations_tiled.json RarePlanes_Public_All_Annotations.geojson; do
  [ -s "$OUT/ann/$f" ] || curl -s "$BASE/real/metadata_annotations/$f" -o "$OUT/ann/$f" || true
done
for split in train test; do
  list_all "real/$split/PS-RGB_tiled/" | sort -u > "$OUT/${split}_keys.txt"
  echo "$split tiles: $(wc -l < "$OUT/${split}_keys.txt")"
  cat "$OUT/${split}_keys.txt" | xargs -P 12 -I{} sh -c 'f="'"$OUT/$split"'/$(basename {})"; [ -s "$f" ] || curl -s -o "$f" "'"$BASE"'/{}" || true'
  echo "$split done: $(ls $OUT/$split | wc -l) files, $(du -sh $OUT/$split | cut -f1)"
done
