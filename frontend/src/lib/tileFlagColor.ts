// Analyst-view helper: `GET /tiles/{id}` returns detector boxes (DetectorBox
// — x1,y1,x2,y2,conf,tp) but never a flagged/is_novel status (see
// backend/API.md's field-mode contract — no scoring on that endpoint). The
// only place a det chip's `is_novel` is available is the collection's own
// det-chip page (`DetChip[]`, from GET /collections/{group_id}), and only
// for chips the UI has already loaded (the current page of that list).
// This matches a tile's detector boxes to that page's det chips by bbox so
// the tile viewer (components/CollectionTiles.tsx) can color a box amber
// when it's known-flagged and cyan otherwise — a best-effort highlight, not
// an authoritative one; see CollectionTiles' caption note on the limitation.
import type { DetChip, DetectorBox } from '../api/types';

export const DETECTION_COLOR = '#0891b2';
export const FLAGGED_DETECTION_COLOR = '#d97706';

// Tolerance for the float round-trip through parquet/JSON — same box should
// come back within a fraction of a pixel.
const EPS = 0.75;

function bboxMatches(d: DetectorBox, bbox: [number, number, number, number]): boolean {
  const [x, y, w, h] = bbox;
  return Math.abs(d.x1 - x) < EPS && Math.abs(d.y1 - y) < EPS && Math.abs(d.x2 - (x + w)) < EPS && Math.abs(d.y2 - (y + h)) < EPS;
}

export function makeFlagColorFn(tileId: number, pageChips: DetChip[]): (d: DetectorBox) => string {
  const flaggedBoxes = pageChips.filter((c) => c.tile_id === tileId && c.is_novel).map((c) => c.bbox);
  if (!flaggedBoxes.length) return () => DETECTION_COLOR;
  return (d) => (flaggedBoxes.some((b) => bboxMatches(d, b)) ? FLAGGED_DETECTION_COLOR : DETECTION_COLOR);
}
