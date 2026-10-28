// Shared box-overlay drawing so every place that renders a 512×512 tile —
// the small in-drawer preview (TileCanvas) and the full-screen viewer
// (TileViewerModal) — draws ground-truth/detector boxes identically.
import type { DetectorBox } from '../api/types';

export const NATIVE_TILE_SIZE = 512;

// Widened from `Chip[]` to this structural subset so field-mode callers can
// draw det-chip boxes too without a fake-Chip cast.
type OverlayChip = { ann_id: number; bbox: [number, number, number, number]; is_novel: boolean };

export function drawTileOverlay(
  ctx: CanvasRenderingContext2D,
  opts: {
    chips: OverlayChip[];
    highlightAnnId: number;
    detections?: DetectorBox[] | null;
    scale: number; // canvas px per tile px
    offsetX?: number; // canvas px added after scaling — lets a panned/zoomed
    offsetY?: number; // caller (the viewer) place boxes without transforming ctx
    highlightColor?: (chip: OverlayChip) => string; // defaults to amber-if-novel/blue-otherwise
    // Field/analyst mode: a chip's own box, drawn on top, that isn't a member
    // of `chips` (det chips have no ground-truth sibling list to draw).
    highlightBox?: { bbox: [number, number, number, number]; color?: string };
    // 'benchmark' (default): detector boxes dashed when `tp` (matched a
    // ground-truth box) is false. 'field': every detector box drawn solid —
    // field/analyst view never surfaces ground-truth match status (see
    // backend/API.md's field-mode contract: "no ground truth anywhere in
    // these payloads").
    detectionStyle?: 'benchmark' | 'field';
    // Per-box color override for detector boxes (e.g. amber for a box
    // matched to a flagged det chip — see lib/tileFlagColor.ts). Defaults to
    // the fixed teal below when omitted.
    detectionColor?: (d: DetectorBox) => string;
  },
) {
  const { chips, highlightAnnId, detections, scale } = opts;
  const ox = opts.offsetX ?? 0;
  const oy = opts.offsetY ?? 0;
  const highlightColor = opts.highlightColor ?? ((c: OverlayChip) => (c.is_novel ? '#d97706' : '#1f5aa6'));
  const detectionStyle = opts.detectionStyle ?? 'benchmark';

  // Ground-truth boxes (oracle localization) — white, with the reviewed
  // chip highlighted (amber if flagged novel, blue otherwise, unless the
  // caller overrides highlightColor).
  for (const c of chips) {
    const [x, y, w, h] = c.bbox;
    const isTarget = c.ann_id === highlightAnnId;
    ctx.setLineDash([]);
    ctx.lineWidth = isTarget ? 2.5 : 1;
    ctx.strokeStyle = isTarget ? highlightColor(c) : '#ffffff';
    ctx.globalAlpha = isTarget ? 1 : 0.75;
    ctx.strokeRect(x * scale + ox, y * scale + oy, w * scale, h * scale);
  }
  ctx.globalAlpha = 1;

  // Detector boxes — teal, thin; dashed when the box has no matching GT
  // (a false positive, tp=false) in benchmark mode only. Colors here are
  // chosen for visibility against the satellite imagery underneath, not the
  // app's light theme.
  for (const d of detections ?? []) {
    ctx.lineWidth = 1.25;
    ctx.strokeStyle = opts.detectionColor ? opts.detectionColor(d) : '#0891b2';
    ctx.setLineDash(detectionStyle === 'field' ? [] : d.tp ? [] : [3, 2]);
    ctx.strokeRect(d.x1 * scale + ox, d.y1 * scale + oy, (d.x2 - d.x1) * scale, (d.y2 - d.y1) * scale);
  }
  ctx.setLineDash([]);

  if (opts.highlightBox) {
    const [x, y, w, h] = opts.highlightBox.bbox;
    ctx.lineWidth = 2.5;
    ctx.strokeStyle = opts.highlightBox.color ?? '#d97706';
    ctx.globalAlpha = 1;
    ctx.strokeRect(x * scale + ox, y * scale + oy, w * scale, h * scale);
  }
}
