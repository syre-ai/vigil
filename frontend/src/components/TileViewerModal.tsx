import { useEffect, useRef, useState, type PointerEvent as ReactPointerEvent, type ReactNode } from 'react';
import { ChevronLeft, ChevronRight } from 'lucide-react';
import { tileImageUrl } from '../api/client';
import type { DetectorBox } from '../api/types';
import { drawTileOverlay, NATIVE_TILE_SIZE } from '../lib/tileOverlay';

type OverlayChip = { ann_id: number; bbox: [number, number, number, number]; is_novel: boolean };

const MIN_ZOOM = 1;
const MAX_ZOOM = 8;

function fitCanvasSize(): number {
  // Fit the (square) tile inside the viewport, leaving room for the close
  // button, legend and caption below/around it.
  const avail = Math.min(window.innerWidth - 64, window.innerHeight - 180);
  return Math.max(240, Math.min(avail, 1400));
}

function clampPan(p: { x: number; y: number }, zoom: number, canvasSize: number): { x: number; y: number } {
  const imgSize = canvasSize * zoom;
  const min = canvasSize - imgSize; // <= 0 for zoom >= 1
  return { x: Math.min(0, Math.max(min, p.x)), y: Math.min(0, Math.max(min, p.y)) };
}

// Full-screen viewer for the parent tile: fit-to-viewport by default,
// mouse-wheel zoom (1x-8x) centred on the cursor, drag to pan, double-click
// to reset. Draws with plain canvas math (no ctx.transform) so box line
// widths stay in screen pixels regardless of zoom.
export function TileViewerModal({
  tileId,
  chips,
  highlightAnnId,
  detections,
  highlightBox,
  detectionStyle,
  detectionColor,
  legend,
  onClose,
  onPrev,
  onNext,
  stepLabel,
}: {
  tileId: number;
  chips: OverlayChip[];
  highlightAnnId: number;
  detections?: DetectorBox[] | null;
  highlightBox?: { bbox: [number, number, number, number]; color?: string };
  detectionStyle?: 'benchmark' | 'field';
  // Per-box color override for detector boxes — see lib/tileFlagColor.ts.
  detectionColor?: (d: DetectorBox) => string;
  // Overrides the default (ground-truth + detector) legend below the
  // canvas — field mode never shows a ground-truth swatch.
  legend?: ReactNode;
  onClose: () => void;
  // Optional prev/next stepping through a caller-defined tile sequence (e.g.
  // a collection's tiles, in CollectionTiles.tsx) — rendered as floating
  // chevrons over the canvas and wired to ←/→. Omit both to render the
  // viewer with no stepping controls (unchanged from before this prop
  // existed).
  onPrev?: () => void;
  onNext?: () => void;
  // Overrides the "Tile {tileId}" lead-in of the caption, e.g. "Tile 3 of 12 · 481".
  stepLabel?: string;
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const imgRef = useRef<HTMLImageElement | null>(null);
  const dragRef = useRef<{ startX: number; startY: number; panX: number; panY: number } | null>(null);

  const [canvasSize, setCanvasSize] = useState(fitCanvasSize);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [imgLoaded, setImgLoaded] = useState(false);

  // Reset zoom/pan when the viewer switches to a different tile. Adjusting
  // state during render (rather than in an effect) avoids an extra render
  // pass — see ReviewActions' resetForAnnId for the same idiom.
  const [loadedForTileId, setLoadedForTileId] = useState(tileId);
  if (tileId !== loadedForTileId) {
    setLoadedForTileId(tileId);
    setImgLoaded(false);
    setZoom(1);
    setPan({ x: 0, y: 0 });
  }

  useEffect(() => {
    function onResize() {
      const newSize = fitCanvasSize();
      setCanvasSize(newSize);
      setPan((p) => clampPan(p, zoom, newSize));
    }
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, [zoom]);

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key === 'Escape') onClose();
      else if (e.key === 'ArrowLeft' && onPrev) onPrev();
      else if (e.key === 'ArrowRight' && onNext) onNext();
    }
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose, onPrev, onNext]);

  // Load the tile image whenever it isn't already loaded (a fresh tileId, per
  // the reset above).
  useEffect(() => {
    if (imgLoaded) return;
    let cancelled = false;
    const img = new Image();
    img.onload = () => {
      if (cancelled) return;
      imgRef.current = img;
      setImgLoaded(true);
    };
    img.src = tileImageUrl(tileId);
    return () => {
      cancelled = true;
      img.onload = null;
    };
  }, [tileId, imgLoaded]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const img = imgRef.current;
    if (!canvas || !img || !imgLoaded) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const baseScale = canvasSize / NATIVE_TILE_SIZE;
    const effScale = baseScale * zoom;
    ctx.clearRect(0, 0, canvasSize, canvasSize);
    ctx.drawImage(img, pan.x, pan.y, effScale * NATIVE_TILE_SIZE, effScale * NATIVE_TILE_SIZE);
    drawTileOverlay(ctx, {
      chips,
      highlightAnnId,
      detections,
      scale: effScale,
      offsetX: pan.x,
      offsetY: pan.y,
      highlightColor: () => '#d97706',
      highlightBox,
      detectionStyle,
      detectionColor,
    });
  }, [canvasSize, zoom, pan, chips, highlightAnnId, detections, highlightBox, detectionStyle, detectionColor, imgLoaded]);

  // Attached as a native (non-passive) listener — React attaches its own
  // synthetic `wheel` handler as passive, which cannot call preventDefault
  // (it would only warn and still let the page scroll).
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !imgLoaded) return;
    function handleWheel(e: WheelEvent) {
      e.preventDefault();
      const rect = canvas!.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      const baseScale = canvasSize / NATIVE_TILE_SIZE;
      const oldEff = baseScale * zoom;
      const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
      const newZoom = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, zoom * factor));
      const newEff = baseScale * newZoom;
      setPan((p) => {
        const imgX = (mx - p.x) / oldEff;
        const imgY = (my - p.y) / oldEff;
        return clampPan({ x: mx - imgX * newEff, y: my - imgY * newEff }, newZoom, canvasSize);
      });
      setZoom(newZoom);
    }
    canvas.addEventListener('wheel', handleWheel, { passive: false });
    return () => canvas.removeEventListener('wheel', handleWheel);
  }, [canvasSize, zoom, imgLoaded]);

  function onPointerDown(e: ReactPointerEvent<HTMLCanvasElement>) {
    e.currentTarget.setPointerCapture(e.pointerId);
    dragRef.current = { startX: e.clientX, startY: e.clientY, panX: pan.x, panY: pan.y };
  }
  function onPointerMove(e: ReactPointerEvent<HTMLCanvasElement>) {
    const drag = dragRef.current;
    if (!drag) return;
    const dx = e.clientX - drag.startX;
    const dy = e.clientY - drag.startY;
    setPan(clampPan({ x: drag.panX + dx, y: drag.panY + dy }, zoom, canvasSize));
  }
  function endDrag(e: ReactPointerEvent<HTMLCanvasElement>) {
    dragRef.current = null;
    if (e.currentTarget.hasPointerCapture(e.pointerId)) e.currentTarget.releasePointerCapture(e.pointerId);
  }
  function onDoubleClick() {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  }

  return (
    <div className="tile-viewer-backdrop" onClick={onClose}>
      <div className="tile-viewer-panel" onClick={(e) => e.stopPropagation()}>
        <button type="button" className="tile-viewer-close" onClick={onClose} aria-label="Close tile viewer">
          ×
        </button>
        <div className="tile-viewer-canvas-wrap" style={{ width: canvasSize, height: canvasSize }}>
          {imgLoaded ? (
            <canvas
              ref={canvasRef}
              width={canvasSize}
              height={canvasSize}
              className="tile-viewer-canvas"
              onPointerDown={onPointerDown}
              onPointerMove={onPointerMove}
              onPointerUp={endDrag}
              onPointerCancel={endDrag}
              onDoubleClick={onDoubleClick}
            />
          ) : (
            <div className="tile-viewer-loading">Loading tile…</div>
          )}
          {onPrev && (
            <button type="button" className="tile-viewer-nav tile-viewer-nav-prev" onClick={onPrev} aria-label="Previous tile (←)" title="Previous tile (←)">
              <ChevronLeft size={18} strokeWidth={2.25} />
            </button>
          )}
          {onNext && (
            <button type="button" className="tile-viewer-nav tile-viewer-nav-next" onClick={onNext} aria-label="Next tile (→)" title="Next tile (→)">
              <ChevronRight size={18} strokeWidth={2.25} />
            </button>
          )}
        </div>
        <div className="tile-legend tile-viewer-legend">
          {legend ?? (
            <>
              <span>
                <span className="legend-swatch" style={{ borderColor: '#9aa3af' }} /> ground truth (oracle localization)
              </span>
              <span>
                <span className="legend-swatch" style={{ borderColor: '#d97706' }} /> this chip
              </span>
              <span>
                <span className="legend-swatch" style={{ borderColor: '#0891b2' }} /> detector box
              </span>
              <span>
                <span className="legend-swatch" style={{ borderColor: '#0891b2', borderTopStyle: 'dashed' }} /> detector box, no GT match
              </span>
            </>
          )}
        </div>
        <div className="caption tile-viewer-caption">
          {stepLabel ?? `Tile ${tileId}`} · scroll/pinch to zoom ({zoom.toFixed(1)}×) · drag to pan · double-click to reset · Esc to close
        </div>
      </div>
    </div>
  );
}
