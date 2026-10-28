import { useEffect, useMemo, useRef, useState, type MouseEvent as ReactMouseEvent } from 'react';
import { chipImageUrl } from '../api/client';
import type { EmbeddingBatchPoint, EmbeddingRefPoint } from '../api/types';
import { NA, score } from '../lib/format';

type Kind = 'reference' | 'batch-normal' | 'batch-novel';
type DrawPoint = { sx: number; sy: number; kind: Kind; ann_id: number; role_true?: string | null; novelty?: number; is_novel?: boolean };

const COLORS: Record<Kind, string> = {
  reference: '#9aa3af',
  'batch-normal': '#1f5aa6',
  'batch-novel': '#d97706',
};

const HEIGHT = 440;
const HIT_RADIUS = 8;

// Canvas-based scatter for large point counts (recharts' SVG scatter gets
// sluggish well before this) — draws every point in one pass, then does
// simple nearest-point hit-testing on mousemove for the hover tooltip.
export function EmbeddingCanvasScatter({ reference, batch }: { reference: EmbeddingRefPoint[]; batch: EmbeddingBatchPoint[] }) {
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const pointsRef = useRef<DrawPoint[]>([]);
  const [width, setWidth] = useState(640);
  const [hover, setHover] = useState<{ x: number; y: number; point: DrawPoint } | null>(null);

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      const w = entries[0]?.contentRect.width;
      if (w) setWidth(w);
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const bounds = useMemo(() => {
    let xMin = Infinity;
    let xMax = -Infinity;
    let yMin = Infinity;
    let yMax = -Infinity;
    for (const p of reference) {
      if (p.x < xMin) xMin = p.x;
      if (p.x > xMax) xMax = p.x;
      if (p.y < yMin) yMin = p.y;
      if (p.y > yMax) yMax = p.y;
    }
    for (const p of batch) {
      if (p.x < xMin) xMin = p.x;
      if (p.x > xMax) xMax = p.x;
      if (p.y < yMin) yMin = p.y;
      if (p.y > yMax) yMax = p.y;
    }
    if (!Number.isFinite(xMin)) return { xMin: -1, xMax: 1, yMin: -1, yMax: 1 };
    const padX = (xMax - xMin) * 0.06 || 1;
    const padY = (yMax - yMin) * 0.06 || 1;
    return { xMin: xMin - padX, xMax: xMax + padX, yMin: yMin - padY, yMax: yMax + padY };
  }, [reference, batch]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = HEIGHT * dpr;
    canvas.style.width = `${width}px`;
    canvas.style.height = `${HEIGHT}px`;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, width, HEIGHT);

    const pad = 10;
    const spanX = bounds.xMax - bounds.xMin || 1;
    const spanY = bounds.yMax - bounds.yMin || 1;
    const toSx = (x: number) => pad + ((x - bounds.xMin) / spanX) * (width - pad * 2);
    const toSy = (y: number) => HEIGHT - pad - ((y - bounds.yMin) / spanY) * (HEIGHT - pad * 2);

    const points: DrawPoint[] = [];

    ctx.globalAlpha = 0.5;
    ctx.fillStyle = COLORS.reference;
    for (const p of reference) {
      const sx = toSx(p.x);
      const sy = toSy(p.y);
      ctx.beginPath();
      ctx.arc(sx, sy, 2, 0, Math.PI * 2);
      ctx.fill();
      points.push({ sx, sy, kind: 'reference', ann_id: p.ann_id, role_true: p.role_true });
    }

    ctx.globalAlpha = 0.85;
    for (const p of batch) {
      const sx = toSx(p.x);
      const sy = toSy(p.y);
      const kind: Kind = p.is_novel ? 'batch-novel' : 'batch-normal';
      ctx.fillStyle = COLORS[kind];
      ctx.beginPath();
      ctx.arc(sx, sy, 3, 0, Math.PI * 2);
      ctx.fill();
      points.push({ sx, sy, kind, ann_id: p.ann_id, novelty: p.novelty, is_novel: p.is_novel });
    }
    ctx.globalAlpha = 1;
    pointsRef.current = points;
  }, [reference, batch, bounds, width]);

  function handleMouseMove(e: ReactMouseEvent<HTMLCanvasElement>) {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    let nearest: DrawPoint | null = null;
    let bestDist = Infinity;
    for (const p of pointsRef.current) {
      const d = (p.sx - mx) ** 2 + (p.sy - my) ** 2;
      if (d < bestDist) {
        bestDist = d;
        nearest = p;
      }
    }
    if (nearest && bestDist <= HIT_RADIUS ** 2) {
      setHover({ x: mx, y: my, point: nearest });
    } else {
      setHover(null);
    }
  }

  return (
    <div>
      <div className="embedding-canvas-wrap" ref={wrapRef}>
        <canvas ref={canvasRef} className="embedding-canvas" onMouseMove={handleMouseMove} onMouseLeave={() => setHover(null)} />
        {hover && (
          <div
            className="embedding-tooltip embedding-canvas-tooltip"
            style={{ left: Math.min(hover.x + 12, Math.max(width - 190, 0)), top: Math.max(hover.y - 44, 4) }}
          >
            <img src={chipImageUrl(hover.point.ann_id)} alt="" width={56} height={56} loading="lazy" />
            <div className="mono">
              <div>ann_id {hover.point.ann_id}</div>
              {hover.point.kind === 'reference' ? (
                <div>role_true {hover.point.role_true ?? NA}</div>
              ) : (
                <>
                  <div>novelty {score(hover.point.novelty)}</div>
                  <div>{hover.point.is_novel ? 'flagged novel' : 'not novel'}</div>
                </>
              )}
            </div>
          </div>
        )}
      </div>
      <div className="embedding-legend">
        <span className="embedding-legend-item">
          <span className="embedding-legend-swatch" style={{ background: COLORS.reference }} /> reference
        </span>
        <span className="embedding-legend-item">
          <span className="embedding-legend-swatch" style={{ background: COLORS['batch-normal'] }} /> batch (not novel)
        </span>
        <span className="embedding-legend-item">
          <span className="embedding-legend-swatch" style={{ background: COLORS['batch-novel'] }} /> batch (novel)
        </span>
      </div>
    </div>
  );
}
