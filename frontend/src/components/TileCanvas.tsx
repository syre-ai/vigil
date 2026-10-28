import { useEffect, useRef } from 'react';
import { tileImageUrl } from '../api/client';
import type { DetectorBox } from '../api/types';
import { drawTileOverlay, NATIVE_TILE_SIZE } from '../lib/tileOverlay';

type OverlayChip = { ann_id: number; bbox: [number, number, number, number]; is_novel: boolean };

export function TileCanvas({
  tileId,
  chips,
  highlightAnnId,
  detections,
  highlightBox,
  detectionStyle,
  size = 380,
}: {
  tileId: number;
  chips: OverlayChip[];
  highlightAnnId: number;
  detections?: DetectorBox[] | null;
  highlightBox?: { bbox: [number, number, number, number]; color?: string };
  detectionStyle?: 'benchmark' | 'field';
  size?: number;
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const scale = size / NATIVE_TILE_SIZE;
    let cancelled = false;
    const img = new Image();
    img.onload = () => {
      if (cancelled) return;
      ctx.clearRect(0, 0, size, size);
      ctx.drawImage(img, 0, 0, size, size);
      drawTileOverlay(ctx, { chips, highlightAnnId, detections, scale, highlightBox, detectionStyle });
    };
    img.src = tileImageUrl(tileId);
    return () => {
      cancelled = true;
      img.onload = null;
      img.src = '';
    };
  }, [tileId, chips, highlightAnnId, detections, highlightBox, detectionStyle, size]);

  return <canvas ref={canvasRef} width={size} height={size} className="tile-canvas" />;
}
