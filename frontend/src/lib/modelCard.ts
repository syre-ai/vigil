// Short, data-driven display names for the model card (shared by ModelCardStrip and the Report
// page) — derived from the model-card fields themselves so they stay correct if the embedder or
// detector changes. Nothing here is hard-coded.
import type { ModelInfo } from '../api/types';

export function shortEmbedderName(embedder: string): string {
  const dinoMatch = embedder.match(/dinov(\d+)/i);
  if (dinoMatch) return `DINOv${dinoMatch[1]}`;
  const firstSegment = embedder.split(',')[0]?.trim() || embedder;
  return firstSegment.split('/').pop() || firstSegment;
}

export function shortDetectorName(model: ModelInfo): string {
  const dc = model.detector_card;
  if (dc) return dc.name.split(' ')[0] || dc.name;
  return model.detector.split(' ')[0] || model.detector;
}
