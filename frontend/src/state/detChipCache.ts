// Client-side snapshot of DetChip payloads keyed by det_id — the field-mode
// twin of state/chipCache.ts, kept in a *separate* Map so a det_id can never
// collide with a ground-truth chip's ann_id in the shared benchmark cache.
import type { DetChip } from '../api/types';

const cache = new Map<number, DetChip>();

export function cacheDetChip(chip: DetChip) {
  cache.set(chip.det_id, chip);
}

export function cacheDetChips(chips: DetChip[]) {
  for (const c of chips) cache.set(c.det_id, c);
}

export function getCachedDetChip(detId: number): DetChip | undefined {
  return cache.get(detId);
}
