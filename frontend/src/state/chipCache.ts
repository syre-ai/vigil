// Client-side snapshot of Chip payloads keyed by ann_id, so the Pending
// feedback panel (and the "reviewed this session" strip) can show a
// thumbnail/role for a chip that has since been reviewed and dropped out of
// the review queue — without re-fetching everything from `/chips/{ann_id}`.
// Populated opportunistically wherever chips are fetched in bulk (the review
// queue) or individually (the chip drawer); read with a `/chips/{ann_id}`
// fallback for ids never seen this session.
import type { Chip } from '../api/types';

const cache = new Map<number, Chip>();

export function cacheChip(chip: Chip) {
  cache.set(chip.ann_id, chip);
}

export function cacheChips(chips: Chip[]) {
  for (const c of chips) cache.set(c.ann_id, c);
}

export function getCachedChip(annId: number): Chip | undefined {
  return cache.get(annId);
}
