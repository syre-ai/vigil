// Shared number-formatting helpers so every page renders the same kind of
// value the same way: percentages to 1 decimal, model scores to 3 decimals,
// counts with thousands separators.
import type { BatchId, CollectionStatus, HealthRecommendation, Review, ReviewAction } from '../api/types';

// Short x-axis-safe labels for batch ids — the full Batch.name strings (e.g.
// "Control (same domain as reference)", "Rare targets present (military
// airfields)") clip on a category axis at chart width. Full names stay in
// tooltips; only the on-axis tick text uses this.
const SHORT_BATCH_NAMES: Record<string, string> = {
  ref_holdout: 'control',
  off_nadir: 'off-nadir',
  foreign: 'foreign',
  haze: 'haze',
  snow: 'snow',
  military: 'military',
  synthetic: 'synthetic',
};

export function shortBatchName(id: BatchId): string {
  return SHORT_BATCH_NAMES[id] ?? id;
}

// Placeholder for a missing/undefined numeric or text value in tables and tiles.
// A bare dash reads as a minus sign in numeric columns, so every empty-value
// placeholder in the UI goes through this one constant.
export const NA = 'n/a';

export function pct(v: number | null | undefined, decimals = 1): string {
  if (v == null || !Number.isFinite(v)) return NA;
  return `${(v * 100).toFixed(decimals)}%`;
}

export function ppDelta(delta: number | null | undefined): string {
  if (delta == null || !Number.isFinite(delta)) return NA;
  const rounded = (delta * 100).toFixed(1);
  // toFixed can render a tiny negative delta as "-0.0" -- collapse to plain "0.0pp".
  return rounded === '-0.0' ? '0.0pp' : `${delta > 0 ? '+' : ''}${rounded}pp`;
}

export function score(v: number | null | undefined, decimals = 3): string {
  if (v == null || !Number.isFinite(v)) return NA;
  return v.toFixed(decimals);
}

export function count(v: number | null | undefined): string {
  if (v == null || !Number.isFinite(v)) return NA;
  return v.toLocaleString();
}

// Human labels for raw snake_case field names that show up as table keys
// (batch condition facts, chip metadata, etc.) — falls back to a generic
// title-cased humanization for anything not explicitly listed.
const FIELD_LABELS: Record<string, string> = {
  weather: 'Weather',
  country: 'Country',
  countries: 'Countries',
  off_nadir: 'Off-nadir angle',
  gsd: 'GSD',
  wingspan_m: 'Wingspan',
  airfield: 'Airfield',
  role_pred: 'Predicted role',
  role_true: 'True role (evaluation only)',
  role_conf: 'Confidence',
  knn_dist: '5-NN distance (baseline)',
  art_match: 'ART match',
  ann_id: 'Annotation #',
  tile_id: 'Tile #',
  n_tiles: 'Tiles',
  n_chips: 'Chips',
  n_parent_images: 'Parent images',
  n_images: 'Images',
  novelty: 'Novelty',
  is_novel: 'Flagged novel',
  detected: 'Detected',
  partial: 'Partial fraction',
  truncated: 'Cut by tile edge',
};

// Edge/partial chips: only worth flagging when the aircraft is actually cut
// off (partial < 1) — `truncated` alone can be true at partial=1.0 in some
// backends, and by default (min_partial=0.75) every visible chip already has
// partial >= 0.75, so the tag needs the fraction to be useful rather than a
// bare "Edge" label.
export function isEdgeChip(c: { truncated?: boolean | null; partial?: number | null }): boolean {
  return c.truncated === true && c.partial != null && c.partial < 1;
}

export function edgeLabel(c: { partial?: number | null }): string {
  return `Edge · ${pct(c.partial, 0)} visible`;
}

const COLLECTION_STATUS_LABELS: Record<CollectionStatus, string> = {
  needs_review: 'needs review',
  nominal: 'nominal',
  insufficient_data: 'insufficient data',
};

export function collectionStatusLabel(status: CollectionStatus): string {
  return COLLECTION_STATUS_LABELS[status];
}

export function humanizeKey(key: string): string {
  if (FIELD_LABELS[key]) return FIELD_LABELS[key];
  return key
    .split('_')
    .map((w) => (w.length ? w[0].toUpperCase() + w.slice(1) : w))
    .join(' ');
}

// ---------- review actions (toast copy, pending-feedback panel, detail-pane
// "reviewed" banner) share this vocabulary so the same action always reads
// the same way everywhere in the UI. ----------

export const REVIEW_ACTION_LABELS: Record<ReviewAction, string> = {
  confirm: 'Confirm',
  new_class: 'New class',
  not_aircraft: 'Not aircraft',
  skip: 'Skip',
  correct: 'Correct role',
  bad_box: 'Bad box',
};

/**
 * One-line human summary of a Review, e.g. `new class "Military fighter"`,
 * `confirmed Small civil`, `corrected Medium civil → Large civil`, `not
 * aircraft`, `bad box (localization error)`, `skipped`. `rolePred` is only
 * used for `confirm`/`correct` — the predicted role isn't stored on the
 * Review itself (only the corrected-to label is, in `review.label`).
 */
export function reviewSummary(review: Pick<Review, 'action' | 'label'>, rolePred?: string | null): string {
  switch (review.action) {
    case 'new_class':
      return `new class "${review.label ?? '?'}"`;
    case 'confirm':
      return `confirmed${rolePred ? ` ${shortRole(rolePred)}` : ''}`;
    case 'correct':
      return `corrected${rolePred ? ` ${shortRole(rolePred)}` : ''} → ${shortRole(review.label)}`;
    case 'not_aircraft':
      return 'not aircraft';
    case 'bad_box':
      return 'bad box (localization error)';
    case 'skip':
      return 'skipped';
    default:
      return review.action;
  }
}

/** Short display form of a RarePlanes role name, e.g. "Small Civil Transport/Utility" → "Small civil". */
export function shortRole(role: string | null | undefined): string {
  if (!role) return NA;
  return role
    .replace(' Transport/Utility/AWAC', ' transport')
    .replace(' Transport/Utility', '')
    .replace(' Fighter/Interceptor/Attack', ' fighter')
    .replace('Military Bomber', 'Military bomber')
    .replace('Military Trainer', 'Military trainer')
    .replace('Small Civil', 'Small civil')
    .replace('Medium Civil', 'Medium civil')
    .replace('Large Civil', 'Large civil');
}

/** Indicative p-value formatting: "<0.001" floor, 3 decimals otherwise, "—" when absent. */
export function pValue(p: number | null | undefined): string {
  if (p == null || !Number.isFinite(p)) return NA;
  if (p < 0.001) return '<0.001';
  return p.toFixed(3);
}

// Model health headline labels (GET /api/health recommendation) — shared by the health panel and the Report header.
export const RECOMMENDATION_LABEL: Record<HealthRecommendation, string> = {
  keep: 'Keep',
  patch: 'Patch',
  schedule_retrain: 'Schedule retrain',
};
