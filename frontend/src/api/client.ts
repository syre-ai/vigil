// Typed fetchers matching backend/API.md. Base path is /api, proxied to the
// mock (or real) backend by vite.config.ts in dev, and by whatever serves
// the built assets in production.
import type {
  Batch,
  BatchDetail,
  ChangesResponse,
  ChangesSort,
  ChangesSummary,
  Chip,
  ChipDetail,
  ChipsPage,
  CollectionDetail,
  CollectionsResponse,
  CollectionsSummary,
  DetChip,
  DetChipDetail,
  DriftResponse,
  EmbeddingPoints,
  EnvelopeResponse,
  EvalResponse,
  HealthResponse,
  MapResponse,
  ReferenceResponse,
  Review,
  ReviewAction,
  ReviewSource,
  ReviewStrategy,
  Summary,
  TileDetail,
  Version,
  VersionsCompare,
} from './types';

const BASE = '/api';

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
    this.name = 'ApiError';
  }
}

async function getJSON<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new ApiError(res.status, body.detail || res.statusText);
  }
  return res.json() as Promise<T>;
}

async function postJSON<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body ?? {}),
  });
  if (!res.ok) {
    const parsed = await res.json().catch(() => ({ detail: res.statusText }));
    throw new ApiError(res.status, parsed.detail || res.statusText);
  }
  return res.json() as Promise<T>;
}

function qs(params: Record<string, string | number | boolean | undefined>): string {
  const entries = Object.entries(params).filter(([, v]) => v !== undefined);
  if (!entries.length) return '';
  const search = new URLSearchParams(entries.map(([k, v]) => [k, String(v)]));
  return `?${search.toString()}`;
}

export function chipImageUrl(annId: number): string {
  return `${BASE}/chips/${annId}/image`;
}
export function tileImageUrl(tileId: number): string {
  return `${BASE}/tiles/${tileId}/image`;
}
export function detImageUrl(detId: number): string {
  return `${BASE}/det/${detId}/image`;
}

export const api = {
  summary: () => getJSON<Summary>('/summary'),
  batches: () => getJSON<Batch[]>('/batches'),
  batchDetail: (id: string) => getJSON<BatchDetail>(`/batches/${encodeURIComponent(id)}`),
  batchChips: (
    id: string,
    params: {
      sort?: 'novelty' | 'role_conf' | 'knn_dist';
      order?: 'asc' | 'desc';
      limit?: number;
      offset?: number;
      only_novel?: boolean;
      reviewed?: 'all' | 'yes' | 'no';
      detected?: 'all' | 'yes' | 'no'; // detector milestone field — no-op on backends that don't support it yet
      min_partial?: number; // edge/partial-chip field — no-op on backends that don't support it yet; default (on the backend) is 0.75
    },
  ) => getJSON<ChipsPage>(`/batches/${encodeURIComponent(id)}/chips${qs(params)}`),
  chipDetail: (annId: number) => getJSON<ChipDetail>(`/chips/${annId}`),
  tileDetail: (tileId: number) => getJSON<TileDetail>(`/tiles/${tileId}`),
  embeddingPoints: (batch: string) => getJSON<EmbeddingPoints>(`/embedding/points${qs({ batch })}`),
  drift: () => getJSON<DriftResponse>('/drift'),
  envelope: () => getJSON<EnvelopeResponse>('/envelope'),
  evalReport: () => getJSON<EvalResponse>('/eval'),
  reviewQueue: (params: { limit?: number; strategy?: ReviewStrategy; min_partial?: number }) => getJSON<Chip[]>(`/review/queue${qs(params)}`),
  // source omitted => GT chip review log (unchanged); source='det' => field-mode review log.
  reviewLog: (source?: ReviewSource | 'all') => getJSON<Review[]>(`/review/log${qs({ source })}`),
  usedLabels: async () => (await getJSON<{ labels: string[] }>('/review/labels')).labels,
  reviewClasses: async () => (await getJSON<{ classes: string[] }>('/review/classes')).classes,
  postReview: (id: number, body: { action: ReviewAction; label?: string; note?: string }, source?: ReviewSource) =>
    postJSON<Review>(`/review/${id}${qs({ source })}`, body),
  adapt: (opts?: { note?: string; assign_min_conf?: number }) =>
    postJSON<Version>('/adapt', { ...(opts?.note ? { note: opts.note } : {}), ...(opts?.assign_min_conf != null ? { assign_min_conf: opts.assign_min_conf } : {}) }),
  versionsCompare: (a: string, b: string) => getJSON<VersionsCompare>(`/versions/compare${qs({ a, b })}`),
  map: () => getJSON<MapResponse>('/map'),
  // ---- field mode (collections / det chips) ----
  collections: () => getJSON<CollectionsResponse>('/collections'),
  collectionsSummary: () => getJSON<CollectionsSummary>('/collections/summary'),
  collectionDetail: (
    groupId: string,
    params: {
      sort?: 'novelty' | 'conf';
      order?: 'asc' | 'desc';
      limit?: number;
      offset?: number;
      only_flagged?: boolean;
      reviewed?: 'all' | 'yes' | 'no';
    },
  ) => getJSON<CollectionDetail>(`/collections/${encodeURIComponent(groupId)}${qs(params)}`),
  detDetail: (detId: number) => getJSON<DetChipDetail>(`/det/${detId}`),
  // Field review queue. Tries the dedicated `/review/queue?source=det`
  // endpoint (backend/API.md, field mode) first. Falls back to building the
  // queue client-side from /collections + /collections/{id} pages when the
  // backend either 404s or hasn't yet learned the `source` param and still
  // returns ground-truth chips (detected by `source !== 'det'` / a
  // `role_true` key — never rendered in analyst view). `strategy`/`min_conf`
  // are newer params (default 'collection' / 0.5 server-side); the fallback
  // path only honors `min_conf` (client-side filter) since it predates
  // `strategy` entirely and always ranks by novelty.
  fieldReviewQueue: async (params: { limit?: number; strategy?: ReviewStrategy; min_conf?: number } = {}): Promise<DetChip[]> => {
    const { limit = 48, strategy, min_conf } = params;
    let items: unknown[] | undefined;
    try {
      const res = await getJSON<unknown>(`/review/queue${qs({ limit, source: 'det', strategy, min_conf })}`);
      items = Array.isArray(res) ? res : Array.isArray((res as { items?: unknown[] })?.items) ? (res as { items: unknown[] }).items : undefined;
    } catch (err) {
      if (!(err instanceof ApiError && err.status === 404)) throw err;
    }
    if (items && items.every((x) => isDetChip(x))) {
      return (items as DetChip[]).slice(0, limit);
    }
    return buildFieldQueueFromCollections(limit, min_conf ?? 0);
  },
  report: async (): Promise<string> => {
    const res = await fetch(`${BASE}/report`);
    if (!res.ok) throw new ApiError(res.status, res.statusText);
    return res.text();
  },
  // ---- "Changed in this version" (analyst view) ----
  changes: (params: { a: string; b: string; limit?: number; offset?: number; group_id?: string; only_unreviewed?: boolean; sort?: ChangesSort }) =>
    getJSON<ChangesResponse>(`/versions/changes${qs({ source: 'det', ...params })}`),
  changesSummary: (a: string, b: string) => getJSON<ChangesSummary>(`/versions/changes/summary${qs({ a, b })}`),
  // ---- Model health · retrain readiness (analyst view; indicative) ----
  health: () => getJSON<HealthResponse>('/health'),
  // ---- Validation set / reference (analyst view landing page) ----
  reference: () => getJSON<ReferenceResponse>('/reference'),
};

function isDetChip(x: unknown): x is DetChip {
  if (!x || typeof x !== 'object') return false;
  const o = x as Record<string, unknown>;
  return o.source === 'det' && typeof o.det_id === 'number' && !('role_true' in o);
}

// Client-side fallback for the field review queue: pulls collections with at
// least one flagged detector chip (most flagged first), pages through each
// one's flagged + unreviewed items, stamps the collection's group_id on each
// chip, and merges/ranks by novelty. Only used when the backend doesn't yet
// serve `/review/queue?source=det`.
async function buildFieldQueueFromCollections(limit: number, minConf = 0): Promise<DetChip[]> {
  const cols = await api.collections();
  const candidates = [...cols.groups].filter((g) => g.n_flagged > 0).sort((a, b) => b.n_flagged - a.n_flagged || (b.flag_rate ?? 0) - (a.flag_rate ?? 0));
  const results: DetChip[] = [];
  for (const g of candidates) {
    if (results.length >= limit * 2) break;
    const page = await api.collectionDetail(g.group_id, { only_flagged: true, sort: 'novelty', order: 'desc', reviewed: 'no', limit: 48, offset: 0 });
    for (const c of page.chips.items) {
      if (minConf > 0 && (c.conf ?? 0) < minConf) continue;
      results.push({ ...c, group_id: c.group_id ?? g.group_id });
    }
  }
  results.sort((a, b) => (b.novelty ?? -1) - (a.novelty ?? -1));
  return results.slice(0, limit);
}
