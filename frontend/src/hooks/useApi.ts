import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { api } from '../api/client';
import type { ChangesSort, ReviewAction, ReviewSource, ReviewStrategy } from '../api/types';

export const qk = {
  summary: ['summary'] as const,
  batches: ['batches'] as const,
  batchDetail: (id: string) => ['batch', id] as const,
  batchChips: (id: string, params: unknown) => ['batchChips', id, params] as const,
  chipDetail: (annId: number) => ['chip', annId] as const,
  tileDetail: (tileId: number) => ['tile', tileId] as const,
  embedding: (batch: string) => ['embedding', batch] as const,
  drift: ['drift'] as const,
  envelope: ['envelope'] as const,
  evalReport: ['eval'] as const,
  reviewQueue: (params: unknown) => ['reviewQueue', params] as const,
  reviewLog: (source?: ReviewSource | 'all') => (source ? (['reviewLog', source] as const) : (['reviewLog'] as const)),
  usedLabels: ['usedLabels'] as const,
  reviewClasses: ['reviewClasses'] as const,
  compare: (a: string, b: string) => ['compare', a, b] as const,
  map: ['map'] as const,
  report: ['report'] as const,
  collections: ['collections'] as const,
  collectionsSummary: ['collectionsSummary'] as const,
  collectionDetail: (groupId: string, params: unknown) => ['collectionDetail', groupId, params] as const,
  detDetail: (detId: number) => ['detDetail', detId] as const,
  fieldReviewQueue: (params: unknown) => ['fieldReviewQueue', params] as const,
  changes: (params: unknown) => ['changes', params] as const,
  changesSummary: (a: string, b: string) => ['changesSummary', a, b] as const,
  health: ['health'] as const,
  reference: ['reference'] as const,
};

export function useSummary() {
  return useQuery({ queryKey: qk.summary, queryFn: api.summary });
}

export function useBatches() {
  return useQuery({ queryKey: qk.batches, queryFn: api.batches });
}

export function useBatchDetail(id: string | undefined) {
  return useQuery({ queryKey: qk.batchDetail(id ?? ''), queryFn: () => api.batchDetail(id as string), enabled: !!id });
}

export function useBatchChips(
  id: string | undefined,
  params: {
    sort?: 'novelty' | 'role_conf' | 'knn_dist';
    order?: 'asc' | 'desc';
    limit?: number;
    offset?: number;
    only_novel?: boolean;
    reviewed?: 'all' | 'yes' | 'no';
    detected?: 'all' | 'yes' | 'no';
    min_partial?: number;
  },
) {
  return useQuery({
    queryKey: qk.batchChips(id ?? '', params),
    queryFn: () => api.batchChips(id as string, params),
    enabled: !!id,
    placeholderData: (prev) => prev,
  });
}

export function useChipDetail(annId: number | undefined) {
  return useQuery({ queryKey: qk.chipDetail(annId ?? -1), queryFn: () => api.chipDetail(annId as number), enabled: annId !== undefined });
}

export function useTileDetail(tileId: number | undefined) {
  return useQuery({ queryKey: qk.tileDetail(tileId ?? -1), queryFn: () => api.tileDetail(tileId as number), enabled: tileId !== undefined });
}

export function useEmbeddingPoints(batch: string) {
  return useQuery({
    queryKey: qk.embedding(batch),
    queryFn: () => api.embeddingPoints(batch),
    enabled: !!batch,
    staleTime: Infinity, // reference+batch points never change for a given version — fetch once, cache per batch
    placeholderData: (prev) => prev, // keep the previous batch's points (and its identical reference layer) visible while the new batch loads
  });
}

export function useDrift() {
  return useQuery({ queryKey: qk.drift, queryFn: api.drift });
}

export function useEnvelope() {
  return useQuery({ queryKey: qk.envelope, queryFn: api.envelope });
}

export function useEval() {
  return useQuery({ queryKey: qk.evalReport, queryFn: api.evalReport });
}

export function useReviewQueue(params: { limit?: number; strategy?: ReviewStrategy; min_partial?: number }) {
  return useQuery({ queryKey: qk.reviewQueue(params), queryFn: () => api.reviewQueue(params) });
}

export function useReviewLog(source?: ReviewSource | 'all') {
  return useQuery({ queryKey: qk.reviewLog(source), queryFn: () => api.reviewLog(source) });
}

export function useUsedLabels() {
  return useQuery({ queryKey: qk.usedLabels, queryFn: api.usedLabels });
}

export function useReviewClasses() {
  return useQuery({ queryKey: qk.reviewClasses, queryFn: api.reviewClasses });
}

export function usePostReview() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ annId, action, label, note, source }: { annId: number; action: ReviewAction; label?: string; note?: string; source?: ReviewSource }) =>
      api.postReview(annId, { action, label, note }, source),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['reviewQueue'] });
      qc.invalidateQueries({ queryKey: ['batchChips'] });
      qc.invalidateQueries({ queryKey: ['reviewLog'] }); // prefix match covers both qk.reviewLog() and qk.reviewLog('det')
      qc.invalidateQueries({ queryKey: qk.usedLabels });
      qc.invalidateQueries({ queryKey: qk.summary });
      qc.invalidateQueries({ queryKey: ['chip'] });
      // field-mode invalidations — cheap no-ops when field mode isn't in use
      qc.invalidateQueries({ queryKey: ['detDetail'] });
      qc.invalidateQueries({ queryKey: ['collectionDetail'] });
      qc.invalidateQueries({ queryKey: ['collections'] });
      qc.invalidateQueries({ queryKey: ['collectionsSummary'] });
      qc.invalidateQueries({ queryKey: ['fieldReviewQueue'] });
      qc.invalidateQueries({ queryKey: ['changes'] });
      qc.invalidateQueries({ queryKey: ['changesSummary'] });
      qc.invalidateQueries({ queryKey: qk.health });
    },
  });
}

export function useAdapt() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (opts?: { note?: string; assign_min_conf?: number }) => api.adapt(opts),
    onSuccess: () => {
      qc.invalidateQueries();
    },
  });
}

export function useVersionsCompare(a: string | undefined, b: string | undefined) {
  return useQuery({ queryKey: qk.compare(a ?? '', b ?? ''), queryFn: () => api.versionsCompare(a as string, b as string), enabled: !!a && !!b });
}

export function useMapPoints() {
  // Parent-image locations/domains for a given version don't change between
  // renders — a 5 min staleTime avoids an unnecessary refetch (and the
  // marker-duplication window that a refetch before style load can cause).
  return useQuery({ queryKey: qk.map, queryFn: api.map, staleTime: 5 * 60 * 1000 });
}

export function useReport() {
  return useQuery({ queryKey: qk.report, queryFn: api.report });
}

// ---- field mode (collections / det chips) ----

export function useCollections() {
  return useQuery({ queryKey: qk.collections, queryFn: api.collections });
}

export function useCollectionsSummary() {
  return useQuery({ queryKey: qk.collectionsSummary, queryFn: api.collectionsSummary });
}

export function useCollectionDetail(
  groupId: string | undefined,
  params: {
    sort?: 'novelty' | 'conf';
    order?: 'asc' | 'desc';
    limit?: number;
    offset?: number;
    only_flagged?: boolean;
    reviewed?: 'all' | 'yes' | 'no';
  },
) {
  return useQuery({
    queryKey: qk.collectionDetail(groupId ?? '', params),
    queryFn: () => api.collectionDetail(groupId as string, params),
    enabled: !!groupId,
    placeholderData: (prev) => prev,
  });
}

export function useDetDetail(detId: number | undefined) {
  return useQuery({ queryKey: qk.detDetail(detId ?? -1), queryFn: () => api.detDetail(detId as number), enabled: detId !== undefined });
}

export function useFieldReviewQueue(params: { limit?: number; strategy?: ReviewStrategy; min_conf?: number } = {}) {
  return useQuery({ queryKey: qk.fieldReviewQueue(params), queryFn: () => api.fieldReviewQueue(params) });
}

// ---- "Changed in this version" (analyst view) ----

export function useChanges(params: {
  a?: string;
  b?: string;
  limit?: number;
  offset?: number;
  group_id?: string;
  only_unreviewed?: boolean;
  sort?: ChangesSort;
}) {
  const { a, b, ...rest } = params;
  return useQuery({
    queryKey: qk.changes(params),
    queryFn: () => api.changes({ a: a as string, b: b as string, ...rest }),
    enabled: !!a && !!b,
    placeholderData: (prev) => prev,
  });
}

export function useChangesSummary(a: string | undefined, b: string | undefined) {
  return useQuery({
    queryKey: qk.changesSummary(a ?? '', b ?? ''),
    queryFn: () => api.changesSummary(a as string, b as string),
    enabled: !!a && !!b,
  });
}

// ---- Model health · retrain readiness (analyst view; indicative) ----

export function useHealth() {
  return useQuery({ queryKey: qk.health, queryFn: api.health });
}

// ---- Validation set / reference (analyst view landing page) ----

export function useReference() {
  return useQuery({ queryKey: qk.reference, queryFn: api.reference });
}
