import { useEffect, useMemo, useRef, useState } from 'react';
import type { CollectionGroup, CollectionStatus, DetChip, ReviewStrategy } from '../api/types';
import { ChipDrawerContent } from '../components/ChipDrawerContent';
import { FieldCompareTable } from '../components/FieldCompareTable';
import { FieldPendingFeedbackPanel } from '../components/FieldPendingFeedbackPanel';
import { InfoPopover } from '../components/InfoPopover';
import { NovelBadge } from '../components/NovelBadge';
import { Loading, QueryBoundary } from '../components/QueryState';
import type { ReviewActionsHandle } from '../components/ReviewActions';
import { usePersistedState } from '../hooks/usePersistedState';
import { useAdapt, useCollections, useCollectionsSummary, useFieldReviewQueue, useReviewLog, useSummary, useVersionsCompare } from '../hooks/useApi';
import { NA, collectionStatusLabel, count, pct, score } from '../lib/format';
import { cacheDetChips, getCachedDetChip } from '../state/detChipCache';
import { useNav } from '../state/useNav';
import { useToast } from '../state/useToast';

const STRATEGIES: { id: ReviewStrategy; label: string }[] = [
  { id: 'collection', label: 'By collection (needs review first)' },
  { id: 'novelty', label: 'Novelty (all collections)' },
  { id: 'uncertainty', label: 'Uncertainty' },
  { id: 'random', label: 'Random' },
];

const MIN_CONF_OPTIONS = [0, 0.3, 0.5, 0.7];

// Optional ADDITIONAL gate on top of the nearest-exemplar assignment rule (backend
// vigil.pipeline.ADAPTATION_RULE), det chips only — off (0) by default, since the exemplar rule
// already removes most of the junk on its own. Separate from MIN_CONF_OPTIONS above, which only
// filters what this queue SHOWS the analyst, not what an analyst class may later CLAIM.
const ASSIGN_MIN_CONF_OPTIONS = [0, 0.4, 0.5];

// "Airfield · date" for a det chip, via its group_id when the queue carries
// one (or the fallback builder stamped it), else the chip's own airfield.
function collectionLabel(chip: DetChip | undefined, groups: Map<string, CollectionGroup>): string | undefined {
  if (!chip) return undefined;
  const g = chip.group_id ? groups.get(chip.group_id) : undefined;
  if (g) return `${g.airfield} · ${g.date}`;
  return chip.meta?.airfield;
}

// One queue row, shared between the flat (novelty/uncertainty/random) and
// grouped (strategy=collection) renderings. `collectionText`, when the caller
// passes it (flat list), shows the collection under the role — omitted
// inside a collection section, since the section header already says it.
function FieldQueueItem({
  chip: c,
  active,
  onSelect,
  collectionText,
}: {
  chip: DetChip;
  active: boolean;
  onSelect: () => void;
  collectionText?: string | null;
}) {
  return (
    <li id={`field-queue-item-${c.ann_id}`}>
      <button type="button" className={`review-list-item${active ? ' active' : ''}`} onClick={onSelect}>
        <img src={c.chip_url} alt="" width={40} height={40} loading="lazy" />
        <span className="review-list-meta">
          <span className="review-list-row1">
            <span className="review-list-role">{c.role_pred ?? 'not yet scored'}</span>
          </span>
          {collectionText !== null && (
            <span className="review-list-collection" title={collectionText ?? undefined}>
              {collectionText ?? NA}
            </span>
          )}
          <span className="mono" title={`role confidence ${score(c.role_conf, 2)} · detector confidence ${score(c.conf, 2)}`}>
            novelty {score(c.novelty, 2)} · conf {score(c.role_conf, 2)}
          </span>
        </span>
        {c.is_novel && <NovelBadge label="FLAGGED" />}
      </button>
    </li>
  );
}

type SectionHeader = { airfield: string; date: string; status: CollectionStatus | null; nFlagged: number | null };
type QueueSection = { key: string; header: SectionHeader | undefined; items: DetChip[] };

// Groups an already-collection-ordered queue (strategy=collection: needs-review
// collections first, then novelty within each) into contiguous per-collection
// sections. Reads each item's own flat airfield/date/group_status/group_n_flagged
// fields (newer backend) and falls back to the /collections list (older backend
// that returns `group_id` only) — a section with neither renders without a header.
function buildQueueSections(items: DetChip[], groups: Map<string, CollectionGroup>): QueueSection[] {
  const sections: QueueSection[] = [];
  for (const item of items) {
    const key = item.group_id ?? `__ungrouped_${item.ann_id}`;
    let section = sections[sections.length - 1];
    if (!section || section.key !== key) {
      const g = item.group_id ? groups.get(item.group_id) : undefined;
      const airfield = item.airfield ?? g?.airfield;
      const date = item.date ?? g?.date;
      const header: SectionHeader | undefined = airfield && date ? { airfield, date, status: item.group_status ?? g?.status ?? null, nFlagged: item.group_n_flagged ?? g?.n_flagged ?? null } : undefined;
      section = { key, header, items: [] };
      sections.push(section);
    }
    section.items.push(item);
  }
  return sections;
}

function sectionHeaderText(h: SectionHeader): string {
  const parts = [h.airfield, h.date];
  if (h.status) parts.push(collectionStatusLabel(h.status));
  if (h.nFlagged != null) parts.push(`${count(h.nFlagged)} flagged`);
  return parts.join(' · ');
}

export function FieldReview() {
  const { goto } = useNav();
  const [strategy, setStrategy] = usePersistedState<ReviewStrategy>('vigil.field.strategy', 'collection');
  const [minConf, setMinConf] = usePersistedState<number>('vigil.field.minConf', 0.5);
  const [assignMinConf, setAssignMinConf] = usePersistedState<number>('vigil.field.assignMinConf', 0);
  const queue = useFieldReviewQueue({ limit: 48, strategy, min_conf: minConf });
  const reviewLog = useReviewLog('det');
  const collectionsSummary = useCollectionsSummary();
  const collections = useCollections();
  const summary = useSummary();
  const pushToast = useToast();
  const [selectedOverride, setSelectedOverride] = useState<number | undefined>(undefined);
  const [viewedDetId, setViewedDetId] = useState<number | undefined>(undefined);
  const actionsRef = useRef<ReviewActionsHandle>(null);
  const adapt = useAdapt();
  const [compareVersions, setCompareVersions] = useState<{ a: string; b: string } | undefined>(undefined);

  useEffect(() => {
    if (queue.data) cacheDetChips(queue.data);
  }, [queue.data]);

  const groupsById = useMemo(() => new Map((collections.data?.groups ?? []).map((g) => [g.group_id, g])), [collections.data]);

  const selected =
    viewedDetId !== undefined
      ? viewedDetId
      : selectedOverride !== undefined && queue.data?.some((c) => c.ann_id === selectedOverride)
        ? selectedOverride
        : queue.data?.[0]?.ann_id;

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (selected === undefined) return;
      if (e.repeat) return;
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
      if (e.key === '1') actionsRef.current?.trigger('confirm');
      else if (e.key === '2') actionsRef.current?.trigger('new_class');
      else if (e.key === '3') actionsRef.current?.trigger('not_aircraft');
      else if (e.key === '4') actionsRef.current?.trigger('correct');
      else if (e.key === '5') actionsRef.current?.trigger('bad_box');
      else if (e.key === 's' || e.key === 'S') actionsRef.current?.trigger('skip');
      else if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
        // Moves the selection through the queue in its current (backend-ranked)
        // order — with strategy=collection that order already crosses section
        // boundaries seamlessly, so this is the same code path either way.
        const items = queue.data;
        if (!items || items.length === 0) return;
        const idx = items.findIndex((c) => c.ann_id === selected);
        const delta = e.key === 'ArrowDown' ? 1 : -1;
        const nextIdx = idx === -1 ? 0 : Math.min(Math.max(idx + delta, 0), items.length - 1);
        if (nextIdx === idx) return;
        e.preventDefault();
        const nextId = items[nextIdx].ann_id;
        selectQueueItem(nextId);
        document.getElementById(`field-queue-item-${nextId}`)?.scrollIntoView({ block: 'nearest' });
      }
    }
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [selected, queue.data]);

  async function handleApplyFeedback() {
    const before = summary.data?.current_version;
    try {
      const result = await adapt.mutateAsync({ assign_min_conf: assignMinConf });
      pushToast(`Applied feedback → ${result.id} (${count(result.n_feedback)} item${result.n_feedback === 1 ? '' : 's'}). Re-scored every detection and re-checked the labeled control set.`);
      if (before) setCompareVersions({ a: before, b: result.id });
    } catch (err) {
      pushToast(`Apply failed: ${err instanceof Error ? err.message : 'unknown error'}`);
    }
  }

  function selectQueueItem(id: number) {
    setViewedDetId(undefined);
    setSelectedOverride(id);
  }

  const pendingCount = reviewLog.data?.filter((r) => r.version_applied == null).length ?? 0;
  const selectedIndex = selected !== undefined ? queue.data?.findIndex((c) => c.ann_id === selected) : undefined;
  const selectedInQueue = selectedIndex != null && selectedIndex >= 0;
  const selectedChip = (selectedInQueue ? queue.data?.[selectedIndex as number] : undefined) ?? (selected !== undefined ? getCachedDetChip(selected) : undefined);
  const selectedCollection = collectionLabel(selectedChip, groupsById);
  const selectedGroupId = selectedChip?.group_id ?? undefined;

  return (
    <div>
      <div className="page-header">
        <div>
          <h1 className="page-title">Review (field)</h1>
          <div className="page-sub">Flagged detections across new collections, ranked by the selected strategy. The analyst decides; the model only finds.</div>
        </div>
        <div className="btn-group" style={{ alignItems: 'center' }}>
          {collectionsSummary.data && (
            <span className="mono" style={{ fontSize: 13, color: 'var(--text-dim)' }}>
              {count(collectionsSummary.data.n_flagged)} flagged of {count(collectionsSummary.data.n_detections)} detections · {count(collectionsSummary.data.n_needs_review)} collections need
              review
            </span>
          )}
          <select value={strategy} onChange={(e) => setStrategy(e.target.value as ReviewStrategy)} title="Queue ranking strategy">
            {STRATEGIES.map((s) => (
              <option key={s.id} value={s.id}>
                {s.label}
              </option>
            ))}
          </select>
          <select
            value={minConf}
            onChange={(e) => setMinConf(Number(e.target.value))}
            title="Minimum detector confidence for chips in this queue: filters out the lowest-confidence detector false positives"
          >
            {MIN_CONF_OPTIONS.map((v) => (
              <option key={v} value={v}>
                min detector confidence {v === 0 ? '0 (any)' : v}
              </option>
            ))}
          </select>
          <FieldPendingFeedbackPanel reviews={reviewLog.data} onSelect={(id) => setViewedDetId(id)} />
          <select
            value={assignMinConf}
            onChange={(e) => setAssignMinConf(Number(e.target.value))}
            title="Optional additional gate on top of the nearest-exemplar assignment rule: a detector chip may only be claimed if its own detection confidence also clears this threshold."
          >
            {ASSIGN_MIN_CONF_OPTIONS.map((v) => (
              <option key={v} value={v}>
                assign only to detections with confidence ≥ {v === 0 ? 'none' : v}
              </option>
            ))}
          </select>
          <InfoPopover label="About the assignment confidence gate">
            Detector false positives are novel by definition and cluster at low confidence; a higher gate keeps the new class cleaner but can skip fused or poorly framed detections. Recorded
            with the version.
          </InfoPopover>
          <button
            type="button"
            className="btn btn-primary"
            onClick={handleApplyFeedback}
            disabled={adapt.isPending || pendingCount === 0}
            title={
              pendingCount === 0
                ? 'No pending feedback to apply'
                : 'Builds the next version: analyst-labeled exemplars from new-class labels (a detection joins a class only via the nearest-exemplar rule, consulted only for detections the trained model already rejected as novel), re-scores every detection, re-checks the labeled control set.'
            }
          >
            {adapt.isPending ? 'Applying…' : `Apply ${count(pendingCount)} feedback item${pendingCount === 1 ? '' : 's'} → new version`}
          </button>
        </div>
      </div>

      <div className="review-layout">
        <div className="panel review-list-panel">
          <div className="panel-title">
            Flagged queue ({count(queue.data?.length ?? 0)})
            <InfoPopover label="About detector false positives in this queue">
              <p className="info-popover-text">
                Detector false positives are novel by definition and will appear here; mark them with <kbd>3</kbd> (not aircraft). Those marks are recorded as evidence of field
                precision on flagged detections.
              </p>
            </InfoPopover>
            <InfoPopover label="About this queue and keyboard shortcuts" width={320}>
              <p className="info-popover-text">
                Unreviewed detector chips that failed the vigilance test, across every new collection. Reviewing a chip removes it from the queue; the collection it came from is
                shown under the predicted role (and as a section header, when sorted by collection).
              </p>
              <p className="info-popover-text">
                <kbd>1</kbd> confirm · <kbd>2</kbd> new class · <kbd>3</kbd> not aircraft · <kbd>4</kbd> correct role · <kbd>s</kbd> skip, applies to the selected chip. Arrow
                keys move the selection up/down the queue.
              </p>
            </InfoPopover>
          </div>
          <QueryBoundary isLoading={queue.isLoading} error={queue.error} data={queue.data} loadingLabel="Loading queue…">
            {(items) =>
              items.length === 0 ? (
                <div className="state-block">No flagged detections pending review across the new collections.</div>
              ) : strategy === 'collection' ? (
                <div className="review-list">
                  {buildQueueSections(items, groupsById).map((sec) => (
                    <div key={sec.key} className="review-list-section">
                      {sec.header && <div className="review-list-section-header">{sectionHeaderText(sec.header)}</div>}
                      <ul className="review-list">
                        {sec.items.map((c) => (
                          <FieldQueueItem key={c.ann_id} chip={c} active={c.ann_id === selected} onSelect={() => selectQueueItem(c.ann_id)} collectionText={null} />
                        ))}
                      </ul>
                    </div>
                  ))}
                </div>
              ) : (
                <ul className="review-list">
                  {items.map((c) => (
                    <FieldQueueItem key={c.ann_id} chip={c} active={c.ann_id === selected} onSelect={() => selectQueueItem(c.ann_id)} collectionText={collectionLabel(c, groupsById)} />
                  ))}
                </ul>
              )
            }
          </QueryBoundary>
        </div>

        <div className="panel review-detail-panel">
          {queue.isLoading ? (
            <Loading label="Loading queue…" />
          ) : selected !== undefined ? (
            <>
              <div className="review-detail-head">
                <span>{selectedInQueue ? `Chip ${(selectedIndex ?? 0) + 1} of ${count(queue.data?.length ?? 0)}` : 'Reviewed chip (not in current queue)'}</span>
                {selectedCollection &&
                  (selectedGroupId ? (
                    <button type="button" className="btn btn-sm" onClick={() => goto('collection', selectedGroupId)} title="Open this collection">
                      {selectedCollection}
                    </button>
                  ) : (
                    <span className="review-detail-batch">{selectedCollection}</span>
                  ))}
              </div>
              <ChipDrawerContent annId={selected} source="det" actionsRef={actionsRef} />
            </>
          ) : (
            <div className="state-block">No flagged detections pending review across the new collections.</div>
          )}
        </div>
      </div>

      {compareVersions && <FieldCompareSection a={compareVersions.a} b={compareVersions.b} groups={groupsById} referenceFlagRate={collectionsSummary.data?.reference_flag_rate} />}
    </div>
  );
}

function FieldCompareSection({
  a,
  b,
  groups,
  referenceFlagRate,
}: {
  a: string;
  b: string;
  groups: Map<string, CollectionGroup>;
  referenceFlagRate: number | null | undefined;
}) {
  const compare = useVersionsCompare(a, b);
  const ref = useRef<HTMLDivElement | null>(null);
  // The compare lands below the queue/detail split — bring it into view once
  // it has data so the analyst doesn't have to hunt for it after Apply.
  useEffect(() => {
    if (compare.data) ref.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }, [compare.data]);
  return (
    <div ref={ref}>
      <div className="section-gap" />
      <QueryBoundary isLoading={compare.isLoading} error={compare.error} data={compare.data} loadingLabel="Computing field comparison…">
        {(data) => <FieldCompareTable compare={data} groups={groups} />}
      </QueryBoundary>
      {referenceFlagRate != null && compare.data && (
        <div className="caption" style={{ marginTop: 6 }}>
          Validation-set flag rate before this change: {pct(referenceFlagRate)}. Collection statuses on the Collections page are recomputed from the new version&apos;s scores.
        </div>
      )}
    </div>
  );
}
