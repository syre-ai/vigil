import { useEffect, useRef, useState } from 'react';
import { chipImageUrl } from '../api/client';
import type { Review, ReviewStrategy } from '../api/types';
import { CompareTable } from '../components/CompareTable';
import { ChipDrawerContent } from '../components/ChipDrawerContent';
import { InfoPopover } from '../components/InfoPopover';
import { NovelBadge } from '../components/NovelBadge';
import { PendingFeedbackPanel } from '../components/PendingFeedbackPanel';
import { Loading, QueryBoundary } from '../components/QueryState';
import type { ReviewActionsHandle } from '../components/ReviewActions';
import { useAdapt, useBatches, useReviewLog, useReviewQueue, useSummary, useVersionsCompare } from '../hooks/useApi';
import { edgeLabel, isEdgeChip, REVIEW_ACTION_LABELS, score } from '../lib/format';
import { cacheChips, getCachedChip } from '../state/chipCache';

const STRATEGIES: { id: ReviewStrategy; label: string }[] = [
  { id: 'novelty', label: 'Novelty (highest first)' },
  { id: 'uncertainty', label: 'Uncertainty (lowest confidence first)' },
  { id: 'random', label: 'Random' },
];

// Small bounded record of reviews submitted in this browser session, kept
// visible below the queue list for a moment after the chip itself drops out
// (the queue only ever lists unreviewed chips) — see task 2e.
const SESSION_STRIP_LIMIT = 6;

export function ReviewQueue() {
  const [strategy, setStrategy] = useState<ReviewStrategy>('novelty');
  const [includeEdgeChips, setIncludeEdgeChips] = useState(false);
  const queue = useReviewQueue({ limit: 24, strategy, min_partial: includeEdgeChips ? 0 : undefined });
  const reviewLog = useReviewLog();
  const summary = useSummary();
  const batchesQuery = useBatches();
  const [selectedOverride, setSelectedOverride] = useState<number | undefined>(undefined);
  // Set when the analyst opens an already-reviewed chip from the Pending
  // feedback panel below — pins the detail pane to that chip even though
  // it's no longer in the (unreviewed-only) queue.
  const [viewedAnnId, setViewedAnnId] = useState<number | undefined>(undefined);
  const [sessionReviews, setSessionReviews] = useState<Review[]>([]);
  const actionsRef = useRef<ReviewActionsHandle>(null);
  const adapt = useAdapt();
  const [compareVersions, setCompareVersions] = useState<{ a: string; b: string } | undefined>(undefined);

  // Cache every chip payload the queue hands us — the Pending feedback panel
  // and the "reviewed this session" strip need to keep showing a chip's
  // thumbnail/role after it's been reviewed and has dropped out of the queue.
  useEffect(() => {
    if (queue.data) cacheChips(queue.data);
  }, [queue.data]);

  // Derived, not stored: falls back to the top of the (possibly just-refetched)
  // queue whenever the explicitly-picked chip is no longer in it — e.g. right
  // after it's reviewed and drops out — so review auto-advances for free.
  // A pinned `viewedAnnId` (from the Pending feedback panel) always wins,
  // even though a reviewed chip is never itself a member of the queue.
  const selected =
    viewedAnnId !== undefined
      ? viewedAnnId
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
    }
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [selected]);

  async function handleApplyFeedback() {
    const before = summary.data?.current_version;
    const result = await adapt.mutateAsync(undefined);
    if (before) setCompareVersions({ a: before, b: result.id });
  }

  function handleReviewed(review: Review) {
    setSessionReviews((prev) => [review, ...prev.filter((r) => r.ann_id !== review.ann_id)].slice(0, SESSION_STRIP_LIMIT));
  }

  function selectQueueItem(annId: number) {
    setViewedAnnId(undefined);
    setSelectedOverride(annId);
  }

  const reviewedCount = reviewLog.data?.length ?? summary.data?.totals.reviewed ?? 0;
  const pendingCount = reviewLog.data?.filter((r) => r.version_applied == null).length ?? 0;

  const selectedIndex = selected !== undefined ? queue.data?.findIndex((c) => c.ann_id === selected) : undefined;
  const selectedChip = (selectedIndex != null && selectedIndex >= 0 ? queue.data?.[selectedIndex] : undefined) ?? (selected !== undefined ? getCachedChip(selected) : undefined);
  const selectedBatchName = selectedChip ? (batchesQuery.data?.find((b) => b.id === selectedChip.batch_id)?.name ?? selectedChip.batch_id) : undefined;
  const selectedInQueue = selectedIndex != null && selectedIndex >= 0;

  return (
    <div>
      <div className="page-header">
        <div>
          <h1 className="page-title">Review queue</h1>
          <div className="page-sub">Analyst-in-the-loop triage, ranked by the selected strategy.</div>
        </div>
        <div className="btn-group" style={{ alignItems: 'center' }}>
          <span className="mono" style={{ fontSize: 13, color: 'var(--text-dim)' }}>
            {reviewedCount.toLocaleString()} reviews recorded
          </span>
          <label className="inline-toggle" title="Aircraft cut by the tile edge are hidden by default (min_partial=0.75 on the backend)">
            <input type="checkbox" checked={includeEdgeChips} onChange={(e) => setIncludeEdgeChips(e.target.checked)} />
            include edge/partial chips
          </label>
          <select value={strategy} onChange={(e) => setStrategy(e.target.value as ReviewStrategy)}>
            {STRATEGIES.map((s) => (
              <option key={s.id} value={s.id}>
                {s.label}
              </option>
            ))}
          </select>
          <PendingFeedbackPanel reviews={reviewLog.data} onSelect={(annId) => setViewedAnnId(annId)} />
          <button
            type="button"
            className="btn btn-primary"
            onClick={handleApplyFeedback}
            disabled={adapt.isPending || pendingCount === 0}
            title={pendingCount === 0 ? 'No pending feedback to apply' : undefined}
          >
            {adapt.isPending ? 'Applying…' : `Apply ${pendingCount.toLocaleString()} feedback item${pendingCount === 1 ? '' : 's'} → new version`}
          </button>
        </div>
      </div>

      <div className="review-layout">
        <div className="panel review-list-panel">
          <div className="panel-title">
            Queue ({queue.data?.length ?? 0})
            <InfoPopover label="Keyboard shortcuts and edge-chip note">
              <kbd>1</kbd> confirm · <kbd>2</kbd> new class · <kbd>3</kbd> not aircraft · <kbd>4</kbd> correct role · <kbd>s</kbd> skip, applies to the selected chip. Aircraft
              cut by the tile edge are hidden by default; they dominate the novelty ranking for the wrong reason.
            </InfoPopover>
          </div>
          <QueryBoundary isLoading={queue.isLoading} error={queue.error} data={queue.data} loadingLabel="Loading queue…">
            {(items) =>
              items.length === 0 ? (
                <div className="state-block">Queue is empty. Every chip has been reviewed under this strategy.</div>
              ) : (
                <ul className="review-list">
                  {items.map((c) => (
                    <li key={c.ann_id}>
                      <button type="button" className={`review-list-item${c.ann_id === selected ? ' active' : ''}`} onClick={() => selectQueueItem(c.ann_id)}>
                        <img src={chipImageUrl(c.ann_id)} alt="" width={40} height={40} loading="lazy" />
                        <span className="review-list-meta">
                          <span className="review-list-row1">
                            <span className="review-list-role">{c.role_pred}</span>
                            {isEdgeChip(c) && <span className="chip-card-tag chip-card-tag-edge">{edgeLabel(c)}</span>}
                          </span>
                          <span className="mono" title={`role confidence ${score(c.role_conf, 2)}`}>
                            novelty {c.novelty.toFixed(2)}
                          </span>
                        </span>
                        {c.is_novel && <NovelBadge />}
                      </button>
                    </li>
                  ))}
                </ul>
              )
            }
          </QueryBoundary>
          {sessionReviews.length > 0 && (
            <div className="review-session-strip">
              <div className="review-session-strip-title">Reviewed this session</div>
              <ul className="review-session-list">
                {sessionReviews.map((r) => {
                  const chip = getCachedChip(r.ann_id);
                  return (
                    <li key={r.ann_id} className="review-session-item" title={`#${r.ann_id} · ${REVIEW_ACTION_LABELS[r.action]}${r.label ? ` → ${r.label}` : ''}`}>
                      {chip ? (
                        <img src={chipImageUrl(chip.ann_id)} alt="" width={26} height={26} loading="lazy" />
                      ) : (
                        <span className="review-session-thumb-skel" />
                      )}
                      <span className="review-session-tag">{REVIEW_ACTION_LABELS[r.action]}</span>
                    </li>
                  );
                })}
              </ul>
            </div>
          )}
        </div>

        <div className="panel review-detail-panel">
          {queue.isLoading ? (
            <Loading label="Loading queue…" />
          ) : selected !== undefined ? (
            <>
              <div className="review-detail-head">
                <span>{selectedInQueue ? `Chip ${(selectedIndex ?? 0) + 1} of ${queue.data?.length ?? 0}` : 'Reviewed chip (not in current queue)'}</span>
                {selectedBatchName && <span className="review-detail-batch">{selectedBatchName}</span>}
              </div>
              <ChipDrawerContent annId={selected} actionsRef={actionsRef} onReviewed={handleReviewed} />
            </>
          ) : (
            <div className="state-block">Queue is empty. Every chip has been reviewed under this strategy.</div>
          )}
        </div>
      </div>

      {compareVersions && <CompareSection a={compareVersions.a} b={compareVersions.b} />}
    </div>
  );
}

// Kept separate so the compare query only mounts once a version pair exists.
function CompareSection({ a, b }: { a: string; b: string }) {
  const compare = useVersionsCompare(a, b);
  return (
    <div>
      <div className="section-gap" />
      <QueryBoundary isLoading={compare.isLoading} error={compare.error} data={compare.data} loadingLabel="Computing regression report…">
        {(data) => <CompareTable compare={data} />}
      </QueryBoundary>
    </div>
  );
}
