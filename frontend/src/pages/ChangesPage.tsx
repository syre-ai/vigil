import { useEffect, useRef, useState } from 'react';
import { ChevronLeft, ChevronRight } from 'lucide-react';
import type { ChangedDetChip, ChangesSort } from '../api/types';
import { ChangedChipGrid } from '../components/ChangedChipGrid';
import { ChipDrawerContent } from '../components/ChipDrawerContent';
import { InfoPopover } from '../components/InfoPopover';
import { Modal } from '../components/Modal';
import { QueryBoundary } from '../components/QueryState';
import type { ReviewActionsHandle } from '../components/ReviewActions';
import { useChanges, useChangesSummary, useSummary } from '../hooks/useApi';
import { usePersistedState } from '../hooks/usePersistedState';
import { count, pct } from '../lib/format';

const PAGE_SIZE = 48;

// Det-chip drawer in a modal with prev/next over the current grid page — same review-key
// convention as CollectionPage/FieldReview's own det-chip modals: 1 confirms the NEW (re-sorted)
// class, 4 corrects the role, 3 marks not-aircraft, 5 marks a bad box.
function ChangedDetChipModal({ items, detId, onSelect, onClose }: { items: ChangedDetChip[]; detId: number; onSelect: (id: number) => void; onClose: () => void }) {
  const actionsRef = useRef<ReviewActionsHandle>(null);
  const idx = items.findIndex((c) => c.ann_id === detId);
  const prev = idx > 0 ? items[idx - 1] : undefined;
  const next = idx >= 0 && idx < items.length - 1 ? items[idx + 1] : undefined;

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.repeat) return;
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
      if (e.key === '1') actionsRef.current?.trigger('confirm');
      else if (e.key === '3') actionsRef.current?.trigger('not_aircraft');
      else if (e.key === '4') actionsRef.current?.trigger('correct');
      else if (e.key === '5') actionsRef.current?.trigger('bad_box');
      else if (e.key === 's' || e.key === 'S') actionsRef.current?.trigger('skip');
      else if (e.key === 'ArrowLeft' && prev) onSelect(prev.ann_id);
      else if (e.key === 'ArrowRight' && next) onSelect(next.ann_id);
    }
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [prev, next, onSelect]);

  return (
    <Modal onClose={onClose} width={920}>
      <div className="drawer-nav">
        <button type="button" className="btn btn-sm" disabled={!prev} onClick={() => prev && onSelect(prev.ann_id)} title="Previous chip (←)">
          <ChevronLeft size={13} strokeWidth={2} /> prev
        </button>
        <span className="mono">{idx >= 0 ? `Changed detection ${idx + 1} of ${items.length} on this page` : 'Detection (not on this page)'}</span>
        <button type="button" className="btn btn-sm" disabled={!next} onClick={() => next && onSelect(next.ann_id)} title="Next chip (→)">
          next <ChevronRight size={13} strokeWidth={2} />
        </button>
      </div>
      <ChipDrawerContent annId={detId} source="det" actionsRef={actionsRef} />
    </Modal>
  );
}

export function ChangesPage() {
  const summary = useSummary();
  const versions = summary.data?.versions ?? [];
  const hasAdaptedVersion = versions.length > 1;
  const a = versions[0]?.id;
  const b = summary.data?.current_version;

  const [groupId, setGroupId] = usePersistedState<string>('vigil.changes.groupId', '');
  const [onlyUnreviewed, setOnlyUnreviewed] = usePersistedState<boolean>('vigil.changes.onlyUnreviewed', false);
  const [sort, setSort] = usePersistedState<ChangesSort>('vigil.changes.sort', 'sim');
  const [offset, setOffset] = useState(0);
  const [openDetId, setOpenDetId] = useState<number | null>(null);

  const changes = useChanges({
    a: hasAdaptedVersion ? a : undefined,
    b: hasAdaptedVersion ? b : undefined,
    limit: PAGE_SIZE,
    offset,
    group_id: groupId || undefined,
    only_unreviewed: onlyUnreviewed,
    sort,
  });
  const changesSummary = useChangesSummary(hasAdaptedVersion ? a : undefined, hasAdaptedVersion ? b : undefined);

  if (summary.isLoading) {
    return <div className="state-block">Loading…</div>;
  }
  if (!hasAdaptedVersion) {
    return <div className="state-block">No adapted version yet. Apply feedback first.</div>;
  }

  return (
    <div>
      <div className="page-header">
        <div>
          <h1 className="page-title">
            Changed in {b} (vs {a})
          </h1>
          <div className="page-sub">
            {changes.data ? `${count(changes.data.total)} detections re-sorted to analyst classes. Verify them` : 'Loading changed detections…'}
          </div>
        </div>
      </div>


      {changesSummary.data && (
        <div className="panel">
          <div className="key-facts-line">
            <span>
              <span className="mono">{count(changesSummary.data.n_reviewed)}</span> of <span className="mono">{count(changesSummary.data.total)}</span> reviewed
            </span>
            {changesSummary.data.n_reviewed > 0 && (
              <span>
                verified precision <span className="mono">{pct(changesSummary.data.verified_precision)}</span>
                <InfoPopover label="About verified precision">
                  <p className="info-popover-text">
                    Of the changed detections an analyst has reviewed, the fraction confirmed as the new class. The rest were corrected to a different role, marked not aircraft, or
                    flagged as a bad box.
                  </p>
                </InfoPopover>
              </span>
            )}
          </div>
        </div>
      )}

      <div className="section-gap" />

      <div className="panel">
        <div className="chip-controls">
          <div className="panel-title" style={{ margin: 0 }}>
            Changed detections
            <InfoPopover label="About this list" width={320}>
              <p className="info-popover-text">
                Detector chips whose predicted role changed between {a} and {b}. Almost always detections the vigilance-gated adaptation rule newly claimed for an analyst class.
                Chips from the same collection as a labeled example are muted: correlated with, not independent confirmation of, the new class.
              </p>
            </InfoPopover>
          </div>
          <div className="chip-controls-right">
            <label className="inline-toggle">
              <input
                type="checkbox"
                checked={onlyUnreviewed}
                onChange={(e) => {
                  setOffset(0);
                  setOnlyUnreviewed(e.target.checked);
                }}
              />
              only unreviewed
            </label>
            <select
              value={groupId}
              aria-label="Collection filter"
              onChange={(e) => {
                setOffset(0);
                setGroupId(e.target.value);
              }}
            >
              <option value="">all collections</option>
              {(changes.data?.by_group ?? []).map((g) => (
                <option key={g.group_id} value={g.group_id}>
                  {g.airfield} · {g.date} ({count(g.n_changed)})
                </option>
              ))}
            </select>
            <select
              value={sort}
              aria-label="Sort"
              onChange={(e) => {
                setOffset(0);
                setSort(e.target.value as ChangesSort);
              }}
            >
              <option value="sim">sort: similarity</option>
              <option value="novelty">sort: novelty</option>
              <option value="conf">sort: detector confidence</option>
            </select>
          </div>
        </div>

        <QueryBoundary isLoading={changes.isLoading} error={changes.error} data={changes.data} loadingLabel="Loading changed detections…">
          {(data) => (
            <>
              <ChangedChipGrid chips={data.items} onSelect={(id) => setOpenDetId(id)} />
              {data.total > 0 && (
                <div className="pager mono">
                  <button type="button" className="btn btn-sm" disabled={offset === 0} onClick={() => setOffset((o) => Math.max(0, o - PAGE_SIZE))}>
                    ← prev
                  </button>
                  <span>
                    {Math.min(offset + 1, data.total)}–{Math.min(offset + PAGE_SIZE, data.total)} of {count(data.total)}
                  </span>
                  <button type="button" className="btn btn-sm" disabled={offset + PAGE_SIZE >= data.total} onClick={() => setOffset((o) => o + PAGE_SIZE)}>
                    next →
                  </button>
                </div>
              )}
            </>
          )}
        </QueryBoundary>
      </div>

      {openDetId !== null && changes.data && <ChangedDetChipModal items={changes.data.items} detId={openDetId} onSelect={setOpenDetId} onClose={() => setOpenDetId(null)} />}
    </div>
  );
}
