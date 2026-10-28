import { useEffect, useRef, useState } from 'react';
import { ChevronLeft, ChevronRight } from 'lucide-react';
import type { CollectionDetail, DetChip } from '../api/types';
import { ChipDrawerContent } from '../components/ChipDrawerContent';
import { ChipGrid } from '../components/ChipGrid';
import { CollectionStatusPill } from '../components/CollectionStatusPill';
import { CollectionTiles } from '../components/CollectionTiles';
import { InfoPopover } from '../components/InfoPopover';
import { Modal } from '../components/Modal';
import { QueryBoundary } from '../components/QueryState';
import type { ReviewActionsHandle } from '../components/ReviewActions';
import { useCollectionDetail, useCollections } from '../hooks/useApi';
import { usePersistedState } from '../hooks/usePersistedState';
import { NA, count, pct, pValue, score } from '../lib/format';
import { cacheDetChips } from '../state/detChipCache';
import { useNav } from '../state/useNav';

type SortKey = 'novelty' | 'conf';
type ReviewedFilter = 'all' | 'yes' | 'no';

const PAGE_SIZE = 48;

function CollectionHeader({ col, referenceFlagRate }: { col: CollectionDetail; referenceFlagRate: number | null | undefined }) {
  const ratio = col.flag_rate != null && referenceFlagRate ? col.flag_rate / referenceFlagRate : null;
  return (
    <div className="panel">
      <div className="panel-title">
        {col.airfield} · {col.date}
        <span style={{ marginLeft: 8 }}>
          <CollectionStatusPill status={col.status} />
        </span>
      </div>
      <div className="key-facts-line">
        <span>
          {col.country}
          {col.state ? ` · ${col.state}` : ''}
        </span>
        <span>{col.weather}</span>
        <span>off-nadir {col.off_nadir != null ? `${col.off_nadir.toFixed(0)}°` : NA}</span>
        <span>GSD {col.gsd != null ? `${col.gsd.toFixed(2)} m` : NA}</span>
        <span>
          <span className="mono">{count(col.n_tiles)}</span> tiles
        </span>
        <span>
          <span className="mono">{count(col.n_detections)}</span> detections
        </span>
        <span>
          <span className={`mono${col.n_flagged > 0 ? ' accent-amber' : ''}`}>{count(col.n_flagged)}</span> flagged (<span className="mono">{pct(col.flag_rate)}</span>
          {referenceFlagRate != null ? (
            <>
              {' '}
              vs <span className="mono">{pct(referenceFlagRate)}</span> on the validation set{ratio != null && ratio >= 1.5 ? `, ${score(ratio, 1)}×` : ''}
            </>
          ) : null}
          )
        </span>
        {col.p_value != null && (
          <span className="cell-dim" title="One-sided binomial test against the validation-set flag rate, indicative only">
            p <span className="mono">{pValue(col.p_value)}</span> (indicative)
          </span>
        )}
        <span>
          <span className="mono">{count(col.n_reviewed)}</span> reviewed
        </span>
      </div>
    </div>
  );
}

// Det-chip drawer in a modal with prev/next over the current grid page and
// the same 1/2/3/4/5/s review keys as the Review (field) page, so an analyst
// can step through a collection's flagged chips without leaving the modal.
function DetChipModal({ items, detId, onSelect, onClose }: { items: DetChip[]; detId: number; onSelect: (id: number) => void; onClose: () => void }) {
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
      else if (e.key === '2') actionsRef.current?.trigger('new_class');
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
        <span className="mono">{idx >= 0 ? `Detection ${idx + 1} of ${items.length} on this page` : 'Detection (not on this page)'}</span>
        <button type="button" className="btn btn-sm" disabled={!next} onClick={() => next && onSelect(next.ann_id)} title="Next chip (→)">
          next <ChevronRight size={13} strokeWidth={2} />
        </button>
      </div>
      <ChipDrawerContent annId={detId} source="det" actionsRef={actionsRef} />
    </Modal>
  );
}

export function CollectionPage() {
  const { route, goto } = useNav();
  const groupId = route.groupId;
  const [sort, setSort] = usePersistedState<SortKey>('vigil.collection.sort', 'novelty');
  const [order, setOrder] = usePersistedState<'asc' | 'desc'>('vigil.collection.order', 'desc');
  const [onlyFlagged, setOnlyFlagged] = usePersistedState<boolean>('vigil.collection.onlyFlagged', true);
  const [reviewedFilter, setReviewedFilter] = usePersistedState<ReviewedFilter>('vigil.collection.reviewed', 'all');
  const [offset, setOffset] = useState(0);
  const [openDetId, setOpenDetId] = useState<number | null>(null);
  // Tile filter chip on the Detections list, set from the Tiles section's
  // "N flagged" link — the API has no server-side tile_id filter for
  // /collections/{id}'s chips (backend/API.md), so this is applied
  // client-side over whatever page of chips is already loaded.
  const [tileFilter, setTileFilter] = useState<number | null>(null);
  // This page instance is reused across collections (routed by groupId, not
  // remounted — see App.tsx), so the filter must be cleared when the analyst
  // navigates to a different collection. Adjusting state during render
  // (rather than in an effect) avoids an extra render pass — same idiom as
  // TileViewerModal's loadedForTileId reset.
  const [tileFilterGroupId, setTileFilterGroupId] = useState(groupId);
  if (groupId !== tileFilterGroupId) {
    setTileFilterGroupId(groupId);
    setTileFilter(null);
  }

  const collections = useCollections(); // for the validation-set flag rate in the header (cached from the Collections page)
  const detail = useCollectionDetail(groupId, {
    sort,
    order,
    limit: PAGE_SIZE,
    offset,
    only_flagged: onlyFlagged,
    reviewed: reviewedFilter,
  });

  useEffect(() => {
    if (detail.data) cacheDetChips(detail.data.chips.items.map((c) => ({ ...c, group_id: c.group_id ?? detail.data?.group_id })));
  }, [detail.data]);

  if (!groupId) {
    return (
      <div className="state-block">
        No collection selected.{' '}
        <button type="button" className="btn btn-sm" onClick={() => goto('collections')}>
          Back to Collections
        </button>
      </div>
    );
  }

  const emptyLabel =
    reviewedFilter === 'yes'
      ? 'No reviewed detections in this collection yet.'
      : reviewedFilter === 'no' && onlyFlagged
        ? 'Every flagged detection in this collection has been reviewed.'
        : onlyFlagged
          ? 'No flagged detections in this collection.'
          : 'No detections in this collection.';

  return (
    <div>
      <div className="page-header">
        <div>
          <button type="button" className="btn btn-sm" onClick={() => goto('collections')} style={{ marginBottom: 8 }}>
            ← Collections
          </button>
        </div>
      </div>

      <QueryBoundary isLoading={detail.isLoading} error={detail.error} data={detail.data} loadingLabel="Loading collection…">
        {(col) => {
          const visibleChips = tileFilter != null ? col.chips.items.filter((c) => c.tile_id === tileFilter) : col.chips.items;
          const gridEmptyLabel = tileFilter != null && visibleChips.length === 0 ? `No detections from tile ${tileFilter} on the current page.` : emptyLabel;
          return (
          <>
            <CollectionHeader col={col} referenceFlagRate={collections.data?.reference_flag_rate} />

            <div className="section-gap" />

            <CollectionTiles groupId={col.group_id} tiles={col.tiles} pageChips={col.chips.items} onFilterTile={(id) => setTileFilter(id)} />

            <div className="section-gap" />

            <div className="panel">
              <div className="chip-controls">
                <div className="panel-title" style={{ margin: 0 }}>
                  Detections
                  <InfoPopover label="About this list" width={320}>
                    <p className="info-popover-text">
                      Each chip is cropped from one of the detector&apos;s own boxes on this collection&apos;s tiles, exactly what a fielded system would produce. Predicted role and
                      confidence come from the delivered role head; novelty is the Fuzzy ART vigilance score against categories learned from the labeled validation set.
                    </p>
                    <p className="info-popover-text">Flagged = fails vigilance. No ground truth exists for these detections, so nothing here is an accuracy.</p>
                  </InfoPopover>
                </div>
                <div className="chip-controls-right">
                  <label className="inline-toggle">
                    <input
                      type="checkbox"
                      checked={onlyFlagged}
                      onChange={(e) => {
                        setOffset(0);
                        setOnlyFlagged(e.target.checked);
                      }}
                    />
                    only flagged
                  </label>
                  <select
                    value={reviewedFilter}
                    aria-label="Reviewed filter"
                    onChange={(e) => {
                      setOffset(0);
                      setReviewedFilter(e.target.value as ReviewedFilter);
                    }}
                  >
                    <option value="all">reviewed: all</option>
                    <option value="yes">reviewed: yes</option>
                    <option value="no">reviewed: no</option>
                  </select>
                  <select
                    value={sort}
                    aria-label="Sort"
                    onChange={(e) => {
                      setOffset(0);
                      setSort(e.target.value as SortKey);
                    }}
                  >
                    <option value="novelty">sort: novelty</option>
                    <option value="conf">sort: detector confidence</option>
                  </select>
                  <select
                    value={order}
                    aria-label="Sort order"
                    onChange={(e) => {
                      setOffset(0);
                      setOrder(e.target.value as 'asc' | 'desc');
                    }}
                  >
                    <option value="desc">desc</option>
                    <option value="asc">asc</option>
                  </select>
                </div>
              </div>

              {tileFilter != null && (
                <div className="tile-filter-row">
                  <span className="filter-chip mono">
                    tile {tileFilter}
                    <button type="button" className="filter-chip-x" onClick={() => setTileFilter(null)} aria-label={`Clear tile ${tileFilter} filter`}>
                      ×
                    </button>
                  </span>
                  <span className="caption" style={{ margin: 0 }}>
                    Filtered to this tile on the current page only. The API has no server-side tile filter for this list yet.
                  </span>
                </div>
              )}

              <ChipGrid chips={visibleChips} onSelect={(id) => setOpenDetId(id)} variant="field" emptyLabel={gridEmptyLabel} />
              {col.chips.total > 0 && (
                <div className="pager mono">
                  <button type="button" className="btn btn-sm" disabled={offset === 0} onClick={() => setOffset((o) => Math.max(0, o - PAGE_SIZE))}>
                    ← prev
                  </button>
                  <span>
                    {Math.min(offset + 1, col.chips.total)}–{Math.min(offset + PAGE_SIZE, col.chips.total)} of {count(col.chips.total)}
                  </span>
                  <button type="button" className="btn btn-sm" disabled={offset + PAGE_SIZE >= col.chips.total} onClick={() => setOffset((o) => o + PAGE_SIZE)}>
                    next →
                  </button>
                </div>
              )}
            </div>

            {openDetId !== null && <DetChipModal items={col.chips.items} detId={openDetId} onSelect={setOpenDetId} onClose={() => setOpenDetId(null)} />}
          </>
          );
        }}
      </QueryBoundary>
    </div>
  );
}
