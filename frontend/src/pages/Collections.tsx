import { useMemo, useState } from 'react';
import { ArrowDown, ArrowUp } from 'lucide-react';
import type { CollectionGroup, CollectionsResponse } from '../api/types';
import { CollectionStatusPill } from '../components/CollectionStatusPill';
import { InfoPopover } from '../components/InfoPopover';
import { QueryBoundary } from '../components/QueryState';
import { useCollections } from '../hooks/useApi';
import { usePersistedState } from '../hooks/usePersistedState';
import { count, pct, pValue } from '../lib/format';
import { useNav } from '../state/useNav';

type SortKey = 'date' | 'flagged' | 'flag_rate' | 'status' | 'detections';
type SortDir = 'asc' | 'desc';

// Status is an ordinal for sorting: what needs attention first.
const STATUS_RANK: Record<CollectionGroup['status'], number> = { needs_review: 0, insufficient_data: 1, nominal: 2 };

// Default direction when a column is first clicked — dates newest-first,
// counts largest-first, status most-urgent-first.
const DEFAULT_DIR: Record<SortKey, SortDir> = { date: 'desc', flagged: 'desc', flag_rate: 'desc', status: 'asc', detections: 'desc' };

function compareBy(key: SortKey, a: CollectionGroup, b: CollectionGroup): number {
  switch (key) {
    case 'date':
      return a.date < b.date ? -1 : a.date > b.date ? 1 : 0;
    case 'flagged':
      return a.n_flagged - b.n_flagged || (a.flag_rate ?? -1) - (b.flag_rate ?? -1);
    case 'flag_rate':
      return (a.flag_rate ?? -1) - (b.flag_rate ?? -1) || a.n_flagged - b.n_flagged;
    case 'detections':
      return a.n_detections - b.n_detections;
    case 'status':
      return STATUS_RANK[a.status] - STATUS_RANK[b.status] || b.n_flagged - a.n_flagged || (b.flag_rate ?? -1) - (a.flag_rate ?? -1);
    default:
      return 0;
  }
}

function conditions(g: CollectionGroup): string {
  return g.weather;
}

function conditionsTitle(g: CollectionGroup): string {
  const parts = [g.weather];
  if (g.off_nadir != null) parts.push(`off-nadir ${g.off_nadir.toFixed(1)}°`);
  if (g.gsd != null) parts.push(`${g.gsd.toFixed(2)} m GSD`);
  parts.push(`${count(g.n_parents)} parent image${g.n_parents === 1 ? '' : 's'} / ${count(g.n_tiles)} tiles`);
  return parts.join(' · ');
}

function SortHeader({
  label,
  sortKey,
  active,
  dir,
  onSort,
  title,
}: {
  label: string;
  sortKey: SortKey;
  active: boolean;
  dir: SortDir;
  onSort: (key: SortKey) => void;
  title?: string;
}) {
  return (
    <th aria-sort={active ? (dir === 'asc' ? 'ascending' : 'descending') : 'none'}>
      <button type="button" className={`th-sort${active ? ' active' : ''}`} onClick={() => onSort(sortKey)} title={title ?? `Sort by ${label}`}>
        {label}
        {active ? dir === 'asc' ? <ArrowUp size={11} strokeWidth={2.25} /> : <ArrowDown size={11} strokeWidth={2.25} /> : null}
      </button>
    </th>
  );
}

function HeaderLine({ data }: { data: CollectionsResponse }) {
  const needsReview = data.groups.filter((g) => g.status === 'needs_review').length;
  return (
    <div className="key-facts-line" style={{ marginTop: -6, marginBottom: 14 }}>
      <span>
        <span className="mono">{count(data.groups.length)}</span> new collections since delivery
      </span>
      <span>
        <span className="mono">{count(needsReview)}</span> need review
      </span>
      <span>
        validation-set flag rate <span className="mono">{pct(data.reference_flag_rate)}</span>
        <InfoPopover label="About the status rule" width={360}>
          <p className="info-popover-text">
            <strong>Status rule</strong> (from the API, verbatim): {data.status_rule}
          </p>
          <p className="info-popover-text">
            Collections are {data.grouping}. {data.min_denominator != null ? `Minimum denominator for a status other than insufficient data: ${count(data.min_denominator)} detections.` : ''}
          </p>
          <p className="info-popover-text">
            Validation-set flag rate = the fraction of detector chips on the labeled validation set ({count(data.reference.n_parents)} parent images / {count(data.reference.n_tiles)} tiles) that the
            novelty layer flags. {data.reference.description}.
          </p>
          <p className="info-popover-text">Flagged = fails the Fuzzy ART vigilance test against categories learned from the validation set. No ground truth is used on this page.</p>
        </InfoPopover>
      </span>
    </div>
  );
}

export function Collections() {
  const { goto } = useNav();
  const collections = useCollections();
  const [needsReviewOnly, setNeedsReviewOnly] = usePersistedState<boolean>('vigil.collections.needsReviewOnly', false);
  const [sortKey, setSortKey] = usePersistedState<SortKey>('vigil.collections.sortKey', 'date');
  const [sortDir, setSortDir] = usePersistedState<SortDir>('vigil.collections.sortDir', 'desc');
  const [filter, setFilter] = useState('');

  function onSort(key: SortKey) {
    if (key === sortKey) setSortDir((d) => (d === 'asc' ? 'desc' : 'asc'));
    else {
      setSortKey(key);
      setSortDir(DEFAULT_DIR[key]);
    }
  }

  const groups = collections.data?.groups;
  const rows = useMemo(() => {
    if (!groups) return [];
    const q = filter.trim().toLowerCase();
    let out = groups.filter((g) => !needsReviewOnly || g.status === 'needs_review');
    if (q) out = out.filter((g) => `${g.airfield} ${g.country} ${g.state ?? ''} ${g.weather} ${g.date}`.toLowerCase().includes(q));
    const sign = sortDir === 'asc' ? 1 : -1;
    out.sort((a, b) => sign * compareBy(sortKey, a, b) || (a.date < b.date ? 1 : a.date > b.date ? -1 : 0));
    return out;
  }, [groups, needsReviewOnly, filter, sortKey, sortDir]);

  const showPValue = groups?.some((g) => g.p_value != null) ?? false;

  return (
    <div>
      <div className="page-header">
        <div>
          <h1 className="page-title">Collections</h1>
          <div className="page-sub">New, unlabeled imagery since the detector was delivered, grouped by airfield and collection date.</div>
        </div>
      </div>

      <QueryBoundary isLoading={collections.isLoading} error={collections.error} data={collections.data} loadingLabel="Loading collections…">
        {(data) => (
          <>
            <HeaderLine data={data} />

            <div className="panel">
              <div className="chip-controls">
                <div className="panel-title" style={{ margin: 0 }}>
                  {count(rows.length)} of {count(data.groups.length)} collections
                </div>
                <div className="chip-controls-right">
                  <input
                    type="text"
                    className="filter-input"
                    placeholder="filter airfield / country…"
                    value={filter}
                    onChange={(e) => setFilter(e.target.value)}
                    aria-label="Filter collections by airfield or country"
                  />
                  <label className="inline-toggle">
                    <input type="checkbox" checked={needsReviewOnly} onChange={(e) => setNeedsReviewOnly(e.target.checked)} />
                    needs review only
                  </label>
                </div>
              </div>

              {data.groups.length === 0 ? (
                <div className="state-block">No new collections yet. Detector chips haven&apos;t been extracted for any imagery outside the validation set.</div>
              ) : rows.length === 0 ? (
                <div className="state-block">No collections match the current filter.</div>
              ) : (
                <div style={{ overflowX: 'auto' }}>
                  <table className="data-table collections-table">
                    <thead>
                      <tr>
                        <SortHeader label="date" sortKey="date" active={sortKey === 'date'} dir={sortDir} onSort={onSort} />
                        <th className="th-left">airfield</th>
                        <th className="th-left">country</th>
                        <th className="th-left" title="Weather; hover a row for off-nadir angle, GSD and tile count">conditions</th>
                        <SortHeader label="detections" sortKey="detections" active={sortKey === 'detections'} dir={sortDir} onSort={onSort} title="Detector boxes at conf ≥ 0.25" />
                        <SortHeader label="flagged" sortKey="flagged" active={sortKey === 'flagged'} dir={sortDir} onSort={onSort} title="Detections failing the vigilance test" />
                        <SortHeader label="flag rate" sortKey="flag_rate" active={sortKey === 'flag_rate'} dir={sortDir} onSort={onSort} />
                        {showPValue && (
                          <th className="muted-col" title="One-sided binomial test of this collection's flagged count against the validation-set flag rate, indicative only, not a decision rule">
                            p
                          </th>
                        )}
                        <th>reviewed</th>
                        <th className="th-left" aria-sort={sortKey === 'status' ? (sortDir === 'asc' ? 'ascending' : 'descending') : 'none'}>
                          <button type="button" className={`th-sort${sortKey === 'status' ? ' active' : ''}`} onClick={() => onSort('status')} title="Sort: needs review first">
                            status
                            {sortKey === 'status' ? sortDir === 'asc' ? <ArrowUp size={11} strokeWidth={2.25} /> : <ArrowDown size={11} strokeWidth={2.25} /> : null}
                          </button>
                        </th>
                      </tr>
                    </thead>
                    <tbody>
                      {rows.map((g) => (
                        <tr
                          key={g.group_id}
                          className="row-link"
                          tabIndex={0}
                          onClick={() => goto('collection', g.group_id)}
                          onKeyDown={(e) => {
                            if (e.key === 'Enter' || e.key === ' ') {
                              e.preventDefault();
                              goto('collection', g.group_id);
                            }
                          }}
                        >
                          <td className="mono">{g.date}</td>
                          <td className="cell-airfield td-left" title={g.state ? `${g.airfield} · ${g.state}` : g.airfield}>
                            {g.airfield}
                          </td>
                          <td className="td-left">{g.country}</td>
                          <td className="cell-dim td-left" title={conditionsTitle(g)}>
                            {conditions(g)}
                          </td>
                          <td className="mono" title={`${count(g.n_tiles)} tiles`}>
                            {count(g.n_detections)}
                          </td>
                          <td
                            className={`mono${g.n_flagged > 0 ? ' accent-amber' : ''}`}
                            title={g.excess_flagged != null ? `${g.excess_flagged >= 0 ? '+' : ''}${g.excess_flagged.toFixed(1)} over the validation-set expectation` : undefined}
                          >
                            {count(g.n_flagged)}
                          </td>
                          <td className="mono">{pct(g.flag_rate)}</td>
                          {showPValue && <td className="mono muted-col">{pValue(g.p_value)}</td>}
                          <td className="mono cell-dim">{count(g.n_reviewed)}</td>
                          <td className="td-left">
                            <CollectionStatusPill status={g.status} />
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          </>
        )}
      </QueryBoundary>
    </div>
  );
}
