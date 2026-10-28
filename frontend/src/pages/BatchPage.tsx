import { useState } from 'react';
import { usePersistedState } from '../hooks/usePersistedState';
import { ChipDrawerModal } from '../components/ChipDrawerModal';
import { ChipGrid } from '../components/ChipGrid';
import { Details } from '../components/Details';
import { InfoPopover } from '../components/InfoPopover';
import { NoveltyHistogram } from '../components/NoveltyHistogram';
import { QueryBoundary } from '../components/QueryState';
import { useBatchChips, useBatchDetail, useBatches, useSummary } from '../hooks/useApi';
import { useNav } from '../state/useNav';
import { NA, count, humanizeKey, pct, score } from '../lib/format';
import type { Batch } from '../api/types';

type SortKey = 'novelty' | 'role_conf' | 'knn_dist';
type DetectedFilter = 'all' | 'yes' | 'no';

const PAGE_SIZE = 48;

// Pulls the three headline condition facts (weather / country / off-nadir)
// out of Batch.condition for the always-visible key-facts line — the rest of
// `condition`, plus drift/role-accuracy/detection numbers, live in the Details
// table below. Keys are looked up flexibly since the backend has used both
// singular and plural forms (see backend/API.md's Batch.condition comment).
function pickCondition(condition: Record<string, string | number>, keys: string[]): string | number | undefined {
  for (const k of keys) {
    if (k in condition) return condition[k];
  }
  return undefined;
}

function keyFacts(batch: Batch): string {
  const weather = pickCondition(batch.condition, ['weather']);
  const country = pickCondition(batch.condition, ['country', 'countries']);
  const offNadir = pickCondition(batch.condition, ['off_nadir', 'off_nadir_deg']);
  const parts: string[] = [];
  if (weather != null) parts.push(String(weather));
  if (country != null) parts.push(String(country));
  if (offNadir != null) parts.push(`off-nadir ${offNadir}`);
  parts.push(`${count(batch.n_tiles)} tiles / ${count(batch.n_chips)} chips / ${count(batch.n_parent_images)} images`);
  return parts.join(' · ');
}

export function BatchPage() {
  const { route, goto } = useNav();
  const batchesQuery = useBatches();
  const summary = useSummary();
  const sortedBatches = batchesQuery.data ? [...batchesQuery.data].sort((a, b) => a.order - b.order) : undefined;
  const selectedId = route.batchId ?? sortedBatches?.[0]?.id;
  // Grid controls persist across navigation (sessionStorage) so a prepared view survives leaving the page.
  const [sort, setSort] = usePersistedState<SortKey>('vigil.batch.sort', 'novelty');
  const [order, setOrder] = usePersistedState<'asc' | 'desc'>('vigil.batch.order', 'desc');
  const [onlyNovel, setOnlyNovel] = usePersistedState<boolean>('vigil.batch.onlyNovel', false);
  const [detectedFilter, setDetectedFilter] = usePersistedState<DetectedFilter>('vigil.batch.detected', 'all');
  const [includeEdgeChips, setIncludeEdgeChips] = usePersistedState<boolean>('vigil.batch.edge', false);
  const [offset, setOffset] = useState(0);
  const [openAnnId, setOpenAnnId] = useState<number | null>(null);

  const detail = useBatchDetail(selectedId);
  const chipsPage = useBatchChips(selectedId, {
    sort,
    order,
    limit: PAGE_SIZE,
    offset,
    only_novel: onlyNovel,
    detected: detectedFilter,
    min_partial: includeEdgeChips ? 0 : undefined,
  });

  return (
    <div>
      <div className="page-header">
        <div>
          <h1 className="page-title">Batch</h1>
          <div className="page-sub">Condition facts, novelty distribution, and the chip-level evidence for one operational batch.</div>
        </div>
        {sortedBatches && (
          <select
            value={selectedId ?? ''}
            onChange={(e) => {
              setOffset(0);
              goto('batch', e.target.value);
            }}
          >
            {sortedBatches.map((b) => (
              <option key={b.id} value={b.id}>
                {b.name}
              </option>
            ))}
          </select>
        )}
      </div>

      <QueryBoundary isLoading={detail.isLoading} error={detail.error} data={detail.data} loadingLabel="Loading batch…">
        {(batch) => (
          <>
            <div className="two-col">
              <div className="panel">
                <div className="panel-title">
                  {batch.name}
                  <InfoPopover label="About this batch's condition facts">
                    Condition fields are derived from the RarePlanes tile metadata used to assemble this batch. Drift statistic
                    {batch.drift.level ? ` (${batch.drift.level}-level)` : ''}: {batch.drift.statistic}. Null: {batch.drift.null_method}. p-value is indicative, not inferential.
                  </InfoPopover>
                </div>
                <p style={{ marginTop: 4, marginBottom: 8, color: 'var(--text-dim)', fontSize: 13 }}>{batch.description}</p>
                <div className="key-facts-line">{keyFacts(batch)}</div>
                <Details id="batch-facts" label="Full condition table">
                  <table className="meta-table">
                    <tbody>
                      {Object.entries(batch.condition)
                        .filter(([k]) => k !== 'has_military')
                        .map(([k, v]) => (
                          <tr key={k}>
                            <td>{humanizeKey(k)}</td>
                            <td className={typeof v === 'number' ? 'mono' : ''}>{v}</td>
                          </tr>
                        ))}
                      <tr>
                        <td>Tiles</td>
                        <td className="mono">{count(batch.n_tiles)}</td>
                      </tr>
                      <tr>
                        <td>Chips</td>
                        <td className="mono">{count(batch.n_chips)}</td>
                      </tr>
                      <tr>
                        <td>Parent images</td>
                        <td className="mono">{count(batch.n_parent_images)}</td>
                      </tr>
                      <tr>
                        <td>Drift score / null p95</td>
                        <td className="mono">
                          {score(batch.drift.score)} / {score(batch.drift.null_p95)} (p={score(batch.drift.p_value)} indicative, {batch.drift.ratio_to_calib.toFixed(2)}× calib
                          mean)
                        </td>
                      </tr>
                      <tr>
                        <td>Role accuracy</td>
                        <td className="mono">{pct(batch.role_accuracy)}</td>
                      </tr>
                      <tr>
                        <td>Detector recall / precision</td>
                        <td className="mono">
                          {batch.detection != null
                            ? `${pct(batch.detection.recall)} / ${pct(batch.detection.precision)} (n_gt=${batch.detection.n_gt.toLocaleString()}${
                                batch.detection.n_det != null ? `, n_det=${batch.detection.n_det.toLocaleString()}` : ''
                              })`
                            : `${NA} (detector metrics not yet available)`}
                        </td>
                      </tr>
                      {batch.novelty_rate_military != null && batch.novelty_rate_civil != null && (
                        <tr>
                          <td>Flag rate (military / civil)</td>
                          <td className="mono">
                            {pct(batch.novelty_rate_military)} / {pct(batch.novelty_rate_civil)} (ground truth, evaluation only)
                          </td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </Details>
              </div>

              <div className="panel">
                <div className="panel-title">
                  Novelty distribution vs reference
                  <InfoPopover label="About the novelty distribution">
                    Density of chip novelty scores (0–1) in this batch vs the reference domain. A rightward shift means more chips failing the vigilance test.
                  </InfoPopover>
                </div>
                <NoveltyHistogram batchHist={batch.novelty_hist} refHist={batch.ref_novelty_hist} vigilance={summary.data?.model.vigilance} />
              </div>
            </div>

            <div className="section-gap" />
            <div className="panel">
              <div className="chip-controls">
                <div className="panel-title" style={{ margin: 0 }}>
                  Chips
                  <InfoPopover label="About the chip list">
                    Paginated from /batches/{selectedId}/chips, {PAGE_SIZE} per page. role_pred and confidence come from the nearest-prototype role head; novelty is the ART
                    vigilance score. Aircraft cut by the tile edge are hidden by default; they dominate the novelty ranking for the wrong reason.
                  </InfoPopover>
                </div>
                <div className="chip-controls-right">
                  <label className="inline-toggle">
                    <input
                      type="checkbox"
                      checked={onlyNovel}
                      onChange={(e) => {
                        setOffset(0);
                        setOnlyNovel(e.target.checked);
                      }}
                    />{' '}
                    only novel
                  </label>
                  <label className="inline-toggle" title="Aircraft cut by the tile edge are hidden by default (min_partial=0.75 on the backend)">
                    <input
                      type="checkbox"
                      checked={includeEdgeChips}
                      onChange={(e) => {
                        setOffset(0);
                        setIncludeEdgeChips(e.target.checked);
                      }}
                    />{' '}
                    include edge/partial chips
                  </label>
                  <select
                    value={detectedFilter}
                    onChange={(e) => {
                      setOffset(0);
                      setDetectedFilter(e.target.value as DetectedFilter);
                    }}
                    title="Filters on Chip.detected: a no-op until the backend's detector-chip flags land"
                  >
                    <option value="all">detected: all</option>
                    <option value="yes">detected: yes</option>
                    <option value="no">detected: no (missed)</option>
                  </select>
                  <select
                    value={sort}
                    onChange={(e) => {
                      setOffset(0);
                      setSort(e.target.value as SortKey);
                    }}
                  >
                    <option value="novelty">sort: novelty</option>
                    <option value="role_conf">sort: role confidence</option>
                    <option value="knn_dist">sort: knn distance</option>
                  </select>
                  <select
                    value={order}
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

              <QueryBoundary isLoading={chipsPage.isLoading} error={chipsPage.error} data={chipsPage.data} loadingLabel="Loading chips…">
                {(page) => (
                  <>
                    <ChipGrid chips={page.items} onSelect={setOpenAnnId} />
                    <div className="pager mono">
                      <button type="button" className="btn btn-sm" disabled={offset === 0} onClick={() => setOffset((o) => Math.max(0, o - PAGE_SIZE))}>
                        ← prev
                      </button>
                      <span>
                        {Math.min(offset + 1, page.total)}–{Math.min(offset + PAGE_SIZE, page.total)} of {page.total}
                      </span>
                      <button type="button" className="btn btn-sm" disabled={offset + PAGE_SIZE >= page.total} onClick={() => setOffset((o) => o + PAGE_SIZE)}>
                        next →
                      </button>
                    </div>
                  </>
                )}
              </QueryBoundary>
            </div>
          </>
        )}
      </QueryBoundary>

      {openAnnId !== null && <ChipDrawerModal annId={openAnnId} onClose={() => setOpenAnnId(null)} />}
    </div>
  );
}
