import { useEffect, useState, type Ref } from 'react';
import { Maximize2 } from 'lucide-react';
import type { ChipDetail, DetChipDetail, Review, ReviewSource } from '../api/types';
import { useChipDetail, useDetDetail, useTileDetail } from '../hooks/useApi';
import { cacheChip } from '../state/chipCache';
import { cacheDetChip } from '../state/detChipCache';
import { Details } from './Details';
import { DetectedBadge } from './DetectedBadge';
import { InfoPopover } from './InfoPopover';
import { NeighborStrip } from './NeighborStrip';
import { NovelBadge } from './NovelBadge';
import { ErrorBlock, QueryBoundary } from './QueryState';
import { ReviewActions, type ReviewActionsHandle } from './ReviewActions';
import { TileCanvas } from './TileCanvas';
import { TileViewerModal } from './TileViewerModal';
import { NA, edgeLabel, isEdgeChip, reviewSummary, score, shortRole } from '../lib/format';

// Normalized view model so the JSX below never has to branch on the
// ChipDetail/DetChipDetail union field-by-field. GT-only fields
// (roleTrue/detected/artMatch/truncated/partial/wingspanM) are left
// undefined/null for det chips and never rendered — field mode never
// carries ground truth (backend/API.md's field-mode contract).
type DrawerVM = {
  id: number;
  imageUrl: string;
  rolePred: string | null; // det chips can be unscored (no score row yet) — see api/types.ts DetChip
  roleConf: number | null;
  novelty: number | null;
  isNovel: boolean | null;
  knnDist: number | null;
  conf?: number; // det-only: the detector's own confidence for this box
  tileId: number;
  bbox: [number, number, number, number];
  meta: { airfield: string; country: string; weather: string; off_nadir: number | null; gsd: number | null; wingspanM: number | null };
  neighbors: { ann_id: number; chip_url: string; role_true: string | null; dist?: number }[];
  review: Review | null;
  roleTrue: string | null;
  detected: boolean | null | undefined;
  artMatch: number | undefined;
  truncated: boolean | null | undefined;
  partial: number | null | undefined;
};

function buildVM(chip: ChipDetail | DetChipDetail, isDet: boolean): DrawerVM {
  if (isDet) {
    const d = chip as DetChipDetail;
    return {
      id: d.det_id,
      imageUrl: d.chip_url,
      rolePred: d.role_pred,
      roleConf: d.role_conf,
      novelty: d.novelty,
      isNovel: d.is_novel,
      knnDist: d.knn_dist,
      conf: d.conf,
      tileId: d.tile_id,
      bbox: d.bbox,
      meta: { airfield: d.meta.airfield, country: d.meta.country, weather: d.meta.weather, off_nadir: d.meta.off_nadir, gsd: d.meta.gsd, wingspanM: null },
      neighbors: d.neighbors,
      review: d.review,
      roleTrue: null,
      detected: undefined,
      artMatch: undefined,
      truncated: undefined,
      partial: undefined,
    };
  }
  const g = chip as ChipDetail;
  return {
    id: g.ann_id,
    imageUrl: g.chip_url,
    rolePred: g.role_pred,
    roleConf: g.role_conf,
    novelty: g.novelty,
    isNovel: g.is_novel,
    knnDist: g.knn_dist,
    tileId: g.tile_id,
    bbox: g.bbox,
    meta: { airfield: g.meta.airfield, country: g.meta.country, weather: g.meta.weather, off_nadir: g.meta.off_nadir, gsd: g.meta.gsd, wingspanM: g.meta.wingspan_m },
    neighbors: g.neighbors,
    review: g.review,
    roleTrue: g.role_true,
    detected: g.detected,
    artMatch: g.art_match,
    truncated: g.truncated,
    partial: g.partial,
  };
}

export function ChipDrawerContent({
  annId,
  source = 'gt',
  onReviewed,
  actionsRef,
}: {
  annId: number;
  source?: ReviewSource;
  onReviewed?: (review: Review) => void;
  actionsRef?: Ref<ReviewActionsHandle>;
}) {
  const isDet = source === 'det';
  const gtQuery = useChipDetail(isDet ? undefined : annId);
  const detQuery = useDetDetail(isDet ? annId : undefined);
  const chipQuery = isDet ? detQuery : gtQuery;
  const tileId = chipQuery.data?.tile_id;
  const tileQuery = useTileDetail(tileId);
  const [tileViewerOpen, setTileViewerOpen] = useState(false);

  // Snapshot every chip fetched here into the shared cache so it can still
  // be shown (thumbnail, role) after it's reviewed and drops out of the
  // review queue — see state/chipCache.ts / state/detChipCache.ts. Kept in
  // two separate caches (keyed by ann_id vs det_id) so the two id spaces can
  // never collide.
  useEffect(() => {
    if (!isDet && gtQuery.data) cacheChip(gtQuery.data);
  }, [isDet, gtQuery.data]);
  useEffect(() => {
    if (isDet && detQuery.data) cacheDetChip(detQuery.data);
  }, [isDet, detQuery.data]);

  return (
    <QueryBoundary isLoading={chipQuery.isLoading} error={chipQuery.error} data={chipQuery.data}>
      {(chipData) => {
        const chip = chipData as ChipDetail | DetChipDetail;
        const vm = buildVM(chip, isDet);
        const showEdge = !isDet && isEdgeChip({ truncated: vm.truncated, partial: vm.partial });
        const tileCanvasChips = isDet ? [] : (tileQuery.data?.chips ?? []);
        const tileHighlightAnnId = isDet ? -1 : vm.id;
        const tileHighlightBox = isDet ? { bbox: vm.bbox, color: '#d97706' } : undefined;
        const tileDetectionStyle: 'benchmark' | 'field' = isDet ? 'field' : 'benchmark';
        const viewerLegend = isDet ? (
          <>
            <span>
              <span className="legend-swatch" style={{ borderColor: '#d97706' }} /> this chip
            </span>
            <span>
              <span className="legend-swatch" style={{ borderColor: '#0891b2' }} /> detector box
            </span>
          </>
        ) : undefined;
        return (
          <div className="chip-drawer">
            <div className="chip-drawer-left">
              <div className="chip-drawer-big-img-wrap">
                <img src={vm.imageUrl} alt={vm.rolePred ?? 'unscored chip'} width={224} height={224} loading="lazy" />
                {(vm.isNovel || vm.detected === false) && (
                  <div className="chip-card-badge">
                    {vm.isNovel && <NovelBadge label={isDet ? 'FLAGGED' : 'NOVEL'} />}
                    {!isDet && vm.detected === false && <DetectedBadge />}
                  </div>
                )}
              </div>
              {showEdge && (
                <div className="chip-drawer-tags">
                  <span className="chip-card-tag chip-card-tag-edge">{edgeLabel({ partial: vm.partial })} · cut by tile edge</span>
                </div>
              )}

              <div className="chip-drawer-role-line">
                <span className="chip-drawer-role-text">{vm.rolePred ?? 'not yet scored'}</span>
                <span className="mono chip-drawer-conf">conf {score(vm.roleConf, 2)}</span>
              </div>
              <div className="chip-drawer-novelty-line mono">
                novelty {score(vm.novelty)}
                <InfoPopover label="About novelty and the baseline distance">
                  Novelty = 1 − best Fuzzy ART category match, evaluated against every known category. 5-NN distance (baseline) is the mean 5-NN cosine distance to the reference
                  set, shown for comparison.
                </InfoPopover>
              </div>
              <Details id="chip-drawer-secondary-stats" label="More chip-level numbers">
                {!isDet && (
                  <div className="chip-drawer-stat-row">
                    <span className="stat-k">ART match</span>
                    <span className="mono">{score(vm.artMatch)}</span>
                  </div>
                )}
                <div className="chip-drawer-stat-row">
                  <span className="stat-k">5-NN distance (baseline)</span>
                  <span className="mono">{score(vm.knnDist)}</span>
                </div>
                <div className="chip-drawer-stat-row">
                  <span className="stat-k">{isDet ? 'Detection #' : 'Annotation #'}</span>
                  <span className="mono">{vm.id}</span>
                </div>
                {isDet && (
                  <div className="chip-drawer-stat-row">
                    <span className="stat-k">Detector confidence</span>
                    <span className="mono">{score(vm.conf, 2)}</span>
                  </div>
                )}
                {!isDet && vm.detected != null && (
                  <div className="chip-drawer-stat-row">
                    <span className="stat-k">Detected</span>
                    <span className={`mono${vm.detected ? '' : ' accent-red'}`}>{vm.detected ? 'yes' : 'no · missed by detector'}</span>
                  </div>
                )}
              </Details>

              <div className="section-gap-sm" />
              <div className="panel-title">
                Parent tile
                <InfoPopover label="About the parent tile view" width={300}>
                  {isDet ? (
                    <div className="tile-legend-popover">
                      <div>
                        <span className="legend-swatch" style={{ borderColor: '#d97706' }} /> this chip
                      </div>
                      <div>
                        <span className="legend-swatch" style={{ borderColor: '#0891b2' }} /> detector box
                      </div>
                    </div>
                  ) : (
                    <div className="tile-legend-popover">
                      <div>
                        <span className="legend-swatch" style={{ borderColor: '#9aa3af' }} /> ground truth (oracle localization)
                      </div>
                      <div>
                        <span className="legend-swatch" style={{ borderColor: '#d97706' }} /> this chip
                      </div>
                      <div>
                        <span className="legend-swatch" style={{ borderColor: '#0891b2' }} /> detector box
                      </div>
                      {tileQuery.data?.detections !== undefined ? (
                        <div>
                          <span className="legend-swatch" style={{ borderColor: '#0891b2', borderTopStyle: 'dashed' }} /> detector box, no GT match
                        </div>
                      ) : (
                        <div>detector boxes not yet available for this tile</div>
                      )}
                    </div>
                  )}
                  <p className="info-popover-text">
                    Tile {vm.tileId}, bbox [{vm.bbox.map((v) => v.toFixed(1)).join(', ')}] px in the 512×512 source tile.{' '}
                    {isDet ? 'This chip is highlighted; other detector boxes on this tile are shown for context.' : "Sibling ground-truth boxes in this tile shown black; the reviewed chip is highlighted."}
                  </p>
                </InfoPopover>
              </div>
              {tileQuery.data ? (
                <>
                  <div
                    className="tile-canvas-wrap"
                    role="button"
                    tabIndex={0}
                    aria-label="Open full-screen tile viewer"
                    onClick={() => setTileViewerOpen(true)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' || e.key === ' ') {
                        e.preventDefault();
                        setTileViewerOpen(true);
                      }
                    }}
                  >
                    <TileCanvas
                      tileId={vm.tileId}
                      chips={tileCanvasChips}
                      highlightAnnId={tileHighlightAnnId}
                      detections={tileQuery.data.detections}
                      highlightBox={tileHighlightBox}
                      detectionStyle={tileDetectionStyle}
                    />
                    <span className="tile-canvas-open-icon">
                      <Maximize2 size={13} strokeWidth={2} />
                    </span>
                  </div>
                  {tileViewerOpen && (
                    <TileViewerModal
                      tileId={vm.tileId}
                      chips={tileCanvasChips}
                      highlightAnnId={tileHighlightAnnId}
                      detections={tileQuery.data.detections}
                      highlightBox={tileHighlightBox}
                      detectionStyle={tileDetectionStyle}
                      legend={viewerLegend}
                      onClose={() => setTileViewerOpen(false)}
                    />
                  )}
                </>
              ) : tileQuery.isError ? (
                <div>
                  <ErrorBlock error={tileQuery.error} />
                  <button type="button" className="btn btn-sm" style={{ marginTop: 6 }} onClick={() => tileQuery.refetch()}>
                    Retry
                  </button>
                </div>
              ) : (
                <Loading />
              )}
            </div>

            <div className="chip-drawer-right">
              <div className="panel-title">Metadata</div>
              <table className="meta-table">
                <tbody>
                  <tr>
                    <td>Airfield</td>
                    <td>{vm.meta.airfield}</td>
                  </tr>
                  <tr>
                    <td>Country</td>
                    <td>{vm.meta.country}</td>
                  </tr>
                  <tr>
                    <td>Weather</td>
                    <td>{vm.meta.weather}</td>
                  </tr>
                  <tr>
                    <td>Off-nadir angle</td>
                    <td className="mono">{vm.meta.off_nadir != null ? `${vm.meta.off_nadir.toFixed(1)}°` : NA}</td>
                  </tr>
                  <tr>
                    <td>GSD</td>
                    <td className="mono">{vm.meta.gsd != null ? `${vm.meta.gsd.toFixed(3)} m` : NA}</td>
                  </tr>
                  {!isDet && (
                    <tr>
                      <td>Wingspan</td>
                      <td className="mono">{vm.meta.wingspanM != null ? `${vm.meta.wingspanM.toFixed(1)} m` : NA}</td>
                    </tr>
                  )}
                </tbody>
              </table>

              {!isDet && (
                <Details id="chip-drawer-ground-truth" label="ground truth (evaluation only)">
                  True role: <span className="mono">{vm.roleTrue ?? 'unknown'}</span>
                </Details>
              )}

              <div className="section-gap-sm" />
              <div className="panel-title">{isDet ? 'Nearest labeled validation-set chips' : '5 nearest reference chips'}</div>
              <NeighborStrip neighbors={vm.neighbors} />

              <div className="section-gap-sm" />
              <div className="panel-title">Review</div>
              {vm.review && (
                <div className={`review-status-banner${vm.review.version_applied == null ? ' review-status-pending' : ' review-status-applied'}`}>
                  Reviewed:{' '}
                  {vm.review.action === 'correct'
                    ? `corrected to ${shortRole(vm.review.label)}. Recorded as disagreement; does not alter the model`
                    : vm.review.action === 'bad_box'
                      ? 'bad box. Localization error; does not alter the model'
                      : reviewSummary(vm.review, vm.rolePred)}
                  {vm.review.version_applied == null ? ' (pending)' : ''}
                </div>
              )}
              <ReviewActions annId={vm.id} rolePred={vm.rolePred} currentReview={vm.review} onDone={onReviewed} ref={actionsRef} source={source} />
            </div>
          </div>
        );
      }}
    </QueryBoundary>
  );
}

function Loading() {
  return (
    <div className="skeleton-block">
      <div className="skeleton-line" style={{ width: '100%', height: 220 }} />
    </div>
  );
}
