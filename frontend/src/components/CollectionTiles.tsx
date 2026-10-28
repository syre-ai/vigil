import { useState } from 'react';
import { tileImageUrl } from '../api/client';
import type { CollectionTile, DetChip } from '../api/types';
import { useTileDetail } from '../hooks/useApi';
import { count } from '../lib/format';
import { makeFlagColorFn } from '../lib/tileFlagColor';
import { Details } from './Details';
import { TileViewerModal } from './TileViewerModal';

const INITIAL_SHOWN = 24;

function sortTiles(tiles: CollectionTile[]): CollectionTile[] {
  return [...tiles].sort((a, b) => b.n_flagged - a.n_flagged || a.tile_id - b.tile_id);
}

// "Tiles (N)" section for the analyst-view collection page: every parent-tile
// thumbnail in the collection (GET /collections/{group_id}'s `tiles`), a
// full-screen stepper viewer with detector boxes only (never ground truth —
// see backend/API.md's field-mode contract), and a hook back to the
// Detections list below via `onFilterTile`.
export function CollectionTiles({
  groupId,
  tiles,
  pageChips,
  onFilterTile,
}: {
  groupId: string;
  tiles: CollectionTile[];
  pageChips: DetChip[];
  onFilterTile: (tileId: number) => void;
}) {
  const [showAll, setShowAll] = useState(false);
  const [openIndex, setOpenIndex] = useState<number | null>(null);

  const sorted = sortTiles(tiles);
  const shown = showAll ? sorted : sorted.slice(0, INITIAL_SHOWN);
  const openTile = openIndex != null ? sorted[openIndex] : undefined;
  const tileDetail = useTileDetail(openTile?.tile_id);

  return (
    <div className="panel">
      <Details id={`collection-tiles-${groupId}`} label={`Tiles (${count(tiles.length)})`} defaultOpen>
        {tiles.length === 0 ? (
          <div className="state-block">No tiles in this collection.</div>
        ) : (
          <>
            <div className="tile-thumb-grid">
              {shown.map((t, idx) => (
                <div key={t.tile_id} className={`tile-thumb${t.n_flagged > 0 ? ' tile-thumb-flagged' : ''}`}>
                  <button
                    type="button"
                    className="tile-thumb-img-btn"
                    onClick={() => setOpenIndex(idx)}
                    aria-label={`Open tile ${t.tile_id} in the full-screen viewer`}
                  >
                    <img src={tileImageUrl(t.tile_id)} alt="" loading="lazy" width={128} height={128} />
                  </button>
                  <div className="tile-thumb-caption mono">
                    <span>{count(t.n_detections)} det</span>
                    {t.n_flagged > 0 && (
                      <>
                        {' · '}
                        <button
                          type="button"
                          className="tile-thumb-flag-link"
                          onClick={() => onFilterTile(t.tile_id)}
                          title={`Filter the detections list below to tile ${t.tile_id}`}
                        >
                          {count(t.n_flagged)} flagged
                        </button>
                      </>
                    )}
                  </div>
                </div>
              ))}
            </div>
            {!showAll && sorted.length > INITIAL_SHOWN && (
              <button type="button" className="btn btn-sm" style={{ marginTop: 10 }} onClick={() => setShowAll(true)}>
                show all {sorted.length} tiles
              </button>
            )}
          </>
        )}
      </Details>

      {openTile && openIndex != null && (
        <TileViewerModal
          tileId={openTile.tile_id}
          chips={[]}
          highlightAnnId={-1}
          detections={tileDetail.data?.detections}
          detectionStyle="field"
          detectionColor={makeFlagColorFn(openTile.tile_id, pageChips)}
          legend={
            <>
              <span>
                <span className="legend-swatch" style={{ borderColor: '#0891b2' }} /> detector box
              </span>
              <span>
                <span className="legend-swatch" style={{ borderColor: '#d97706' }} /> flagged (matched from the loaded detections page)
              </span>
            </>
          }
          stepLabel={`Tile ${openIndex + 1} of ${sorted.length} · ${openTile.tile_id}`}
          onPrev={openIndex > 0 ? () => setOpenIndex(openIndex - 1) : undefined}
          onNext={openIndex < sorted.length - 1 ? () => setOpenIndex(openIndex + 1) : undefined}
          onClose={() => setOpenIndex(null)}
        />
      )}
    </div>
  );
}
