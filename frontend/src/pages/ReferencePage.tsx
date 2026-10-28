import { useState, type ReactNode } from 'react';
import type { ReferenceExampleTile } from '../api/types';
import { InfoPopover } from '../components/InfoPopover';
import { QueryBoundary } from '../components/QueryState';
import { TileViewerModal } from '../components/TileViewerModal';
import { useReference, useTileDetail } from '../hooks/useApi';
import { count, pct, score } from '../lib/format';
import { useNav } from '../state/useNav';

// Analyst view's landing page: what the model shipped knowing. The ONLY
// labeled data anywhere in this system (ref_train ∪ ref_calib ∪ ref_holdout —
// see backend/API.md's vocabulary section) — everything else (Collections)
// is unlabeled imagery collected since delivery.

function StatTile({ label, value, sub, info }: { label: string; value: string; sub?: string; info?: ReactNode }) {
  return (
    <div className="reference-stat">
      <div className="reference-stat-label">
        {label}
        {info && <InfoPopover label={`About ${label}`}>{info}</InfoPopover>}
      </div>
      <div className="reference-stat-value">{value}</div>
      {sub && <div className="reference-stat-sub">{sub}</div>}
    </div>
  );
}

function ExampleTilesGrid({ tiles }: { tiles: ReferenceExampleTile[] }) {
  const [openIndex, setOpenIndex] = useState<number | null>(null);
  const openTile = openIndex != null ? tiles[openIndex] : undefined;
  const tileDetail = useTileDetail(openTile?.tile_id);

  if (tiles.length === 0) {
    return <div className="state-block">No labeled tiles yet.</div>;
  }

  return (
    <>
      <div className="tile-thumb-grid">
        {tiles.map((t, idx) => (
          <div key={t.tile_id} className="tile-thumb">
            <button type="button" className="tile-thumb-img-btn" onClick={() => setOpenIndex(idx)} aria-label={`Open tile ${t.tile_id} in the full-screen viewer`}>
              <img src={`${t.tile_url}`} alt="" loading="lazy" width={128} height={128} />
            </button>
            <div className="tile-thumb-caption reference-tile-caption" title={`${t.airfield} · ${count(t.n_aircraft)} aircraft`}>
              <span className="reference-tile-airfield">{t.airfield}</span>
              <span className="mono">{count(t.n_aircraft)}</span>
            </div>
          </div>
        ))}
      </div>

      {openTile && openIndex != null && (
        <TileViewerModal
          tileId={openTile.tile_id}
          chips={tileDetail.data?.chips ?? []}
          highlightAnnId={-1}
          detections={tileDetail.data?.detections}
          legend={
            <span>
              <span className="legend-swatch" style={{ borderColor: '#ffffff' }} /> ground truth (labeled: this is the reference set)
            </span>
          }
          stepLabel={`Tile ${openIndex + 1} of ${tiles.length} · ${openTile.airfield}`}
          onPrev={openIndex > 0 ? () => setOpenIndex(openIndex - 1) : undefined}
          onNext={openIndex < tiles.length - 1 ? () => setOpenIndex(openIndex + 1) : undefined}
          onClose={() => setOpenIndex(null)}
        />
      )}
    </>
  );
}

export function ReferencePage() {
  const { goto } = useNav();
  const reference = useReference();

  return (
    <div>
      <div className="page-header">
        <div>
          <h1 className="page-title">Validation set: what the model knows</h1>
          <div className="page-sub">The only labeled data in the system: the set the detector shipped with. Everything else is unlabeled imagery since delivery.</div>
        </div>
      </div>

      <QueryBoundary isLoading={reference.isLoading} error={reference.error} data={reference.data} loadingLabel="Loading validation set…">
        {(data) => (
          <>
            <div className="reference-stats">
              <StatTile label="images" value={count(data.n_images)} sub={`${count(data.n_tiles)} tiles`} />
              <StatTile label="aircraft chips" value={count(data.n_chips)} sub={`${data.classes.length} known classes`} />
              <StatTile label="fit vs control" value={`${count(data.splits.fit)} / ${count(data.splits.control)}`} sub="images used to fit vs. held out as control" />
              <StatTile label="detector val mAP50" value={score(data.detector.val_map50, 3)} sub={data.detector.trained_on ?? undefined} />
              <StatTile
                label="flag rate"
                value={pct(data.flag_rate)}
                sub="at calibrated vigilance"
                info="Calibrated so ~3% of validation chips fail the vigilance test."
              />
            </div>

            <div className="section-gap" />

            <div className="panel">
              <div className="panel-title">
                Known classes
                <InfoPopover label="About known classes">
                  The detector's role head was fit on these classes only. The set contains no military aircraft (military_aircraft: {data.military_aircraft}). Any analyst-created
                  class comes from feedback recorded since, never from this labeled set.
                </InfoPopover>
              </div>
              <div className="reference-classes">
                {data.classes.map((c) => (
                  <div className="reference-class-card" key={c.label}>
                    <div className="reference-class-title">{c.label}</div>
                    <div className="reference-class-count mono">{count(c.n_chips)} chips</div>
                    <div className="reference-example-row">
                      {c.examples.map((ex) => (
                        <img key={ex.ann_id} className="reference-example-thumb" src={ex.chip_url} alt={c.label} loading="lazy" width={48} height={48} />
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="section-gap" />

            <div className="panel">
              <div className="panel-title">Conditions covered</div>
              <div className="reference-pill-row">
                {Object.entries(data.conditions.weather).map(([label, n]) => (
                  <span className="reference-pill" key={`weather-${label}`}>
                    {label} · {count(n)} images
                  </span>
                ))}
                {Object.entries(data.conditions.countries).map(([label, n]) => (
                  <span className="reference-pill" key={`country-${label}`}>
                    {label} · {count(n)} images
                  </span>
                ))}
                {data.conditions.off_nadir.min != null && data.conditions.off_nadir.max != null && (
                  <span className="reference-pill">
                    off-nadir {data.conditions.off_nadir.min.toFixed(1)}°–{data.conditions.off_nadir.max.toFixed(1)}°
                  </span>
                )}
                {data.conditions.gsd.min != null && data.conditions.gsd.max != null && (
                  <span className="reference-pill">
                    GSD {data.conditions.gsd.min.toFixed(2)}–{data.conditions.gsd.max.toFixed(2)} m
                  </span>
                )}
                {data.conditions.dates.min && data.conditions.dates.max && (
                  <span className="reference-pill">
                    collected {data.conditions.dates.min} – {data.conditions.dates.max}
                  </span>
                )}
              </div>
            </div>

            <div className="section-gap" />

            <div className="panel">
              <div className="panel-title">
                Example tiles
                <InfoPopover label="About example tiles">
                  {data.example_tiles.length} labeled parent-image tiles, most aircraft first. Ground-truth boxes (white) are drawn because this is the labeled set. The analyst
                  view never shows ground truth anywhere else.
                </InfoPopover>
              </div>
              <ExampleTilesGrid tiles={data.example_tiles} />
            </div>

            <div className="section-gap" />
            <div className="caption">Fit set: ref_train and ref_calib. Control: ref_holdout, never used for fitting or calibration. Aircraft count shown per tile.</div>

            <div className="section-gap" />
            <button type="button" className="btn btn-primary" onClick={() => goto('collections')}>
              Go to collections →
            </button>
          </>
        )}
      </QueryBoundary>
    </div>
  );
}
