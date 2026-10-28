import { useState } from 'react';
import { EnvelopeTable } from '../components/EnvelopeTable';
import { InfoPopover } from '../components/InfoPopover';
import { QueryBoundary } from '../components/QueryState';
import { useEnvelope } from '../hooks/useApi';

export function EnvelopePage() {
  const envelope = useEnvelope();
  const [showDeltas, setShowDeltas] = useState(false);
  return (
    <div>
      <div className="page-header">
        <div>
          <h1 className="page-title">Tested envelope</h1>
          <div className="page-sub">Which conditions have enough evidence behind them, and where the model is degraded or untested.</div>
        </div>
      </div>

      <QueryBoundary isLoading={envelope.isLoading} error={envelope.error} data={envelope.data} loadingLabel="Loading envelope…">
        {(data) => {
          const hasLegacyReferenceStatus = data.dimensions.some((d) => d.buckets.some((b) => b.status === 'reference'));
          const hasRefTag = data.dimensions.some((d) => d.buckets.some((b) => b.in_reference_definition));
          const degradedPts = (data.thresholds.degraded_margin * 100).toFixed(0);
          const controlNoveltyRate = data.thresholds.control?.novelty_rate;
          let noveltyRuleText: string;
          if (data.thresholds.novelty_multiplier != null) {
            noveltyRuleText = `novelty rate > ${data.thresholds.novelty_multiplier}× control${controlNoveltyRate != null ? ` (${(controlNoveltyRate * 100).toFixed(1)}%)` : ''}`;
          } else {
            const noveltyMarginPct = data.thresholds.novelty_margin != null ? (data.thresholds.novelty_margin * 100).toFixed(0) : '25';
            noveltyRuleText = `novelty rate > ${noveltyMarginPct}%`;
          }
          return (
            <>
              <div className="chip-controls" style={{ marginBottom: 4 }}>
                <div className="section-heading" style={{ margin: 0 }}>
                  Legend
                  <InfoPopover label="Status legend and methodology" width={360}>
                    <div className="legend-row" style={{ marginTop: 0 }}>
                      {hasLegacyReferenceStatus && (
                        <>
                          <span className="pill pill-reference">reference</span> the bucket that defines the reference domain itself
                        </>
                      )}
                      {hasRefTag && (
                        <>
                          <span className="tag-ref">ref</span> part of the reference-domain definition
                        </>
                      )}
                      <span className="pill pill-tested">tested</span> n ≥ {data.thresholds.min_n} chips, role accuracy / detection recall within {degradedPts} points of control
                      <span className="pill pill-untested">untested</span> fewer than {data.thresholds.min_n} chips scored
                      <span className="pill pill-degraded">degraded</span> role accuracy or detection recall more than {degradedPts} points below control, or {noveltyRuleText}
                    </div>
                    <p className="info-popover-text">
                      Every bucket is computed from real per-chip metadata (weather, off-nadir, country, GSD, wingspan) across the reference set and all operational batches. Δ
                      columns are this bucket&apos;s metric minus the control batch&apos;s (ref_holdout), where available.
                    </p>
                  </InfoPopover>
                </div>
                <label className="inline-toggle">
                  <input type="checkbox" checked={showDeltas} onChange={(e) => setShowDeltas(e.target.checked)} />
                  Show deltas &amp; images
                </label>
              </div>
              {data.dimensions.map((dim) => (
                <EnvelopeTable key={dim.name} dimension={dim} showDeltas={showDeltas} />
              ))}
            </>
          );
        }}
      </QueryBoundary>
    </div>
  );
}
