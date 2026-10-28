import { BatchCard } from '../components/BatchCard';
import { DriftChart } from '../components/DriftChart';
import { FindingsPanel } from '../components/FindingsPanel';
import { InfoPopover } from '../components/InfoPopover';
import { ModelCardStrip } from '../components/ModelCardStrip';
import { NoveltyRateChart } from '../components/NoveltyRateChart';
import { QueryBoundary } from '../components/QueryState';
import { ENVELOPE_STATUS_CAPTION } from '../lib/envelopeStatus';
import { useDrift, useEval, useSummary } from '../hooks/useApi';

export function Overview() {
  const summary = useSummary();
  const drift = useDrift();
  const evalQuery = useEval();

  return (
    <div>
      <div className="page-header">
        <div>
          <h1 className="page-title">Overview</h1>
          <div className="page-sub">Current prototype version, operational stream, and drift/novelty trend.</div>
        </div>
      </div>

      <QueryBoundary isLoading={summary.isLoading} error={summary.error} data={summary.data}>
        {(data) => {
          const control = data.batches.find((b) => b.id === 'ref_holdout') ?? data.batches[0];
          return (
          <>
            <ModelCardStrip model={data.model} currentVersion={data.current_version} versions={data.versions} />

            <div className="section-gap" />
            <div className="panel-title" style={{ marginLeft: 2 }}>
              Operational stream
              <InfoPopover label="About the operational stream">
                Stream order as batches arrive. Click a card to inspect the batch; hover a card for drift score/p-value, role accuracy, and chip counts. {ENVELOPE_STATUS_CAPTION}
              </InfoPopover>
            </div>
            <div className="batch-strip">
              {[...data.batches]
                .sort((a, b) => a.order - b.order)
                .map((b) => (
                  <BatchCard key={b.id} batch={b} control={control} />
                ))}
            </div>

            <div className="section-gap" />
            <FindingsPanel batches={data.batches} evalData={evalQuery.data} />

            <div className="section-gap" />
            <div className="two-col">
              <div className="panel">
                <div className="panel-title">
                  Drift timeline
                  <InfoPopover label="About the drift timeline">
                    {drift.data?.level ? `${drift.data.level}-level drift: ` : ''}
                    {drift.data?.statistic ?? 'mean 5-NN cosine distance to reference'}; dashed line is the null p95 from{' '}
                    {drift.data?.null_method ?? '500 bootstraps of the calibration split'}. Bars colored by indicative p-value (p&lt;0.01 red, p&lt;0.05 amber, else blue).
                  </InfoPopover>
                </div>
                {drift.isLoading && <div className="state-block">Loading…</div>}
                {drift.error && <div className="state-block state-error">Failed to load drift.</div>}
                {drift.data && <DriftChart series={drift.data.series} />}
              </div>
              <div className="panel">
                <div className="panel-title">
                  Novelty rate by batch
                  <InfoPopover label="About novelty rate by batch">
                    Fraction of chips per batch that failed the vigilance test against every known ART category, at the calibrated threshold.
                  </InfoPopover>
                </div>
                <NoveltyRateChart batches={data.batches} />
              </div>
            </div>
          </>
          );
        }}
      </QueryBoundary>
    </div>
  );
}
