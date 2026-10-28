import type { HealthIndicator, HealthResponse, HealthStatus } from '../api/types';
import { useHealth } from '../hooks/useApi';
import { Details } from './Details';
import { InfoPopover } from './InfoPopover';
import { RECOMMENDATION_LABEL } from '../lib/format';

// "Model health · retrain readiness" — an INDICATIVE keep / patch / schedule-retrain headline
// with four indicator tiles (GET /api/health). Every threshold and rule shown here comes verbatim
// from the API (`rule`/`rules`); nothing is restated or computed client-side. Wording rules
// (backend/API.md): say "indicative"; never "deployed", "accredited", "significant".
//
// `collapsed` (Changes page): headline + reasons only, with the tiles behind a Details toggle.
// Default (Collections page, the analyst home): tiles always visible, evidence behind Details.

const STATUS_LABEL: Record<HealthStatus, string> = {
  nominal: 'nominal',
  watch: 'watch',
  retrain: 'retrain',
};

function StatusPill({ status }: { status: HealthStatus }) {
  return <span className={`pill pill-${status}`}>{STATUS_LABEL[status]}</span>;
}

export function IndicatorTile({ ind }: { ind: HealthIndicator }) {
  return (
    <div className="health-tile" title={ind.evidence}>
      <div className="health-tile-label">
        {ind.label}
        <InfoPopover label={`Rule for ${ind.label}`} width={360}>
          <p className="info-popover-text" style={{ marginTop: 0 }}>
            <strong>Rule</strong> (from the API, verbatim): {ind.rule}
          </p>
          <p className="info-popover-text">
            <strong>Evidence:</strong> {ind.evidence}
          </p>
        </InfoPopover>
      </div>
      <div className="health-tile-value">
        <span className="mono">{ind.display}</span>
        <StatusPill status={ind.status} />
      </div>
    </div>
  );
}

function Tiles({ h }: { h: HealthResponse }) {
  return (
    <div className="health-tiles">
      {h.indicators.map((ind) => (
        <IndicatorTile key={ind.key} ind={ind} />
      ))}
    </div>
  );
}

function Evidence({ h }: { h: HealthResponse }) {
  return (
    <ul className="health-evidence">
      {h.indicators.map((ind) => (
        <li key={ind.key}>
          <strong>{ind.label}:</strong> {ind.evidence}
        </li>
      ))}
    </ul>
  );
}

export function ModelHealthPanel({ id, collapsed = false }: { id: string; collapsed?: boolean }) {
  const health = useHealth();
  // An older backend without /api/health: show nothing rather than an error block on the analyst home.
  if (health.error) return null;
  if (health.isLoading || !health.data) {
    return (
      <div className="panel health-panel">
        <div className="caption">Loading model health…</div>
      </div>
    );
  }
  const h = health.data;
  const reasons = h.reasons.length > 0 ? h.reasons.join(' ') : 'Every indicator is nominal.';

  return (
    <div className="panel health-panel">
      <div className="health-headline">
        <span className="panel-title">
          Model health · retrain readiness
          <InfoPopover label="About model health" width={440}>
            <p className="info-popover-text" style={{ marginTop: 0 }}>
              <strong>Indicative:</strong> {h.note}. An input to a retraining decision, not the decision. Rules (from the API, verbatim):
            </p>
            <ul className="health-evidence" style={{ marginTop: 6 }}>
              {h.rules.map((r) => (
                <li key={r}>{r}</li>
              ))}
            </ul>
          </InfoPopover>
        </span>
        <span className={`pill pill-${h.recommendation}`} title={`${RECOMMENDATION_LABEL[h.recommendation]} · indicative`}>
          {RECOMMENDATION_LABEL[h.recommendation]}
        </span>
        <span className="health-reasons">{reasons}</span>
        <span className="caption" title={h.note}>
          indicative{h.version ? ` · ${h.version}` : ''}
        </span>
      </div>

      {collapsed ? (
        <Details id={`health-${id}-indicators`} label="Indicators" closeLabel="Hide indicators">
          <Tiles h={h} />
          <Details id={`health-${id}-evidence`} label="Evidence" closeLabel="Hide evidence">
            <Evidence h={h} />
          </Details>
        </Details>
      ) : (
        <>
          <Tiles h={h} />
          <Details id={`health-${id}-evidence`} label="Details" closeLabel="Hide details">
            <Evidence h={h} />
          </Details>
        </>
      )}
    </div>
  );
}
