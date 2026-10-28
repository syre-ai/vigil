import type { Batch } from '../api/types';
import { computeEnvelopeStatus, ENVELOPE_STATUS_LABEL, ENVELOPE_STATUS_RULE } from '../lib/envelopeStatus';
import { pct, score } from '../lib/format';
import { useNav } from '../state/useNav';

// Overview cards show only the headline pair (novelty rate, detector recall)
// plus name and status — drift score/p-value, role accuracy, chip counts and
// the military/civil flag-rate split move into the card's hover tooltip; the
// full picture stays one click away on the Batch page.
export function BatchCard({ batch, control }: { batch: Batch; control: Batch }) {
  const { goto } = useNav();
  const status = computeEnvelopeStatus(batch, control);
  const cardClass = status === 'nominal' ? 'ok' : status === 'watch' ? 'watch' : 'alert';
  const flagRateDetail =
    batch.novelty_rate_military != null && batch.novelty_rate_civil != null
      ? ` · flag rate military ${pct(batch.novelty_rate_military, 0)} / civil ${pct(batch.novelty_rate_civil, 0)} (ground truth, evaluation only)`
      : '';
  const hoverDetail =
    `${batch.n_chips.toLocaleString()} chips · role acc. ${pct(batch.role_accuracy)} · ` +
    `drift score ${score(batch.drift.score)} (p indicative ${score(batch.drift.p_value)}) vs null p95 ${score(batch.drift.null_p95)}` +
    flagRateDetail;
  return (
    <button type="button" className={`batch-card batch-card-${cardClass}`} onClick={() => goto('batch', batch.id)} title={hoverDetail}>
      <div className="batch-card-head">
        <span className="batch-card-order mono">{String(batch.order + 1).padStart(2, '0')}</span>
        <span className="batch-card-name">{batch.name}</span>
      </div>
      <div className="batch-card-status-row">
        <span className={`pill pill-${status}`} title={ENVELOPE_STATUS_RULE}>
          {ENVELOPE_STATUS_LABEL[status]}
        </span>
      </div>

      <div className="batch-card-stats">
        <div>
          <div className="batch-card-stat-label">Novelty</div>
          <div className="batch-card-stat-value mono">{pct(batch.novelty_rate)}</div>
        </div>
        <div>
          <div className="batch-card-stat-label">Det. recall</div>
          <div className="batch-card-stat-value mono">{pct(batch.detection?.recall)}</div>
        </div>
      </div>
    </button>
  );
}
