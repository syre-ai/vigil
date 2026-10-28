import type { EnvelopeDimension } from '../api/types';
import { StatusPill } from './StatusPill';
import { NA, ppDelta } from '../lib/format';

function deltaCell(value: number | null | undefined) {
  const cls = value != null && Number.isFinite(value) ? (value > 0.0005 ? 'accent-green' : value < -0.0005 ? 'accent-red' : '') : '';
  return <td className={`mono ${cls}`}>{ppDelta(value)}</td>;
}

// Default columns are the headline six (bucket, chips, novelty, role
// accuracy, detector recall, status); the parent-image count and the two
// control-delta columns are secondary and only shown once "Show deltas" is
// toggled on (see EnvelopePage). Rendered as a plain section — no card
// border — per the five-dimension-tables-as-sections layout.
export function EnvelopeTable({ dimension, showDeltas }: { dimension: EnvelopeDimension; showDeltas: boolean }) {
  return (
    <div className="envelope-section">
      <div className="panel-title">{dimension.name}</div>
      <table className="data-table">
        <thead>
          <tr>
            <th>bucket</th>
            <th>chips</th>
            {showDeltas && <th>images</th>}
            <th>novelty</th>
            <th>role accuracy</th>
            <th>detector recall</th>
            {showDeltas && <th>Δ role acc</th>}
            {showDeltas && <th>Δ det recall</th>}
            <th>status</th>
          </tr>
        </thead>
        <tbody>
          {dimension.buckets.map((b) => (
            <tr key={b.label}>
              <td>
                {b.label}
                {b.in_reference_definition && (
                  <span className="tag-ref" title="Part of the reference-domain definition">
                    ref
                  </span>
                )}
              </td>
              <td className="mono">{b.n_chips.toLocaleString()}</td>
              {showDeltas && <td className="mono">{(b.n_parent_images ?? b.n_images).toLocaleString()}</td>}
              <td className="mono">{(b.novelty_rate * 100).toFixed(1)}%</td>
              <td className="mono">{b.role_accuracy != null ? `${(b.role_accuracy * 100).toFixed(1)}%` : NA}</td>
              <td className="mono">{b.detection_recall != null ? `${(b.detection_recall * 100).toFixed(1)}%` : NA}</td>
              {showDeltas && deltaCell(b.control_delta?.role_accuracy ?? null)}
              {showDeltas && deltaCell(b.control_delta?.detection_recall ?? null)}
              <td>
                <StatusPill status={b.status} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
