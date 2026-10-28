import type { VersionsCompare } from '../api/types';
import { InfoPopover } from './InfoPopover';
import { NA, pct, ppDelta } from '../lib/format';

function DeltaCell({ a, b, invert = false }: { a: number | null; b: number | null; invert?: boolean }) {
  if (a == null || b == null) return <td className="mono">{NA}</td>;
  const delta = b - a;
  const directional = invert ? -delta : delta;
  const regressed = directional < -0.005;
  const improved = directional > 0.005;
  return (
    <td className={`mono ${regressed ? 'accent-red' : improved ? 'accent-green' : ''}`}>
      {pct(a)} → {pct(b)} ({ppDelta(delta)})
    </td>
  );
}

// Primary analyst-class assignment rate, falling back to the rare_target_recall alias
// if the backend hasn't sent analyst_class_assignment_rate_* yet.
function assignmentRate(military: VersionsCompare['military'], which: 'a' | 'b'): number | null {
  const primary = which === 'a' ? military.analyst_class_assignment_rate_a : military.analyst_class_assignment_rate_b;
  const alias = which === 'a' ? military.rare_target_recall_a : military.rare_target_recall_b;
  return primary ?? alias ?? null;
}

export function CompareTable({ compare }: { compare: VersionsCompare }) {
  const regressed = compare.forgetting > 0.01;
  const rateA = assignmentRate(compare.military, 'a');
  const rateB = assignmentRate(compare.military, 'b');
  const civilFalseA = compare.military.civil_false_assignment_rate_a ?? null;
  const civilFalseB = compare.military.civil_false_assignment_rate_b ?? null;
  const hasCivilFalse = civilFalseA != null || civilFalseB != null;
  return (
    <div className="panel">
      <div className="panel-title">
        Version compare · {compare.a} → {compare.b}
        <InfoPopover label="About this comparison" width={340}>
          {compare.military.exclusion_method ? `${compare.military.exclusion_method}. ` : ''}
          {compare.military.exclusion_method ? 'Labeled chips and their parent images are excluded from evaluation.' : 'Labeled chips are excluded from evaluation.'}
          {compare.military.adaptation_rule && <p className="info-popover-text">{compare.military.adaptation_rule}</p>}
          {compare.analyst_disagreement && (
            <p className="info-popover-text">
              Analyst feedback (all time, not per version): {compare.analyst_disagreement.n_confirm} confirmed, {compare.analyst_disagreement.n_correct} corrected.
              Disagreement rate {pct(compare.analyst_disagreement.disagreement_rate)}. Neither action alters the model.
            </p>
          )}
          <p className="info-popover-text">
            Forgetting = drop in role accuracy on ref_holdout (never used for fitting or calibration) between versions. Analyst-class assignment rate is measured only on
            held-out military chips <em>not</em> used to fit any prototype, so the model isn&apos;t graded on what it just adapted to.
          </p>
        </InfoPopover>
      </div>
      <div className={`regression-banner ${regressed ? 'regression-bad' : 'regression-ok'}`}>
        {regressed
          ? `Forgetting detected: control-batch role accuracy dropped ${(compare.forgetting * 100).toFixed(1)} points.`
          : 'No meaningful forgetting on the control batch (ref_holdout).'}
      </div>

      <table className="data-table" style={{ marginTop: 10 }}>
        <thead>
          <tr>
            <th>metric</th>
            <th>{compare.a}</th>
            <th>{compare.b}</th>
            <th>change</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>role accuracy, control batch (ref_holdout)</td>
            <td className="mono">{pct(compare.ref_holdout.role_accuracy_a)}</td>
            <td className="mono">{pct(compare.ref_holdout.role_accuracy_b)}</td>
            <DeltaCell a={compare.ref_holdout.role_accuracy_a} b={compare.ref_holdout.role_accuracy_b} />
          </tr>
          {compare.ref_holdout.per_class.map((row) => (
            <tr key={row.cls}>
              <td>per-class accuracy · {row.cls}</td>
              <td className="mono">{pct(row.a)}</td>
              <td className="mono">{pct(row.b)}</td>
              <DeltaCell a={row.a} b={row.b} />
            </tr>
          ))}
          <tr>
            <td>analyst-class assignment rate (held-out military chips) (n_eval {compare.military.n_eval_a} → {compare.military.n_eval_b})</td>
            <td className="mono">{pct(rateA)}</td>
            <td className="mono">{pct(rateB)}</td>
            <DeltaCell a={rateA} b={rateB} />
          </tr>
          {hasCivilFalse && (
            <tr>
              <td>
                civil chips wrongly assigned
                {compare.military.n_eval_civil_a != null || compare.military.n_eval_civil_b != null
                  ? ` (n_eval_civil ${compare.military.n_eval_civil_a ?? NA} → ${compare.military.n_eval_civil_b ?? NA})`
                  : ''}
              </td>
              <td className="mono">{pct(civilFalseA)}</td>
              <td className="mono">{pct(civilFalseB)}</td>
              <DeltaCell a={civilFalseA} b={civilFalseB} invert />
            </tr>
          )}
          <tr>
            <td>miss rate · still predicted civil</td>
            <td className="mono">{pct(compare.military.miss_rate_civil_a)}</td>
            <td className="mono">{pct(compare.military.miss_rate_civil_b)}</td>
            <DeltaCell a={compare.military.miss_rate_civil_a} b={compare.military.miss_rate_civil_b} invert />
          </tr>
        </tbody>
      </table>

      <div style={{ height: 10 }} />
      <table className="data-table">
        <thead>
          <tr>
            <th>novelty rate by batch</th>
            <th>{compare.a}</th>
            <th>{compare.b}</th>
            <th>change</th>
          </tr>
        </thead>
        <tbody>
          {compare.novelty_rate_by_batch.map((row) => (
            <tr key={row.batch_id}>
              <td>{row.batch_id}</td>
              <td className="mono">{pct(row.a)}</td>
              <td className="mono">{pct(row.b)}</td>
              <td className="mono">{row.a != null && row.b != null ? ppDelta(row.b - row.a) : NA}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
