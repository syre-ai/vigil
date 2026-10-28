import type { Batch, EvalResponse } from '../api/types';
import { computeFindings, ENVELOPE_STATUS_LABEL } from '../lib/envelopeStatus';
import { InfoPopover } from './InfoPopover';

export function FindingsPanel({ batches, evalData }: { batches: Batch[]; evalData?: EvalResponse }) {
  const findings = computeFindings(batches, evalData);
  if (!findings.length) return null;
  return (
    <div className="panel">
      <div className="panel-title">
        Findings
        <InfoPopover label="About the findings list">Auto-generated from the live batch, drift, and evaluation numbers above. Nothing here is hard-coded.</InfoPopover>
      </div>
      <ul className="findings-list">
        {findings.map((f, i) => (
          <li key={i}>
            {f.status ? <span className={`pill pill-${f.status}`}>{ENVELOPE_STATUS_LABEL[f.status]}</span> : <span className="findings-list-marker">›</span>}
            <span>{f.text}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}
