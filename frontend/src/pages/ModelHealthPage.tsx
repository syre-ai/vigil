import { ModelHealthPanel } from '../components/ModelHealthPanel';

/** Analyst view: the retrain-readiness decision on its own page (indicators, rules, evidence). */
export function ModelHealthPage() {
  return (
    <div>
      <div className="page-header">
        <div>
          <h1 className="page-title">Model health</h1>
          <div className="page-sub">
            Keep, patch, or schedule a retrain: from evidence the field can measure plus the labeled control set. Indicative; every rule is stated.
          </div>
        </div>
      </div>
      <ModelHealthPanel id="health" />
    </div>
  );
}
