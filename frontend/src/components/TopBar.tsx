import { HelpCircle } from 'lucide-react';
import { useCollectionsSummary, useSummary } from '../hooks/useApi';
import { count, pct } from '../lib/format';
import { useView } from '../state/useView';
import { ViewToggle } from './ViewToggle';

function AnalystStats() {
  const summary = useCollectionsSummary();
  const data = summary.data;
  if (!data) return null;
  return (
    <span className="topbar-stats">
      {count(data.n_groups)} new collections · {count(data.n_needs_review)} need review · {count(data.n_flagged)} of {count(data.n_detections)} detections flagged
      {data.reference_flag_rate != null ? ` · validation-set flag rate ${pct(data.reference_flag_rate)}` : ''}
    </span>
  );
}

function BenchmarkStats() {
  const summary = useSummary();
  const data = summary.data;
  if (!data) return null;
  return (
    <span className="topbar-stats">
      {data.current_version} · {count(data.totals.chips)} operational chips · {count(data.totals.novel)} flagged · {count(data.totals.reviewed)} reviews
    </span>
  );
}

export function TopBar({ onGuideOpen }: { onGuideOpen: () => void }) {
  const { view } = useView();
  return (
    <header className="topbar">
      <div className="topbar-brand">
        <span className="topbar-name">Vigil</span>
        <span className="topbar-tag">assurance evidence prototype</span>
      </div>
      <div className="topbar-right">
        {view === 'analyst' ? <AnalystStats /> : <BenchmarkStats />}
        <ViewToggle />
        <button type="button" className="btn btn-sm" onClick={onGuideOpen}>
          <HelpCircle size={14} strokeWidth={1.75} />
          How to read this
        </button>
      </div>
    </header>
  );
}
