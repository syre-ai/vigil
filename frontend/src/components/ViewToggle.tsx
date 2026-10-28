import { useNav } from '../state/useNav';
import { useView } from '../state/useView';
import { InfoPopover } from './InfoPopover';

// Analyst view (default) / Benchmark view toggle, persisted in localStorage
// (state/view.tsx). Switching view navigates to that view's home page.
export function ViewToggle() {
  const { view, setView } = useView();
  const { goto } = useNav();

  function switchTo(next: 'analyst' | 'benchmark') {
    if (next === view) return;
    setView(next);
    goto(next === 'analyst' ? 'reference' : 'overview');
  }

  return (
    <span className="view-toggle-wrap">
    <div className="btn-group view-toggle" role="group" aria-label="Analyst view or Benchmark view">
      <button
        type="button"
        className={`btn btn-sm${view === 'analyst' ? ' btn-primary' : ''}`}
        onClick={() => switchTo('analyst')}
        aria-pressed={view === 'analyst'}
      >
        Analyst view
      </button>
      <button
        type="button"
        className={`btn btn-sm${view === 'benchmark' ? ' btn-primary' : ''}`}
        onClick={() => switchTo('benchmark')}
        aria-pressed={view === 'benchmark'}
      >
        Benchmark view
      </button>
    </div>
      <InfoPopover label="About the two views" align="right" width={320}>
        <p className="info-popover-text">
          Benchmark view uses the dataset&apos;s ground truth to check the tool; the analyst view shows only what a fielded system would have: the detector&apos;s own boxes, the
          labeled validation set, and the analyst&apos;s decisions.
        </p>
      </InfoPopover>
    </span>
  );
}
