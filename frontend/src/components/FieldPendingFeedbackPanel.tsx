import type { Review, ReviewAction } from '../api/types';
import { useDetDetail } from '../hooks/useApi';
import { getCachedDetChip } from '../state/detChipCache';
import { Details } from './Details';
import { REVIEW_ACTION_LABELS, shortRole } from '../lib/format';

// 'not_aircraft' and 'bad_box' are each broken out into their own badge below
// (see PendingFeedbackPanel vs this field-mode twin) — they're the field
// precision/localization-error evidence the analyst view's queue info popover
// promises, not just another feedback-action count.
const ACTION_ORDER: ReviewAction[] = ['new_class', 'correct', 'confirm', 'skip'];

// Field-mode twin of PendingFeedbackPanel — same pattern, but for det chips
// (source=det review log), reading from the det chip cache/`/det/{id}`
// instead of the GT chip cache/`/chips/{ann_id}`.
export function FieldPendingFeedbackPanel({ reviews, onSelect }: { reviews: Review[] | undefined; onSelect: (detId: number) => void }) {
  const pending = (reviews ?? []).filter((r) => r.version_applied == null);

  if (pending.length === 0) {
    return <span className="pending-feedback-summary">No pending feedback yet. Review flagged chips, then apply.</span>;
  }

  const counts: Record<ReviewAction, number> = { confirm: 0, new_class: 0, not_aircraft: 0, skip: 0, correct: 0, bad_box: 0 };
  const labelCounts = new Map<string, number>();
  for (const r of pending) {
    counts[r.action] += 1;
    if (r.action === 'new_class' && r.label) {
      labelCounts.set(r.label, (labelCounts.get(r.label) ?? 0) + 1);
    }
  }
  const breakdown = ACTION_ORDER.filter((a) => counts[a] > 0)
    .map((a) => `${counts[a]} ${REVIEW_ACTION_LABELS[a].toLowerCase()}`)
    .join(', ');
  const notAircraftCount = counts.not_aircraft;
  const badBoxCount = counts.bad_box;
  const rows = [...pending].sort((a, b) => (a.ts < b.ts ? 1 : a.ts > b.ts ? -1 : 0));

  return (
    <span className="pending-feedback-summary">
      <span className="mono">{pending.length}</span> pending
      {breakdown && <>: {breakdown}</>}
      {notAircraftCount > 0 && (
        <span className="pending-feedback-not-aircraft" title="Recorded as evidence of field precision on flagged detections">
          <span className="mono">{notAircraftCount}</span> not aircraft
        </span>
      )}
      {badBoxCount > 0 && (
        <span className="pending-feedback-bad-box" title="Localization error (fused/truncated aircraft): never used to fit any class">
          <span className="mono">{badBoxCount}</span> bad box
        </span>
      )}
      <Details id="field-review-pending-feedback" label="Details" closeLabel="Hide">
        <div className="panel pending-feedback-expanded">
          {labelCounts.size > 0 && (
            <div className="pending-feedback-labels">
              {[...labelCounts.entries()].map(([label, n]) => (
                <span key={label} className="pending-feedback-label-pill">
                  {label} ×{n}
                </span>
              ))}
            </div>
          )}
          <ul className="pending-feedback-list">
            {rows.map((r) => (
              <FieldPendingFeedbackRow key={r.ann_id} review={r} onSelect={onSelect} />
            ))}
          </ul>
          <div className="caption" style={{ marginTop: 6 }}>
            Click a row to reopen that chip. Cleared once &quot;Apply feedback&quot; folds it into a new version.
          </div>
        </div>
      </Details>
    </span>
  );
}

function FieldPendingFeedbackRow({ review, onSelect }: { review: Review; onSelect: (detId: number) => void }) {
  const cached = getCachedDetChip(review.ann_id);
  const chipQuery = useDetDetail(cached ? undefined : review.ann_id);
  const chip = cached ?? chipQuery.data;

  return (
    <li>
      <button type="button" className="pending-feedback-row" onClick={() => onSelect(review.ann_id)}>
        {chip ? <img src={chip.chip_url} alt="" width={32} height={32} loading="lazy" /> : <span className="pending-feedback-thumb-skel" />}
        <span className="pending-feedback-row-meta">
          <span className="mono">#{review.ann_id}</span>
          <span>{chip ? shortRole(chip.role_pred) : '…'}</span>
        </span>
        <span className="pending-feedback-row-action">{REVIEW_ACTION_LABELS[review.action]}</span>
        {review.label && <span className="pending-feedback-label-pill">{review.label}</span>}
      </button>
    </li>
  );
}
