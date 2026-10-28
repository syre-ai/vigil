import { chipImageUrl } from '../api/client';
import type { Chip, Review, ReviewAction } from '../api/types';
import { useChipDetail } from '../hooks/useApi';
import { getCachedChip } from '../state/chipCache';
import { Details } from './Details';
import { REVIEW_ACTION_LABELS, shortRole } from '../lib/format';

const ACTION_ORDER: ReviewAction[] = ['new_class', 'correct', 'confirm', 'not_aircraft', 'bad_box', 'skip'];

// Compact one-line summary of every not-yet-applied review
// (`version_applied == null`) shown next to the Apply-feedback button, e.g.
// "4 pending: 3 new class, 1 correct". Details expands the full list an
// analyst can use to see, at a glance, what "Apply feedback" is about to fold
// into the next version — and reopen any of those chips. Chip payloads come
// from the shared session cache (populated by the review queue / chip drawer
// as chips are seen) with a `/chips/{ann_id}` fallback for ids never seen
// this session (e.g. right after a page reload).
export function PendingFeedbackPanel({ reviews, onSelect }: { reviews: Review[] | undefined; onSelect: (annId: number) => void }) {
  const pending = (reviews ?? []).filter((r) => r.version_applied == null);

  if (pending.length === 0) {
    return <span className="pending-feedback-summary">No pending feedback yet. Review chips, then apply.</span>;
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
  const rows = [...pending].sort((a, b) => (a.ts < b.ts ? 1 : a.ts > b.ts ? -1 : 0));

  return (
    <span className="pending-feedback-summary">
      <span className="mono">{pending.length}</span> pending: {breakdown}
      <Details id="review-pending-feedback" label="Details" closeLabel="Hide">
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
              <PendingFeedbackRow key={r.ann_id} review={r} onSelect={onSelect} />
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

function PendingFeedbackRow({ review, onSelect }: { review: Review; onSelect: (annId: number) => void }) {
  const cached = getCachedChip(review.ann_id);
  const chipQuery = useChipDetail(cached ? undefined : review.ann_id);
  const chip: Chip | undefined = cached ?? chipQuery.data;

  return (
    <li>
      <button type="button" className="pending-feedback-row" onClick={() => onSelect(review.ann_id)}>
        {chip ? <img src={chipImageUrl(chip.ann_id)} alt="" width={32} height={32} loading="lazy" /> : <span className="pending-feedback-thumb-skel" />}
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
