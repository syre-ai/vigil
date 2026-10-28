import { forwardRef, useImperativeHandle, useState } from 'react';
import { usePostReview, useReviewClasses, useUsedLabels } from '../hooks/useApi';
import type { Review, ReviewAction, ReviewSource } from '../api/types';
import { useToast } from '../state/useToast';
import { reviewSummary, shortRole } from '../lib/format';

export type ReviewActionsHandle = { trigger: (action: ReviewAction) => void };

function toastMessage(review: Review, rolePred?: string | null): string {
  if (review.action === 'skip') return 'Skipped';
  const summary = reviewSummary(review, rolePred);
  const suffix = review.action === 'new_class' ? ' · pending until Apply feedback' : '';
  return `Recorded: ${summary}${suffix}`;
}

export const ReviewActions = forwardRef<
  ReviewActionsHandle,
  { annId: number; rolePred?: string | null; currentReview?: Review | null; onDone?: (review: Review) => void; source?: ReviewSource }
>(function ReviewActions({ annId, rolePred, currentReview, onDone, source }, ref) {
  const mutation = usePostReview();
  const usedLabelsQuery = useUsedLabels();
  const classesQuery = useReviewClasses();
  const pushToast = useToast();
  const [showNewClass, setShowNewClass] = useState(false);
  const [label, setLabel] = useState(currentReview?.action === 'new_class' ? (currentReview.label ?? '') : '');
  const [showCorrect, setShowCorrect] = useState(false);
  const [correctLabel, setCorrectLabel] = useState(currentReview?.action === 'correct' ? (currentReview.label ?? '') : '');
  // Tracks the annId a "new class"/"correct role" entry belongs to, so a
  // stale label/drawer state from the previous chip never leaks onto the
  // next one (this component instance is reused across chips). Adjusting
  // state during render — rather than in an effect — avoids an extra render
  // pass; see
  // https://react.dev/learn/you-might-not-need-an-effect#adjusting-some-state-when-a-prop-changes
  const [resetForAnnId, setResetForAnnId] = useState(annId);
  if (annId !== resetForAnnId) {
    setResetForAnnId(annId);
    setShowNewClass(false);
    setLabel(currentReview?.action === 'new_class' ? (currentReview.label ?? '') : '');
    setShowCorrect(false);
    setCorrectLabel(currentReview?.action === 'correct' ? (currentReview.label ?? '') : '');
  }

  // Suggestions come only from the backend's review log, so a demo reset clears them.
  const suggestions = Array.from(new Set(usedLabelsQuery.data ?? []));
  const [comboOpen, setComboOpen] = useState(false);
  const [comboIdx, setComboIdx] = useState(-1);
  const filteredSuggestions = suggestions.filter((sug) => sug.toLowerCase().includes(label.trim().toLowerCase()));

  // "Correct role" only offers KNOWN classes (civil + analyst-created, from
  // /review/classes — the same list the backend validates action='correct'
  // against), excluding the chip's own current predicted role.
  const correctOptions = (classesQuery.data ?? []).filter((c) => c !== rolePred);

  function simpleAction(action: Exclude<ReviewAction, 'new_class'>) {
    if (mutation.isPending) return;
    mutation.mutate(
      { annId, action, source },
      {
        onSuccess: (review) => {
          pushToast(toastMessage(review, rolePred));
          onDone?.(review);
        },
      },
    );
  }

  function submitNewClass() {
    if (mutation.isPending) return;
    const trimmed = label.trim();
    if (!trimmed) return;
    mutation.mutate(
      { annId, action: 'new_class', label: trimmed, source },
      {
        onSuccess: (review) => {
          setLabel('');
          setShowNewClass(false);
          pushToast(toastMessage(review, rolePred));
          onDone?.(review);
        },
      },
    );
  }

  function submitCorrect() {
    if (mutation.isPending) return;
    if (!correctLabel) return;
    mutation.mutate(
      { annId, action: 'correct', label: correctLabel, source },
      {
        onSuccess: (review) => {
          setCorrectLabel('');
          setShowCorrect(false);
          pushToast(toastMessage(review, rolePred));
          onDone?.(review);
        },
      },
    );
  }

  useImperativeHandle(ref, () => ({
    trigger: (action) => {
      if (mutation.isPending) return;
      if (action === 'new_class') {
        setShowNewClass(true);
        return;
      }
      if (action === 'correct') {
        setShowCorrect(true);
        return;
      }
      simpleAction(action);
    },
  }));

  const active = currentReview?.action;

  return (
    <div className="review-actions">
      <div className="btn-group">
        <button
          type="button"
          className={`btn btn-sm${active === 'confirm' ? ' btn-review-active' : ''}`}
          onClick={() => simpleAction('confirm')}
          disabled={mutation.isPending}
          title="Records agreement; does not alter the model."
        >
          <kbd>1</kbd> Confirm role
        </button>
        <button
          type="button"
          className={`btn btn-sm${active === 'new_class' ? ' btn-review-active' : ''}`}
          onClick={() => setShowNewClass(true)}
          disabled={mutation.isPending}
        >
          <kbd>2</kbd> New class
        </button>
        <button
          type="button"
          className={`btn btn-sm${active === 'not_aircraft' ? ' btn-review-active' : ''}`}
          onClick={() => simpleAction('not_aircraft')}
          disabled={mutation.isPending}
        >
          <kbd>3</kbd> Not aircraft
        </button>
        <button
          type="button"
          className={`btn btn-sm${active === 'correct' ? ' btn-review-active' : ''}`}
          onClick={() => setShowCorrect(true)}
          disabled={mutation.isPending}
          title="Records disagreement with the predicted role and assigns a known class; does not alter the model."
        >
          <kbd>4</kbd> Correct role
        </button>
        <button
          type="button"
          className={`btn btn-sm${active === 'bad_box' ? ' btn-review-active' : ''}`}
          onClick={() => simpleAction('bad_box')}
          disabled={mutation.isPending}
          title="Detector box covers two aircraft or cuts one. Recorded as a localization error; not used for any class."
        >
          <kbd>5</kbd> Bad box
        </button>
        <button
          type="button"
          className={`btn btn-sm${active === 'skip' ? ' btn-review-active' : ''}`}
          onClick={() => simpleAction('skip')}
          disabled={mutation.isPending}
        >
          <kbd>s</kbd> Skip
        </button>
      </div>
      {showNewClass && (
        <div className="new-class-form">
          <label className="field-label" htmlFor="new-class-label">
            New class label
          </label>
          <div className="combo">
            <input
              id="new-class-label"
              type="text"
              autoComplete="off"
              value={label}
              autoFocus
              onChange={(e) => { setLabel(e.target.value); setComboOpen(true); setComboIdx(-1); }}
              onFocus={() => setComboOpen(true)}
              onBlur={() => window.setTimeout(() => setComboOpen(false), 120)}
              onKeyDown={(e) => {
                const opts = filteredSuggestions;
                if (e.key === 'ArrowDown' && opts.length) { e.preventDefault(); setComboOpen(true); setComboIdx((i) => (i + 1) % opts.length); return; }
                if (e.key === 'ArrowUp' && opts.length) { e.preventDefault(); setComboIdx((i) => (i <= 0 ? opts.length - 1 : i - 1)); return; }
                if (e.key === 'Enter') {
                  if (comboOpen && comboIdx >= 0 && opts[comboIdx] !== undefined) { setLabel(opts[comboIdx]); setComboOpen(false); setComboIdx(-1); return; }
                  submitNewClass(); return;
                }
                if (e.key === 'Escape') { if (comboOpen) { setComboOpen(false); } else { setShowNewClass(false); } }
              }}
              placeholder="e.g. Military Trainer"
            />
            {comboOpen && filteredSuggestions.length > 0 && (
              <ul className="combo-list" role="listbox">
                {filteredSuggestions.map((sug, i) => (
                  <li
                    key={sug}
                    role="option"
                    aria-selected={i === comboIdx}
                    className={i === comboIdx ? 'active' : undefined}
                    onMouseDown={(e) => { e.preventDefault(); setLabel(sug); setComboOpen(false); setComboIdx(-1); }}
                  >
                    {sug}
                  </li>
                ))}
              </ul>
            )}
          </div>
          <div className="btn-group" style={{ marginTop: 6 }}>
            <button type="button" className="btn btn-sm btn-primary" onClick={submitNewClass} disabled={!label.trim() || mutation.isPending}>
              Submit
            </button>
            <button type="button" className="btn btn-sm" onClick={() => setShowNewClass(false)}>
              Cancel
            </button>
          </div>
        </div>
      )}
      {showCorrect && (
        <div className="new-class-form">
          <label className="field-label" id="correct-role-label">
            Corrected role: records disagreement, does not alter the model
          </label>
          {classesQuery.isLoading ? (
            <div className="caption">Loading known classes…</div>
          ) : correctOptions.length === 0 ? (
            <div className="caption">No other known class to correct to yet.</div>
          ) : (
            <div className="correct-role-options" role="radiogroup" aria-labelledby="correct-role-label">
              {correctOptions.map((cls) => (
                <label key={cls} className="correct-role-option">
                  <input type="radio" name="correct-role" value={cls} checked={correctLabel === cls} onChange={() => setCorrectLabel(cls)} />
                  {shortRole(cls)}
                </label>
              ))}
            </div>
          )}
          <div className="btn-group" style={{ marginTop: 6 }}>
            <button type="button" className="btn btn-sm btn-primary" onClick={submitCorrect} disabled={!correctLabel || mutation.isPending}>
              Submit
            </button>
            <button type="button" className="btn btn-sm" onClick={() => setShowCorrect(false)}>
              Cancel
            </button>
          </div>
        </div>
      )}
      {mutation.isError && <div className="state-error" style={{ fontSize: 11, marginTop: 4 }}>Review failed to save.</div>}
    </div>
  );
});
