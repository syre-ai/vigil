import type { ChangedDetChip } from '../api/types';
import { score, shortRole } from '../lib/format';

// A dedicated grid/card for the "Changed in this version" page rather than reusing ChipCard
// directly: these chips need a before/after role line ("Small civil → Military fighter") and the
// analyst-class similarity that drove the re-sort, neither of which ChipCard renders. Reuses
// ChipCard's own CSS classes (chip-card / chip-card-img-wrap / chip-card-tags / chip-card-body /
// etc. — see index.css's "chip grid / card" section) so a changed-chip card reads as the same
// visual family as the field ChipCard, without editing that shared component.
function ChangedChipCard({ chip, onClick }: { chip: ChangedDetChip; onClick: (annId: number) => void }) {
  return (
    <button type="button" className={`chip-card${chip.is_novel ? ' chip-card-novel' : ''}`} onClick={() => onClick(chip.ann_id)}>
      <div className="chip-card-img-wrap">
        <img src={chip.chip_url} alt={chip.role_pred ?? 'unscored chip'} loading="lazy" width={224} height={224} />
        {chip.review && <div className="chip-card-reviewed mono">{chip.review.action}</div>}
      </div>
      <div className="chip-card-tags">
        {chip.is_novel && <span className="chip-card-tag chip-card-tag-novel">Flagged</span>}
        {chip.same_parent && (
          <span className="chip-card-tag chip-card-tag-edge" title="Also from the same parent image as a chip used to fit this analyst class: correlated, not independent, evidence">
            same collection as labels
          </span>
        )}
      </div>
      <div className="chip-card-body">
        <div className="chip-card-role" title={`${chip.role_pred_a ?? 'unscored'} → ${chip.role_pred_b ?? 'unscored'}`}>
          {shortRole(chip.role_pred_a)} → {shortRole(chip.role_pred_b)}
        </div>
        <div className="chip-card-row mono">
          <span title="cosine similarity to the nearest analyst class mean">sim {score(chip.analyst_sim, 2)}</span>
          <span title="detector confidence">conf {score(chip.conf, 2)}</span>
        </div>
      </div>
    </button>
  );
}

export function ChangedChipGrid({
  chips,
  onSelect,
  emptyLabel = 'No changed detections match the current filters.',
}: {
  chips: ChangedDetChip[];
  onSelect: (annId: number) => void;
  emptyLabel?: string;
}) {
  if (!chips.length) return <div className="state-block">{emptyLabel}</div>;
  return (
    <div className="chip-grid">
      {chips.map((c) => (
        <ChangedChipCard key={c.ann_id} chip={c} onClick={onSelect} />
      ))}
    </div>
  );
}
