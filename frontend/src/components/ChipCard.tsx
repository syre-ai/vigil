import type { Review } from '../api/types';
import { edgeLabel, isEdgeChip, score } from '../lib/format';

// Structural subset shared by GT Chip and field-mode DetChip — lets this
// card (and ChipGrid) render either without a cast. `ann_id` doubles as the
// det chip's det_id (DetChip.ann_id is an alias of det_id, see api/types.ts).
export type ChipCardData = {
  ann_id: number;
  chip_url: string;
  role_pred: string | null; // null for an unscored det chip (no score row yet)
  role_conf: number | null;
  novelty: number | null;
  is_novel: boolean | null;
  review: Review | null;
  detected?: boolean | null;
  truncated?: boolean | null;
  partial?: number | null;
};

// `variant='field'` (analyst view, det chips): the vigilance tag reads
// "Flagged" (the field-mode word for is_novel), role confidence is shown on
// the card, and nothing ground-truth-based (Not detected / edge fraction —
// GT-annotation fields) can appear because DetChip never carries it.
export function ChipCard({ chip, onClick, variant = 'benchmark' }: { chip: ChipCardData; onClick: (annId: number) => void; variant?: 'benchmark' | 'field' }) {
  const isField = variant === 'field';
  const showEdge = !isField && isEdgeChip(chip);
  return (
    <button type="button" className={`chip-card${chip.is_novel ? ' chip-card-novel' : ''}`} onClick={() => onClick(chip.ann_id)}>
      <div className="chip-card-img-wrap">
        <img src={chip.chip_url} alt={chip.role_pred ?? 'unscored chip'} loading="lazy" width={224} height={224} />
        {chip.review && <div className="chip-card-reviewed mono">{chip.review.action}</div>}
      </div>
      <div className="chip-card-tags">
        {chip.is_novel && <span className="chip-card-tag chip-card-tag-novel">{isField ? 'Flagged' : 'Novel'}</span>}
        {!isField && chip.detected === false && (
          <span className="chip-card-tag chip-card-tag-missed" title="Missed by the detector at IoU ≥ 0.5">
            Not detected
          </span>
        )}
        {showEdge && <span className="chip-card-tag chip-card-tag-edge">{edgeLabel(chip)}</span>}
      </div>
      <div className="chip-card-body">
        <div className="chip-card-role" title={chip.role_pred ?? undefined}>
          {chip.role_pred ?? 'not yet scored'}
        </div>
        <div className="chip-card-row mono">
          <span className={chip.is_novel ? 'accent-amber' : ''} title={`role confidence ${score(chip.role_conf, 2)}`}>
            novelty {score(chip.novelty, 2)}
          </span>
          {isField && <span title="role confidence">conf {score(chip.role_conf, 2)}</span>}
        </div>
      </div>
    </button>
  );
}
