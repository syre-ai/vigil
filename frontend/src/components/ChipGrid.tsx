import { ChipCard, type ChipCardData } from './ChipCard';

export function ChipGrid({
  chips,
  onSelect,
  variant,
  emptyLabel = 'No chips match the current filters.',
}: {
  chips: ChipCardData[];
  onSelect: (annId: number) => void;
  variant?: 'benchmark' | 'field';
  emptyLabel?: string;
}) {
  if (!chips.length) return <div className="state-block">{emptyLabel}</div>;
  return (
    <div className="chip-grid">
      {chips.map((c) => (
        <ChipCard key={c.ann_id} chip={c} onClick={onSelect} variant={variant} />
      ))}
    </div>
  );
}
