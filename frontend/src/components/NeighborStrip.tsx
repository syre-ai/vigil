import { NA, shortRole } from '../lib/format';

// Shared by the GT chip drawer (ChipNeighbor, dist always present) and the
// field-mode det chip drawer (DetChipNeighbor, dist optional) — see api/types.ts.
type NeighborLike = { ann_id: number; chip_url: string; role_true: string | null; dist?: number };

export function NeighborStrip({ neighbors }: { neighbors: NeighborLike[] }) {
  if (!neighbors.length) return null;
  return (
    <div className="neighbor-strip">
      {neighbors.map((n) => (
        <div className="neighbor-item" key={n.ann_id}>
          <img src={n.chip_url} alt={n.role_true ?? 'reference chip'} width={72} height={72} loading="lazy" />
          <div className="neighbor-meta mono">
            <div title={n.role_true ?? undefined}>{shortRole(n.role_true)}</div>
            <div>d={n.dist != null ? n.dist.toFixed(3) : NA}</div>
          </div>
        </div>
      ))}
    </div>
  );
}
