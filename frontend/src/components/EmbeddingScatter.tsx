import { CartesianGrid, Legend, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis } from 'recharts';
import { chipImageUrl } from '../api/client';
import type { EmbeddingBatchPoint, EmbeddingRefPoint } from '../api/types';
import { NA, score } from '../lib/format';

type TooltipPoint = {
  ann_id: number;
  x: number;
  y: number;
  z: number;
  kind: 'reference' | 'batch';
  role_true?: string | null;
  novelty?: number;
  is_novel?: boolean;
};

// recharts 3.10's default circle "symbol" shape renders a <path> with no `d`
// attribute in this environment (a bug — points end up invisible even though
// they're positioned correctly). A custom shape sidesteps it by drawing a
// plain <circle>.
function dot(radius: number) {
  return function Dot(props: { cx?: number; cy?: number; fill?: string; fillOpacity?: number }) {
    if (props.cx === undefined || props.cy === undefined) return null;
    return <circle cx={props.cx} cy={props.cy} r={radius} fill={props.fill} fillOpacity={props.fillOpacity ?? 1} />;
  };
}

function CustomTooltip({ active, payload }: { active?: boolean; payload?: { payload: TooltipPoint }[] }) {
  if (!active || !payload?.length) return null;
  const p = payload[0].payload;
  return (
    <div className="embedding-tooltip">
      <img src={chipImageUrl(p.ann_id)} alt="" width={56} height={56} loading="lazy" />
      <div className="mono">
        <div>ann_id {p.ann_id}</div>
        {p.kind === 'reference' ? (
          <div>role_true {p.role_true ?? NA}</div>
        ) : (
          <>
            <div>novelty {score(p.novelty)}</div>
            <div>{p.is_novel ? 'flagged novel' : 'not novel'}</div>
          </>
        )}
      </div>
    </div>
  );
}

export function EmbeddingScatter({ reference, batch }: { reference: EmbeddingRefPoint[]; batch: EmbeddingBatchPoint[] }) {
  const refData: TooltipPoint[] = reference.map((r) => ({ ann_id: r.ann_id, x: r.x, y: r.y, z: 1, kind: 'reference', role_true: r.role_true }));
  const normal: TooltipPoint[] = batch.filter((b) => !b.is_novel).map((b) => ({ ann_id: b.ann_id, x: b.x, y: b.y, z: 1, kind: 'batch', novelty: b.novelty, is_novel: b.is_novel }));
  const novel: TooltipPoint[] = batch.filter((b) => b.is_novel).map((b) => ({ ann_id: b.ann_id, x: b.x, y: b.y, z: 1, kind: 'batch', novelty: b.novelty, is_novel: b.is_novel }));

  return (
    <ResponsiveContainer width="100%" height={440}>
      <ScatterChart margin={{ top: 10, right: 16, bottom: 10, left: 0 }}>
        <CartesianGrid stroke="var(--border-soft)" />
        <XAxis type="number" dataKey="x" tick={{ fill: '#8a93a1', fontSize: 11 }} name="dim 1" />
        <YAxis type="number" dataKey="y" tick={{ fill: '#8a93a1', fontSize: 11 }} name="dim 2" />
        <ZAxis dataKey="z" range={[24, 24]} />
        <Tooltip content={<CustomTooltip />} />
        <Legend wrapperStyle={{ fontSize: 11 }} />
        <Scatter name="reference" data={refData} fill="#9aa3af" fillOpacity={0.55} shape={dot(2.5)} isAnimationActive={false} />
        <Scatter name="batch (not novel)" data={normal} fill="#1f5aa6" shape={dot(4)} isAnimationActive={false} />
        <Scatter name="batch (novel)" data={novel} fill="#d97706" shape={dot(4)} isAnimationActive={false} />
      </ScatterChart>
    </ResponsiveContainer>
  );
}
