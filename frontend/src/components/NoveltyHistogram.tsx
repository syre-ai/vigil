import { Area, CartesianGrid, ComposedChart, Legend, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import type { Histogram } from '../api/types';

type DensityPoint = { x: number; density: number };

// The backend has moved novelty_hist/ref_novelty_hist from "one count per
// bin label" (bins.length === counts.length) to "bin edges" (bins.length ===
// counts.length + 1, ~30 edges spanning the observed range). Detect which
// shape arrived and always emit one {x, density} point per count, with x at
// the bin's center for the edges shape.
function toDensitySeries(hist: Histogram): DensityPoint[] {
  const total = hist.counts.reduce((a, b) => a + b, 0) || 1;
  if (hist.bins.length === hist.counts.length + 1) {
    return hist.counts.map((c, i) => ({ x: (hist.bins[i] + hist.bins[i + 1]) / 2, density: c / total }));
  }
  const n = Math.min(hist.bins.length, hist.counts.length);
  const out: DensityPoint[] = [];
  for (let i = 0; i < n; i++) out.push({ x: hist.bins[i], density: hist.counts[i] / total });
  return out;
}

export function NoveltyHistogram({ batchHist, refHist, vigilance }: { batchHist: Histogram; refHist: Histogram; vigilance?: number | null }) {
  const batchData = toDensitySeries(batchHist);
  const refData = toDensitySeries(refHist);
  const allX = [...batchData, ...refData].map((d) => d.x);
  const xMin = allX.length ? Math.min(...allX) : 0;
  const xMax = allX.length ? Math.max(...allX) : 1;
  const vigilanceX = vigilance != null ? 1 - vigilance : null;

  return (
    <ResponsiveContainer width="100%" height={220}>
      <ComposedChart margin={{ top: 8, right: 16, left: -12, bottom: 4 }}>
        <CartesianGrid stroke="var(--border-soft)" vertical={false} />
        <XAxis dataKey="x" type="number" domain={[xMin, xMax]} tick={{ fill: '#8a93a1', fontSize: 11 }} tickFormatter={(v: number) => v.toFixed(2)} />
        <YAxis tick={{ fill: '#8a93a1', fontSize: 11 }} width={42} tickFormatter={(v: number) => `${Math.round(v * 100)}%`} />
        <Tooltip
          contentStyle={{ background: '#ffffff', border: '1px solid #e2e5ea', fontSize: 11, borderRadius: 6 }}
          labelStyle={{ color: '#111827' }}
          labelFormatter={(v) => `novelty ${Number(v).toFixed(2)}`}
          formatter={(value, name) => [`${(Number(value) * 100).toFixed(1)}%`, String(name)]}
        />
        <Legend wrapperStyle={{ fontSize: 11 }} />
        <Area data={refData} dataKey="density" name="reference domain" type="stepAfter" stroke="#1f5aa6" fill="#1f5aa6" fillOpacity={0.12} strokeWidth={1.5} isAnimationActive={false} />
        <Area data={batchData} dataKey="density" name="this batch" type="stepAfter" stroke="#d97706" fill="#d97706" fillOpacity={0.15} strokeWidth={1.5} isAnimationActive={false} />
        {vigilanceX != null && (
          <ReferenceLine
            x={vigilanceX}
            stroke="#dc2626"
            strokeDasharray="4 3"
            label={{ value: 'vigilance threshold', position: 'insideTopRight', fill: '#dc2626', fontSize: 11 }}
          />
        )}
      </ComposedChart>
    </ResponsiveContainer>
  );
}
