import { Bar, BarChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import type { Batch } from '../api/types';
import { pct, shortBatchName } from '../lib/format';

export function NoveltyRateChart({ batches }: { batches: Batch[] }) {
  // Short, non-clipping axis labels derived from batch id; the tooltip still
  // shows the full Batch.name via labelFormatter.
  const data = batches.map((b) => ({ name: b.name, novelty_rate: b.novelty_rate, shortName: shortBatchName(b.id) }));
  const fullNameByShort = new Map(data.map((d) => [d.shortName, d.name]));
  return (
    <ResponsiveContainer width="100%" height={170}>
      <BarChart data={data} margin={{ top: 6, right: 8, left: 0, bottom: 4 }}>
        <XAxis dataKey="shortName" tick={{ fill: 'var(--text-faint)', fontSize: 11 }} axisLine={false} tickLine={false} interval={0} height={24} />
        <YAxis tick={{ fill: 'var(--text-faint)', fontSize: 11 }} axisLine={false} tickLine={false} width={30} tickCount={4} tickFormatter={(v: number) => `${Math.round(v * 100)}%`} />
        <Tooltip
          contentStyle={{ background: 'var(--surface)', border: '1px solid var(--border)', fontSize: 11, borderRadius: 0 }}
          labelStyle={{ color: 'var(--text)' }}
          labelFormatter={(label) => fullNameByShort.get(String(label)) ?? label}
          formatter={(value) => [pct(Number(value)), 'novelty rate']}
        />
        <Bar dataKey="novelty_rate" fill="var(--series-flagged)" radius={0} />
      </BarChart>
    </ResponsiveContainer>
  );
}
