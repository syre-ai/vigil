import { Bar, BarChart, Cell, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import type { DriftSeriesEntry } from '../api/types';
import { score, shortBatchName } from '../lib/format';

const COLORS = { ok: '#1f5aa6', watch: '#d97706', alert: '#dc2626' };

function colorFor(entry: DriftSeriesEntry): string {
  if (entry.p_value < 0.01) return COLORS.alert;
  if (entry.p_value < 0.05) return COLORS.watch;
  return COLORS.ok;
}

function DriftTooltip({ active, payload }: { active?: boolean; payload?: { payload: DriftSeriesEntry }[] }) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div className="embedding-tooltip" style={{ flexDirection: 'column', alignItems: 'flex-start', gap: 2 }}>
      <div style={{ fontWeight: 600 }}>{d.name}</div>
      <div className="mono">score {score(d.score)}</div>
      <div className="mono">
        p-value (indicative) {score(d.p_value)} · {d.ratio_to_calib.toFixed(2)}× calibration mean
      </div>
      <div className="mono">
        null p95 {score(d.null_p95)} (n={d.n})
      </div>
    </div>
  );
}

export function DriftChart({ series }: { series: DriftSeriesEntry[] }) {
  const nullP95 = series[0]?.null_p95 ?? 0;
  // Short, non-clipping axis labels derived from batch_id; the tooltip (which
  // reads the full data row) still shows the full Batch.name.
  const data = series.map((entry) => ({ ...entry, shortName: shortBatchName(entry.batch_id) }));
  return (
    <ResponsiveContainer width="100%" height={170}>
      <BarChart data={data} margin={{ top: 6, right: 8, left: 0, bottom: 4 }}>
        <XAxis dataKey="shortName" tick={{ fill: '#8a93a1', fontSize: 11 }} axisLine={false} tickLine={false} interval={0} height={24} />
        <YAxis tick={{ fill: '#8a93a1', fontSize: 11 }} axisLine={false} tickLine={false} width={30} tickCount={4} />
        <Tooltip content={<DriftTooltip />} />
        <ReferenceLine y={nullP95} stroke={COLORS.alert} strokeDasharray="4 3" label={{ value: 'null p95', position: 'insideTopRight', fill: COLORS.alert, fontSize: 11 }} />
        <Bar dataKey="score" name="drift score" radius={[3, 3, 0, 0]}>
          {data.map((entry) => (
            <Cell key={entry.batch_id} fill={colorFor(entry)} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
