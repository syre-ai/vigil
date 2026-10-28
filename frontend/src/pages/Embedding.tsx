import { useMemo, useState } from 'react';
import { EmbeddingCanvasScatter } from '../components/EmbeddingCanvasScatter';
import { InfoPopover } from '../components/InfoPopover';
import { EmbeddingScatter } from '../components/EmbeddingScatter';
import { Loading } from '../components/QueryState';
import { useBatches, useEmbeddingPoints } from '../hooks/useApi';
import { strideDownsample } from '../lib/sample';

const MAX_BATCH_POINTS = 3000;
// Above this many combined points, recharts' SVG scatter gets sluggish —
// switch to the canvas renderer instead.
const CANVAS_THRESHOLD = 2000;

export function Embedding() {
  const batchesQuery = useBatches();
  const sortedBatches = batchesQuery.data ? [...batchesQuery.data].sort((a, b) => a.order - b.order) : undefined;
  const [batchIdOverride, setBatchIdOverride] = useState<string | undefined>(undefined);
  const batchId = batchIdOverride ?? sortedBatches?.[0]?.id;
  const selectedBatch = sortedBatches?.find((b) => b.id === batchId);

  const points = useEmbeddingPoints(batchId ?? '');
  const downsampled = useMemo(() => (points.data ? strideDownsample(points.data.batch, MAX_BATCH_POINTS) : undefined), [points.data]);

  // useEmbeddingPoints keeps the previous batch's response visible
  // (placeholderData + staleTime: Infinity) while the new one loads, so the
  // reference layer — identical across batches — renders immediately on
  // every switch after the first; only the very first load has nothing to
  // show yet, when we fall back to a loading hint sized from the batch's own
  // known chip count.
  const showLoadingHint = points.isLoading && !points.data;

  return (
    <div>
      <div className="page-header">
        <div>
          <h1 className="page-title">Embedding</h1>
          <div className="page-sub">
            2-D UMAP projection, fit on the reference domain and transformed for each batch.
            <InfoPopover label="About this projection">
              UMAP fit on up to 3,000 ref_train chips; showing a 1,500-point reference sample (grey). Amber points failed the vigilance test against every known ART category.
              Hover a point for its chip thumbnail.
              {downsampled?.downsampled && (
                <>
                  {' '}
                  Batch has {points.data?.batch.length.toLocaleString()} points, showing a downsampled {downsampled.sample.length.toLocaleString()}-point subset for
                  performance.
                </>
              )}
            </InfoPopover>
          </div>
        </div>
        {sortedBatches && (
          <select value={batchId ?? ''} onChange={(e) => setBatchIdOverride(e.target.value)}>
            {sortedBatches.map((b) => (
              <option key={b.id} value={b.id}>
                {b.name}
              </option>
            ))}
          </select>
        )}
      </div>

      <div className="panel">
        {showLoadingHint ? (
          <>
            <div className="state-block" style={{ padding: '4px 0 12px' }}>
              {selectedBatch ? `Loading ${selectedBatch.n_chips.toLocaleString()} points…` : 'Loading embedding…'}
            </div>
            <Loading />
          </>
        ) : points.error ? (
          <div className="state-block state-error">Failed to load embedding.</div>
        ) : points.data ? (
          (() => {
            const batchPts = downsampled?.sample ?? points.data.batch;
            const total = points.data.reference.length + batchPts.length;
            return total > CANVAS_THRESHOLD ? (
              <EmbeddingCanvasScatter reference={points.data.reference} batch={batchPts} />
            ) : (
              <EmbeddingScatter reference={points.data.reference} batch={batchPts} />
            );
          })()
        ) : (
          <Loading label="Loading embedding…" />
        )}
      </div>
    </div>
  );
}
