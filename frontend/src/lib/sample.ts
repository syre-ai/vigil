/** Deterministic stride downsample: keeps a spread-out, reproducible subset instead of
 * just the first N (which would bias toward whatever order the API returned). */
export function strideDownsample<T>(arr: T[], max: number): { sample: T[]; downsampled: boolean } {
  if (arr.length <= max) return { sample: arr, downsampled: false };
  const step = arr.length / max;
  const out: T[] = new Array(max);
  for (let i = 0; i < max; i++) out[i] = arr[Math.floor(i * step)];
  return { sample: out, downsampled: true };
}
