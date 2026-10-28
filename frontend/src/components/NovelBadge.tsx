// "NOVEL" in Benchmark view; the analyst view (det chips) says "FLAGGED" —
// the field-mode word for a chip that fails the vigilance test.
export function NovelBadge({ label = 'NOVEL' }: { label?: string }) {
  return <span className="novel-badge">{label}</span>;
}
