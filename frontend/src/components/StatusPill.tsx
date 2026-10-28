import type { EnvelopeStatus } from '../api/types';

export function StatusPill({ status }: { status: EnvelopeStatus }) {
  return <span className={`pill pill-${status}`}>{status}</span>;
}
