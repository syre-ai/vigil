import type { CollectionStatus } from '../api/types';
import { collectionStatusLabel } from '../lib/format';

export function CollectionStatusPill({ status }: { status: CollectionStatus }) {
  return <span className={`pill pill-${status}`}>{collectionStatusLabel(status)}</span>;
}
