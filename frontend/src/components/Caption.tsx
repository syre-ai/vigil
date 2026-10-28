import type { ReactNode } from 'react';

/** Small "what this number is computed from" caption shown under every panel. */
export function Caption({ children }: { children: ReactNode }) {
  return <p className="caption">{children}</p>;
}
