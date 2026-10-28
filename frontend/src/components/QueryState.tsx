import type { ReactNode } from 'react';
import { ApiError } from '../api/client';

export function Loading({ label = 'Loading…' }: { label?: string }) {
  return (
    <div className="skeleton-block">
      <span className="sr-only">{label}</span>
      <div className="skeleton-line" style={{ width: '60%' }} />
      <div className="skeleton-line" style={{ width: '90%' }} />
      <div className="skeleton-line" style={{ width: '40%' }} />
    </div>
  );
}

export function ErrorBlock({ error }: { error: unknown }) {
  const message = error instanceof ApiError ? `${error.status}: ${error.message}` : error instanceof Error ? error.message : 'Unknown error';
  return <div className="state-block state-error">Failed to load: {message}</div>;
}

/** Renders loading/error states, or the children render-prop with the resolved data. */
export function QueryBoundary<T>({
  isLoading,
  error,
  data,
  children,
  loadingLabel,
}: {
  isLoading: boolean;
  error: unknown;
  data: T | undefined;
  children: (data: T) => ReactNode;
  loadingLabel?: string;
}) {
  if (isLoading) return <Loading label={loadingLabel} />;
  if (error) return <ErrorBlock error={error} />;
  if (data === undefined) return <Loading label={loadingLabel} />;
  return <>{children(data)}</>;
}
