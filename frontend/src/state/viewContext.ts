import { createContext } from 'react';

// Analyst view (default) vs Benchmark view (the original app, unchanged) —
// see backend/API.md's field-mode contract and STATE.md's "Field mode"
// section. Persisted in localStorage so it survives a reload.
export type ViewMode = 'analyst' | 'benchmark';

export type ViewContextValue = {
  view: ViewMode;
  setView: (view: ViewMode) => void;
};

export const ViewContext = createContext<ViewContextValue | null>(null);
