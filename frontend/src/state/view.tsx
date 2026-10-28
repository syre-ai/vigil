import { useCallback, useMemo, useState, type ReactNode } from 'react';
import { ViewContext, type ViewMode } from './viewContext';

const KEY = 'vigil.view';

function readView(): ViewMode {
  try {
    return window.localStorage.getItem(KEY) === 'benchmark' ? 'benchmark' : 'analyst';
  } catch {
    return 'analyst';
  }
}

export function ViewProvider({ children }: { children: ReactNode }) {
  const [view, setViewState] = useState<ViewMode>(readView);
  const setView = useCallback((next: ViewMode) => {
    setViewState(next);
    try {
      window.localStorage.setItem(KEY, next);
    } catch {
      // storage unavailable (private mode, quota, etc.)
    }
  }, []);
  const value = useMemo(() => ({ view, setView }), [view, setView]);
  return <ViewContext.Provider value={value}>{children}</ViewContext.Provider>;
}
