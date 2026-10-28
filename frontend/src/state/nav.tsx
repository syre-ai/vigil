import { useCallback, useMemo, useState, type ReactNode } from 'react';
import { NavContext, type Route } from './navContext';

// Default landing page depends on the persisted view (read directly from
// localStorage — the same key state/view.tsx's ViewProvider owns — so the
// initial route is right on first paint without a cross-context dependency).
function initialRoute(): Route {
  let view: 'analyst' | 'benchmark' = 'analyst';
  try {
    view = window.localStorage.getItem('vigil.view') === 'benchmark' ? 'benchmark' : 'analyst';
  } catch {
    // storage unavailable — default to analyst
  }
  return { page: view === 'analyst' ? 'reference' : 'overview' };
}

export function NavProvider({ children }: { children: ReactNode }) {
  const [route, setRoute] = useState<Route>(initialRoute);
  // The Batch page remembers its last selection across navigation (and across a reload, via sessionStorage).
  const goto = useCallback((page: Route['page'], param?: string) => {
    if (page === 'collection') {
      setRoute({ page, groupId: param });
      window.scrollTo({ top: 0 });
      return;
    }
    let remembered: string | undefined;
    try { remembered = window.sessionStorage.getItem('vigil.lastBatchId') ?? undefined; } catch { remembered = undefined; }
    const resolved = page === 'batch' ? (param ?? remembered) : param;
    if (page === 'batch' && resolved) {
      try { window.sessionStorage.setItem('vigil.lastBatchId', resolved); } catch { /* storage unavailable */ }
    }
    setRoute({ page, batchId: resolved });
    window.scrollTo({ top: 0 });
  }, []);
  const value = useMemo(() => ({ route, goto }), [route, goto]);
  return <NavContext.Provider value={value}>{children}</NavContext.Provider>;
}
