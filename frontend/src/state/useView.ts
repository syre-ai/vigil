import { useContext } from 'react';
import { ViewContext, type ViewContextValue } from './viewContext';

export function useView(): ViewContextValue {
  const ctx = useContext(ViewContext);
  if (!ctx) throw new Error('useView must be used within ViewProvider');
  return ctx;
}
