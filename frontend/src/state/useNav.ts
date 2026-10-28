import { useContext } from 'react';
import { NavContext, type NavContextValue } from './navContext';

export function useNav(): NavContextValue {
  const ctx = useContext(NavContext);
  if (!ctx) throw new Error('useNav must be used within NavProvider');
  return ctx;
}
