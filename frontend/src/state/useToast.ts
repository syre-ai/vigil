import { useContext } from 'react';
import { ToastContext } from './toastContext';

export function useToast(): (message: string) => void {
  const push = useContext(ToastContext);
  if (!push) throw new Error('useToast must be used within a ToastProvider');
  return push;
}
