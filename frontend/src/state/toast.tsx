// Minimal, dependency-free toast stack. Mounted once at the app root;
// `useToast()` (see useToast.ts) returns a `push(message)` function any
// component can call — used by ReviewActions so every successful review
// gets a visible confirmation, regardless of which page/drawer it was
// submitted from.
import { useCallback, useRef, useState, type ReactNode } from 'react';
import { ToastContext } from './toastContext';

type ToastItem = { id: number; message: string };

const DURATION_MS = 4000;

export function ToastProvider({ children }: { children: ReactNode }) {
  const [items, setItems] = useState<ToastItem[]>([]);
  const idRef = useRef(0);

  const push = useCallback((message: string) => {
    const id = ++idRef.current;
    setItems((prev) => [...prev, { id, message }]);
    window.setTimeout(() => {
      setItems((prev) => prev.filter((t) => t.id !== id));
    }, DURATION_MS);
  }, []);

  return (
    <ToastContext.Provider value={push}>
      {children}
      <div className="toast-stack" aria-live="polite">
        {items.map((t) => (
          <div key={t.id} className="toast">
            {t.message}
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  );
}
