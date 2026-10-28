import { useCallback, useState, type Dispatch, type SetStateAction } from 'react';

/** useState that mirrors its value into sessionStorage under `key` (JSON); falls back silently if storage is unavailable. */
export function usePersistedState<T>(key: string, initial: T): [T, Dispatch<SetStateAction<T>>] {
  const [value, setValue] = useState<T>(() => {
    try {
      const raw = window.sessionStorage.getItem(key);
      return raw === null ? initial : (JSON.parse(raw) as T);
    } catch {
      return initial;
    }
  });
  const set = useCallback<Dispatch<SetStateAction<T>>>((next) => {
    setValue((prev) => {
      const resolved = typeof next === 'function' ? (next as (p: T) => T)(prev) : next;
      try { window.sessionStorage.setItem(key, JSON.stringify(resolved)); } catch { /* storage unavailable */ }
      return resolved;
    });
  }, [key]);
  return [value, set];
}
