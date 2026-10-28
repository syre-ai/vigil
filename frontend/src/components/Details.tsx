import { useState, type ReactNode } from 'react';
import { ChevronDown, ChevronRight } from 'lucide-react';

// Generic "Details" disclosure: expands a full table/grid inline. Open state
// is remembered per panel in sessionStorage (keyed by `id`) so it survives
// re-renders and navigating away and back within the tab, but resets on a
// fresh session — matching the localStorage idiom used by state/labelsCache.ts
// for cross-session memory, one tier down.
const STORAGE_PREFIX = 'vigil.details.';

function readOpen(id: string, defaultOpen: boolean): boolean {
  try {
    const raw = sessionStorage.getItem(STORAGE_PREFIX + id);
    return raw == null ? defaultOpen : raw === '1';
  } catch {
    return defaultOpen;
  }
}

function writeOpen(id: string, open: boolean) {
  try {
    sessionStorage.setItem(STORAGE_PREFIX + id, open ? '1' : '0');
  } catch {
    // ignore storage failures (private mode, quota, etc.)
  }
}

export function Details({
  id,
  label = 'Details',
  closeLabel,
  defaultOpen = false,
  children,
}: {
  id: string;
  label?: string;
  closeLabel?: string;
  defaultOpen?: boolean;
  children: ReactNode;
}) {
  const [open, setOpen] = useState(() => readOpen(id, defaultOpen));

  function toggle() {
    setOpen((prev) => {
      const next = !prev;
      writeOpen(id, next);
      return next;
    });
  }

  return (
    <div className="details-block">
      <button type="button" className="details-toggle" onClick={toggle} aria-expanded={open}>
        {open ? <ChevronDown size={12} strokeWidth={2.25} /> : <ChevronRight size={12} strokeWidth={2.25} />}
        {open ? (closeLabel ?? label) : label}
      </button>
      {open && <div className="details-content">{children}</div>}
    </div>
  );
}
