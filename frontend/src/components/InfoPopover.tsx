import { useEffect, useId, useRef, useState, type ReactNode } from 'react';
import { Info } from 'lucide-react';

// Small `ⓘ` icon button that opens a popover card with methodology/caption
// text on click. Esc or a click outside closes it. Every "computed from…"
// paragraph that used to sit under a panel as a <Caption> lives in one of
// these now — see the simplification pass notes in STATE.md. Also reused
// for keyboard-shortcut hints (children can be any markup, incl. <kbd>).
export function InfoPopover({
  children,
  label = 'More info',
  align = 'left',
  width,
}: {
  children: ReactNode;
  label?: string;
  align?: 'left' | 'right';
  /** Override the default max-width (px) for popovers with denser content (e.g. a data grid). */
  width?: number;
}) {
  const [open, setOpen] = useState(false);
  const wrapRef = useRef<HTMLSpanElement | null>(null);
  const popoverId = useId();

  useEffect(() => {
    if (!open) return;
    function onDocPointer(e: PointerEvent) {
      if (wrapRef.current && !wrapRef.current.contains(e.target as Node)) setOpen(false);
    }
    function onKey(e: KeyboardEvent) {
      if (e.key === 'Escape') setOpen(false);
    }
    document.addEventListener('pointerdown', onDocPointer);
    window.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('pointerdown', onDocPointer);
      window.removeEventListener('keydown', onKey);
    };
  }, [open]);

  return (
    <span className="info-popover" ref={wrapRef}>
      <button
        type="button"
        className="info-popover-btn"
        aria-label={label}
        aria-expanded={open}
        aria-controls={popoverId}
        onClick={() => setOpen((v) => !v)}
      >
        <Info size={13} strokeWidth={2} />
      </button>
      {open && (
        <div
          id={popoverId}
          role="tooltip"
          className={`info-popover-card info-popover-card-${align}`}
          style={width ? { maxWidth: width } : undefined}
        >
          {children}
        </div>
      )}
    </span>
  );
}
