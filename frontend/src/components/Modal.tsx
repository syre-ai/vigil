import { useEffect, type ReactNode } from 'react';

export function Modal({ onClose, children, width = 880 }: { onClose: () => void; children: ReactNode; width?: number }) {
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      // A full-screen tile viewer stacked on top of this modal owns Escape —
      // one press closes the viewer, the next closes the modal.
      if (e.key === 'Escape' && !document.querySelector('.tile-viewer-backdrop')) onClose();
    }
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-panel" style={{ maxWidth: width }} onClick={(e) => e.stopPropagation()}>
        <button type="button" className="modal-close" onClick={onClose} aria-label="Close">
          ×
        </button>
        {children}
      </div>
    </div>
  );
}
