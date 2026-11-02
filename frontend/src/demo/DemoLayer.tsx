import { useCallback, useEffect, useState } from 'react';
import { useNav } from '../state/useNav';
import { useView } from '../state/useView';
import { TOUR } from './tour';

// Start screen and guided tour, shown only in the public snapshot build.
const SEEN = 'vigil.demo.seen';

function readSeen(): boolean {
  try {
    return window.sessionStorage.getItem(SEEN) === '1';
  } catch {
    return false;
  }
}

function markSeen() {
  try {
    window.sessionStorage.setItem(SEEN, '1');
  } catch {
    // storage unavailable
  }
}

function focusElement(selector: string, text?: string): () => void {
  let el: HTMLElement | null = null;
  let tries = 0;
  const timer = window.setInterval(() => {
    const all = Array.from(document.querySelectorAll<HTMLElement>(selector));
    el = (text ? all.find((e) => e.textContent?.includes(text)) : all[0]) ?? null;
    if (el || ++tries > 40) {
      window.clearInterval(timer);
      if (el) {
        el.classList.add('tour-focus');
        el.scrollIntoView({ block: 'center', behavior: 'smooth' });
      }
    }
  }, 100);
  return () => {
    window.clearInterval(timer);
    el?.classList.remove('tour-focus');
  };
}

export const TOUR_EVENT = 'vigil:tour';

export function DemoLayer() {
  const { goto, route } = useNav();
  const { view, setView } = useView();
  const [start, setStart] = useState(() => !readSeen());
  const [step, setStep] = useState<number | null>(null);

  const go = useCallback(
    (i: number) => {
      const s = TOUR[i];
      setView(s.view);
      goto(s.page, s.param);
      setStep(i);
    },
    [goto, setView],
  );

  // Leaving the tour's page or view by hand ends the tour, so its text never describes another page.
  useEffect(() => {
    if (step === null) return;
    const s = TOUR[step];
    if (route.page !== s.page || view !== s.view || (s.param && route.groupId !== s.param)) setStep(null);
  }, [route, view, step]);

  // The top bar's Tour button restarts the tour.
  useEffect(() => {
    const restart = () => {
      setStart(false);
      go(0);
    };
    window.addEventListener(TOUR_EVENT, restart);
    return () => window.removeEventListener(TOUR_EVENT, restart);
  }, [go]);

  useEffect(() => {
    if (step === null) return;
    const f = TOUR[step].focus;
    if (!f) {
      window.scrollTo({ top: 0 });
      return;
    }
    return focusElement(f.selector, f.text);
  }, [step]);

  const begin = () => {
    markSeen();
    setStart(false);
    go(0);
  };
  const explore = () => {
    markSeen();
    setStart(false);
    setView('analyst');
    goto('reference');
  };

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (step === null || (e.target as HTMLElement)?.closest('input, select, textarea')) return;
      if (e.key === 'ArrowRight' && e.altKey && step < TOUR.length - 1) go(step + 1);
      if (e.key === 'ArrowLeft' && e.altKey && step > 0) go(step - 1);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [step, go]);

  return (
    <>
      {start && (
        <div className="demo-start-backdrop">
          <div className="demo-start" role="dialog" aria-modal="true" aria-labelledby="demo-start-title">
            <h1 id="demo-start-title" className="demo-start-title">Is a deployed detector still working?</h1>
            <p>
              An aircraft detector ships with a labeled validation set, then runs on new satellite imagery with no way to
              say "I don't know." Vigil flags detections unlike anything it was validated on, lets an analyst add a new class
              without retraining, and recommends when to retrain. It can't recover aircraft the detector never found, and it
              shows where that happens.
            </p>
            <p className="demo-start-meta">
              A read-only snapshot of a working prototype on public WorldView-3 imagery (RarePlanes). Every number was
              computed by the real pipeline. Best on a desktop screen.
            </p>
            <div className="demo-start-actions">
              <button type="button" className="btn btn-primary" onClick={begin} autoFocus>
                Take the 2-minute tour
              </button>
              <button type="button" className="btn" onClick={explore}>
                Explore on my own
              </button>
            </div>
            <p className="demo-start-foot">
              Built by Shane Cairns · <a href="https://github.com/syre-ai/vigil">code on GitHub</a>
            </p>
          </div>
        </div>
      )}
      {step !== null && (
        <div className="tour-bar" role="region" aria-label="Guided tour">
          <span className="tour-step mono">
            {step + 1} / {TOUR.length}
          </span>
          <div className="tour-text">
            <strong>{TOUR[step].title}.</strong> {TOUR[step].body}
          </div>
          <div className="tour-actions">
            <button type="button" className="btn btn-sm" disabled={step === 0} onClick={() => go(step - 1)}>
              Back
            </button>
            {step < TOUR.length - 1 ? (
              <button type="button" className="btn btn-sm btn-primary" onClick={() => go(step + 1)}>
                Next
              </button>
            ) : (
              <button type="button" className="btn btn-sm btn-primary" onClick={() => setStep(null)}>
                Explore on my own
              </button>
            )}
            <button type="button" className="btn btn-sm tour-exit" onClick={() => setStep(null)} aria-label="Exit tour">
              Exit
            </button>
          </div>
        </div>
      )}
    </>
  );
}
