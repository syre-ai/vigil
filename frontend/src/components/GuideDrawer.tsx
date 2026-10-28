import { X } from 'lucide-react';
import { useEffect } from 'react';

// Kept to ~250 words total, one short paragraph per page. Vocabulary matches
// backend/API.md exactly: never "deployed"/"accredited"/"significant", and
// never "recall" for the Rule B (analyst-class assignment) metric.
const SECTIONS: { title: string; body: string }[] = [
  {
    title: 'Validation set',
    body:
      'The analyst view\'s landing page: the only labeled data anywhere in the system (the set the detector shipped with): known classes, conditions covered, and the flag rate it produces on itself. Then move on to Collections, which is entirely unlabeled imagery collected since delivery.',
  },
  {
    title: 'Analyst view',
    body:
      "The default view: only what a fielded system would have. Collections lists every new, unlabeled imagery collection since the detector was delivered, grouped by airfield and date, with how many aircraft the detector found, how many the novelty layer flagged as unlike the labeled validation set, and a needs-review / nominal / insufficient-data status from the stated rule (open the info icon next to the flag rate). Open a collection to see the detector's own chips. predicted role, confidence, novelty. and the parent tile with detector boxes only. Review (field) queues flagged detections across all collections: confirm, new class, not aircraft, correct role, bad box, or skip. Bad box marks a localization error (the detector's box covers two aircraft or cuts one). it is never used to fit any class. Apply feedback builds the next version and shows two things: how many detections across the new collections each analyst class is now assigned (descriptive. there are no labels to grade against), and the labeled control-set regression check, the only before/after accuracy, computed on the labeled validation set. Reviewed flagged detections also give a field estimate of detector precision (not aircraft) and localization error (bad box). No ground truth, drift statistic, or dataset batch name appears in this view; Benchmark view (top right) is the proof harness underneath. Model health is the decision page: keep, patch, or schedule a retrain, from four indicators the field can measure (novelty pressure, new-class demand, detector field quality, patch safety) with every rule stated. indicative.",
  },
  {
    title: 'Overview',
    body: 'Current prototype version, operational batch stream, and drift/novelty trend. Read the nominal/watch/alert pills and the Findings panel for a plain-language summary, drawn live from /summary, /drift, and /eval.',
  },
  {
    title: 'Batch',
    body: 'Condition facts, the novelty distribution against the reference domain, and chip-level evidence for one batch. Watch for a rightward shift; computed from data/scores/{version}.parquet.',
  },
  {
    title: 'Review queue',
    body: 'Analyst triage ranked by novelty, uncertainty, or random order. Confirm records agreement only. Correct role records disagreement and assigns a known class, evidence only, like Confirm, never alters the model. New class fits an analyst prototype, but only for chips already called novel. Apply feedback builds the next version.',
  },
  {
    title: 'Embedding',
    body: '2-D UMAP fit on the reference domain, transformed per batch. Grey points are reference chips; amber points failed the vigilance test. Look for amber clusters apart from the reference cloud.',
  },
  {
    title: 'Tested envelope',
    body: 'Evidence table of role accuracy, novelty rate, and detection recall by condition (weather, off-nadir, country, GSD, size), judged against the control batch. Status is tested/untested/degraded, never inferred from a label.',
  },
  {
    title: 'Evaluation',
    body: 'Ground truth for evaluation only, never fitting or calibration: novelty AUROC, role-head accuracy, the analyst-class assignment rate on held-out military chips, and detector recall/precision.',
  },
  {
    title: 'Map',
    body: 'Parent images by centroid of their aircraft annotations, colored by domain, sized by chip count. Hover or click a point for novelty rate and drift score.',
  },
  {
    title: 'Report',
    body: 'The generated assurance-evidence markdown report for the current version: method, limitations, and detector performance, ready to copy.',
  },
];

export function GuideDrawer({ open, onClose }: { open: boolean; onClose: () => void }) {
  useEffect(() => {
    if (!open) return;
    function onKey(e: KeyboardEvent) {
      if (e.key === 'Escape') onClose();
    }
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open, onClose]);

  if (!open) return null;
  return (
    <div className="guide-backdrop" onClick={onClose}>
      <aside className="guide-drawer" onClick={(e) => e.stopPropagation()}>
        <div className="guide-drawer-head">
          <div className="panel-title" style={{ margin: 0 }}>
            How to read this
          </div>
          <button type="button" className="modal-close" onClick={onClose} aria-label="Close">
            <X size={18} />
          </button>
        </div>
        <div className="guide-drawer-body">
          {SECTIONS.map((s) => (
            <div className="guide-section" key={s.title}>
              <div className="guide-section-title">{s.title}</div>
              <p className="guide-section-text">{s.body}</p>
            </div>
          ))}
          <div className="guide-section">
            <div className="guide-section-title">Keyboard shortcuts</div>
            <p className="guide-section-text">
              Review queue: <kbd>1</kbd> confirm role · <kbd>2</kbd> new class · <kbd>3</kbd> not aircraft · <kbd>4</kbd> correct role · <kbd>5</kbd> bad box · <kbd>s</kbd> skip.{' '}
              <kbd>Esc</kbd> closes any open drawer or modal.
            </p>
          </div>
        </div>
      </aside>
    </div>
  );
}
