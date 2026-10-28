import type { ModelInfo, Version } from '../api/types';
import { Details } from './Details';
import { InfoPopover } from './InfoPopover';
import { NA } from '../lib/format';
import { shortDetectorName, shortEmbedderName } from '../lib/modelCard';

function fmt3(v: number | null | undefined): string {
  return v == null ? NA : v.toFixed(3);
}

function fmtNum(v: unknown): string {
  return typeof v === 'number' ? (Number.isInteger(v) ? String(v) : v.toFixed(3)) : String(v);
}

// calibration_check has been observed as either a plain string or a structured
// split-stability summary (e.g. {min, median, max, n_parents}); render whatever
// shape arrives rather than assuming one.
function renderCalibrationCheck(v: ModelInfo['calibration_check']): string | null {
  if (v == null) return null;
  if (typeof v === 'string') return v;
  const { min, median, max, n_parents, ...rest } = v;
  const parts: string[] = [];
  if (min !== undefined) parts.push(`min ${fmtNum(min)}`);
  if (median !== undefined) parts.push(`median ${fmtNum(median)}`);
  if (max !== undefined) parts.push(`max ${fmtNum(max)}`);
  if (!parts.length) {
    const restEntries = Object.entries(rest);
    if (!restEntries.length) return null;
    return restEntries.map(([k, val]) => `${k}: ${fmtNum(val)}`).join(', ');
  }
  return n_parents !== undefined ? `${parts.join(', ')} (n=${fmtNum(n_parents)} parent-image folds)` : parts.join(', ');
}

// The backend's model.novelty_method / model.detector strings are full,
// multi-clause sentences (fine for the report, too long for a one-line
// summary). Short, data-driven forms — derived from the model-card fields
// themselves (never hard-coded) so these stay correct if the embedder, PCA
// dimension, detector, or cross-fitting setup changes.

function shortNoveltyLabel(model: ModelInfo): string {
  return /fuzzy art/i.test(model.novelty_method) ? 'Fuzzy ART novelty' : model.novelty_method.split(' ').slice(0, 3).join(' ');
}

function shortNoveltyMethod(model: ModelInfo): string {
  const pcaMatch = model.novelty_method.match(/PCA-(\d+)/);
  if (!pcaMatch) return model.novelty_method;
  const embedderName = model.embedder.split(',')[0]?.trim() || model.embedder;
  const crossFitted = model.calibrated_on.toLowerCase().includes('cross-fit');
  return `Fuzzy ART on ${embedderName} chip embeddings (PCA-${pcaMatch[1]}), vigilance ${fmt3(model.vigilance)}${crossFitted ? ', cross-fitted' : ''}`;
}


function shortDetector(model: ModelInfo): string {
  const dc = model.detector_card;
  if (!dc) return model.detector;
  const shortName = dc.name.split(' ')[0] || dc.name;
  const trainedOnClause = (dc.trained_on || '').split('(')[0].trim();
  if (!trainedOnClause) return model.detector;
  return `${shortName}, trained on ${trainedOnClause} · val mAP50 ${fmt3(dc.val_map50)}`;
}

// The single line shown by default, e.g. "YOLO11n detector · DINOv2 + Fuzzy
// ART novelty (vigilance 0.588) · reference 3,942 chips / 43 images · v1".
// Every clause is read off `model`/`currentVersion` — nothing is hard-coded.
function oneLineSummary(model: ModelInfo, currentVersion: string): string {
  const vigilanceText = model.vigilance != null ? ` (vigilance ${fmt3(model.vigilance)})` : '';
  return (
    `${shortDetectorName(model)} detector · ${shortEmbedderName(model.embedder)} + ${shortNoveltyLabel(model)}${vigilanceText} · ` +
    `reference ${model.n_ref_chips.toLocaleString()} chips / ${model.n_ref_images.toLocaleString()} images · ${currentVersion}`
  );
}

export function ModelCardStrip({ model, currentVersion, versions }: { model: ModelInfo; currentVersion: string; versions: Version[] }) {
  const v = versions.find((x) => x.id === currentVersion);
  const dc = model.detector_card;
  const items: { label: string; value: string }[] = [
    { label: 'detector', value: shortDetector(model) },
    { label: 'embedder', value: model.embedder },
    { label: 'novelty method', value: shortNoveltyMethod(model) },
    { label: 'reference domain', value: model.reference_domain },
    { label: 'reference set', value: `${model.n_ref_chips.toLocaleString()} chips / ${model.n_ref_images.toLocaleString()} images` },
    { label: 'version', value: `${currentVersion}${v ? ` · ${v.classes.length} classes` : ''}` },
  ];
  if (dc) {
    items.push({ label: 'detector val mAP', value: `${fmt3(dc.val_map50)} / ${fmt3(dc.val_map50_95)} (50 / 50-95) on ${dc.val_split}` });
  }
  const calibrationCheckText = renderCalibrationCheck(model.calibration_check);

  return (
    <div className="model-card-line">
      <span className="model-card-summary">{oneLineSummary(model, currentVersion)}</span>
      <InfoPopover label="Full model card" width={520} align="right">
        <div className="model-card-grid">
          {items.map((item) => (
            <div className="model-card-item" key={item.label}>
              <div className="model-card-label">{item.label}</div>
              <div className="model-card-value">{item.value}</div>
            </div>
          ))}
        </div>
        <p className="info-popover-text">
          Calibrated on {model.calibrated_on}. Localization: {model.localization}
          {dc ? (
            <>
              {' '}
              Detector: {dc.name}, trained on {dc.trained_on}. {dc.note}
            </>
          ) : null}
        </p>
      </InfoPopover>
      <Details id="overview-model-card-notes" label="Calibration & model-selection notes">
        <div className="model-card-note-row">
          <span className="model-card-label">novelty method (full)</span>
          {model.novelty_method}
        </div>
        <div className="model-card-note-row">
          <span className="model-card-label">detector (full)</span>
          {model.detector}
        </div>
        {calibrationCheckText && (
          <div className="model-card-note-row">
            <span className="model-card-label">calibration check</span>
            {calibrationCheckText}
          </div>
        )}
        {model.model_selection_note && (
          <div className="model-card-note-row">
            <span className="model-card-label">model selection</span>
            {model.model_selection_note}
          </div>
        )}
        {model.adaptation_rule && (
          <div className="model-card-note-row">
            <span className="model-card-label">adaptation rule</span>
            {model.adaptation_rule}
          </div>
        )}
      </Details>
    </div>
  );
}
