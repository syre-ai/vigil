import type { Batch, EvalResponse } from '../api/types';

// Client-side "story at a glance" ALERT for the Overview batch cards — this is
// explicitly not a tested-envelope determination (see EnvelopePage for the
// evidence table). Vocabulary is exactly "nominal" / "watch" / "alert":
//
//   nominal — drift score <= null p95 AND novelty rate <= 2x control
//   watch   — either of those is exceeded
//   alert   — (drift p<0.01 AND novelty rate >= 3x control), OR
//             detection recall < control recall - 0.10
//
// "control" is the ref_holdout batch (same domain as reference, never used
// for fitting/calibration). Every number is read from the API; nothing here
// is hard-coded. p-values are indicative (block-bootstrap null), never
// described as "significant".
export type EnvelopeStatusKind = 'nominal' | 'watch' | 'alert';

// Detailed rule text — used as a hover tooltip on the pill, not as the
// Overview caption (which must use ENVELOPE_STATUS_CAPTION verbatim).
export const ENVELOPE_STATUS_RULE =
  'nominal: drift score ≤ null p95 and novelty rate ≤ 2× control · watch: either exceeded · ' +
  'alert: drift p<0.01 (indicative) and novelty rate ≥ 3× control, or detector recall > 10 points below control';

export const ENVELOPE_STATUS_CAPTION =
  'Alerts are a simple client-side rule over the numbers shown (drift score vs null p95, novelty rate vs control, detector recall vs control); ' +
  'the Tested envelope page is the evidence table.';

export const ENVELOPE_STATUS_LABEL: Record<EnvelopeStatusKind, string> = {
  nominal: 'nominal',
  watch: 'watch',
  alert: 'alert',
};

export function computeEnvelopeStatus(batch: Batch, control: Batch): EnvelopeStatusKind {
  const controlNovelty = control.novelty_rate;
  const noveltyExceeds2x = controlNovelty > 0 ? batch.novelty_rate > 2 * controlNovelty : batch.novelty_rate > 0;
  const noveltyExceeds3x = controlNovelty > 0 ? batch.novelty_rate >= 3 * controlNovelty : batch.novelty_rate > 0;
  const driftExceedsP95 = batch.drift.score > batch.drift.null_p95;
  const driftPBelow01 = batch.drift.p_value < 0.01; // indicative, not a significance test

  const batchRecall = batch.detection?.recall ?? null;
  const controlRecall = control.detection?.recall ?? null;
  const detectionDegraded = batchRecall != null && controlRecall != null && batchRecall < controlRecall - 0.1;

  if ((driftPBelow01 && noveltyExceeds3x) || detectionDegraded) return 'alert';
  if (!driftExceedsP95 && !noveltyExceeds2x) return 'nominal';
  return 'watch';
}

export function noveltyRatioToControl(batch: Batch, control: Batch): number {
  if (control.novelty_rate > 0) return batch.novelty_rate / control.novelty_rate;
  return batch.novelty_rate > 0 ? Infinity : 1;
}

// A Finding pairs its plain-language sentence with the same nominal/watch/alert
// status shown on the Overview batch cards — rendered as the same colored
// pill, so the Findings panel and the card strip read as one consistent
// vocabulary — instead of spelling the status out as trailing sentence text.
export type Finding = { text: string; status?: EnvelopeStatusKind };

/** Auto-generated, data-driven "story at a glance" sentences for the Overview Findings panel. */
export function computeFindings(batches: Batch[], evalData?: EvalResponse): Finding[] {
  const control = batches.find((b) => b.id === 'ref_holdout');
  if (!control) return [];
  const others = [...batches].filter((b) => b.id !== control.id).sort((a, b) => a.order - b.order);

  const scored = others.map((b) => {
    const status = computeEnvelopeStatus(b, control);
    const ratio = noveltyRatioToControl(b, control);
    const severity = (status === 'alert' ? 2 : status === 'watch' ? 1 : 0) + Math.min(ratio, 5) / 5;
    return { b, status, severity };
  });
  scored.sort((a, b) => b.severity - a.severity);

  const findings: Finding[] = [];
  for (const { b, status } of scored) {
    if (findings.length >= 4) break;
    const bRecall = b.detection?.recall ?? null;
    const cRecall = control.detection?.recall ?? null;
    const hasDetection = bRecall != null && cRecall != null;
    // The military batch mixes civil and military chips, so its own
    // batch-level novelty_rate is diluted by the civil majority — compare
    // the disaggregated military-vs-civil flag rate within the batch instead
    // of that diluted rate against control.
    const hasMilitarySplit = b.novelty_rate_military != null && b.novelty_rate_civil != null;
    // off_nadir/foreign are geometry/acquisition-domain shifts, not scene
    // content shifts — when the client-side rule reads "nominal" for them,
    // say what the statistic actually shows instead of implying an envelope
    // determination.
    const noScopeShiftBatch = (b.id === 'off_nadir' || b.id === 'foreign') && status === 'nominal';
    let s: string;
    if (hasMilitarySplit) {
      const milRate = b.novelty_rate_military as number;
      const civRate = b.novelty_rate_civil as number;
      s = `${b.name}: military aircraft flagged novel ${(milRate * 100).toFixed(0)}% vs civil ${(civRate * 100).toFixed(0)}% within the batch (ground truth, evaluation only)`;
      if (hasDetection) s += `; detector recall ${(bRecall! * 100).toFixed(0)}% vs ${(cRecall! * 100).toFixed(0)}% control`;
    } else {
      s = `${b.name}: novelty rate ${(b.novelty_rate * 100).toFixed(0)}% vs ${(control.novelty_rate * 100).toFixed(0)}% control`;
      if (noScopeShiftBatch) {
        s += `; no scene-embedding shift detected by this statistic`;
        if (hasDetection) s += `; detector recall ${(bRecall! * 100).toFixed(0)}% vs control ${(cRecall! * 100).toFixed(0)}%`;
      } else if (hasDetection) {
        s += `; detector recall ${(bRecall! * 100).toFixed(0)}% vs ${(cRecall! * 100).toFixed(0)}% control`;
      }
    }
    findings.push({ text: s, status });
  }

  const rg = evalData?.detection?.by_role_group;
  if (rg && rg.military.recall != null && rg.civil.recall != null) {
    const scopeText = rg.scope ? ` (${rg.scope})` : '';
    findings.push({
      text:
        `Military airfields: detector recall on military aircraft ${(rg.military.recall * 100).toFixed(0)}% vs ${(rg.civil.recall * 100).toFixed(0)}% civil${scopeText} ` +
        `(ground truth, evaluation only; n=${rg.military.n} military / ${rg.civil.n} civil)`,
    });
    const pooled = evalData?.detection?.by_role_group_pooled ?? evalData?.detection?.recall_by_role_group_pooled;
    if (pooled && pooled.military.recall != null && pooled.civil.recall != null) {
      const pooledScopeText = pooled.scope ? ` (${pooled.scope})` : '';
      findings.push({
        text:
          `Pooled comparison${pooledScopeText}: detector recall on military aircraft ${(pooled.military.recall * 100).toFixed(0)}% vs ${(pooled.civil.recall * 100).toFixed(0)}% civil ` +
          `(ground truth, evaluation only; n=${pooled.military.n} military / ${pooled.civil.n} civil)`,
      });
    }
  } else if (evalData?.military) {
    const rate = evalData.military.analyst_class_assignment_rate ?? evalData.military.rare_target_recall;
    if (Number.isFinite(rate)) {
      findings.push({
        text: `Held-out military chips never used to fit the model: ${(rate! * 100).toFixed(0)}% now assigned to an analyst-created class (n=${evalData.military.n_eval}).`,
      });
    }
  }

  return findings.slice(0, 5);
}
