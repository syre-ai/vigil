import { Caption } from '../components/Caption';
import { Details } from '../components/Details';
import { QueryBoundary } from '../components/QueryState';
import type { WingspanBin } from '../api/types';
import { useBatches, useEval, useSummary } from '../hooks/useApi';
import { NA, pct, score as fmt } from '../lib/format';

function binRangeLabel(bin: WingspanBin): string {
  const [lo, hi] = bin.range;
  if (lo == null && hi != null) return `≤${hi.toFixed(1)} m`;
  if (lo != null && hi == null) return `>${lo.toFixed(1)} m`;
  if (lo != null && hi != null) return `${lo.toFixed(1)}–${hi.toFixed(1)} m`;
  return NA;
}

export function Evaluation() {
  const evalQuery = useEval();
  const batchesQuery = useBatches();
  const summaryQuery = useSummary();
  const currentVersion = summaryQuery.data?.versions.find((v) => v.id === summaryQuery.data.current_version);
  const noAnalystClasses = (currentVersion?.n_new_class ?? 0) === 0;
  const batchNames: Record<string, string> = {};
  const batchOrder: Record<string, number> = {};
  const batchNDet: Record<string, number | null | undefined> = {};
  for (const b of batchesQuery.data ?? []) {
    batchNames[b.id] = b.name;
    batchOrder[b.id] = b.order;
    batchNDet[b.id] = b.detection?.n_det;
  }
  return (
    <div>
      <div className="page-header">
        <div>
          <h1 className="page-title">Evaluation</h1>
          <div className="page-sub">Ground truth used for evaluation only, never for fitting or calibration.</div>
        </div>
      </div>

      <QueryBoundary isLoading={evalQuery.isLoading} error={evalQuery.error} data={evalQuery.data} loadingLabel="Loading evaluation…">
        {(data) => (
          <>
            {/* ---------- Novelty section: headline = AUROC + flag rates ---------- */}
            <div className="panel">
              <div className="panel-title">Novelty detection</div>
              <div className="stat-row mono">
                <div>
                  <div className="stat-value">{fmt(data.novelty.auroc_military_vs_civil)}</div>
                  <div className="stat-label">AUROC, military vs civil (ART)</div>
                </div>
                <div>
                  <div className="stat-value">{fmt(data.novelty.baseline_knn_auroc)}</div>
                  <div className="stat-label">AUROC, kNN baseline</div>
                </div>
                <div>
                  <div className="stat-value">{fmt(data.novelty.auprc)}</div>
                  <div className="stat-label">AUPRC</div>
                </div>
                <div>
                  <div className="stat-value">{fmt(data.novelty.auroc_wingspan_stratified)}</div>
                  <div className="stat-label">AUROC, wingspan-stratified</div>
                </div>
                {data.novelty.novelty_flag_rate && (data.novelty.novelty_flag_rate.military != null || data.novelty.novelty_flag_rate.civil != null) && (
                  <>
                    <div>
                      <div className="stat-value">{pct(data.novelty.novelty_flag_rate.military)}</div>
                      <div className="stat-label">
                        flag rate, military{data.novelty.novelty_flag_rate.n_military != null ? ` (n=${data.novelty.novelty_flag_rate.n_military})` : ''}
                      </div>
                    </div>
                    <div>
                      <div className="stat-value">{pct(data.novelty.novelty_flag_rate.civil)}</div>
                      <div className="stat-label">
                        flag rate, civil{data.novelty.novelty_flag_rate.n_civil != null ? ` (n=${data.novelty.novelty_flag_rate.n_civil})` : ''}
                      </div>
                    </div>
                  </>
                )}
              </div>

              <Details id="eval-novelty-details" label="Per-role AUROC, wingspan bins & adaptation">
                <Caption>{data.novelty.wingspan_stratified_method}</Caption>
                {data.novelty.size_matched_bins && data.novelty.size_matched_bins.length > 0 ? (
                  <>
                    <table className="data-table" style={{ marginTop: 8 }}>
                      <thead>
                        <tr>
                          <th>wingspan bin</th>
                          <th>n</th>
                          <th>AUROC</th>
                        </tr>
                      </thead>
                      <tbody>
                        {data.novelty.size_matched_bins.map((bin) => (
                          <tr key={bin.bin}>
                            <td>{binRangeLabel(bin)}</td>
                            <td className="mono">{bin.n.toLocaleString()}</td>
                            <td className="mono">{fmt(bin.auroc)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                    <Caption>Per-quartile-bin AUROC backing the wingspan-stratified figure above (quartile edges from civil chips).</Caption>
                  </>
                ) : (
                  <Caption>Per-bin wingspan breakdown not yet exposed by /eval.</Caption>
                )}
                {data.novelty.novelty_flag_rate && (data.novelty.novelty_flag_rate.military != null || data.novelty.novelty_flag_rate.civil != null) && (
                  <Caption>Fraction of chips flagged novel at the calibrated vigilance threshold, by true role group (ground truth, evaluation only).</Caption>
                )}

                <div className="section-gap-sm" />
                <div className="panel-title" style={{ fontSize: 13 }}>
                  Novelty AUROC per role
                </div>
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>role</th>
                      <th>n</th>
                      <th>AUROC</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.novelty.per_role.map((row) => (
                      <tr key={row.role}>
                        <td>{row.role}</td>
                        <td className="mono">{row.n}</td>
                        <td className="mono">{fmt(row.auroc)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                <Caption>
                  For a military role, AUROC of the novelty score separating that role from the known civil population. For a civil role, AUROC separating it from the other
                  civil roles. This should sit near 0.5, since novelty is not meant to discriminate between classes it already knows.
                </Caption>

                <div className="section-gap-sm" />
                <div className="panel-title" style={{ fontSize: 13 }}>
                  Analyst-class assignment rate (held-out military chips)
                </div>
                {noAnalystClasses && (data.military.analyst_class_assignment_rate ?? data.military.rare_target_recall) === 0 ? (
                  <div className="state-block" style={{ padding: '10px 0' }}>
                    no analyst classes yet: use the Review queue
                  </div>
                ) : (
                  <>
                    <div className="stat-row mono">
                      <div>
                        <div className="stat-value">{fmt(data.military.analyst_class_assignment_rate ?? data.military.rare_target_recall)}</div>
                        <div className="stat-label">analyst-class assignment rate (held-out military chips)</div>
                      </div>
                      {data.military.civil_false_assignment_rate != null && (
                        <div>
                          <div className="stat-value" style={{ color: 'var(--accent-amber)' }}>
                            {fmt(data.military.civil_false_assignment_rate)}
                          </div>
                          <div className="stat-label">
                            civil chips wrongly assigned{data.military.n_eval_civil != null ? ` (n_eval_civil=${data.military.n_eval_civil})` : ''}
                          </div>
                        </div>
                      )}
                      <div>
                        <div className="stat-value" style={{ color: 'var(--accent-red)' }}>
                          {fmt(data.military.miss_rate_civil)}
                        </div>
                        <div className="stat-label">miss rate · still predicted civil</div>
                      </div>
                      <div>
                        <div className="stat-value">{data.military.n_eval}</div>
                        <div className="stat-label">n evaluated (excludes fit exemplars)</div>
                      </div>
                    </div>
                    <Caption>
                      Among held-out military chips never used to fit a prototype, the fraction assigned to some analyst-created class at all (label-agnostic), never
                      &quot;recall&quot;: analyst classes are reachable only through the Fuzzy-ART vigilance gate below. rare_target_recall/recall_new_class are exact aliases of
                      the same number. This is the headline &quot;did adaptation actually generalize&quot; metric.
                    </Caption>
                    {data.military.exclusion_method && <Caption>{data.military.exclusion_method}</Caption>}
                    {data.military.adaptation_rule && <Caption>{data.military.adaptation_rule}</Caption>}
                  </>
                )}
              </Details>
            </div>

            <div className="section-gap" />
            {/* ---------- Role head section: headline = accuracy ---------- */}
            <div className="panel">
              <div className="panel-title">Role head</div>
              <div className="stat-row mono">
                <div>
                  <div className="stat-value">{fmt(data.role_head.accuracy_ref_holdout)}</div>
                  <div className="stat-label">accuracy, control batch (ref_holdout)</div>
                </div>
              </div>

              <Details id="eval-role-head-details" label="Per-class accuracy & analyst feedback">
                <table className="data-table" style={{ marginTop: 8 }}>
                  <thead>
                    <tr>
                      <th>class</th>
                      <th>n</th>
                      <th>accuracy</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.role_head.per_class.map((row) => (
                      <tr key={row.cls}>
                        <td>{row.cls}</td>
                        <td className="mono">{row.n}</td>
                        <td className="mono">{row.accuracy != null ? `${(row.accuracy * 100).toFixed(1)}%` : NA}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                <Caption>Nearest-prototype role head accuracy on ref_holdout, the domain the model was fit on but this specific batch was never used for fitting or calibration.</Caption>

                <div className="section-gap-sm" />
                <div className="panel-title" style={{ fontSize: 13 }}>
                  Analyst feedback
                </div>
                {data.analyst_disagreement && (data.analyst_disagreement.n_confirm > 0 || data.analyst_disagreement.n_correct > 0) ? (
                  <>
                    <div className="stat-row mono">
                      <div>
                        <div className="stat-value">{data.analyst_disagreement.n_confirm}</div>
                        <div className="stat-label">confirmed</div>
                      </div>
                      <div>
                        <div className="stat-value">{data.analyst_disagreement.n_correct}</div>
                        <div className="stat-label">corrected</div>
                      </div>
                      <div>
                        <div className="stat-value">{pct(data.analyst_disagreement.disagreement_rate)}</div>
                        <div className="stat-label">disagreement rate</div>
                      </div>
                      <div>
                        <div className="stat-value">{data.analyst_disagreement.n_reviewed}</div>
                        <div className="stat-label">total reviewed</div>
                      </div>
                    </div>
                    {data.analyst_disagreement.by_pair.length > 0 && (
                      <table className="data-table" style={{ marginTop: 8 }}>
                        <thead>
                          <tr>
                            <th>predicted role</th>
                            <th>corrected to</th>
                            <th>n</th>
                          </tr>
                        </thead>
                        <tbody>
                          {data.analyst_disagreement.by_pair.map((p) => (
                            <tr key={`${p.from_role}->${p.to_role}`}>
                              <td>{p.from_role}</td>
                              <td>{p.to_role}</td>
                              <td className="mono">{p.n}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    )}
                    <Caption>
                      Confirm and Correct role both mark a chip reviewed and record the analyst&apos;s agreement or disagreement with the predicted role as evidence. Neither
                      alters the model. Disagreement rate = corrected / (confirmed + corrected).
                    </Caption>
                  </>
                ) : (
                  <div className="state-block" style={{ padding: '10px 0' }}>
                    no confirms or corrections recorded yet
                  </div>
                )}
              </Details>
            </div>

            <div className="section-gap" />
            {/* ---------- Detector section: headline = mAP + civil-vs-military recall ---------- */}
            <div className="panel">
              <div className="panel-title">Detector</div>
              {data.detection ? (
                <>
                  <div className="stat-row mono">
                    <div>
                      <div className="stat-value">{fmt(data.detection.val_map50)}</div>
                      <div className="stat-label">val mAP50</div>
                    </div>
                    <div>
                      <div className="stat-value">{fmt(data.detection.val_map50_95)}</div>
                      <div className="stat-label">val mAP50-95</div>
                    </div>
                  </div>

                  {data.detection.by_role_group && (
                    <div className="civil-military-compare">
                      <div className="cmc-bar">
                        <div className="cmc-bar-label">
                          <span>civil recall (GT, eval only)</span>
                          <span className="mono">
                            {pct(data.detection.by_role_group.civil.recall)} (n={data.detection.by_role_group.civil.n})
                          </span>
                        </div>
                        <div className="cmc-bar-track">
                          <div
                            className="cmc-bar-fill"
                            style={{ width: `${((data.detection.by_role_group.civil.recall ?? 0) * 100).toFixed(1)}%`, background: 'var(--accent-blue)' }}
                          />
                        </div>
                      </div>
                      <div className="cmc-bar">
                        <div className="cmc-bar-label">
                          <span>military recall (GT, eval only)</span>
                          <span className="mono">
                            {pct(data.detection.by_role_group.military.recall)} (n={data.detection.by_role_group.military.n})
                          </span>
                        </div>
                        <div className="cmc-bar-track">
                          <div
                            className="cmc-bar-fill"
                            style={{ width: `${((data.detection.by_role_group.military.recall ?? 0) * 100).toFixed(1)}%`, background: 'var(--accent-red)' }}
                          />
                        </div>
                      </div>
                    </div>
                  )}
                  <Caption>
                    Detector recall/precision measured against the ground-truth aircraft boxes (IoU ≥ {data.detection.iou_threshold ?? 0.5}), per operational batch. The
                    civil-vs-military comparison above is evaluation-only. It never feeds back into detection or novelty scoring.
                    {data.detection.by_role_group?.scope ? ` Scope: ${data.detection.by_role_group.scope}.` : ''}
                  </Caption>

                  {data.detection.by_batch && Object.keys(data.detection.by_batch).length > 0 && (
                    <Details id="eval-detector-details" label="Per-batch recall & precision">
                      <table className="data-table">
                        <thead>
                          <tr>
                            <th>batch</th>
                            <th>recall</th>
                            <th>precision</th>
                            <th>n_gt</th>
                            <th>n_det</th>
                          </tr>
                        </thead>
                        <tbody>
                          {Object.entries(data.detection.by_batch)
                            .sort(([a], [b]) => (batchOrder[a] ?? 99) - (batchOrder[b] ?? 99))
                            .map(([batchId, row]) => {
                              const nDet = row.n_det ?? batchNDet[batchId];
                              return (
                                <tr key={batchId}>
                                  <td>{batchNames[batchId] ?? batchId}</td>
                                  <td className="mono">{pct(row.recall)}</td>
                                  <td className="mono">{pct(row.precision)}</td>
                                  <td className="mono">{row.n_gt.toLocaleString()}</td>
                                  <td className="mono">{nDet != null ? nDet.toLocaleString() : NA}</td>
                                </tr>
                              );
                            })}
                        </tbody>
                      </table>
                    </Details>
                  )}
                </>
              ) : (
                <div className="state-block">Detector evaluation not yet available. This block will populate once the detector milestone lands.</div>
              )}
            </div>

            <div className="section-gap" />
            <div className="panel" style={{ borderColor: 'var(--accent-blue)' }}>
              <span className="mono" style={{ color: 'var(--accent-blue)' }}>
                {data.note}
              </span>
            </div>
          </>
        )}
      </QueryBoundary>
    </div>
  );
}
