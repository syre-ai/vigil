import { useMemo, useState, type ReactNode } from 'react';
import { useQueries } from '@tanstack/react-query';
import { Copy, Download } from 'lucide-react';
import { api } from '../api/client';
import type { Batch, CollectionGroup, EvalResponse, HealthResponse, ModelInfo, Review, ReviewAction, Summary, Version, VersionsCompare } from '../api/types';
import { InfoPopover } from '../components/InfoPopover';
import { IndicatorTile } from '../components/ModelHealthPanel';
import { QueryBoundary } from '../components/QueryState';
import { qk, useBatches, useChangesSummary, useCollections, useCollectionsSummary, useEval, useHealth, useReviewLog, useSummary } from '../hooks/useApi';
import { normalizeFieldCompare, type FieldCompareVM } from '../lib/fieldCompare';
import { NA, RECOMMENDATION_LABEL, REVIEW_ACTION_LABELS, count, pValue, pct, ppDelta, score, shortBatchName, shortRole } from '../lib/format';
import { shortDetectorName, shortEmbedderName } from '../lib/modelCard';
import { useToast } from '../state/useToast';
import { useView } from '../state/useView';

// One-page, data-driven assurance summary. Every value is read off an existing endpoint
// (/summary, /health, /collections, /collections/summary, /review/log, /versions/compare,
// /versions/changes/summary, and in Benchmark view /eval + /batches). No prose beyond one-line
// labels; the full markdown report (/api/report) stays available as a file export.

const ACTION_ORDER: ReviewAction[] = ['new_class', 'confirm', 'correct', 'not_aircraft', 'bad_box', 'skip'];

// Fixed limitation bullets (≤ 12 words each) — the only static text on the page.
const LIMITATIONS = [
  'Single sensor: WorldView-3 only (public RarePlanes).',
  'Small military sample; military-vs-civil is acquisition-domain confounded.',
  'Role/novelty scores use ground-truth boxes in Benchmark view only.',
  'Generic embedding (DINOv2-small), not tuned for overhead imagery.',
  'Drift and collection p-values are indicative, not inferential.',
];

function generatedStamp(): string {
  return `${new Date().toISOString().slice(0, 16).replace('T', ' ')} UTC`;
}

function shortNoveltyMethod(model: ModelInfo): string {
  const pca = model.novelty_method.match(/PCA-(\d+)/);
  const art = /fuzzy art/i.test(model.novelty_method) ? 'Fuzzy ART' : model.novelty_method.split(' ').slice(0, 2).join(' ');
  return pca ? `${art} · PCA-${pca[1]}` : art;
}

function pctOf(n: number, d: number): string {
  return d > 0 ? pct(n / d) : NA;
}

function byPAsc(a: CollectionGroup, b: CollectionGroup): number {
  const pa = a.p_value ?? Infinity;
  const pb = b.p_value ?? Infinity;
  return pa - pb || b.n_flagged - a.n_flagged;
}

// ---------- small layout primitives ----------

function Section({ title, info, children, className }: { title: string; info?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <section className={`panel report-section${className ? ` ${className}` : ''}`}>
      <div className="report-section-title">
        {title}
        {info}
      </div>
      {children}
    </section>
  );
}

function Tile({ label, value, sub, info }: { label: string; value: ReactNode; sub?: ReactNode; info?: ReactNode }) {
  return (
    <div className="report-tile">
      <div className="report-tile-label">
        {label}
        {info}
      </div>
      <div className="report-tile-value mono">{value}</div>
      {sub != null && <div className="report-tile-sub">{sub}</div>}
    </div>
  );
}

function Rule({ label, text }: { label: string; text: string }) {
  return (
    <InfoPopover label={label} width={380}>
      <p className="info-popover-text" style={{ margin: 0 }}>
        {text}
      </p>
    </InfoPopover>
  );
}

// ---------- header ----------

function ReportActions() {
  const toast = useToast();
  const [busy, setBusy] = useState<'download' | 'copy' | null>(null);

  async function withReport(kind: 'download' | 'copy', fn: (md: string) => Promise<void> | void) {
    setBusy(kind);
    try {
      await fn(await api.report());
    } catch (e) {
      toast(`Report ${kind} failed: ${e instanceof Error ? e.message : 'unknown error'}`);
    } finally {
      setBusy(null);
    }
  }

  function download() {
    return withReport('download', (md) => {
      const blob = new Blob([md], { type: 'text/markdown;charset=utf-8' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `vigil-report-${new Date().toISOString().slice(0, 10)}.md`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    });
  }

  function copy() {
    return withReport('copy', async (md) => {
      await navigator.clipboard.writeText(md);
      toast('Report markdown copied');
    });
  }

  return (
    <div className="report-actions">
      <button type="button" className="btn btn-sm" onClick={download} disabled={busy != null} title="Full assurance evidence report as markdown (/api/report)">
        <Download size={13} strokeWidth={1.75} />
        {busy === 'download' ? 'Preparing…' : 'Download .md'}
      </button>
      <button type="button" className="btn btn-sm" onClick={copy} disabled={busy != null} title="Copy the full markdown report to the clipboard">
        <Copy size={13} strokeWidth={1.75} />
        {busy === 'copy' ? 'Copying…' : 'Copy'}
      </button>
    </div>
  );
}

function Header({ summary, health, stamp }: { summary: Summary; health: HealthResponse | undefined; stamp: string }) {
  return (
    <div className="report-header">
      <div className="report-header-line">
        <span className="report-header-title">Assurance evidence</span>
        <span className="report-header-sep">·</span>
        <span>{shortDetectorName(summary.model)}</span>
        <span className="report-header-sep">·</span>
        <span>
          version <span className="mono">{summary.current_version}</span>
        </span>
        <span className="report-header-sep">·</span>
        <span className="cell-dim">
          generated <span className="mono">{stamp}</span>
        </span>
        {health && (
          <span className={`pill pill-${health.recommendation}`} title={`${RECOMMENDATION_LABEL[health.recommendation]} · indicative (${health.note})`}>
            {RECOMMENDATION_LABEL[health.recommendation]}
          </span>
        )}
      </div>
      <ReportActions />
    </div>
  );
}

// ---------- 1. model ----------

function ModelSection({ summary, referenceFlagRate }: { summary: Summary; referenceFlagRate: number | null | undefined }) {
  const m = summary.model;
  const dc = m.detector_card;
  return (
    <Section title="Model">
      <div className="report-tiles">
        <Tile label="detector" value={shortDetectorName(m)} sub={dc ? `val mAP50 ${score(dc.val_map50)} on ${dc.val_split.split(' ')[0]}` : `val mAP50 ${NA}`} />
        <Tile label="embedder" value={shortEmbedderName(m.embedder)} sub={m.embedder.split(',')[0]} />
        <Tile label="novelty method" value={shortNoveltyMethod(m)} sub="fails vigilance vs every category = flagged" />
        <Tile label="vigilance" value={score(m.vigilance)} sub={m.calibrated_on.split(';')[0].split('(')[0].trim()} />
        <Tile label="reference set" value={`${count(m.n_ref_chips)} chips`} sub={`${count(m.n_ref_images)} images · ${m.reference_domain}`} />
        <Tile
          label="validation-set flag rate"
          value={pct(referenceFlagRate)}
          sub="flagged share on the labeled validation set"
          info={<Rule label="About the validation-set flag rate" text="Fraction of detector chips on the labeled validation set (reference domain) that the novelty layer flags. The baseline every new collection is compared against." />}
        />
      </div>
    </Section>
  );
}

// ---------- 2. since delivery ----------

function SinceDeliverySection({ groups, statusRule }: { groups: CollectionGroup[]; statusRule: string }) {
  const summary = useCollectionsSummary();
  const s = summary.data;
  const nGroups = s?.n_groups ?? groups.length;
  const nNeeds = s?.n_needs_review ?? groups.filter((g) => g.status === 'needs_review').length;
  const nDet = s?.n_detections ?? groups.reduce((acc, g) => acc + g.n_detections, 0);
  const nFlag = s?.n_flagged ?? groups.reduce((acc, g) => acc + g.n_flagged, 0);
  const needsReview = groups.filter((g) => g.status === 'needs_review').sort(byPAsc);
  const showP = needsReview.some((g) => g.p_value != null);

  return (
    <Section title="Since delivery" info={<Rule label="Status rule" text={`Status rule (from the API, verbatim): ${statusRule}`} />}>
      <div className="report-tiles report-tiles-4">
        <Tile label="collections" value={count(nGroups)} sub="airfield × date groups" />
        <Tile label="need review" value={count(nNeeds)} sub={`${pctOf(nNeeds, nGroups)} of collections`} />
        <Tile label="detections" value={count(nDet)} sub="detector boxes, conf ≥ 0.25" />
        <Tile label="flagged" value={count(nFlag)} sub={`${pctOf(nFlag, nDet)} of detections`} />
      </div>
      {needsReview.length === 0 ? (
        <div className="caption" style={{ marginTop: 8 }}>
          No collection needs review.
        </div>
      ) : (
        <div className="report-table-wrap">
          <table className="data-table report-table">
            <thead>
              <tr>
                <th>date</th>
                <th className="th-left">airfield</th>
                <th className="th-left">country</th>
                <th className="th-left">conditions</th>
                <th>detections</th>
                <th>flagged</th>
                <th>flag rate</th>
                {showP && <th title="One-sided binomial test against the validation-set flag rate, indicative only">p (indicative)</th>}
                <th>reviewed</th>
              </tr>
            </thead>
            <tbody>
              {needsReview.map((g) => (
                <tr key={g.group_id}>
                  <td className="mono">{g.date}</td>
                  <td className="td-left">{g.airfield}</td>
                  <td className="td-left">{g.country}</td>
                  <td className="td-left cell-dim">
                    {g.weather}
                    {g.off_nadir != null ? ` · ${g.off_nadir.toFixed(0)}°` : ''}
                  </td>
                  <td className="mono">{count(g.n_detections)}</td>
                  <td className="mono accent-amber">{count(g.n_flagged)}</td>
                  <td className="mono">{pct(g.flag_rate)}</td>
                  {showP && <td className="mono">{pValue(g.p_value)}</td>}
                  <td className="mono cell-dim">{count(g.n_reviewed)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Section>
  );
}

// ---------- 3. analyst decisions ----------

function AnalystDecisionsSection({ reviews, latest, a, b }: { reviews: Review[]; latest: FieldCompareVM | null; a: string | undefined; b: string | undefined }) {
  const changes = useChangesSummary(a, b);
  const counts = useMemo(() => {
    const c: Record<ReviewAction, number> = { confirm: 0, new_class: 0, not_aircraft: 0, skip: 0, correct: 0, bad_box: 0 };
    for (const r of reviews) if (r.action in c) c[r.action] += 1;
    return c;
  }, [reviews]);
  const rf = latest?.b.reviewedFlagged ?? latest?.a.reviewedFlagged ?? null;
  const fieldPrecision = rf && rf.n > 0 ? pct(rf.fieldPrecisionOnFlagged) : NA;
  const badBoxRate = rf && rf.n > 0 ? pct(rf.nBadBox / rf.n) : NA;
  const cs = changes.data;

  return (
    <Section title="Analyst decisions" info={<Rule label="About analyst decisions" text="Counts of every analyst review (source = detector chips and ground-truth chips). Confirm, correct and bad box are recorded as evidence and never alter the model; only new class fits anything." />}>
      <div className="report-tiles report-tiles-6">
        {ACTION_ORDER.map((act) => (
          <Tile key={act} label={REVIEW_ACTION_LABELS[act].toLowerCase()} value={count(counts[act])} />
        ))}
      </div>
      <div className="report-tiles report-tiles-3" style={{ marginTop: 8 }}>
        <Tile
          label="field precision on flagged"
          value={fieldPrecision}
          sub={rf && rf.n > 0 ? `${count(rf.n)} reviewed · ${count(rf.nNotAircraft)} not aircraft · ${count(rf.nBadBox)} bad box` : 'no reviewed flagged detections in a built version'}
          info={<Rule label="Field precision rule" text="Field precision on flagged = (reviewed − not aircraft − bad box) / reviewed, over reviewed flagged detections in the latest version's field block." />}
        />
        <Tile label="bad-box rate" value={badBoxRate} sub={rf && rf.n > 0 ? `${count(rf.nBadBox)} of ${count(rf.n)} reviewed flagged` : NA} info={<Rule label="Bad-box rate rule" text="Bad-box rate = bad box / reviewed, over reviewed flagged detections in the latest version's field block." />} />
        <Tile
          label="verified precision of changes"
          value={a && b ? pct(cs?.verified_precision) : NA}
          sub={a && b && cs ? `${count(cs.n_confirmed)} confirmed of ${count(cs.n_reviewed)} reviewed · ${count(cs.total)} changed ${a}→${b}` : `needs a version beyond ${a ?? 'v1'}`}
          info={<Rule label="Verified precision rule" text="Verified precision = confirmed / (confirmed + corrected + not aircraft + bad box) over re-sorted detections reviewed while the later version was current." />}
        />
      </div>
    </Section>
  );
}

// ---------- 4. versions ----------

type VersionRow = { version: Version; compare: VersionsCompare | undefined; vm: FieldCompareVM | null };

function VersionsSection({ rows, a }: { rows: VersionRow[]; a: string }) {
  const rule = rows.find((r) => r.compare?.military?.adaptation_rule)?.compare?.military?.adaptation_rule;
  return (
    <Section title="Versions" info={rule ? <Rule label="Adaptation rule" text={`Adaptation rule (from the API, verbatim): ${rule}`} /> : undefined}>
      <div className="report-table-wrap">
        <table className="data-table report-table">
          <thead>
            <tr>
              <th>version</th>
              <th>labels used</th>
              <th>assignments</th>
              <th>assignment rate</th>
              <th>assignment gate</th>
              <th>control overall ({a}→v)</th>
              <th className="th-left">worst class</th>
              <th className="th-left">note</th>
            </tr>
          </thead>
          <tbody>
            {rows.map(({ version, compare, vm }) => {
              const side = vm?.b;
              const labels = side && side.classes.length > 0 ? side.classes.reduce((acc, c) => acc + (c.nLabelsUsed ?? 0), 0) : version.n_new_class;
              const assigned = side ? side.classes.reduce((acc, c) => acc + (c.nAssigned ?? 0), 0) : null;
              const rate = side && side.classes.length === 1 ? side.classes[0].assignmentRate : assigned != null && side?.nEligible ? assigned / side.nEligible : null;
              const gate = version.assign_min_conf ?? side?.assignMinConf ?? null;
              const ctlA = vm?.a.control?.overall ?? compare?.ref_holdout.role_accuracy_a ?? null;
              const ctlB = vm?.b.control?.overall ?? compare?.ref_holdout.role_accuracy_b ?? null;
              const worst = vm?.worst ?? null;
              const worstBad = worst?.change != null && worst.change < -0.005;
              return (
                <tr key={version.id}>
                  <td className="mono">{version.id}</td>
                  <td className="mono">{count(labels)}</td>
                  <td className="mono">{compare ? count(assigned) : '…'}</td>
                  <td className="mono">{compare ? pct(rate, 2) : '…'}</td>
                  <td className="mono">{gate ? `conf ≥ ${score(gate, 2)}` : 'off'}</td>
                  <td className="mono">
                    {compare ? (
                      <>
                        {pct(ctlA)} → {pct(ctlB)}
                      </>
                    ) : (
                      '…'
                    )}
                  </td>
                  <td className={`td-left${worstBad ? ' accent-red' : ''}`}>
                    {compare ? (worst ? `${shortRole(worst.cls)} ${ppDelta(worst.change)}` : 'none') : '…'}
                  </td>
                  <td className="td-left cell-dim report-note-cell" title={version.note}>
                    {version.note.split(';')[0]}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {rows.map(({ version, vm }) =>
        vm?.b.labeledCollections.map((g) => (
          <div key={`${version.id}-${g.groupId}`} className="caption report-labeled-line">
            {version.id} · labeled collection {g.airfield} · {g.date}: {count(g.nLabeled)} labeled, {count(g.nFlaggedUnlabeledAssigned)} of {count(g.nFlaggedUnlabeled)} remaining flagged re-sorted,{' '}
            {count(g.nBadBox)} bad box, {count(g.nNotAircraft)} not aircraft (excluded from assignments).
          </div>
        )),
      )}
    </Section>
  );
}

// ---------- 5. model health ----------

function HealthSection({ health }: { health: HealthResponse }) {
  return (
    <Section title="Model health" info={<Rule label="About model health" text={`Indicative: ${health.note}. Recommendation = worst indicator status.`} />}>
      <div className="health-tiles" style={{ marginTop: 0 }}>
        {health.indicators.map((ind) => (
          <IndicatorTile key={ind.key} ind={ind} />
        ))}
      </div>
      <ul className="report-bullets">
        {(health.reasons.length > 0 ? health.reasons.slice(0, 3) : ['Every indicator is nominal.']).map((r) => (
          <li key={r}>{r}</li>
        ))}
      </ul>
    </Section>
  );
}

// ---------- 6. limitations ----------

function LimitationsSection() {
  return (
    <Section title="Limitations">
      <ul className="report-bullets">
        {LIMITATIONS.map((l) => (
          <li key={l}>{l}</li>
        ))}
      </ul>
    </Section>
  );
}

// ---------- 7. benchmark evidence (benchmark view only) ----------

function BenchmarkSection({ batches, ev }: { batches: Batch[]; ev: EvalResponse }) {
  const nov = ev.novelty;
  const fr = nov.novelty_flag_rate;
  const det = ev.detection?.by_role_group;
  const ordered = [...batches].sort((x, y) => x.order - y.order);
  return (
    <Section title="Benchmark evidence" info={<Rule label="About benchmark evidence" text={`Ground truth used for evaluation only. ${ev.detection?.note ?? ''}`.trim()} />}>
      <div className="report-tiles report-tiles-4">
        <Tile label="AUROC military vs civil" value={score(nov.auroc_military_vs_civil)} sub={`kNN baseline ${score(nov.baseline_knn_auroc)}`} />
        <Tile label="wingspan-stratified AUROC" value={score(nov.auroc_wingspan_stratified)} info={<Rule label="Wingspan-stratified method" text={nov.wingspan_stratified_method} />} />
        <Tile label="flag rate military / civil" value={fr ? `${pct(fr.military)} / ${pct(fr.civil)}` : NA} sub={fr ? `n ${count(fr.n_military)} / ${count(fr.n_civil)}` : undefined} />
        <Tile label="detector recall military / civil" value={det ? `${pct(det.military.recall)} / ${pct(det.civil.recall)}` : NA} sub={det?.scope ?? undefined} />
      </div>
      <div className="report-table-wrap">
        <table className="data-table report-table">
          <thead>
            <tr>
              <th>batch</th>
              <th>chips</th>
              <th>novelty rate</th>
              <th>drift ratio</th>
              <th>p (indicative)</th>
              <th>detector recall</th>
              <th>role accuracy</th>
            </tr>
          </thead>
          <tbody>
            {ordered.map((b) => (
              <tr key={b.id}>
                <td title={b.name}>{shortBatchName(b.id)}</td>
                <td className="mono">{count(b.n_chips)}</td>
                <td className="mono">{pct(b.novelty_rate)}</td>
                <td className="mono">{score(b.drift.ratio_to_calib, 2)}×</td>
                <td className="mono">{pValue(b.drift.p_value)}</td>
                <td className="mono">{pct(b.detection?.recall)}</td>
                <td className="mono">{pct(b.role_accuracy)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Section>
  );
}

function BenchmarkEvidence() {
  const batches = useBatches();
  const ev = useEval();
  if (batches.error || ev.error) return null;
  if (!batches.data || !ev.data) return <div className="caption">Loading benchmark evidence…</div>;
  return <BenchmarkSection batches={batches.data} ev={ev.data} />;
}

// ---------- page ----------

function ReportBody({ summary }: { summary: Summary }) {
  const { view } = useView();
  const [stamp] = useState(generatedStamp);
  const health = useHealth();
  const collections = useCollections();
  const reviews = useReviewLog('all');
  const versions = summary.versions;
  const a = versions[0]?.id;
  const later = versions.slice(1);
  const b = later.length > 0 ? summary.current_version : undefined;

  const compares = useQueries({
    queries: later.map((v) => ({ queryKey: qk.compare(a ?? '', v.id), queryFn: () => api.versionsCompare(a as string, v.id), enabled: !!a })),
  });
  const rows: VersionRow[] = later.map((v, i) => {
    const compare = compares[i]?.data;
    return { version: v, compare, vm: compare ? normalizeFieldCompare(compare) : null };
  });
  const latest = rows.length > 0 ? (rows[rows.length - 1].vm ?? null) : null;

  return (
    <div className="report-page">
      <Header summary={summary} health={health.data} stamp={stamp} />
      <ModelSection summary={summary} referenceFlagRate={collections.data?.reference_flag_rate} />
      {collections.error ? null : collections.data ? (
        <SinceDeliverySection groups={collections.data.groups} statusRule={collections.data.status_rule} />
      ) : (
        <div className="caption">Loading collections…</div>
      )}
      <AnalystDecisionsSection reviews={reviews.data ?? []} latest={latest} a={b ? a : undefined} b={b} />
      {a && rows.length > 0 && <VersionsSection rows={rows} a={a} />}
      {health.data && <HealthSection health={health.data} />}
      <LimitationsSection />
      {view === 'benchmark' && <BenchmarkEvidence />}
      <div className="report-footer caption">prototype · public RarePlanes (WorldView-3) · evidence only, not a T&amp;E judgement</div>
    </div>
  );
}

export function Report() {
  const summary = useSummary();
  return (
    <div>
      <div className="page-header report-page-header">
        <div>
          <h1 className="page-title">Report</h1>
          <div className="page-sub">One-page assurance summary; the full markdown report is the file export.</div>
        </div>
      </div>
      <QueryBoundary isLoading={summary.isLoading} error={summary.error} data={summary.data} loadingLabel="Loading report…">
        {(data) => <ReportBody summary={data} />}
      </QueryBoundary>
    </div>
  );
}
