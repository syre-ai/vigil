import type { CollectionGroup, VersionsCompare } from '../api/types';
import { InfoPopover } from './InfoPopover';
import { NA, count, pct, ppDelta } from '../lib/format';
import { normalizeFieldCompare, type FieldClassVM, type FieldGroupCount, type FieldLabeledCollectionVM } from '../lib/fieldCompare';

function ChangeCell({ a, b }: { a: number | null | undefined; b: number | null | undefined }) {
  if (a == null || b == null) return <td className="mono">{NA}</td>;
  const delta = b - a;
  const regressed = delta < -0.005;
  const improved = delta > 0.005;
  return <td className={`mono ${regressed ? 'accent-red' : improved ? 'accent-green' : ''}`}>{ppDelta(delta)}</td>;
}

function groupName(g: FieldGroupCount, groups: Map<string, CollectionGroup> | undefined): string {
  const known = groups?.get(g.key);
  if (known) return `${known.airfield} · ${known.date}`;
  if (g.airfield) return `${g.airfield}${g.date ? ` · ${g.date}` : ''}`;
  return g.key;
}

// One data-driven line per labeled collection (a collection containing >=1 analyst-labeled det
// chip) — DISPLAY ONLY, explicitly excluded from the assignments table above (its own labeled
// exemplars and every other chip from the same collection's flagged-detection story are counted
// here, never folded into `n_assigned`/`assignment_rate`). See backend/API.md's
// FieldBlock.labeled_collections and vigil.field._labeled_collections_block.
function labeledCollectionLine(g: FieldLabeledCollectionVM, groups: Map<string, CollectionGroup> | undefined): string {
  const known = groups?.get(g.groupId);
  const name = known ? `${known.airfield} · ${known.date}` : `${g.airfield} · ${g.date}`;
  const resortText =
    g.perClass.length === 0
      ? `${count(g.nFlaggedUnlabeledAssigned)} of ${count(g.nFlaggedUnlabeled)} remaining flagged detections re-sorted`
      : `${count(g.nFlaggedUnlabeledAssigned)} of ${count(g.nFlaggedUnlabeled)} remaining flagged detections re-sorted to ${g.perClass
          .map((r) => (g.perClass.length > 1 ? `${r.label} (${count(r.n)})` : r.label))
          .join(', ')}`;
  const badBoxText =
    g.nBadBox > 0
      ? `${count(g.nBadBox)} bad box (${count(g.nBadBoxAssigned)} of them now carry the new class: poor boxes, but the aircraft inside were recognised)`
      : '0 bad box';
  const remainingText = g.nFlaggedUnlabeled > 0 ? `${resortText}; ` : '';
  return `In the labeled collection ${name} (excluded from the counts below): ${count(g.nLabeled)} labeled; ${remainingText}${badBoxText}, ${count(g.nNotAircraft)} not aircraft.`;
}

function LabeledCollectionsNote({ collections, groups }: { collections: FieldLabeledCollectionVM[]; groups: Map<string, CollectionGroup> | undefined }) {
  if (collections.length === 0) return null;
  return (
    <div className="field-compare-note" style={{ marginBottom: 8 }}>
      {collections.map((g) => (
        <div key={g.groupId} className="caption">
          {labeledCollectionLine(g, groups)}
        </div>
      ))}
    </div>
  );
}

function TopCollections({ cls, groups }: { cls: FieldClassVM | undefined; groups: Map<string, CollectionGroup> | undefined }) {
  const top = cls?.byGroup.slice(0, 3) ?? [];
  if (top.length === 0) return <td className="cell-dim">{NA}</td>;
  return (
    <td style={{ textAlign: 'left' }}>
      <ul className="compact-list">
        {top.map((g) => (
          <li key={g.key}>
            {groupName(g, groups)} <span className="mono">({count(g.n)})</span>
          </li>
        ))}
      </ul>
    </td>
  );
}

// Field-mode compare — what the analyst's decision did, in field-computable
// terms only: (1) per analyst class, labels used and how many detections
// across the new collections are now assigned to it (descriptive — there is
// no ground truth for a detector chip); (2) the labeled control-set
// regression check — accuracy on the labeled validation set before/after,
// the only accuracy-style number here. Never military AUROC, never an
// assignment rate on ground truth, never dataset domain names.
export function FieldCompareTable({ compare, groups }: { compare: VersionsCompare; groups?: Map<string, CollectionGroup> }) {
  const vm = normalizeFieldCompare(compare);
  const labels = [...new Set([...vm.a.classes.map((c) => c.label), ...vm.b.classes.map((c) => c.label)])];
  const aByLabel = new Map(vm.a.classes.map((c) => [c.label, c]));
  const bByLabel = new Map(vm.b.classes.map((c) => [c.label, c]));
  const control = vm.b.control ?? vm.a.control;
  const aCtl = new Map(vm.a.control?.perClass.map((r) => [r.cls, r]) ?? []);
  const bCtl = new Map(vm.b.control?.perClass.map((r) => [r.cls, r]) ?? []);
  const ctlClasses = [...new Set([...aCtl.keys(), ...bCtl.keys()])];
  const note = vm.b.note ?? vm.a.note ?? 'assignments are descriptive; not ground-truth accuracy';
  const hasFieldBlock = compare.field != null && (vm.b.nDetections != null || vm.b.classes.length > 0);
  const reviewedFlagged = vm.b.reviewedFlagged ?? vm.a.reviewedFlagged;
  const labeledCollections = vm.b.labeledCollections.length > 0 ? vm.b.labeledCollections : vm.a.labeledCollections;
  const assignMinConf = vm.b.assignMinConf ?? vm.a.assignMinConf;

  return (
    <div className="panel">
      <div className="panel-title">
        What changed · {compare.a} → {compare.b}
        <InfoPopover label="About this comparison" width={360}>
          <p className="info-popover-text">
            <strong>Assignments</strong> count detections across the new collections whose predicted role is now the analyst&apos;s class. The new class may only claim detections the
            trained model already rejected as novel; the labeled exemplars and every detection from the same parent image are excluded from the counts. Assignments are descriptive,
            not accuracy. Nothing in the new collections is labeled.
          </p>
          <p className="info-popover-text">
            <strong>Labeled control-set regression check</strong>: role accuracy on the labeled validation set before and after: the one place a before/after accuracy is
            legitimate. A drop in a civil class means the new class pulled in aircraft the validated model used to get right.
          </p>
        </InfoPopover>
      </div>

      {!hasFieldBlock && (
        <div className="state-block" style={{ paddingTop: 8, paddingBottom: 8 }}>
          This backend version did not return field-mode assignment counts, showing the labeled control-set regression check only.
        </div>
      )}

      {hasFieldBlock && (
        <>
          <div className="panel-title" style={{ fontSize: 13, marginTop: 10, display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', gap: 8 }}>
            <span>Analyst classes · assignments across new collections</span>
            <span className="caption cell-dim" title="Optional additional gate on top of the nearest-exemplar rule: a detector chip may only be claimed if its own detection confidence also clears this threshold.">
              assignment gate: {assignMinConf ? `conf ≥ ${assignMinConf}` : 'off'}
            </span>
          </div>
          <LabeledCollectionsNote collections={labeledCollections} groups={groups} />
          <table className="data-table">
            <thead>
              <tr>
                <th>class</th>
                <th>labels used</th>
                <th>assigned {compare.a}</th>
                <th>assigned {compare.b}</th>
                <th title="assigned ÷ eligible detections (unlabeled, not same-parent) across new collections">rate {compare.b}</th>
                <th style={{ textAlign: 'left' }}>top collections by assignments</th>
              </tr>
            </thead>
            <tbody>
              {labels.length === 0 ? (
                <tr>
                  <td colSpan={6} className="cell-dim">
                    No analyst classes yet. Only new-class labels create one.
                  </td>
                </tr>
              ) : (
                labels.map((label) => {
                  const ra = aByLabel.get(label);
                  const rb = bByLabel.get(label);
                  return (
                    <tr key={label}>
                      <td>{label}</td>
                      <td className="mono">{count(rb?.nLabelsUsed ?? ra?.nLabelsUsed)}</td>
                      <td className="mono">{count(ra?.nAssigned ?? 0)}</td>
                      <td className="mono">{count(rb?.nAssigned ?? 0)}</td>
                      <td className="mono">{pct(rb?.assignmentRate, 2)}</td>
                      <TopCollections cls={rb} groups={groups} />
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
          <div className="caption field-compare-note">
            {count(vm.b.nDetections)} detections across new collections
            {vm.b.nExcludedLabeled != null || vm.b.nExcludedSameParent != null ? (
              <>
                {' '}
                · excluded from the counts: {count(vm.b.nExcludedLabeled ?? 0)} labeled exemplar{(vm.b.nExcludedLabeled ?? 0) === 1 ? '' : 's'}, {count(vm.b.nExcludedSameParent ?? 0)} from the same
                parent image{(vm.b.nExcludedSameParent ?? 0) === 1 ? '' : 's'}
              </>
            ) : null}
            {vm.b.nEligible != null ? ` · ${count(vm.b.nEligible)} eligible` : ''}. {note}.
          </div>
          {reviewedFlagged && reviewedFlagged.n > 0 && (
            <div className="caption field-compare-note">
              Of {count(reviewedFlagged.n)} reviewed flagged detections: {count(reviewedFlagged.nNotAircraft)} not aircraft, {count(reviewedFlagged.nBadBox)} bad box → field precision on
              flagged ≈ {pct(reviewedFlagged.fieldPrecisionOnFlagged)}.
            </div>
          )}
        </>
      )}

      <div className="panel-title" style={{ fontSize: 13, marginTop: 14 }}>
        Labeled control-set regression check
        <InfoPopover label="About this check" width={300}>
          Role accuracy on the labeled validation set (never the new collections), before and after applying feedback. The worst per-class change is highlighted.
        </InfoPopover>
      </div>
      {!control ? (
        <div className="state-block">Control-set numbers are not available for these versions.</div>
      ) : (
        <>
          <table className="data-table">
            <thead>
              <tr>
                <th>class</th>
                <th>n</th>
                <th>{compare.a}</th>
                <th>{compare.b}</th>
                <th>change</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>overall</td>
                <td className="mono cell-dim">{count(ctlClasses.reduce((s, c) => s + (bCtl.get(c)?.n ?? aCtl.get(c)?.n ?? 0), 0) || null)}</td>
                <td className="mono">{pct(vm.a.control?.overall)}</td>
                <td className="mono">{pct(vm.b.control?.overall)}</td>
                <ChangeCell a={vm.a.control?.overall} b={vm.b.control?.overall} />
              </tr>
              {ctlClasses.map((cls) => {
                const ra = aCtl.get(cls);
                const rb = bCtl.get(cls);
                const isWorst = vm.worst?.cls === cls && (vm.worst.change ?? 0) < 0;
                return (
                  <tr key={cls} className={isWorst ? 'field-compare-worst' : undefined} title={isWorst ? 'Worst regression' : undefined}>
                    <td>{cls}</td>
                    <td className="mono cell-dim">{count(rb?.n ?? ra?.n)}</td>
                    <td className="mono">{pct(ra?.accuracy)}</td>
                    <td className="mono">{pct(rb?.accuracy)}</td>
                    <ChangeCell a={ra?.accuracy} b={rb?.accuracy} />
                  </tr>
                );
              })}
            </tbody>
          </table>
          <div className="caption" style={{ marginTop: 6 }}>
            {vm.worst && (vm.worst.change ?? 0) < 0
              ? `Worst regression: ${vm.worst.cls} ${pct(vm.worst.a)} → ${pct(vm.worst.b)} (${ppDelta(vm.worst.change)}).`
              : 'No per-class regression on the labeled control set.'}{' '}
            A check against the one labeled set, not an accreditation decision.
          </div>
        </>
      )}
    </div>
  );
}
