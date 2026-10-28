// Normalizes /versions/compare's `field` block — whichever of the two shapes
// backend/API.md ends up with (see api/types.ts FieldCompare) — into one
// view model for FieldCompareTable. Nothing benchmark-flavored survives the
// normalization: `by_domain`/`domains` (dataset domain names) are dropped on
// purpose; only per-collection (`by_group`) breakdowns are kept.
import type { FieldBlock, FieldByGroup, FieldCompare, FieldControlBlock, FieldLabeledCollection, FieldReviewedFlagged, FieldVersionEntry, VersionsCompare } from '../api/types';

export type FieldGroupCount = { key: string; airfield?: string; date?: string; n: number };

export type FieldClassVM = {
  label: string;
  nLabelsUsed: number | null;
  nAssigned: number | null;
  assignmentRate: number | null;
  byGroup: FieldGroupCount[];
};

export type FieldControlVM = {
  overall: number | null;
  perClass: { cls: string; accuracy: number | null; n: number | null }[];
};

export type FieldReviewedFlaggedVM = {
  n: number;
  nNotAircraft: number;
  nBadBox: number;
  fieldPrecisionOnFlagged: number | null;
};

export type FieldLabeledCollectionVM = {
  groupId: string;
  airfield: string;
  date: string;
  nLabeled: number;
  nBadBox: number;
  nBadBoxAssigned: number;
  nNotAircraft: number;
  nFlaggedUnlabeled: number;
  nFlaggedUnlabeledAssigned: number;
  perClass: { label: string; n: number }[];
};

export type FieldSideVM = {
  nDetections: number | null;
  nExcludedLabeled: number | null;
  nExcludedSameParent: number | null;
  nEligible: number | null;
  classes: FieldClassVM[];
  control: FieldControlVM | null;
  reviewedFlagged: FieldReviewedFlaggedVM | null;
  labeledCollections: FieldLabeledCollectionVM[];
  assignMinConf: number | null;
  note: string | null;
};

export type FieldCompareVM = {
  a: FieldSideVM;
  b: FieldSideVM;
  // Class with the largest control-set accuracy drop a → b, if any.
  worst: { cls: string; a: number | null; b: number | null; change: number | null } | null;
};

function num(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

function normalizeByGroup(bg: FieldByGroup | undefined): FieldGroupCount[] {
  if (!bg) return [];
  const out: FieldGroupCount[] = [];
  if (Array.isArray(bg)) {
    for (const e of bg) {
      const n = num(e.n_assigned) ?? num(e.n);
      const key = e.group_id ?? (e.airfield ? `${e.airfield}_${e.date ?? ''}` : null);
      if (key == null || n == null) continue;
      out.push({ key, airfield: e.airfield, date: e.date, n });
    }
  } else {
    for (const [key, v] of Object.entries(bg)) {
      const n = num(v);
      if (n != null) out.push({ key, n });
    }
  }
  return out.filter((g) => g.n > 0).sort((x, y) => y.n - x.n);
}

function normalizeControl(c: FieldControlBlock | undefined): FieldControlVM | null {
  if (!c) return null;
  const perClass: FieldControlVM['perClass'] = [];
  const pc = c.per_class;
  if (Array.isArray(pc)) {
    for (const e of pc) {
      const cls = e.cls ?? e.label;
      if (cls) perClass.push({ cls, accuracy: num(e.accuracy), n: num(e.n) });
    }
  } else if (pc && typeof pc === 'object') {
    for (const [cls, e] of Object.entries(pc)) perClass.push({ cls, accuracy: num(e?.accuracy), n: num(e?.n) });
  }
  return { overall: num(c.overall) ?? num(c.accuracy_ref_holdout), perClass };
}

function normalizeReviewedFlagged(rf: FieldReviewedFlagged | null | undefined): FieldReviewedFlaggedVM | null {
  const n = num(rf?.n);
  if (n == null) return null;
  return {
    n,
    nNotAircraft: num(rf?.n_not_aircraft) ?? 0,
    nBadBox: num(rf?.n_bad_box) ?? 0,
    fieldPrecisionOnFlagged: num(rf?.field_precision_on_flagged),
  };
}

function normalizeLabeledCollections(lc: FieldLabeledCollection[] | null | undefined): FieldLabeledCollectionVM[] {
  if (!Array.isArray(lc)) return [];
  return lc.map((g) => ({
    groupId: g.group_id,
    airfield: g.airfield,
    date: g.date,
    nLabeled: num(g.n_labeled) ?? 0,
    nBadBox: num(g.n_bad_box) ?? 0,
    nBadBoxAssigned: num(g.n_bad_box_assigned) ?? 0,
    nNotAircraft: num(g.n_not_aircraft) ?? 0,
    nFlaggedUnlabeled: num(g.n_flagged_unlabeled) ?? 0,
    nFlaggedUnlabeledAssigned: num(g.n_flagged_unlabeled_assigned) ?? 0,
    perClass: Object.entries(g.per_class ?? {})
      .map(([label, n]) => ({ label, n: num(n) ?? 0 }))
      .filter((r) => r.n > 0)
      .sort((x, y) => y.n - x.n),
  }));
}

function unwrap(entry: FieldVersionEntry | undefined): { block: FieldBlock | null; control: FieldControlBlock | undefined } {
  if (!entry) return { block: null, control: undefined };
  const e = entry as FieldBlock & { field?: FieldBlock | null; control?: FieldControlBlock };
  // old shape: {field, control}; new shape: the block itself (with .control)
  if (e.field !== undefined && e.classes === undefined && e.n_detections === undefined) {
    return { block: e.field ?? null, control: e.control ?? e.field?.control };
  }
  return { block: e, control: e.control };
}

function normalizeSide(entry: FieldVersionEntry | undefined): FieldSideVM {
  const { block, control } = unwrap(entry);
  const classes: FieldClassVM[] = (block?.classes ?? []).map((r) => ({
    label: r.label,
    nLabelsUsed: num(r.n_labels_used),
    nAssigned: num(r.n_assigned),
    assignmentRate: num(r.assignment_rate),
    byGroup: normalizeByGroup(r.by_group),
  }));
  return {
    nDetections: num(block?.n_detections),
    nExcludedLabeled: num(block?.n_excluded_labeled),
    nExcludedSameParent: num(block?.n_excluded_same_parent),
    nEligible: num(block?.n_eligible),
    classes,
    control: normalizeControl(control),
    reviewedFlagged: normalizeReviewedFlagged(block?.reviewed_flagged),
    labeledCollections: normalizeLabeledCollections(block?.labeled_collections),
    assignMinConf: num(block?.assign_min_conf),
    note: block?.note ?? null,
  };
}

// Fallback control block from the compare's top-level `ref_holdout` — the
// same labeled validation-set role-accuracy numbers under the field-mode
// label — used only when a version's own `control` block is missing.
function controlFromRefHoldout(compare: VersionsCompare, side: 'a' | 'b'): FieldControlVM | null {
  const rh = compare.ref_holdout;
  if (!rh) return null;
  return {
    overall: num(side === 'a' ? rh.role_accuracy_a : rh.role_accuracy_b),
    perClass: (rh.per_class ?? []).map((r) => ({ cls: r.cls, accuracy: num(side === 'a' ? r.a : r.b), n: null })),
  };
}

export function normalizeFieldCompare(compare: VersionsCompare): FieldCompareVM {
  const field: FieldCompare | null | undefined = compare.field;
  const a = normalizeSide(field?.a);
  const b = normalizeSide(field?.b);
  if (!a.control) a.control = controlFromRefHoldout(compare, 'a');
  if (!b.control) b.control = controlFromRefHoldout(compare, 'b');

  // Worst regression: prefer the backend's, else compute from per-class a/b.
  let worst: FieldCompareVM['worst'] = null;
  const wr = field?.worst_regression;
  const wrCls = wr?.cls ?? wr?.label;
  if (wr && wrCls) {
    const wa = num(wr.a);
    const wb = num(wr.b);
    // API.md: delta = a − b (positive = forgetting). Display change as b − a.
    const change = wa != null && wb != null ? wb - wa : num(wr.delta) != null ? -(num(wr.delta) as number) : null;
    worst = { cls: wrCls, a: wa, b: wb, change };
  } else if (a.control && b.control) {
    const bMap = new Map(b.control.perClass.map((r) => [r.cls, r.accuracy]));
    for (const r of a.control.perClass) {
      const bv = bMap.get(r.cls);
      if (r.accuracy == null || bv == null) continue;
      const change = bv - r.accuracy;
      if (change < 0 && (worst == null || (worst.change ?? 0) > change)) worst = { cls: r.cls, a: r.accuracy, b: bv, change };
    }
  }
  return { a, b, worst };
}
