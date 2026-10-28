// Types mirroring backend/API.md verbatim. This file is the single source
// of truth for the shape of every API response the UI consumes.

export type BatchId = 'ref_holdout' | 'off_nadir' | 'foreign' | 'haze' | 'snow' | 'military' | 'synthetic' | string;

export type Drift = {
  score: number;
  null_p95: number;
  p_value: number; // indicative, not inferential (block-bootstrap null) — never call this "significant"
  n_boot: number;
  statistic: string; // "mean 5-NN cosine distance to reference"
  null_method: string; // e.g. "tile-level block bootstrap over ref_calib (...)"
  calib_mean: number; // mean of `statistic` over ref_calib itself
  ratio_to_calib: number; // score / calib_mean
  level?: string; // e.g. "scene" — granularity the statistic is computed at; may be absent on older backends
};

export type ConfShift = {
  ref_mean: number;
  batch_mean: number;
  ks_p: number;
} | null;

export type Detection = {
  recall: number;
  precision: number;
  n_gt: number;
  n_det?: number | null; // detector milestone field — may be absent on older backends
} | null;

export type Batch = {
  id: BatchId;
  order: number;
  name: string;
  description: string;
  condition: Record<string, string | number>;
  n_tiles: number;
  n_chips: number;
  n_parent_images: number;
  drift: Drift;
  novelty_rate: number;
  novelty_rate_military?: number | null; // military-batch-only field: flag rate among GT military chips (ground truth, evaluation only)
  novelty_rate_civil?: number | null; // military-batch-only field: flag rate among GT civil chips (ground truth, evaluation only)
  role_accuracy: number | null;
  detection: Detection;
  conf_shift: ConfShift;
};

export type Histogram = { bins: number[]; counts: number[] };

export type ChipMeta = {
  airfield: string;
  country: string;
  weather: string;
  off_nadir: number;
  gsd: number;
  wingspan_m: number | null;
};

export type ReviewAction = 'confirm' | 'new_class' | 'not_aircraft' | 'skip' | 'correct' | 'bad_box';

export type Review = {
  ann_id: number;
  action: ReviewAction;
  label: string | null;
  note: string | null;
  ts: string;
  version_applied: string | null;
  // The chip's role_pred / the current version id at the moment this review was POSTed
  // (snapshot, overwritten on a re-review) — absent/null on reviews recorded before this
  // field existed. Used as the "from" side of analyst-disagreement evidence.
  predicted_role?: string | null;
  predicted_version?: string | null;
};

export type Chip = {
  ann_id: number;
  tile_id: number;
  batch_id: BatchId;
  version: string;
  bbox: [number, number, number, number];
  chip_url: string;
  tile_url: string;
  role_pred: string;
  role_conf: number;
  novelty: number;
  is_novel: boolean;
  art_category: number | null;
  knn_dist: number;
  umap: [number, number] | null;
  used_in_fit: boolean; // true if used to fit a new prototype/ART category — excluded from evaluation metrics
  meta: ChipMeta;
  role_true: string | null;
  review: Review | null;
  detected?: boolean | null; // detector milestone field: whether the detector localized this GT chip; null/absent = not yet available
  partial?: number | null; // edge/partial-chip field: 0-1 fraction of the aircraft inside the tile; absent on older backends
  truncated?: boolean | null; // edge/partial-chip field: true if the aircraft is cut by the tile edge; absent on older backends
};

export type ChipNeighbor = { ann_id: number; chip_url: string; role_true: string | null; dist: number };

export type ChipDetail = Chip & { neighbors: ChipNeighbor[]; art_match: number };

export type Version = {
  id: string;
  created: string;
  classes: string[];
  n_feedback: number;
  n_confirm: number;
  n_new_class: number;
  n_not_aircraft: number;
  n_correct?: number; // corrections folded into this version — recorded as disagreement, never alters the model
  n_bad_box?: number; // localization-error marks folded into this version — recorded as evidence, never alters the model
  assign_min_conf?: number | null; // this version's POST /adapt gate on det-chip assignment (0/null = off) — see backend/API.md
  note: string;
};

export type ModelInfo = {
  name: string;
  detector: string;
  embedder: string;
  novelty_method: string;
  reference_domain: string;
  n_ref_chips: number;
  n_ref_images: number;
  vigilance: number;
  calibrated_on: string;
  localization: string;
  detector_card?: DetectorCard | null; // detector milestone field — structured detector model-card facts
  // optional model-selection fields — shown in a collapsible notes row if present. calibration_check has
  // been observed as either a plain string or a structured split-stability summary (e.g. {min, median,
  // max, n_parents}); render defensively for either shape.
  calibration_check?: string | Record<string, unknown> | null;
  model_selection_note?: string | null;
  // newer field — the analyst-class adaptation-gate rule, verbatim, e.g. "analyst-created
  // classes can only claim chips that fail the vigilance test against every trained
  // category (novel); among those, the nearest prototype ... decides". Shown in the
  // collapsible calibration & model-selection notes when present.
  adaptation_rule?: string | null;
};

// detector milestone field: /summary.model.detector_card
export type DetectorCard = {
  name: string;
  trained_on: string;
  val_split: string;
  val_map50: number | null;
  val_map50_95: number | null;
  val_precision: number | null;
  val_recall: number | null;
  conf_threshold: number | null;
  iou_threshold: number | null;
  note: string;
};

export type Summary = {
  model: ModelInfo;
  versions: Version[];
  current_version: string;
  batches: Batch[];
  totals: { chips: number; novel: number; reviewed: number };
};

export type BatchDetail = Batch & {
  novelty_hist: Histogram;
  ref_novelty_hist: Histogram;
  top_novel: Chip[];
};

export type ChipsPage = { total: number; items: Chip[] };

// detector milestone field: one detector-predicted box on a tile, in the same
// 512×512 pixel space as Chip.bbox. `tp` = matched a ground-truth box (true positive).
export type DetectorBox = { x1: number; y1: number; x2: number; y2: number; conf: number; tp: boolean };

export type TileDetail = {
  tile_id: number;
  file_name: string;
  meta: Omit<ChipMeta, 'wingspan_m'> & { lat?: number; lon?: number };
  chips: Chip[];
  detections?: DetectorBox[] | null; // detector milestone field — absent/undefined on older backends, null on newer ones
};

export type EmbeddingRefPoint = { ann_id: number; x: number; y: number; role_true: string | null };
export type EmbeddingBatchPoint = { ann_id: number; x: number; y: number; novelty: number; is_novel: boolean; art_category: number | null };
export type EmbeddingPoints = { reference: EmbeddingRefPoint[]; batch: EmbeddingBatchPoint[] };

export type DriftSeriesEntry = {
  batch_id: BatchId;
  order: number;
  name: string;
  score: number;
  null_p95: number;
  p_value: number;
  n: number;
  calib_mean: number;
  ratio_to_calib: number;
};
export type DriftResponse = { statistic: string; null_method: string; series: DriftSeriesEntry[]; level?: string };

// 'reference' is kept only for backward compatibility with older backends —
// the current contract only ever returns tested/untested/degraded and marks
// the reference-defining bucket via `in_reference_definition` instead.
export type EnvelopeStatus = 'reference' | 'tested' | 'untested' | 'degraded';
export type EnvelopeControlDelta = { role_accuracy: number | null; detection_recall: number | null };
export type EnvelopeBucket = {
  label: string;
  n_chips: number;
  n_images: number;
  n_parent_images?: number; // newer field, preferred over n_images when present
  novelty_rate: number;
  role_accuracy: number | null;
  detection_recall: number | null;
  status: EnvelopeStatus;
  in_reference_definition?: boolean; // newer field — true if this bucket is part of the reference-domain definition
  control_delta?: EnvelopeControlDelta | null; // newer field — this bucket's metrics minus the control batch's
};
export type EnvelopeDimension = { name: string; buckets: EnvelopeBucket[] };
export type EnvelopeControlThresholds = {
  role_accuracy?: number | null;
  novelty_rate?: number | null;
  detection_recall?: number | null;
  n_chips?: number | null;
  n_parent_images?: number | null;
};
export type EnvelopeThresholds = {
  min_n: number;
  degraded_margin: number;
  novelty_multiplier?: number | null; // degraded if bucket novelty_rate > novelty_multiplier × control novelty_rate
  novelty_margin?: number | null; // alternate absolute-margin shape seen in some drafts — kept for safety
  detection_reference_recall?: number | null; // alternate shape seen in some drafts — kept for safety
  control?: EnvelopeControlThresholds | null; // newer field — the control batch's reference values the rule compares against
};
export type EnvelopeResponse = { dimensions: EnvelopeDimension[]; thresholds: EnvelopeThresholds };

export type EvalPerRole = { role: string; n: number; auroc: number | null };
export type EvalPerClass = { cls: string; n: number; accuracy: number | null };
// `analyst_class_assignment_rate` is the primary metric (backend adaptation-rule rewrite):
// among held-out true-military chips (never used to fit any prototype/ART category),
// the fraction assigned to any analyst-created class. Never call this "recall" in the
// UI — analyst classes are reachable only through the Fuzzy-ART vigilance gate described
// in `adaptation_rule`. `rare_target_recall`/`recall_new_class` are exact aliases of the
// same value, kept for older-frontend/backend compatibility — fall back to them if
// `analyst_class_assignment_rate` hasn't landed yet. `civil_false_assignment_rate` is the
// mirror-image error: the fraction of held-out true-civil chips wrongly assigned to an
// analyst-created class. The remaining fields (n_eval_civil, n_excluded_*,
// exclusion_method, adaptation_rule) are newer additions — all optional so this type
// tolerates both the old and new backend shapes.
export type EvalMilitary = {
  analyst_class_assignment_rate?: number | null;
  rare_target_recall: number;
  recall_new_class: number;
  alias_note?: string | null;
  civil_false_assignment_rate?: number | null;
  miss_rate_civil: number;
  n_eval: number;
  n_eval_civil?: number | null;
  n_excluded_used_in_fit?: number | null;
  n_excluded_same_parent?: number | null;
  exclusion_method?: string | null;
  adaptation_rule?: string | null;
};

// wingspan-stratified control, broken out per quartile bin (bins computed from civil chips).
export type WingspanBin = { bin: number; range: [number | null, number | null]; n: number; auroc: number | null };

// detector milestone field: /eval.detection — val-set detector mAP, a civil-vs-military
// recall comparison (ground truth, evaluation only), and per-batch recall/precision.
export type EvalDetectionRoleGroup = { recall: number | null; n: number };
// `scope` (newer field) describes what population/collections both recall
// figures were computed over, e.g. "within the military batch (same
// collections for both groups)" — print it verbatim next to the comparison.
export type EvalDetectionByRoleGroup = { civil: EvalDetectionRoleGroup; military: EvalDetectionRoleGroup; scope?: string };
export type EvalDetectionBatchRow = { recall: number | null; precision: number | null; n_gt: number; n_det?: number | null };
export type EvalDetection = {
  val_map50: number | null;
  val_map50_95: number | null;
  conf_threshold?: number | null;
  iou_threshold?: number | null;
  by_role_group: EvalDetectionByRoleGroup;
  by_role_group_pooled?: EvalDetectionByRoleGroup | null; // newer field — pooled-across-batches variant of by_role_group, own scope
  recall_by_role_group_pooled?: EvalDetectionByRoleGroup | null; // alternate name seen in some drafts of this field — kept for safety
  by_batch: Record<string, EvalDetectionBatchRow>;
  note?: string;
};

// Analyst feedback evidence (confirm = agreement, correct = disagreement + a known class
// label) — see backend/API.md's AnalystDisagreement. Neither action ever alters the model;
// only new_class does. `by_pair` = (predicted role -> corrected-to label) counts, sorted
// desc by n. `/eval.analyst_disagreement` is scoped to the current version;
// `/versions/compare.analyst_disagreement` is deliberately all-time, not per version a/b.
export type AnalystDisagreementPair = { from_role: string; to_role: string; n: number };
export type AnalystDisagreementByRole = { role: string; n_confirm: number; n_correct: number; disagreement_rate: number | null };
export type AnalystDisagreement = {
  n_reviewed: number;
  n_confirm: number;
  n_correct: number;
  n_new_class: number;
  n_not_aircraft: number;
  n_skip: number;
  n_bad_box?: number; // localization-error marks — recorded as evidence, never alters the model
  disagreement_rate: number | null;
  by_pair: AnalystDisagreementPair[];
  by_predicted_role: AnalystDisagreementByRole[];
};

export type NoveltyFlagRatePerRole = { role_id?: number; role_name?: string; n?: number; novelty_flag_rate?: number | null };
export type NoveltyFlagRate = {
  military: number | null;
  civil: number | null;
  n_military?: number;
  n_civil?: number;
  per_role?: NoveltyFlagRatePerRole[];
};

export type EvalResponse = {
  version: string;
  novelty: {
    auroc_military_vs_civil: number | null;
    auprc: number | null;
    auroc_size_matched: number | null;
    auroc_wingspan_stratified: number | null;
    wingspan_stratified_method: string;
    size_matched_bins?: WingspanBin[]; // per-bin AUROCs backing auroc_wingspan_stratified — may be absent
    per_role: EvalPerRole[];
    baseline_knn_auroc: number | null;
    novelty_flag_rate?: NoveltyFlagRate | null; // newer field — military vs civil flag rate at the calibrated vigilance
  };
  role_head: { accuracy_ref_holdout: number | null; per_class: EvalPerClass[] };
  military: EvalMilitary;
  detection?: EvalDetection | null; // detector milestone field — absent/null until it lands
  analyst_disagreement?: AnalystDisagreement | null; // confirm/correct feedback evidence, scoped to this version
  note: string;
};

// 'collection' = det-queue only (see API.md's flagged_det_queue): needs-review
// collections first (by excess flagged desc), then novelty within a collection.
export type ReviewStrategy = 'novelty' | 'uncertainty' | 'random' | 'collection';

export type VersionsCompare = {
  a: string;
  b: string;
  ref_holdout: {
    role_accuracy_a: number | null;
    role_accuracy_b: number | null;
    per_class: { cls: string; a: number | null; b: number | null }[];
  };
  military: {
    // primary metric, both versions (adaptation-rule rewrite) — fall back to the
    // rare_target_recall_*/recall_new_class_* aliases if this hasn't landed yet.
    analyst_class_assignment_rate_a?: number | null;
    analyst_class_assignment_rate_b?: number | null;
    rare_target_recall_a: number;
    rare_target_recall_b: number;
    recall_new_class_a: number;
    recall_new_class_b: number;
    // newer field — fraction of held-out true-civil chips wrongly assigned to an analyst-created class, per version
    civil_false_assignment_rate_a?: number | null;
    civil_false_assignment_rate_b?: number | null;
    miss_rate_civil_a: number;
    miss_rate_civil_b: number;
    n_eval_a: number;
    n_eval_b: number;
    n_eval_civil_a?: number | null; // newer field
    n_eval_civil_b?: number | null; // newer field
    n_excluded_used_in_fit_a?: number | null; // newer field
    n_excluded_used_in_fit_b?: number | null; // newer field
    n_excluded_same_parent_a?: number | null; // newer field
    n_excluded_same_parent_b?: number | null; // newer field
    exclusion_method?: string | null; // describes what's excluded from the held-out eval pool (e.g. whether parent images are excluded too)
    adaptation_rule?: string | null; // newer field — the ART-gate adaptation rule, verbatim
  };
  novelty_rate_by_batch: { batch_id: BatchId; a: number | null; b: number | null }[];
  forgetting: number;
  // All-time, NOT per version a/b — see backend/API.md.
  analyst_disagreement?: AnalystDisagreement | null;
  // Field-mode compare block (det chips) — absent/null until the backend adds it.
  field?: FieldCompare | null;
};

export type MapPoint = {
  parent_id: number;
  lat: number;
  lon: number;
  airfield: string;
  country: string;
  domain: BatchId;
  n_chips: number;
  novelty_rate: number;
  drift_score: number | null;
};
export type MapResponse = { points: MapPoint[] };

export type ApiError = { detail: string };

// ==================== Field mode (analyst view) ====================
// Collections of new, unlabeled imagery grouped by airfield + date, and the
// detector-chip ("det") review/adapt/compare surface. No ground truth is
// ever present in these payloads — see backend/API.md's field-mode contract.
// Backend endpoints are being built concurrently by two agents; every type
// here is deliberately defensive (optional fields, tolerant unions) so the
// UI degrades gracefully rather than crashing while the real backend lands.

export type ReviewSource = 'det' | 'gt';

export type CollectionStatus = 'needs_review' | 'nominal' | 'insufficient_data';

export type CollectionGroup = {
  group_id: string;
  airfield: string;
  country: string;
  state?: string | null;
  date: string; // YYYY-MM-DD, collection date (UTC day) — grouped by airfield x date
  weather: string;
  off_nadir: number | null; // mean over this group's distinct tiles
  gsd: number | null; // mean over this group's distinct tiles
  lat: number | null;
  lon: number | null;
  n_parents: number;
  n_tiles: number;
  n_detections: number;
  n_flagged: number;
  flag_rate: number | null;
  n_reviewed: number;
  n_bad_box: number; // of n_reviewed, how many are action="bad_box" (localization error)
  status: CollectionStatus;
  // Newer backend (binomial status test): the one-sided p-value of this
  // group's flagged count against the validation-set flag rate, and the
  // excess flagged count over that rate's expectation. Both optional until
  // the backend lane lands them — shown muted and labeled "indicative".
  p_value?: number | null;
  excess_flagged?: number | null;
};

export type CollectionsResponse = {
  groups: CollectionGroup[];
  grouping: string; // e.g. "airfield and date"
  min_denominator: number;
  reference: { n_parents: number; n_tiles: number; description: string };
  reference_flag_rate: number | null; // flag rate on the labeled validation set (reference domain)
  status_rule: string; // free-text describing needs_review/nominal/insufficient_data
};

export type CollectionsSummary = {
  n_groups: number;
  n_needs_review: number;
  n_detections: number;
  n_flagged: number;
  reference_flag_rate: number | null;
};

export type DetChipMeta = { airfield: string; country: string; weather: string; off_nadir: number | null; gsd: number | null };

// Cropped from the detector's own boxes (not ground-truth annotations).
// `ann_id` is an alias of `det_id` kept so shared components (ChipCard,
// ChipGrid, ReviewActions) that key off `ann_id` work unchanged for det
// chips too. Never carries role_true / matched_ann_id / a "not detected" flag.
export type DetChip = {
  source: 'det';
  det_id: number;
  ann_id: number; // == det_id
  tile_id: number;
  bbox: [number, number, number, number];
  conf: number;
  chip_url: string;
  tile_url: string;
  // Nullable: a det chip with no score yet (left-joined onto the det chip
  // index) still counts toward a collection's n_detections, just not
  // n_flagged — see backend/API.md's field-mode Data model note.
  role_pred: string | null;
  role_conf: number | null;
  novelty: number | null;
  is_novel: boolean | null;
  knn_dist: number | null;
  meta: DetChipMeta;
  review: Review | null;
  // Which collection (airfield x date group) this det chip belongs to —
  // carried by `/review/queue?source=det` items (newer backend) and stamped
  // client-side by the fallback queue builder; absent on `/det/{id}`.
  group_id?: string | null;
  // `strategy=collection` only: the owning collection's own facts, stamped
  // flat onto each queue item (backend/app/collections.py's flagged_det_queue)
  // so the UI can render a section header without a second /collections
  // round-trip. Absent for other strategies, or on older backends — the UI
  // falls back to looking `group_id` up in the /collections list.
  airfield?: string | null;
  date?: string | null;
  group_status?: CollectionStatus | null;
  group_n_flagged?: number | null;
};

export type CollectionTile = { tile_id: number; n_detections: number; n_flagged: number };

// backend/API.md: `CollectionGroup & { tiles, chips: {total, items} }` — chips
// nested (not flat total/items) confirmed against the running backend.
export type CollectionDetail = CollectionGroup & {
  tiles: CollectionTile[];
  chips: { total: number; items: DetChip[] };
};

// Reference (labeled validation set) chip a det chip's embedding is nearest
// to — reference labels are legitimate here (they're the labeled control
// set), unlike anything derived from the new collections themselves. `dist`
// is optional so NeighborStrip degrades gracefully if a backend omits it.
export type DetChipNeighbor = { ann_id: number; chip_url: string; role_true: string | null; dist?: number };

export type DetChipDetail = DetChip & { neighbors: DetChipNeighbor[] };

// ---- /versions/compare's field block ----
// Two shapes are tolerated (see backend/API.md, which the backend lane is
// finalizing): (old) `field.a = {field: FieldBlock, control: ControlBlock}`
// with `control.per_class` as a Record and `field.by_domain` (never shown:
// domain names are benchmark vocabulary); (new) `field.a = FieldBlock &
// {control}` with `control.per_class` as an array and per-class `by_group`.
// lib/fieldCompare.ts normalizes both into one view model.

export type FieldByGroup =
  | Record<string, number>
  | { group_id?: string; airfield?: string; date?: string; n?: number; n_assigned?: number }[]
  | null;

export type FieldClassRow = {
  label: string;
  n_labels_used?: number | null;
  n_assigned?: number | null;
  assignment_rate?: number | null;
  by_group?: FieldByGroup;
};

export type FieldControlPerClassEntry = { cls?: string; label?: string; accuracy?: number | null; n?: number | null };

export type FieldControlBlock = {
  label?: string;
  overall?: number | null;
  accuracy_ref_holdout?: number | null; // old name for `overall`
  per_class?: FieldControlPerClassEntry[] | Record<string, { accuracy?: number | null; n?: number | null }> | null;
} | null;

// Field estimate of detector precision/localization error on FLAGGED (is_novel) det chips, from
// det-chip reviews of any action — see backend/API.md's FieldBlock.reviewed_flagged. Always present
// on a current backend (an all-zero/null default if this version's eval artifact predates the key).
export type FieldReviewedFlagged = {
  n: number; // det-chip reviews (source='det', any action) on chips that WERE flagged
  n_not_aircraft: number; // of those, action="not_aircraft" (false positive)
  n_bad_box: number; // of those, action="bad_box" (localization error)
  field_precision_on_flagged: number | null; // (n - n_not_aircraft - n_bad_box) / n; null if n == 0
};

// DISPLAY ONLY (never folded into `classes`/n_assigned/assignment_rate above) — one entry per
// airfield x date collection containing >=1 analyst-labeled ('new_class') det chip. See
// backend/API.md's FieldBlock.labeled_collections.
export type FieldLabeledCollection = {
  group_id: string;
  airfield: string;
  date: string;
  n_labeled: number;
  n_bad_box: number;
  n_not_aircraft: number;
  n_flagged_unlabeled: number;
  n_flagged_unlabeled_assigned: number;
  n_bad_box_assigned?: number;
  per_class: Record<string, number>;
};

export type FieldBlock = {
  n_detections?: number | null;
  n_excluded_labeled?: number | null;
  n_excluded_same_parent?: number | null;
  n_eligible?: number | null;
  classes?: FieldClassRow[] | null;
  control?: FieldControlBlock;
  reviewed_flagged?: FieldReviewedFlagged | null;
  labeled_collections?: FieldLabeledCollection[] | null;
  assign_min_conf?: number | null; // this version's POST /adapt gate (0/null = off) — see backend/API.md
  note?: string | null;
  // old shape only — benchmark domain names; deliberately never rendered
  domains?: string[];
  by_domain?: unknown;
};

// old shape: a version's entry wraps the block; new shape: the entry IS the block
export type FieldVersionEntry = (FieldBlock & { field?: FieldBlock | null }) | { field: FieldBlock | null; control?: FieldControlBlock } | null;

export type FieldWorstRegression = { cls?: string; label?: string; delta?: number | null; a?: number | null; b?: number | null } | null;

export type FieldCompare = {
  a: FieldVersionEntry;
  b: FieldVersionEntry;
  worst_regression?: FieldWorstRegression;
};

// ==================== "Changed in this version" (analyst view) ====================
// GET /api/versions/changes* — detector chips whose predicted role differs between two versions'
// own det-chip scores (an /adapt re-sort, usually into a newly-created analyst class), for
// spot-check review. `source` is currently always 'det' — GT chips aren't reviewed this way.

export type ChangesSort = 'sim' | 'novelty' | 'conf';

// Same shape as DetChip (source/det_id/ann_id/bbox/conf/chip_url/tile_url/role_pred/role_conf/
// novelty/is_novel/knn_dist/meta/review/group_id/airfield/date), plus the before/after role labels,
// the analyst-class similarity that drove the re-sort, and whether this chip shares a parent image
// with a labeled (used_in_fit) chip.
export type ChangedDetChip = DetChip & {
  role_pred_a: string | null; // this chip's role_pred in version `a`
  role_pred_b: string | null; // this chip's role_pred in version `b` (== `role_pred` above)
  analyst_sim: number | null; // cosine similarity to the nearest analyst class mean, from `b`
  // true if this chip's parent image also produced a chip used to fit an analyst class in `b` —
  // visually/positionally correlated with that label, so the UI mutes it (never excluded).
  same_parent: boolean;
};

export type ChangesByGroup = { group_id: string; airfield: string; date: string; n_changed: number };
export type ChangesByClass = { label: string; n: number };

export type ChangesResponse = {
  a: string;
  b: string;
  total: number;
  by_group: ChangesByGroup[]; // facets over the full (only_unreviewed-filtered) changed set
  by_class: ChangesByClass[];
  items: ChangedDetChip[];
};

// A review counts here only if it was recorded while `b` was the current version — see
// backend/API.md; `verified_precision` = n_confirmed / (n_confirmed + n_corrected + n_not_aircraft +
// n_bad_box), null if that denominator is 0 (nothing reviewed yet).
export type ChangesSummary = {
  total: number;
  n_reviewed: number;
  n_confirmed: number;
  n_corrected: number;
  n_not_aircraft: number;
  n_bad_box: number;
  verified_precision: number | null;
};

// ---- Validation set / reference (GET /api/reference; analyst view landing page) ----
// The ONLY labeled data in the system (ref_train ∪ ref_calib ∪ ref_holdout — see
// backend/API.md's vocabulary section). Ground-truth class/role information is
// legitimate here (unlike /collections*, which never carries it) — this endpoint IS
// the labeled set.
export type ReferenceExample = { ann_id: number; chip_url: string };
export type ReferenceClass = { label: string; n_chips: number; examples: ReferenceExample[] };
export type ReferenceRange = { min: number | null; max: number | null };
export type ReferenceDateRange = { min: string | null; max: string | null };
export type ReferenceConditions = {
  weather: Record<string, number>;
  countries: Record<string, number>;
  off_nadir: ReferenceRange;
  gsd: ReferenceRange;
  dates: ReferenceDateRange;
};
export type ReferenceExampleTile = { tile_id: number; tile_url: string; airfield: string; n_aircraft: number };
export type ReferenceDetector = { name: string | null; val_map50: number | null; val_map50_95: number | null; trained_on: string | null };
export type ReferenceResponse = {
  name: string;
  n_images: number;
  n_tiles: number;
  n_chips: number;
  military_aircraft: number;
  splits: { fit: number; control: number };
  classes: ReferenceClass[];
  conditions: ReferenceConditions;
  example_tiles: ReferenceExampleTile[];
  detector: ReferenceDetector;
  flag_rate: number | null;
  definition: string | null;
};

// ---- Model health · retrain readiness (GET /api/health; analyst view) ----
// Indicative only: computed from field-available evidence plus the labeled control set. Every
// threshold is stated verbatim in `rule`/`rules` by the backend (backend/app/health.py) — the UI
// shows those strings, never restates them.
export type HealthStatus = 'nominal' | 'watch' | 'retrain';
export type HealthRecommendation = 'keep' | 'patch' | 'schedule_retrain';
export type HealthIndicator = {
  key: 'novelty_pressure' | 'new_class_demand' | 'detector_quality_field' | 'patch_safety' | string;
  label: string;
  value: number | null;
  display: string;
  status: HealthStatus;
  rule: string;
  evidence: string;
};
export type HealthResponse = {
  version: string | null;
  recommendation: HealthRecommendation;
  reasons: string[];
  indicators: HealthIndicator[];
  rules: string[];
  note: string;
};
