// Synthetic dataset for the Vigil mock API. Everything the endpoints return
// is derived from this generated population (chip-by-chip), not hand-typed
// constants, so the numbers stay internally consistent the way the real
// backend's would be (drift/novelty rise across the stream, accuracy is a
// real mean over generated role_pred/role_true pairs, etc).
import { mulberry32, gaussian, clamp01, pick, pickWeighted, randRange } from './rng.mjs';

const rng = mulberry32(20260905);

export const CIVIL_ROLES = [
  'Small Civil Transport/Utility',
  'Medium Civil Transport/Utility',
  'Large Civil Transport/Utility',
];
export const MILITARY_ROLES = [
  'Military Fighter/Interceptor/Attack',
  'Military Bomber',
  'Military Transport/Cargo',
  'Military Trainer',
];
const ALL_ROLES = [...CIVIL_ROLES, ...MILITARY_ROLES];

const WINGSPAN = {
  'Small Civil Transport/Utility': [6, 15],
  'Medium Civil Transport/Utility': [15, 30],
  'Large Civil Transport/Utility': [30, 68],
  'Military Fighter/Interceptor/Attack': [9, 17],
  'Military Bomber': [30, 56],
  'Military Transport/Cargo': [30, 65],
  'Military Trainer': [8, 13],
};

const AIRFIELDS = {
  USA: [
    ['Davis-Monthan AFB', 32.1665, -110.883],
    ['March ARB', 33.9008, -117.258],
    ['Offutt AFB', 41.1183, -95.9128],
    ['Travis AFB', 38.2626, -121.9375],
    ['Wright-Patterson AFB', 39.826, -84.0483],
    ['San Diego Intl', 32.7338, -117.1933],
  ],
  Germany: [['Ramstein AB', 49.4369, 7.6003]],
  'Saudi Arabia': [['King Khalid Intl', 24.9576, 46.6988]],
  UAE: [['Al Dhafra AB', 24.2481, 54.5479]],
  Japan: [['Kadena AB', 26.3556, 127.7686]],
};

const THETA = 0.62; // synthetic vigilance threshold on the mock's continuous novelty score

const BATCH_DEFS = [
  {
    id: 'ref_holdout',
    order: 0,
    name: 'Control (same domain)',
    description: 'Held-out reference-domain tiles, never used for fitting or calibration.',
    weather: ['Clear'],
    offNadir: [2, 24],
    countries: ['USA'],
    gsd: [0.26, 0.36],
    militaryFrac: 0,
    nChips: 150,
    noveltyBaseMean: 0.36,
    novelCount: 3,
  },
  {
    id: 'haze',
    order: 1,
    name: 'Haze / cloud',
    description: 'Tiles flagged Haze or Cloud in the RarePlanes weather metadata.',
    weather: ['Haze'],
    offNadir: [2, 24],
    countries: ['USA'],
    gsd: [0.26, 0.36],
    militaryFrac: 0,
    nChips: 140,
    noveltyBaseMean: 0.42,
    novelCount: 6,
  },
  {
    id: 'snow',
    order: 2,
    name: 'Snow',
    description: 'Tiles flagged Snow in the RarePlanes weather metadata.',
    weather: ['Snow'],
    offNadir: [2, 24],
    countries: ['USA'],
    gsd: [0.26, 0.36],
    militaryFrac: 0,
    nChips: 130,
    noveltyBaseMean: 0.46,
    novelCount: 8,
  },
  {
    id: 'off_nadir',
    order: 3,
    name: 'High off-nadir (>25°)',
    description: 'Tiles captured beyond the reference off-nadir envelope (25 deg).',
    weather: ['Clear'],
    offNadir: [26, 48],
    countries: ['USA'],
    gsd: [0.3, 0.48],
    militaryFrac: 0,
    nChips: 150,
    noveltyBaseMean: 0.5,
    novelCount: 14,
  },
  {
    id: 'foreign',
    order: 4,
    name: 'New countries',
    description: 'Tiles from countries absent from the reference domain.',
    weather: ['Clear', 'Haze'],
    offNadir: [2, 24],
    countries: ['Germany', 'Saudi Arabia', 'UAE', 'Japan'],
    gsd: [0.26, 0.4],
    militaryFrac: 0,
    nChips: 160,
    noveltyBaseMean: 0.54,
    novelCount: 18,
  },
  {
    id: 'military',
    order: 5,
    name: 'Military aircraft present',
    description: 'Tiles containing military aircraft roles held out of training entirely.',
    weather: ['Clear', 'Haze'],
    offNadir: [2, 30],
    countries: ['USA', 'Germany'],
    gsd: [0.26, 0.4],
    militaryFrac: 0.16,
    nChips: 170,
    noveltyBaseMean: 0.56,
    novelCount: 20,
  },
];

let annSeq = 1;
let tileSeq = 1;

const tiles = new Map(); // tile_id -> {tile_id, file_name, meta, chipIds: number[]}
const chips = new Map(); // ann_id -> chip record (mutable; POST /review and /adapt mutate in place)
const referenceIds = [];
const batchChipIds = new Map(); // batch_id -> ann_id[]

function makeTile(country, weatherOptions, offNadirRange, gsdRange, airfieldName) {
  const tile_id = tileSeq++;
  const weather = pick(rng, weatherOptions);
  const pool = AIRFIELDS[country] || AIRFIELDS.USA;
  // Field-mode collections pin a specific airfield (so a collection's
  // displayed airfield/lat/lon always matches its own tiles); GT batches
  // never pass airfieldName and keep the original random-within-country pick.
  const forced = airfieldName ? pool.find(([name]) => name === airfieldName) : undefined;
  const [af_name, lat0, lon0] = forced || pick(rng, pool);
  const meta = {
    airfield: af_name,
    country,
    weather,
    off_nadir: Math.round(randRange(rng, offNadirRange[0], offNadirRange[1]) * 10) / 10,
    gsd: Math.round(randRange(rng, gsdRange[0], gsdRange[1]) * 1000) / 1000,
    lat: lat0 + randRange(rng, -0.5, 0.5),
    lon: lon0 + randRange(rng, -0.5, 0.5),
  };
  tiles.set(tile_id, {
    tile_id,
    file_name: `tile_${String(tile_id).padStart(5, '0')}.png`,
    meta,
    chipIds: [],
  });
  return tile_id;
}

function randomBBox() {
  const w = Math.round(randRange(rng, 28, 130));
  const h = Math.round(randRange(rng, 28, 130));
  const x = Math.round(randRange(rng, 4, 512 - w - 4));
  const y = Math.round(randRange(rng, 4, 512 - h - 4));
  return [x, y, w, h];
}

function roleClusterCenter(role) {
  // fixed 2D "embedding" centers so the scatter plot shows real clustering;
  // civil roles sit close together, military roles sit further out.
  const centers = {
    'Small Civil Transport/Utility': [-3, -2],
    'Medium Civil Transport/Utility': [-1, 1],
    'Large Civil Transport/Utility': [1, -1],
    'Military Fighter/Interceptor/Attack': [7, 4],
    'Military Bomber': [8, -3],
    'Military Transport/Cargo': [5, 6],
    'Military Trainer': [6, 1],
  };
  return centers[role] || [0, 0];
}

function otherCivilRole(role) {
  const rest = CIVIL_ROLES.filter((r) => r !== role);
  return pick(rng, rest);
}

function makeChip({ batch_id, tile_id, roleTrue, noveltyScore, misclassProb = 0 }) {
  const ann_id = annSeq++;
  const tile = tiles.get(tile_id);
  const isMilitary = MILITARY_ROLES.includes(roleTrue);
  const misclassified = !isMilitary && rng() < misclassProb;
  const rolePred = isMilitary ? pick(rng, CIVIL_ROLES) : misclassified ? otherCivilRole(roleTrue) : roleTrue;
  const roleConfBase = isMilitary || misclassified ? randRange(rng, 0.3, 0.6) : randRange(rng, 0.72, 0.98);
  const [cx, cy] = roleClusterCenter(roleTrue);
  const scatter = 0.6 + noveltyScore * 2.4;
  const wsRange = WINGSPAN[roleTrue];
  const chip = {
    ann_id,
    tile_id,
    batch_id,
    version: 'v1',
    bbox: randomBBox(),
    chip_url: `/api/chips/${ann_id}/image`,
    tile_url: `/api/tiles/${tile_id}/image`,
    role_pred: rolePred,
    role_conf: Math.round(roleConfBase * 1000) / 1000,
    novelty: Math.round(clamp01(noveltyScore) * 1000) / 1000,
    is_novel: false, // finalized per-batch after ranking, see finalizeBatchNovelty()
    art_category: null,
    knn_dist: Math.round(clamp01(noveltyScore * 0.55 + randRange(rng, -0.12, 0.28)) * 1000) / 1000,
    used_in_fit: false, // set true only for the exact exemplar chip(s) an analyst used to fit a new prototype
    umap: [
      Math.round((cx + gaussian(rng) * scatter) * 100) / 100,
      Math.round((cy + gaussian(rng) * scatter) * 100) / 100,
    ],
    meta: {
      airfield: tile.meta.airfield,
      country: tile.meta.country,
      weather: tile.meta.weather,
      off_nadir: tile.meta.off_nadir,
      gsd: tile.meta.gsd,
      wingspan_m: wsRange ? Math.round(randRange(rng, wsRange[0], wsRange[1]) * 10) / 10 : null,
    },
    role_true: roleTrue,
    review: null,
  };
  tile.chipIds.push(ann_id);
  chips.set(ann_id, chip);
  return chip;
}

function finalizeBatchNovelty(batchId, targetNovelCount) {
  const ids = batchChipIds.get(batchId);
  const ranked = [...ids].sort((a, b) => chips.get(b).novelty - chips.get(a).novelty);
  const novelSet = new Set(ranked.slice(0, targetNovelCount));
  for (const id of ids) {
    const c = chips.get(id);
    c.is_novel = novelSet.has(id);
    if (!c.is_novel) {
      c.art_category = (CIVIL_ROLES.indexOf(c.role_pred) % 3) + 1;
    } else {
      c.art_category = null;
      c.novelty = Math.max(c.novelty, THETA + randRange(rng, 0.01, 0.1));
    }
  }
}

// ---- build reference pool (n_ref_chips ~= 3,912) --------------------------
const N_REF_CHIPS = 3912;
const N_REF_TILES = 1284;
const refTileIds = [];
for (let i = 0; i < N_REF_TILES; i++) {
  refTileIds.push(makeTile('USA', ['Clear'], [2, 24], [0.26, 0.36]));
}
for (let i = 0; i < N_REF_CHIPS; i++) {
  const tile_id = pick(rng, refTileIds);
  const roleTrue = pickWeighted(rng, CIVIL_ROLES, [0.48, 0.33, 0.19]);
  const noveltyScore = clamp01(0.1 + Math.abs(gaussian(rng)) * 0.08);
  const c = makeChip({ batch_id: 'reference', tile_id, roleTrue, noveltyScore, misclassProb: 0.03 });
  c.is_novel = c.novelty > THETA; // ~never, by construction
  c.art_category = (CIVIL_ROLES.indexOf(c.role_pred) % 3) + 1;
  referenceIds.push(c.ann_id);
}

// ---- build operational batches --------------------------------------------
for (const def of BATCH_DEFS) {
  const ids = [];
  batchChipIds.set(def.id, ids);
  const chipsPerTile = 3;
  const nTiles = Math.max(1, Math.ceil(def.nChips / chipsPerTile));
  const tileIds = [];
  for (let i = 0; i < nTiles; i++) {
    const country = pick(rng, def.countries);
    tileIds.push(makeTile(country, def.weather, def.offNadir, def.gsd));
  }
  const nMilitary = Math.round(def.nChips * def.militaryFrac);
  const misclassProb = 0.03 + def.order * 0.025;
  for (let i = 0; i < def.nChips; i++) {
    const tile_id = pick(rng, tileIds);
    const isMilitary = i < nMilitary;
    const roleTrue = isMilitary ? pick(rng, MILITARY_ROLES) : pickWeighted(rng, CIVIL_ROLES, [0.48, 0.33, 0.19]);
    const noise = gaussian(rng) * 0.15;
    const noveltyScore = isMilitary
      ? clamp01(0.8 + Math.abs(gaussian(rng)) * 0.12)
      : clamp01(def.noveltyBaseMean + noise);
    const c = makeChip({ batch_id: def.id, tile_id, roleTrue, noveltyScore, misclassProb });
    ids.push(c.ann_id);
  }
  finalizeBatchNovelty(def.id, def.novelCount);
}

// ---- aggregates -------------------------------------------------------------
function batchChips(batchId) {
  if (batchId === 'reference') return referenceIds.map((id) => chips.get(id));
  return (batchChipIds.get(batchId) || []).map((id) => chips.get(id));
}

function mean(arr) {
  return arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;
}

function r3(x) {
  return Math.round(x * 1000) / 1000;
}

// "not_aircraft" chips are dropped from scoring; used_in_fit chips are
// excluded so the model isn't graded on the exemplars it just adapted to.
function isScorable(c) {
  return !c.review || c.review.action !== 'not_aircraft';
}
function evalPool(list) {
  return list.filter((c) => isScorable(c) && !c.used_in_fit);
}

const NULL_METHOD = 'tile-level block bootstrap over ref_calib (chips within a tile are correlated; p-value is indicative, not inferential)';
const LOCALIZATION_NOTE = 'ground-truth aircraft boxes (oracle localization); every metric here is on GT chips, never a detector’s own output, until the detector milestone lands';
const WINGSPAN_STRATIFIED_METHOD =
  'support-weighted mean of within-wingspan-quartile AUROCs (quartiles from civil chips); military positives come only from the military batch, so acquisition-domain confounding is not removed';

function computeMilitaryMetrics() {
  const militaryAll = [...batchChipIds.values()]
    .flat()
    .map((id) => chips.get(id))
    .filter((c) => MILITARY_ROLES.includes(c.role_true));
  const held = evalPool(militaryAll);
  const recognized = held.filter((c) => !CIVIL_ROLES.includes(c.role_pred)).length;
  const rare_target_recall = held.length ? r3(recognized / held.length) : 0;
  return {
    rare_target_recall,
    recall_new_class: rare_target_recall,
    miss_rate_civil: held.length ? r3(1 - rare_target_recall) : 1,
    n_eval: held.length,
  };
}

function roleAccuracy(chipList) {
  const known = chipList.filter((c) => CIVIL_ROLES.includes(c.role_true));
  if (!known.length) return null;
  return Math.round(mean(known.map((c) => (c.role_pred === c.role_true ? 1 : 0))) * 1000) / 1000;
}

const NULL_P95 = 0.32;
const NULL_SIGMA = 0.04;
const NULL_MEAN = NULL_P95 - 1.645 * NULL_SIGMA; // null_p95 is the 95th pctile of 500 bootstraps of ref_calib
const normCDF = (x) => 1 / (1 + Math.exp(-1.702 * x)); // logistic approximation to the standard normal CDF

function driftFor(def, chipList) {
  const score = Math.round((mean(chipList.map((c) => c.knn_dist)) + 0.02) * 1000) / 1000;
  const z = (score - NULL_MEAN) / NULL_SIGMA;
  const p_value = Math.max(0.002, Math.round((1 - normCDF(z)) * 1000) / 1000);
  const calib_mean = NULL_MEAN;
  return {
    score,
    null_p95: NULL_P95,
    p_value,
    n_boot: 500,
    statistic: 'mean 5-NN cosine distance to reference',
    null_method: NULL_METHOD,
    calib_mean: r3(calib_mean),
    ratio_to_calib: r3(score / calib_mean),
    level: 'scene',
  };
}

function batchSummary(def) {
  const list = batchChips(def.id);
  const nTilesActual = new Set(list.map((c) => c.tile_id)).size;
  const summary = {
    id: def.id,
    order: def.order,
    name: def.name,
    description: def.description,
    condition: {
      weather: def.weather.join('/'),
      countries: def.countries.join(', '),
      off_nadir: `${def.offNadir[0]}–${def.offNadir[1]}°`,
    },
    n_tiles: nTilesActual,
    n_chips: list.length,
    n_parent_images: Math.max(1, Math.round(nTilesActual * 0.62)),
    drift: driftFor(def, list),
    novelty_rate: Math.round(mean(list.map((c) => (c.is_novel ? 1 : 0))) * 1000) / 1000,
    role_accuracy: roleAccuracy(list),
    detection: null,
    conf_shift: {
      ref_mean: 0.91,
      batch_mean: Math.round(mean(list.map((c) => c.role_conf)) * 1000) / 1000,
      ks_p: Math.round(clamp01(0.5 - def.order * 0.08 + randRange(rng, -0.02, 0.02)) * 1000) / 1000,
    },
  };
  // military-batch-only fields: flag rate split by ground-truth role group
  // (ground truth, evaluation only — never fed back into scoring).
  if (def.id === 'military') {
    const militaryChips = list.filter((c) => MILITARY_ROLES.includes(c.role_true));
    const civilChips = list.filter((c) => CIVIL_ROLES.includes(c.role_true));
    summary.novelty_rate_military = militaryChips.length ? r3(mean(militaryChips.map((c) => (c.is_novel ? 1 : 0)))) : null;
    summary.novelty_rate_civil = civilChips.length ? r3(mean(civilChips.map((c) => (c.is_novel ? 1 : 0)))) : null;
  }
  return summary;
}

function currentBatches() {
  return BATCH_DEFS.map(batchSummary);
}

// ---- versions ---------------------------------------------------------------
export const versions = [
  {
    id: 'v1',
    created: new Date('2026-09-01T09:00:00Z').toISOString(),
    classes: [...CIVIL_ROLES],
    n_feedback: 0,
    n_confirm: 0,
    n_new_class: 0,
    n_not_aircraft: 0,
    note: 'Initial prototype version (heads fit on reference domain).',
  },
];
export let currentVersion = 'v1';

const snapshots = new Map(); // versionId -> { batches: Batch[] (deep), refHoldoutAcc, perClass, military }

function takeSnapshot(versionId) {
  const batches = currentBatches();
  const refHoldout = evalPool(batchChips('ref_holdout'));
  const perClass = CIVIL_ROLES.map((cls) => {
    const subset = refHoldout.filter((c) => c.role_true === cls);
    return { cls, acc: roleAccuracy(subset) ?? 0, n: subset.length };
  });
  snapshots.set(versionId, {
    batches,
    refHoldoutAcc: roleAccuracy(refHoldout) ?? 0,
    perClass,
    military: computeMilitaryMetrics(),
    field: takeFieldSnapshot(versionId),
  });
}
const reviews = new Map(); // ann_id -> Review
const usedLabelsOrder = []; // most-recently-used first

// ---- AUROC / AUPRC ------------------------------------------------------------
function auroc(scores, labels) {
  const pairs = scores.map((s, i) => [s, labels[i]]).sort((a, b) => a[0] - b[0]);
  const nPos = labels.filter(Boolean).length;
  const nNeg = labels.length - nPos;
  if (!nPos || !nNeg) return null;
  let rankSum = 0;
  let rank = 1;
  let i = 0;
  while (i < pairs.length) {
    let j = i;
    while (j < pairs.length && pairs[j][0] === pairs[i][0]) j++;
    const avgRank = (rank + (rank + (j - i) - 1)) / 2;
    for (let k = i; k < j; k++) if (pairs[k][1]) rankSum += avgRank;
    rank += j - i;
    i = j;
  }
  const auc = (rankSum - (nPos * (nPos + 1)) / 2) / (nPos * nNeg);
  return Math.round(auc * 1000) / 1000;
}

function auprc(scores, labels) {
  const pairs = scores.map((s, i) => [s, labels[i]]).sort((a, b) => b[0] - a[0]);
  const nPos = labels.filter(Boolean).length;
  if (!nPos) return null;
  let tp = 0;
  let sum = 0;
  for (let i = 0; i < pairs.length; i++) {
    if (pairs[i][1]) {
      tp++;
      sum += tp / (i + 1);
    }
  }
  return Math.round((sum / nPos) * 1000) / 1000;
}

function allChipsList() {
  return [...chips.values()];
}

// ================= public API used by server.mjs =============================

export function getSummary() {
  const batches = currentBatches();
  const totals = {
    chips: allChipsList().filter((c) => c.batch_id !== 'reference').length,
    novel: allChipsList().filter((c) => c.batch_id !== 'reference' && c.is_novel).length,
    reviewed: reviews.size,
  };
  return {
    model: {
      name: 'Vigil',
      detector: 'YOLO11n (RarePlanes real/aircraft, reference domain)',
      embedder: 'DINOv2 ViT-S/14',
      novelty_method: 'Fuzzy ART vigilance-gated resonance over L2-normalized chip embeddings',
      vigilance: THETA,
      reference_domain: 'clear skies, USA, off-nadir ≤ 25°, no military aircraft',
      n_ref_chips: N_REF_CHIPS,
      n_ref_images: N_REF_TILES,
      calibrated_on: 'ref_calib (600 chips held out from the reference domain)',
      localization: LOCALIZATION_NOTE,
      calibration_check: 'Vigilance re-checked on a second, disjoint held-out split of the reference domain; flag rate stayed within 2 points of the ref_calib figure.',
      model_selection_note: 'Prototype/ART hyperparameters were chosen on ref_calib only; ref_holdout was never touched during model selection.',
    },
    versions,
    current_version: currentVersion,
    batches,
    totals,
  };
}

export function getBatches() {
  return currentBatches();
}

export function getBatchDetail(id) {
  const def = BATCH_DEFS.find((b) => b.id === id);
  if (!def) return null;
  const list = batchChips(id);
  const refList = batchChips('reference');
  const bins = 12;
  const hist = (arr) => {
    const counts = new Array(bins).fill(0);
    for (const c of arr) {
      const idx = Math.min(bins - 1, Math.floor(c.novelty * bins));
      counts[idx]++;
    }
    return { bins: Array.from({ length: bins }, (_, i) => Math.round((i / bins) * 100) / 100), counts };
  };
  const topNovel = [...list].sort((a, b) => b.novelty - a.novelty).slice(0, 10);
  return {
    ...batchSummary(def),
    novelty_hist: hist(list),
    ref_novelty_hist: hist(refList),
    top_novel: topNovel,
  };
}

export function getBatchChipsPage(id, query) {
  const list = batchChips(id);
  let filtered = list;
  if (query.only_novel === 'true') filtered = filtered.filter((c) => c.is_novel);
  if (query.reviewed === 'yes') filtered = filtered.filter((c) => reviews.has(c.ann_id));
  if (query.reviewed === 'no') filtered = filtered.filter((c) => !reviews.has(c.ann_id));
  const sortKey = query.sort === 'role_conf' ? 'role_conf' : query.sort === 'knn_dist' ? 'knn_dist' : 'novelty';
  const order = query.order === 'asc' ? 1 : -1;
  filtered = [...filtered].sort((a, b) => (a[sortKey] - b[sortKey]) * order);
  const total = filtered.length;
  const offset = Number(query.offset) || 0;
  const limit = Number(query.limit) || 48;
  const items = filtered.slice(offset, offset + limit).map((c) => ({ ...c, review: reviews.get(c.ann_id) || null }));
  return { total, items };
}

export function getChipDetail(annId) {
  const c = chips.get(Number(annId));
  if (!c) return null;
  const refPool = referenceIds.map((id) => chips.get(id)).filter((r) => r.role_true !== undefined);
  const sameRole = refPool.filter((r) => r.role_true === c.role_pred);
  const pool = sameRole.length >= 5 ? sameRole : refPool;
  const neighbors = [...pool]
    .sort(() => rng() - 0.5)
    .slice(0, 5)
    .map((r, i) => ({ ann_id: r.ann_id, chip_url: r.chip_url, role_true: r.role_true, dist: Math.round((0.04 + i * 0.05 + rng() * 0.03) * 1000) / 1000 }))
    .sort((a, b) => a.dist - b.dist);
  return { ...c, review: reviews.get(c.ann_id) || null, neighbors, art_match: Math.round((1 - c.novelty) * 1000) / 1000 };
}

export function getChipImageInfo(annId) {
  const c = chips.get(Number(annId));
  if (!c) return null;
  return { seed: annId * 2654435761, role: c.role_pred, isNovel: c.is_novel };
}

export function getTileImageInfo(tileId) {
  const t = tiles.get(Number(tileId));
  if (!t) return null;
  return { seed: tileId * 2246822519, weather: t.meta.weather };
}

export function getTileDetail(tileId) {
  const t = tiles.get(Number(tileId));
  if (!t) return null;
  return {
    tile_id: t.tile_id,
    file_name: t.file_name,
    meta: t.meta,
    chips: t.chipIds.map((id) => ({ ...chips.get(id), review: reviews.get(id) || null })),
    detections: tileDetections.get(t.tile_id) ?? null,
  };
}

export function getEmbeddingPoints(batchId) {
  const refSample = referenceIds.slice(0, 1500).map((id) => {
    const c = chips.get(id);
    return { ann_id: c.ann_id, x: c.umap[0], y: c.umap[1], role_true: c.role_true };
  });
  const id = batchId && BATCH_DEFS.some((b) => b.id === batchId) ? batchId : 'ref_holdout';
  const batchPoints = batchChips(id).map((c) => ({
    ann_id: c.ann_id,
    x: c.umap[0],
    y: c.umap[1],
    novelty: c.novelty,
    is_novel: c.is_novel,
    art_category: c.art_category,
  }));
  return { reference: refSample, batch: batchPoints };
}

export function getDrift() {
  const series = BATCH_DEFS.map((def) => {
    const d = batchSummary(def).drift;
    return {
      batch_id: def.id,
      order: def.order,
      name: def.name,
      score: d.score,
      null_p95: d.null_p95,
      p_value: d.p_value,
      n: d.n_boot,
      calib_mean: d.calib_mean,
      ratio_to_calib: d.ratio_to_calib,
    };
  });
  return { statistic: 'mean 5-NN cosine distance to reference', null_method: NULL_METHOD, series, level: 'scene' };
}

function bucketize(list, keyFn, labelOrder) {
  const groups = new Map();
  for (const c of list) {
    const k = keyFn(c);
    if (!groups.has(k)) groups.set(k, []);
    groups.get(k).push(c);
  }
  const labels = labelOrder || [...groups.keys()];
  return labels.filter((l) => groups.has(l)).map((label) => {
    const arr = groups.get(label);
    const nImages = new Set(arr.map((c) => c.tile_id)).size;
    return {
      label,
      n_chips: arr.length,
      n_images: nImages,
      n_parent_images: Math.max(1, Math.round(nImages * 0.62)),
      novelty_rate: Math.round(mean(arr.map((c) => (c.is_novel ? 1 : 0))) * 1000) / 1000,
      role_accuracy: roleAccuracy(arr),
      detection_recall: null,
    };
  });
}

// Status is tested/untested/degraded only — the reference-defining bucket is
// marked separately via `in_reference_definition`, not a "reference" status.
function statusFor(bucket, thresholds) {
  if (bucket.n_chips < thresholds.min_n) return 'untested';
  const accDegraded = bucket.control_delta.role_accuracy != null && bucket.control_delta.role_accuracy < -thresholds.degraded_margin;
  const controlNovelty = thresholds.control.novelty_rate;
  const noveltyDegraded = controlNovelty != null && controlNovelty > 0 && bucket.novelty_rate > thresholds.novelty_multiplier * controlNovelty;
  return accDegraded || noveltyDegraded ? 'degraded' : 'tested';
}

export function getEnvelope() {
  const all = [...referenceIds.map((id) => chips.get(id)), ...[...batchChipIds.values()].flat().map((id) => chips.get(id))];
  const refAcc = roleAccuracy(all.filter((c) => c.meta.weather === 'Clear' && c.meta.country === 'USA' && c.meta.off_nadir <= 25));
  const controlChips = batchChips('ref_holdout');
  const thresholds = {
    min_n: 30,
    degraded_margin: 0.08,
    novelty_multiplier: 3.0,
    control: {
      role_accuracy: roleAccuracy(controlChips),
      novelty_rate: controlChips.length ? r3(mean(controlChips.map((c) => (c.is_novel ? 1 : 0)))) : null,
      detection_recall: null, // detector milestone not simulated in the mock
      n_chips: controlChips.length,
      n_parent_images: Math.max(1, Math.round(new Set(controlChips.map((c) => c.tile_id)).size * 0.62)),
    },
  };

  const dims = [
    {
      name: 'weather',
      buckets: bucketize(all, (c) => c.meta.weather, ['Clear', 'Haze', 'Snow']),
      refTest: (b) => b.label === 'Clear',
    },
    {
      name: 'off_nadir',
      buckets: bucketize(
        all,
        (c) => (c.meta.off_nadir <= 15 ? '≤15°' : c.meta.off_nadir <= 25 ? '15–25°' : '>25°'),
        ['≤15°', '15–25°', '>25°'],
      ),
      refTest: (b) => b.label !== '>25°',
    },
    {
      name: 'country',
      buckets: bucketize(all, (c) => (c.meta.country === 'USA' ? 'USA' : 'Other'), ['USA', 'Other']),
      refTest: (b) => b.label === 'USA',
    },
    {
      name: 'gsd',
      buckets: bucketize(all, (c) => (c.meta.gsd < 0.35 ? '<0.35 m' : '≥0.35 m'), ['<0.35 m', '≥0.35 m']),
      refTest: (b) => b.label === '<0.35 m',
    },
    {
      name: 'aircraft size',
      buckets: bucketize(
        all,
        (c) => (c.meta.wingspan_m == null ? 'unknown' : c.meta.wingspan_m < 15 ? 'Small (<15 m)' : c.meta.wingspan_m < 30 ? 'Medium (15–30 m)' : 'Large (>30 m)'),
        ['Small (<15 m)', 'Medium (15–30 m)', 'Large (>30 m)'],
      ),
      refTest: () => false,
    },
  ].map((d) => ({
    name: d.name,
    buckets: d.buckets.map((b) => {
      const control_delta = {
        role_accuracy: b.role_accuracy != null && refAcc != null ? r3(b.role_accuracy - refAcc) : null,
        detection_recall: null,
      };
      const withDelta = { ...b, control_delta, in_reference_definition: d.refTest(b) };
      return { ...withDelta, status: statusFor(withDelta, thresholds) };
    }),
  }));

  return { dimensions: dims, thresholds };
}

export function getEval() {
  // Every metric here excludes chips with used_in_fit=true (and, like all
  // scoring, chips dropped as not_aircraft) so the model isn't graded on
  // exemplars it just adapted to.
  const all = evalPool(allChipsList().filter((c) => c.batch_id !== 'reference'));
  const scores = all.map((c) => c.novelty);
  const labels = all.map((c) => (MILITARY_ROLES.includes(c.role_true) ? 1 : 0));
  const knnScores = all.map((c) => c.knn_dist);

  // per-role AUROC: for a military role, novelty score vs. every civil chip
  // (does the score pick this held-out role out of the known population?);
  // for a civil role, novelty score vs. the other civil roles (should sit
  // near chance — novelty is not supposed to separate *known* classes).
  const perRole = ALL_ROLES.map((role) => {
    const n = all.filter((c) => c.role_true === role).length;
    const roleScores = [];
    const roleLabels = [];
    for (const c of all) {
      if (MILITARY_ROLES.includes(role)) {
        if (c.role_true === role) {
          roleScores.push(c.novelty);
          roleLabels.push(1);
        } else if (CIVIL_ROLES.includes(c.role_true)) {
          roleScores.push(c.novelty);
          roleLabels.push(0);
        }
      } else if (CIVIL_ROLES.includes(c.role_true)) {
        roleScores.push(c.novelty);
        roleLabels.push(c.role_true === role ? 1 : 0);
      }
    }
    return { role, n, auroc: auroc(roleScores, roleLabels) };
  });

  const medianMilitaryWingspan = 20;
  const sizeMatched = all.filter((c) => Math.abs((c.meta.wingspan_m ?? 20) - medianMilitaryWingspan) < 12);
  const sizeMatchedLabels = sizeMatched.map((c) => (MILITARY_ROLES.includes(c.role_true) ? 1 : 0));

  const refHoldout = evalPool(batchChips('ref_holdout'));
  const perClass = CIVIL_ROLES.map((cls) => {
    const subset = refHoldout.filter((c) => c.role_true === cls);
    return { cls, n: subset.length, accuracy: roleAccuracy(subset) };
  });

  const sizeMatchedAuroc = auroc(sizeMatched.map((c) => c.novelty), sizeMatchedLabels);

  return {
    version: currentVersion,
    novelty: {
      auroc_military_vs_civil: auroc(scores, labels),
      auprc: auprc(scores, labels),
      auroc_size_matched: sizeMatchedAuroc,
      auroc_wingspan_stratified: sizeMatchedAuroc, // same value, kept for compatibility
      wingspan_stratified_method: WINGSPAN_STRATIFIED_METHOD,
      per_role: perRole,
      baseline_knn_auroc: auroc(knnScores, labels),
    },
    role_head: {
      accuracy_ref_holdout: roleAccuracy(refHoldout),
      per_class: perClass,
    },
    military: computeMilitaryMetrics(),
    note: 'ground truth used for evaluation only — never for fitting or calibration',
  };
}

export function getReviewQueue(query) {
  const limit = Number(query.limit) || 24;
  const strategy = query.strategy || 'novelty';
  const unreviewed = [...batchChipIds.values()].flat().map((id) => chips.get(id)).filter((c) => !reviews.has(c.ann_id));
  let ranked;
  if (strategy === 'uncertainty') {
    ranked = [...unreviewed].sort((a, b) => Math.abs(a.role_conf - 0.5) - Math.abs(b.role_conf - 0.5));
  } else if (strategy === 'random') {
    ranked = [...unreviewed].sort(() => rng() - 0.5);
  } else {
    ranked = [...unreviewed].sort((a, b) => b.novelty - a.novelty);
  }
  return ranked.slice(0, limit);
}

export function postReview(annId, body, source) {
  const id = Number(annId);
  if (source === 'det') {
    const c = detChips.get(id);
    if (!c) return null;
    const review = {
      ann_id: c.det_id,
      action: body.action,
      label: body.label ?? null,
      note: body.note ?? null,
      ts: new Date().toISOString(),
      version_applied: null,
    };
    fieldReviews.set(c.det_id, review);
    c.review = review;
    if (review.label) {
      const i = usedLabelsOrder.indexOf(review.label);
      if (i !== -1) usedLabelsOrder.splice(i, 1);
      usedLabelsOrder.unshift(review.label);
    }
    return review;
  }
  const c = chips.get(id);
  if (!c) return null;
  const review = {
    ann_id: c.ann_id,
    action: body.action,
    label: body.label ?? null,
    note: body.note ?? null,
    ts: new Date().toISOString(),
    version_applied: null,
  };
  reviews.set(c.ann_id, review);
  c.review = review;
  if (review.label) {
    const i = usedLabelsOrder.indexOf(review.label);
    if (i !== -1) usedLabelsOrder.splice(i, 1);
    usedLabelsOrder.unshift(review.label);
  }
  return review;
}

export function getReviewLog(source) {
  if (source === 'det') return [...fieldReviews.values()].sort((a, b) => (a.ts < b.ts ? 1 : -1));
  return [...reviews.values()].sort((a, b) => (a.ts < b.ts ? 1 : -1));
}

export function getUsedLabels() {
  return [...usedLabelsOrder];
}

export function getReviewClasses() {
  return { classes: versions[versions.length - 1].classes };
}

export function postAdapt(body) {
  const pending = [...reviews.values()].filter(
    (r) => r.version_applied === null && (r.action === 'confirm' || r.action === 'new_class' || r.action === 'not_aircraft'),
  );
  const newClassGroups = new Map(); // role_true -> { label, exemplarAnnIds }
  for (const r of pending) {
    if (r.action !== 'new_class' || !r.label) continue;
    const chip = chips.get(r.ann_id);
    if (!chip) continue;
    if (!newClassGroups.has(chip.role_true)) newClassGroups.set(chip.role_true, { label: r.label, exemplarAnnIds: [] });
    newClassGroups.get(chip.role_true).exemplarAnnIds.push(r.ann_id);
  }

  const newVersionId = `v${versions.length + 1}`;
  let artCursor = 900 + versions.length;
  const addedClasses = [];
  for (const [roleTrue, { label, exemplarAnnIds }] of newClassGroups) {
    addedClasses.push(label);
    const cat = artCursor++;
    // The prototype generalizes to every chip of this hidden true class, but
    // only the analyst's actual exemplar(s) are used_in_fit (and so excluded
    // from evaluation) — the rest are legitimate generalization test points.
    for (const c of chips.values()) {
      if (c.role_true === roleTrue) {
        c.is_novel = false;
        c.art_category = cat;
        c.novelty = Math.round(randRange(rng, 0.05, 0.16) * 1000) / 1000;
        c.role_pred = label;
        c.role_conf = Math.round(randRange(rng, 0.82, 0.97) * 1000) / 1000;
        if (exemplarAnnIds.includes(c.ann_id)) c.used_in_fit = true;
      }
    }
  }
  let n_confirm = 0;
  let n_new_class = 0;
  let n_not_aircraft = 0;
  for (const r of pending) {
    // `confirm` is recorded as analyst agreement only — it never alters the
    // model (no prototype/ART update, no score change).
    if (r.action === 'confirm') n_confirm++;
    else if (r.action === 'new_class') n_new_class++;
    else if (r.action === 'not_aircraft') n_not_aircraft++;
    r.version_applied = newVersionId;
  }

  // ---- field-mode (det chip) adaptation, same version bump, separate
  // namespace — analyst classes created from det chips only claim det
  // chips, and vice versa; both feed the same version's `classes` list.
  const pendingDet = [...fieldReviews.values()].filter(
    (r) => r.version_applied === null && (r.action === 'confirm' || r.action === 'new_class' || r.action === 'not_aircraft'),
  );
  const detClassGroups = new Map(); // hidden _trueRole -> { label, exemplarDetIds }
  for (const r of pendingDet) {
    if (r.action !== 'new_class' || !r.label) continue;
    const dc = detChips.get(r.ann_id);
    if (!dc) continue;
    if (!detClassGroups.has(dc._trueRole)) detClassGroups.set(dc._trueRole, { label: r.label, exemplarDetIds: [] });
    detClassGroups.get(dc._trueRole).exemplarDetIds.push(r.ann_id);
  }
  for (const [trueRole, { label, exemplarDetIds }] of detClassGroups) {
    if (!addedClasses.includes(label)) addedClasses.push(label);
    if (!fieldClassLabelsEver.includes(label)) fieldClassLabelsEver.push(label);
    for (const c of detChips.values()) {
      if (c._trueRole === trueRole) {
        c.is_novel = false;
        c.novelty = Math.round(randRange(rng, 0.05, 0.16) * 1000) / 1000;
        c.role_pred = label;
        c.role_conf = Math.round(randRange(rng, 0.82, 0.97) * 1000) / 1000;
        if (exemplarDetIds.includes(c.det_id)) c._usedInFit = true;
      }
    }
  }
  let n_confirm_det = 0;
  let n_new_class_det = 0;
  let n_not_aircraft_det = 0;
  for (const r of pendingDet) {
    if (r.action === 'confirm') n_confirm_det++;
    else if (r.action === 'new_class') n_new_class_det++;
    else if (r.action === 'not_aircraft') n_not_aircraft_det++;
    r.version_applied = newVersionId;
  }

  const prevClasses = versions[versions.length - 1].classes;
  const baseNote = body?.note ? `${body.note} — ` : '';
  const version = {
    id: newVersionId,
    created: new Date().toISOString(),
    classes: [...new Set([...prevClasses, ...addedClasses])],
    n_feedback: pending.length + pendingDet.length,
    n_confirm: n_confirm + n_confirm_det,
    n_new_class: n_new_class + n_new_class_det,
    n_not_aircraft: n_not_aircraft + n_not_aircraft_det,
    note: `${baseNote}confirm actions are recorded as agreement and do not alter the model`,
  };
  versions.push(version);
  currentVersion = newVersionId;
  for (const c of chips.values()) c.version = newVersionId;
  for (const c of detChips.values()) c.version = newVersionId;
  takeSnapshot(newVersionId);
  return version;
}

export function getVersionsCompare(a, b) {
  const snapA = snapshots.get(a);
  const snapB = snapshots.get(b);
  if (!snapA || !snapB) return null;
  const forgetting = Math.max(0, Math.round((snapA.refHoldoutAcc - snapB.refHoldoutAcc) * 1000) / 1000);
  const fieldPerClass = CIVIL_ROLES.map((cls) => ({
    cls,
    a: snapA.perClass.find((p) => p.cls === cls)?.acc ?? null,
    b: snapB.perClass.find((p) => p.cls === cls)?.acc ?? null,
  }));
  let worst_regression = null;
  for (const row of fieldPerClass) {
    if (row.a != null && row.b != null) {
      const delta = r3(row.b - row.a);
      if (!worst_regression || delta < worst_regression.delta) worst_regression = { cls: row.cls, delta, a: row.a, b: row.b };
    }
  }
  return {
    a,
    b,
    field: {
      a: snapA.field,
      b: snapB.field,
      control: { overall_a: snapA.refHoldoutAcc, overall_b: snapB.refHoldoutAcc, per_class: fieldPerClass },
      worst_regression,
    },
    ref_holdout: {
      role_accuracy_a: snapA.refHoldoutAcc,
      role_accuracy_b: snapB.refHoldoutAcc,
      per_class: CIVIL_ROLES.map((cls) => ({
        cls,
        a: snapA.perClass.find((p) => p.cls === cls)?.acc ?? null,
        b: snapB.perClass.find((p) => p.cls === cls)?.acc ?? null,
      })),
    },
    military: {
      rare_target_recall_a: snapA.military.rare_target_recall,
      rare_target_recall_b: snapB.military.rare_target_recall,
      recall_new_class_a: snapA.military.recall_new_class,
      recall_new_class_b: snapB.military.recall_new_class,
      miss_rate_civil_a: snapA.military.miss_rate_civil,
      miss_rate_civil_b: snapB.military.miss_rate_civil,
      n_eval_a: snapA.military.n_eval,
      n_eval_b: snapB.military.n_eval,
      exclusion_method: "analyst-labeled chips and every other chip from that chip's parent tile are dropped from the held-out eval pool",
    },
    novelty_rate_by_batch: BATCH_DEFS.map((def) => ({
      batch_id: def.id,
      a: snapA.batches.find((x) => x.id === def.id)?.novelty_rate ?? null,
      b: snapB.batches.find((x) => x.id === def.id)?.novelty_rate ?? null,
    })),
    forgetting,
  };
}

export function getMap() {
  const points = [];
  const groups = new Map();
  const consider = [...batchChipIds.entries()];
  for (const [batchId, ids] of consider) {
    for (const id of ids) {
      const c = chips.get(id);
      const key = c.tile_id;
      if (!groups.has(key)) groups.set(key, { batchId, chips: [] });
      groups.get(key).chips.push(c);
    }
  }
  const tileIds = [...groups.keys()];
  for (let i = tileIds.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [tileIds[i], tileIds[j]] = [tileIds[j], tileIds[i]];
  }
  for (const tileId of tileIds) {
    const g = groups.get(tileId);
    const t = tiles.get(tileId);
    points.push({
      parent_id: tileId,
      lat: Math.round(t.meta.lat * 10000) / 10000,
      lon: Math.round(t.meta.lon * 10000) / 10000,
      airfield: t.meta.airfield,
      country: t.meta.country,
      domain: g.batchId,
      n_chips: g.chips.length,
      novelty_rate: Math.round(mean(g.chips.map((c) => (c.is_novel ? 1 : 0))) * 1000) / 1000,
      drift_score: Math.round(mean(g.chips.map((c) => c.knn_dist)) * 1000) / 1000,
    });
  }
  return { points };
}

export function getReportMarkdown() {
  const summary = getSummary();
  const envelope = getEnvelope();
  const evalData = getEval();
  const lines = [];
  lines.push(`# Vigil — Assurance Evidence Report`);
  lines.push('');
  lines.push(`*prototype · public RarePlanes data (WorldView-3) · not an accreditation*`);
  lines.push('');
  lines.push(`**Version:** ${summary.current_version}   **Generated:** ${new Date().toISOString()}`);
  lines.push('');
  lines.push(`## Model card`);
  lines.push(`- Detector: ${summary.model.detector}`);
  lines.push(`- Embedder: ${summary.model.embedder}`);
  lines.push(`- Novelty method: ${summary.model.novelty_method} (vigilance = ${summary.model.vigilance})`);
  lines.push(`- Reference domain: ${summary.model.reference_domain}`);
  lines.push(`- Reference set: ${summary.model.n_ref_chips.toLocaleString()} chips / ${summary.model.n_ref_images.toLocaleString()} images`);
  lines.push(`- Calibrated on: ${summary.model.calibrated_on}`);
  lines.push('');
  lines.push(`## Operational stream`);
  lines.push(`| batch | n chips | drift score | null p95 | p | novelty rate | role acc |`);
  lines.push(`|---|---|---|---|---|---|---|`);
  for (const b of summary.batches) {
    lines.push(`| ${b.name} | ${b.n_chips} | ${b.drift.score} | ${b.drift.null_p95} | ${b.drift.p_value} | ${(b.novelty_rate * 100).toFixed(1)}% | ${b.role_accuracy != null ? (b.role_accuracy * 100).toFixed(1) + '%' : '—'} |`);
  }
  lines.push('');
  lines.push(`## Tested operating envelope`);
  for (const dim of envelope.dimensions) {
    lines.push(`### ${dim.name}`);
    lines.push(`| bucket | n chips | novelty rate | role accuracy | status |`);
    lines.push(`|---|---|---|---|---|`);
    for (const b of dim.buckets) {
      lines.push(`| ${b.label} | ${b.n_chips} | ${(b.novelty_rate * 100).toFixed(1)}% | ${b.role_accuracy != null ? (b.role_accuracy * 100).toFixed(1) + '%' : '—'} | ${b.status} |`);
    }
    lines.push('');
  }
  lines.push(`## Evaluation (ground truth used for evaluation only)`);
  lines.push(`- Novelty AUROC (military vs civil, ART): ${evalData.novelty.auroc_military_vs_civil}`);
  lines.push(`- Novelty AUROC (kNN baseline): ${evalData.novelty.baseline_knn_auroc}`);
  lines.push(`- Size-matched AUROC: ${evalData.novelty.auroc_size_matched}`);
  lines.push(`- Role-head accuracy (control batch): ${(evalData.role_head.accuracy_ref_holdout * 100).toFixed(1)}%`);
  lines.push('');
  lines.push(`## Feedback log`);
  lines.push(`- Total reviews recorded: ${reviews.size}`);
  lines.push(`- Versions: ${versions.map((v) => v.id).join(' → ')}`);
  lines.push('');
  lines.push(`*This is assurance evidence for the current prototype version, not an accreditation decision.*`);
  return lines.join('\n');
}

// ============================================================================
// FIELD MODE — collections of new, unlabeled imagery + detector ("det")
// chips. See research/08-field-mode-proposal.md and backend/API.md's
// field-mode contract. No ground truth is ever exposed on a serialized det
// chip — `_trueRole`/`_usedInFit` below are internal-only generation state,
// stripped by serializeDetChip(). Tiles are created via the same makeTile()
// used by the GT batches above, so the existing /tiles/{id} (+ /image)
// endpoints work unchanged for field-mode tiles too.
// ============================================================================

const US_STATES = {
  'Davis-Monthan AFB': 'Arizona',
  'March ARB': 'California',
  'Offutt AFB': 'Nebraska',
  'Travis AFB': 'California',
  'Wright-Patterson AFB': 'Ohio',
  'San Diego Intl': 'California',
};

const FIELD_COLLECTIONS_DEFS = [
  { id: 'davis-monthan-2026-09-02', airfield: 'Davis-Monthan AFB', country: 'USA', date: '2026-09-02', weather: 'Clear', offNadir: [4, 12], gsd: [0.28, 0.32], nTiles: 6, detsPerTile: [10, 16], militaryFrac: 0.03, flagBias: 0.03 },
  { id: 'travis-2026-09-03', airfield: 'Travis AFB', country: 'USA', date: '2026-09-03', weather: 'Clear', offNadir: [3, 9], gsd: [0.26, 0.3], nTiles: 2, detsPerTile: [8, 12], militaryFrac: 0.0, flagBias: 0.02 },
  { id: 'march-2026-09-03', airfield: 'March ARB', country: 'USA', date: '2026-09-03', weather: 'Snow', offNadir: [5, 10], gsd: [0.3, 0.36], nTiles: 5, detsPerTile: [10, 16], militaryFrac: 0.04, flagBias: 0.13 },
  { id: 'ramstein-2026-09-04', airfield: 'Ramstein AB', country: 'Germany', date: '2026-09-04', weather: 'Haze', offNadir: [8, 18], gsd: [0.3, 0.4], nTiles: 4, detsPerTile: [10, 15], militaryFrac: 0.18, flagBias: 0.10 },
  { id: 'wright-patterson-2026-09-05', airfield: 'Wright-Patterson AFB', country: 'USA', date: '2026-09-05', weather: 'Clear', offNadir: [10, 20], gsd: [0.3, 0.36], nTiles: 7, detsPerTile: [14, 22], militaryFrac: 0.32, flagBias: 0.24 },
  { id: 'offutt-2026-09-06', airfield: 'Offutt AFB', country: 'USA', date: '2026-09-06', weather: 'Clear', offNadir: [6, 14], gsd: [0.28, 0.34], nTiles: 6, detsPerTile: [12, 18], militaryFrac: 0.22, flagBias: 0.19 },
];

const MIN_DENOMINATOR = 20;
const NEEDS_REVIEW_MIN_FLAGGED = 10;
const NEEDS_REVIEW_FLAG_RATE_MULTIPLIER = 2.0;
const STATUS_RULE_TEXT =
  `insufficient_data if n_detections < ${MIN_DENOMINATOR}; needs_review if flag_rate >= ${NEEDS_REVIEW_FLAG_RATE_MULTIPLIER}x reference_flag_rate OR n_flagged >= ${NEEDS_REVIEW_MIN_FLAGGED}; else nominal.`;

const detChips = new Map(); // det_id -> record (incl. hidden _trueRole/_usedInFit, never serialized)
const tileDetections = new Map(); // tile_id -> DetectorBox[] (this tile's detector output)
const collectionDetIds = new Map(); // group_id -> det_id[]
const fieldReviews = new Map(); // det_id -> Review (separate namespace from GT `reviews`)
const fieldClassLabelsEver = []; // analyst-class labels ever created from a det chip, in creation order

let detSeq = 900000; // guarantees no collision with GT ann_ids in this mock's small id space

for (const def of FIELD_COLLECTIONS_DEFS) {
  const detIds = [];
  collectionDetIds.set(def.id, detIds);
  const tileIds = [];
  for (let i = 0; i < def.nTiles; i++) {
    tileIds.push(makeTile(def.country, [def.weather], def.offNadir, def.gsd, def.airfield));
  }
  def._tileIds = tileIds;

  for (const tileId of tileIds) {
    const tile = tiles.get(tileId);
    const nDets = Math.round(randRange(rng, def.detsPerTile[0], def.detsPerTile[1]));
    const nMilitary = Math.round(nDets * def.militaryFrac);
    const detectionsForTile = [];
    for (let i = 0; i < nDets; i++) {
      const isMilitary = i < nMilitary;
      const trueRole = isMilitary ? pick(rng, MILITARY_ROLES) : pickWeighted(rng, CIVIL_ROLES, [0.5, 0.32, 0.18]);
      const noise = gaussian(rng) * 0.12;
      const noveltyScore = isMilitary ? clamp01(0.78 + Math.abs(gaussian(rng)) * 0.14) : clamp01(0.3 + def.flagBias * 0.6 + noise);
      const roleConfBase = isMilitary ? randRange(rng, 0.3, 0.55) : randRange(rng, 0.7, 0.97);
      const rolePred = isMilitary ? pick(rng, CIVIL_ROLES) : trueRole;
      const det_id = detSeq++;
      const bbox = randomBBox();
      const conf = Math.round(randRange(rng, 0.28, 0.95) * 1000) / 1000;
      const detChip = {
        det_id,
        tile_id: tileId,
        bbox,
        conf,
        chip_url: `/api/det/${det_id}/image`,
        tile_url: `/api/tiles/${tileId}/image`,
        role_pred: rolePred,
        role_conf: Math.round(roleConfBase * 1000) / 1000,
        novelty: Math.round(clamp01(noveltyScore) * 1000) / 1000,
        is_novel: false, // finalized below, per collection, at its target flag rate
        knn_dist: Math.round(clamp01(noveltyScore * 0.5 + randRange(rng, -0.1, 0.25)) * 1000) / 1000,
        meta: { airfield: tile.meta.airfield, country: tile.meta.country, weather: tile.meta.weather, off_nadir: tile.meta.off_nadir, gsd: tile.meta.gsd },
        review: null,
        _trueRole: trueRole, // hidden — never serialized (no ground truth in field mode)
        _usedInFit: false,
      };
      detChips.set(det_id, detChip);
      detIds.push(det_id);
      detectionsForTile.push({ x1: bbox[0], y1: bbox[1], x2: bbox[0] + bbox[2], y2: bbox[1] + bbox[3], conf, tp: false });
    }
    tileDetections.set(tileId, detectionsForTile);
  }

  // finalize novelty flags for this collection at its target flag rate
  const targetFlagged = Math.max(0, Math.round(detIds.length * def.flagBias));
  const ranked = [...detIds].sort((x, y) => detChips.get(y).novelty - detChips.get(x).novelty);
  const novelSet = new Set(ranked.slice(0, targetFlagged));
  for (const id of detIds) {
    detChips.get(id).is_novel = novelSet.has(id);
  }

  // collection-level facts (lat/lon/off-nadir/gsd) averaged from its tiles
  const tileMetas = tileIds.map((tid) => tiles.get(tid).meta);
  def._lat = r3(mean(tileMetas.map((m) => m.lat)));
  def._lon = r3(mean(tileMetas.map((m) => m.lon)));
  def._offNadir = r3(mean(tileMetas.map((m) => m.off_nadir)));
  def._gsd = r3(mean(tileMetas.map((m) => m.gsd)));
}

// Flag rate over the labeled validation set (reference domain) — the
// baseline every collection's status is judged against.
const REFERENCE_FLAG_RATE = referenceIds.length ? r3(mean(referenceIds.map((id) => (chips.get(id).is_novel ? 1 : 0)))) : null;

function serializeDetChip(c) {
  return {
    source: 'det',
    det_id: c.det_id,
    ann_id: c.det_id,
    tile_id: c.tile_id,
    bbox: c.bbox,
    conf: c.conf,
    chip_url: c.chip_url,
    tile_url: c.tile_url,
    role_pred: c.role_pred,
    role_conf: c.role_conf,
    novelty: c.novelty,
    is_novel: c.is_novel,
    knn_dist: c.knn_dist,
    meta: c.meta,
    review: fieldReviews.get(c.det_id) ?? null,
  };
}

function collectionStatusFor(nDet, nFlagged, flagRate) {
  if (nDet < MIN_DENOMINATOR) return 'insufficient_data';
  const rateExceeded = REFERENCE_FLAG_RATE != null && REFERENCE_FLAG_RATE > 0 && flagRate != null && flagRate >= NEEDS_REVIEW_FLAG_RATE_MULTIPLIER * REFERENCE_FLAG_RATE;
  return nFlagged >= NEEDS_REVIEW_MIN_FLAGGED || rateExceeded ? 'needs_review' : 'nominal';
}

function collectionSummaryFor(def) {
  const detIds = collectionDetIds.get(def.id) || [];
  const detList = detIds.map((id) => detChips.get(id));
  const flagged = detList.filter((c) => c.is_novel);
  const reviewedCount = detList.filter((c) => fieldReviews.has(c.det_id)).length;
  const flagRate = detList.length ? r3(flagged.length / detList.length) : null;
  return {
    group_id: def.id,
    airfield: def.airfield,
    country: def.country,
    state: US_STATES[def.airfield] ?? null,
    date: def.date,
    weather: def.weather,
    off_nadir: def._offNadir,
    gsd: def._gsd,
    lat: def._lat,
    lon: def._lon,
    n_parents: def._tileIds.length,
    n_tiles: def._tileIds.length,
    n_detections: detList.length,
    n_flagged: flagged.length,
    flag_rate: flagRate,
    n_reviewed: reviewedCount,
    status: collectionStatusFor(detList.length, flagged.length, flagRate),
  };
}

// Per-version snapshot of field-mode (det chip) adaptation state — called
// from takeSnapshot() so it stays in lockstep with the GT snapshot used for
// /versions/compare.
function takeFieldSnapshot() {
  const allDet = [...detChips.values()];
  const usedInFitIds = new Set(allDet.filter((c) => c._usedInFit).map((c) => c.det_id));
  const usedInFitTiles = new Set(allDet.filter((c) => c._usedInFit).map((c) => c.tile_id));
  const evalPoolDet = allDet.filter((c) => !usedInFitTiles.has(c.tile_id));

  const classes = fieldClassLabelsEver.map((label) => {
    const nLabelsUsed = [...fieldReviews.values()].filter((r) => r.action === 'new_class' && r.label === label && r.version_applied != null).length;
    const assigned = evalPoolDet.filter((c) => c.role_pred === label);
    return {
      label,
      n_labels_used: nLabelsUsed,
      n_assigned: assigned.length,
      assignment_rate: evalPoolDet.length ? r3(assigned.length / evalPoolDet.length) : null,
    };
  });

  return {
    n_detections: allDet.length,
    n_excluded_labeled: usedInFitIds.size,
    n_excluded_same_parent: allDet.length - evalPoolDet.length,
    classes,
    note: "Assignments are descriptive, not accuracy — the labeled control-set regression check (below) is the only accuracy-style number in field mode.",
  };
}

export function getCollections() {
  const groups = FIELD_COLLECTIONS_DEFS.map(collectionSummaryFor).sort((a, b) => (a.date < b.date ? -1 : a.date > b.date ? 1 : 0));
  return {
    groups,
    grouping: 'airfield and date',
    min_denominator: MIN_DENOMINATOR,
    reference: {
      n_parents: N_REF_TILES,
      n_tiles: N_REF_TILES,
      description: 'Reference domain used to fit and calibrate the model: clear skies, USA, off-nadir ≤ 25°, no military aircraft.',
    },
    reference_flag_rate: REFERENCE_FLAG_RATE,
    status_rule: STATUS_RULE_TEXT,
  };
}

export function getCollectionsSummary() {
  const groups = FIELD_COLLECTIONS_DEFS.map(collectionSummaryFor);
  return {
    n_groups: groups.length,
    n_needs_review: groups.filter((g) => g.status === 'needs_review').length,
    n_detections: groups.reduce((s, g) => s + g.n_detections, 0),
    n_flagged: groups.reduce((s, g) => s + g.n_flagged, 0),
    reference_flag_rate: REFERENCE_FLAG_RATE,
  };
}

export function getCollectionDetail(groupId, query) {
  const def = FIELD_COLLECTIONS_DEFS.find((d) => d.id === groupId);
  if (!def) return null;
  const detIds = collectionDetIds.get(groupId) || [];
  let list = detIds.map((id) => detChips.get(id));
  if (query.only_flagged === 'true') list = list.filter((c) => c.is_novel);
  if (query.reviewed === 'yes') list = list.filter((c) => fieldReviews.has(c.det_id));
  if (query.reviewed === 'no') list = list.filter((c) => !fieldReviews.has(c.det_id));
  const sortKey = query.sort === 'conf' ? 'conf' : 'novelty';
  const order = query.order === 'asc' ? 1 : -1;
  list = [...list].sort((a, b) => (a[sortKey] - b[sortKey]) * order);
  const total = list.length;
  const offset = Number(query.offset) || 0;
  const limit = Number(query.limit) || 48;
  const items = list.slice(offset, offset + limit).map(serializeDetChip);
  const summary = collectionSummaryFor(def);
  const allForGroup = (collectionDetIds.get(groupId) || []).map((id) => detChips.get(id));
  const tilesOut = def._tileIds.map((tid) => {
    const tileDets = allForGroup.filter((c) => c.tile_id === tid);
    return { tile_id: tid, n_detections: tileDets.length, n_flagged: tileDets.filter((c) => c.is_novel).length };
  });
  // Matches backend/API.md's field-mode contract: chips nested under `chips`
  // (not flat total/items) — confirmed against the real backend's shape.
  return {
    ...summary,
    tiles: tilesOut,
    chips: { total, items },
  };
}

export function getDetDetail(detId) {
  const c = detChips.get(Number(detId));
  if (!c) return null;
  const refPool = referenceIds.map((id) => chips.get(id));
  const sameRole = refPool.filter((r) => r.role_true === c.role_pred);
  const pool = sameRole.length >= 5 ? sameRole : refPool;
  const neighbors = [...pool]
    .sort(() => rng() - 0.5)
    .slice(0, 5)
    .map((r, i) => ({ ann_id: r.ann_id, chip_url: r.chip_url, role_true: r.role_true, dist: Math.round((0.04 + i * 0.05 + rng() * 0.03) * 1000) / 1000 }))
    .sort((a, b) => a.dist - b.dist);
  return { ...serializeDetChip(c), neighbors };
}

export function getDetImageInfo(detId) {
  const c = detChips.get(Number(detId));
  if (!c) return null;
  return { seed: c.det_id * 2654435761, role: c.role_pred, isNovel: c.is_novel };
}

export function getFieldReviewQueue(query) {
  const limit = Number(query.limit) || 24;
  const flaggedUnreviewed = [...detChips.values()].filter((c) => c.is_novel && !fieldReviews.has(c.det_id));
  const ranked = [...flaggedUnreviewed].sort((a, b) => b.novelty - a.novelty);
  return ranked.slice(0, limit).map(serializeDetChip);
}

// Now that field-mode data (detChips, referenceIds-derived REFERENCE_FLAG_RATE)
// exists, take the v1 snapshot — see the moved call near `const reviews = new Map()` above.
takeSnapshot('v1');
