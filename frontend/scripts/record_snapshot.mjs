// Records a read-only snapshot of the running app for static hosting.
//
// Drives the dev server (which proxies /api to the real backend) through every
// page in both views, every benchmark batch and every field collection, and
// saves each GET response under snap/api/<snapKey(path)>.json. Then fetches the
// detail payloads a chip drawer opens (/chips, /det, /tiles) for every chip the
// pages listed. Image URLs inside payloads are rewritten to snap/img/*.webp; the
// images themselves are listed in snap/images.txt for scripts/snapshot_images.py.
// Non-GET requests are aborted, so recording never changes backend state.
//
//   APP=http://localhost:5173 BACKEND=http://localhost:8000 OUT=snap BASE=/vigil/ \
//     node scripts/record_snapshot.mjs
import { mkdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { chromium } from 'playwright';

const APP = process.env.APP || 'http://localhost:5173';
const BACKEND = process.env.BACKEND || 'http://localhost:8000';
const OUT = process.env.OUT || 'snap';
const BASE = process.env.BASE || '/vigil/';

// FNV-1a, 32-bit. Must match snapKey in src/api/client.ts.
function snapKey(path) {
  let h = 0x811c9dc5;
  for (let i = 0; i < path.length; i++) {
    h ^= path.charCodeAt(i);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h.toString(16).padStart(8, '0');
}

mkdirSync(join(OUT, 'api'), { recursive: true });
const saved = new Map(); // path -> parsed JSON
const images = new Set();
const IMG = /\/api\/(chips|tiles|det)\/(\d+)\/image/g;

function save(path, text, collect = true) {
  if (path === '/report') {
    writeFileSync(join(OUT, 'report.md'), text);
    return;
  }
  const rewritten = text.replace(IMG, (_, kind, id) => {
    if (collect) images.add(`${kind}/${id}`);
    return `${BASE}snap/img/${kind}_${id}.webp`;
  });
  writeFileSync(join(OUT, 'api', `${snapKey(path)}.json`), rewritten);
  try {
    saved.set(path, JSON.parse(text));
  } catch {
    // non-JSON body; keep the file, skip the walk
  }
}

async function fetchDirect(path, collect = true) {
  if (saved.has(path)) return;
  const res = await fetch(`${BACKEND}/api${path}`);
  if (!res.ok) return;
  save(path, await res.text(), collect);
}

const browser = await chromium.launch({ executablePath: process.env.CHROMIUM });
const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
let inflight = 0;
await page.route('**/api/**', async (route) => {
  const req = route.request();
  if (req.method() !== 'GET') return route.abort();
  const url = new URL(req.url());
  const path = url.pathname.slice('/api'.length) + url.search;
  if (/^\/(chips|tiles|det)\/\d+\/image$/.test(url.pathname.slice(4))) {
    images.add(url.pathname.slice(5).replace('/image', ''));
    return route.fulfill({ status: 200, contentType: 'image/png', body: Buffer.alloc(0) });
  }
  inflight++;
  try {
    const res = await route.fetch();
    const body = await res.text();
    if (res.ok()) save(path, body);
    await route.fulfill({ response: res, body });
  } finally {
    inflight--;
  }
});

async function settle() {
  await page.waitForTimeout(300);
  for (let quiet = 0; quiet < 4; ) {
    await page.waitForTimeout(150);
    quiet = inflight === 0 ? quiet + 1 : 0;
  }
}

async function go(name, param) {
  await page.evaluate(([n, p]) => window.__vigilGoto(n, p), [name, param]);
  await settle();
}

const PAGES = ['reference', 'overview', 'batch', 'review', 'embedding', 'envelope', 'eval', 'map', 'report',
  'collections', 'field-review', 'changes', 'health'];

const batches = await (await fetch(`${BACKEND}/api/batches`)).json();
const collections = (await (await fetch(`${BACKEND}/api/collections`)).json()).groups;

for (const view of ['analyst', 'benchmark']) {
  await page.goto(APP);
  await page.evaluate((v) => localStorage.setItem('vigil.view', v), view);
  await page.goto(APP);
  await page.waitForFunction(() => typeof window.__vigilGoto === 'function');
  await settle();
  for (const p of PAGES) await go(p);
  if (view === 'benchmark') for (const b of batches) await go('batch', b.id);
  else for (const c of collections) await go('collection', c.group_id);
  console.log(`${view}: ${saved.size} payloads`);
}
await browser.close();

for (const b of batches) await fetchDirect(`/embedding/points?${new URLSearchParams({ batch: b.id })}`);

// Detail payloads a chip drawer opens, for every chip any page listed.
const chips = new Set();
const dets = new Set();
const tiles = new Set();
function walk(x) {
  if (Array.isArray(x)) return x.forEach(walk);
  if (!x || typeof x !== 'object') return;
  const isDet = typeof x.det_id === 'number';
  const isChip = !isDet && typeof x.ann_id === 'number' && 'role_true' in x && 'novelty' in x;
  if (isDet) dets.add(x.det_id);
  if (isChip) chips.add(x.ann_id);
  // Only the tiles behind listed chips: a drawer's tile view is one click from a chip.
  if ((isDet || isChip) && typeof x.tile_id === 'number') tiles.add(x.tile_id);
  Object.values(x).forEach(walk);
}
[...saved.entries()].filter(([p]) => !p.startsWith('/tiles/')).forEach(([, v]) => walk(v));
for (const id of chips) await fetchDirect(`/chips/${id}`);
for (const id of dets) await fetchDirect(`/det/${id}`);
// Tile payloads list every chip on the tile; their thumbnails aren't shown, so they aren't fetched.
for (const id of tiles) {
  await fetchDirect(`/tiles/${id}`, false);
  images.add(`tiles/${id}`);
}

writeFileSync(join(OUT, 'images.txt'), [...images].sort().join('\n') + '\n');
console.log(`payloads ${saved.size}, chips ${chips.size}, dets ${dets.size}, tiles ${tiles.size}, images ${images.size}`);
