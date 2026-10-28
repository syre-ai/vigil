// Vigil demo rehearsal: walks the demo path with Playwright, saves
// frontend/screenshots/real-*.png, performs the adaptation rehearsal
// (label + confirm chips, apply feedback), and scans every page for
// console errors / forbidden wording / formatting bugs.
//
// Usage: node scripts/screenshots.mjs   (run from frontend/, dev server on :5173)
import { chromium } from 'playwright';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const SHOT_DIR = path.join(__dirname, '..', 'screenshots');
const BASE_URL = 'http://localhost:5173/';
const API_BASE = 'http://localhost:8000/api';
const CHROME_PATH = process.env.CHROME_PATH; // unset = Playwright's bundled Chromium

const FORBIDDEN_WORDS = ['deployed', 'accredited', 'significant'];

const issues = []; // { page, kind, detail }

function record(pageLabel, kind, detail) {
  issues.push({ page: pageLabel, kind, detail });
  console.log(`[ISSUE] ${pageLabel} :: ${kind} :: ${detail}`);
}

async function scanPage(page, label) {
  const text = await page.locator('body').innerText();
  // forbidden vocabulary (word-boundary, case-insensitive)
  for (const w of FORBIDDEN_WORDS) {
    const re = new RegExp(`\\b${w}\\b`, 'i');
    if (re.test(text)) record(label, 'forbidden-word', `found "${w}"`);
  }
  // NaN / undefined literal text
  if (/\bNaN\b/.test(text)) record(label, 'NaN-text', 'literal NaN found in rendered text');
  if (/\bundefined\b/.test(text)) record(label, 'undefined-text', 'literal undefined found in rendered text');
  // suspiciously long floats (>= 10 fractional digits, or >=15 total digits)
  const numMatches = text.match(/\d+\.\d{6,}/g);
  if (numMatches) {
    for (const m of numMatches) {
      const digitCount = m.replace('.', '').length;
      if (digitCount >= 10) record(label, 'long-float', m);
    }
  }
}

async function collectConsole(page, label) {
  page.on('console', (msg) => {
    if (msg.type() === 'error') record(label, 'console-error', msg.text());
  });
  page.on('pageerror', (err) => {
    record(label, 'page-error', err.message);
  });
}

async function nav(page, label) {
  await page.locator('button.nav-item', { hasText: label }).click();
  await page.waitForTimeout(200);
}

async function settle(page) {
  try {
    await page.waitForLoadState('networkidle', { timeout: 8000 });
  } catch {
    // some pages (map tiles) never fully go idle -- fall back to a fixed wait
    await page.waitForTimeout(1200);
  }
}

async function shot(page, name, opts = {}) {
  const file = path.join(SHOT_DIR, `real-${name}.png`);
  await page.screenshot({ path: file, fullPage: !!opts.fullPage });
  console.log('saved', file);
}

async function openChipByAnnId(page, annId, { timeout = 15000 } = {}) {
  const img = page.locator(`img[src="/api/chips/${annId}/image"]`).first();
  await img.waitFor({ state: 'visible', timeout });
  await img.click();
  await page.waitForSelector('.chip-drawer', { timeout });
  await page.waitForTimeout(500); // tile canvas draw + neighbor thumbnails
}

async function closeModal(page) {
  const closeBtn = page.locator('.modal-close');
  if (await closeBtn.count()) {
    await closeBtn.click();
    await page.waitForTimeout(150);
  }
}

async function submitNewClass(page, label) {
  await page.locator('.review-actions button', { hasText: 'New class' }).click();
  const input = page.locator('#new-class-label');
  await input.waitFor({ state: 'visible' });
  await input.fill(label);
  await page.locator('.new-class-form button.btn-primary', { hasText: 'Submit' }).click();
  await page.waitForTimeout(400);
}

async function submitConfirm(page) {
  await page.locator('.review-actions button', { hasText: 'Confirm role' }).click();
  await page.waitForTimeout(400);
}

async function main() {
  fs.mkdirSync(SHOT_DIR, { recursive: true });
  // delete stale real-* screenshots
  for (const f of fs.readdirSync(SHOT_DIR)) {
    if (f.startsWith('real-')) fs.unlinkSync(path.join(SHOT_DIR, f));
  }

  const browser = await chromium.launch({ executablePath: CHROME_PATH });
  const page = await browser.newPage({ viewport: { width: 1560, height: 960 } });
  await collectConsole(page, 'global');

  console.log('== navigating to app ==');
  await page.goto(BASE_URL, { waitUntil: 'networkidle' });
  await page.waitForSelector('.model-card-grid', { timeout: 20000 });

  // ---------- 01 Overview ----------
  await settle(page);
  await scanPage(page, '01-overview');
  await shot(page, '01-overview');
  await shot(page, '01-overview-full', { fullPage: true });

  // ---------- 02 Batch -> Snow, missed by detector ----------
  await nav(page, 'Batch');
  await page.waitForSelector('.chip-grid', { timeout: 15000 });
  await page.locator('.page-header select').selectOption('snow');
  await page.waitForTimeout(300);
  await page.locator('.chip-controls-right select').nth(0).selectOption('no'); // detected filter
  await settle(page);
  await scanPage(page, '02-batch-snow-missed');
  await shot(page, '02-batch-snow-missed');

  // ---------- 03 chip drawer: snow chip missed by detector ----------
  // sort=novelty desc is already the default -> first card is the target chip
  await page.locator('.chip-card').first().click();
  await page.waitForSelector('.chip-drawer', { timeout: 15000 });
  await page.waitForTimeout(700); // let TileCanvas image draw
  await scanPage(page, '03-chip-drawer-snow-missed');
  await shot(page, '03-chip-drawer-snow-missed');
  await closeModal(page);

  // ---------- 04 Batch -> military, sorted by novelty desc ----------
  await page.locator('.page-header select').selectOption('military');
  await page.waitForTimeout(300);
  await page.locator('.chip-controls-right select').nth(0).selectOption('all'); // clear detected filter
  await settle(page);
  await scanPage(page, '04-batch-military');
  await shot(page, '04-batch-military');

  // ---------- 05 chip drawer: top-novelty military-role chip (neighbors strip) ----------
  const FIGHTER_1 = 22119; // top-novelty chip whose GT role is actually military (fighter)
  await openChipByAnnId(page, FIGHTER_1);
  await scanPage(page, '05-chip-drawer-military');
  await shot(page, '05-chip-drawer-military');
  await closeModal(page);

  // ---------- 06 Review queue ----------
  await nav(page, 'Review queue');
  await page.waitForSelector('.review-list', { timeout: 15000 });
  await settle(page);
  await scanPage(page, '06-review-queue');
  await shot(page, '06-review-queue');

  // ---------- 07 Embedding (military batch) ----------
  await nav(page, 'Embedding');
  await page.waitForSelector('.page-header select', { timeout: 15000 });
  await page.locator('.page-header select').selectOption('military');
  await settle(page);
  await page.waitForTimeout(500);
  await scanPage(page, '07-embedding-military');
  await shot(page, '07-embedding-military');

  // ---------- 08 Tested envelope ----------
  await nav(page, 'Tested envelope');
  await page.waitForSelector('.data-table', { timeout: 15000 });
  await settle(page);
  await scanPage(page, '08-envelope');
  await shot(page, '08-envelope', { fullPage: true });

  // ---------- 09 Evaluation (full page) ----------
  await nav(page, 'Evaluation');
  await page.waitForSelector('.data-table', { timeout: 15000 });
  await settle(page);
  await scanPage(page, '09-evaluation-full');
  await shot(page, '09-evaluation-full', { fullPage: true });

  // ---------- 10 Map ----------
  await nav(page, 'Map');
  await page.waitForSelector('.map-canvas', { timeout: 15000 });
  await page.waitForTimeout(2500); // OSM tiles + markers
  await scanPage(page, '10-map');
  await shot(page, '10-map');

  // ---------- 11 Report (top) ----------
  await nav(page, 'Report');
  await page.waitForSelector('.markdown-body', { timeout: 15000 });
  await settle(page);
  await scanPage(page, '11-report');
  await shot(page, '11-report');

  // =====================================================================
  // Adaptation rehearsal
  // =====================================================================
  console.log('== adaptation rehearsal ==');
  await nav(page, 'Review queue');
  await page.waitForSelector('.review-list', { timeout: 15000 });

  const CIVIL_1 = 17762;
  const CIVIL_2 = 23765;
  const FIGHTERS_VIA_QUEUE = [22119, 13174, 9977, 12379]; // all "Military fighter"
  const REMAINING_VIA_BATCH = [
    { annId: 4374, label: 'Military transport' },
    { annId: 15794, label: 'Military fighter' },
  ];

  // 2 civil confirms, via the review queue list
  for (const annId of [CIVIL_1, CIVIL_2]) {
    const img = page.locator(`.review-list-item img[src="/api/chips/${annId}/image"]`).first();
    await img.waitFor({ state: 'visible', timeout: 15000 });
    await img.click();
    await page.waitForSelector('.chip-drawer', { timeout: 15000 });
    await submitConfirm(page);
  }

  // first 4 military-fighter labels, via the review queue list (still in the
  // visible novelty-ranked window at this point)
  for (const annId of FIGHTERS_VIA_QUEUE) {
    const img = page.locator(`.review-list-item img[src="/api/chips/${annId}/image"]`).first();
    await img.waitFor({ state: 'visible', timeout: 15000 });
    await img.click();
    await page.waitForSelector('.chip-drawer', { timeout: 15000 });
    await submitNewClass(page, 'Military fighter');
  }

  // remaining 2 military labels: these have fallen outside the review queue's
  // fixed 24-item novelty-ranked window after the 6 prior reviews (a real
  // rehearsal finding -- see report), so open them from Batch -> military instead.
  await nav(page, 'Batch');
  await page.waitForSelector('.chip-grid', { timeout: 15000 });
  await page.locator('.page-header select').selectOption('military');
  await page.waitForTimeout(300);
  for (const { annId, label } of REMAINING_VIA_BATCH) {
    await openChipByAnnId(page, annId);
    await submitNewClass(page, label);
    await closeModal(page);
  }

  // Apply feedback -> new version, timed. Guarded against a concurrent
  // backend file edit (this repo has another autonomous process touching
  // backend/app files -- observed mid-rehearsal) triggering uvicorn's
  // --reload and dropping the in-flight connection: if the UI-level wait
  // times out, fall back to polling /api/summary directly rather than
  // re-clicking (which would double-apply the same feedback into two
  // versions).
  await nav(page, 'Review queue');
  await page.waitForSelector('.review-list', { timeout: 15000 });
  const applyBtn = page.locator('button.btn-primary', { hasText: 'Apply feedback' });
  await applyBtn.waitFor({ state: 'visible', timeout: 15000 });

  const beforeSummary = await (await fetch(`${API_BASE}/summary`)).json();
  const versionBefore = beforeSummary.current_version;

  const adaptStart = Date.now();
  let adaptBody = null;
  try {
    const [adaptResponse] = await Promise.all([
      page.waitForResponse((r) => r.url().includes('/api/adapt') && r.request().method() === 'POST', { timeout: 60000 }),
      applyBtn.click(),
    ]);
    adaptBody = await adaptResponse.json().catch(() => null);
  } catch (err) {
    console.log('waitForResponse for /api/adapt timed out (', err.message, ') -- polling /api/summary instead');
    for (let i = 0; i < 30 && !adaptBody; i++) {
      await page.waitForTimeout(2000);
      const s = await (await fetch(`${API_BASE}/summary`)).json();
      if (s.current_version !== versionBefore) {
        adaptBody = { id: s.current_version };
      }
    }
    if (!adaptBody) throw new Error('adapt never completed (current_version did not advance after fallback polling)');
  }
  const adaptEnd = Date.now();
  const adaptMs = adaptEnd - adaptStart;
  console.log('POST /api/adapt took', adaptMs, 'ms; new version:', adaptBody && adaptBody.id);

  // Sanity check: exactly one new version should have been created.
  const afterSummary = await (await fetch(`${API_BASE}/summary`)).json();
  console.log('versions after adapt:', afterSummary.versions.map((v) => v.id));
  if (afterSummary.versions.length !== 2) {
    console.log('WARNING: expected exactly 2 versions (v1 + new), found', afterSummary.versions.length);
  }

  await page.waitForSelector('.regression-banner', { timeout: 30000 });
  await settle(page);
  await scanPage(page, '12-compare-table');
  await shot(page, '12-compare-table', { fullPage: true });

  const versionsPair = { a: 'v1', b: adaptBody ? adaptBody.id : null };

  await browser.close();

  const result = { issues, adaptMs, versionsPair, timestamp: new Date().toISOString() };
  const outPath = process.env.REHEARSAL_OUT || path.join('/tmp', 'rehearsal_run.json');
  fs.writeFileSync(outPath, JSON.stringify(result, null, 2));
  console.log('== DONE ==');
  console.log(JSON.stringify(result, null, 2));
}

main().catch((err) => {
  console.error('FATAL', err);
  process.exit(1);
});
