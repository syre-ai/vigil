// Read-only visual-polish screenshot pass: walks every page and a couple of
// interaction states (chip drawer, guide drawer, an InfoPopover, a Details
// disclosure) with Playwright and saves frontend/screenshots/simple-*.png.
// Unlike scripts/screenshots.mjs this script performs NO reviews and never
// calls /api/adapt — it never mutates backend state, so it's safe to re-run
// at any time against the live dev server (vite on :5173, backend on :8000).
// Selectors track the simplification pass (info-popover/details disclosures
// replaced the old always-visible model-card-grid/caption paragraphs).
//
// Usage: node scripts/polish-screenshots.mjs   (run from frontend/)
import { chromium } from 'playwright';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const SHOT_DIR = path.join(__dirname, '..', 'screenshots');
const BASE_URL = 'http://localhost:5173/';
const CHROME_PATH = process.env.CHROME_PATH; // unset = Playwright's bundled Chromium

async function nav(page, label) {
  await page.locator('button.nav-item', { hasText: label }).click();
  await page.waitForTimeout(200);
}

async function settle(page) {
  try {
    await page.waitForLoadState('networkidle', { timeout: 8000 });
  } catch {
    await page.waitForTimeout(1000);
  }
}

async function shot(page, name, opts = {}) {
  const file = path.join(SHOT_DIR, `simple-${name}.png`);
  await page.screenshot({ path: file, fullPage: !!opts.fullPage });
  console.log('saved', file);
}

async function main() {
  fs.mkdirSync(SHOT_DIR, { recursive: true });

  const browser = await chromium.launch({ executablePath: CHROME_PATH });
  const page = await browser.newPage({ viewport: { width: 1560, height: 960 } });
  page.on('pageerror', (err) => console.log('[pageerror]', err.message));
  page.on('console', (msg) => {
    if (msg.type() === 'error') console.log('[console-error]', msg.text());
  });

  await page.goto(BASE_URL, { waitUntil: 'networkidle' });
  await page.waitForSelector('.model-card-line', { timeout: 20000 });
  await settle(page);
  await shot(page, '01-overview', { fullPage: true });

  // Demonstrate the InfoPopover pattern: open the model-card ⓘ.
  await page.locator('.model-card-line .info-popover-btn').click();
  await page.waitForTimeout(200);
  await shot(page, '01b-overview-info-popover');
  await page.keyboard.press('Escape');
  await page.waitForTimeout(150);

  await nav(page, 'Batch');
  await page.waitForSelector('.chip-grid', { timeout: 15000 });
  await settle(page);
  await shot(page, '02-batch', { fullPage: true });

  // Demonstrate the Details disclosure pattern: expand the full condition table.
  await page.locator('.details-toggle', { hasText: 'Full condition table' }).click();
  await page.waitForTimeout(200);
  await shot(page, '02b-batch-details-expanded', { fullPage: true });

  await page.locator('.chip-card').first().click();
  await page.waitForSelector('.chip-drawer', { timeout: 15000 });
  await page.waitForTimeout(600);
  await shot(page, '03-chip-drawer');
  await page.locator('.modal-close').click();
  await page.waitForTimeout(150);

  await nav(page, 'Review queue');
  await page.waitForSelector('.review-list', { timeout: 15000 });
  await settle(page);
  // The detail pane depends on the queue resolving before it can auto-select
  // and fetch chip detail (a second, dependent round-trip) — give it a beat
  // longer than the generic settle() so the shot isn't a skeleton mid-flight.
  await page.waitForSelector('.chip-drawer', { timeout: 15000 });
  await page.waitForTimeout(500);
  await shot(page, '04-review-queue', { fullPage: true });

  await nav(page, 'Embedding');
  await page.waitForSelector('.embedding-legend', { timeout: 15000 });
  await page.waitForTimeout(1200);
  await shot(page, '05-embedding');

  await nav(page, 'Tested envelope');
  await page.waitForSelector('.data-table', { timeout: 15000 });
  await settle(page);
  await shot(page, '06-envelope', { fullPage: true });

  await nav(page, 'Evaluation');
  await page.waitForSelector('.stat-value', { timeout: 15000 });
  await settle(page);
  await shot(page, '07-evaluation', { fullPage: true });

  await nav(page, 'Map');
  await page.waitForSelector('.map-canvas', { timeout: 15000 });
  await page.waitForTimeout(2000);
  await shot(page, '08-map');

  await nav(page, 'Report');
  await page.waitForSelector('.markdown-body', { timeout: 15000 });
  await settle(page);
  await shot(page, '09-report', { fullPage: true });

  await page.locator('button', { hasText: 'How to read this' }).click();
  await page.waitForTimeout(300);
  await shot(page, '10-guide-drawer');

  await browser.close();
  console.log('== DONE ==');
}

main().catch((err) => {
  console.error('FATAL', err);
  process.exit(1);
});
