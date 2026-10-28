// One-off screenshot pass for the field-mode (Analyst view) build, taken
// against frontend/mock (port 8099) via a scratch Vite instance on :5183 —
// see the field-mode build report for why (real backend's det-chip data
// hasn't landed yet). Saves frontend/screenshots/field-*.png.
import { chromium } from 'playwright';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const SHOT_DIR = path.join(__dirname, '..', 'screenshots');
const BASE_URL = 'http://localhost:5183/';
const CHROME_PATH = process.env.CHROME_PATH; // unset = Playwright's bundled Chromium

fs.mkdirSync(SHOT_DIR, { recursive: true });

async function shot(page, name, opts = {}) {
  const file = path.join(SHOT_DIR, `field-${name}.png`);
  await page.screenshot({ path: file, fullPage: !!opts.fullPage });
  console.log('saved', file);
}

async function settle(page) {
  try {
    await page.waitForLoadState('networkidle', { timeout: 6000 });
  } catch {
    await page.waitForTimeout(800);
  }
}

async function main() {
  const browser = await chromium.launch({ executablePath: fs.existsSync(CHROME_PATH) ? CHROME_PATH : undefined });
  const page = await browser.newPage({ viewport: { width: 1440, height: 960 } });
  page.on('pageerror', (err) => console.log('[page-error]', err.message));
  page.on('console', (msg) => {
    if (msg.type() === 'error') console.log('[console-error]', msg.text());
  });
  page.on('response', (res) => {
    if (res.status() >= 400) console.log('[http]', res.status(), res.url());
  });

  // Force Analyst view + clear any stale queue/tile state from a previous run.
  await page.addInitScript(() => {
    try {
      window.localStorage.setItem('vigil.view', 'analyst');
    } catch {
      /* ignore */
    }
  });

  await page.goto(BASE_URL);
  await settle(page);
  await shot(page, '01-collections');

  // needs-review filter
  await page.getByText('needs review only').click();
  await page.waitForTimeout(300);
  await shot(page, '02-collections-needs-review-only');
  await page.getByText('needs review only').click();
  await page.waitForTimeout(300);

  // Snow collection
  await page.locator('td.mono:has-text("2026-09-03")').first();
  const marchRow = page.locator('tr', { hasText: 'March ARB' });
  await marchRow.click();
  await settle(page);
  await shot(page, '03-collection-snow');

  // Back (client-side router, no browser history), open the military-heavy collection
  await page.locator('button.btn:has-text("Collections")').click();
  await settle(page);
  const wpRow = page.locator('tr', { hasText: 'Wright-Patterson AFB' });
  await wpRow.click();
  await settle(page);
  await shot(page, '04-collection-military-needs-review');

  // Open a flagged chip's drawer
  const chipCard = page.locator('.chip-card').first();
  await chipCard.waitFor({ state: 'visible', timeout: 10000 });
  await chipCard.click();
  await page.waitForSelector('.chip-drawer', { timeout: 10000 });
  await settle(page);
  await shot(page, '05-det-chip-drawer');

  // Open the tile viewer (detector boxes only)
  await page.locator('.tile-canvas-wrap').click();
  await page.waitForSelector('.tile-viewer-panel', { timeout: 10000 });
  await page.waitForTimeout(400);
  await shot(page, '06-det-chip-tile-viewer');
  // Close via the × button, not Escape — Escape is caught by both the tile
  // viewer AND the outer chip drawer modal's window-level listeners (a
  // pre-existing shared-component quirk, unrelated to field mode), which
  // would close both at once.
  await page.locator('.tile-viewer-close').click();
  await page.waitForTimeout(200);

  // Label it a new analyst class, fill label, submit
  await page.getByRole('button', { name: /New class/ }).click();
  await page.waitForTimeout(150);
  await page.fill('#new-class-label', 'Suspected Fighter');
  await page.getByRole('button', { name: 'Submit' }).click();
  await page.waitForTimeout(400);
  await shot(page, '07-det-chip-labeled');

  await page.keyboard.press('Escape');
  await page.waitForTimeout(200);

  // Field Review queue
  await page.locator('button.nav-item', { hasText: 'Review (field)' }).click();
  await settle(page);
  await shot(page, '08-field-review-queue');

  // Apply feedback -> field compare
  const applyBtn = page.getByRole('button', { name: /Apply \d+ feedback/ });
  await applyBtn.waitFor({ state: 'visible', timeout: 10000 });
  const enabled = await applyBtn.isEnabled();
  if (enabled) {
    await applyBtn.click();
    await page.waitForSelector('.panel:has-text("Field compare")', { timeout: 15000 });
    await page.waitForTimeout(300);
    await shot(page, '09-field-compare', { fullPage: true });
  } else {
    console.log('[skip] Apply button disabled — no pending feedback (label step above may not have landed)');
  }

  // Report page (analyst view)
  await page.locator('button.nav-item', { hasText: 'Report' }).click();
  await settle(page);
  await shot(page, '10-report');

  // Guide drawer's Analyst view section
  await page.locator('button:has-text("How to read this")').click();
  await page.waitForSelector('.guide-drawer', { timeout: 5000 });
  await page.waitForTimeout(200);
  await shot(page, '11-guide-drawer');
  await page.keyboard.press('Escape');

  // Benchmark view toggle sanity check (unchanged app still works)
  await page.locator('.view-toggle button:has-text("Benchmark view")').click();
  await settle(page);
  await shot(page, '12-benchmark-view-unchanged');

  await browser.close();
  console.log('done');
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
