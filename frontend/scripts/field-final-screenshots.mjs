// Final analyst-view screenshot pass against the REAL backend (:8000 via the
// dev server on :5173), reset state (no reviews / no adapt). Saves
// frontend/screenshots/field-final-*.png and logs any page/console errors or
// HTTP >= 400 responses (there should be none).
import { chromium } from 'playwright';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const SHOT_DIR = path.join(__dirname, '..', 'screenshots');
const BASE_URL = process.env.BASE_URL ?? 'http://localhost:5173/';
const CHROME_PATH = process.env.CHROME_PATH; // unset = Playwright's bundled Chromium


fs.mkdirSync(SHOT_DIR, { recursive: true });

async function shot(page, name, opts = {}) {
  const file = path.join(SHOT_DIR, `field-final-${name}.png`);
  await page.screenshot({ path: file, fullPage: !!opts.fullPage });
  console.log('saved', file);
}

async function settle(page, ms = 600) {
  try {
    await page.waitForLoadState('networkidle', { timeout: 8000 });
  } catch {
    /* fall through */
  }
  await page.waitForTimeout(ms);
}

async function main() {
  const browser = await chromium.launch({ executablePath: fs.existsSync(CHROME_PATH) ? CHROME_PATH : undefined });
  const page = await browser.newPage({ viewport: { width: 1440, height: 960 } });
  const problems = [];
  page.on('pageerror', (err) => problems.push(`[page-error] ${err.message}`));
  page.on('console', (msg) => {
    if (msg.type() === 'error') problems.push(`[console-error] ${msg.text()}`);
  });
  page.on('response', (res) => {
    if (res.status() >= 400) problems.push(`[http] ${res.status()} ${res.url()}`);
  });

  await page.addInitScript(() => {
    try {
      window.localStorage.setItem('vigil.view', 'analyst');
      window.sessionStorage.clear();
    } catch {
      /* ignore */
    }
  });

  // 1. Collections (default: date, newest first) with the status-rule popover open.
  await page.goto(BASE_URL);
  await settle(page);
  await page.locator('table.data-table tbody tr').first().waitFor({ state: 'visible', timeout: 15000 });
  await page.getByRole('button', { name: 'About the status rule' }).click();
  await page.waitForTimeout(250);
  await shot(page, '01-collections');
  await page.keyboard.press('Escape');

  // 1b. Sorted by flagged, needs-review only.
  await page.locator('th button.th-sort', { hasText: /^flagged$/ }).click();
  await page.getByText('needs review only').click();
  await settle(page, 300);
  await shot(page, '01b-collections-sorted-flagged-needs-review');
  await page.getByText('needs review only').click();

  // 2. Tirana collection (object story).
  await page.getByPlaceholder('filter airfield / country…').fill('Tirana');
  await settle(page, 300);
  await page.locator('table.data-table tbody tr', { hasText: '2019-02-07' }).first().click();
  await settle(page);
  await page.locator('.chip-card').first().waitFor({ state: 'visible', timeout: 15000 });
  await page.waitForTimeout(800); // lazy chip images
  await shot(page, '02-tirana-collection');

  // 3. Top-novelty flagged chip drawer (a fighter predicted as a civil role).
  await page.locator('.chip-card').first().click();
  await page.waitForSelector('.chip-drawer', { timeout: 10000 });
  await settle(page);
  await page.waitForTimeout(1500); // parent-tile canvas Image.onload
  await shot(page, '03-tirana-fighter-chip-drawer');
  // Full-screen tile viewer
  await page.locator('.tile-canvas-wrap').click();
  await page.waitForSelector('.tile-viewer-canvas', { timeout: 10000 });
  await page.waitForTimeout(600);
  await shot(page, '03b-tirana-tile-viewer');
  await page.keyboard.press('Escape');
  await page.waitForTimeout(200);
  // Step to the next chip with the arrow key
  await page.keyboard.press('ArrowRight');
  await settle(page);
  await page.waitForTimeout(1200);
  await shot(page, '03c-tirana-second-chip-drawer');
  await page.locator('.modal-close').click();
  await page.waitForTimeout(200);

  // 4. Salt Lake City snow collection (imagery story).
  await page.locator('button.nav-item', { hasText: 'Collections' }).click();
  await settle(page);
  await page.getByPlaceholder('filter airfield / country…').fill('Salt Lake');
  await settle(page, 300);
  await page.locator('table.data-table tbody tr', { hasText: '2019-02-11' }).first().click();
  await settle(page);
  await page.locator('.chip-card').first().waitFor({ state: 'visible', timeout: 15000 });
  await page.waitForTimeout(800);
  await shot(page, '04-salt-lake-city-snow-collection');

  // 5. Review (field) queue.
  await page.locator('button.nav-item', { hasText: 'Review (field)' }).click();
  await settle(page);
  await page.waitForSelector('.review-list-item, .state-block', { timeout: 20000 });
  await page.waitForTimeout(1500);
  await shot(page, '05-field-review-queue');

  // 6. Guide drawer (analyst paragraph).
  await page.getByRole('button', { name: 'How to read this' }).click();
  await page.waitForTimeout(300);
  await shot(page, '06-guide-drawer');
  await page.keyboard.press('Escape');

  // 7. View toggle info + Benchmark view (unchanged Overview).
  await page.getByRole('button', { name: 'About the two views' }).click();
  await page.waitForTimeout(250);
  await shot(page, '07-view-toggle-info');
  await page.keyboard.press('Escape');
  await page.getByRole('button', { name: 'Benchmark view' }).click();
  await settle(page);
  await page.waitForTimeout(1500);
  await shot(page, '08-benchmark-view');

  await browser.close();
  if (problems.length) {
    console.log('PROBLEMS:');
    for (const p of problems) console.log(' ', p);
  } else {
    console.log('no page/console errors, no HTTP >= 400');
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
