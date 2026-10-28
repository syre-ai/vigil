// Screenshot pass against the REAL backend (:8000, via the already-running
// dev server on :5173) — taken once /api/collections returned real,
// populated groups. Det chips aren't scored yet (role_pred/novelty/is_novel
// are all null), so this run proves graceful degradation to an "unscored"
// state rather than a fully-flagged demo (see frontend/screenshots/field-*
// for the fully-populated mock-backed walkthrough). Saves
// frontend/screenshots/field-real-*.png.
import { chromium } from 'playwright';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const SHOT_DIR = path.join(__dirname, '..', 'screenshots');
const BASE_URL = 'http://localhost:5173/';
const CHROME_PATH = process.env.CHROME_PATH; // unset = Playwright's bundled Chromium

fs.mkdirSync(SHOT_DIR, { recursive: true });

async function shot(page, name, opts = {}) {
  const file = path.join(SHOT_DIR, `field-real-${name}.png`);
  await page.screenshot({ path: file, fullPage: !!opts.fullPage });
  console.log('saved', file);
}

async function settle(page) {
  try {
    await page.waitForLoadState('networkidle', { timeout: 8000 });
  } catch {
    await page.waitForTimeout(1000);
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

  const firstRow = page.locator('table.data-table tbody tr').first();
  await firstRow.waitFor({ state: 'visible', timeout: 10000 });
  await firstRow.click();
  await settle(page);
  await shot(page, '02-collection-detail-needs-review-bug');

  // Uncheck "only flagged" — nothing is scored yet, so the flagged-only
  // view is empty; this reveals the real (unscored) det chips instead.
  const onlyFlagged = page.getByText('only flagged');
  await onlyFlagged.click();
  await settle(page);
  await shot(page, '03-collection-unscored-chips');

  const chipCard = page.locator('.chip-card').first();
  await chipCard.waitFor({ state: 'visible', timeout: 10000 });
  await chipCard.click();
  await page.waitForSelector('.chip-drawer', { timeout: 10000 });
  await settle(page);
  await page.waitForTimeout(1500); // let the parent-tile <canvas> finish drawing its Image.onload
  await shot(page, '04-det-chip-drawer-unscored');
  await page.locator('.modal-close').click();
  await page.waitForTimeout(200);

  await page.locator('button.nav-item', { hasText: 'Review (field)' }).click();
  await settle(page);
  await shot(page, '05-field-review-empty-queue');

  await page.locator('button.nav-item', { hasText: 'Report' }).click();
  await settle(page);
  await shot(page, '06-report');

  await browser.close();
  console.log('done');
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
