// End-to-end analyst path against the REAL backend: Tirana collection →
// label 5 flagged chips "Military fighter" (key 2) + confirm 1 (key 1) from the
// chip modal → Review (field) → Apply → field compare. MUTATES demo state
// (reviews + a v2 version); run `bash scripts/reset_demo.sh` afterwards.
// Saves frontend/screenshots/field-final-09*.png.
import { chromium } from 'playwright';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const SHOT_DIR = path.join(__dirname, '..', 'screenshots');
const BASE_URL = process.env.BASE_URL ?? 'http://localhost:5173/';
const CHROME_PATH = process.env.CHROME_PATH; // unset = Playwright's bundled Chromium

async function shot(page, name, opts = {}) {
  const file = path.join(SHOT_DIR, `field-final-${name}.png`);
  await page.screenshot({ path: file, fullPage: !!opts.fullPage });
  console.log('saved', file);
}

async function settle(page, ms = 500) {
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

  await page.goto(BASE_URL);
  await settle(page);
  await page.getByPlaceholder('filter airfield / country…').fill('Tirana');
  await settle(page, 300);
  await page.locator('table.data-table tbody tr', { hasText: '2019-02-07' }).first().click();
  await settle(page);
  await page.locator('.chip-card').first().waitFor({ state: 'visible', timeout: 15000 });
  await page.locator('.chip-card').first().click();
  await page.waitForSelector('.chip-drawer', { timeout: 10000 });
  await settle(page);

  const t0 = Date.now();
  for (let i = 0; i < 5; i++) {
    await page.locator('.drawer-nav', { hasText: `Detection ${i + 1} of` }).waitFor({ timeout: 10000 });
    await page.locator('.review-actions').waitFor({ state: 'visible', timeout: 15000 });
    await page.waitForTimeout(150);
    await page.keyboard.press('2');
    const input = page.locator('#new-class-label');
    await input.waitFor({ state: 'visible', timeout: 5000 });
    await input.fill('Military fighter');
    await input.press('Enter');
    await page.locator('.review-status-banner', { hasText: 'Military fighter' }).waitFor({ timeout: 10000 });
    if (i === 0) await shot(page, '09a-labeled-fighter-drawer');
    await page.keyboard.press('ArrowRight');
    await page.waitForTimeout(700);
  }
  // Confirm one (the 6th chip, whatever it is)
  await page.locator('.drawer-nav', { hasText: 'Detection 6 of' }).waitFor({ timeout: 10000 });
  await page.locator('.review-actions').waitFor({ state: 'visible', timeout: 15000 });
  await page.waitForTimeout(150);
  await page.keyboard.press('1');
  await page.locator('.review-status-banner', { hasText: 'confirmed' }).waitFor({ timeout: 10000 });
  console.log('labeling took', ((Date.now() - t0) / 1000).toFixed(1), 's');
  await page.locator('.modal-close').click();
  await settle(page);
  await shot(page, '09b-tirana-after-labels');

  await page.locator('button.nav-item', { hasText: 'Review (field)' }).click();
  await settle(page);
  await page.waitForSelector('.review-list-item', { timeout: 20000 });
  await page.getByRole('button', { name: 'Details' }).first().click();
  await page.waitForTimeout(400);
  await shot(page, '09c-field-review-pending');

  const apply = page.getByRole('button', { name: /Apply 6 feedback items/ });
  await apply.waitFor({ timeout: 10000 });
  const t1 = Date.now();
  await apply.click();
  await page.getByText('What changed —').waitFor({ timeout: 240000 });
  console.log('apply + compare took', ((Date.now() - t1) / 1000).toFixed(1), 's');
  await settle(page, 800);
  await page.getByText('What changed —').scrollIntoViewIfNeeded();
  await page.waitForTimeout(300);
  await shot(page, '09-field-compare');
  await shot(page, '09-field-compare-full', { fullPage: true });

  // Collections page after adapt — statuses recomputed from v2 scores.
  await page.locator('button.nav-item', { hasText: 'Collections' }).click();
  await settle(page);
  await shot(page, '09d-collections-after-apply');

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
