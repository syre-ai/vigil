// Screenshot the overhauled Report page (analyst view) against the live backend via the
// dev server on :5173. Saves frontend/screenshots/report-final.png (full page) and logs any
// page/console errors or HTTP >= 400 responses. Read-only: never posts reviews or adapts.
import { chromium } from 'playwright';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const SHOT_DIR = path.join(__dirname, '..', 'screenshots');
const BASE_URL = process.env.BASE_URL ?? 'http://localhost:5173/';
const VIEW = process.env.VIEW ?? 'analyst';
const OUT = process.env.OUT ?? 'report-final.png';
const CHROME_PATH = process.env.CHROME_PATH; // unset = Playwright's bundled Chromium

fs.mkdirSync(SHOT_DIR, { recursive: true });

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

  await page.addInitScript((view) => {
    try {
      window.localStorage.setItem('vigil.view', view);
      window.sessionStorage.clear();
    } catch {
      /* ignore */
    }
  }, VIEW);

  await page.goto(BASE_URL);
  await page.getByRole('button', { name: /^Report/ }).click();
  await page.locator('.report-page').waitFor({ state: 'visible', timeout: 15000 });
  try {
    await page.waitForLoadState('networkidle', { timeout: 10000 });
  } catch {
    /* fall through */
  }
  await page.waitForTimeout(800);
  const file = path.join(SHOT_DIR, OUT);
  await page.screenshot({ path: file, fullPage: true });
  console.log('saved', file);

  // Report the smallest rendered font size on the page (nothing should be under 11px).
  const minFont = await page.evaluate(() => {
    let min = Infinity;
    for (const el of document.querySelectorAll('.report-page *')) {
      if (!el.textContent?.trim()) continue;
      const fs = parseFloat(getComputedStyle(el).fontSize);
      if (fs < min) min = fs;
    }
    return min;
  });
  console.log('min font-size on the report page:', minFont);

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
