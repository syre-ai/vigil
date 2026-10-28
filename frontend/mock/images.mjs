// Procedural placeholder imagery for chips and tiles. No source imagery is
// available in data/chips for this mock, so every "photo" is a deterministic
// synthetic PNG: a tinted, noised background plus a simple silhouette so
// chips are visually distinguishable in a grid.
import { encodePNG } from './png.mjs';
import { mulberry32, clamp8 } from './rng.mjs';

const ROLE_COLORS = {
  'Small Civil Transport/Utility': [92, 138, 148],
  'Medium Civil Transport/Utility': [110, 128, 168],
  'Large Civil Transport/Utility': [126, 110, 158],
  'Military Fighter/Interceptor/Attack': [150, 84, 78],
  'Military Bomber': [120, 70, 74],
  'Military Transport/Cargo': [104, 96, 70],
  'Military Trainer': [96, 110, 84],
  'Not Aircraft': [80, 80, 80],
};

const WEATHER_COLORS = {
  Clear: [58, 68, 78],
  Haze: [128, 130, 128],
  Snow: [206, 210, 216],
};

function roleColor(role) {
  return ROLE_COLORS[role] || [100, 110, 120];
}

const chipCache = new Map();
const tileCache = new Map();

export function chipImage(seed, role, isNovel) {
  const key = `${seed}:${role}:${isNovel}`;
  if (chipCache.has(key)) return chipCache.get(key);
  const w = 224;
  const h = 224;
  const buf = Buffer.alloc(w * h * 3);
  const rnd = mulberry32(seed);
  const [r0, g0, b0] = roleColor(role);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const n = (rnd() - 0.5) * 26;
      const idx = (y * w + x) * 3;
      buf[idx] = clamp8(r0 + n);
      buf[idx + 1] = clamp8(g0 + n);
      buf[idx + 2] = clamp8(b0 + n);
    }
  }
  // simple cross-shaped silhouette to suggest a detected aircraft
  const cx = w / 2 + (rnd() - 0.5) * 20;
  const cy = h / 2 + (rnd() - 0.5) * 20;
  const armX = w * (0.32 + rnd() * 0.08);
  const armY = h * 0.05;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const inWing = Math.abs(x - cx) < armX && Math.abs(y - cy) < armY;
      const inBody = Math.abs(y - cy) < armX * 0.55 && Math.abs(x - cx) < armY * 1.3;
      if (inWing || inBody) {
        const idx = (y * w + x) * 3;
        buf[idx] = clamp8(buf[idx] * 0.5);
        buf[idx + 1] = clamp8(buf[idx + 1] * 0.5);
        buf[idx + 2] = clamp8(buf[idx + 2] * 0.5);
      }
    }
  }
  if (isNovel) drawBorder(buf, w, h, 5, [232, 168, 62]);
  const png = encodePNG(w, h, buf);
  chipCache.set(key, png);
  return png;
}

export function tileImage(seed, weather) {
  const key = `${seed}:${weather}`;
  if (tileCache.has(key)) return tileCache.get(key);
  const w = 512;
  const h = 512;
  const buf = Buffer.alloc(w * h * 3);
  const rnd = mulberry32(seed);
  const [r0, g0, b0] = WEATHER_COLORS[weather] || WEATHER_COLORS.Clear;
  // coarse blocky terrain-ish noise, cheap to generate
  const block = 8;
  for (let by = 0; by < h; by += block) {
    for (let bx = 0; bx < w; bx += block) {
      const n = (rnd() - 0.5) * 34;
      const rr = clamp8(r0 + n);
      const gg = clamp8(g0 + n);
      const bb = clamp8(b0 + n);
      for (let y = by; y < Math.min(by + block, h); y++) {
        for (let x = bx; x < Math.min(bx + block, w); x++) {
          const idx = (y * w + x) * 3;
          buf[idx] = rr;
          buf[idx + 1] = gg;
          buf[idx + 2] = bb;
        }
      }
    }
  }
  const png = encodePNG(w, h, buf);
  tileCache.set(key, png);
  return png;
}

function drawBorder(buf, w, h, thickness, color) {
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const onBorder = x < thickness || y < thickness || x >= w - thickness || y >= h - thickness;
      if (onBorder) {
        const idx = (y * w + x) * 3;
        buf[idx] = color[0];
        buf[idx + 1] = color[1];
        buf[idx + 2] = color[2];
      }
    }
  }
}
