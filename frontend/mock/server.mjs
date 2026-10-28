#!/usr/bin/env node
// Mock backend for Vigil. Plain node http, no deps. Implements every
// endpoint in backend/API.md against the synthetic dataset in ./data.mjs.
import http from 'node:http';
import { URL } from 'node:url';
import * as api from './data.mjs';
import { chipImage, tileImage } from './images.mjs';

const PORT = Number(process.env.PORT) || 8000;

function sendJSON(res, status, body) {
  const buf = Buffer.from(JSON.stringify(body));
  res.writeHead(status, {
    'Content-Type': 'application/json',
    'Access-Control-Allow-Origin': '*',
    'Content-Length': buf.length,
  });
  res.end(buf);
}

function sendPNG(res, buf) {
  res.writeHead(200, { 'Content-Type': 'image/png', 'Access-Control-Allow-Origin': '*', 'Content-Length': buf.length, 'Cache-Control': 'public, max-age=3600' });
  res.end(buf);
}

function sendMarkdown(res, text) {
  const buf = Buffer.from(text);
  res.writeHead(200, { 'Content-Type': 'text/markdown; charset=utf-8', 'Access-Control-Allow-Origin': '*', 'Content-Length': buf.length });
  res.end(buf);
}

function notFound(res, msg = 'not found') {
  sendJSON(res, 404, { detail: msg });
}

function readBody(req) {
  return new Promise((resolve, reject) => {
    let data = '';
    req.on('data', (chunk) => (data += chunk));
    req.on('end', () => {
      if (!data) return resolve({});
      try {
        resolve(JSON.parse(data));
      } catch {
        reject(new Error('invalid JSON body'));
      }
    });
    req.on('error', reject);
  });
}

const server = http.createServer(async (req, res) => {
  if (req.method === 'OPTIONS') {
    res.writeHead(204, {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET,POST,OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type',
    });
    return res.end();
  }

  const url = new URL(req.url, `http://localhost:${PORT}`);
  const path = url.pathname.replace(/^\/api/, '') || '/';
  const q = Object.fromEntries(url.searchParams.entries());
  let m;

  try {
    if (req.method === 'GET' && path === '/summary') {
      return sendJSON(res, 200, api.getSummary());
    }
    if (req.method === 'GET' && path === '/batches') {
      return sendJSON(res, 200, api.getBatches());
    }
    if (req.method === 'GET' && (m = path.match(/^\/batches\/([^/]+)$/))) {
      const b = api.getBatchDetail(m[1]);
      return b ? sendJSON(res, 200, b) : notFound(res, `batch ${m[1]} not found`);
    }
    if (req.method === 'GET' && (m = path.match(/^\/batches\/([^/]+)\/chips$/))) {
      const b = api.getBatchDetail(m[1]);
      if (!b) return notFound(res, `batch ${m[1]} not found`);
      return sendJSON(res, 200, api.getBatchChipsPage(m[1], q));
    }
    if (req.method === 'GET' && (m = path.match(/^\/chips\/(\d+)$/))) {
      const c = api.getChipDetail(m[1]);
      return c ? sendJSON(res, 200, c) : notFound(res, `chip ${m[1]} not found`);
    }
    if (req.method === 'GET' && (m = path.match(/^\/chips\/(\d+)\/image$/))) {
      const info = api.getChipImageInfo(m[1]);
      if (!info) return notFound(res, `chip ${m[1]} not found`);
      return sendPNG(res, chipImage(info.seed, info.role, info.isNovel));
    }
    if (req.method === 'GET' && (m = path.match(/^\/tiles\/(\d+)\/image$/))) {
      const info = api.getTileImageInfo(m[1]);
      if (!info) return notFound(res, `tile ${m[1]} not found`);
      return sendPNG(res, tileImage(info.seed, info.weather));
    }
    if (req.method === 'GET' && (m = path.match(/^\/tiles\/(\d+)$/))) {
      const t = api.getTileDetail(m[1]);
      return t ? sendJSON(res, 200, t) : notFound(res, `tile ${m[1]} not found`);
    }
    if (req.method === 'GET' && path === '/embedding/points') {
      return sendJSON(res, 200, api.getEmbeddingPoints(q.batch));
    }
    if (req.method === 'GET' && path === '/drift') {
      return sendJSON(res, 200, api.getDrift());
    }
    if (req.method === 'GET' && path === '/envelope') {
      return sendJSON(res, 200, api.getEnvelope());
    }
    if (req.method === 'GET' && path === '/eval') {
      return sendJSON(res, 200, api.getEval());
    }
    if (req.method === 'GET' && path === '/review/queue') {
      if (q.source === 'det') return sendJSON(res, 200, api.getFieldReviewQueue(q));
      return sendJSON(res, 200, api.getReviewQueue(q));
    }
    if (req.method === 'GET' && path === '/review/log') {
      return sendJSON(res, 200, api.getReviewLog(q.source));
    }
    if (req.method === 'GET' && path === '/review/labels') {
      // distinct non-null labels ever used across all reviews, most-recently-used first
      return sendJSON(res, 200, { labels: api.getUsedLabels() });
    }
    if (req.method === 'GET' && path === '/review/classes') {
      return sendJSON(res, 200, api.getReviewClasses());
    }
    if (req.method === 'POST' && (m = path.match(/^\/review\/(\d+)$/))) {
      const body = await readBody(req);
      if (!body.action) return sendJSON(res, 400, { detail: 'action is required' });
      const review = api.postReview(m[1], body, q.source);
      return review ? sendJSON(res, 200, review) : notFound(res, `chip ${m[1]} not found`);
    }
    if (req.method === 'POST' && path === '/adapt') {
      const body = await readBody(req).catch(() => ({}));
      return sendJSON(res, 200, api.postAdapt(body));
    }
    if (req.method === 'GET' && path === '/versions/compare') {
      const result = api.getVersionsCompare(q.a, q.b);
      return result ? sendJSON(res, 200, result) : sendJSON(res, 400, { detail: 'unknown version(s)' });
    }
    if (req.method === 'GET' && path === '/map') {
      return sendJSON(res, 200, api.getMap());
    }
    if (req.method === 'GET' && path === '/report') {
      return sendMarkdown(res, api.getReportMarkdown());
    }

    // ---- field mode (collections / det chips) ----
    if (req.method === 'GET' && path === '/collections') {
      return sendJSON(res, 200, api.getCollections());
    }
    if (req.method === 'GET' && path === '/collections/summary') {
      return sendJSON(res, 200, api.getCollectionsSummary());
    }
    if (req.method === 'GET' && (m = path.match(/^\/collections\/([^/]+)$/))) {
      const c = api.getCollectionDetail(decodeURIComponent(m[1]), q);
      return c ? sendJSON(res, 200, c) : notFound(res, `collection ${m[1]} not found`);
    }
    if (req.method === 'GET' && (m = path.match(/^\/det\/(\d+)$/))) {
      const c = api.getDetDetail(m[1]);
      return c ? sendJSON(res, 200, c) : notFound(res, `det chip ${m[1]} not found`);
    }
    if (req.method === 'GET' && (m = path.match(/^\/det\/(\d+)\/image$/))) {
      const info = api.getDetImageInfo(m[1]);
      if (!info) return notFound(res, `det chip ${m[1]} not found`);
      return sendPNG(res, chipImage(info.seed, info.role, info.isNovel));
    }

    return notFound(res, `no route for ${req.method} ${path}`);
  } catch (err) {
    return sendJSON(res, 500, { detail: String(err?.message || err) });
  }
});

server.listen(PORT, () => {
  console.log(`Vigil mock API listening on http://localhost:${PORT}/api`);
});
