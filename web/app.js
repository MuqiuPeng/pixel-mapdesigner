'use strict';
// The editor keeps a copy of the open map's cells and a picture of it. The picture is drawn by
// the server (render.py), because the ground is not a grid of stamps: borders wander, texture
// never repeats, tall things overlap. A stroke paints flat colour at once, is sent as a list
// of changed cells, and then the part of the picture around it is fetched again.
// The files on disk are the truth: /api/rev is polled, and when something else (the MCP
// server) has changed them the copy is thrown away and reloaded.

const $ = s => document.querySelector(s);
const $$ = s => [...document.querySelectorAll(s)];
const enc = encodeURIComponent;
const LAYERS = ['ground', 'plant', 'object'];
const ALPHABET = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789';
const PLACEHOLDER = new Set(['body', 'body_s', 'acc', 'acc_s']);  // recolour keys for characters
const T = 16;
const REACH = 4;   // cells around a change whose picture may differ (tall sprites, shadows, cliffs)
const CHUNK = 20;  // cells per side of the pieces the picture is loaded in
const WALK = 32;   // screen pixels per cell when walking: the art at 2x, the size it is played at
const PAN = 14;    // cells per second that WASD moves the view
const KEEP = 48;   // moving pieces kept in memory; the furthest from the view are dropped first

const S = {
  palette: null, shapes: [], tiles: [], byId: {}, img: {}, maps: [],
  map: null, frames: [], f: 0, play: true, nFrames: 8, fps: 6, motions: [], speckAnims: [],
  rev: {tileset: null, map: null},
  sel: null, tool: 'brush', grid: false, contours: false, geo: null,
  cell: 32, ox: 0, oy: 0, hover: null, space: false,
  // Two ways to look. 'whole': the whole map fitted to the window, a still picture. 'walk': at
  // the size it is played at, moved with WASD, never zoomed; only what is in view is drawn, and
  // only that moves. still: one picture of the whole map (frame 0), filled in piece by piece;
  // have: pieces of it that are up to date; anim: piece -> all frames of that piece.
  view: 'whole', still: null, have: new Set(), anim: new Map(), keys: new Set(),
  undo: [], redo: [], inflight: 0, seq: 0, needReload: false, dirty: false,
};

async function api(method, url, body) {
  const r = await fetch(url, {
    method, headers: body ? {'Content-Type': 'application/json'} : {},
    body: body ? JSON.stringify(body) : undefined,
  });
  const data = await r.json().catch(() => ({error: r.statusText}));
  if (!r.ok) throw new Error(data.error || r.statusText);
  return data;
}

// ---------------------------------------------------------------- messages

function dialogResult(title, body, withCancel) {
  $('#msgTitle').textContent = title;
  $('#msgBody').textContent = body;
  $('#msgCancel').hidden = !withCancel;
  const d = $('#msgDlg');
  d.returnValue = 'cancel';
  d.showModal();
  return new Promise(done => d.addEventListener('close', () => done(d.returnValue === 'ok'), {once: true}));
}
const msg = (title, body) => dialogResult(title, body, false);
const ask = (title, body) => dialogResult(title, body, true);

let syncTimer = 0;
function sync(text, warn, hold) {
  const el = $('#sync');
  el.textContent = text;
  el.classList.toggle('warn', !!warn);
  clearTimeout(syncTimer);
  if (hold) syncTimer = setTimeout(() => sync('已保存'), hold);
}

// ---------------------------------------------------------------- tiles

function artCanvas(art) {
  const c = document.createElement('canvas');
  c.width = art.rows[0].length; c.height = art.rows.length;
  const g = c.getContext('2d');
  art.rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const name = art.key[row[x]];
      if (name) { g.fillStyle = S.palette.colors[name]; g.fillRect(x, y, 1, 1); }
    }
  });
  return c;
}

function tilesOf(layer) {
  const list = S.tiles.filter(t => t.layer === layer);
  return layer === 'ground' ? list.sort((a, b) => a.rank - b.rank) : list;
}

function renderTiles() {
  for (const layer of LAYERS) {
    const box = $('#tiles-' + layer);
    box.replaceChildren();
    for (const t of tilesOf(layer)) {
      const el = document.createElement('div');
      el.className = 'tile' + (t.id === S.sel ? ' on' : '');
      el.title = `${t.name}（${t.id}）· ${t.pass}${t.tags.length ? ' · ' + t.tags.join(', ') : ''}\n双击编辑`;
      let icon;
      if (layer !== 'object') {   // grounds and what grows on them: a square of the real thing
        icon = new Image();
        icon.src = `/api/swatch/${enc(t.id)}.png?rev=${S.rev.tileset}`;
      } else {
        icon = document.createElement('canvas');
        icon.width = icon.height = 32;
        const src = S.img[t.id], fit = 32 / Math.max(src.width, src.height), g = icon.getContext('2d');
        const k = fit >= 1 ? Math.min(4, Math.floor(fit)) : fit;  // small things enlarged by whole steps
        g.imageSmoothingEnabled = false;
        g.drawImage(src, Math.round((32 - src.width * k) / 2), Math.round(32 - src.height * k), src.width * k, src.height * k);
      }
      const label = document.createElement('span');
      label.textContent = t.name;
      el.append(icon, label);
      el.onclick = () => { S.sel = t.id; if (S.tool === 'erase' || S.tool === 'pick') setTool('brush'); renderTiles(); };
      el.ondblclick = () => (layer === 'ground' ? openLook(t) : layer === 'object' ? openSprite(t)
        : msg(t.name, '植被是长出来的，不是画出来的：它的生长方式（高矮、疏密、颜色、越出格子多远）用 MCP 的 save_tile 改 grow。\n' + JSON.stringify(t.grow)));
      box.append(el);
    }
  }
}

function setTool(tool) {
  S.tool = tool;
  $$('#tools button').forEach(b => b.classList.toggle('on', b.dataset.tool === tool));
}

// ---------------------------------------------------------------- loading

async function loadState() {
  const s = await api('GET', '/api/state');
  S.palette = s.palette;
  S.shapes = s.shapes;
  S.nFrames = s.frames; S.fps = s.fps; S.motions = s.motions; S.speckAnims = s.speck_anims; S.geo = s.geo;
  S.tiles = s.tileset.tiles;
  S.byId = Object.fromEntries(S.tiles.map(t => [t.id, t]));
  S.img = Object.fromEntries(S.tiles.filter(t => t.layer === 'object').map(t => [t.id, artCanvas(t.variants[0])]));
  S.maps = s.maps;
  S.rev.tileset = s.rev.tileset;
  if (!S.byId[S.sel]) S.sel = S.tiles.length ? tilesOf('ground').concat(tilesOf('object'))[0].id : null;
  renderTiles();
  renderMapSel();
  draw();
}

function renderMapSel() {
  const sel = $('#mapSel');
  sel.replaceChildren(...S.maps.map(m => new Option(m.name, m.name)));
  if (S.map) sel.value = S.map.name;
  for (const id of ['resizeMap', 'delMap', 'validate', 'png']) $('#' + id).disabled = !S.map;
}

// Everything that touches the server's copy of the map or our picture of it goes through one
// queue, so a picture fetched before a save can never land on top of one fetched after it.
let chain = Promise.resolve();
function queue(fn) {
  S.inflight++;
  chain = chain.then(fn).catch(e => { S.needReload = true; sync('出错：' + e.message, true); }).finally(() => { S.inflight--; });
  return chain;
}

async function picture(m, x0, y0, x1, y1, all) {
  // frame 0 only, or every frame of the loop stacked top to bottom in one picture
  const r = await fetch(`/api/map/${enc(m.name)}/render.png?scale=1&${all ? 'frames=all' : 'frame=0'}` +
                        `&x=${x0}&y=${y0}&w=${x1 - x0}&h=${y1 - y0}`);
  if (!r.ok) throw new Error('画面没有取到');
  return createImageBitmap(await r.blob());
}

// The whole view's own picture, at the size it is shown (cell px per cell). Pieces are scaled
// into it once as they come, so a repaint is one small copy, not a shrink of the whole map.
function small() {
  const m = S.map;
  if (!m || S.view !== 'whole') return null;
  const w = Math.max(1, Math.round(m.w * S.cell)), h = Math.max(1, Math.round(m.h * S.cell));
  if (!S.small || S.small.width !== w || S.small.height !== h || S.small.map !== m) {
    S.small = document.createElement('canvas');
    S.small.width = w; S.small.height = h; S.small.map = m;
    const g = S.small.getContext('2d');
    g.imageSmoothingEnabled = true;
    g.drawImage(S.still, 0, 0, w, h);      // what has come so far, once
  }
  return S.small;
}

function toSmall(x0, y0, img, cw, ch, whole = false) {   // whole: img is the full still picture
  const sm = small();
  if (!sm) return;
  const g = sm.getContext('2d'), k = sm.width / S.map.w;
  g.imageSmoothingEnabled = true;
  const X0 = Math.floor(x0 * k), Y0 = Math.floor(y0 * k), X1 = Math.ceil((x0 + cw) * k), Y1 = Math.ceil((y0 + ch) * k);
  const sx = X0 * T / k - (whole ? 0 : x0 * T), sy = Y0 * T / k - (whole ? 0 : y0 * T);
  g.drawImage(img, sx, sy, (X1 - X0) * T / k, (Y1 - Y0) * T / k, X0, Y0, X1 - X0, Y1 - Y0);
}

// After an edit: that part of the still picture again, and its moving pieces are out of date.
async function refresh(m, x, y, w, h) {
  const x0 = Math.max(0, x), y0 = Math.max(0, y), x1 = Math.min(m.w, x + w), y1 = Math.min(m.h, y + h);
  if (x0 >= x1 || y0 >= y1 || S.map !== m) return;
  const bmp = await picture(m, x0, y0, x1, y1, false);
  if (S.map !== m) return;
  const g = S.still.getContext('2d');
  g.clearRect(x0 * T, y0 * T, bmp.width, bmp.height);
  g.drawImage(bmp, x0 * T, y0 * T);
  if (S.small && S.small.map === m) S.small = null;    // rebuilt from the still picture at the next paint
  for (const k of [...S.anim.keys()]) {
    const [cx, cy] = k.split(',').map(Number);
    if (cx < x1 && cx + CHUNK > x0 && cy < y1 && cy + CHUNK > y0) S.anim.delete(k);
  }
  want();
  draw();
}

// The pieces the view needs, nearest the middle of the window first: the still picture always;
// the moving one too when walking with the animation on.
const loading = new Set();
let wanted = [];
function want() {
  const m = S.map;
  if (!m) return;
  const W = cv.clientWidth, H = cv.clientHeight, c = S.cell;
  const margin = S.view === 'walk' ? CHUNK : 0;   // a piece ahead in every direction, for walking into
  const x0 = Math.max(0, Math.floor((-S.ox / c - margin) / CHUNK) * CHUNK), y0 = Math.max(0, Math.floor((-S.oy / c - margin) / CHUNK) * CHUNK);
  const x1 = Math.min(m.w, (W - S.ox) / c + margin), y1 = Math.min(m.h, (H - S.oy) / c + margin);
  const mx = (W / 2 - S.ox) / c, my = (H / 2 - S.oy) / c, jobs = [];
  for (let y = y0; y < y1; y += CHUNK) {
    for (let x = x0; x < x1; x += CHUNK) {
      const k = x + ',' + y, d = Math.hypot(x + CHUNK / 2 - mx, y + CHUNK / 2 - my);
      if (!S.have.has(k)) jobs.push({k, x, y, all: false, d});
      if (S.view === 'walk' && S.play && !S.anim.has(k)) jobs.push({k, x, y, all: true, d: d + 1e4});
    }
  }
  wanted = jobs.sort((a, b) => a.d - b.d);
  pump();
}

function pump() {
  const m = S.map;
  while (loading.size < 12 && wanted.length) {  // the server draws up to this many at once, in its own processes
    const j = wanted.shift(), id = j.k + (j.all ? '*' : '');
    if (loading.has(id)) continue;
    loading.add(id);
    picture(m, j.x, j.y, Math.min(m.w, j.x + CHUNK), Math.min(m.h, j.y + CHUNK), j.all).then(bmp => {
      if (S.map !== m) return;
      if (j.all) {
        S.anim.set(j.k, bmp);
        if (S.anim.size > KEEP) {       // drop the moving pieces furthest from the view
          const W = cv.clientWidth, H = cv.clientHeight, mx = (W / 2 - S.ox) / S.cell, my = (H / 2 - S.oy) / S.cell;
          const far = [...S.anim.keys()].sort((a, b) => {
            const [ax, ay] = a.split(',').map(Number), [bx, by] = b.split(',').map(Number);
            return Math.hypot(bx - mx, by - my) - Math.hypot(ax - mx, ay - my);
          });
          for (const k of far.slice(0, S.anim.size - KEEP)) S.anim.get(k).close(), S.anim.delete(k);
        }
      } else {
        const g = S.still.getContext('2d');
        g.clearRect(j.x * T, j.y * T, bmp.width, bmp.height);
        g.drawImage(bmp, j.x * T, j.y * T);
        toSmall(j.x, j.y, bmp, bmp.width / T, bmp.height / T);
        S.have.add(j.k);
        bmp.close();
      }
      draw();
    }).catch(e => sync('出错：' + e.message, true)).finally(() => { loading.delete(id); pump(); });
  }
}

async function loadMap(name, keepView) {
  const d = await api('GET', '/api/map/' + enc(name));
  const old = S.map, m = {name: d.name, w: d.width, h: d.height, layers: {}, relief: d.relief ? d.relief.flat() : null,
                          biome: d.biome ? d.biome.cells.flat().map(i => d.biome.legend[i] || null) : null};
  for (const L of LAYERS) m.layers[L] = d[L].flat().map(i => (i < 0 ? null : d.legend[i]));
  if (!(keepView && old && S.still && old.w === m.w && old.h === m.h)) {
    // otherwise the old picture stays up while the new one arrives
    S.still = document.createElement('canvas');
    S.still.width = m.w * T; S.still.height = m.h * T;
  }
  S.have = new Set();
  for (const bmp of S.anim.values()) bmp.close();
  S.anim = new Map();
  S.map = m;
  S.rev.map = d.rev;
  S.undo = []; S.redo = []; S.needReload = false;
  try { localStorage.setItem('mapdesigner.map', name); } catch (e) { /* private mode */ }
  if (!keepView) fitView();
  renderMapSel();
  updateHistory();
  status();
  draw();
  want();
}

async function openFirstMap() {
  let want = null;
  try { want = localStorage.getItem('mapdesigner.map'); } catch (e) { /* private mode */ }
  const m = S.maps.find(m => m.name === want) || S.maps[0];
  if (m) await loadMap(m.name);
  else { S.map = null; renderMapSel(); draw(); }
}

// ---------------------------------------------------------------- the map view

const cv = $('#map');

function fitView(keepMiddle) {
  const W = cv.clientWidth, H = cv.clientHeight, m = S.map;
  if (!m) return;
  if (S.view === 'whole') {
    // the whole map in the window, as large as fits
    S.cell = Math.min((W - 24) / m.w, (H - 24) / m.h);
    S.ox = Math.round((W - m.w * S.cell) / 2);
    S.oy = Math.round((H - m.h * S.cell) / 2);
  } else {
    // at the size it is played at, on the same middle as before (or the middle of the map)
    const mx = keepMiddle ? (W / 2 - S.ox) / S.cell : m.w / 2, my = keepMiddle ? (H / 2 - S.oy) / S.cell : m.h / 2;
    S.cell = WALK;
    S.ox = Math.round(W / 2 - mx * WALK);
    S.oy = Math.round(H / 2 - my * WALK);
    clampView();
  }
}

function clampView() {
  // walking may go a little past the edge of the map, not off it
  const W = cv.clientWidth, H = cv.clientHeight, m = S.map, c = S.cell, pad = 4 * c;
  const lo = (win, size) => Math.min(pad, win - size * c - pad), hi = (win, size) => Math.max(pad, win - size * c - pad);
  S.ox = Math.round(Math.min(Math.max(S.ox, Math.min(lo(W, m.w), (W - m.w * c) / 2)), Math.max(hi(W, m.w), (W - m.w * c) / 2)));
  S.oy = Math.round(Math.min(Math.max(S.oy, Math.min(lo(H, m.h), (H - m.h * c) / 2)), Math.max(hi(H, m.h), (H - m.h * c) / 2)));
}

function setView(view) {
  S.view = view;
  try { localStorage.setItem('mapdesigner.view', view); } catch (e) { /* private mode */ }
  $$('#views button').forEach(b => b.classList.toggle('on', b.dataset.view === view));
  $('#play').disabled = view === 'whole';
  $('#playRow').title = view === 'whole' ? '全图是静态的；切到「行走比例」看动画' : '';
  fitView(true);
  status();
  draw();
  want();
}

function draw() {
  if (S.dirty) return;
  S.dirty = true;
  requestAnimationFrame(paint);
  setTimeout(paint, 60);  // frames do not run while the page is hidden
}

function paint() {
  if (!S.dirty) return;
  S.dirty = false;
  const dpr = window.devicePixelRatio || 1, W = cv.clientWidth, H = cv.clientHeight;
  if (cv.width !== Math.round(W * dpr) || cv.height !== Math.round(H * dpr)) {
    cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr);
  }
  const g = cv.getContext('2d');
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.imageSmoothingEnabled = false;
  g.fillStyle = '#14121a';
  g.fillRect(0, 0, W, H);
  const m = S.map;
  if (!m) {
    g.fillStyle = '#9a96a6';
    g.font = '14px sans-serif';
    g.fillText('还没有地图，点上方「新建」。', 24, 36);
    return;
  }
  const c = S.cell, ox = S.ox, oy = S.oy;
  // only the part of the picture in the window
  const vx0 = Math.max(0, Math.floor(-ox / c)), vy0 = Math.max(0, Math.floor(-oy / c));
  const vx1 = Math.min(m.w, Math.ceil((W - ox) / c)), vy1 = Math.min(m.h, Math.ceil((H - oy) / c));
  const sm = S.view === 'whole' ? small() : null;
  if (sm) {
    g.drawImage(sm, ox, oy);           // already the size it is shown
  } else if (vx1 > vx0 && vy1 > vy0) {
    g.imageSmoothingEnabled = c < T;   // the whole map shrunk: smooth; at whole multiples: crisp pixels
    g.drawImage(S.still, vx0 * T, vy0 * T, (vx1 - vx0) * T, (vy1 - vy0) * T, ox + vx0 * c, oy + vy0 * c, (vx1 - vx0) * c, (vy1 - vy0) * c);
    g.imageSmoothingEnabled = false;
  }
  if (S.view === 'walk' && S.play) {
    const f = S.f % S.nFrames;
    for (let y = Math.floor(vy0 / CHUNK) * CHUNK; y < vy1; y += CHUNK) {
      for (let x = Math.floor(vx0 / CHUNK) * CHUNK; x < vx1; x += CHUNK) {
        const bmp = S.anim.get(x + ',' + y);
        if (!bmp) continue;
        const pw = bmp.width, ph = bmp.height / S.nFrames;
        g.drawImage(bmp, 0, f * ph, pw, ph, ox + x * c, oy + y * c, pw / T * c, ph / T * c);
      }
    }
  }
  if (S.grid && c >= 8) {
    const x0 = Math.max(0, Math.floor(-ox / c)), y0 = Math.max(0, Math.floor(-oy / c));
    const x1 = Math.min(m.w, Math.ceil((W - ox) / c)), y1 = Math.min(m.h, Math.ceil((H - oy) / c));
    g.strokeStyle = 'rgba(0,0,0,.3)';
    g.lineWidth = 1;
    g.beginPath();
    for (let x = x0; x <= x1; x++) { g.moveTo(ox + x * c + .5, oy + y0 * c); g.lineTo(ox + x * c + .5, oy + y1 * c); }
    for (let y = y0; y <= y1; y++) { g.moveTo(ox + x0 * c, oy + y * c + .5); g.lineTo(ox + x1 * c, oy + y * c + .5); }
    g.stroke();
  }
  if (S.contours && m.relief && c >= 8) {
    // A line wherever the ground steps from one level to the next; every fifth level heavier.
    const x0 = Math.max(0, Math.floor(-ox / c)), y0 = Math.max(0, Math.floor(-oy / c));
    const x1 = Math.min(m.w, Math.ceil((W - ox) / c)), y1 = Math.min(m.h, Math.ceil((H - oy) / c));
    const lev = i => Math.ceil(m.relief[i] / S.geo.fine);
    for (const heavy of [false, true]) {
      g.strokeStyle = heavy ? 'rgba(23,22,28,.75)' : 'rgba(23,22,28,.4)';
      g.lineWidth = heavy ? 2 : 1;
      g.beginPath();
      for (let y = y0; y < y1; y++) {
        for (let x = x0; x < x1; x++) {
          const a = lev(y * m.w + x);
          for (const [nx, ny, ax, ay, bx, by] of [[x + 1, y, 1, 0, 1, 1], [x, y + 1, 0, 1, 1, 1]]) {
            if (nx >= m.w || ny >= m.h) continue;
            const b = lev(ny * m.w + nx);
            if (a === b || (Math.max(a, b) % 5 === 0) !== heavy) continue;
            g.moveTo(ox + (x + ax) * c + .5, oy + (y + ay) * c + .5);
            g.lineTo(ox + (x + bx) * c + .5, oy + (y + by) * c + .5);
          }
        }
      }
      g.stroke();
    }
    g.lineWidth = 1;
  }
  g.strokeStyle = '#9a96a6';
  g.strokeRect(ox - .5, oy - .5, m.w * c + 1, m.h * c + 1);
  if (drag && drag.rect) {
    const r = rectOf(drag.a, drag.b);
    g.fillStyle = drag.erase ? 'rgba(176,58,72,.35)' : 'rgba(242,162,60,.3)';
    g.fillRect(ox + r.x * c, oy + r.y * c, r.w * c, r.h * c);
    g.strokeStyle = '#f2a23c';
    g.strokeRect(ox + r.x * c + .5, oy + r.y * c + .5, r.w * c - 1, r.h * c - 1);
  } else if (S.hover && inside(S.hover)) {
    g.strokeStyle = '#fff';
    g.strokeRect(ox + S.hover.x * c + .5, oy + S.hover.y * c + .5, c - 1, c - 1);
  }
}

const inside = p => S.map && p.x >= 0 && p.y >= 0 && p.x < S.map.w && p.y < S.map.h;
const at = (layer, p) => S.map.layers[layer][p.y * S.map.w + p.x];
const rectOf = (a, b) => ({x: Math.min(a.x, b.x), y: Math.min(a.y, b.y), w: Math.abs(a.x - b.x) + 1, h: Math.abs(a.y - b.y) + 1});

function cellAt(e) {
  const r = cv.getBoundingClientRect();
  return {x: Math.floor((e.clientX - r.left - S.ox) / S.cell), y: Math.floor((e.clientY - r.top - S.oy) / S.cell)};
}

function lineCells(a, b) {
  const out = [];
  let x = a.x, y = a.y;
  const dx = Math.abs(b.x - x), dy = -Math.abs(b.y - y), sx = x < b.x ? 1 : -1, sy = y < b.y ? 1 : -1;
  let err = dx + dy;
  for (;;) {
    out.push({x, y});
    if (x === b.x && y === b.y) return out;
    const e2 = 2 * err;
    if (e2 >= dy) { err += dy; x += sx; }
    if (e2 <= dx) { err += dx; y += sy; }
  }
}

function status() {
  const m = S.map, p = S.hover;
  if (!m) { $('#status').textContent = ' '; return; }
  let text = `${m.name}  ${m.w}×${m.h}   ${S.view === 'whole' ? '全图（静态）' : '行走比例' + (S.play ? '（动画）' : '')}`;
  if (p && inside(p)) {
    const name = L => { const id = at(L, p); return id ? (S.byId[id] ? S.byId[id].name : id + '（缺失）') : '—'; };
    text += `   (${p.x}, ${p.y})   地面：${name('ground')}   植被：${name('plant')}   物件：${name('object')}`;
    if (m.relief) text += `   高度：${Math.ceil(m.relief[p.y * m.w + p.x] / S.geo.fine)} 级`;
    if (m.biome) { const b = m.biome[p.y * m.w + p.x]; if (b) text += `   群系：${(S.geo.biomes || {})[b] || b}`; }
  }
  $('#status').textContent = text;
}

// ---------------------------------------------------------------- editing

let stroke = null;  // "layer,index" -> [layer, x, y, before, after]
let drag = null;

// A rough stand-in on the picture until the server's version of this spot arrives.
function sketch(layer, x, y, id) {
  const t = S.byId[id];
  const g = S.still.getContext('2d');
  if (layer === 'ground') {
    g.fillStyle = t ? S.palette.colors[t.look.base] : '#14121a';
    g.fillRect(x * T, y * T, T, T);
  } else if (layer === 'plant') {
    if (t) { g.fillStyle = S.palette.colors[t.grow.under]; g.fillRect(x * T, y * T, T, T); }
  } else if (t && S.img[id]) {
    const img = S.img[id];
    g.drawImage(img, x * T + T / 2 - Math.floor(img.width / 2), y * T + T - 2 - img.height + 1);
  }
  const k = Math.floor(x / CHUNK) * CHUNK + ',' + Math.floor(y / CHUNK) * CHUNK;
  if (S.anim.has(k)) { S.anim.get(k).close(); S.anim.delete(k); }
  if (S.small) toSmall(x, y, S.still, 1, 1, true);   // the still one shows the change until the new one comes
}

function put(layer, p, id) {
  if (!inside(p)) return;
  const i = p.y * S.map.w + p.x, cur = S.map.layers[layer][i];
  if (cur === id) return;
  const k = layer + ',' + i, e = stroke.get(k);
  if (e) e[4] = id; else stroke.set(k, [layer, p.x, p.y, cur, id]);
  S.map.layers[layer][i] = id;
  sketch(layer, p.x, p.y, id);
}

function commit() {
  const changes = [...stroke.values()].filter(e => e[3] !== e[4]);
  stroke = null;
  if (!changes.length) return;
  S.undo.push(changes);
  S.redo = [];
  send(changes, 4);
  updateHistory();
  draw();
}

function step(from, to, k) {
  const changes = from.pop();
  if (!changes) return;
  for (const e of changes) { S.map.layers[e[0]][e[2] * S.map.w + e[1]] = e[k]; sketch(e[0], e[1], e[2], e[k]); }
  to.push(changes);
  send(changes, k);
  updateHistory();
  draw();
}
const undo = () => step(S.undo, S.redo, 3);
const redo = () => step(S.redo, S.undo, 4);

function updateHistory() {
  $('#undo').disabled = !S.undo.length;
  $('#redo').disabled = !S.redo.length;
}

function send(changes, k) {
  const m = S.map, cells = changes.map(e => [e[0], e[1], e[2], e[k]]);
  const xs = changes.map(e => e[1]), ys = changes.map(e => e[2]);
  const x = Math.min(...xs) - REACH, y = Math.min(...ys) - REACH;
  const w = Math.max(...xs) - x + 1 + REACH, h = Math.max(...ys) - y + 1 + REACH;
  S.seq++;
  sync('保存中…');
  queue(async () => {
    const r = await api('POST', `/api/map/${enc(m.name)}/patch`, {cells, base_rev: S.rev.map});
    if (S.map !== m) return;
    S.rev.map = r.rev;
    if (r.stale) S.needReload = true;  // someone else changed the file; reload once idle
    sync('已保存');
    await refresh(m, x, y, w, h);
  });
}

function needTile() {
  if (S.sel && S.byId[S.sel]) return S.byId[S.sel];
  sync('先在左边选一个图块', true, 2500);
  return null;
}

function flood(p, tile) {
  if (!inside(p)) return;
  const m = S.map, cells = m.layers[tile.layer], old = at(tile.layer, p);
  if (old === tile.id) return;
  const todo = [p];
  put(tile.layer, p, tile.id);
  while (todo.length) {
    const c = todo.pop();
    for (const n of [{x: c.x + 1, y: c.y}, {x: c.x - 1, y: c.y}, {x: c.x, y: c.y + 1}, {x: c.x, y: c.y - 1}]) {
      if (inside(n) && cells[n.y * m.w + n.x] === old) { put(tile.layer, n, tile.id); todo.push(n); }
    }
  }
}

function pick(p) {
  if (!inside(p)) return;
  const id = at('object', p) || at('plant', p) || at('ground', p);
  if (id && S.byId[id]) { S.sel = id; setTool('brush'); renderTiles(); }
}

function dab(p) {
  if (drag.layer) put(drag.layer, p, null);
  else put(drag.tile.layer, p, drag.tile.id);
}

cv.addEventListener('pointerdown', e => {
  cv.setPointerCapture(e.pointerId);
  if (e.button === 1 || e.button === 2 || S.space) {
    if (S.view === 'walk') drag = {pan: true, x: e.clientX, y: e.clientY, ox: S.ox, oy: S.oy};
    return;
  }
  if (e.button !== 0 || !S.map) return;
  const p = cellAt(e), tool = e.altKey ? 'pick' : S.tool;
  if (tool === 'pick') return pick(p);
  if (tool === 'erase') {
    if (!inside(p)) return;
    stroke = new Map();
    drag = {stroke: true, last: p, layer: at('object', p) ? 'object' : at('plant', p) ? 'plant' : 'ground'};
    dab(p);
    return draw();
  }
  const tile = needTile();
  if (!tile) return;
  if (tool === 'fill') { stroke = new Map(); flood(p, tile); return commit(); }
  if (tool === 'rect') { drag = {rect: true, a: p, b: p, erase: e.shiftKey, tile}; return draw(); }
  stroke = new Map();
  drag = {stroke: true, last: p, tile};
  dab(p);
  draw();
});

cv.addEventListener('pointermove', e => {
  const p = cellAt(e);
  if (drag && drag.pan) {
    S.ox = drag.ox + e.clientX - drag.x; S.oy = drag.oy + e.clientY - drag.y;
    clampView();
    want();
  } else if (drag && drag.stroke) {
    for (const c of lineCells(drag.last, p)) dab(c);
    drag.last = p;
  } else if (drag && drag.rect) {
    drag.b = p;
  }
  S.hover = p;
  status();
  draw();
});

function endDrag() {
  const d = drag;
  drag = null;
  if (!d) return;
  if (d.stroke) commit();
  if (d.rect) {
    const r = rectOf(d.a, d.b);
    stroke = new Map();
    for (let y = r.y; y < r.y + r.h; y++) {
      for (let x = r.x; x < r.x + r.w; x++) put(d.tile.layer, {x, y}, d.erase ? null : d.tile.id);
    }
    commit();
  }
  draw();
}
cv.addEventListener('pointerup', endDrag);
cv.addEventListener('pointercancel', endDrag);
cv.addEventListener('pointerleave', () => { if (!drag) { S.hover = null; status(); draw(); } });
cv.addEventListener('contextmenu', e => e.preventDefault());

cv.addEventListener('wheel', e => e.preventDefault(), {passive: false});   // no zooming: two fixed views

new ResizeObserver(() => { if (S.view === 'whole') fitView(); else clampView(); draw(); want(); }).observe($('#stage'));

// WASD walks the view, smoothly, for as long as the keys are held
let walkT = 0;
function walk(t) {
  if (!S.keys.size || S.view !== 'walk' || !S.map) { walkT = 0; return; }
  const dt = walkT ? Math.min((t - walkT) / 1000, 0.1) : 0;
  walkT = t;
  const k = S.keys, step = PAN * S.cell * dt;
  S.ox -= ((k.has('d') ? 1 : 0) - (k.has('a') ? 1 : 0)) * step;
  S.oy -= ((k.has('s') ? 1 : 0) - (k.has('w') ? 1 : 0)) * step;
  clampView();
  draw();
  want();
  requestAnimationFrame(walk);
}

document.addEventListener('keydown', e => {
  if ($('dialog[open]') || /INPUT|SELECT|TEXTAREA/.test(e.target.tagName)) return;
  if (e.code === 'Space') { S.space = true; e.preventDefault(); return; }
  if (e.ctrlKey || e.metaKey) {
    const k = e.key.toLowerCase();
    if (k === 'z' && !e.shiftKey) { undo(); e.preventDefault(); }
    else if (k === 'y' || (k === 'z' && e.shiftKey)) { redo(); e.preventDefault(); }
    return;
  }
  const key = e.key.toLowerCase();
  if ('wasd'.includes(key) && key.length === 1) {
    if (S.view === 'walk' && !S.keys.has(key)) { S.keys.add(key); if (S.keys.size === 1) requestAnimationFrame(walk); }
    e.preventDefault();
    return;
  }
  if (key === 'v') setView(S.view === 'whole' ? 'walk' : 'whole');
  const tool = {b: 'brush', e: 'erase', r: 'rect', f: 'fill', i: 'pick'}[e.key.toLowerCase()];
  if (tool) setTool(tool);
  if (e.key.toLowerCase() === 'g') { $('#grid').checked = S.grid = !S.grid; draw(); }
  if (e.key.toLowerCase() === 'p' && S.view === 'walk') { $('#play').checked = S.play = !S.play; status(); draw(); want(); }
  if (e.key.toLowerCase() === 'h') { $('#contours').checked = S.contours = !S.contours; draw(); }
});
document.addEventListener('keyup', e => { if (e.code === 'Space') S.space = false; S.keys.delete(e.key.toLowerCase()); });
window.addEventListener('blur', () => S.keys.clear());
$$('#views button').forEach(b => { b.onclick = () => setView(b.dataset.view); });

$$('#tools button').forEach(b => { b.onclick = () => setTool(b.dataset.tool); });
$('#grid').onchange = e => { S.grid = e.target.checked; draw(); };
$('#play').onchange = e => { S.play = e.target.checked; status(); draw(); want(); };
$('#contours').onchange = e => { S.contours = e.target.checked; draw(); };

// One clock for everything that moves: the map, and the previews in the two tile editors.
const ticking = [];
function startClock() {
  setInterval(() => {
    if (document.hidden) return;
    S.f = (S.f + 1) % S.nFrames;
    if (S.play && S.view === 'walk' && S.map && !$('dialog[open]')) draw();
    for (const fn of ticking) fn(S.f);
  }, 1000 / S.fps);
}

// A strip of frames (stacked top to bottom) played on a canvas at a whole-number scale.
function player(canvas, scale) {
  let strip = null;
  ticking.push(f => {
    if (!strip || !canvas.isConnected || !canvas.offsetParent) return;
    const w = strip.width, h = strip.height / S.nFrames, g = canvas.getContext('2d');
    if (canvas.width !== w * scale || canvas.height !== h * scale) { canvas.width = w * scale; canvas.height = h * scale; }
    g.imageSmoothingEnabled = false;
    g.clearRect(0, 0, canvas.width, canvas.height);
    g.drawImage(strip, 0, f * h, w, h, 0, 0, w * scale, h * scale);
  });
  return bmp => { strip = bmp; };
}
$('#undo').onclick = undo;
$('#redo').onclick = redo;

// ---------------------------------------------------------------- staying in step with the files

async function poll() {
  if (S.inflight || S.polling || drag) return;
  S.polling = true;
  const seq = S.seq;
  try {
    const r = await api('GET', '/api/rev');
    if (S.inflight || seq !== S.seq || drag) return;  // our own save raced the poll; ask again later
    let reloaded = false, tiles = false;
    if (r.tileset !== S.rev.tileset || Object.keys(r.maps).join('\n') !== S.maps.map(m => m.name).join('\n')) {
      tiles = r.tileset !== S.rev.tileset;
      await loadState();
      reloaded = true;
    }
    if (S.map) {
      const rev = r.maps[S.map.name];
      if (rev === undefined) { await openFirstMap(); reloaded = true; }
      else if (rev !== S.rev.map || S.needReload || tiles) { await loadMap(S.map.name, true); reloaded = true; }
    } else if (S.maps.length) {
      await openFirstMap();
    }
    if (reloaded) sync('已同步外部修改', true, 2500);
    else if ($('#sync').textContent.startsWith('连接')) sync('已保存');
  } catch (e) {
    sync('连接断开：' + e.message, true);
  } finally {
    S.polling = false;
  }
}

// ---------------------------------------------------------------- map dialogs

let mapMode = 'new';
function openMapDlg(mode) {
  mapMode = mode;
  const resize = mode === 'resize';
  $('#mapTitle').textContent = resize ? `调整「${S.map.name}」的尺寸` : '新建地图';
  $('#mNameRow').hidden = resize;
  $('#mShiftRow').hidden = !resize;
  $('#mName').value = '';
  $('#mHowRow').hidden = resize;
  $('#mHow').value = 'blank';
  const pick = (id, entries) => $(id).replaceChildren(...entries.map(([v, text]) => new Option(text, v)));
  pick('#mValley', [['', '生成（抬升与河流侵蚀）'], ...Object.entries(S.geo.valleys).map(([k, v]) => [k, '实测对照：' + v.name])]);
  pick('#mClimate', Object.entries(S.geo.climates));
  pick('#mPlace', Object.entries(S.geo.places));
  $('#mPlace').value = 'river';
  $('#mValley').onchange = () => { const v = S.geo.valleys[$('#mValley').value]; if (v) $('#mClimate').value = v.climate; };
  $('#mValley').onchange();
  const how = () => { const geo = !resize && $('#mHow').value === 'geo'; $('#mGeo').hidden = !geo; $('#mFillRow').hidden = geo;
    if (geo && +$('#mW').value === 40) { $('#mW').value = 60; $('#mH').value = 40; } };
  $('#mHow').onchange = how;
  how();
  $('#mW').value = resize ? S.map.w : 40;
  $('#mH').value = resize ? S.map.h : 28;
  $('#mDx').value = $('#mDy').value = 0;
  const grounds = tilesOf('ground');
  $('#mFill').replaceChildren(new Option('（留空）', ''), ...grounds.map(t => new Option(t.name, t.id)));
  const grass = grounds.find(t => t.id === 'grass') || grounds[0];
  if (!resize && grass) $('#mFill').value = grass.id;
  $('#mapDlg').showModal();
}

$('#mOk').onclick = async () => {
  const body = {width: +$('#mW').value, height: +$('#mH').value, fill: $('#mFill').value || null};
  try {
    await chain;
    if (mapMode === 'new' && $('#mHow').value === 'geo') {
      sync('正在推演地形…', true);
      const d = await api('POST', '/api/generate', {name: $('#mName').value.trim(), width: body.width, height: body.height,
        valley: $('#mValley').value, climate: $('#mClimate').value, place: $('#mPlace').value, seed: +$('#mSeed').value});
      $('#mapDlg').close();
      await loadState();
      await loadMap(d.name);
      sync('已保存');
      msg('这块地为什么长这样', d.report.map(line => '• ' + line).join('\n'));
    } else if (mapMode === 'new') {
      const d = await api('POST', '/api/map', {...body, name: $('#mName').value.trim()});
      $('#mapDlg').close();
      await loadState();
      await loadMap(d.name);
    } else {
      await api('POST', `/api/map/${enc(S.map.name)}/resize`, {...body, dx: +$('#mDx').value, dy: +$('#mDy').value});
      $('#mapDlg').close();
      await loadMap(S.map.name);
    }
  } catch (e) {
    msg('没有成功', e.message);
  }
};
$('#mCancel').onclick = () => $('#mapDlg').close();
$('#newMap').onclick = () => openMapDlg('new');
$('#resizeMap').onclick = () => openMapDlg('resize');
$('#mapSel').onchange = async e => { await chain; loadMap(e.target.value); };

$('#delMap').onclick = async () => {
  const name = S.map.name;
  if (!await ask('删除地图', `把「${name}」移到 .trash 文件夹？（文件不会被真正删除，可以手动移回来。）`)) return;
  try {
    await chain;
    await api('DELETE', '/api/map/' + enc(name));
    S.map = null;
    await loadState();
    await openFirstMap();
  } catch (e) { msg('没有成功', e.message); }
};

$('#validate').onclick = async () => {
  try {
    await chain;
    const r = await api('POST', `/api/map/${enc(S.map.name)}/validate`, {});
    const used = Object.entries(r.stats.tiles).map(([id, n]) => `  ${S.byId[id] ? S.byId[id].name : id}：${n}`).join('\n');
    msg(r.ok ? '没有发现问题' : `发现 ${r.problems.length} 个问题`,
      [...r.problems.map(p => '• ' + p), ...r.notes.map(n => '（提示）' + n)].join('\n') + (r.problems.length + r.notes.length ? '\n\n' : '') +
      `可行走的陆地 ${r.stats.walkable_regions} 块，最大 ${r.stats.largest_region} 格\n图块用量：\n${used || '  （空）'}`);
  } catch (e) { msg('没有成功', e.message); }
};

$('#png').onclick = async () => { await chain; window.open(`/api/map/${enc(S.map.name)}/render.png?scale=2`, '_blank'); };

$('#export').onclick = async () => {
  try {
    await chain;
    const r = await api('POST', '/api/export', {});
    msg('已导出到游戏', `${r.objects} 个物件的图集：\n${r.png}\n${r.json}\n\n烘焙好的地面：\n${r.grounds.join('\n') || '（没有地图）'}` +
      (r.problems.length ? '\n\n' + r.problems.join('\n') : ''));
  } catch (e) { msg('没有成功', e.message); }
};

// ---------------------------------------------------------------- shared by both tile editors

const colorNames = () => Object.keys(S.palette.colors).filter(n => !PLACEHOLDER.has(n));
const splitList = text => text.split(/[,，\s]+/).map(s => s.trim()).filter(Boolean);

function tintSelect(sel) {
  const hex = S.palette.colors[sel.value];
  sel.style.background = hex || '';
  if (!hex) { sel.style.color = ''; return; }
  const n = parseInt(hex.slice(1), 16), lum = (n >> 16) * .3 + ((n >> 8) & 255) * .59 + (n & 255) * .11;
  sel.style.color = lum > 140 ? '#000' : '#fff';
}

function colorSelect(value, allowNone) {
  const sel = document.createElement('select');
  sel.className = 'color';
  if (allowNone) sel.add(new Option('无', ''));
  for (const n of colorNames()) sel.add(new Option(n, n));
  sel.value = value || '';
  tintSelect(sel);
  sel.addEventListener('change', () => tintSelect(sel));
  return sel;
}

async function deleteTile(id, dlg) {
  try {
    try {
      await api('DELETE', '/api/tile/' + enc(id));
    } catch (e) {
      if (!await ask('图块还在使用', e.message + '\n\n仍然删除，并清掉用到它的地方？')) return;
      await chain;
      await api('DELETE', `/api/tile/${enc(id)}?force=1`);
    }
    dlg.close();
    await loadState();
    if (S.map) await loadMap(S.map.name, true);
  } catch (e) { msg('没有成功', e.message); }
}

async function saveTile(tile, isNew, dlg) {
  if (isNew && S.byId[tile.id]) return msg('没有成功', `已经有图块 ${tile.id} 了，换一个 id。`);
  try {
    await chain;
    await api('POST', '/api/tile', tile);
    dlg.close();
    S.sel = tile.id;
    await loadState();
    if (S.map) await loadMap(S.map.name, true);  // the look of the map may have changed
  } catch (e) { msg('没有成功', e.message); }
}

// ---------------------------------------------------------------- the material editor (ground)

const L = {tile: null, lists: {}, timer: 0, url: null};

// Rows of small controls for a list of {key: value} items.
function listEditor(box, items, cols, onChange) {
  const rows = [];
  box.replaceChildren();
  const add = item => {
    const row = document.createElement('div'), read = {};
    row.className = 'row li';
    for (const c of cols) {
      let el;
      if (c.type === 'color') el = colorSelect(item[c.key]);
      else if (c.type === 'pick') { el = document.createElement('select'); for (const [v, text] of c.options()) el.add(new Option(text, v)); el.value = item[c.key] ?? ''; }
      else if (c.type === 'text') { el = document.createElement('input'); el.type = 'text'; el.placeholder = c.placeholder || ''; el.value = (item[c.key] || []).join(', '); }
      else { el = document.createElement('input'); el.type = 'number'; el.min = c.min; el.max = c.max; el.step = c.step; el.value = item[c.key] ?? c.def ?? ''; }
      el.addEventListener('change', onChange);
      const label = document.createElement('label');
      label.title = c.hint || '';
      label.append(c.label + ' ', el);
      row.append(label);
      read[c.key] = () => (c.type === 'num' ? +el.value : c.type === 'text' ? splitList(el.value) : el.value);
    }
    const del = document.createElement('button');
    del.type = 'button'; del.textContent = '×';
    del.onclick = () => { rows.splice(rows.indexOf(read), 1); row.remove(); onChange(); };
    row.append(del);
    rows.push(read);
    box.append(row);
  };
  items.forEach(add);
  return {
    add: item => { add(item); onChange(); },
    value: () => rows.map(r => Object.fromEntries(Object.entries(r).map(([k, f]) => [k, f()]))),
  };
}

function collectLook() {
  const cliff = splitList($('#lCliff').value), shore = splitList($('#lShore').value);
  const surf = L.surfFoam.value && L.surfWater.value
    ? {foam: L.surfFoam.value, water: L.surfWater.value, reach: +$('#lSurfReach').value, onto: splitList($('#lSurfOnto').value)} : null;
  return {
    edge: L.lists.edge.value(), apron: L.lists.apron.value(), surf, joins: splitList($('#lJoins').value), flows: $('#lFlows').checked,
    base: L.base.value, rim: L.rim.value || null,
    patches: L.lists.patches.value(), specks: L.lists.specks.value(), decor: L.lists.decor.value().filter(d => d.tile),
    cliff: cliff.length ? {colors: cliff, over: splitList($('#lCliffOver').value), cracks: $('#lCracks').checked} : null,
    shore: shore.length ? {colors: shore, lap: $('#lLap').checked} : null,
  };
}

function previewLook() {
  clearTimeout(L.timer);
  L.timer = setTimeout(async () => {
    try {
      const r = await fetch('/api/preview', {method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({id: $('#lId').value.trim(), rank: +$('#lRank').value, look: collectLook()})});
      if (!r.ok) throw new Error((await r.json()).error);
      L.show(await createImageBitmap(await r.blob()));
      $('#lNote').textContent = '预览：这种材质与相邻层级材质相接时的样子（会动的部分在动）。';
    } catch (e) { $('#lNote').textContent = '预览不了：' + e.message; }
  }, 250);
}

function openLook(tile) {
  L.tile = tile || null;
  const grounds = tilesOf('ground');
  const look = tile ? tile.look : {base: 'g2', patches: [], specks: [], rim: null, edge: [], apron: [], cliff: null,
    shore: null, surf: null, decor: []};
  $('#lookTitle').textContent = tile ? `编辑地面材质「${tile.name}」` : '新的地面材质';
  $('#lId').value = tile ? tile.id : '';
  $('#lId').readOnly = !!tile;
  $('#lName').value = tile ? tile.name : '';
  $('#lPass').value = tile ? tile.pass : 'walk';
  $('#lRank').value = tile ? tile.rank : (grounds.length ? grounds[grounds.length - 1].rank + 1 : 0);
  $('#lTags').value = tile ? tile.tags.join(', ') : '';
  L.base = colorSelect(look.base); L.rim = colorSelect(look.rim, true);
  $('#lBase').replaceChildren(L.base); $('#lRim').replaceChildren(L.rim);
  L.base.addEventListener('change', previewLook); L.rim.addEventListener('change', previewLook);
  const color = {key: 'color', label: '颜色', type: 'color'};
  L.lists.patches = listEditor($('#lPatches'), look.patches, [color,
    {key: 'scale', label: '大小', type: 'num', min: 4, max: 96, step: 2, hint: '色斑的尺度，像素'},
    {key: 'amount', label: '占比', type: 'num', min: 0, max: 1, step: 0.05},
    {key: 'fill', label: '实', type: 'num', min: 0.05, max: 1, step: 0.05, def: 1,
      hint: '1 是实心色块；小于 1 打散成散点，读起来是纹理而不是另一种地形的边界'}], previewLook);
  const zone = [color,
    {key: 'width', label: '宽', type: 'num', min: 1, max: 16, step: 1, hint: '像素'},
    {key: 'wobble', label: '起伏', type: 'num', min: 0, max: 8, step: 0.5, def: 0, hint: '让宽度不均匀'},
    {key: 'density', label: '密', type: 'num', min: 0.01, max: 1, step: 0.05, def: 1, hint: '1 是实心带；小于 1 是散点，离边界越远越稀'},
    {key: 'shape', label: '形', type: 'pick', options: () => S.shapes.map(s => [s, s])}];
  L.lists.edge = listEditor($('#lEdge'), look.edge || [], zone, previewLook);
  L.lists.apron = listEditor($('#lApron'), look.apron || [], [...zone,
    {key: 'over', label: '只洒到', type: 'text', placeholder: '留空=全部', hint: '只洒到这些地形上，逗号分隔'}], previewLook);
  L.surfFoam = colorSelect(look.surf && look.surf.foam, true); L.surfWater = colorSelect(look.surf && look.surf.water, true);
  $('#lSurfFoam').replaceChildren(L.surfFoam); $('#lSurfWater').replaceChildren(L.surfWater);
  L.surfFoam.addEventListener('change', previewLook); L.surfWater.addEventListener('change', previewLook);
  $('#lSurfReach').value = look.surf ? look.surf.reach : 8;
  $('#lSurfOnto').value = look.surf ? look.surf.onto.join(', ') : '';
  $('#lJoins').value = (look.joins || []).join(', ');
  $('#lFlows').checked = !!look.flows;
  L.lists.specks = listEditor($('#lSpecks'), look.specks, [color,
    {key: 'shape', label: '形状', type: 'pick', options: () => S.shapes.map(s => [s, s])},
    {key: 'density', label: '密度', type: 'num', min: 0, max: 0.3, step: 0.002, hint: '每个像素出现的概率'},
    {key: 'anim', label: '动', type: 'pick', hint: '闪：忽隐忽现；漂：滑过去，像水纹',
      options: () => [['', '不动'], ['twinkle', '闪'], ['drift', '漂'], ['gust', '风过时'], ['flow', '顺流']]}], previewLook);
  L.lists.decor = listEditor($('#lDecor'), look.decor, [
    {key: 'tile', label: '物件', type: 'pick', options: () => tilesOf('object').map(t => [t.id, t.name])},
    {key: 'density', label: '密度', type: 'num', min: 0, max: 1, step: 0.01, hint: '每格出现的概率'}], previewLook);
  $('#lCliff').value = look.cliff ? look.cliff.colors.join(', ') : '';
  $('#lCliffOver').value = look.cliff ? look.cliff.over.join(', ') : '';
  $('#lCracks').checked = !!(look.cliff && look.cliff.cracks);
  $('#lShore').value = look.shore ? look.shore.colors.join(', ') : '';
  $('#lLap').checked = !!(look.shore && look.shore.lap);
  $('#lDelete').hidden = !tile;
  previewLook();
  $('#lookDlg').showModal();
}

$('#lAddPatch').onclick = () => L.lists.patches.add({color: L.base.value, scale: 24, amount: 0.3, fill: 1});
$('#lAddEdge').onclick = () => L.lists.edge.add({color: L.base.value, width: 3, wobble: 2, density: 1, shape: 'dot'});
$('#lAddApron').onclick = () => L.lists.apron.add({color: L.base.value, width: 6, wobble: 2, density: 0.4, shape: 'dot', over: []});
$('#lAddSpeck').onclick = () => L.lists.specks.add({color: L.base.value, shape: 'dot', density: 0.01, anim: ''});
L.show = player($('#lPreview'), 1);
$('#lAddDecor').onclick = () => { const o = tilesOf('object')[0]; L.lists.decor.add({tile: o ? o.id : '', density: 0.05}); };
for (const id of ['lCliff', 'lCliffOver', 'lCracks', 'lShore', 'lLap', 'lRank', 'lSurfReach', 'lSurfOnto', 'lJoins', 'lFlows']) $('#' + id).addEventListener('change', previewLook);
$('#lCancel').onclick = () => $('#lookDlg').close();
$('#lDelete').onclick = () => deleteTile(L.tile.id, $('#lookDlg'));
$('#lSave').onclick = () => saveTile({
  id: $('#lId').value.trim(), name: $('#lName').value.trim(), layer: 'ground', pass: $('#lPass').value,
  rank: +$('#lRank').value, tags: splitList($('#lTags').value), look: collectLook(),
}, !L.tile, $('#lookDlg'));

// ---------------------------------------------------------------- the sprite editor (objects)

const E = {variants: [], vi: 0, color: 'g2', tool: 'pen', hist: [], tile: null, down: false, last: 0, timer: 0, busy: false};
const pxc = $('#px');
const cur = () => E.variants[E.vi];
const pxScale = v => Math.max(4, Math.floor(336 / Math.max(v.w, v.h)));

function pxToArt(v) {
  const chars = {}, key = {}, rows = [];
  for (let y = 0; y < v.h; y++) {
    let row = '';
    for (let x = 0; x < v.w; x++) {
      const name = v.px[y * v.w + x];
      if (!name) { row += '.'; continue; }
      if (!(name in chars)) { chars[name] = ALPHABET[Object.keys(chars).length]; key[chars[name]] = name; }
      row += chars[name];
    }
    rows.push(row);
  }
  return {rows, key};
}
const artToPx = art => ({w: art.rows[0].length, h: art.rows.length, px: art.rows.flatMap(row => [...row].map(ch => art.key[ch] || null))});

function setColor(name) {
  E.color = name;
  $('#curColor').textContent = `${name}  ${S.palette.colors[name]}`;
  $$('#swatches i').forEach(i => i.classList.toggle('on', i.dataset.name === name));
}

function setPTool(tool) {
  E.tool = tool;
  $$('[data-ptool]').forEach(b => b.classList.toggle('on', b.dataset.ptool === tool));
}

function buildPalette() {
  const names = colorNames();
  $('#swatches').replaceChildren(...names.map(name => {
    const i = document.createElement('i');
    i.dataset.name = name;
    i.title = `${name}  ${S.palette.colors[name]}`;
    i.style.background = S.palette.colors[name];
    i.onclick = () => { setColor(name); if (E.tool === 'erase' || E.tool === 'pick') setPTool('pen'); };
    return i;
  }));
  $('#gRamp').replaceChildren(...Object.keys(S.palette.ramps || {}).map(n => new Option(n, n)));
  if (!S.palette.colors[E.color]) E.color = names[0];
  setColor(E.color);
}

function renderVariants() {
  $('#vSel').replaceChildren(...E.variants.map((v, i) => new Option(`${i + 1}（${v.w}×${v.h}）`, i)));
  $('#vSel').value = E.vi;
  $('#vDel').disabled = E.variants.length < 2;
  $('#sW').value = cur().w; $('#sH').value = cur().h;
}

function openSprite(tile) {
  E.tile = tile || null;
  E.variants = tile ? tile.variants.map(artToPx) : [{w: T, h: T, px: new Array(T * T).fill(null)}];
  E.vi = 0; E.hist = [];
  buildPalette();
  setPTool('pen');
  $('#spriteTitle').textContent = tile ? `编辑物件「${tile.name}」` : '新物件';
  $('#tId').value = tile ? tile.id : '';
  $('#tId').readOnly = !!tile;
  $('#tName').value = tile ? tile.name : '';
  $('#tPass').value = tile ? tile.pass : 'walk';
  $('#tTags').value = tile ? tile.tags.join(', ') : '';
  $('#tShadow').checked = tile ? tile.shadow : true;
  $('#tFlip').checked = tile ? tile.flip : false;
  $('#tJitter').value = tile ? tile.jitter : 0;
  $('#tMotion').value = tile ? tile.motion : 'none';
  $('#tFw').value = tile ? tile.footprint[0] : 1;
  $('#tFh').value = tile ? tile.footprint[1] : 1;
  $('#tDelete').hidden = !tile;
  renderVariants();
  drawPx();
  $('#spriteDlg').showModal();
}

function drawPx() {
  const v = cur(), s = pxScale(v);
  if (pxc.width !== v.w * s || pxc.height !== v.h * s) { pxc.width = v.w * s; pxc.height = v.h * s; }
  const g = pxc.getContext('2d');
  for (let y = 0; y < v.h; y++) {
    for (let x = 0; x < v.w; x++) {
      const name = v.px[y * v.w + x];
      g.fillStyle = name ? S.palette.colors[name] : ((x + y) % 2 ? '#34323f' : '#2a2836');
      g.fillRect(x * s, y * s, s, s);
    }
  }
  if (s >= 6) {
    g.strokeStyle = 'rgba(0,0,0,.22)';
    g.beginPath();
    for (let i = 1; i < v.w; i++) { g.moveTo(i * s + .5, 0); g.lineTo(i * s + .5, v.h * s); }
    for (let i = 1; i < v.h; i++) { g.moveTo(0, i * s + .5); g.lineTo(v.w * s, i * s + .5); }
    g.stroke();
  }
  const img = artCanvas(pxToArt(v)), pv1 = $('#pv1');
  pv1.width = v.w; pv1.height = v.h;
  pv1.getContext('2d').drawImage(img, 0, 0);

  clearTimeout(E.timer);  // once the hand has stopped: pixelkit's check, and the sprite in motion
  E.timer = setTimeout(async () => {
    const art = pxToArt(cur());
    try { showProblems((await api('POST', '/api/art', {op: 'check', art})).problems); }
    catch (e) { showProblems([e.message]); }
    try {
      const r = await fetch('/api/motion', {method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({art, motion: $('#tMotion').value})});
      if (r.ok) E.show(await createImageBitmap(await r.blob()));
    } catch (e) { /* the still preview is enough */ }
  }, 400);
}

function showProblems(list) {
  const ul = $('#problems');
  ul.classList.toggle('ok', !list.length);
  ul.replaceChildren(...(list.length ? list : ['pixelkit 检查通过']).map(t => {
    const li = document.createElement('li');
    li.textContent = t;
    return li;
  }));
}

function snapshot() {
  const v = cur();
  E.hist.push({vi: E.vi, w: v.w, h: v.h, px: v.px.slice()});
  if (E.hist.length > 60) E.hist.shift();
}

function pxAt(e) {
  const r = pxc.getBoundingClientRect(), v = cur();
  const x = Math.floor((e.clientX - r.left) / r.width * v.w), y = Math.floor((e.clientY - r.top) / r.height * v.h);
  return x >= 0 && y >= 0 && x < v.w && y < v.h ? y * v.w + x : -1;
}

function pxFill(i, name) {
  const v = cur(), old = v.px[i];
  if (old === name) return;
  const todo = [i];
  v.px[i] = name;
  while (todo.length) {
    const c = todo.pop(), x = c % v.w, y = (c - x) / v.w;
    for (const [nx, ny] of [[x + 1, y], [x - 1, y], [x, y + 1], [x, y - 1]]) {
      const n = ny * v.w + nx;
      if (nx >= 0 && ny >= 0 && nx < v.w && ny < v.h && v.px[n] === old) { v.px[n] = name; todo.push(n); }
    }
  }
}

pxc.addEventListener('pointerdown', e => {
  const i = pxAt(e), v = cur();
  if (i < 0 || E.busy) return;
  pxc.setPointerCapture(e.pointerId);
  const tool = e.altKey ? 'pick' : (e.button === 2 ? 'erase' : E.tool);
  if (tool === 'pick') { if (v.px[i]) { setColor(v.px[i]); setPTool('pen'); } return; }
  snapshot();
  if (tool === 'fill') pxFill(i, E.color);
  else { E.down = tool; E.last = i; v.px[i] = tool === 'erase' ? null : E.color; }
  drawPx();
});
pxc.addEventListener('pointermove', e => {
  const i = pxAt(e), v = cur();
  if (!E.down || i < 0) return;
  // A fast hand skips pixels between two events; fill in the line.
  for (const p of lineCells({x: E.last % v.w, y: Math.floor(E.last / v.w)}, {x: i % v.w, y: Math.floor(i / v.w)})) {
    v.px[p.y * v.w + p.x] = E.down === 'erase' ? null : E.color;
  }
  E.last = i;
  drawPx();
});
pxc.addEventListener('pointerup', () => { E.down = false; });
pxc.addEventListener('pointercancel', () => { E.down = false; });
pxc.addEventListener('contextmenu', e => e.preventDefault());

$$('[data-ptool]').forEach(b => { b.onclick = () => setPTool(b.dataset.ptool); });
$('#pxUndo').onclick = () => {
  const h = E.hist.pop();
  if (!h || !E.variants[h.vi]) return;
  E.vi = h.vi;
  E.variants[h.vi] = {w: h.w, h: h.h, px: h.px};
  renderVariants();
  drawPx();
};
$('#pxClear').onclick = () => { snapshot(); cur().px.fill(null); drawPx(); };
E.show = player($('#pvMove'), 2);
$('#tMotion').onchange = drawPx;
$('#vSel').onchange = e => { E.vi = +e.target.value; renderVariants(); drawPx(); };
$('#vAdd').onclick = () => {
  if (E.variants.length >= 8) return;
  const v = cur();
  E.variants.push({w: v.w, h: v.h, px: v.px.slice()});
  E.vi = E.variants.length - 1; E.hist = [];
  renderVariants();
  drawPx();
};
$('#vDel').onclick = () => {
  if (E.variants.length < 2) return;
  E.variants.splice(E.vi, 1);
  E.vi = Math.max(0, E.vi - 1); E.hist = [];
  renderVariants();
  drawPx();
};
$('#sResize').onclick = () => {
  const v = cur(), w = Math.max(1, Math.min(64, +$('#sW').value | 0)), h = Math.max(1, Math.min(64, +$('#sH').value | 0));
  if (w === v.w && h === v.h) return;
  snapshot();
  const px = new Array(w * h).fill(null), dx = Math.floor((w - v.w) / 2), dy = h - v.h;  // feet stay put
  for (let y = 0; y < v.h; y++) {
    for (let x = 0; x < v.w; x++) {
      const nx = x + dx, ny = y + dy;
      if (nx >= 0 && ny >= 0 && nx < w && ny < h) px[ny * w + nx] = v.px[y * v.w + x];
    }
  }
  E.variants[E.vi] = {w, h, px};
  renderVariants();
  drawPx();
};

// Everything below runs pixelkit on the server and puts the result back in the editor.
async function pixelkit(body) {
  if (E.busy) return;
  E.busy = true;  // no drawing until the answer is in, or it would be overwritten
  try {
    const r = await api('POST', '/api/art', body);
    snapshot();
    E.variants[E.vi] = artToPx(r.art);
    renderVariants();
    drawPx();
  } catch (e) {
    msg('pixelkit', e.message);
  } finally {
    E.busy = false;
  }
}
$$('[data-shift]').forEach(b => {
  b.onclick = () => { const [dx, dy] = b.dataset.shift.split(',').map(Number); pixelkit({op: 'shift', art: pxToArt(cur()), params: {dx, dy}}); };
});
$('#mirror').onclick = () => pixelkit({op: 'mirror', art: pxToArt(cur())});
$('#outline').onclick = () => pixelkit({op: 'outline', art: pxToArt(cur()), params: {color: E.color}});
$('#genBlob').onclick = () => {
  const v = cur(), art = pxToArt(v);
  pixelkit({op: 'recipe', recipe: {size: [v.w, v.h], ops: [
    {op: 'grid', rows: art.rows, key: art.key},
    {op: 'blob', cx: v.w / 2, cy: v.h / 2, rx: Math.max(1, v.w * 0.38), ry: Math.max(1, v.h * 0.32), ramp: $('#gRamp').value},
  ]}});
};

$('#tSave').onclick = () => saveTile({
  id: $('#tId').value.trim(), name: $('#tName').value.trim(), layer: E.tile ? E.tile.layer : (E.layer || 'object'), pass: $('#tPass').value,
  tags: splitList($('#tTags').value), shadow: $('#tShadow').checked, flip: $('#tFlip').checked,
  jitter: +$('#tJitter').value, motion: $('#tMotion').value, footprint: [+$('#tFw').value, +$('#tFh').value], variants: E.variants.map(pxToArt),
}, !E.tile, $('#spriteDlg'));
$('#tCancel').onclick = () => $('#spriteDlg').close();
$('#tDelete').onclick = () => deleteTile(E.tile.id, $('#spriteDlg'));
$$('[data-new]').forEach(b => { b.onclick = () => (b.dataset.new === 'ground' ? openLook(null) : b.dataset.new === 'plant'
  ? msg('新植被', '植被用 MCP 的 save_tile 新建：layer 为 plant，给出 grow：{height:[最矮,最高], density, colors:[根…尖], under, spill}。')
  : (E.layer = 'object', openSprite(null))); });
for (const id of ['spriteDlg', 'lookDlg', 'mapDlg']) $(`#${id} form`).addEventListener('submit', e => e.preventDefault());

// ---------------------------------------------------------------- start

(async () => {
  setTool('brush');
  let view = 'whole';
  try { view = localStorage.getItem('mapdesigner.view') || 'whole'; } catch (e) { /* private mode */ }
  setView(view === 'walk' ? 'walk' : 'whole');
  try {
    await loadState();
    await openFirstMap();
    sync('已保存');
  } catch (e) {
    sync('无法连接服务：' + e.message, true);
  }
  status();
  startClock();
  setInterval(poll, 1200);
})();
