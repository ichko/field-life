// The fluid page, driven headless: samples worlds and scores them, or replays chosen ones for frames.
// Needs playwright (npm i -g playwright) and a Chromium it can find; see the launch line.
// node sweep.js score <count> <start> <out.json> [port]
// node sweep.js capture <params.json> <outdir> [port]   (frames as PNGs + meta)
const { chromium } = require('playwright');
const http = require('http'), fs = require('fs'), path = require('path');
const root = path.resolve(__dirname, '..');
const [mode, a1, a2, a3, a4] = process.argv.slice(2);
const port = +a4 || 8800 + Math.floor(Math.random()*100);
http.createServer((req, res) => {
  fs.readFile(path.join(root, decodeURIComponent(req.url.split('?')[0])), (e, d) => {
    if(e){ res.writeHead(404); res.end(); return; } res.writeHead(200, {'Content-Type':'text/html'}); res.end(d); });
}).listen(port);

function mulberry(a){ return () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a);
  t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0)/4294967296; }; }
const NCH = 4;
function sampleMany(idx){
  // Five to eight fluids, more shapes of matrix, the shaped scalar prior.
  const R = mulberry(1000003*idx + 7);
  const u = (a, b) => a + R()*(b - a), pick = arr => arr[Math.floor(R()*arr.length)];
  const n = pick([5, 6, 6, 7, 8, 8]);
  const kind = pick(['free', 'free', 'sym', 'anti', 'cycle', 'cycle', 'hunter', 'pairs', 'clans', 'ring2']);
  const M = new Array(n*n).fill(0);
  const set = (c, d, v) => M[c*n + d] = +Math.max(-1, Math.min(1, v)).toFixed(2);
  const half = Math.ceil(n/2);
  if(kind === 'free') for(let c = 0; c < n; c++) for(let d = 0; d < n; d++) set(c, d, u(-1, 1));
  if(kind === 'sym') for(let c = 0; c < n; c++) for(let d = c; d < n; d++){ const v = u(-1, 1); set(c, d, v); set(d, c, v); }
  if(kind === 'anti') for(let c = 0; c < n; c++) for(let d = c; d < n; d++){
    if(c === d) set(c, c, u(-0.3, 0.9)); else { const v = u(-1, 1); set(c, d, v); set(d, c, -v); } }
  if(kind === 'cycle' || kind === 'ring2'){
    const self = u(-0.3, 0.9), fwd = u(0.3, 1), back = u(-1, -0.2), rest = u(-0.7, 0.3), step = kind === 'cycle' ? 1 : 2;
    for(let c = 0; c < n; c++) for(let d = 0; d < n; d++)
      set(c, d, c === d ? self : d === (c + step) % n ? fwd : c === (d + step) % n ? back : rest); }
  if(kind === 'hunter'){ const hs = u(-0.4, 0.6), hunt = u(0.4, 1), flee = u(-1, -0.3), ps = u(-0.4, 0.9), pp = u(-0.8, 0.4);
    for(let c = 0; c < n; c++) for(let d = 0; d < n; d++)
      set(c, d, c === 0 ? (d === 0 ? hs : hunt) : d === 0 ? flee : c === d ? ps : pp); }
  if(kind === 'pairs'){ const s0 = u(-0.3, 0.9), mate = u(0.2, 1), other = u(-1, 0.2);
    for(let c = 0; c < n; c++) for(let d = 0; d < n; d++)
      set(c, d, c === d ? s0 : (c >> 1) === (d >> 1) ? mate : other); }
  if(kind === 'clans'){ const s0 = u(-0.3, 0.9), kin = u(0.1, 0.9), foe = u(-1, -0.1);
    for(let c = 0; c < n; c++) for(let d = 0; d < n; d++)
      set(c, d, c === d ? s0 : Math.floor(c/half) === Math.floor(d/half) ? kin : foe); }
  if(R() < 0.75) for(let c = 0; c < n; c++) set(c, c, u(0.2, 1.0));
  if(kind !== 'free' && R() < 0.5) for(let i = 0; i < n*n; i++) set(Math.floor(i/n), i % n, M[i] + u(-0.15, 0.15));
  return {
    idx, kind, nf: n, M, seed: Math.floor(R()*1e9),
    g: +u(0.8, 3.0).toFixed(2), range: Math.round(u(3, 12)),
    stiff: +u(0.3, 1.5).toFixed(2), gamma: +u(1.5, 3.0).toFixed(2),
    dt: +u(0.08, 0.25).toFixed(3), drag: +u(0.005, 0.05).toFixed(3),
    visc: +u(0, 0.3).toFixed(2), couple: +u(0, 0.3).toFixed(2), vmax: 0.9,
    spread: +u(0.3, 0.5).toFixed(2), seep: +u(0, 0.5).toFixed(2),
    seedMode: R() < 0.75 ? 0 : 1, amt: +u(1.5, 3.5).toFixed(1), rad: Math.round(u(10, 28)), fill: +u(0.15, 0.45).toFixed(2),
  };
}
function sample(idx){
  if(idx >= 2000) return sampleMany(idx);
  const R = mulberry(1000003*idx + 7);
  const u = (a, b) => a + R()*(b - a), pick = arr => arr[Math.floor(R()*arr.length)];
  const kind = pick(['free', 'free', 'sym', 'anti', 'cycle', 'hunter', 'pairs']);
  const M = new Array(16).fill(0);
  const set = (c, d, v) => M[c*NCH + d] = +Math.max(-1, Math.min(1, v)).toFixed(2);
  if(kind === 'free') for(let c = 0; c < 4; c++) for(let d = 0; d < 4; d++) set(c, d, u(-1, 1));
  if(kind === 'sym') for(let c = 0; c < 4; c++) for(let d = c; d < 4; d++){ const v = u(-1, 1); set(c, d, v); set(d, c, v); }
  if(kind === 'anti') for(let c = 0; c < 4; c++) for(let d = c; d < 4; d++){
    if(c === d) set(c, c, u(-0.6, 0.9)); else { const v = u(-1, 1); set(c, d, v); set(d, c, -v); } }
  if(kind === 'cycle'){ const self = u(-0.3, 0.9), fwd = u(0.3, 1), back = u(-1, -0.2), rest = u(-0.7, 0.3);
    for(let c = 0; c < 4; c++) for(let d = 0; d < 4; d++)
      set(c, d, c === d ? self : d === (c + 1) % 4 ? fwd : c === (d + 1) % 4 ? back : rest); }
  if(kind === 'hunter'){ const hs = u(-0.4, 0.6), hunt = u(0.4, 1), flee = u(-1, -0.3), ps = u(-0.4, 0.9), pp = u(-0.8, 0.4);
    for(let c = 0; c < 4; c++) for(let d = 0; d < 4; d++)
      set(c, d, c === 0 ? (d === 0 ? hs : hunt) : d === 0 ? flee : c === d ? ps : pp); }
  if(kind === 'pairs'){ const s = u(-0.3, 0.9), mate = u(0.2, 1), other = u(-1, 0.2);
    for(let c = 0; c < 4; c++) for(let d = 0; d < 4; d++)
      set(c, d, c === d ? s : (c >> 1) === (d >> 1) ? mate : other); }
  // A little asymmetry on the structured ones, so they are not all the same world.
  if(kind !== 'free' && R() < 0.5) for(let i = 0; i < 16; i++) set(i >> 2, i & 3, M[i] + u(-0.15, 0.15));
  // The second prior, shaped like the presets that work: cohesion on the
  // diagonal most of the time, a stronger interaction, a narrower spread.
  const shaped = idx >= 1000;
  if(shaped && R() < 0.75) for(let c = 0; c < 4; c++) set(c, c, u(0.2, 1.0));
  if(shaped) return {
    idx, kind, M, seed: Math.floor(R()*1e9),
    g: +u(0.8, 3.0).toFixed(2), range: Math.round(u(3, 12)),
    stiff: +u(0.3, 1.5).toFixed(2), gamma: +u(1.5, 3.0).toFixed(2),
    dt: +u(0.08, 0.25).toFixed(3), drag: +u(0.005, 0.05).toFixed(3),
    visc: +u(0, 0.3).toFixed(2), couple: +u(0, 0.3).toFixed(2), vmax: 0.9,
    spread: +u(0.3, 0.5).toFixed(2), seep: +u(0, 0.5).toFixed(2),
    seedMode: R() < 0.75 ? 0 : 1, amt: +u(1.5, 3.5).toFixed(1), rad: Math.round(u(10, 28)), fill: +u(0.15, 0.45).toFixed(2),
  };
  const seedMode = R() < 0.6 ? 0 : 1;
  return {
    idx, kind, M, seed: Math.floor(R()*1e9),
    g: +u(0.5, 2.6).toFixed(2), range: Math.round(u(2, 14)),
    stiff: +u(0.3, 2.0).toFixed(2), gamma: +u(1.2, 3.0).toFixed(2),
    dt: +u(0.06, 0.30).toFixed(3), drag: +u(0.003, 0.08).toFixed(3),
    visc: +u(0, 0.5).toFixed(2), couple: +u(0, 0.5).toFixed(2), vmax: 0.9,
    spread: +u(0.3, 0.7).toFixed(2), seep: +u(0, 0.8).toFixed(2),
    seedMode, amt: +u(1, 3.5).toFixed(1), rad: Math.round(u(8, 28)), fill: +u(0.1, 0.5).toFixed(2),
  };
}

async function open(){
  const br = await chromium.launch({ executablePath: '/opt/pw-browsers/chromium-1194/chrome-linux/chrome',
    args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] });
  const pg = await br.newPage({ viewport: { width: 300, height: 300 }, deviceScaleFactor: 1 });
  await pg.route(/fonts\.(googleapis|gstatic)\.com/, r => r.abort());
  await pg.goto(`http://localhost:${port}/staging/fluid.html#p=` + encodeURIComponent(JSON.stringify({N:128, pause:1})),
                { waitUntil: 'domcontentloaded' });
  await pg.waitForFunction(() => typeof step === 'function' && typeof G !== 'undefined' && G.N === 128, null, { timeout: 60000 });
  await pg.addStyleTag({ content: '.page{padding:0;max-width:none} header,.below,.rack,.hud,#hint{display:none!important} .app{aspect-ratio:1;width:256px;border:0;border-radius:0}' });
  return { br, pg };
}

// Runs in the page. Metrics from the density texture read back to the CPU.
const PAGE = {
  readRho: `function readRho(){ const N = G.N, a = new Float32Array(N*N*4), b = new Float32Array(N*N*4);
    target(G.rho, N, N); gl.readPixels(0, 0, N, N, gl.RGBA, gl.FLOAT, a);
    target(G.rho2, N, N); gl.readPixels(0, 0, N, N, gl.RGBA, gl.FLOAT, b);
    const o = new Float32Array(N*N*8);
    for(let i = 0; i < N*N; i++){ for(let c = 0; c < 4; c++){ o[8*i + c] = a[4*i + c]; o[8*i + 4 + c] = b[4*i + c]; } }
    return o; }`,
  metrics: `function metrics(a, b){
    const C = 8, n = a.length/C; let sum = 0, sq = 0, pur = 0, act = 0, occ = 0, mx = 0, edge = 0, bad = 0;
    const N = Math.round(Math.sqrt(n));
    const tot = new Float32Array(n);
    for(let i = 0; i < n; i++){ let t = 0; for(let c = 0; c < C; c++) t += a[C*i + c]; tot[i] = t; if(!isFinite(t)) bad++; sum += t; }
    const m = sum/n;
    for(let i = 0; i < n; i++){
      const t = tot[i]; sq += (t - m)*(t - m); if(t > mx) mx = t; if(t > 0.2*m) occ++;
      if(t > 1e-9){ let best = 0; for(let c = 0; c < C; c++) best = Math.max(best, a[C*i+c]); pur += best; }
      if(b) for(let c = 0; c < C; c++) act += Math.abs(a[C*i+c] - b[C*i+c]);
      const x = i % N, y = (i / N) | 0, r = y*N + (x + 1) % N, u = ((y + 1) % N)*N + x;
      edge += Math.abs(tot[r] - t) + Math.abs(tot[u] - t);
    }
    return { m, cv: Math.sqrt(sq/n)/m, purity: pur/sum, act: act/n/m, occ: occ/n, maxr: mx/m, edge: edge/n/m, bad };
  }`,
};

(async () => {
  const { br, pg } = await open();
  await pg.addScriptTag({ content: `${PAGE.readRho}\n${PAGE.metrics}\nwindow.readRho = readRho; window.metrics = metrics;` });
  if(mode === 'score'){
    const count = +a1, start = +a2, out = a3, res = [];
    const t0 = Date.now();
    for(let i = start; i < start + count; i++){
      const p = sample(i);
      const r = await pg.evaluate(p => {
        setParams(p); reseed(p.seed);
        for(let s = 0; s < 300; s++) step();
        const e = readRho(); const early = metrics(e);
        for(let s = 0; s < 350; s++) step();
        const a = readRho();
        for(let s = 0; s < 50; s++) step();
        const b = readRho();
        const late = metrics(b, a);
        late.cvEarly = early.cv; late.actEarly = early.act;
        return late;
      }, p);
      res.push({ ...p, ...r });
      if((i - start) % 5 === 4){
        fs.writeFileSync(out, JSON.stringify(res));
        console.log(`${i + 1 - start}/${count} ${((Date.now() - t0)/1000/(i + 1 - start)).toFixed(1)}s each`);
      }
    }
    fs.writeFileSync(out, JSON.stringify(res));
  } else if(mode === 'capture'){
    const list = JSON.parse(fs.readFileSync(a1, 'utf8')), outdir = a2;
    fs.mkdirSync(outdir, { recursive: true });
    const every = 20, nframes = 48;
    for(const p of list){
      const frames = await pg.evaluate(([p, every, nframes]) => {
        setParams(p); reseed(p.seed); S.view = 0; S.chan = -1; S.exp = 1.0;
        resize();
        const out = [];
        for(let f = 0; f < nframes; f++){
          if(f) for(let s = 0; s < every; s++) step();
          render();
          out.push(cv.toDataURL('image/png').split(',')[1]);
        }
        return out;
      }, [p, every, nframes]);
      const dir = path.join(outdir, String(p.idx).padStart(4, '0'));
      fs.mkdirSync(dir, { recursive: true });
      frames.forEach((b, i) => fs.writeFileSync(path.join(dir, `f${String(i).padStart(3, '0')}.png`), Buffer.from(b, 'base64')));
      fs.writeFileSync(path.join(dir, 'params.json'), JSON.stringify(p));
      console.log('captured', p.idx);
    }
  }
  await br.close(); process.exit(0);
})().catch(e => { console.error(e); process.exit(1); });
