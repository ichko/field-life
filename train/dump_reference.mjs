/*
 * Step the REAL simulation and write the result out, so train/parity.py can
 * check the PyTorch port against it rather than against my reading of the
 * shaders.
 *
 * It loads index.html in headless Chromium, pins the grid, installs a known
 * field and a known kernel bank, calls step() N times, and dumps rho.
 *
 *   node train/dump_reference.mjs [out.json]
 *
 * The kernels here have three lobes on purpose: that makes separablePlan()
 * return null, so the browser takes the 2-D stencil path -- the one the port
 * covers, and the one any trained kernel will run on.
 */
import { chromium } from "playwright";
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { extname, join, resolve } from "node:path";
import { writeFileSync } from "node:fs";

const ROOT = resolve(import.meta.dirname, "..");
const CHROME = "/opt/pw-browsers/chromium-1194/chrome-linux/chrome";
const OUT = process.argv[2] ?? join(ROOT, "train", "reference.json");

const N = 64, C = 6;
const CHECKPOINTS = [0, 1, 2, 4, 8, 16, 32];   // snapshot the field at each
const STEPS = CHECKPOINTS[CHECKPOINTS.length - 1];

/* Two regimes, because they test different things.

   "hot" is what the page actually ships -- force 90, beta 3 -- and it is
   chaotic: the softmax in MaCE is winner-take-all there, so the smallest
   difference between two runs doubles every step or so. No port can track it
   for long in a different float width, and pretending otherwise would make the
   test meaningless. What it does test is that the port amplifies at the same
   rate the simulation does.

   "cool" turns the exponent down until the step stops amplifying. There the
   comparison is a real one, and the tolerance can be tight. */
const REGIMES = {
  cool: { force: 12, repel: 1.0, beta: 0.35 },
  hot:  { force: 90, repel: 6.0, beta: 3.0 },
};

/* A fixed, ugly-on-purpose bank: three lobes each, different reaches, so the
   stencil path runs and every channel exercises a different inset. */
const KERNELS = [
  { R: 12.0, terms: [[1.0, 0.00, 0.35], [-6.25, 0.00, 0.14], [0.30, 0.55, 0.18]] },
  { R: 9.5,  terms: [[0.8, 0.10, 0.28], [-3.10, 0.00, 0.12], [-0.40, 0.70, 0.22]] },
  { R: 13.0, terms: [[-0.6, 0.00, 0.40], [2.00, 0.30, 0.16], [0.15, 0.80, 0.25]] },
  { R: 7.0,  terms: [[1.2, 0.00, 0.30], [-2.00, 0.45, 0.20], [0.50, 0.85, 0.15]] },
  { R: 11.0, terms: [[0.4, 0.20, 0.33], [-1.50, 0.00, 0.18], [0.90, 0.65, 0.21]] },
  { R: 10.0, terms: [[-1.0, 0.05, 0.26], [1.70, 0.40, 0.19], [-0.25, 0.75, 0.24]] },
].map(k => ({
  type: "radial", sym: "radial", R: k.R, feather: 0.22, seed: 1, oct: 3,
  terms: k.terms.map(([a, r, w]) => ({ a, r, w })),
}));

/* A deterministic field, generated the same way on both sides (see parity.py). */
function makeField(n, c) {
  const rho = new Float64Array(c * n * n);
  let s = 12345 >>> 0;
  const rnd = () => (s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 4294967296;
  for (let i = 0; i < rho.length; i++) rho[i] = rnd() * 0.4 + 0.02;
  return rho;
}

const MIME = { ".html": "text/html", ".json": "application/json", ".png": "image/png" };
const server = createServer(async (req, res) => {
  const p = join(ROOT, decodeURIComponent(req.url.split("?")[0]).replace(/^\/+/, "") || "index.html");
  try {
    const body = await readFile(p);
    res.writeHead(200, { "content-type": MIME[extname(p)] ?? "application/octet-stream" });
    res.end(body);
  } catch { res.writeHead(404); res.end("no"); }
});
await new Promise(r => server.listen(0, "127.0.0.1", r));
const url = `http://127.0.0.1:${server.address().port}/index.html`;

const FIELD = makeField(N, C);

const browser = await chromium.launch({
  executablePath: CHROME,
  args: ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"],
});
const page = await browser.newPage({ viewport: { width: 900, height: 900 } });
page.on("pageerror", e => console.error("page error:", e.message));

await page.goto(url, { waitUntil: "domcontentloaded" });
await page.waitForFunction(() => typeof S !== "undefined" && T && T.rhoA, null, { timeout: 60000 });

async function runRegime(page, name, cfg) {
  return page.evaluate(async ({ N, C, STEPS, CHECKPOINTS, cfg, KERNELS, rho0 }) => {
  S.running = false;                                  // nothing steps but us
  await new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));

  // Pin the lattice. sizeGrid() derives it from the canvas, and we want a
  // square grid of a known size regardless of how the page laid itself out.
  globalThis.sizeGrid = () => { S.NX = N; S.NY = N; };
  S.N = N; S.C = C; S.noise = 0; S.steps = 1;
  S.force = cfg.force; S.repel = cfg.repel; S.beta = cfg.beta;
  S.paintMask = new Uint8Array(C).fill(1);
  kernels = KERNELS.map(k => JSON.parse(JSON.stringify(k)));
  mat = new Float32Array(C * C);
  for (let i = 0; i < C * C; i++) mat[i] = Math.sin(i * 1.7) * 0.8;   // deterministic
  allocate();
  uploadMatrix(); uploadKernels();

  if (separablePlan() !== null) throw new Error("expected the stencil path, got separable");
  if (kernelMip !== 0) throw new Error("expected kernelMip 0, got " + kernelMip);

  // Install the field: restoreField is an identity copy at matching dimensions.
  const L = S.L, px = new Float32Array(S.NX * L * S.NY * 4);
  for (let c = 0; c < C; c++)
    for (let y = 0; y < N; y++)
      for (let x = 0; x < N; x++)
        px[(y * S.NX * L + (c >> 2) * S.NX + x) * 4 + (c & 3)] = rho0[(c * N + y) * N + x];
  if (!restoreField({ px, nx: S.NX, ny: S.NY, L })) throw new Error("restoreField refused");

  const read = () => {
    const snap = snapshotField(), out = new Array(C * N * N);
    for (let c = 0; c < C; c++)
      for (let y = 0; y < N; y++)
        for (let x = 0; x < N; x++)
          out[(c * N + y) * N + x] = snap.px[(y * S.NX * L + (c >> 2) * S.NX + x) * 4 + (c & 3)];
    return out;
  };
  const frames = {};
  for (let i = 0; i <= STEPS; i++) {
    if (CHECKPOINTS.includes(i)) frames[i] = read();
    if (i < STEPS) step();
  }
  const out = frames[STEPS];

  // The baked bank too, so a kernel mismatch is distinguishable from a step one.
  const KR = kernelKR, K = 2 * KR + 1, bank = [];
  for (let c = 0; c < C; c++) bank.push(Array.from(bakeKernel(kernels[c],
      Math.max(2, Math.round(KR * Math.min(1, kernels[c].R / Math.max(...kernels.slice(0, C).map(k => k.R), 1)))))));
  return { frames, bank, kernelKR: KR, NX: S.NX, NY: S.NY,
           mat: Array.from(mat), renderer: gl.getParameter(gl.RENDERER) };
  }, { N, C, STEPS, CHECKPOINTS, cfg, KERNELS, rho0: Array.from(FIELD) });
}

const regimes = {};
for (const [name, cfg] of Object.entries(REGIMES)) regimes[name] = { cfg, ...await runRegime(page, name, cfg) };

const any = regimes[Object.keys(regimes)[0]];
writeFileSync(OUT, JSON.stringify({
  N, C, steps: STEPS, checkpoints: CHECKPOINTS, kernels: KERNELS,
  rho0: Array.from(FIELD), regimes,
  kernelKR: any.kernelKR, NX: any.NX, NY: any.NY, renderer: any.renderer,
}));
console.log(`wrote ${OUT}  (${any.NX}x${any.NY}, kernelKR ${any.kernelKR}, ${any.renderer})`);
console.log(`  regimes: ${Object.keys(regimes).join(", ")}`);

await browser.close();
server.close();
