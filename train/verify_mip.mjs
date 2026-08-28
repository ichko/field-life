/*
 * Check the ported mip path against index.html.
 *
 *   node train/verify_mip.mjs
 *
 * A reach wider than the stencil is served by convolving a halved copy of the
 * field, and that path had never been ported -- the port asserted "kernelMip
 * is 0 at these grids" and was right only while the grids were small. This
 * builds a world that forces mip 1, steps it in the browser, and writes the
 * field out for train/verify_mip.py to subtract.
 */
import { chromium } from "playwright";
import { createServer } from "node:http";
import { readFile, writeFile } from "node:fs/promises";
import { extname, join, resolve } from "node:path";

const ROOT = resolve(import.meta.dirname, "..");
const STEPS = Number(process.env.STEPS ?? 24), N = 144, C = 12;
const MIME = { ".html":"text/html", ".json":"application/json", ".png":"image/png" };
const server = createServer(async (q, r) => {
  const p = join(ROOT, decodeURIComponent(q.url.split("?")[0]).replace(/^\/+/,"") || "index.html");
  try { r.writeHead(200, {"content-type": MIME[extname(p)] ?? "application/octet-stream"});
        r.end(await readFile(p)); } catch { r.writeHead(404); r.end(); }
});
await new Promise(r => server.listen(0, "127.0.0.1", r));

// a world whose widest reach is 26 cells: past KMAX, so the browser must mip
const rnd = (s => () => (s = (Math.imul(s,1664525)+1013904223)>>>0)/4294967296)(99);
const kernels = Array.from({length: C}, (_, c) => ({
  type: "radial", sym: "radial", R: 26 - c*1.4, feather: 0.25, seed: 1, oct: 3,
  terms: [{a: rnd()*2-1, r: rnd()*0.6, w: 0.25 + rnd()*0.3, m: c % 3, phase: rnd()*6},
          {a: rnd()*2-1, r: rnd()*0.6, w: 0.20 + rnd()*0.3, m: (c+1) % 4, phase: rnd()*6},
          {a: rnd()*2-1, r: rnd()*0.6, w: 0.30 + rnd()*0.3, m: 0, phase: 0}],
}));
const mat = Array.from({length: C*C}, () => rnd()*1.2 - 0.6);
const cfg = { seed: 1, mat, kernels, N, C, density: 0.12,
  radMin: Math.min(...kernels.map(k=>k.R)), radMax: Math.max(...kernels.map(k=>k.R)),
  force: 18, repel: 0.4, beta: 0.5, noise: 0, steps: 1, expo: 1,
  seedMode: "disc", palette: "Spectrum", blend: 5, square: true,
  kterms: 2, kwidth: 0.7, ksym: "radial", cfreq: 1.1, cdepth: 2 };

const rho0 = Array.from({length: C*N*N}, () => rnd()*0.35 + 0.02);

const browser = await chromium.launch({
  executablePath: "/opt/pw-browsers/chromium-1194/chrome-linux/chrome",
  args: ["--use-gl=angle","--use-angle=swiftshader","--enable-unsafe-swiftshader"] });
const page = await browser.newPage({ viewport: { width: 900, height: 900 } });
const errs = []; page.on("pageerror", e => errs.push(e.message));
await page.goto(`http://127.0.0.1:${server.address().port}/staging/index.html`, {waitUntil:"domcontentloaded"});
await page.waitForFunction(() => typeof S !== "undefined" && T && T.rhoA, null, {timeout:60000});

const out = await page.evaluate(async ({cfg, rho0, STEPS, N, C}) => {
  S.running = false;
  await new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));
  globalThis.sizeGrid = () => { S.NX = S.NY = N; };
  if (!applyConfig(cfg)) throw new Error("preset rejected");
  allocate();
  const flat = new Float32Array(C*N*N);
  for (let i = 0; i < flat.length; i++) flat[i] = rho0[i];
  FL.writeRho(flat);
  for (let i = 0; i < STEPS; i++) step();
  // T.far holds the interaction integral computed from the field as it was at
  // the start of the last step, so after one step it is U(rho0) -- the
  // convolution on its own, before the transport touches it.
  const readTex = (tex) => {
    const w = S.NX*S.L, h = S.NY, px = new Float32Array(w*h*4);
    const fb = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    gl.drawBuffers([gl.COLOR_ATTACHMENT0]);
    gl.readPixels(0, 0, w, h, gl.RGBA, gl.FLOAT, px);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null); gl.deleteFramebuffer(fb);
    const out = new Array(C*N*N);
    for (let c = 0; c < C; c++)
      for (let y = 0; y < N; y++)
        for (let x = 0; x < N; x++)
          out[(c*N + y)*N + x] = px[(y*w + (c>>2)*S.NX + x)*4 + (c&3)];
    return out;
  };
  const Rmax = Math.max(...kernels.slice(0, C).map(k => k.R), 1), bank = [];
  for (let c = 0; c < C; c++)
    bank.push(Array.from(bakeKernel(kernels[c], kernelKR,
      Math.max(2, kernelKR*Math.min(1, kernels[c].R/Rmax)))));
  return { rho: Array.from(FL.readRho()), far: readTex(T.far), near: readTex(T.near),
           bank, mip: kernelMip, kernelKR,
           NX: S.NX, NY: S.NY, pyramid: pyramid.length,
           separable: separablePlan() !== null };
}, {cfg, rho0, STEPS, N, C});

await writeFile(join(ROOT,"train/mip_run.json"),
  JSON.stringify({ cfg, rho0, steps: STEPS, N, C, ...out }));
console.log(`grid ${out.NX}x${out.NY}  pyramid levels ${out.pyramid}  mip ${out.mip}  stencil ${out.kernelKR}  separable ${out.separable}`);
if (errs.length) console.log("PAGE ERRORS: " + errs.join(" | "));
await browser.close(); server.close();
