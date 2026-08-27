/*
 * Load a trained preset into the real simulation and dump what it does.
 *
 *   node train/verify_browser.mjs train/runs/polar3/preset.json [out.json]
 *
 * The claim being tested is that a trained world runs in index.html. It is
 * only true for kernels the shipped bakeKernel can actually bake, and the
 * point of running it in a browser rather than reasoning about it is that a
 * kernel it cannot bake does not fail loudly -- an angular term is simply not
 * read, and the preset loads as a different, radially symmetric world.
 */
import { chromium } from "playwright";
import { createServer } from "node:http";
import { readFile, writeFile } from "node:fs/promises";
import { extname, join, resolve } from "node:path";

const ROOT = resolve(import.meta.dirname, "..");
const CHROME = "/opt/pw-browsers/chromium-1194/chrome-linux/chrome";
const PRESET = process.argv[2] ?? join(ROOT, "train/runs/polar3/preset.json");
const OUT = process.argv[3] ?? join(ROOT, "train/browser_run.json");
const STEPS = Number(process.env.STEPS ?? 64);

const preset = JSON.parse(await readFile(PRESET, "utf8"));
const seed = JSON.parse(await readFile(join(ROOT, "train/seed_field.json"), "utf8"));

const MIME = { ".html": "text/html", ".json": "application/json", ".png": "image/png" };
const server = createServer(async (req, res) => {
  const p = join(ROOT, decodeURIComponent(req.url.split("?")[0]).replace(/^\/+/, "") || "index.html");
  try {
    res.writeHead(200, { "content-type": MIME[extname(p)] ?? "application/octet-stream" });
    res.end(await readFile(p));
  } catch { res.writeHead(404); res.end("no"); }
});
await new Promise(r => server.listen(0, "127.0.0.1", r));

const browser = await chromium.launch({
  executablePath: CHROME,
  args: ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"],
});
const page = await browser.newPage({ viewport: { width: 900, height: 900 } });
page.on("pageerror", e => console.error("page error:", e.message));
await page.goto(`http://127.0.0.1:${server.address().port}/index.html`,
                { waitUntil: "domcontentloaded" });
await page.waitForFunction(() => typeof S !== "undefined" && T && T.rhoA, null, { timeout: 60000 });

const out = await page.evaluate(async ({ preset, seed, STEPS }) => {
  S.running = false;
  await new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));
  const N = preset.N, C = preset.C;
  globalThis.sizeGrid = () => { S.NX = N; S.NY = N; };
  if (!applyConfig(preset)) throw new Error("applyConfig rejected the preset");
  S.noise = 0;

  const L = S.L, px = new Float32Array(S.NX * L * S.NY * 4);
  for (let c = 0; c < C; c++)
    for (let y = 0; y < N; y++)
      for (let x = 0; x < N; x++)
        px[(y * S.NX * L + (c >> 2) * S.NX + x) * 4 + (c & 3)] = seed[(c * N + y) * N + x];
  if (!restoreField({ px, nx: S.NX, ny: S.NY, L })) throw new Error("restoreField refused");

  for (let i = 0; i < STEPS; i++) step();

  const snap = snapshotField(), rho = new Array(C * N * N);
  for (let c = 0; c < C; c++)
    for (let y = 0; y < N; y++)
      for (let x = 0; x < N; x++)
        rho[(c * N + y) * N + x] = snap.px[(y * S.NX * L + (c >> 2) * S.NX + x) * 4 + (c & 3)];

  // and the bank the browser actually baked, which is the thing in question
  const KR = kernelKR, Rmax = Math.max(...kernels.slice(0, C).map(k => k.R), 1), bank = [];
  for (let c = 0; c < C; c++)
    bank.push(Array.from(bakeKernel(kernels[c],
      Math.max(2, Math.round(KR * Math.min(1, kernels[c].R / Rmax))))));
  return { rho, bank, kernelKR: KR, separable: separablePlan() !== null, C, N,
           renderer: gl.getParameter(gl.RENDERER) };
}, { preset, seed, STEPS });

await writeFile(OUT, JSON.stringify({ steps: STEPS, preset: PRESET, ...out }));
console.log(`ran ${STEPS} steps of ${PRESET} in the browser -> ${OUT}`);
console.log(`  kernelKR ${out.kernelKR}  separable path: ${out.separable}`);
await browser.close();
server.close();
