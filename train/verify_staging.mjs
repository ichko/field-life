/*
 * Run staging/index.html's own copy of the world and dump the field, so it can
 * be compared against the same preset stepped in torch.
 *
 *   node train/verify_staging.mjs [steps]   ->  train/staging_run.json
 *
 * The page accepting a preset proves nothing: an unknown field on a kernel term
 * is simply not read, and the world loads as a different one without complaint.
 * Only stepping both and subtracting settles it.
 */
import { chromium } from "playwright";
import { createServer } from "node:http";
import { readFile, writeFile } from "node:fs/promises";
import { extname, join, resolve } from "node:path";

const ROOT = resolve(import.meta.dirname, "..");
const STEPS = Number(process.argv[2] ?? 64);
const MIME = { ".html":"text/html", ".json":"application/json", ".png":"image/png" };
const server = createServer(async (req, res) => {
  const p = join(ROOT, decodeURIComponent(req.url.split("?")[0]).replace(/^\/+/,"") || "index.html");
  try { res.writeHead(200, { "content-type": MIME[extname(p)] ?? "application/octet-stream" });
        res.end(await readFile(p)); }
  catch { res.writeHead(404); res.end("no"); }
});
await new Promise(r => server.listen(0, "127.0.0.1", r));

const browser = await chromium.launch({
  executablePath: "/opt/pw-browsers/chromium-1194/chrome-linux/chrome",
  args: ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"] });
const page = await browser.newPage({ viewport: { width: 900, height: 900 } });
const errors = [];
page.on("pageerror", e => errors.push(e.message));
await page.goto(`http://127.0.0.1:${server.address().port}/staging/index.html`,
                { waitUntil: "domcontentloaded" });
await page.waitForFunction(() => typeof S !== "undefined" && T && T.rhoA, null, { timeout: 60000 });
await page.waitForFunction(() => S.seedMode === "masses" && S.square === true, null, { timeout: 30000 });

const out = await page.evaluate(async (steps) => {
  S.running = false;
  await new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));
  reseed();                                   // back to the seed, deterministically
  const C = S.C, N = S.NX;
  const seed = Array.from(FL.readRho());
  for (let i = 0; i < steps; i++) step();
  const rho = Array.from(FL.readRho());

  const Rmax = Math.max(...kernels.slice(0, C).map(k => k.R), 1), bank = [];
  for (let c = 0; c < C; c++)
    bank.push(Array.from(bakeKernel(kernels[c], kernelKR,
      Math.max(2, kernelKR*Math.min(1, kernels[c].R/Rmax)))));

  return { C, N, NY: S.NY, seed, rho, bank, kernelKR,
           separable: separablePlan() !== null, mip: kernelMip,
           force: S.force, repel: S.repel, beta: S.beta, noise: S.noise,
           steps: S.steps, radiusCap: radiusCap(),
           Rs: kernels.slice(0, C).map(k => k.R) };
}, STEPS);

await writeFile(join(ROOT, "train/staging_run.json"), JSON.stringify({ ranSteps: STEPS, ...out }));
console.log(`grid ${out.N}x${out.NY}  C ${out.C}  kernelKR ${out.kernelKR}  mip ${out.mip}`);
console.log(`  force ${out.force}  repel ${out.repel}  beta ${out.beta}  noise ${out.noise}`);
console.log(`  separable ${out.separable}  steps/frame ${out.steps}  radiusCap ${out.radiusCap}`);
if (errors.length) console.log("PAGE ERRORS: " + errors.join(" | "));
await browser.close(); server.close();
