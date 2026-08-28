/*
 * Does index.html bake the kernel the trainer thinks it does?
 *
 *   node train/verify_bake.mjs [preset.json] > train/browser_bank.json
 *
 * Bakes a preset's kernel bank inside the real page and prints it, so
 * train/verify_bake.py can hold it against the PyTorch bake. This is the check
 * the lizard experiment learned to run: a kernel field index.html does not know
 * about is not rejected, it is silently ignored, and the preset then loads as a
 * different, rotationally symmetric world with nothing anywhere reporting it.
 *
 * Only the bake is exercised -- no simulation, no WebGL -- which is why it is
 * seconds rather than minutes and safe to run beside a training job.
 */
import { chromium } from "playwright";
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { extname, join, resolve } from "node:path";

const ROOT = resolve(import.meta.dirname, "..");
const CHROME = "/opt/pw-browsers/chromium-1194/chrome-linux/chrome";
const PRESET = process.argv[2] ?? join(ROOT, "train/runs/add4/preset.json");
const preset = JSON.parse(await readFile(PRESET, "utf8"));

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
const page = await browser.newPage();
page.on("pageerror", e => console.error("page error:", e.message));
await page.goto(`http://127.0.0.1:${server.address().port}/index.html`,
                { waitUntil: "domcontentloaded" });
await page.waitForFunction(() => typeof bakeKernel === "function", null, { timeout: 60000 });

const out = await page.evaluate(({ preset }) => {
  const C = preset.C, ks = preset.kernels.slice(0, C);
  // uploadKernels' own arithmetic: one shared grid from the widest reach, and
  // every other channel resampled to a coarser one in proportion to its own.
  const Rmax = Math.max(...ks.map(k => k.R), 1);
  const KR = Math.max(3, Math.min(15, Math.round(Rmax)));
  return {
    KR, Rmax,
    banks: ks.map(k => {
      const kr = Math.max(2, Math.round(KR * Math.min(1, k.R / Rmax)));
      return { kr, w: Array.from(bakeKernel(k, kr)) };
    }),
    separable: separablePlan === undefined ? null : "checked-elsewhere",
  };
}, { preset });

console.log(JSON.stringify(out));
await browser.close();
server.close();
