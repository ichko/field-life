/*
 * Screenshot the canvas alone, at an exact tick, with the world paused.
 *
 *   node train/shot_canvas.mjs [steps] [out.png]
 *
 * shot.mjs photographs the page while it runs, which answers "does it load"
 * but not "does it draw what it computed": the tick is whatever the frame
 * clock reached, and swiftshader's is slow and uneven. Comparing the page's
 * own picture against an offline render of the same field needs both at the
 * same step, so this pauses, reseeds, steps a fixed number of times, and
 * photographs only the canvas.
 */
import { chromium } from "playwright";
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { extname, join, resolve } from "node:path";

const ROOT = resolve(import.meta.dirname, "..");
const STEPS = Number(process.argv[2] ?? 128);
const OUT = process.argv[3] ?? join(ROOT, "train/canvas.png");
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
const page = await browser.newPage({ viewport: { width: 1100, height: 1100 } });
const errors = [];
page.on("pageerror", e => errors.push(e.message));
await page.goto(`http://127.0.0.1:${server.address().port}/staging/index.html`,
                { waitUntil: "domcontentloaded" });
await page.waitForFunction(() => typeof S !== "undefined" && T && T.rhoA, null, { timeout: 60000 });
await page.waitForFunction(() => S.seedMode === "masses" && S.seedMasses, null, { timeout: 30000 });

const info = await page.evaluate(async (steps) => {
  S.running = false;
  await new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));
  reseed();
  for (let i = 0; i < steps; i++) step();
  render();
  await new Promise(r => requestAnimationFrame(r));
  return { tick: S.tick, blend: S.blend, expo: S.expo, C: S.C, grid: `${S.NX}x${S.NY}` };
}, STEPS);
console.log(JSON.stringify(info));
if (errors.length) console.log("PAGE ERRORS:\n  " + errors.join("\n  "));
// Clip a full-page shot to the canvas rather than shooting the element: the
// element screenshot waits for the node to be "stable", and a canvas the page
// keeps repainting never is, so it times out having drawn nothing.
const box = await page.locator("canvas").first().boundingBox();
await page.screenshot({ path: OUT, clip: box });
await browser.close(); server.close();
