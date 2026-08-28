/*
 * Grow the staged world in the page and screenshot it, so its orientation can
 * be checked against the target.
 *
 *   node train/check_orientation.mjs [steps] [out.png]
 *
 * This needs a screenshot, not a subtraction: reading the field back through
 * the same convention it was written with cancels any flip, so the array-level
 * checks in verify_staging.mjs are blind to exactly this bug -- as they were.
 */
import { chromium } from "playwright";
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { extname, join, resolve } from "node:path";

const ROOT = resolve(import.meta.dirname, "..");
const STEPS = Number(process.argv[2] ?? 64);
const OUT = process.argv[3] ?? join(ROOT, "train/orientation.png");
const MIME = {".html":"text/html", ".json":"application/json", ".png":"image/png"};
const server = createServer(async (q, r) => {
  const p = join(ROOT, decodeURIComponent(q.url.split("?")[0]).replace(/^\/+/,"") || "index.html");
  try { r.writeHead(200, {"content-type": MIME[extname(p)] ?? "application/octet-stream"});
        r.end(await readFile(p)); } catch { r.writeHead(404); r.end(); }
});
await new Promise(r => server.listen(0, "127.0.0.1", r));

const b = await chromium.launch({ executablePath: "/opt/pw-browsers/chromium-1194/chrome-linux/chrome",
  args: ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"] });
const p = await b.newPage({ viewport: { width: 900, height: 900 } });
await p.goto(`http://127.0.0.1:${server.address().port}/staging/index.html`, {waitUntil:"domcontentloaded"});
await p.waitForFunction(() => typeof S !== "undefined" && T && T.rhoA, null, {timeout:60000});
await p.waitForFunction(() => S.seedMode === "masses", null, {timeout:30000});
const box = await p.evaluate(async (steps) => {
  S.running = false;
  await new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));
  reseed();
  for (let i = 0; i < steps; i++) step();
  render();
  // where the world sits inside the canvas: it is fitted to the shorter axis
  const side = Math.min(cv.width, cv.height);
  return { x: (cv.width - side)/2, y: (cv.height - side)/2, side,
           w: cv.width, h: cv.height };
}, STEPS);
await p.waitForTimeout(300);
await p.locator("#cv").screenshot({ path: OUT, clip: box });
console.log(`step ${STEPS} of the staged world -> ${OUT} (${box.side}px of world)`);
await b.close(); server.close();
