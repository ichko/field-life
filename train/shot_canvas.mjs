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
import { readFile, writeFile } from "node:fs/promises";
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
// loadLizard is gone: a world now arrives by clicking its entry on the shelf,
// and a preset whose seedMode is "masses" lands on the creature rail. Click it
// the way a person would, so this verifies the path the page actually uses.
await page.waitForSelector("#cshelf .w-item, #wshelf .w-item", { timeout: 60000 });
const picked = await page.evaluate(() => {
  const b = document.querySelector("#cshelf .w-item") ||
            document.querySelector("#wshelf .w-item");
  if (!b) return null;
  // Choose a rate BEFORE the click, so the shot answers whether loading a
  // world stamps its own rate over the one the viewer set.
  S.fps = 7;
  b.click();
  return b.querySelector("b")?.textContent ?? b.dataset.tip ?? "(unnamed)";
});
if (!picked) { console.error("no world on the shelf to load"); process.exit(1); }
console.log(`loaded "${picked}" off the shelf`);
await page.waitForFunction(() => S.seedMode === "masses" && S.seedMasses, null, { timeout: 30000 });

const info = await page.evaluate(async (steps) => {
  S.running = false;
  await new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));
  reseed();
  for (let i = 0; i < steps; i++) step();
  render();
  // Copy the drawing buffer here, synchronously, the way the page's own camera
  // button does: the context is created with preserveDrawingBuffer false, so
  // anything that waits for a frame -- an element screenshot, toBlob, even
  // page.screenshot's own font and stability waits -- photographs a buffer
  // that has already been wiped, or times out waiting for a canvas that never
  // holds still.
  const c = document.createElement("canvas");
  c.width = cv.width; c.height = cv.height;
  c.getContext("2d").drawImage(cv, 0, 0);
  return { tick: S.tick, blend: S.blend, expo: S.expo, C: S.C, fps: S.fps,
           grid: `${S.NX}x${S.NY}`, png: c.toDataURL("image/png") };
}, STEPS);
const { png, ...rest } = info;
console.log(JSON.stringify(rest));
if (errors.length) console.log("PAGE ERRORS:\n  " + errors.join("\n  "));
await writeFile(OUT, Buffer.from(png.split(",")[1], "base64"));
console.log(`wrote ${OUT}`);
await browser.close(); server.close();
