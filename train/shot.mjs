/*
 * Open a page in headless Chromium, report what the simulation is running, and
 * save a screenshot. Used to check that staging/index.html really loads a
 * trained world rather than merely accepting the file.
 *
 *   node train/shot.mjs staging/index.html out.png
 */
import { chromium } from "playwright";
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { extname, join, resolve } from "node:path";

const ROOT = resolve(import.meta.dirname, "..");
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
const page = await browser.newPage({ viewport: { width: 1280, height: 860 } });
const errors = [];
page.on("pageerror", e => errors.push(e.message));
await page.goto(`http://127.0.0.1:${server.address().port}/${process.argv[2] ?? "index.html"}`,
                { waitUntil: "domcontentloaded" });
await page.waitForFunction(() => typeof S !== "undefined" && T && T.rhoA, null, { timeout: 60000 });
await page.waitForTimeout(Number(process.env.WAIT ?? 7000));

const info = await page.evaluate(() => ({
  C: S.C, grid: `${S.NX}x${S.NY}`, blend: S.blend, tick: S.tick, running: S.running,
  shelf: document.querySelectorAll(".w-item").length,
  first: document.querySelector(".w-item")?.dataset.tip ?? null,
  separable: separablePlan() !== null,
  glError: document.getElementById("gl-error")?.textContent || "",
}));
console.log(JSON.stringify(info, null, 1));
if (errors.length) console.log("PAGE ERRORS:\n  " + errors.join("\n  "));
if (process.argv[3]) await page.screenshot({ path: process.argv[3] });
await browser.close(); server.close();
