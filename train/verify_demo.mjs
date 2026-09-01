/*
 * Does staging/ncpu.html run the trained world, or something close to it?
 *
 *   node train/verify_demo.mjs > train/demo_run.json
 *   python3 train/verify_demo.py
 *
 * Steps the page in a real browser on a fixed sample and dumps the pointer
 * channel and the region vote, so the PyTorch port can be held against it. The
 * page reimplements the law in plain JavaScript -- the static channels are
 * convolved once and the crowding blur is folded into a single convolution --
 * and both of those are only sound if the numbers come out the same.
 */
import { chromium } from "playwright";
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { extname, join, resolve } from "node:path";

const ROOT = resolve(import.meta.dirname, "..");
const CHROME = "/opt/pw-browsers/chromium-1194/chrome-linux/chrome";
const STEPS = (process.env.STEPS ?? "1,4,16,64").split(",").map(Number);
const SAMPLE = Number(process.env.SAMPLE ?? 0);

const MIME = { ".html":"text/html", ".js":"text/javascript", ".json":"application/json" };
const server = createServer(async (req, res) => {
  const p = join(ROOT, decodeURIComponent(req.url.split("?")[0]).replace(/^\/+/, ""));
  // read BEFORE writing headers: a miss used to throw after the 200 was already
  // on the wire, and the catch then tried to send a second set
  let body;
  try { body = await readFile(p); }
  catch { res.writeHead(404); res.end("no"); return; }
  res.writeHead(200, { "content-type": MIME[extname(p)] ?? "application/octet-stream" });
  res.end(body);
});
await new Promise(r => server.listen(0, "127.0.0.1", r));

const browser = await chromium.launch({ executablePath: CHROME });
const page = await browser.newPage();
page.on("pageerror", e => console.error("page error:", e.message));
await page.goto(`http://127.0.0.1:${server.address().port}/staging/ncpu.html`,
                { waitUntil: "domcontentloaded" });
await page.waitForFunction(() => window.__ncpu, null, { timeout: 30000 });

const out = await page.evaluate(({ STEPS, SAMPLE }) => {
  window.__ncpu.setSample(SAMPLE);
  const res = { sample: SAMPLE, frames: [] };
  let done = 0;
  for (const s of STEPS) {
    window.__ncpu.tick(s - done); done = s;
    res.frames.push({ step: s, scores: window.__ncpu.scores(),
                      mass: window.__ncpu.mass(),
                      pointer: window.__ncpu.pointer() });
  }
  return res;
}, { STEPS, SAMPLE });

console.log(JSON.stringify(out));
await browser.close();
server.close();
