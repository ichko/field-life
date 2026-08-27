"""
Every run side by side.

    python3 train/compare.py            # table of all runs
    python3 train/compare.py --png out.png

Reads what the runs already wrote -- no simulation -- so it is safe to call
while they are still going. The horizon columns are the ones that matter: a
model that reaches the target and then loses it has a curve rising from h16,
and one that holds has a curve that stays flat or dips later.
"""

import argparse
import csv
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def read(run):
    log = os.path.join(run, "log.csv")
    if not os.path.exists(log):
        return None
    rows = list(csv.DictReader(open(log)))
    if not rows:
        return None
    last = rows[-1]
    best = min(float(r["best"]) for r in rows)
    cfg = {}
    p = os.path.join(run, "preset.json")
    if os.path.exists(p):
        try:
            cfg = json.load(open(p))
        except json.JSONDecodeError:      # a checkpoint mid-write
            cfg = {}
    lobes = len(cfg.get("kernels", [{}])[0].get("terms", [])) if cfg.get("kernels") else 0
    orders = sorted({t.get("m", 0) for k in cfg.get("kernels", []) for t in k.get("terms", [])})
    return {
        "name": os.path.basename(run),
        "iter": int(last["iter"]),
        "best": best,
        "C": cfg.get("C", 0),
        "lobes": lobes,
        "orders": ",".join(str(o) for o in orders) if orders else "-",
        "rung": 1 if "_note" in cfg else 0,
        "h": [last.get(k) for k in ("h16", "h32", "h64", "h128")],
        "mins": float(last["secs"]) / 60,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--png", default=None, help="stack every run's progress strip")
    a = ap.parse_args()

    runs = [r for r in (read(d) for d in sorted(glob.glob(os.path.join(HERE, "runs", "*"))))
            if r]
    if not runs:
        return print("no runs yet")

    print(f"{'run':<10} {'rung':>4} {'C':>3} {'lobes':>5} {'orders':>8} {'iter':>6} "
          f"{'best':>9}   {'h16':>8} {'h32':>8} {'h64':>8} {'h128':>8}  {'min':>5}")
    for r in runs:
        h = "".join(f"{(float(v) if v else 0):>9.5f}" if v else f"{'-':>9}" for v in r["h"])
        print(f"{r['name']:<10} {r['rung']:>4} {r['C']:>3} {r['lobes']:>5} "
              f"{r['orders']:>8} {r['iter']:>6} {r['best']:>9.5f}  {h}  {r['mins']:>5.0f}")
    print("\nh<N> is the loss from the seed after N steps. Training window is 16, so a")
    print("curve rising left to right means the shape is reached and then lost.")

    if a.png:
        from PIL import Image, ImageDraw
        strips = [(r["name"], os.path.join(HERE, "runs", r["name"], "progress.png"))
                  for r in runs]
        strips = [(n, p) for n, p in strips if os.path.exists(p)]
        imgs = [Image.open(p).convert("RGB") for _, p in strips]
        w = max(i.width for i in imgs)
        out = Image.new("RGB", (w, sum(i.height for i in imgs)))
        y = 0
        for (name, _), im in zip(strips, imgs):
            out.paste(im, (0, y))
            ImageDraw.Draw(out).text((8, y + 6), name, fill=(255, 255, 255))
            y += im.height
        out.save(a.png)
        print(f"\nwrote {a.png}  (rows: {', '.join(n for n, _ in strips)})")


if __name__ == "__main__":
    main()
