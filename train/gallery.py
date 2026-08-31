"""
What the classification actually looks like, on digits it gets right and wrong.

    python3 train/gallery.py --name dig10 --images 300

Two strips: the ones it answers correctly and the ones it does not, each row a
digit rolled forward in time. The point is to make the mechanism visible rather
than summarised -- a number says 0.52, this says what the field is doing to earn
it and how it fails when it fails.

Each row reads left to right: the raw digit, then the field at increasing step
counts. Grey is the static digit, dim green the ten class regions, yellow the
pointer -- and the pointer is the answer, so watching it leave the centre and
commit to a region IS the classification. The last panel scores the regions: a
bar per class, the true one marked, so a confident wrong answer and a near-tie
look different.
"""

import argparse, json, os, sys
import numpy as np, torch
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import digits as dg, fieldlife as fl
from train_digits import Task, roll
from eval_digits import Replay

HERE = os.path.dirname(os.path.abspath(__file__))


def panel(f, task, ring, scale):
    """One frame: static digit grey, regions dim green, pointer yellow."""
    v = ring[..., None] * np.array([0.0, 0.28, 0.0])
    d = f[0].numpy()
    v = v + (d / max(d.max(), 1e-9))[..., None] * np.array([.40, .40, .45])
    p = f[task.ptr].numpy()
    v = v + (p / max(p.max(), 1e-9))[..., None] * np.array([1.0, .95, .25])
    return np.clip(v, 0, 1)


def bars(scores, truth, pred, grid):
    """Region scores as a column of bars, the true class marked."""
    v = np.zeros((grid, grid, 3))
    s = scores / max(scores.max(), 1e-9)
    h = max(1, grid // len(s))
    for k, val in enumerate(s):
        y0 = k * h
        w = int(val * (grid - 6))
        col = (np.array([.3, 1., .4]) if k == truth
               else (np.array([1., .35, .3]) if k == pred
                     else np.array([.45, .45, .5])))
        v[y0:y0 + h - 1, 3:3 + max(w, 1)] = col
    return v


def strip(rows, task, world, ring, steps, scale, title, path, siren):
    imgs, labs = rows
    seeds, _, y = task.build(imgs, labs, np.random.default_rng(3), siren)
    frames, rho, done = [], seeds.clone(), 0
    with torch.no_grad():
        for s in steps:
            rho = roll(world, task, rho, s - done)
            done = s
            frames.append(rho.clone())
        final = task.scores(frames[-1]).numpy()
    g = task.geo.grid
    out = []
    for k in range(len(labs)):
        raw = np.stack([task.geo.place(imgs[k])] * 3, -1)
        raw = raw / max(raw.max(), 1e-9)
        tiles = [raw] + [panel(f[k], task, ring, scale) for f in frames]
        tiles.append(bars(final[k], int(y[k]), int(final[k].argmax()), g))
        out.append(np.concatenate(tiles, axis=1))
    grid_img = (np.concatenate(out, axis=0) * 255).astype(np.uint8)
    im = Image.fromarray(grid_img).resize(
        (grid_img.shape[1] * scale, grid_img.shape[0] * scale), Image.NEAREST)
    band = Image.new("RGB", (im.width, im.height + 18), (0, 0, 0))
    band.paste(im, (0, 18))
    d = ImageDraw.Draw(band)
    heads = ["digit"] + [f"{s}" for s in steps] + ["scores"]
    for i, h in enumerate(heads):
        d.text((i * g * scale + 4, 4), h, fill=(190, 190, 190))
    for k in range(len(labs)):
        d.text((4, 18 + k * g * scale + 4),
               f"{int(y[k])}→{int(final[k].argmax())}", fill=(255, 255, 120))
    d.text((im.width - 150, 4), title, fill=(190, 190, 190))
    band.save(path)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="dig10")
    ap.add_argument("--images", type=int, default=300)
    ap.add_argument("--show", type=int, default=8, help="rows in each strip")
    ap.add_argument("--steps", default="0,8,16,32,64,128")
    ap.add_argument("--classes", type=int, default=10)
    ap.add_argument("--static", type=int, default=4)
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    run = os.path.join(HERE, "runs", a.name)
    cfg = json.load(open(os.path.join(run, "preset.json")))
    C = cfg["C"]; t = cfg.get("_task", {})
    task = Task(C, a.classes, cfg["N"], t.get("digit", dg.DIGIT), a.static,
                t.get("ring", dg.RING))
    world = Replay(cfg, C)
    siren = None
    ck = os.path.join(run, "ckpt.pt")
    if os.path.exists(ck):
        st = torch.load(ck, weights_only=False)
        if st.get("siren"):
            siren = dg.SirenSeed(cfg["N"], C - a.static - 1, 32, 2)
            siren.load_state_dict(st["siren"])

    x, y = dg.load("test")
    keep = y < a.classes
    x, y = x[keep][:a.images], y[keep][:a.images]
    steps = [int(s) for s in a.steps.split(",")]

    right, wrong = [], []
    with torch.no_grad():
        for lo in range(0, len(y), 50):
            seeds, _, lab = task.build(x[lo:lo + 50], y[lo:lo + 50],
                                       np.random.default_rng(3), siren)
            rho = roll(world, task, seeds.clone(), steps[-1])
            pred = task.scores(rho).argmax(1)
            for i, (p, l) in enumerate(zip(pred.tolist(), lab.tolist())):
                (right if p == l else wrong).append(lo + i)

    ring = task.geo.regions.sum(0); ring = ring / ring.max()
    print(f"{len(right)} right, {len(wrong)} wrong of {len(y)} "
          f"({len(right)/len(y):.3f})")
    for tag, idxs in (("correct", right), ("wrong", wrong)):
        sel = idxs[:a.show]
        if not sel:
            continue
        p = strip((x[sel], y[sel]), task, world, ring, steps, a.scale,
                  tag, os.path.join(run, f"gallery_{tag}.png"), siren)
        print(f"  wrote {p}  ({len(sel)} rows)")


if __name__ == "__main__":
    main()
