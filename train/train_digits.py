"""
Train field-life to classify a digit by moving it.

    python3 train/train_digits.py --name dig10

The world is rung 0 of docs/nca-experiment.md: the interaction matrix, one polar
kernel per channel, force/repel/beta. About 190 numbers. The task is
train/digits.py -- an MNIST digit in channel 0, chemicals in the rest, ten class
regions on a ring, and the answer is which region the digit's matter ends up in.

Three things the task forces on the loop.

**The chemicals land somewhere new every time.** A reseed draws a new digit AND
a new ball position, so a rule that only works when the reagents happen to sit
on the digit never gets to keep the score it got that way.

**A frame is scored on its absolute age.** The digit has to travel fourteen
cells to reach the ring and MaCE moves mass one cell per step, so nothing can be
right before step fourteen, and with a sample pool a state's age is not its
position in the backprop window.

**Loss trains, accuracy decides.** L2 against "all of the digit inside region k"
gives a dense gradient from anywhere on the field, which softmax over region
occupancy does not when the digit is still in the middle touching no region at
all. But what is reported, and what picks the checkpoint to keep, is whether
argmax over the ten regions is the right class on digits never trained on.
"""

import argparse
import atexit
import csv
import json
import math
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import digits as dg
from train import World, divergence_rate

HERE = os.path.dirname(os.path.abspath(__file__))
HORIZONS = [32, 64, 96, 128, 192, 256]


class Task:
    def __init__(self, C, nclass=10, grid=dg.GRID, digit=dg.DIGIT,
                 dtype=torch.float32):
        self.geo = dg.Geometry(grid=grid, digit=digit, nclass=nclass)
        self.C, self.dtype, self.nclass = C, dtype, nclass
        self.w = torch.tensor(self.geo.w, dtype=dtype)

    def build(self, imgs, labs, rng):
        n = len(labs)
        g = self.geo
        seeds = torch.zeros(n, self.C, g.grid, g.grid, dtype=self.dtype)
        targets = torch.zeros(n, g.grid, g.grid, dtype=self.dtype)
        for k in range(n):
            s = g.seed(imgs[k], self.C, rng)
            seeds[k] = torch.tensor(s, dtype=self.dtype)
            targets[k] = torch.tensor(g.target(int(labs[k]), s[0].sum()),
                                      dtype=self.dtype)
        return seeds, targets, torch.tensor(np.asarray(labs), dtype=torch.long)

    def scores(self, rho):
        return torch.einsum("khw,bhw->bk", self.w, rho[:, 0])

    def loss(self, rho, target):
        return ((rho[:, 0] - target) ** 2).mean(dim=(1, 2))


def accuracy(model, task, imgs, labs, horizons, rng, chunk=32):
    hits = [0] * len(horizons)
    n = len(labs)
    with torch.no_grad():
        for lo in range(0, n, chunk):
            seeds, _, y = task.build(imgs[lo:lo + chunk], labs[lo:lo + chunk], rng)
            rho, done = seeds.clone(), 0
            for j, h in enumerate(horizons):
                rho = model.rollout(rho, h - done)
                done = h
                hits[j] += int((task.scores(rho).argmax(1) == y).sum())
    return [v / max(n, 1) for v in hits]


def save_progress(path, model, task, imgs, labs, horizons, rng):
    seeds, targets, y = task.build(imgs, labs, rng)
    ring = torch.tensor(task.geo.regions.sum(0), dtype=torch.float32)
    ring = ring / ring.max()
    with torch.no_grad():
        rho, done, frames = seeds.clone(), 0, [seeds.clone()]
        for h in horizons:
            rho = model.rollout(rho, h - done)
            done = h
            frames.append(rho.clone())
    rows = []
    for k in range(len(labs)):
        tiles = []
        for f in frames:
            d = f[k, 0].numpy()
            chem = f[k, 1:].sum(0).numpy()
            tiles.append(np.stack([d / max(d.max(), 1e-9), ring.numpy(),
                                   chem / max(chem.max(), 1e-9)], -1))
        rows.append(np.concatenate([np.clip(t, 0, 1) for t in tiles], axis=1))
    img = Image.fromarray((np.concatenate(rows, axis=0) * 255).astype(np.uint8))
    img.resize((img.width * 3, img.height * 3), Image.NEAREST).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="dig10")
    ap.add_argument("--classes", type=int, default=10,
                    help="how many digits to tell apart. Ten is the task; fewer "
                         "is the honest way to find out whether the substrate "
                         "can do any of it before blaming the class count.")
    ap.add_argument("--grid", type=int, default=dg.GRID)
    ap.add_argument("--digit", type=int, default=dg.DIGIT)
    ap.add_argument("--orders", default="0,0,0,1,1,2")
    ap.add_argument("--channels", type=int, default=6)
    ap.add_argument("--kr", type=int, default=9)
    ap.add_argument("--mat-init", default="zeros", choices=["random", "zeros"])
    ap.add_argument("--hidden", type=int, default=0)
    ap.add_argument("--iters", type=int, default=1000000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--settle", type=int, default=24)
    ap.add_argument("--loss-every", type=int, default=4)
    ap.add_argument("--pool", type=int, default=64)
    ap.add_argument("--max-age", type=int, default=128)
    ap.add_argument("--young-frac", type=float, default=0.5)
    ap.add_argument("--young-age", type=int, default=48)
    ap.add_argument("--train-images", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--lam-penalty", type=float, default=0.02)
    ap.add_argument("--ckpt", type=int, default=100)
    ap.add_argument("--eval-images", type=int, default=200)
    ap.add_argument("--minutes", type=float, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    orders = tuple(int(x) for x in a.orders.split(","))
    run = os.path.join(HERE, "runs", a.name)
    os.makedirs(run, exist_ok=True)
    lock = os.path.join(run, "RUNNING")
    if os.path.exists(lock):
        pid = open(lock).read().strip()
        if pid.isdigit() and os.path.exists(f"/proc/{pid}"):
            sys.exit(f"{a.name} is already being trained by pid {pid}.")
    with open(lock, "w") as fh:
        fh.write(str(os.getpid()))
    atexit.register(lambda: os.path.exists(lock) and os.remove(lock))

    task = Task(a.channels, a.classes, a.grid, a.digit)
    rng = np.random.default_rng(a.seed)
    xtr, ytr = dg.load("train")
    xte, yte = dg.load("test")
    keep = ytr < a.classes
    xtr, ytr = xtr[keep][:a.train_images], ytr[keep][:a.train_images]
    keept = yte < a.classes
    xte, yte = xte[keept][:a.eval_images], yte[keept][:a.eval_images]

    model = World(a.channels, orders, a.kr, seed=a.seed, hidden=a.hidden,
                  mat_init=a.mat_init, mip=0)
    with torch.no_grad():
        model.kern.logR.fill_(math.log(float(a.kr)))
    model.kern.logR.requires_grad_(False)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], a.lr)

    pick = rng.integers(0, len(ytr), a.pool)
    pool, pool_t, pool_y = task.build(xtr[pick], ytr[pick], rng)
    ages = torch.zeros(a.pool, dtype=torch.long)
    n_young = max(1, min(a.pool - 1, int(a.pool * a.young_frac)))
    lifespan = torch.full((a.pool,), a.max_age, dtype=torch.long)
    lifespan[:n_young] = a.young_age
    start_it = 0

    def fresh(k):
        """A new digit AND a new place for the chemicals to land."""
        j = int(rng.integers(0, len(ytr)))
        s, t, y = task.build(xtr[j:j + 1], ytr[j:j + 1], rng)
        pool[k], pool_t[k], pool_y[k], ages[k] = s[0], t[0], y[0], 0

    ck = os.path.join(run, "ckpt.pt")
    if a.resume and os.path.exists(ck):
        st = torch.load(ck, weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
        pool, pool_t, pool_y = st["pool"], st["pool_t"], st["pool_y"]
        ages, start_it = st["ages"], st["iter"]
        print(f"resumed {a.name} at iteration {start_it}")

    logp = os.path.join(run, "log.csv")
    if not os.path.exists(logp):
        with open(logp, "w", newline="") as f:
            csv.writer(f).writerow(["iter", "loss", "lam", "force", "beta"]
                                   + [f"acc{h}" for h in HORIZONS]
                                   + ["acc_test", "age", "secs"])

    print(f"run {a.name}: {a.classes}-way MNIST by transport, grid {a.grid}, "
          f"digit {a.digit}, {len(ytr)} training images")
    print(f"  window {a.window}  batch {a.batch}  pool {a.pool}  settle "
          f"{a.settle}  stencil {a.kr}  orders {orders}")
    print(f"  {sum(p.numel() for p in model.parameters() if p.requires_grad)} "
          f"trainable numbers; chance is {1 / a.classes:.2f}")

    t0, best = time.time(), -1.0
    bp = os.path.join(run, "preset-best.json")
    for it in range(start_it, a.iters):
        half = max(1, a.batch // 2)
        idx = torch.cat([torch.randint(0, n_young, (half,)),
                         torch.randint(n_young, a.pool, (a.batch - half,))])
        for k in idx.tolist():
            if ages[k] >= lifespan[k]:
                fresh(k)
        batch, btgt, bage = pool[idx].clone(), pool_t[idx], ages[idx]

        kern = model.kern()
        out, scored = batch, []
        for t in range(a.window):
            out = model.rollout(out, 1, kern=kern)
            step_no = t + 1
            if not (step_no == a.window or not a.loss_every
                    or step_no % a.loss_every == 0):
                continue
            ready = (bage + step_no) >= a.settle
            if not ready.any():
                continue
            err = task.loss(out, btgt)
            scored.append((err * ready).sum() / ready.sum())
        loss = torch.stack(scored).mean() if scored else task.loss(out, btgt).mean()

        lam = 0.0
        if a.lam_penalty > 0 and it % 10 == 0:
            lam = float(np.nan_to_num(divergence_rate(model, batch.detach()), nan=0.0))
        total = loss + a.lam_penalty * max(lam, 0.0) * loss.detach()

        opt.zero_grad()
        total.backward()
        for p in model.parameters():
            if p.grad is not None:
                p.grad /= p.grad.norm() + 1e-8
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        with torch.no_grad():
            good = torch.isfinite(out).all(dim=(1, 2, 3))
            pool[idx[good]] = out[good].detach()
            ages[idx[good]] += a.window
            for k in idx[~good].tolist():
                fresh(k)

        if it % a.ckpt == 0 or it == a.iters - 1:
            f, r, b = model.scalars()
            secs = time.time() - t0
            ev = np.random.default_rng(1234)     # the same digits every checkpoint
            acc = accuracy(model, task, xte, yte, HORIZONS, ev)
            score = max(acc)
            with open(logp, "a", newline="") as fh:
                csv.writer(fh).writerow(
                    [it, f"{loss.item():.6f}", f"{lam:.3f}", f"{f:.2f}", f"{b:.3f}"]
                    + [f"{v:.4f}" for v in acc]
                    + [f"{acc[-1]:.4f}", int(ages.float().mean()), f"{secs:.0f}"])
            cfg = model.to_config(a.channels, a.grid)
            cfg["_task"] = {"classes": a.classes, "grid": a.grid,
                            "digit": a.digit, "ring": dg.RING}
            json.dump(cfg, open(os.path.join(run, "preset.json"), "w"))
            if score > best:
                best = score
                json.dump(dict(cfg, _bestAt=it, _acc=acc), open(bp, "w"))
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "pool": pool, "pool_t": pool_t, "pool_y": pool_y,
                        "ages": ages, "iter": it}, ck)
            save_progress(os.path.join(run, "progress.png"), model, task,
                          xte[:4], yte[:4], HORIZONS, np.random.default_rng(7))
            print(f"  it {it:6d}  loss {loss.item():.5f}  force {f:6.1f}  "
                  f"beta {b:.2f}  lam {lam:+.2f}  age~{int(ages.float().mean())}  "
                  f"test acc {'/'.join(f'{v:.3f}' for v in acc)}  best {best:.3f}  "
                  f"{secs / max(it - start_it + 1, 1):.2f}s/it", flush=True)

        if a.minutes and time.time() - t0 > a.minutes * 60:
            print(f"stopping after {a.minutes} minutes at iteration {it}")
            break


if __name__ == "__main__":
    main()
