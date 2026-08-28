"""
Train field-life to add two 8-bit numbers.

    python3 train/train_adder.py --name add8

The world is rung 0 of docs/nca-experiment.md, unchanged: the interaction
matrix, one polar kernel per channel, and force/repel/beta. About 190 numbers,
against 65536 possible input pairs -- nothing here can memorise a table, so
held-out accuracy is the only number worth printing.

The task is train/adder.py: two input columns written into channel 0, one output
column read back out of it, everything else steering. Three things the layout
forces on the loop.

**Only the read-out band is scored.** The answer costs less mass than the inputs
carry, and the surplus has to go somewhere; charging for it wherever it lands
would be asking for a second, invented task.

**A frame is scored on its absolute age.** The mass that spells the answer has
to cross twenty-two cells and MaCE moves it one cell per step, so nothing can be
right before step twenty-two, and with a sample pool the age of a state is not
its position in the backprop window.

**Accuracy, not loss, decides what is kept.** A row reads 1 when its disc is more
than half filled, and a world can halve its L2 while filling nothing past half.

Checkpoints land in train/runs/<name>/ every --ckpt iterations.
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
import adder as ad
from train import World, divergence_rate

HERE = os.path.dirname(os.path.abspath(__file__))
HORIZONS = [32, 64, 96, 128, 192, 256]


class Task:
    def __init__(self, nbits, C, dtype=torch.float32):
        self.geo = ad.Geometry(nbits)
        self.C, self.dtype = C, dtype
        self.O = torch.tensor(self.geo.O, dtype=dtype)
        self.mask = torch.tensor(self.geo.mask, dtype=dtype)
        self.norm = self.geo.norm

    def build(self, ab):
        n = len(ab)
        g = self.geo
        seeds = torch.zeros(n, self.C, g.H, g.W, dtype=self.dtype)
        targets = torch.zeros(n, g.H, g.W, dtype=self.dtype)
        bits = torch.zeros(n, g.nbits, dtype=torch.long)
        for k, (a, b) in enumerate(ab):
            a_bits, b_bits, s_bits = ad.problem(a, b, g)
            seeds[k] = torch.tensor(ad.seed_field(a_bits, b_bits, self.C, g),
                                    dtype=self.dtype)
            targets[k] = torch.tensor(ad.target_field(s_bits, g), dtype=self.dtype)
            bits[k] = torch.tensor(s_bits)
        return seeds, targets, bits

    def decode(self, rho):
        c = torch.einsum("shw,bhw->bs", self.O, rho[:, ad.CH_DATA]) / self.norm
        return (c > 0.5).long()

    def loss(self, rho, target):
        """L2 inside the read-out band only, per sample."""
        d = (rho[:, ad.CH_DATA] - target) ** 2 * self.mask
        return d.sum(dim=(1, 2)) / self.mask.sum()


def accuracy(model, task, ab, horizons, chunk=32):
    bit = [0] * len(horizons)
    exact = [0] * len(horizons)
    seen = 0
    with torch.no_grad():
        for lo in range(0, len(ab), chunk):
            seeds, _, bits = task.build(ab[lo:lo + chunk])
            rho, done = seeds.clone(), 0
            seen += bits.numel()
            for j, h in enumerate(horizons):
                rho = model.rollout(rho, h - done)
                done = h
                ok = task.decode(rho) == bits
                bit[j] += int(ok.sum())
                exact[j] += int(ok.all(dim=1).sum())
    return ([v / max(seen, 1) for v in bit],
            [v / max(len(ab), 1) for v in exact])


def save_progress(path, model, task, ab, horizons):
    seeds, targets, _ = task.build(ab)
    with torch.no_grad():
        rho, done, frames = seeds.clone(), 0, []
        for h in horizons:
            rho = model.rollout(rho, h - done)
            done = h
            frames.append(rho[:, ad.CH_DATA].clone())
    rows = []
    for k in range(len(ab)):
        # red is channel 0, green is the answer it is being asked for
        tiles = [np.stack([seeds[k, ad.CH_DATA].numpy(), targets[k].numpy(),
                           np.zeros_like(targets[k].numpy())], -1)]
        for f in frames:
            tiles.append(np.stack([f[k].numpy(), targets[k].numpy(),
                                   np.zeros_like(targets[k].numpy())], -1))
        rows.append(np.concatenate([np.clip(t, 0, 1) for t in tiles], axis=1))
    strip = np.concatenate(rows, axis=0)
    img = Image.fromarray((strip * 255).astype(np.uint8))
    img.resize((img.width * 3, img.height * 3), Image.NEAREST).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="add8")
    ap.add_argument("--bits", type=int, default=8)
    ap.add_argument("--orders", default="0,0,0,1,1,2")
    ap.add_argument("--channels", type=int, default=6)
    ap.add_argument("--kr", type=int, default=7)
    ap.add_argument("--mat-init", default="zeros", choices=["random", "zeros"])
    ap.add_argument("--hidden", type=int, default=0)
    ap.add_argument("--iters", type=int, default=1000000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--settle", type=int, default=32,
                    help="absolute age before a frame is scored; the answer's "
                         "mass cannot have crossed the field before then")
    ap.add_argument("--loss-every", type=int, default=4)
    ap.add_argument("--pool", type=int, default=64)
    ap.add_argument("--max-age", type=int, default=128)
    ap.add_argument("--young-frac", type=float, default=0.5)
    ap.add_argument("--young-age", type=int, default=48)
    ap.add_argument("--train-pairs", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--lam-penalty", type=float, default=0.02)
    ap.add_argument("--ckpt", type=int, default=100)
    ap.add_argument("--eval-pairs", type=int, default=32)
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

    task = Task(a.bits, a.channels)
    rng = np.random.default_rng(a.seed)
    lim = 1 << a.bits
    seen = set()
    train_pairs = []
    while len(train_pairs) < min(a.train_pairs, lim * lim):
        p = (int(rng.integers(0, lim)), int(rng.integers(0, lim)))
        if p not in seen:
            seen.add(p)
            train_pairs.append(p)
    test_pairs = []
    while len(test_pairs) < a.eval_pairs:
        p = (int(rng.integers(0, lim)), int(rng.integers(0, lim)))
        if p not in seen:
            test_pairs.append(p)
    eval_tr = train_pairs[:a.eval_pairs]
    seeds_tr, targets_tr, _ = task.build(train_pairs)

    model = World(a.channels, orders, a.kr, seed=a.seed, hidden=a.hidden,
                  mat_init=a.mat_init, mip=0)
    with torch.no_grad():
        model.kern.logR.fill_(math.log(float(a.kr)))
    model.kern.logR.requires_grad_(False)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], a.lr)

    pick = torch.from_numpy(rng.integers(0, len(train_pairs), a.pool))
    pool, pool_t = seeds_tr[pick].clone(), targets_tr[pick].clone()
    ages = torch.zeros(a.pool, dtype=torch.long)
    n_young = max(1, min(a.pool - 1, int(a.pool * a.young_frac)))
    lifespan = torch.full((a.pool,), a.max_age, dtype=torch.long)
    lifespan[:n_young] = a.young_age
    start_it = 0

    def fresh(k):
        j = int(rng.integers(0, len(train_pairs)))
        pool[k], pool_t[k], ages[k] = seeds_tr[j], targets_tr[j], 0

    ck = os.path.join(run, "ckpt.pt")
    if a.resume and os.path.exists(ck):
        st = torch.load(ck, weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
        pool, pool_t, ages, start_it = st["pool"], st["pool_t"], st["ages"], st["iter"]
        print(f"resumed {a.name} at iteration {start_it}")

    logp = os.path.join(run, "log.csv")
    if not os.path.exists(logp):
        with open(logp, "w", newline="") as f:
            csv.writer(f).writerow(
                ["iter", "loss", "lam", "force", "beta"]
                + [f"bit{h}" for h in HORIZONS] + [f"ex{h}" for h in HORIZONS]
                + ["bit_test", "exact_test", "age", "secs"])

    print(f"run {a.name}: {a.bits}-bit adder, grid {task.geo.W} x {task.geo.H}, "
          f"{len(train_pairs)} training pairs of {lim * lim} possible")
    print(f"  the answer's mass crosses {ad.COL_O - ad.COL_A} cells at one cell "
          f"per step, so settle is {a.settle}")
    print(f"  window {a.window}  batch {a.batch}  pool {a.pool}  "
          f"stencil {a.kr}  orders {orders}")
    print(f"  {sum(p.numel() for p in model.parameters() if p.requires_grad)} "
          f"trainable numbers; a coin flip scores 0.50 per bit and "
          f"{0.5 ** a.bits:.4f} exact")

    t0, best_score = time.time(), -1.0
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
            bit_tr, ex_tr = accuracy(model, task, eval_tr, HORIZONS)
            bit_te, ex_te = accuracy(model, task, test_pairs, HORIZONS)
            score = ex_te[-1] + 0.01 * bit_te[-1]
            with open(logp, "a", newline="") as fh:
                csv.writer(fh).writerow(
                    [it, f"{loss.item():.6f}", f"{lam:.3f}", f"{f:.2f}", f"{b:.3f}"]
                    + [f"{v:.4f}" for v in bit_tr] + [f"{v:.4f}" for v in ex_tr]
                    + [f"{bit_te[-1]:.4f}", f"{ex_te[-1]:.4f}",
                       int(ages.float().mean()), f"{secs:.0f}"])
            cfg = model.to_config(a.channels, task.geo.W)
            cfg["_task"] = {"bits": a.bits, "pitch": ad.PITCH_Y,
                            "cols": [ad.COL_A, ad.COL_B, ad.COL_O]}
            json.dump(cfg, open(os.path.join(run, "preset.json"), "w"))
            if score > best_score:
                best_score = score
                json.dump(dict(cfg, _bestAt=it, _bit=bit_te, _exact=ex_te),
                          open(bp, "w"))
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "pool": pool, "pool_t": pool_t, "ages": ages, "iter": it}, ck)
            save_progress(os.path.join(run, "progress.png"), model, task,
                          test_pairs[:4], HORIZONS)
            print(f"  it {it:6d}  loss {loss.item():.5f}  force {f:6.1f}  "
                  f"beta {b:.2f}  lam {lam:+.2f}  age~{int(ages.float().mean())}  "
                  f"bit {'/'.join(f'{v:.2f}' for v in bit_tr)}  "
                  f"exact {'/'.join(f'{v:.2f}' for v in ex_tr)}  "
                  f"held-out {bit_te[-1]:.2f}/{ex_te[-1]:.2f}  "
                  f"{secs / max(it - start_it + 1, 1):.2f}s/it", flush=True)

        if a.minutes and time.time() - t0 > a.minutes * 60:
            print(f"stopping after {a.minutes} minutes at iteration {it}")
            break


if __name__ == "__main__":
    main()
