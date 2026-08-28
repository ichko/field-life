"""
Train field-life to add.

    python3 train/train_adder.py --name add4 --slots 5

The world is rung 0 of docs/nca-experiment.md -- the interaction matrix, one
polar kernel per channel, and force/repel/beta, about two hundred numbers -- and
it is the SAME World the lizard used. Only the task is new. What changes with it
is what a "sample" is: the lizard had one target and a pool of aged states, and
this has a different target for every input pair, so a pool slot has to carry
its problem along with its state.

Three things follow from the task that did not apply to the lizard.

**A frame is scored only once the answer could exist.** A carry has to ripple
along the slots, and the field cannot know slot 4's answer before slot 3 has
decided. Charging for the arrangement at step 2 asks the impossible, so a frame
is scored on its ABSOLUTE age -- how many steps that state has run since its
seed -- not on where it sits inside the backprop window. With a pool the two are
different numbers, and the lizard's `--loss-from` measured the wrong one.

**Loss is not the metric.** L2 against the target arrangement is what trains,
but what is being asked is whether the field gets the arithmetic RIGHT: each
slot is decoded by which rail holds more of its blob, and the numbers that
matter are per-bit accuracy and the fraction of input pairs answered exactly.
A world can halve its L2 and answer nothing correctly.

**The interesting test is a wider adder.** The rule is the same in every cell,
so a world fitted on five slots can be dropped on nine and asked to be an 8-bit
adder it has never seen -- see train/adder.py for why the slot axis is cyclic
and why that is sound. Every checkpoint reports it, because it is the result:
generalising in width is something a fitted table cannot do.

Checkpoints land in train/runs/<name>/ every --ckpt iterations:

    preset.json  / preset-best.json   the trained world
    log.csv                           loss, accuracy, and the wider adders
    progress.png                      seed, target, and the rollout at each horizon
    ckpt.pt                           optimiser state, for --resume
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
import fieldlife as fl
from train import World

HERE = os.path.dirname(os.path.abspath(__file__))

# Rollout lengths every checkpoint decodes at. They run well past the backprop
# window: reaching the right arrangement and standing in it are different
# things, and only the long ones tell them apart.
HORIZONS = [32, 64, 96, 128, 192, 256]


# ------------------------------------------------------------------ the task

class Task:
    """Problems at one width, as tensors: seeds, targets, and the answers."""

    def __init__(self, slots, C, op="add", dtype=torch.float32):
        self.geo = ad.Geometry(slots)
        self.C, self.dtype, self.op = C, dtype, op
        self.T1 = torch.tensor(self.geo.T1, dtype=dtype)
        self.T0 = torch.tensor(self.geo.T0, dtype=dtype)
        self.Tm = torch.tensor(self.geo.mid(), dtype=dtype)

    def build(self, ab):
        """(seeds, targets, answer bits) for a list of (a, b) pairs."""
        n = len(ab)
        seeds = torch.zeros(n, self.C, self.geo.H, self.geo.W, dtype=self.dtype)
        targets = torch.zeros(n, self.geo.H, self.geo.W, dtype=self.dtype)
        bits = torch.zeros(n, self.geo.nslots, dtype=torch.long)
        for k, (a, b) in enumerate(ab):
            a_bits, b_bits, s_bits, _ = ad.problem(a, b, self.geo, self.op)
            seeds[k] = torch.tensor(
                ad.seed_field(a_bits, b_bits, self.C, self.geo), dtype=self.dtype)
            targets[k] = torch.tensor(
                ad.target_field(s_bits, self.geo), dtype=self.dtype)
            bits[k] = torch.tensor(s_bits)
        return seeds, targets, bits

    def decode(self, rho):
        """Bits read off channel 2 of a batch of fields: (B, nslots)."""
        f = rho[:, ad.CH_S]
        s1 = torch.einsum("shw,bhw->bs", self.T1, f)
        s0 = torch.einsum("shw,bhw->bs", self.T0, f)
        return (s1 > s0).long()


def accuracy(model, task, ab, horizons, chunk=64):
    """Per-bit and exact-answer accuracy at each horizon, from a fresh seed."""
    bit_hits = [0] * len(horizons)
    exact_hits = [0] * len(horizons)
    total_bits = 0
    with torch.no_grad():
        for lo in range(0, len(ab), chunk):
            seeds, _, bits = task.build(ab[lo:lo + chunk])
            rho, done = seeds.clone(), 0
            total_bits += bits.numel()
            for j, h in enumerate(horizons):
                rho = model.rollout(rho, h - done)
                done = h
                got = task.decode(rho)
                ok = got == bits
                bit_hits[j] += int(ok.sum())
                exact_hits[j] += int(ok.all(dim=1).sum())
    n = len(ab)
    return ([h / max(total_bits, 1) for h in bit_hits],
            [h / max(n, 1) for h in exact_hits])


def save_progress(path, model, task, ab, horizons):
    """One row per problem: seed, target, then the rollout at each horizon."""
    seeds, targets, _ = task.build(ab)
    rows = []
    with torch.no_grad():
        rho, done, frames = seeds.clone(), 0, []
        for h in horizons:
            rho = model.rollout(rho, h - done)
            done = h
            frames.append(rho[:, :3].clone())
    for k in range(len(ab)):
        tgt = torch.stack([targets[k]] * 3).numpy()
        tiles = [seeds[k, :3].numpy(), tgt] + [f[k].numpy() for f in frames]
        rows.append(np.concatenate(
            [np.clip(t.transpose(1, 2, 0), 0, 1) for t in tiles], axis=1))
    strip = np.concatenate(rows, axis=0)
    img = Image.fromarray((strip * 255).astype(np.uint8))
    img = img.resize((img.width * 4, img.height * 4), Image.NEAREST)
    img.save(path)


# --------------------------------------------------------------------- train

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="add4")
    ap.add_argument("--op", default="add", choices=list(ad.OPS),
                    help="the ladder of train/adder.py: copy, and, or, xor, add")
    ap.add_argument("--slots", type=int, default=5,
                    help="slots to TRAIN on; the adder is one bit narrower, "
                         "since the top slot is pinned to zero and terminates "
                         "the carry ripple")
    ap.add_argument("--wider", default="9,17",
                    help="slot counts to also evaluate at, never trained on")
    ap.add_argument("--wider-pairs", type=int, default=64,
                    help="input pairs sampled at each wider width")
    ap.add_argument("--wide-every", type=int, default=5,
                    help="checkpoints between wide-width evaluations. A rollout "
                         "at 17 slots costs about what forty training iterations "
                         "do, so measuring it every checkpoint spends more of the "
                         "budget watching than training.")
    ap.add_argument("--train-frac", type=float, default=0.75,
                    help="fraction of the width's input pairs used for training; "
                         "the rest is held out")
    ap.add_argument("--orders", default="0,0,0,1,1,2")
    ap.add_argument("--channels", type=int, default=6)
    ap.add_argument("--kr", type=int, default=7,
                    help="stencil half-width. Must stay well inside the grid: "
                         "the field is a torus and a kernel wider than it makes "
                         "a cell see itself from the far side.")
    ap.add_argument("--mat-init", default="zeros", choices=["random", "zeros"])
    ap.add_argument("--free-reach", action="store_true",
                    help="let each channel learn its own reach. Off by default, "
                         "and the reason is portability: index.html bakes a "
                         "channel whose reach is short at a COARSER grid rather "
                         "than as a radial scale on the shared one, so a learned "
                         "R exports to a different kernel than the one trained "
                         "(docs/nca-experiment.md 7.3). Pinning every reach to "
                         "the stencil makes the two bakers agree exactly, and "
                         "costs little: mu and w still decide where inside that "
                         "reach a lobe's weight sits.")
    ap.add_argument("--hidden", type=int, default=0, help="rung 1 width; 0 keeps the matrix")
    ap.add_argument("--iters", type=int, default=100000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--window", type=int, default=16, help="BPTT length")
    ap.add_argument("--settle", type=int, default=24,
                    help="steps of absolute age before a frame is scored. Below "
                         "this the carry has not had time to ripple and the "
                         "answer cannot exist yet.")
    ap.add_argument("--loss-every", type=int, default=4,
                    help="score every N steps inside the window; 0 scores the end only")
    ap.add_argument("--pool", type=int, default=64)
    ap.add_argument("--max-age", type=int, default=512,
                    help="steps a pooled state may live before it is retired and "
                         "replaced by a fresh problem")
    ap.add_argument("--young-frac", type=float, default=0.5)
    ap.add_argument("--young-age", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--init-from", default=None,
                    help="start from another run's world; kernels are in absolute "
                         "cells, so a world fitted at one width is a legitimate "
                         "start at another")
    ap.add_argument("--lam-penalty", type=float, default=0.02)
    ap.add_argument("--ckpt", type=int, default=50)
    ap.add_argument("--eval-pairs", type=int, default=128)
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

    # ---- problems
    task = Task(a.slots, a.channels, a.op)
    nbits = task.geo.nbits
    rng = np.random.default_rng(a.seed)
    all_pairs = [(x, y) for x in range(1 << nbits) for y in range(1 << nbits)]
    rng.shuffle(all_pairs)
    ntr = max(1, int(len(all_pairs) * a.train_frac))
    train_pairs, test_pairs = all_pairs[:ntr], all_pairs[ntr:]
    seeds_tr, targets_tr, _ = task.build(train_pairs)

    eval_tr = train_pairs[:a.eval_pairs]
    eval_te = test_pairs[:a.eval_pairs] or train_pairs[:a.eval_pairs]
    wider = [int(x) for x in a.wider.split(",") if x] if a.wider else []
    wide_tasks = []
    for s in wider:
        wt = Task(s, a.channels, a.op)
        wp = [(int(rng.integers(0, 1 << wt.geo.nbits)),
               int(rng.integers(0, 1 << wt.geo.nbits)))
              for _ in range(a.wider_pairs)]
        wide_tasks.append((wt, wp))

    model = World(a.channels, orders, a.kr, seed=a.seed, hidden=a.hidden,
                  mat_init=a.mat_init, mip=0)
    if a.init_from:
        src = os.path.join(HERE, "runs", a.init_from, "ckpt.pt")
        if not os.path.exists(src):
            sys.exit(f"no checkpoint at {src}")
        model.load_state_dict(torch.load(src, weights_only=False)["model"])
        print(f"started from {a.init_from}'s world")
    if not a.free_reach:
        with torch.no_grad():
            model.kern.logR.fill_(math.log(float(a.kr)))
        model.kern.logR.requires_grad_(False)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], a.lr)

    # ---- the pool: a state, the problem it belongs to, and its age
    pick = torch.from_numpy(rng.integers(0, len(train_pairs), a.pool))
    pool = seeds_tr[pick].clone()
    pool_t = targets_tr[pick].clone()
    ages = torch.zeros(a.pool, dtype=torch.long)
    n_young = max(1, min(a.pool - 1, int(a.pool * a.young_frac)))
    lifespan = torch.full((a.pool,), a.max_age, dtype=torch.long)
    lifespan[:n_young] = a.young_age
    start_it = 0

    def fresh(k):
        """Send pool slot k back to the seed of a newly drawn problem."""
        j = int(rng.integers(0, len(train_pairs)))
        pool[k], pool_t[k], ages[k] = seeds_tr[j], targets_tr[j], 0

    ck = os.path.join(run, "ckpt.pt")
    if a.resume and os.path.exists(ck):
        st = torch.load(ck, weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
        pool, pool_t = st["pool"], st["pool_t"]
        ages, start_it = st["ages"], st["iter"]
        print(f"resumed {a.name} at iteration {start_it}")

    logp = os.path.join(run, "log.csv")
    if not os.path.exists(logp):
        cols = (["iter", "loss", "best", "lam", "force", "beta"]
                + [f"bit{h}" for h in HORIZONS] + [f"ex{h}" for h in HORIZONS]
                + ["bit_test", "exact_test"]
                + [f"bit_s{s}" for s in wider] + [f"exact_s{s}" for s in wider]
                + ["age", "secs"])
        with open(logp, "w", newline="") as f:
            csv.writer(f).writerow(cols)

    print(f"run {a.name}: {nbits}-bit `{a.op}` on {a.slots} slots, grid "
          f"{task.geo.W} x {task.geo.H}, {len(train_pairs)} train pairs / "
          f"{len(test_pairs)} held out")
    print(f"  stencil half-width {a.kr} (reach {a.kr} cells, "
          f"{a.kr / ad.PITCH:.1f} slots), orders {orders}, C {a.channels}")
    print(f"  window {a.window}  batch {a.batch}  pool {a.pool}  settle {a.settle}")
    print(f"  also evaluated, never trained, at slots {wider}")
    print(f"  reach {'learned per channel' if a.free_reach else f'pinned to {a.kr} cells, so the export bakes exactly'}")
    print(f"  {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable numbers "
          f"against {len(all_pairs)} input pairs at this width")

    last_wide = [([float("nan")], [float("nan")]) for _ in wide_tasks]
    t0, best, best_score = time.time(), float("inf"), -1.0
    bp = os.path.join(run, "preset-best.json")
    if a.resume and os.path.exists(bp):
        try:
            best_score = json.load(open(bp)).get("_score", -1.0)
            print(f"carrying the best score {best_score:.4f}")
        except (json.JSONDecodeError, TypeError):
            pass

    for it in range(start_it, a.iters):
        half = max(1, a.batch // 2)
        idx = torch.cat([torch.randint(0, n_young, (half,)),
                         torch.randint(n_young, a.pool, (a.batch - half,))])
        # Retire on age alone. The lizard's trainer also reseeded the batch's
        # oldest every few iterations; here every pool slot already carries its
        # own lifespan and a retirement draws a NEW problem, so the pool turns
        # over on its own and the input distribution is refreshed with it.
        for k in idx.tolist():
            if ages[k] >= lifespan[k]:
                fresh(k)

        batch, btgt, bage = pool[idx].clone(), pool_t[idx], ages[idx]
        kern = model.kern()
        out, scored = batch, []
        for t in range(a.window):
            out = model.rollout(out, 1, kern=kern)
            step_no = t + 1
            last = step_no == a.window
            if not (last or not a.loss_every or step_no % a.loss_every == 0):
                continue
            # absolute age, not position in the window: with a pool the two are
            # different, and it is the absolute one that says whether the carry
            # has had time to arrive
            ready = (bage + step_no) >= a.settle
            if not ready.any():
                continue
            err = ((out[:, ad.CH_S] - btgt) ** 2).mean(dim=(1, 2))
            scored.append((err * ready).sum() / ready.sum())
        loss = (torch.stack(scored).mean() if scored
                else ((out[:, ad.CH_S] - btgt) ** 2).mean())

        lam = 0.0
        if a.lam_penalty > 0 and it % 10 == 0:
            lam = float(np.nan_to_num(
                fl_divergence(model, batch.detach()), nan=0.0))
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
            keep = idx[good]
            pool[keep] = out[good].detach()
            ages[keep] += a.window
            for k in idx[~good].tolist():
                fresh(k)                      # a blown-up state poisons the pool

        best = min(best, loss.item())
        if it % a.ckpt == 0 or it == a.iters - 1:
            f, r, b = model.scalars()
            secs = time.time() - t0
            bit_tr, ex_tr = accuracy(model, task, eval_tr, HORIZONS)
            bit_te, ex_te = accuracy(model, task, eval_te, HORIZONS)
            if (it // a.ckpt) % a.wide_every == 0 or it == a.iters - 1:
                wide = [accuracy(model, wt, wp, [HORIZONS[-1]])
                        for wt, wp in wide_tasks]
                last_wide[:] = wide
            else:
                wide = last_wide
            # what "best" means: the held-out exact-answer rate at the longest
            # horizon, tie-broken by per-bit accuracy. Not the loss -- the loss
            # is measured off a pool whose contents wander.
            score = ex_te[-1] + 0.01 * bit_te[-1]
            with open(logp, "a", newline="") as fh:
                csv.writer(fh).writerow(
                    [it, f"{loss.item():.6f}", f"{best:.6f}", f"{lam:.3f}",
                     f"{f:.2f}", f"{b:.3f}"]
                    + [f"{v:.4f}" for v in bit_tr] + [f"{v:.4f}" for v in ex_tr]
                    + [f"{bit_te[-1]:.4f}", f"{ex_te[-1]:.4f}"]
                    + [f"{w[0][0]:.4f}" for w in wide]
                    + [f"{w[1][0]:.4f}" for w in wide]
                    + [int(ages.float().mean()), f"{secs:.0f}"])
            cfg = model.to_config(a.channels, task.geo.W)
            cfg["_task"] = {"op": a.op, "slots": a.slots, "bits": nbits, "pitch": ad.PITCH,
                            "height": task.geo.H, "rail": ad.RAIL}
            if a.hidden:
                cfg["_note"] = "rung 1: affinity is a per-cell network, not this matrix."
            json.dump(cfg, open(os.path.join(run, "preset.json"), "w"))
            if score > best_score:
                best_score = score
                json.dump(dict(cfg, _bestAt=it, _score=score,
                               _bit=bit_te, _exact=ex_te),
                          open(bp, "w"))
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "pool": pool, "pool_t": pool_t, "ages": ages, "iter": it}, ck)
            save_progress(os.path.join(run, "progress.png"), model, task,
                          eval_te[:4], HORIZONS)
            wide_txt = "  ".join(
                f"s{s}:{w[0][0]:.2f}/{w[1][0]:.2f}" for s, w in zip(wider, wide))
            print(f"  it {it:6d}  loss {loss.item():.5f}  force {f:6.1f}  "
                  f"beta {b:.2f}  lam {lam:+.2f}  age~{int(ages.float().mean())}  "
                  f"bit {'/'.join(f'{v:.2f}' for v in bit_tr)}  "
                  f"exact {'/'.join(f'{v:.2f}' for v in ex_tr)}  "
                  f"held-out {bit_te[-1]:.2f}/{ex_te[-1]:.2f}  {wide_txt}  "
                  f"{secs / max(it - start_it + 1, 1):.2f}s/it"
                  f"{'  *best*' if score >= best_score else ''}", flush=True)

        if a.minutes and time.time() - t0 > a.minutes * 60:
            print(f"stopping after {a.minutes} minutes at iteration {it}")
            break

    print(f"done. best loss {best:.6f}   best held-out score {best_score:.4f}   {run}")


def fl_divergence(model, rho, steps=8):
    """train.py's divergence_rate, on a batch whose first element is enough."""
    from train import divergence_rate
    return divergence_rate(model, rho, steps)


if __name__ == "__main__":
    main()
