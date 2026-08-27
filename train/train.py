"""
Train field-life to sculpt the lizard.

    python3 train/train.py --name polar
    python3 train/train.py --name radial --orders 0,0,0,0     # the m=0 ablation

What is being trained (rung 0 of docs/nca-experiment.md): the interaction
matrix, one polar kernel per channel, and the three globals. About 190 numbers.
The law is untouched -- MaCE still conserves mass exactly, so the seed arrives
holding the picture's own per-channel mass and the only thing to learn is where
to put it.

Two things shape the loop, both measured in train/parity.py:

  * mass moves ONE cell per step, and the lizard's furthest cell is 26 cells
    from the seed, so a rollout has to be long;
  * the step amplifies differences by up to e^0.4 per step, so a long rollout
    cannot be backpropagated through.

Growing NCA's sample pool resolves exactly this tension: backpropagate through
a short window, but start that window from a state the model itself produced,
and write the result back. Rollouts get arbitrarily long; the gradient never
sees more than `--window` steps.

Checkpoints land in train/runs/<name>/ every --ckpt iterations:

    preset.json    the trained world, loadable in index.html as it stands
    log.csv        loss per iteration
    progress.png   target vs the current rollout
    ckpt.pt        optimiser state, for --resume (not committed)
"""

import argparse
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
import fieldlife as fl
import target as tgt

HERE = os.path.dirname(os.path.abspath(__file__))


# ------------------------------------------------------------------ the model

class World(torch.nn.Module):
    """Everything trainable: the kernels, the matrix, and the three globals.

    force/repel/beta are held as logs so they stay positive, and they start in
    the regime parity.py found does NOT amplify. The optimiser may heat them up
    if that helps; --lam-penalty is what stops it from running off into chaos
    where the gradient it is following stops meaning anything.
    """

    def __init__(self, C, orders, KR, seed=0):
        super().__init__()
        self.kern = fl.PolarKernels(C, orders=orders, KR=KR, seed=seed)
        g = torch.Generator().manual_seed(seed + 1)
        self.mat = torch.nn.Parameter(torch.randn(C, C, generator=g) * 0.5)
        self.log_force = torch.nn.Parameter(torch.tensor(math.log(12.0)))
        self.log_repel = torch.nn.Parameter(torch.tensor(math.log(1.0)))
        self.log_beta = torch.nn.Parameter(torch.tensor(math.log(0.35)))

    def globals(self):
        return (self.log_force.exp().clamp(0.1, 200.0),
                self.log_repel.exp().clamp(0.0, 50.0),
                self.log_beta.exp().clamp(0.01, 10.0))

    def rollout(self, rho, steps, kern=None):
        kern = self.kern() if kern is None else kern
        f, r, b = self.globals()
        for _ in range(steps):
            rho = fl.step(rho, kern, self.mat, f, r, b)
        return rho

    def scalars(self):
        return tuple(float(v.detach()) for v in self.globals())

    def to_config(self, C, N):
        """A preset index.html can load, with no changes to it."""
        f, r, b = self.scalars()
        R = self.kern.radii().detach().tolist()
        return {"seed": 1, "mat": self.mat.detach().flatten().tolist(),
                "kernels": self.kern.to_config(),
                "N": N, "C": C, "density": 0.12,
                "radMin": min(R), "radMax": max(R),
                "force": f, "repel": r, "beta": b,
                "noise": 0.0, "steps": 1, "expo": 2.2,
                "seedMode": "disc", "palette": "Spectrum", "blend": 2,
                "kterms": 2, "kwidth": 0.7, "ksym": "radial",
                "cfreq": 1.1, "cdepth": 2}


# ---------------------------------------------------------------- diagnostics

def divergence_rate(model, rho, steps=12):
    """Measured e^lambda per step: how fast two near-identical fields separate.

    This is the number that says whether the gradient still means anything.
    Perturb one cell by a hair, run both, fit the growth of the gap.
    """
    with torch.no_grad():
        kern = model.kern()
        a, b = rho[:1].clone(), rho[:1].clone()
        b[0, 0, b.shape[-2] // 2, b.shape[-1] // 2] *= 1.0 + 1e-6
        gaps, base = [], None
        for i in range(1, steps + 1):
            a, b = model.rollout(a, 1, kern), model.rollout(b, 1, kern)
            g = (a - b).abs().max().item()
            if g <= 0:
                continue
            if base is None:
                base, i0 = g, i
            gaps.append((i, math.log(g)))
        if len(gaps) < 2:
            return 0.0
        mx = sum(p[0] for p in gaps) / len(gaps)
        my = sum(p[1] for p in gaps) / len(gaps)
        den = sum((p[0] - mx) ** 2 for p in gaps)
        return (sum((p[0] - mx) * (p[1] - my) for p in gaps) / den) if den else 0.0


def save_progress(path, target, model, seed, steps_list):
    """Target on the left, then the rollout at each checkpoint length."""
    with torch.no_grad():
        tiles = [target.cpu().numpy()]
        rho, done = seed[None].clone(), 0
        for s in steps_list:
            rho = model.rollout(rho, s - done)
            done = s
            tiles.append(rho[0, :3].cpu().numpy())
    strip = np.concatenate([np.clip(t.transpose(1, 2, 0), 0, 1) for t in tiles], axis=1)
    img = Image.fromarray((strip * 255).astype(np.uint8))
    img = img.resize((img.width * 5, img.height * 5), Image.NEAREST)
    img.save(path)


# --------------------------------------------------------------------- train

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="polar")
    ap.add_argument("--orders", default="0,0,0,1,1,2",
                    help="angular order per kernel lobe; all zeros = radial only")
    ap.add_argument("--channels", type=int, default=6)
    ap.add_argument("--grid", type=int, default=64)
    ap.add_argument("--kr", type=int, default=13)
    ap.add_argument("--iters", type=int, default=100000)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--window", type=int, default=16, help="BPTT length")
    ap.add_argument("--pool", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--lam-penalty", type=float, default=0.02,
                    help="weight on the measured divergence rate")
    ap.add_argument("--ckpt", type=int, default=50)
    ap.add_argument("--minutes", type=float, default=0, help="stop after this long")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    orders = tuple(int(x) for x in a.orders.split(","))
    run = os.path.join(HERE, "runs", a.name)
    os.makedirs(run, exist_ok=True)

    # ---- the task
    target_np = tgt.render_emoji(span=tgt.SPAN, grid=a.grid)
    masses = tgt.seed_masses(target_np, a.channels,
                             rng=np.random.default_rng(a.seed))
    seed_np = tgt.seed_field(masses, grid=a.grid)
    target = torch.tensor(target_np, dtype=torch.float32)
    seed = torch.tensor(seed_np, dtype=torch.float32)

    model = World(a.channels, orders, a.kr, seed=a.seed)
    opt = torch.optim.Adam(model.parameters(), a.lr)
    pool = seed[None].repeat(a.pool, 1, 1, 1).clone()
    start_it = 0

    ck = os.path.join(run, "ckpt.pt")
    if a.resume and os.path.exists(ck):
        st = torch.load(ck, weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
        pool = st["pool"]; start_it = st["iter"]
        print(f"resumed {a.name} at iteration {start_it}")

    logp = os.path.join(run, "log.csv")
    if not os.path.exists(logp):
        with open(logp, "w", newline="") as f:
            csv.writer(f).writerow(["iter", "loss", "best", "lam", "force", "beta", "secs"])

    print(f"run {a.name}: orders {orders}  C {a.channels}  grid {a.grid}  "
          f"window {a.window}  batch {a.batch}  pool {a.pool}")
    print(f"  {sum(p.numel() for p in model.parameters())} trainable numbers")

    t0 = time.time()
    best = float("inf")
    for it in range(start_it, a.iters):
        idx = torch.randint(0, a.pool, (a.batch,))
        batch = pool[idx].clone()
        # the worst state in the batch goes back to the seed, so the model has
        # to keep working from scratch and cannot drift into a pool of its own
        # comfortable states -- Growing NCA's trick, and the reason the pattern
        # is asked to persist rather than merely to be reached once
        with torch.no_grad():
            worst = ((batch[:, :3] - target) ** 2).mean(dim=(1, 2, 3)).argmax()
        batch[worst] = seed

        out = model.rollout(batch, a.window)
        loss = ((out[:, :3] - target) ** 2).mean()

        lam = 0.0
        if a.lam_penalty > 0 and it % 10 == 0:
            lam = divergence_rate(model, batch.detach())
        total = loss + a.lam_penalty * max(lam, 0.0) * loss.detach()

        opt.zero_grad()
        total.backward()
        # per-parameter normalisation, as Growing NCA does: the scales here
        # differ by orders of magnitude (a log-beta against 36 matrix entries)
        # and a shared learning rate cannot serve both.
        for p in model.parameters():
            if p.grad is not None:
                p.grad /= p.grad.norm() + 1e-8
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        with torch.no_grad():
            good = torch.isfinite(out).all(dim=(1, 2, 3))
            pool[idx[good]] = out[good].detach()
            pool[idx[~good]] = seed          # a blown-up state poisons the pool

        best = min(best, loss.item())
        if it % a.ckpt == 0 or it == a.iters - 1:
            f, r, b = model.scalars()
            secs = time.time() - t0
            with open(logp, "a", newline="") as fh:
                csv.writer(fh).writerow([it, f"{loss.item():.6f}", f"{best:.6f}",
                                         f"{lam:.3f}", f"{f:.2f}", f"{b:.3f}",
                                         f"{secs:.0f}"])
            json.dump(model.to_config(a.channels, a.grid),
                      open(os.path.join(run, "preset.json"), "w"))
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "pool": pool, "iter": it}, ck)
            save_progress(os.path.join(run, "progress.png"), target, model, seed,
                          [16, 32, 64, 128])
            print(f"  it {it:6d}  loss {loss.item():.5f}  best {best:.5f}  "
                  f"lam {lam:+.2f}  force {f:6.1f}  beta {b:.2f}  "
                  f"{secs / max(it - start_it + 1, 1):.2f}s/it", flush=True)

        if a.minutes and time.time() - t0 > a.minutes * 60:
            print(f"stopping after {a.minutes} minutes at iteration {it}")
            break

    print(f"done. best loss {best:.6f}   {run}")


if __name__ == "__main__":
    main()
