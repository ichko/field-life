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
import fieldlife as fl
import target as tgt

HERE = os.path.dirname(os.path.abspath(__file__))
# Rollout lengths every checkpoint reports. They run well past anything that
# is backpropagated through, because reaching the shape and holding it are
# different things and only the long ones can tell them apart: a world can
# score 0.0019 at step 64 and 0.0077 at step 512.
HORIZONS = [16, 32, 64, 128, 256, 512]
# What "best" means. Weighted hard toward the far end -- a world that decays is
# not a better world for having been briefly sharper.
HZ_WEIGHTS = [0.25, 0.5, 1, 2, 3, 4]


# ------------------------------------------------------------------ the model

class Mix(torch.nn.Module):
    """Rung 1: a hidden layer where the interaction matrix was.

    The shipped law computes the affinity as `M . U` -- one matrix multiply per
    cell, linear in the convolved fields, and that linearity is the capacity
    ceiling, not the kernels. This replaces it with `W2 . tanh(W1 . U + b1)`,
    applied per cell, which is exactly the shape of an NCA's update MLP.

    It is 1x1 convolution, so it costs almost nothing next to the kernels, and
    it leaves MaCE alone -- mass is still conserved exactly. What it does cost
    is portability: index.html's FS_AFF computes a matrix multiply, so a rung 1
    world needs the shader change of docs/nca-experiment.md before it will load.
    """

    def __init__(self, C, hidden, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed + 2)
        self.w1 = torch.nn.Parameter(torch.randn(hidden, C, generator=g) / C ** 0.5)
        self.b1 = torch.nn.Parameter(torch.zeros(hidden))
        self.w2 = torch.nn.Parameter(torch.randn(C, hidden, generator=g) / hidden ** 0.5)

    def forward(self, U):
        ax1 = "hc,chw->hhw" if U.ndim == 3 else "hc,bchw->bhhw"
        ax2 = "ch,hhw->chw" if U.ndim == 3 else "ch,bhhw->bchw"
        # einsum cannot reuse a subscript for two different sizes; do it as a
        # 1x1 convolution instead, which is what a per-cell dense layer is.
        v = torch.nn.functional.conv2d(U if U.ndim == 4 else U[None],
                                       self.w1[:, :, None, None], self.b1)
        v = torch.nn.functional.conv2d(torch.tanh(v), self.w2[:, :, None, None])
        return v if U.ndim == 4 else v[0]


class World(torch.nn.Module):
    """Everything trainable: the kernels, the matrix, and the three globals.

    force/repel/beta are held as logs so they stay positive, and they start in
    the regime parity.py found does NOT amplify. The optimiser may heat them up
    if that helps; --lam-penalty is what stops it from running off into chaos
    where the gradient it is following stops meaning anything.
    """

    def __init__(self, C, orders, KR, seed=0, hidden=0):
        super().__init__()
        self.kern = fl.PolarKernels(C, orders=orders, KR=KR, seed=seed)
        g = torch.Generator().manual_seed(seed + 1)
        self.hidden = hidden
        self.mix = Mix(C, hidden, seed) if hidden else None
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
        mix = self.mix if self.hidden else self.mat
        for _ in range(steps):
            rho = fl.step(rho, kern, mix, f, r, b)
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


def horizon_losses(model, target, seed, steps_list):
    """Loss from the seed at each rollout length.

    The training window is short by necessity, so the number that matters is
    not the loss at the window but how it behaves well past it: a model that
    reaches the shape and then smears has a rising curve here, and one that
    holds has a flat one. Without this the difference is only visible by eye.
    """
    with torch.no_grad():
        out, rho, done = [], seed[None].clone(), 0
        for s in steps_list:
            rho = model.rollout(rho, s - done)
            done = s
            out.append(((rho[0, :3] - target) ** 2).mean().item())
    return out


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
    ap.add_argument("--span", type=int, default=None,
                    help="cells the animal spans. Growing this WITHOUT growing "
                         "the kernels is the point: the reach sets a blob size, "
                         "so a target much larger than the reach has to be built "
                         "out of blobs of it. At span 40 the reach is a third of "
                         "the animal; at 200 it is a fourteenth.")
    ap.add_argument("--init-from", default=None,
                    help="start from another run's world. Kernels are in "
                         "absolute cells, so a world fitted at one target size "
                         "is a legitimate starting point at another -- only the "
                         "matrix has to adapt to the new scale.")
    ap.add_argument("--kr", type=int, default=13)
    ap.add_argument("--iters", type=int, default=100000)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--window", type=int, default=16, help="BPTT length")
    ap.add_argument("--loss-from", type=int, default=0,
                    help="first step inside the window that is scored; frames "
                         "before the target could physically be reached are "
                         "asking the impossible and only add a constant")
    ap.add_argument("--loss-every", type=int, default=4,
                    help="score the field every N steps inside the window, not "
                         "only at its end; 0 scores the end only")
    ap.add_argument("--pool", type=int, default=64)
    ap.add_argument("--max-age", type=int, default=1024,
                    help="steps a pooled state may live before it goes back to "
                         "the seed; the pool's oldest states are what teach the "
                         "shape to hold rather than merely to arrive")
    ap.add_argument("--reseed-every", type=int, default=4,
                    help="iterations between sending a pool state back to the "
                         "seed. This sets how old the pool gets, and the "
                         "relationship is arithmetic, not a matter of taste: "
                         "each iteration adds window*batch step-years to the "
                         "pool and reseeding removes the oldest at rate "
                         "1/reseed_every, so the mean age settles at about "
                         "window*batch*reseed_every/2. At 32x2x8 that is 256, "
                         "which is what it sat at while the far horizons "
                         "refused to move")
    ap.add_argument("--young-frac", type=float, default=0.5,
                    help="fraction of the pool kept young. One reseed rate cannot "
                         "serve both ends: rare reseeding taught the shape to "
                         "hold (h512 0.0054 -> 0.0032) and to grow worse (h16 "
                         "0.0032 -> 0.0070), because growing and holding were "
                         "competing for the same pool slots. Splitting the pool "
                         "puts one of each in every batch instead.")
    ap.add_argument("--young-age", type=int, default=96,
                    help="steps a state in the young half may live")
    ap.add_argument("--reseed-policy", default="oldest", choices=["oldest", "worst"],
                    help="which pooled state goes back to the seed. Growing NCA "
                         "retires the worst, to stop a pool filling with "
                         "blown-up states; mass conservation makes that "
                         "impossible here, and retiring the worst culls exactly "
                         "the aged, drifting states that persistence has to be "
                         "taught on")
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--hidden", type=int, default=0,
                    help="rung 1: hidden width of the per-cell network; 0 keeps the matrix")
    ap.add_argument("--lam-penalty", type=float, default=0.02,
                    help="weight on the measured divergence rate")
    ap.add_argument("--ckpt", type=int, default=50)
    ap.add_argument("--minutes", type=float, default=0, help="stop after this long")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=2,
                    help="torch intra-op threads; keep the total across concurrent "
                         "runs at or under the core count")
    a = ap.parse_args()

    # Torch's OpenMP pool spin-waits, so oversubscribing cores does not merely
    # share them -- a run that asks for more threads than are free burns its
    # quantum spinning and can crawl to a near halt while a LARGER run beside
    # it, holding fewer threads, keeps its pace. Default low and set it here
    # rather than leaving it to whatever the environment happens to carry.
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    orders = tuple(int(x) for x in a.orders.split(","))
    run = os.path.join(HERE, "runs", a.name)
    os.makedirs(run, exist_ok=True)

    # One writer per run directory. Two trainers sharing one is not a crash --
    # they interleave rows in the log and take turns overwriting the
    # checkpoint, and the run looks fine until the numbers stop making sense.
    # It happened once here, because a kill matched nothing: /proc/pid/cmdline
    # separates arguments with NULs, so grepping it for "name polar3" never
    # matches. A lock is cheaper than remembering that.
    lock = os.path.join(run, "RUNNING")
    if os.path.exists(lock):
        pid = open(lock).read().strip()
        if pid.isdigit() and os.path.exists(f"/proc/{pid}"):
            sys.exit(f"{a.name} is already being trained by pid {pid}. "
                     f"Stop it first, or use a different --name.")
    with open(lock, "w") as fh:
        fh.write(str(os.getpid()))
    atexit.register(lambda: os.path.exists(lock) and os.remove(lock))

    # ---- the task
    span = a.span or tgt.SPAN
    target_np = tgt.render_emoji(span=span, grid=a.grid)
    masses = tgt.seed_masses(target_np, a.channels,
                             rng=np.random.default_rng(a.seed))
    seed_np = tgt.seed_field(masses, grid=a.grid)
    target = torch.tensor(target_np, dtype=torch.float32)
    seed = torch.tensor(seed_np, dtype=torch.float32)

    model = World(a.channels, orders, a.kr, seed=a.seed, hidden=a.hidden)
    if a.init_from:
        src = os.path.join(HERE, "runs", a.init_from, "ckpt.pt")
        if not os.path.exists(src):
            sys.exit(f"no checkpoint at {src}")
        model.load_state_dict(torch.load(src, weights_only=False)["model"])
        print(f"started from {a.init_from}'s world (its pool is not carried: "
              f"a field grown at one size is not a state at another)")
    opt = torch.optim.Adam(model.parameters(), a.lr)
    pool = seed[None].repeat(a.pool, 1, 1, 1).clone()
    # How many steps each pooled state has lived. Persistence cannot be learned
    # from states that are always young: if nothing in the pool has run longer
    # than a couple of windows, nothing is ever asked to still be a lizard at
    # step 500. Age is tracked so the reseeding can be driven by it.
    ages = torch.zeros(a.pool, dtype=torch.long)
    # The pool is two bands, not one. The young half is retired quickly, so
    # there is always a field partway through being built; the old half runs to
    # --max-age, so there is always one that has been standing for hundreds of
    # steps. Every batch draws from both, and growing and holding stop being
    # the same slot.
    n_young = max(1, min(a.pool - 1, int(a.pool * a.young_frac)))
    lifespan = torch.full((a.pool,), a.max_age, dtype=torch.long)
    lifespan[:n_young] = a.young_age
    start_it = 0

    ck = os.path.join(run, "ckpt.pt")
    if a.resume and os.path.exists(ck):
        st = torch.load(ck, weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
        pool = st["pool"]; start_it = st["iter"]
        ages = st.get("ages", torch.zeros(a.pool, dtype=torch.long))
        print(f"resumed {a.name} at iteration {start_it}")

    logp = os.path.join(run, "log.csv")
    if not os.path.exists(logp):
        with open(logp, "w", newline="") as f:
            csv.writer(f).writerow(["iter", "loss", "best", "lam", "force", "beta"]
                                   + [f"h{h}" for h in HORIZONS] + ["age", "secs"])

    reach = tgt.budget(target_np, a.grid)
    print(f"run {a.name}: grid {a.grid}, animal {span} cells, kernels up to "
          f"{a.kr} -> reach is {a.kr/span:.2f} of the animal; "
          f"the far end is {reach} cells from the seed")
    print(f"run {a.name}: orders {orders}  C {a.channels}  grid {a.grid}  "
          f"window {a.window}  batch {a.batch}  pool {a.pool} "
          f"({n_young} young to {a.young_age}, {a.pool - n_young} old to {a.max_age})")
    print(f"  {sum(p.numel() for p in model.parameters())} trainable numbers")

    t0 = time.time()
    best = float("inf")
    # The training loss is measured from pooled states and swings with whatever
    # the pool happens to hold, so the newest checkpoint is not the best one --
    # h128 was 0.0022 at iteration 7700 and 0.0097 at 7800. What to export is
    # decided by the horizon curve from a fresh seed instead, and the world
    # that scored it is kept beside the running one.
    best_h = float("inf")
    # ...carried across restarts. Without this every resumed run crowns its own
    # local best and overwrites a better world with a worse one: iteration 9500
    # replaced 8520 while losing on five horizons of six, purely because the
    # process had started again in between.
    bp = os.path.join(run, "preset-best.json")
    if a.resume and os.path.exists(bp):
        try:
            prev = json.load(open(bp))
            if "_horizons" in prev:
                best_h = (sum(h * w for h, w in zip(prev["_horizons"], HZ_WEIGHTS))
                          / sum(HZ_WEIGHTS))
                print(f"carrying the best from iteration {prev.get('_bestAt')}: "
                      f"score {best_h:.5f}")
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
    for it in range(start_it, a.iters):
        # draw from both bands, so a batch always holds one of each
        half = max(1, a.batch // 2)
        idx = torch.cat([torch.randint(0, n_young, (half,)),
                         torch.randint(n_young, a.pool, (a.batch - half,))])
        batch = pool[idx].clone()
        # The worst state in the batch goes back to the seed, so the model has
        # to keep working from scratch -- Growing NCA's trick. But the RATE
        # matters as much as the trick: resetting one of a batch of two every
        # iteration reseeds half the batch, and then no pooled state ever ages
        # past about two windows, so persistence is never actually asked for.
        # Reseeding every few iterations instead lets the pool grow old.
        if it % a.reseed_every == 0:
            # Retiring the worst state looks sensible and quietly defeats the
            # whole point: a lizard that has drifted by step 400 IS the worst
            # state, so it gets deleted instead of repaired, and the pool's age
            # plateaus well below its cap -- it sat at ~325 of 1024 doing this.
            # Retiring the oldest lets drift survive long enough to be fixed.
            with torch.no_grad():
                if a.reseed_policy == "worst":
                    pick = ((batch[:, :3] - target) ** 2).mean(dim=(1, 2, 3)).argmax()
                else:
                    # Oldest RELATIVE TO ITS BAND. Plain argmax over age always
                    # lands on the old band -- its states are older by
                    # construction -- so the reseed meant to keep fresh seeds
                    # coming was executing the long-lived states instead, and
                    # the old band sat at 233 of a 2048 cap. Age over span puts
                    # a young state at 90 of 96 ahead of an old one at 300 of
                    # 2048, which is what "due to be retired" actually means.
                    pick = (ages[idx].double() / lifespan[idx].double()).argmax()
            batch[pick] = seed
            ages[idx[pick]] = 0
        # and retire a state once it has lived its full span, so the pool holds
        # a spread of ages rather than drifting to all-old or all-young
        stale = ages[idx] >= lifespan[idx]
        if stale.any():
            batch[stale] = seed
            ages[idx[stale]] = 0

        # Score the field THROUGHOUT the window, not only where it ends.
        # Scoring the last frame alone asks for the shape to be reached at step
        # 16 and says nothing about step 15 or 17, and a rule can satisfy that
        # by sweeping through the right arrangement on its way somewhere else.
        # Charging for every frame asks for the shape to be STOOD IN, which is
        # what persistence means and what the horizon columns measure.
        kern = model.kern()
        out, scored = batch, []
        for t in range(a.window):
            out = model.rollout(out, 1, kern=kern)
            step_no = t + 1
            if step_no < a.loss_from and step_no != a.window:
                continue          # too early for the shape to exist at all
            if not a.loss_every or step_no % a.loss_every == 0 or t == a.window - 1:
                scored.append(((out[:, :3] - target) ** 2).mean())
        loss = torch.stack(scored).mean()

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
            ages[idx[good]] += a.window
            pool[idx[~good]] = seed          # a blown-up state poisons the pool
            ages[idx[~good]] = 0

        best = min(best, loss.item())
        if it % a.ckpt == 0 or it == a.iters - 1:
            f, r, b = model.scalars()
            secs = time.time() - t0
            hz = horizon_losses(model, target, seed, HORIZONS)
            # weighted toward the long horizons: holding the shape is the point
            score = sum(h * w for h, w in zip(hz, HZ_WEIGHTS)) / sum(HZ_WEIGHTS)
            with open(logp, "a", newline="") as fh:
                csv.writer(fh).writerow([it, f"{loss.item():.6f}", f"{best:.6f}",
                                         f"{lam:.3f}", f"{f:.2f}", f"{b:.3f}",
                                         *[f"{v:.6f}" for v in hz],
                                         int(ages.float().mean()), f"{secs:.0f}"])
            # a rung 1 world is not expressible as a preset: index.html's
            # FS_AFF is a matrix multiply. Write the kernels anyway -- they are
            # still legal -- and say so, rather than emitting a preset that
            # silently loads as something else.
            cfg = model.to_config(a.channels, a.grid)
            if a.hidden:
                cfg["_note"] = ("rung 1: affinity is a per-cell network, not this "
                                "matrix. Needs the FS_AFF change to load faithfully.")
            json.dump(cfg, open(os.path.join(run, "preset.json"), "w"))
            if score < best_h:
                best_h = score
                cfg_best = dict(cfg, _bestAt=it, _horizons=hz)
                json.dump(cfg_best, open(os.path.join(run, "preset-best.json"), "w"))
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "pool": pool, "ages": ages, "iter": it}, ck)
            save_progress(os.path.join(run, "progress.png"), target, model, seed,
                          HORIZONS)
            json.dump({"span": span, "grid": a.grid},
                      open(os.path.join(run, "scale.json"), "w"))
            print(f"  it {it:6d}  loss {loss.item():.5f}  best {best:.5f}  "
                  f"lam {lam:+.2f}  force {f:6.1f}  beta {b:.2f}  "
                  f"age~{int(ages[:n_young].float().mean())}/"
                  f"{int(ages[n_young:].float().mean())}  "
                  f"horizon {'/'.join(f'{v:.4f}' for v in hz)}"
                  f"{' *best*' if score <= best_h else ''}  "
                  f"{secs / max(it - start_it + 1, 1):.2f}s/it", flush=True)

        if a.minutes and time.time() - t0 > a.minutes * 60:
            print(f"stopping after {a.minutes} minutes at iteration {it}")
            break

    print(f"done. best loss {best:.6f}   {run}")


if __name__ == "__main__":
    main()
