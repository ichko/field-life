"""Fit the brains rule so a seed grows into the gecko.

Three things the volume fit paid for and wrote down, all of which apply here:

  The target must be REACHABLE. The rule's smallest feature is about a cell, so
  the target is blurred to that and no finer.

  Colours are SPECIES, and species separate. The animal is cut into four parts
  that do not overlap, one per species, and put back together with colour at the
  end. Better still here: species mass is conserved exactly, so each species is
  SEEDED with the mass its part will need. Conservation then works for the fit
  instead of against it -- the right amount of each is already in the world and
  the rule only has to arrange it.

  The pattern must be a FIXED POINT, not a waypoint. A rule tuned to look right
  at step 30 will happily keep going at step 40. So keep a pool of states the
  rule has already made and restart from them as often as from the seed: a state
  that is already the animal then gets scored on whether it is still the animal
  afterwards, which is the only way an attractor is learned.
"""
import argparse, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import field_brains as FB
from gecko2d import build2d

torch.set_num_threads(4)


def seed_state(N, parts, fill, gen):
    """One blob in the middle, holding each species in the proportion its part
    will need, and pointing every which way."""
    frac = parts.sum((1, 2))/parts.sum()
    yy, xx = torch.meshgrid(torch.arange(N).float(), torch.arange(N).float(), indexing="ij")
    r = torch.hypot(xx - N/2 + 0.5, yy - N/2 + 0.5)
    disc = (r < N*0.13).float()
    C = disc[None]*frac[:, None, None]*(fill*N*N/max(float(disc.sum()), 1.0))
    a = torch.rand(FB.NS, N, N, generator=gen)*2*np.pi
    V = torch.stack([torch.cos(a), torch.sin(a)], 1)*0.25
    return FB.State(C, V)


def loss_of(st, P, wsil):
    """Everything normalised to sum one, so the fit is about ARRANGEMENT and not
    about how much there is -- which conservation has already settled."""
    C = st.C/torch.clamp(st.C.sum(), min=1e-9)
    part = ((C - P)**2).mean()*C.shape[-1]**2
    sil = ((C.sum(0) - P.sum(0))**2).mean()*C.shape[-1]**2
    return part + wsil*sil, float(part), float(sil)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--N", type=int, default=64)
    ap.add_argument("--warm", type=int, default=26)
    ap.add_argument("--chunk", type=int, default=13)
    ap.add_argument("--tail", type=int, default=10)
    ap.add_argument("--pool", type=int, default=12)
    ap.add_argument("--fresh", type=float, default=0.45)
    ap.add_argument("--lr", type=float, default=6e-3)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--wsil", type=float, default=1.5)
    ap.add_argument("--blur", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", default="geckoBrains.pt")
    ap.add_argument("--resume", default=None)
    a = ap.parse_args()

    parts, _ = build2d(a.N, blur=a.blur)
    P = torch.tensor(parts); P = P/P.sum()
    cfg = dict(shape=1, mirror=1, dist=2.5, size=1, feather=0.2, spread=162,
               crowd=0.0, fill=0.1)
    rule = FB.Rule(seed=a.seed)
    if a.resume and os.path.exists(a.resume):
        ck = torch.load(a.resume, weights_only=False)
        with torch.no_grad():
            rule.frq.copy_(ck["frq"]); rule.amp.copy_(ck["amp"])
            for k, v in ck["raw"].items(): rule.raw[k].copy_(v)
        print("resumed from", a.resume, flush=True)
    opt = torch.optim.Adam(rule.params(), lr=a.lr)
    gen = torch.Generator().manual_seed(a.seed)
    pool, best, hist, skipped = [], 1e9, [], 0

    for it in range(a.iters):
        t0 = time.time()
        fresh = (not pool) or (torch.rand(1, generator=gen).item() < a.fresh)
        if fresh:
            st, n = seed_state(a.N, parts, cfg["fill"], gen), a.warm
        else:
            st = pool[int(torch.randint(len(pool), (1,), generator=gen))].detach()
            n = a.chunk
        # TRUNCATED. Measured, the gradient through this rule grows about
        # sixteenfold a step -- 3e2 at five steps, 2e8 at ten, past float32 at
        # twenty -- which is not a bug but what differentiating a chaotic map
        # does. So the field is still run the whole way, and only the last few
        # steps carry a gradient: the fit sees a state that took twenty-six
        # steps to build and is asked what to change about the end of it.
        with torch.no_grad():
            for _ in range(max(0, n - a.tail)):
                st = FB.step(st, rule, cfg)
        st = st.detach()
        for _ in range(min(n, a.tail)):
            st = FB.step(st, rule, cfg)
        L, lp, ls = loss_of(st, P, a.wsil)
        opt.zero_grad(set_to_none=True)
        L.backward()
        gn = torch.nn.utils.clip_grad_norm_(rule.params(), a.clip)
        # A single non-finite gradient poisons every weight through Adam's
        # running moments, and the run never recovers. Drop the step instead.
        if torch.isfinite(gn):
            opt.step()
        else:
            skipped += 1
            opt.zero_grad(set_to_none=True)

        with torch.no_grad():
            d = st.detach()
            if torch.isfinite(d.C).all() and torch.isfinite(d.M).all():
                pool.append(d)
                if len(pool) > a.pool: pool.pop(0)
        hist.append((float(L), fresh))
        if float(L) < best and fresh:
            best = float(L)
            torch.save({"frq": rule.frq.detach(), "amp": rule.amp.detach(),
                        "raw": {k: v.detach() for k, v in rule.raw.items()},
                        "cfg": cfg, "loss": best, "iter": it}, a.out)
        if it % 10 == 0 or it == a.iters - 1:
            kn = {k: round(float(v), 3) for k, v in rule.knobs().items()}
            print(f"{it:4d} {'seed' if fresh else 'pool'} L {float(L):.4f} "
                  f"(part {lp:.4f} sil {ls:.4f}) |g| {float(gn):.1e} "
                  f"{time.time()-t0:.1f}s best {best:.4f} skip {skipped}", flush=True)
            if it % 50 == 0: print("     ", kn, flush=True)
    fr = [l for l, f in hist if f]
    for i in range(0, len(fr) - 9, max(1, len(fr)//6)):
        print(f"seed runs {i:3d}-{i+9:3d}  mean {np.mean(fr[i:i+10]):.4f}", flush=True)


if __name__ == "__main__":
    main()
