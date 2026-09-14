"""Fit the brains rule so a seed disc grows into the lizard emoji.

The same target the flat page's lizard was fitted to, and the same shape of
problem: a disc holding the mass the picture costs, grown until it is the
picture, and still the picture if you keep going.

The thing that had to change from the first attempt is the SEED. That one drew
a fresh random velocity field every iteration, so every fresh run broke the
rotational symmetry a different way and the fit averaged over all of them -- and
the average of a lizard over every angle is a disc, which is exactly what it
made. The rule has no preferred direction of its own; the seed has to give it
one, and give it the SAME one every time. So each species starts with one
coherent heading, fixed, a different angle each. Nothing else about the seed is
random at all.
"""
import argparse, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import field_brains as FB
from emoji_gecko import build as build_emoji

torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "4")))


def seed_state(N, parts, fill, disc=0.115):
    """One disc in the middle holding each species' share of what the picture
    costs, and one fixed coherent heading per species. Deterministic: the fit
    sees the same world every time, which is the only way it can commit to an
    orientation instead of averaging over them."""
    P = torch.as_tensor(parts)
    frac = P.sum((1, 2))/torch.clamp(P.sum(), min=1e-9)
    yy, xx = torch.meshgrid(torch.arange(N).float(), torch.arange(N).float(), indexing="ij")
    d = (torch.hypot(xx - N/2 + 0.5, yy - N/2 + 0.5) < N*disc).float()
    C = d[None]*frac[:, None, None]*(fill*N*N/torch.clamp(d.sum(), min=1.0))
    ang = torch.tensor([2*np.pi*s/FB.NS + 0.3 for s in range(FB.NS)])
    V = torch.stack([torch.cos(ang), torch.sin(ang)], 1)[:, :, None, None]*0.25
    V = V.expand(FB.NS, 2, N, N).clone()
    return FB.State(C, V)


def loss_of(st, P, wsil):
    C = st.C/torch.clamp(st.C.sum(), min=1e-9)
    part = ((C - P)**2).mean()*C.shape[-1]**2
    a, b = C.sum(0), P.sum(0)
    dice = 1 - 2*(a*b).sum()/torch.clamp((a*a).sum() + (b*b).sum(), min=1e-12)
    return part + wsil*dice, float(part), float(dice)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--N", type=int, default=64)
    ap.add_argument("--warm", type=int, default=32)
    ap.add_argument("--tail", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--pool", type=int, default=24)
    ap.add_argument("--fresh", type=float, default=0.4)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--wsil", type=float, default=3.0)
    ap.add_argument("--blur", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "emojiGecko.pt"))
    ap.add_argument("--resume", default=None)
    a = ap.parse_args()

    parts, cols, _ = build_emoji(a.N, blur=a.blur)
    P = torch.tensor(parts); P = P/P.sum()
    cfg = dict(shape=1, mirror=0, dist=2.5, size=1, feather=0.2, spread=162,
               crowd=0.0, fill=0.12)
    # Mirror OFF: the emoji is a specific handed animal, and averaging a brain
    # with its own reflection is asking for one that cannot tell left from right.
    rule = FB.Rule(seed=a.seed)
    if a.resume and os.path.exists(a.resume):
        ck = torch.load(a.resume, weights_only=False)
        with torch.no_grad():
            rule.frq.copy_(ck["frq"]); rule.amp.copy_(ck["amp"])
            for k, v in ck["raw"].items(): rule.raw[k].copy_(v)
        print("resumed from", a.resume, flush=True)
    opt = torch.optim.Adam(rule.params(), lr=a.lr)
    gen = torch.Generator().manual_seed(a.seed)
    pool, best, skipped = [], 1e9, 0

    for it in range(a.iters):
        t0 = time.time()
        fresh = (not pool) or (torch.rand(1, generator=gen).item() < a.fresh)
        if fresh:
            st, n = seed_state(a.N, parts, cfg["fill"]), a.warm
        else:
            st = pool[int(torch.randint(len(pool), (1,), generator=gen))].detach()
            n = a.chunk
        with torch.no_grad():
            for _ in range(max(0, n - a.tail)):
                st = FB.step(st, rule, cfg)
        st = st.detach()
        for _ in range(min(n, a.tail)):
            st = FB.step(st, rule, cfg)
        L, lp, ld = loss_of(st, P, a.wsil)
        opt.zero_grad(set_to_none=True)
        L.backward()
        gn = torch.nn.utils.clip_grad_norm_(rule.params(), a.clip)
        if torch.isfinite(gn):
            opt.step()
        else:
            skipped += 1; opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            d = st.detach()
            if torch.isfinite(d.C).all() and torch.isfinite(d.M).all():
                pool.append(d)
                if len(pool) > a.pool: pool.pop(0)
        if fresh and float(L) < best:
            best = float(L)
            torch.save({"frq": rule.frq.detach(), "amp": rule.amp.detach(),
                        "raw": {k: v.detach() for k, v in rule.raw.items()},
                        "cfg": cfg, "cols": cols, "loss": best, "iter": it}, a.out)
        if it % 20 == 0 or it == a.iters - 1:
            print(f"{it:4d} {'seed' if fresh else 'pool'} L {float(L):.4f} "
                  f"(part {lp:.4f} dice {ld:.4f}) |g| {float(gn):.1e} "
                  f"{time.time()-t0:.1f}s best {best:.4f} skip {skipped}", flush=True)


if __name__ == "__main__":
    main()
