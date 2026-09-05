"""Random search over the brains parameter space, scored for structure.

What counts as interesting here is stated rather than assumed: a field that is
still moving, does not fill everything or nothing, and whose matter is drawn
into COHERENT DIRECTIONAL structure -- filaments, not blobs and not noise. That
last one is the structure-tensor coherence, and it is the metric that separates
a physarum-looking field from a Keller-Segel blob or a boiling mess.
"""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from explore import Brains, NS

N_GRID, STEPS, WARM = 72, 200, 120


def metrics(hist, C):
    u = hist[-1]
    m = u.mean()
    if not np.isfinite(m) or m <= 1e-6:
        return None
    u = u / m
    gx = np.roll(u, -1, 1) - np.roll(u, 1, 1)
    gy = np.roll(u, -1, 0) - np.roll(u, 1, 0)
    g2 = gx * gx + gy * gy
    # structure tensor, smoothed over a 5x5, then coherence weighted by edge
    def box(a):
        for ax in (0, 1):
            a = sum(np.roll(a, s, ax) for s in (-2, -1, 0, 1, 2)) / 5.0
        return a
    Jxx, Jyy, Jxy = box(gx * gx), box(gy * gy), box(gx * gy)
    tr = Jxx + Jyy
    det = Jxx * Jyy - Jxy * Jxy
    disc = np.sqrt(np.maximum(tr * tr - 4 * det, 0))
    coh = np.where(tr > 1e-12, disc / np.maximum(tr, 1e-12), 0)
    w = np.sqrt(g2)
    aniso = float((coh * w).sum() / max(w.sum(), 1e-12))
    tot = C.sum(-1)
    mask = tot > 1e-9
    pur = 0.0
    if mask.any():
        frac = C[mask] / tot[mask][:, None]
        best = frac.max(-1)
        pur = float(np.average((best * NS - 1) / (NS - 1), weights=tot[mask]))
    churn = float(np.abs(hist[-1] - hist[-2]).mean() / m)
    return dict(mean=float(m), contrast=float(u.std()), edge=float(np.abs(gx).mean() + np.abs(gy).mean()),
                aniso=aniso, occ=float((u > 0.5).mean()), churn=churn, purity=pur,
                drift=float(abs(C.sum() / max(tot.sum(), 1e-12) - 1)))


PAL = np.array([[1,.35,0],[1,.85,0],[.35,1,0],[0,1,.55],[0,.85,1],[.3,.4,1],
                [.7,.25,1],[1,.15,.7],[1,.55,.35]], np.float32)[:8]


def thumb(M, mnorm, floor=0.05, expo=1.5):
    """The page's own blend: hue is the weighted mean of the channels' hues,
    weighted by the SQUARE of how much of each is there, brightness a separate
    curve on the total. Adding instead drives everything to white."""
    a = np.maximum(M, 0) * mnorm
    w = a * a
    hue = w @ PAL
    tot, wsum = a.sum(-1), w.sum(-1)
    al = 1 - np.exp(-np.maximum(tot / a.shape[-1] - floor, 0) * expo)
    col = hue / np.maximum(wsum, 1e-12)[..., None] * al[..., None] + np.array([.016, .024, .04], np.float32)
    return (np.clip(col, 0, 1) * 255).astype(np.uint8)


def sample(rng):
    lg = lambda a, b: float(np.exp(rng.uniform(np.log(a), np.log(b))))
    return dict(
        N=N_GRID, seed=int(rng.integers(1e9)),
        shape=int(rng.integers(0, 3)), mirror=int(rng.integers(0, 2)),
        gain=round(lg(0.2, 5.0), 3), speed=round(float(rng.uniform(0, 5)), 2),
        beta=round(lg(3.0, 35.0), 2), turn=round(float(rng.uniform(3, 90)), 1),
        dist=round(lg(1.0, 16.0), 2), spread=round(float(rng.uniform(8, 178)), 0),
        size=round(lg(1.0, 4.0), 2), feather=round(float(rng.uniform(0, 1)), 2),
        decay=round(float(rng.uniform(0.80, 0.99)), 3),
        diff=round(float(rng.uniform(0, 0.5)), 2),
        moff=round(lg(0.2, 3.0), 2), crowd=round(float(rng.choice([0, 0, 0, rng.uniform(0, 3)])), 2),
        fill=0.1, ball=0.55)


def run(args):
    P, bseed = args
    try:
        b = Brains(P, bseed)
        hist = []
        for i in range(STEPS):
            b.step()
            if not np.isfinite(b.M).all():
                return None
            if i in (WARM, STEPS - 1):
                hist.append(b.M.sum(-1).copy())
        if len(hist) < 2:
            return None
        mid = metrics(hist, b.C)
        if mid is None:
            return None
        mid["massdrift"] = float(abs(b.C.sum() / b.mass0 - 1))
        mn = (1 - P["decay"]) / max(1e-6, P["moff"] * P["fill"])
        return dict(P=P, bseed=int(bseed), thumb=thumb(b.M, mn).tolist(), **mid)
    except Exception as e:
        return None


if __name__ == "__main__":
    import multiprocessing as mp
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    rng = np.random.default_rng(int(sys.argv[2]) if len(sys.argv) > 2 else 1)
    jobs = [(sample(rng), int(rng.integers(1e9))) for _ in range(n)]
    t = time.time()
    with mp.Pool(4) as pool:
        out = [r for r in pool.imap_unordered(run, jobs) if r]
    print(f"{len(out)}/{n} survived in {time.time()-t:.0f}s", file=sys.stderr)
    path = sys.argv[3] if len(sys.argv) > 3 else "sweep.json"
    thumbs = np.array([r.pop("thumb") for r in out], np.uint8)
    np.save(path.replace(".json", "_thumbs.npy"), thumbs)
    json.dump(out, open(path, "w"))
