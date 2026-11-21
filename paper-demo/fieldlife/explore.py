"""Reproducible exploration of the parameter space.

Draws random rules (see model.random_world), runs each from noise (or from a disc), measures what it
does, keeps the samples that form structure and move it coherently, and tags every kept sample.
All randomness comes from one seed, so the same command gives the same worlds.

    python -m fieldlife.explore --n 24 --seed 0 --out results

writes results/worlds.json (same layout as data/worlds.json), results/measures.json, and a PNG of
every sample.  Runtime is about one minute per sample at N = 512 and 300 steps.

The measurements are heuristics.  They are the ones used to build the catalogue in data/worlds.json,
reimplemented here with their thresholds stated in the code.
"""
import argparse, json, os
import numpy as np
from .model import World, Sim, random_world
from .render import to_image

# ---- measurements ------------------------------------------------------------------------------
def activity(prev, rho):
    """Fraction of the mass that moved in one step."""
    return np.abs(rho - prev).sum() / rho.sum()


def components(mask):
    """Label the connected regions of a boolean mask on the torus (4-neighbourhood)."""
    N = mask.shape[0]
    lab = np.where(mask, np.arange(N * N).reshape(N, N) + 1, 0)
    while True:
        new = lab.copy()
        for s in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            r = np.roll(lab, s, (0, 1)); new = np.where(mask & (r > 0) & (r < new), r, new)
        if np.array_equal(new, lab): return lab
        lab = new


def structure_size(total):
    """Typical size of the structures, in cells: the mass-weighted median of sqrt(area) over the
    connected regions that lie above a tenth of the peak concentration."""
    lab = components(total > 0.1 * np.percentile(total, 99.5))
    ids, area = np.unique(lab[lab > 0], return_counts=True)
    if len(ids) == 0: return 0.0
    mass = np.array([total[lab == i].sum() for i in ids]); order = np.argsort(area)
    cum = np.cumsum(mass[order]) / mass.sum()
    return float(np.sqrt(area[order][np.searchsorted(cum, 0.5)]))


def local_flow(prev, rho, patch=32, maxshift=4):
    """Displacement of the mass in each patch between two steps, by the best-matching shift.
    Returns (changed mass per patch, shift per patch as (dy, dx))."""
    tot0, tot1 = prev.sum(0), rho.sum(0); N = tot0.shape[0]; n = N // patch
    moved = np.abs(tot1 - tot0)
    masses = np.zeros((n, n)); shifts = np.zeros((n, n, 2))
    cands = [(dy, dx) for dy in range(-maxshift, maxshift + 1) for dx in range(-maxshift, maxshift + 1)]
    rolled = {s: np.roll(tot0, s, (0, 1)) for s in cands}
    for i in range(n):
        for j in range(n):
            sl = (slice(i * patch, (i + 1) * patch), slice(j * patch, (j + 1) * patch))
            a = tot1[sl]; masses[i, j] = moved[sl].sum()      # weight: how much mass changed in the patch
            if masses[i, j] <= 0: continue
            err = [np.abs(a - rolled[s][sl]).sum() for s in cands]
            shifts[i, j] = cands[int(np.argmin(err))]
    return masses, shifts


def measure(sim, steps, every=10):
    """Run `steps` steps and return the measurements used for scoring and tagging."""
    acts, speeds, aligns = [], [], []
    for t in range(1, steps + 1):
        prev = sim.rho.copy(); sim.step()
        if t % every == 0:
            acts.append(activity(prev, sim.rho))
            m, v = local_flow(prev, sim.rho)
            sp = np.hypot(v[..., 0], v[..., 1]); w = m / max(m.sum(), 1e-12)
            speeds.append((w * sp).sum())                                       # speed of the mass that moves, cells/step
            mv = (w[..., None] * v).sum((0, 1)); aligns.append(np.hypot(*mv) / max((w * sp).sum(), 1e-12))
    rho = sim.rho; total = rho.sum(0); N = total.shape[0]
    top = np.percentile(total, 99.5)
    occupied = (total > 0.1 * top).mean()                                       # fraction of cells that hold structure
    haze = total[(total > 0.002 * top) & (total < 0.05 * top)].sum() / total.sum()  # mass sitting in a faint halo
    purity = ((rho ** 2).sum(0) / np.maximum(total ** 2, 1e-12) * total).sum() / total.sum()  # 1 = every cell one colour
    # stripes: anisotropy of one colour's power spectrum, largest over colours
    aniso = 0
    for c in range(len(rho)):
        P = np.abs(np.fft.fftshift(np.fft.fft2(rho[c] - rho[c].mean()))) ** 2
        ky, kx = np.mgrid[-N // 2:N // 2, -N // 2:N // 2]; ang = np.arctan2(ky, kx) % np.pi; r = np.hypot(kx, ky)
        sel = (r > 2) & (r < N / 6)
        hist = np.bincount((ang[sel] / np.pi * 12).astype(int) % 12, weights=P[sel], minlength=12)
        aniso = max(aniso, hist.max() / max(hist.mean(), 1e-12))
    return {"activity": float(np.mean(acts)), "activity_end": float(np.mean(acts[-3:])),
            "speed": float(np.mean(speeds)), "alignment": float(np.mean(aligns)),
            "size": float(structure_size(total)), "occupied": float(occupied), "haze": float(haze),
            "purity": float(purity), "anisotropy": float(aniso)}


# ---- scoring and tags --------------------------------------------------------------------------
def keep(m, reach):
    """A sample is kept when it moves, forms structure larger than its kernel core, and moves it coherently."""
    return m["activity"] > 0.02 and m["size"] > 0.15 * reach and m["speed"] > 0.05


def scale(x, edges, names):
    return names[int(np.searchsorted(edges, x))]


def tags(m, C):
    t = [scale(m["size"], [8, 16, 32, 64], ["fine", "small", "medium", "large", "coarse"]),
         scale(m["speed"], [0.1, 0.25, 0.5, 1.0], ["slow", "steady", "brisk", "fast", "furious"]),
         scale(m["occupied"], [0.08, 0.2, 0.4, 0.7], ["scattered", "spread", "clustered", "filling", "colony"])]
    if m["speed"] > 0.25 and m["occupied"] < 0.4 and m["purity"] > 0.6: t.append("gliders")
    if m["haze"] > 0.12: t.append("gassy")
    if m["alignment"] > 0.7 and m["speed"] > 0.1: t.append("drifting")
    if m["activity_end"] > 0.02: t.append("restless")
    if m["purity"] > 0.8: t.append("sorted")
    if m["anisotropy"] > 4.0: t.append("corduroy")
    return t


# ---- main --------------------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=24, help="number of rules to draw")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--start", choices=["noise", "disc"], default="noise", help="initial state: noise over the world, or a disc")
    p.add_argument("--random-scalars", action="store_true", help="also draw reach, force, crowding and sharpness")
    p.add_argument("--symmetric", action="store_true", help="replace each drawn matrix by its symmetric part (M + M^T) / 2")
    p.add_argument("--out", default="results")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    kept, measures = [], []
    for i in range(a.n):
        w = random_world(rng, N=a.size, random_scalars=a.random_scalars)
        if a.symmetric:
            w.M = (w.M + w.M.T) / 2
        sim = Sim(w, np.float32); (sim.seed_noise if a.start == "noise" else sim.seed_disc)(rng)
        m = measure(sim, a.steps); m["sample"] = i; m["kept"] = keep(m, float(w.reach.mean()))
        measures.append(m)
        to_image(sim.rho).save(f"{a.out}/sample{i:03d}.png")
        if m["kept"]:
            t = tags(m, w.C)
            kept.append({"name": " ".join(t[:3]), "tags": t, "cfg": w.to_config(), "sample": i})
        print(f"sample {i:3d}  C={w.C}  kept={m['kept']}  " + "  ".join(f"{k}={v:.3f}" for k, v in m.items() if isinstance(v, float)), flush=True)
    json.dump({"setups": kept}, open(f"{a.out}/worlds.json", "w"), indent=1)
    json.dump(measures, open(f"{a.out}/measures.json", "w"), indent=1)
    print(f"kept {len(kept)} of {a.n}; wrote {a.out}/worlds.json")


if __name__ == "__main__":
    main()
