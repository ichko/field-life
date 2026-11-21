"""Symmetrisation test of the seven gliders. Run from the repository root: python -m fieldlife.symmetry

Each glider runs alone for 150 steps under (a) its own matrix, (b) the symmetric part (M + M^T) / 2,
(c) the symmetric part plus a fresh random antisymmetric part (two draws). Prints the speed of the centre of mass."""
import sys, json, numpy as np
sys.path.insert(0, ".")
from fieldlife.model import World, Sim, load_setups
G = load_setups("data/gliders.json")
N = 512; STEPS = 150
def com(rho):
    tot = rho.sum(0); m = tot.sum()
    ys, xs = np.mgrid[0:N, 0:N]
    return (xs * tot).sum() / m, (ys * tot).sum() / m
def speed(w, pattern):
    sim = Sim(w, np.float32).seed_pattern(pattern)
    path = [com(sim.rho)]
    for t in range(STEPS):
        sim.step()
        x, y = com(sim.rho); px, py = path[-1]
        path.append((px + ((x - px + N / 2) % N - N / 2), py + ((y - py + N / 2) % N - N / 2)))
    p = np.array(path); return float(np.hypot(*np.diff(p, axis=0).T).sum() / STEPS)
out = []
rng = np.random.default_rng(0)
for i, g in enumerate(G):
    w = World.from_config(g["cfg"]); pat = g["cfg"]["seedPattern"]
    M = w.M.copy(); S = (M + M.T) / 2
    row = {"body": f"B{i}", "own": speed(w, pat)}
    w.M = S; row["sym"] = speed(w, pat)
    for k in range(2):
        A = rng.uniform(-1, 1, M.shape); A = (A - A.T) / 2
        scale = np.abs((M - M.T) / 2).mean() / max(np.abs(A).mean(), 1e-9)   # same typical size as the original antisymmetric part
        w.M = S + A * scale; row[f"resym{k}"] = speed(w, pat)
    out.append(row); print(row, flush=True)
json.dump(out, open("symmetry.json", "w"), indent=1)
