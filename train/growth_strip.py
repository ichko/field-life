"""
Pack a rollout into one sprite strip, for a page that can scrub it.

    python3 train/growth_strip.py --run polar3-w16 --steps 160

A GIF plays at whatever pace the viewer's client decides, and some clients
flatten it to its first frame. A strip of frames drawn to a canvas is just an
image, so it survives that, and it can be stepped through one simulation tick
at a time -- which is the only way to actually see mass travelling one cell
per step.
"""

import argparse
import base64
import io
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fieldlife as fl
import target as tgt

HERE = os.path.dirname(os.path.abspath(__file__))


def hue_ring(t):
    """index.html's Spectrum palette, so a channel is the colour the sim gives it."""
    h = ((t % 1) + 1) % 1 * 6
    x = 1 - abs(h % 2 - 1)
    return [[1, x, 0], [x, 1, 0], [0, 1, x], [0, x, 1], [x, 0, 1], [1, 0, x]][int(h) % 6]


def palette(C):
    return np.array([hue_ring(i / C) for i in range(C)])


def shade(density, colour, expo):
    """One channel drawn as the sim draws it: its hue, brightness from its mass."""
    a = 1.0 - np.exp(-np.maximum(density, 0) * expo)
    return a[..., None] * np.asarray(colour)[None, None, :]


def rollout_frames(cfg, steps, kr=13):
    C, N = cfg["C"], cfg["N"]
    tg = tgt.render_emoji(span=tgt.SPAN, grid=N)
    masses = cfg.get("seedMasses") or tgt.seed_masses(tg, C, rng=np.random.default_rng(0))
    rho = torch.tensor(tgt.seed_field(np.asarray(masses), grid=N), dtype=torch.float64)[None]
    kern = fl.bake_from_config(cfg["kernels"], C, kr)
    mat = torch.tensor(cfg["mat"], dtype=torch.float64).reshape(C, C)
    g = (cfg["force"], cfg["repel"], cfg["beta"])
    tgt_t = torch.tensor(tg)

    frames, losses, mass, chans, permass = [], [], [], [], []
    for i in range(steps + 1):
        if i:
            rho = fl.step(rho, kern, mat, *g)
        losses.append(round(((rho[0, :3] - tgt_t) ** 2).mean().item(), 6))
        mass.append(round(float(rho[0].sum()), 3))
        permass.append([round(float(v), 2) for v in rho[0].sum(dim=(1, 2))])
        frames.append(np.clip(rho[0, :3].numpy().transpose(1, 2, 0), 0, 1))
        chans.append(rho[0].numpy())
    return tg, frames, losses, mass, chans, permass


def b64_png(arr):
    buf = io.BytesIO()
    Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8)).save(buf, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="polar3")
    ap.add_argument("--steps", type=int, default=160)
    ap.add_argument("--kr", type=int, default=13)
    ap.add_argument("--every", type=int, default=1, help="keep every Nth step")
    ap.add_argument("--expo", type=float, default=2.2, help="display brightness, as the sim's")
    ap.add_argument("--out", default=os.path.join(HERE, "growth_data.json"))
    a = ap.parse_args()

    cfg = json.load(open(os.path.join(HERE, "runs", a.run, "preset.json")))
    tg, frames, losses, mass, chans, permass = rollout_frames(cfg, a.steps, a.kr)
    N, C = cfg["N"], cfg["C"]

    keep = list(range(0, a.steps + 1, a.every))
    frames = [frames[i] for i in keep]

    # Every channel, laid out as one image: a row per channel, a column per
    # kept step, each drawn in the hue index.html gives it. One fetch, and a
    # canvas can cut any (channel, step) cell out of it.
    pal = palette(C)
    grid = np.concatenate([
        np.concatenate([shade(chans[i][c], pal[c], a.expo) for i in keep], axis=1)
        for c in range(C)], axis=0)                 # (C*N, len(keep)*N, 3)

    strip = np.concatenate(frames, axis=1)          # (N, N*len(keep), 3)
    data = {
        "run": a.run, "N": N, "C": C, "steps": a.steps,
        "kept": keep, "expo": a.expo,
        "palette": [[round(v, 3) for v in c] for c in pal],
        "perMass": [permass[i] for i in keep],
        "channels": b64_png(grid),
        "force": round(cfg["force"], 2), "beta": round(cfg["beta"], 3),
        "repel": round(cfg["repel"], 3),
        "lobes": len(cfg["kernels"][0]["terms"]),
        "orders": sorted({t.get("m", 0) for k in cfg["kernels"] for t in k["terms"]}),
        "losses": [losses[i] for i in keep], "mass": [mass[i] for i in keep],
        "target": b64_png(tg.transpose(1, 2, 0)),
        "strip": b64_png(strip),
    }
    json.dump(data, open(a.out, "w"))
    kl = [losses[i] for i in keep]
    best = keep[int(np.argmin(kl))]
    print(f"{len(keep)} frames of {N}x{N}, {C} channels -> {a.out} "
          f"({os.path.getsize(a.out)/1e6:.2f} MB)")
    print(f"  channel sheet {grid.shape[1]}x{grid.shape[0]} px")
    print(f"  best step {best} (loss {min(kl):.5f}), "
          f"last {kl[-1]:.5f}, mass drift {abs(mass[-1]-mass[0]):.2e}")


if __name__ == "__main__":
    main()
