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


def rollout_frames(cfg, steps, kr=13):
    C, N = cfg["C"], cfg["N"]
    tg = tgt.render_emoji(span=tgt.SPAN, grid=N)
    masses = cfg.get("seedMasses") or tgt.seed_masses(tg, C, rng=np.random.default_rng(0))
    rho = torch.tensor(tgt.seed_field(np.asarray(masses), grid=N), dtype=torch.float64)[None]
    kern = fl.bake_from_config(cfg["kernels"], C, kr)
    mat = torch.tensor(cfg["mat"], dtype=torch.float64).reshape(C, C)
    g = (cfg["force"], cfg["repel"], cfg["beta"])
    tgt_t = torch.tensor(tg)

    frames, losses, mass = [], [], []
    for i in range(steps + 1):
        if i:
            rho = fl.step(rho, kern, mat, *g)
        losses.append(round(((rho[0, :3] - tgt_t) ** 2).mean().item(), 6))
        mass.append(round(float(rho[0].sum()), 3))
        frames.append(np.clip(rho[0, :3].numpy().transpose(1, 2, 0), 0, 1))
    return tg, frames, losses, mass


def b64_png(arr):
    buf = io.BytesIO()
    Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8)).save(buf, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="polar3")
    ap.add_argument("--steps", type=int, default=160)
    ap.add_argument("--kr", type=int, default=13)
    ap.add_argument("--out", default=os.path.join(HERE, "growth_data.json"))
    a = ap.parse_args()

    cfg = json.load(open(os.path.join(HERE, "runs", a.run, "preset.json")))
    tg, frames, losses, mass = rollout_frames(cfg, a.steps, a.kr)
    N = cfg["N"]

    strip = np.concatenate(frames, axis=1)          # (N, N*(steps+1), 3)
    data = {
        "run": a.run, "N": N, "C": cfg["C"], "steps": a.steps,
        "force": round(cfg["force"], 2), "beta": round(cfg["beta"], 3),
        "repel": round(cfg["repel"], 3),
        "lobes": len(cfg["kernels"][0]["terms"]),
        "orders": sorted({t.get("m", 0) for k in cfg["kernels"] for t in k["terms"]}),
        "losses": losses, "mass": mass,
        "target": b64_png(tg.transpose(1, 2, 0)),
        "strip": b64_png(strip),
    }
    json.dump(data, open(a.out, "w"))
    best = int(np.argmin(losses))
    print(f"{len(frames)} frames of {N}x{N} -> {a.out} "
          f"({os.path.getsize(a.out)/1e6:.2f} MB)")
    print(f"  best step {best} (loss {losses[best]:.5f}), "
          f"last {losses[-1]:.5f}, mass drift {abs(mass[-1]-mass[0]):.2e}")


if __name__ == "__main__":
    main()
