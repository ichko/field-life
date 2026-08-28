"""
Replay a trained preset and report how well it holds the target.

    python3 train/evaluate.py train/runs/polar/preset.json

Everything comes from the preset -- the same file index.html would load -- so
this also checks that nothing the trainer learned lives outside it.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fieldlife as fl
import target as tgt

HORIZONS = [8, 16, 32, 64, 128, 256]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("preset")
    ap.add_argument("--kr", type=int, default=13)
    ap.add_argument("--png", default=None)
    a = ap.parse_args()

    cfg = json.load(open(a.preset))
    C, N = cfg["C"], cfg["N"]
    tg = torch.tensor(tgt.render_emoji(span=tgt.SPAN, grid=N), dtype=torch.float64)
    masses = tgt.seed_masses(tg.numpy(), C, rng=np.random.default_rng(0))
    rho = torch.tensor(tgt.seed_field(masses, grid=N), dtype=torch.float64)[None]

    kern = fl.bake_from_config(cfg["kernels"], C, a.kr)
    mat = torch.tensor(cfg["mat"], dtype=torch.float64).reshape(C, C)
    g = (cfg["force"], cfg["repel"], cfg["beta"])
    print(f"{os.path.basename(os.path.dirname(a.preset))}: C {C}  grid {N}  "
          f"force {g[0]:.1f}  repel {g[1]:.2f}  beta {g[2]:.3f}")

    tiles, done = [tg.numpy()], 0
    print(f"  {'steps':>6} {'loss':>10}")
    for h in HORIZONS:
        for _ in range(h - done):
            rho = fl.step(rho, kern, mat, *g)
        done = h
        loss = ((rho[0, :3] - tg) ** 2).mean().item()
        print(f"  {h:>6} {loss:>10.6f}")
        tiles.append(rho[0, :3].numpy())

    drift = (rho[0].sum(dim=(1, 2)) - torch.tensor(masses)).abs().max().item()
    print(f"  mass drift over {done} steps: {drift:.2e}")

    if a.png:
        strip = np.concatenate([np.clip(t.transpose(1, 2, 0), 0, 1) for t in tiles], axis=1)
        img = Image.fromarray((strip * 255).astype(np.uint8))
        img.resize((img.width * 4, img.height * 4), Image.NEAREST).save(a.png)
        print(f"  wrote {a.png}")


if __name__ == "__main__":
    main()
