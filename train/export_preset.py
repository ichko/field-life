"""
Publish a trained run into the staging page.

    python3 train/export_preset.py --run polar3

Writes three things:

    staging/lizard.json                 the world, for the Lizard button
    staging/worlds.json                 the same world as the shelf's first item
    staging/worlds/000-lizard.thumb.png a thumbnail, rendered from the rollout

The preset carries `seedMasses` alongside the usual fields. It has to: MaCE
only moves mass, so the world is meaningless without the mass its picture
costs, and a preset that arrived without it would load into whatever the
field happened to be holding.

`staging/index.html` is the only page that can run this faithfully -- the
shipped bakeKernel drops angular terms silently. See §7 of
docs/nca-experiment.md.
"""

import argparse
import json
import os
import shutil
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fieldlife as fl
import target as tgt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
STAGING = os.path.join(ROOT, "staging")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="polar3")
    ap.add_argument("--name", default="Lizard")
    ap.add_argument("--kr", type=int, default=13)
    ap.add_argument("--steps", type=int, default=64, help="rollout length for the thumbnail")
    a = ap.parse_args()

    src = os.path.join(HERE, "runs", a.run, "preset.json")
    cfg = json.load(open(src))
    C, N = cfg["C"], cfg["N"]
    if "_note" in cfg:
        sys.exit(f"{a.run} is a rung 1 world; index.html's FS_AFF cannot run it")

    tg = tgt.render_emoji(span=tgt.SPAN, grid=N)
    masses = tgt.seed_masses(tg, C, rng=np.random.default_rng(0))

    cfg = dict(cfg)
    cfg["seedMasses"] = [float(m) for m in masses]
    cfg["seedMode"] = "masses"
    cfg["square"] = True
    cfg["blend"] = 4          # the raw RGB read-out
    cfg["expo"] = 1.0
    cfg["palette"] = "Spectrum"

    os.makedirs(os.path.join(STAGING, "worlds"), exist_ok=True)
    with open(os.path.join(STAGING, "lizard.json"), "w") as f:
        json.dump(cfg, f)

    # roll it out here, both for the thumbnail and to report what it scores
    kern = fl.bake_from_config(cfg["kernels"], C, a.kr)
    mat = torch.tensor(cfg["mat"], dtype=torch.float64).reshape(C, C)
    rho = torch.tensor(tgt.seed_field(masses, grid=N), dtype=torch.float64)[None]
    rho = fl.run(rho, kern, mat, cfg["force"], cfg["repel"], cfg["beta"], a.steps)
    loss = ((rho[0, :3] - torch.tensor(tg)) ** 2).mean().item()

    thumb = "000-lizard.thumb.png"
    img = np.clip(rho[0, :3].numpy().transpose(1, 2, 0), 0, 1)
    Image.fromarray((img * 255).astype(np.uint8)).resize((128, 128), Image.NEAREST) \
        .save(os.path.join(STAGING, "worlds", thumb))

    # first in the shelf, ahead of whatever was already there
    wpath = os.path.join(STAGING, "worlds.json")
    if os.path.exists(wpath):
        shelf = json.load(open(wpath))
    else:
        base = os.path.join(ROOT, "worlds.json")
        shelf = json.load(open(base)) if os.path.exists(base) else {"dir": "worlds", "setups": []}
    shelf.setdefault("dir", "worlds")
    shelf["setups"] = [s for s in shelf.get("setups", []) if s.get("name") != a.name]
    shelf["setups"].insert(0, {"name": a.name, "n": "▶", "thumb": thumb,
                               "tags": ["trained", f"{C} channels", "angular kernels"],
                               "cfg": cfg})
    for i, s in enumerate(shelf["setups"][1:], 1):
        s["n"] = i
    json.dump(shelf, open(wpath, "w"))

    print(f"exported {a.run}: C {C}, grid {N}, "
          f"force {cfg['force']:.1f} beta {cfg['beta']:.3f}")
    print(f"  loss after {a.steps} steps: {loss:.5f}")
    print(f"  staging/lizard.json, staging/worlds/{thumb}, "
          f"and first of {len(shelf['setups'])} in staging/worlds.json")


if __name__ == "__main__":
    main()
