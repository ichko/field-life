"""
Publish a trained run into the staging page.

    python3 train/export_preset.py --run polar3

Writes three things:

    staging/worlds.json                 the world, as the shelf's first item
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
    ap.add_argument("--kr", type=int, default=0,
                    help="stencil half-width; 0 derives it from the preset the "
                         "way uploadKernels does")
    ap.add_argument("--seed-radius", type=float, default=0.0,
                    help="0 reads it from the run's scale.json")
    ap.add_argument("--steps", type=int, default=64, help="rollout length for the thumbnail")
    ap.add_argument("--no-flip", action="store_true",
                    help="skip the vertical flip that matches the page's y axis")
    ap.add_argument("--latest", action="store_true",
                    help="export the newest checkpoint rather than the best one")
    a = ap.parse_args()

    # prefer the best checkpoint the run has seen over its newest one
    best = os.path.join(HERE, "runs", a.run, "preset-best.json")
    src = best if (os.path.exists(best) and not a.latest) \
        else os.path.join(HERE, "runs", a.run, "preset.json")
    cfg = json.load(open(src))
    if "_bestAt" in cfg:
        print(f"exporting the best checkpoint, from iteration {cfg['_bestAt']} "
              f"(horizons {'/'.join(f'{v:.4f}' for v in cfg['_horizons'])})")
    C, N = cfg["C"], cfg["N"]
    if "_note" in cfg:
        sys.exit(f"{a.run} is a rung 1 world; index.html's FS_AFF cannot run it")

    # The run's own scale, not the module default. A world fitted to a
    # 120-cell animal exported against the 40-cell one gets the wrong seed
    # masses, and MaCE only moves mass -- wrong masses is a different world.
    scale = {}
    sp = os.path.join(HERE, "runs", a.run, "scale.json")
    if os.path.exists(sp):
        scale = json.load(open(sp))
    span = scale.get("span", tgt.SPAN)
    tg = tgt.render_emoji(span=span, grid=N)
    masses = tgt.seed_masses(tg, C, rng=np.random.default_rng(0))
    tg_shown = tg

    # The seed disc, likewise. The page defaults to a tenth of the grid; a run
    # that trained on a wider one starts from a disc it never saw.
    radius = a.seed_radius or scale.get("seed_radius") or N * 0.10
    if not a.seed_radius and not scale.get("seed_radius"):
        print(f"  WARNING: {a.run}/scale.json does not record a seed radius, so "
              f"this falls back to the page default of {N * 0.10:g}. If the run "
              f"trained on a different disc the export is a different world -- "
              f"pass --seed-radius.")
    mip, KR = fl.plan_from_config(cfg["kernels"], C, N)
    kr = a.kr or KR
    print(f"  span {span}, seed radius {radius:g}, mip {mip}, stencil {kr}")

    cfg = dict(cfg)
    cfg.pop("_bestAt", None); cfg.pop("_horizons", None)

    # index.html draws array row 0 at the BOTTOM -- gl_FragCoord counts y up
    # from the bottom of the framebuffer -- while the target is an image, whose
    # row 0 is its top. A world fitted against the image therefore renders
    # upside down, head at the bottom.
    #
    # Rather than flip the renderer, which would turn every existing world over
    # for no reason, flip the world. Mirroring in y sends theta to -theta, so
    # cos(m*theta + phase) becomes cos(m*theta - phase): negating every phase
    # mirrors each kernel, and mirroring the kernels mirrors the whole
    # trajectory, because everything else in the step -- the crowding blur, the
    # matrix, the 3x3 softmax -- is symmetric, and the seed disc is too.
    # Verified: the flipped bank matches the flipped bake to 2e-17 and the
    # flipped trajectory to 1e-13, at identical loss.
    if not a.no_flip:
        cfg["kernels"] = json.loads(json.dumps(cfg["kernels"]))
        for k in cfg["kernels"]:
            for t in k.get("terms", []):
                if t.get("phase"):
                    t["phase"] = -t["phase"]
        print("  flipped in y to match the page's axis (phases negated)")
    cfg["seedMasses"] = [float(m) for m in masses]
    cfg["seedMode"] = "masses"
    # seedDisc, not a name of our own: index.html already carries this field
    # ("seed radius in cells, 0 = a tenth of the world") and reads it in
    # applyConfig. A second field meaning the same thing would load as nothing.
    cfg["seedDisc"] = float(radius)
    cfg["square"] = True
    # blend 5: the three visible channels as themselves, the hidden ones
    # screened in behind them. RGB alone hides that the shape sits inside a
    # much larger scaffold; a flat blend of everything drowns the shape.
    cfg["blend"] = 5
    # Exposure 1.0 leaves the three visible channels dim -- blend 5 draws them
    # as clamp(rgb*expo), so mass around 0.3 stays a third lit -- while the
    # hidden scaffold behind them is screened in through 1-exp(-total*expo),
    # which saturates regardless. The animal ends up underneath its own
    # scaffold. 2.2 is what the offline renders use and what the page's own
    # worlds ship with.
    cfg["expo"] = 2.2
    cfg["palette"] = "Spectrum"

    os.makedirs(os.path.join(STAGING, "worlds"), exist_ok=True)

    # roll it out here, both for the thumbnail and to report what it scores
    kern = fl.bake_from_config(cfg["kernels"], C, kr, mip=mip)
    mat = torch.tensor(cfg["mat"], dtype=torch.float64).reshape(C, C)
    rho = torch.tensor(tgt.seed_field(masses, grid=N, radius=radius),
                       dtype=torch.float64)[None]
    rho = fl.run(rho, kern, mat, cfg["force"], cfg["repel"], cfg["beta"], a.steps,
                 mip=mip)
    if not a.no_flip:
        tg_shown = tg[:, ::-1, :].copy()      # the target as the flipped world builds it
    loss = ((rho[0, :3] - torch.tensor(tg_shown)) ** 2).mean().item()

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
    print(f"  staging/worlds/{thumb}, and first of "
          f"{len(shelf['setups'])} in staging/worlds.json")


if __name__ == "__main__":
    main()
