"""
An animated GIF of a trained world growing from its seed.

    python3 train/growth_gif.py --run polar3 --steps 160 --out growth.gif

Every frame is one simulation step, so the pace on screen is the pace the
field actually moves at -- which is slow by construction, because MaCE carries
mass one cell per step and the far end of the lizard is 26 cells from the
seed. `--hold` freezes the last frame so a loop does not snap away from the
finished animal.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fieldlife as fl
import render as rn
import target as tgt

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="polar3")
    ap.add_argument("--preset", default=None, help="a preset.json, instead of a run")
    ap.add_argument("--steps", type=int, default=160)
    ap.add_argument("--kr", type=int, default=13)
    ap.add_argument("--scale", type=int, default=6, help="pixels per cell")
    ap.add_argument("--ms", type=int, default=110, help="milliseconds per frame")
    ap.add_argument("--hold", type=int, default=12, help="extra frames on the last one")
    ap.add_argument("--every", type=int, default=1, help="keep every Nth step")
    ap.add_argument("--blend", default="dominant",
                    choices=["dominant", "soft", "additive", "winner", "rgb"],
                    help="how the channels combine, as index.html's Blend control. "
                         "dominant is the simulation's default and shows ALL "
                         "channels; rgb shows only the three visible ones")
    ap.add_argument("--expo", type=float, default=2.2, help="display brightness")
    ap.add_argument("--side-by-side", action="store_true",
                    help="put the target beside it, for comparison")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    path = a.preset or os.path.join(HERE, "runs", a.run, "preset.json")
    cfg = json.load(open(path))
    C, N = cfg["C"], cfg["N"]
    if "_note" in cfg:
        sys.exit("that run is rung 1; its affinity is a network, not this matrix")

    tg = tgt.render_emoji(span=tgt.SPAN, grid=N)
    masses = cfg.get("seedMasses") or tgt.seed_masses(tg, C, rng=np.random.default_rng(0))
    rho = torch.tensor(tgt.seed_field(np.asarray(masses), grid=N), dtype=torch.float64)[None]
    kern = fl.bake_from_config(cfg["kernels"], C, a.kr)
    mat = torch.tensor(cfg["mat"], dtype=torch.float64).reshape(C, C)
    g = (cfg["force"], cfg["repel"], cfg["beta"])

    pal = rn.palette(C)
    # the target is a picture, not a field, so it is drawn as itself either way
    tgt_img = np.clip(tg.transpose(1, 2, 0), 0, 1)
    if a.blend != "rgb":
        tgt_img = np.clip(tgt_img + rn.GROUND, 0, 1)
    frames, losses = [], []
    for i in range(a.steps + 1):
        if i:
            rho = fl.step(rho, kern, mat, *g)
        losses.append(((rho[0, :3] - torch.tensor(tg)) ** 2).mean().item())
        if i % a.every:
            continue
        v = rn.blend(rho[0].numpy(), pal, a.expo, a.blend, ground=a.blend != "rgb")
        if a.side_by_side:
            gap = np.zeros((N, 2, 3))
            v = np.concatenate([tgt_img, gap, v], axis=1)
        im = Image.fromarray((v * 255).astype(np.uint8))
        im = im.resize((im.width * a.scale, im.height * a.scale), Image.NEAREST)
        # the step count, so the pace is readable rather than guessed at
        ImageDraw.Draw(im).text((6, im.height - 14), f"step {i}", fill=(140, 150, 160))
        frames.append(im)

    frames += [frames[-1]] * a.hold
    out = a.out or os.path.join(HERE, "runs", a.run, "growth.gif")
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=a.ms, loop=0, optimize=True)

    best = int(np.argmin(losses))
    print(f"{len(frames)} frames, {a.ms}ms each, blend {a.blend} "
          f"over {'3 visible' if a.blend == 'rgb' else f'all {C}'} channels -> {out}")
    print(f"  loss is lowest at step {best} ({losses[best]:.5f}); "
          f"at the end, step {a.steps}, it is {losses[-1]:.5f}")
    print(f"  size {os.path.getsize(out)/1e6:.2f} MB")


if __name__ == "__main__":
    main()
