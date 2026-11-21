"""Rendering: turn the concentration fields into an RGB image, the way the web page draws them.

Brightness comes from the total mass in a cell, 1 - exp(-total * expo), and the hue from the
colours present, weighted by the square of their concentration so the dominant colour wins.
"""
import sys
import numpy as np
from PIL import Image


def palette(C):
    """Spectrum palette: C fully saturated hues spread around the colour wheel."""
    out = []
    for i in range(C):
        h = (i / C % 1) * 6; x = 1 - abs(h % 2 - 1)
        out.append([[1, x, 0], [x, 1, 0], [0, 1, x], [0, x, 1], [x, 0, 1], [1, 0, x]][int(h) % 6])
    return np.array(out, float)


def to_rgb(rho, expo=2.2):
    """rho: (C, N, N) concentrations -> (N, N, 3) floats in [0, 1]."""
    m = np.maximum(rho, 0); pal = palette(len(rho))
    total = m.sum(0); w = m * m
    hue = np.einsum("cyx,ck->yxk", w, pal) / np.maximum(w.sum(0), 1e-12)[..., None]
    a = 1 - np.exp(-total * expo)
    return np.clip(hue * a[..., None] + np.array([0.016, 0.024, 0.040]), 0, 1)


def to_image(rho, expo=2.2, scale=1):
    img = Image.fromarray((to_rgb(rho, expo) * 255).astype(np.uint8))
    return img.resize((img.width * scale, img.height * scale), Image.NEAREST) if scale != 1 else img


def strip(frames, expo=2.2, gap=4):
    """Several states side by side, for a rollout figure."""
    ims = [to_image(f, expo) for f in frames]
    W = sum(i.width for i in ims) + gap * (len(ims) - 1)
    out = Image.new("RGB", (W, ims[0].height), (4, 6, 10)); x = 0
    for i in ims:
        out.paste(i, (x, 0)); x += i.width + gap
    return out


if __name__ == "__main__":
    # python -m fieldlife.render data/worlds.json W1 --steps 0,100,300,1000,2000 --out w1.png
    # python -m fieldlife.render data/gliders.json B3 --steps 0,100,200,300 --out b3.png
    import argparse
    from .model import World, Sim, load_setups, find_setup
    p = argparse.ArgumentParser()
    p.add_argument("file"); p.add_argument("which", help="W<n>, B<n>, an index, or a name")
    p.add_argument("--steps", default="0,100,300", help="steps at which to take a frame")
    p.add_argument("--seed", type=int, default=0); p.add_argument("--noise", action="store_true", help="start from noise instead of a disc")
    p.add_argument("--out", default="out.png")
    a = p.parse_args()
    s = find_setup(load_setups(a.file), a.which)
    w = World.from_config(s["cfg"]); sim = Sim(w)
    if s["cfg"].get("seedPattern"):
        sim.seed_pattern(s["cfg"]["seedPattern"])
    elif a.noise:
        sim.seed_noise(np.random.default_rng(a.seed))
    else:
        sim.seed_disc(np.random.default_rng(a.seed))
    steps = [int(x) for x in a.steps.split(",")]; frames = []; t = 0
    for st in steps:
        sim.run(st - t); t = st; frames.append(sim.rho.copy())
        print("step", st, "mass", sim.mass().sum())
    strip(frames).save(a.out); print("wrote", a.out)
