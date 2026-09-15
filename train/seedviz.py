"""What is actually in the seed.

The animal is lopsided -- head at one end, tail at the other -- and the only
thing in the whole run that can tell one end from the other is what the seed has
inside it. The rule is the same everywhere and the world wraps, so if the seed
is isotropic then so is everything that can grow from it, and no amount of
training fixes that. This prints where each channel sits inside the ball and
draws the slices, so the question can be answered by looking rather than
guessed at.

    python3 seedviz.py cuteK900.pt seed.png
"""
import argparse, sys, warnings
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

CELL, GAP, LAB = 54, 3, 14


def pattern(ckpt):
    """The seed as it is placed: (C, D, D, D), each channel holding its mass."""
    sd = torch.load(ckpt, map_location="cpu")
    pat = F.softplus(sd["seed_raw"])*sd["seed_ball"]
    pat = pat/pat.sum((1, 2, 3), keepdim=True).clamp_min(1e-9)
    return pat.numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt"); ap.add_argument("out", nargs="?", default="seed.png")
    a = ap.parse_args()
    pat = pattern(a.ckpt)
    C, D = pat.shape[0], pat.shape[-1]
    lin = np.arange(D) - (D - 1)/2.0

    # where each channel's mass sits, in cells from the middle of the ball
    cen = np.stack([(pat*lin.reshape(s)).sum((1, 2, 3))
                    for s in ((-1, 1, 1), (1, -1, 1), (1, 1, -1))], -1)
    off = np.linalg.norm(cen, axis=-1)
    print("channel offsets from the centre of the seed, in cells:")
    for c in range(C):
        print("  %d  %6.3f   (z %+.2f  y %+.2f  x %+.2f)"
              % (c, off[c], cen[c, 0], cen[c, 1], cen[c, 2]))
    print("mean %.3f, largest %.3f, spread between channels %.3f cells"
          % (off.mean(), off.max(),
             float(np.linalg.norm(cen - cen.mean(0), axis=-1).mean())))

    zs = list(range(D))
    W = len(zs)*(CELL + GAP) + GAP + 40
    H = C*(CELL + GAP) + GAP + LAB
    img = np.full((H, W, 3), 0.08)
    for c in range(C):
        lim = float(pat[c].max())
        for i, z in enumerate(zs):
            sl = np.clip(pat[c, z]/max(lim, 1e-12), 0, 1)**0.6
            sl = np.repeat(np.repeat(sl, CELL//D + 1, 0), CELL//D + 1, 1)[:CELL, :CELL]
            y0 = LAB + GAP + c*(CELL + GAP)
            x0 = GAP + 40 + i*(CELL + GAP)
            img[y0:y0 + CELL, x0:x0 + CELL] = sl[..., None]*np.array([0.55, 0.85, 0.65])
    pic = Image.fromarray((img*255).astype(np.uint8))
    d = ImageDraw.Draw(pic)
    d.text((4, 3), "off", fill=(190, 195, 205))
    for i, z in enumerate(zs):
        d.text((GAP + 40 + i*(CELL + GAP) + 2, 3), "z%d" % z, fill=(150, 156, 166))
    for c in range(C):
        d.text((4, LAB + GAP + c*(CELL + GAP) + CELL//2 - 5), "%.2f" % off[c],
               fill=(190, 195, 205))
    pic.save(a.out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
