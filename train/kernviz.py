"""Look at the kernels the network draws: slices through each channel's stencil.

Red is attraction, blue repulsion, and the panel is the same size for every
channel, so a channel with a short reach shows as a small mark in a large frame
-- which is the point of giving each one its own radius.
"""
import sys, warnings
import numpy as np
import torch
from PIL import Image, ImageDraw

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
from field3d import Field3D

CELL, GAP = 76, 4


def colour(sl, lim):
    """Diverging: red positive, blue negative, dark at zero."""
    t = np.clip(sl/max(lim, 1e-12), -1, 1)
    pos = np.clip(t, 0, 1)[..., None]*np.array([1.00, 0.42, 0.24])
    neg = np.clip(-t, 0, 1)[..., None]*np.array([0.30, 0.62, 1.00])
    return np.clip((pos + neg)**0.75, 0, 1)


def main():
    ck = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] != "-" else None
    out = sys.argv[2] if len(sys.argv) > 2 else "kernels.png"
    C, K = 8, 7
    m = Field3D(C=C, S=5, T=12, N=40, kernel="cppn", K=K)
    if ck:
        m.load_state_dict(torch.load(ck, map_location="cpu"))
    with torch.no_grad():
        k = m.kern.bake()[:, 0].numpy()               # (C, D, D, D)
        reach = m.kern.reach().numpy()
    D = k.shape[-1]
    zs = [D//2 - 4, D//2 - 2, D//2, D//2 + 2, D//2 + 4]
    zs = [z for z in zs if 0 <= z < D]
    lab = 15
    W = len(zs)*(CELL + GAP) + GAP + 54
    H = C*(CELL + GAP) + GAP + lab
    img = np.full((H, W, 3), 0.09)
    for c in range(C):
        lim = float(np.abs(k[c]).max())
        for i, z in enumerate(zs):
            sl = k[c, z]
            sl = np.repeat(np.repeat(sl, CELL//D + 1, 0), CELL//D + 1, 1)[:CELL, :CELL]
            y0 = lab + GAP + c*(CELL + GAP)
            x0 = GAP + 54 + i*(CELL + GAP)
            img[y0:y0 + CELL, x0:x0 + CELL] = colour(sl, lim)
    pic = Image.fromarray((img*255).astype(np.uint8))
    d = ImageDraw.Draw(pic)
    for i, z in enumerate(zs):
        d.text((GAP + 54 + i*(CELL + GAP) + 2, 3), "z%+d" % (z - D//2), fill=(190, 195, 205))
    d.text((4, 3), "reach", fill=(190, 195, 205))
    for c in range(C):
        y0 = lab + GAP + c*(CELL + GAP)
        d.text((4, y0 + CELL//2 - 5), "%.0f cells" % (2*reach[c]*K), fill=(190, 195, 205))
    pic.save(out)
    print("wrote", out, "reach in full cells:",
          [round(float(2*r*K), 1) for r in reach])


if __name__ == "__main__":
    main()
