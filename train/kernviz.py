"""Look at the kernels the network draws: slices through each channel's stencil.

Red is attraction, blue repulsion, and the panel is the same size for every
channel, so a channel with a short reach shows as a small mark in a large frame
-- which is the point of giving each one its own radius.

Only the bank is loaded, not the field around it. The kernels are what this
draws, and rebuilding a whole Field3D to reach them means a checkpoint trained
with a different seed radius or a different target cannot be looked at at all.
"""
import argparse, math, sys, warnings
import numpy as np
import torch
from PIL import Image, ImageDraw

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
from kernel_cppn import KernelCPPN

CELL, GAP, LAB = 76, 4, 15


def load_bank(ckpt, orders=3, rings=5):
    """Rebuild just the KernelCPPN, with its shape read off the checkpoint."""
    sd = {k[len("kern."):]: v for k, v in torch.load(ckpt, map_location="cpu").items()
          if k.startswith("kern.")}
    if not sd:
        raise SystemExit("%s holds no cppn bank -- was it fitted with --kernel cppn?" % ckpt)
    C, emb = sd["embed"].shape
    K = (round(sd["off"].shape[0]**(1/3)) - 1)//2
    axes, hidden = sd["axis"].shape[0], sd["head"].shape[1]
    nin = sd["net.0.weight"].shape[1]
    want = 2 + 3 + axes*orders + 2*rings + emb
    if nin != want:
        raise SystemExit("cannot tell orders from rings: the net takes %d inputs, "
                         "orders=%d rings=%d would give %d. Pass --orders/--rings."
                         % (nin, orders, rings, want))
    m = KernelCPPN(C, K=K, axes=axes, orders=orders, rings=rings, emb=emb, hidden=hidden)
    m.load_state_dict(sd)
    return m


def colour(sl, lim):
    """Diverging: red positive, blue negative, dark at zero."""
    t = np.clip(sl/max(lim, 1e-12), -1, 1)
    pos = np.clip(t, 0, 1)[..., None]*np.array([1.00, 0.42, 0.24])
    neg = np.clip(-t, 0, 1)[..., None]*np.array([0.30, 0.62, 1.00])
    return np.clip((pos + neg)**0.75, 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("out", nargs="?", default="kernels.png")
    ap.add_argument("--orders", type=int, default=3)
    ap.add_argument("--rings", type=int, default=5)
    a = ap.parse_args()

    m = load_bank(a.ckpt, a.orders, a.rings)
    with torch.no_grad():
        k = m.bake()[:, 0].numpy()
        reach = m.reach().numpy()
    C, D = k.shape[0], k.shape[-1]
    zs = [z for z in (D//2 - 4, D//2 - 2, D//2, D//2 + 2, D//2 + 4) if 0 <= z < D]

    W = len(zs)*(CELL + GAP) + GAP + 54
    H = C*(CELL + GAP) + GAP + LAB
    img = np.full((H, W, 3), 0.09)
    for c in range(C):
        lim = float(np.abs(k[c]).max())
        for i, z in enumerate(zs):
            sl = np.repeat(np.repeat(k[c, z], CELL//D + 1, 0), CELL//D + 1, 1)[:CELL, :CELL]
            y0 = LAB + GAP + c*(CELL + GAP)
            x0 = GAP + 54 + i*(CELL + GAP)
            img[y0:y0 + CELL, x0:x0 + CELL] = colour(sl, lim)
    pic = Image.fromarray((img*255).astype(np.uint8))
    d = ImageDraw.Draw(pic)
    for i, z in enumerate(zs):
        d.text((GAP + 54 + i*(CELL + GAP) + 2, 3), "z%+d" % (z - D//2), fill=(190, 195, 205))
    d.text((4, 3), "reach", fill=(190, 195, 205))
    for c in range(C):
        d.text((4, LAB + GAP + c*(CELL + GAP) + CELL//2 - 5),
               "%.0f cells" % (2*reach[c]*m.K), fill=(190, 195, 205))
    pic.save(a.out)
    print("wrote", a.out, "reach across the bank, in full cells:",
          [round(float(2*r*m.K), 1) for r in reach])


if __name__ == "__main__":
    main()
