"""Run a fitted rule from its seed and look at what grows."""
import argparse, sys, warnings
import numpy as np
import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import gecko, view
from field3d import Field3D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--N", type=int, default=40)
    ap.add_argument("--C", type=int, default=8)
    ap.add_argument("--T", type=int, default=8)
    ap.add_argument("--S", type=int, default=3)
    ap.add_argument("--seedR", type=float, default=3.5)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--at", default="")
    ap.add_argument("--out", default="fit-views.png")
    ap.add_argument("--hard", action="store_true", help="the page's hard clamp")
    a = ap.parse_args()
    torch.set_num_threads(4)

    m = Field3D(C=a.C, S=a.S, T=a.T, N=a.N, seedR=a.seedR)
    m.load_state_dict(torch.load(a.ckpt, map_location="cpu"))
    m.soft = not a.hard
    rgb_t, occ_t = gecko.build(a.N)
    vis = torch.from_numpy(rgb_t).permute(3, 0, 1, 2).unsqueeze(0).sum((0, 2, 3, 4))
    masses = torch.cat([vis, F.softplus(m.seed_mass.detach())[3:]])

    at = [int(v) for v in a.at.split(",")] if a.at else [a.steps]
    with torch.no_grad():
        rho, snaps = m.run(masses, max(at), keep=tuple(at))

    tiles = []
    for k in at:
        v = snaps[k][0, :3].clamp(min=0).permute(1, 2, 3, 0).numpy()
        o = np.clip(v.sum(-1)/1.1, 0, 1)
        tiles.append((v, o, k))
    W = 260
    imgs = [view.render(v, o, 0.0, 1.45, W) for v, o, _ in tiles] \
         + [view.render(v, o, 0.7, 0.35, W) for v, o, _ in tiles]
    n = len(at)
    sheet = np.full((2*W + 6, n*W + 6*(n - 1), 3), 26, np.uint8)
    for i in range(n):
        sheet[:W, i*(W + 6):i*(W + 6) + W] = imgs[i]
        sheet[W + 6:, i*(W + 6):i*(W + 6) + W] = imgs[n + i]
    from PIL import Image
    Image.fromarray(sheet).save(a.out)
    print("steps", at, "-> ", a.out,
          "peak %.2f" % float(snaps[at[-1]][:, :3].max()),
          "mass", [round(float(x), 1) for x in snaps[at[-1]][0, :3].sum((1, 2, 3))])


if __name__ == "__main__":
    main()
