"""Run a fitted rule from its seed and draw what grew.

    python3 look.py fit.pt 20,34,48 40 10 5 12 out.png --target cute
"""
import argparse, importlib, sys, warnings
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import view
from field3d import Field3D

# One colour per part, in the order the target cuts them. A target that has its
# own palette gets to use it, so what grew and what was aimed at are drawn in
# the same colours and can be put side by side.
FALLBACK = np.array([[0.82, 0.30, 0.16], [0.96, 0.78, 0.28], [0.16, 0.52, 0.30],
                     [0.86, 0.84, 0.62], [0.36, 0.26, 0.58], [0.20, 0.62, 0.72],
                     [0.90, 0.45, 0.60], [0.55, 0.70, 0.22]], np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt"); ap.add_argument("at")
    ap.add_argument("N", type=int); ap.add_argument("C", type=int)
    ap.add_argument("S", type=int); ap.add_argument("T", type=int)
    ap.add_argument("out")
    ap.add_argument("--target", default="cute")
    ap.add_argument("--kernel", default="gauss", choices=("gauss", "cppn"))
    ap.add_argument("--K", type=int, default=7)
    ap.add_argument("--hard", action="store_true", help="the page's clamp, not the fit's")
    a = ap.parse_args()
    at = [int(v) for v in a.at.split(",")]
    torch.set_num_threads(4)

    m = Field3D(C=a.C, S=a.S, T=a.T, N=a.N, kernel=a.kernel, K=a.K)
    m.load_state_dict(torch.load(a.ckpt, map_location="cpu"))
    if a.hard: m.soft = False

    mod = importlib.import_module(a.target)
    parts, _ = mod.build_parts(a.N)
    P = parts.shape[0]
    PAL = np.asarray(getattr(mod, "PAL", FALLBACK), np.float32)
    if len(PAL) < P:
        PAL = np.concatenate([PAL, FALLBACK[:P - len(PAL)]], 0)
    vis = torch.from_numpy(parts).unsqueeze(0).sum((0, 2, 3, 4))
    masses = torch.cat([vis, F.softplus(m.seed_mass.detach())[P:]])
    with torch.no_grad():
        _, sn = m.run(masses, max(at), keep=tuple(at))

    W, tiles = 250, []
    for k in at:
        p = sn[k][0, :P].clamp(min=0).numpy()
        tiles.append((np.einsum("pzyx,pc->zyxc", p, PAL[:P]), np.clip(p.sum(0), 0, 1)))
    rows = [[view.render(r, o, y, q, W) for r, o in tiles]
            for y, q in ((0.9, 0.28), (0.0, 0.06), (0.0, 1.35))]
    n = len(at)
    sheet = np.full((len(rows)*(W + 6) - 6, n*(W + 6) - 6, 3), 26, np.uint8)
    for j, row in enumerate(rows):
        for i, t in enumerate(row):
            sheet[j*(W + 6):j*(W + 6) + W, i*(W + 6):i*(W + 6) + W] = t
    Image.fromarray(sheet).save(a.out)
    print("peak %.2f" % float(sn[at[-1]][:, :P].max()),
          "mass", [round(float(x), 1) for x in sn[at[-1]][0, :P].sum((1, 2, 3))])


if __name__ == "__main__":
    main()
