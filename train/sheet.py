"""Contact sheets: the target, and what each stage of the fit actually grew."""
import sys, warnings
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import gecko, view
from field3d import Field3D

torch.set_num_threads(4)
W = 240
PAL = np.array([[0.16, 0.55, 0.20], [0.86, 0.83, 0.55], [0.95, 0.62, 0.20]], np.float32)
VIEWS = [(0.0, 1.45), (0.72, 0.34)]          # from above, and three-quarters


def to_rgb(p, parts):
    """p: (3,N,N,N) -> premultiplied colour and occupancy."""
    p = np.clip(p, 0, None)
    if parts:
        return np.einsum("pzyx,pc->zyxc", p, PAL), np.clip(p.sum(0), 0, 1)
    return p.transpose(1, 2, 3, 0), np.clip(p.sum(0)/1.1, 0, 1)


def run(ckpt, N, C, S, T, at, parts=True):
    m = Field3D(C=C, S=S, T=T, N=N)
    m.load_state_dict(torch.load(ckpt, map_location="cpu"))
    if parts:
        tgt, _ = gecko.build_parts(N)
        vis = torch.from_numpy(tgt).unsqueeze(0).sum((0, 2, 3, 4))
    else:
        rgb, _ = gecko.build(N)
        vis = torch.from_numpy(rgb).permute(3, 0, 1, 2).unsqueeze(0).sum((0, 2, 3, 4))
    masses = torch.cat([vis, F.softplus(m.seed_mass.detach())[3:]])
    out = {}
    with torch.no_grad():
        m.dscale.fill_(float(N**3)/float(masses.sum()))
        rho = m.seed(masses)
        if 0 in at:
            out[0] = to_rgb(rho[0, :3].numpy(), parts)
        for i in range(1, max(at) + 1):
            rho = m.step(rho)
            if i in at:
                out[i] = to_rgb(rho[0, :3].numpy(), parts)
    return out


def grid(cells, labels, path, cols):
    """cells: list of (rgb, occ). Two rows of views per cell row."""
    rows = (len(cells) + cols - 1)//cols
    pad, lab = 6, 20
    sheet = np.full((rows*(2*W + pad + lab) + pad, cols*(W + pad) + pad, 3), 22, np.uint8)
    for i, (rgb, occ) in enumerate(cells):
        r, c = i//cols, i % cols
        x0 = pad + c*(W + pad)
        y0 = pad + r*(2*W + pad + lab) + lab
        for k, (yaw, pit) in enumerate(VIEWS):
            sheet[y0 + k*W:y0 + (k + 1)*W, x0:x0 + W] = view.render(rgb, occ, yaw, pit, W)
    img = Image.fromarray(sheet)
    d = ImageDraw.Draw(img)
    for i, t in enumerate(labels):
        r, c = i//cols, i % cols
        d.text((pad + c*(W + pad) + 3, pad + r*(2*W + pad + lab) + 4), t, fill=(210, 214, 220))
    img.save(path)
    print("wrote", path)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "stages"
    N = 40
    if which == "stages":
        cells, labels = [], []
        parts, occ = gecko.build_parts(N)
        cells.append((np.einsum("pzyx,pc->zyxc", parts, PAL), np.clip(parts.sum(0), 0, 1)))
        labels.append("TARGET  the gecko, in three parts")
        for ck, S, T, pr, lab in [
                ("gecko40_stage1.pt", 3, 8, False, "1  rgb target, sharp"),
                ("geckoB.pt",         5, 12, False, "2  rgb target, softened"),
                ("geckoP_best.pt",    5, 12, True,  "3  parts target"),
                ("geckoQ_last.pt",    5, 12, True,  "4  parts + persistence pool")]:
            r = run(ck, N, 8, S, T, [36], parts=pr)
            cells.append(r[36]); labels.append(lab)
        grid(cells, labels, "progress-stages.png", 5)
    else:
        at = [0, 6, 14, 24, 40, 70]
        r = run("geckoQ_last.pt", N, 8, 5, 12, at, parts=True)
        grid([r[k] for k in at], ["seed"] + ["step %d" % k for k in at[1:]],
             "progress-growth.png", 6)
