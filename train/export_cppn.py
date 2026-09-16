"""Write a fitted ring-bank rule out in the form a page can run.

The page's own kernel is a difference of two Gaussians, which is three separable
blurs and a subtraction. This one is a bank of dense stencils a network drew, so
what gets written out is the stencils themselves, baked.

They are cheaper than they look. Each channel has its own reach and most of them
are small -- a radius of one or two half-pitch cells where the widest is six --
so the channels are SORTED by radius and packed in fours, and each group of four
is convolved over only as many taps as its widest member needs. Sorted, the four
groups cost about five thousand taps a voxel between them; unsorted, every group
would pay the widest channel's price and it would be four times that.

    python3 export_cppn.py cuteE_last.pt --N 40 --out ../staging/gecko-fit.json
"""
import argparse, json
import numpy as np
import torch

import cute
from field3d import load_field, gauss1d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--N", type=int, default=40)
    ap.add_argument("--target", default="cute")
    ap.add_argument("--steps", type=int, default=56)
    ap.add_argument("--eps", type=float, default=1e-5)
    ap.add_argument("--out", default="../staging/gecko-fit.json")
    a = ap.parse_args()

    m = load_field(a.ckpt, a.N)
    m.soft = False                       # the page clamps hard; check it here too
    C, K = m.C, m.kern.K
    parts, _ = __import__(a.target).build_parts(a.N)
    P = parts.shape[0]
    vis = torch.from_numpy(parts).sum((1, 2, 3))
    masses = torch.cat([vis, torch.nn.functional.softplus(m.seed_mass.detach())[P:]])
    dscale = float(a.N**3)/float(masses.sum())

    with torch.no_grad():
        bank = m.kern.bake()[:, 0].numpy()             # (C, S, S, S)
        pat = torch.nn.functional.softplus(m.seed_raw)*m.seed_ball
        pat = pat/pat.sum((1, 2, 3), keepdim=True).clamp_min(1e-9)
        pat = (pat*masses.view(-1, 1, 1, 1)).numpy()
        mat = m.mat.numpy()
        sig0 = float(torch.exp(m.log_sig)[0])
        kb = min(max(2, int(np.ceil(4.0*sig0))), a.N//4 - 1)
        blur = gauss1d(torch.tensor(sig0), kb, torch.device("cpu"), torch.float32).numpy()

    # each channel's own radius, from where its weights actually stop
    d = np.arange(-K, K + 1)
    gz, gy, gx = np.meshgrid(d, d, d, indexing="ij")
    rad = np.zeros(C, np.int32)
    for c in range(C):
        nz = np.abs(bank[c]) > a.eps
        rad[c] = max(int(np.abs(gz[nz]).max()), int(np.abs(gy[nz]).max()),
                     int(np.abs(gx[nz]).max())) if nz.any() else 0

    # sort by radius, then take them four at a time, so a group of tight kernels
    # is not dragged over a wide one's neighbourhood
    order = np.argsort(rad, kind="stable")
    G = (C + 3)//4
    groups = []
    for g in range(G):
        ch = order[4*g:4*g + 4]
        R = int(rad[ch].max())
        w = np.zeros(((2*R + 1)**3, 4), np.float32)
        for j, c in enumerate(ch):
            sub = bank[c][K - R:K + R + 1, K - R:K + R + 1, K - R:K + R + 1]
            w[:, j] = sub.reshape(-1)                   # z slowest, x fastest
        groups.append({"radius": R,
                       "channels": [int(v) for v in ch],
                       "w": [round(float(v), 7) for v in w.reshape(-1)]})
    taps = sum((2*g["radius"] + 1)**3 for g in groups)
    print("channel radii", list(rad), "-> groups",
          [(g["radius"], g["channels"]) for g in groups])
    print("taps per voxel: %d sorted, %d if every group paid the widest" %
          (taps, G*(2*int(rad.max()) + 1)**3))

    # the matrix, permuted to match, so the page never has to unpermute
    perm = order
    matp = mat[np.ix_(perm, perm)]

    out = {
        "kind": "gecko-fit-v1",
        "N": a.N, "C": C, "P": P, "steps": a.steps,
        "order": [int(v) for v in perm],
        "groups": groups,
        "mat": [[round(float(v), 6) for v in row] for row in matp],
        "force": round(float(torch.exp(m.log_force))*dscale, 6),
        "repel": round(float(m.repel)*dscale, 6),
        "beta": round(float(torch.exp(m.log_beta)), 6),
        "crowdBlur": [round(float(v), 7) for v in blur],
        "seedHalf": m.seed_half,
        "seedMass": [round(float(masses[c]), 4) for c in perm],
        # plain [channel][z][y][x] with x fastest, permuted like everything else
        "seedPattern": [round(float(v), 6) for v in pat[perm].reshape(-1)],
        "palette": [[round(float(v), 4) for v in col] for col in cute.PAL[:P]],
    }
    with open(a.out, "w") as f:
        json.dump(out, f)
    print("wrote", a.out,
          "force %.3f repel %.4f beta %.3f" % (out["force"], out["repel"], out["beta"]))


if __name__ == "__main__":
    main()
