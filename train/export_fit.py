"""Write a fitted rule out in the form the volume page reads.

Two changes of units on the way out. The fit divides the affinity by the mean
density so that force and repel mean the same thing whatever the animal weighs;
the page has no such notion, so the constant is folded into the two numbers
here. And the fit works in (z, y, x) because that is how torch indexes a volume,
while the page works in (x, y, z), so every displacement is reversed.
"""
import argparse, json, math
import torch
import torch.nn.functional as F

import gecko
from field3d import Field3D, gauss1d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--N", type=int, default=40)
    ap.add_argument("--C", type=int, default=8)
    ap.add_argument("--T", type=int, default=8)
    ap.add_argument("--S", type=int, default=3)
    ap.add_argument("--seedR", type=float, default=3.5)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--out", default="gecko-fit.json")
    a = ap.parse_args()

    m = Field3D(C=a.C, S=a.S, T=a.T, N=a.N, seedR=a.seedR)
    m.load_state_dict(torch.load(a.ckpt, map_location="cpu"))

    rgb, occ = gecko.build(a.N)
    vis = torch.from_numpy(rgb).permute(3, 0, 1, 2).unsqueeze(0).sum((0, 2, 3, 4))
    masses = torch.cat([vis, F.softplus(m.seed_mass.detach())[3:]])
    dscale = float(a.N**3)/float(masses.sum())

    with torch.no_grad():
        amp = m.amp/m.amp.abs().sum(1, keepdim=True).clamp_min(1e-6)
        sig = torch.exp(m.log_sig)
        # the stencil each width needs, the way the model picks it
        half = [min(max(2, int(math.ceil(4.0*float(s)))), a.N//4 - 1) for s in sig]
        pat = F.softplus(m.seed_raw)*m.seed_ball
        pat = pat/pat.sum((1, 2, 3), keepdim=True).clamp_min(1e-9)*masses.view(-1, 1, 1, 1)

    out = {
        "kind": "volume-fit-v1",
        "N": a.N, "C": a.C, "S": a.S, "T": a.T, "steps": a.steps,
        "sigma": [round(float(s), 5) for s in sig],
        "stencil": half,
        "termSig": list(m.term_sig),
        # (z,y,x) in the fit, (x,y,z) on the page
        "amp": [[round(float(v), 6) for v in row] for row in amp],
        "off": [[[round(float(o[2]), 5), round(float(o[1]), 5), round(float(o[0]), 5)]
                 for o in row] for row in m.off.detach()],
        "mat": [[round(float(v), 6) for v in row] for row in m.mat.detach()],
        "force": round(float(torch.exp(m.log_force))*dscale, 6),
        "repel": round(float(m.repel)*dscale, 6),
        "beta": round(float(torch.exp(m.log_beta)), 6),
        "seedR": a.seedR,
        "seedHalf": m.seed_half,
        "seedMass": [round(float(v), 4) for v in masses],
        # the pattern, flattened x fastest, as the page uploads it
        "seedPattern": [round(float(v), 6)
                        for v in pat.permute(0, 3, 2, 1).reshape(-1)],
    }
    with open(a.out, "w") as f:
        json.dump(out, f)
    print("wrote", a.out,
          "force %.3f repel %.4f beta %.3f" % (out["force"], out["repel"], out["beta"]),
          "sigma", out["sigma"], "stencil", out["stencil"])


if __name__ == "__main__":
    main()
