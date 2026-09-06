"""Write a fitted rule out as a preset the page can load.

Every other preset carries a brain SEED, and the page draws its weights from
that -- which is why a preset is three numbers and replays exactly. A fitted
brain has no seed: its weights are where the fit put them, so they have to
travel with it. 4*10*(29+13) = 1680 numbers, rounded to four figures and written
as a flat array, which is about 12 KB of the page.
"""
import argparse, json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import field_brains as FB
from gecko2d import build2d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--N", type=int, default=64)
    ap.add_argument("--name", default="gecko")
    ap.add_argument("--out", default="-")
    a = ap.parse_args()
    ck = torch.load(a.ckpt, weights_only=False)
    rule = FB.Rule(seed=0)
    with torch.no_grad():
        rule.frq.copy_(ck["frq"]); rule.amp.copy_(ck["amp"])
        for k, v in ck["raw"].items(): rule.raw[k].copy_(v)
        kn = {k: float(v) for k, v in rule.knobs().items()}
    cfg = ck["cfg"]

    # the page's own packing: frq then amp, four to a vec4, species-major
    frq = rule.frq.detach().numpy().reshape(-1)
    amp = rule.amp.detach().numpy().reshape(-1)
    r4 = lambda v: [float(f"{x:.4g}") for x in v]

    # the seed the fit was given, as fractions and centres in grid units, so the
    # page can lay the same blobs down at any resolution
    parts, _ = build2d(a.N, blur=1.0)
    P = torch.as_tensor(parts)
    tot = float(P.sum())
    blobs = []
    yy, xx = np.mgrid[0:a.N, 0:a.N]
    for s in range(FB.NS):
        w = parts[s]; ws = max(w.sum(), 1e-9)
        blobs.append(dict(frac=round(float(w.sum()/tot), 4),
                          x=round(float((w*xx).sum()/ws)/a.N, 4),
                          y=round(float((w*yy).sum()/ws)/a.N, 4)))
    out = dict(name=a.name, fitted=True,
               p=dict(shape=cfg["shape"], mirror=cfg["mirror"], dist=cfg["dist"],
                      size=cfg["size"], spread=cfg["spread"], feather=cfg["feather"],
                      crowd=cfg["crowd"], fill=cfg["fill"], sweep=0, view=1,
                      **{k: round(v, 4) for k, v in kn.items()}),
               blobs=blobs, frq=r4(frq), amp=r4(amp))
    js = json.dumps(out, separators=(",", ":"))
    if a.out == "-":
        print(js)
    else:
        open(a.out, "w").write(js)
        print(f"{a.out}: {len(js)/1024:.1f} KB, loss {ck.get('loss'):.4f}, iter {ck.get('iter')}")
        print("knobs", {k: round(v, 3) for k, v in kn.items()})


main()
