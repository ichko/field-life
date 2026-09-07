"""Look at what grew, in the emoji's own colours: each species painted the
colour of the cluster it was given."""
import sys, os
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import field_brains as FB
from emoji_gecko import build as build_emoji
from train_emoji_gecko import seed_state
from PIL import Image


def paint(C, cols, gain=0.62):
    a = np.maximum(np.asarray(C), 0)
    a = a/max(a.sum(), 1e-9)*a.shape[-1]*a.shape[-2]*gain
    w = a*a
    hue = np.tensordot(w, cols, ([0], [0]))
    al = 1 - np.exp(-a.sum(0)*2.4)
    return np.clip(hue/np.maximum(w.sum(0), 1e-9)[..., None]*al[..., None]
                   + np.array([.02, .03, .04]), 0, 1)


def main():
    ck = torch.load(sys.argv[1], weights_only=False)
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    stages = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "0,8,18,32,50,80,130").split(",")]
    out = sys.argv[4] if len(sys.argv) > 4 else "emoji_grown.png"
    parts, cols, _ = build_emoji(N, blur=0.8)
    rule = FB.Rule(seed=0)
    with torch.no_grad():
        rule.frq.copy_(ck["frq"]); rule.amp.copy_(ck["amp"])
        for k, v in ck["raw"].items(): rule.raw[k].copy_(v)
    cfg = ck["cfg"]
    print("iter", ck.get("iter"), "loss", round(float(ck.get("loss", -1)), 4))
    print({k: round(float(v), 3) for k, v in rule.knobs().items()})
    st = seed_state(N, parts, cfg["fill"])
    shots, mx = [], max(stages)
    m0 = float(st.C.sum())
    with torch.no_grad():
        for i in range(mx + 1):
            if i in stages: shots.append(paint(st.C.numpy(), cols))
            if i < mx: st = FB.step(st, rule, cfg)
    print("mass drift %.2e" % abs(float(st.C.sum())/m0 - 1))
    shots.append(paint(parts/parts.sum(), cols))
    row = np.concatenate([np.pad(s, ((1, 1), (1, 1), (0, 0))) for s in shots], 1)
    z = max(2, 900//(N*len(shots)))
    Image.fromarray((row*255).astype(np.uint8)).resize(
        (row.shape[1]*z, row.shape[0]*z), Image.NEAREST).save(out)
    print("stages", stages, "+ target")


main()
