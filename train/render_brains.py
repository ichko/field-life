"""Look at what grew: the seed, the run, and the target beside it."""
import sys, os
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import field_brains as FB
from gecko2d import build2d
from PIL import Image

PAL = np.array([[1, .75, .15], [.25, .85, .35], [.35, .6, 1], [1, .4, .55]], np.float32)

def paint(C):
    a = np.maximum(np.asarray(C), 0)
    a = a/max(a.sum(), 1e-9)*a.shape[-1]**2*0.6
    w = a*a
    hue = np.tensordot(w, PAL, ([0], [0]))
    al = 1 - np.exp(-a.sum(0)*2.2)
    return np.clip(hue/np.maximum(w.sum(0), 1e-9)[..., None]*al[..., None]
                   + np.array([.02, .03, .04]), 0, 1)

def main():
    ck = torch.load(sys.argv[1], weights_only=False)
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    stages = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "0,8,16,28,44,70").split(",")]
    parts, _ = build2d(N, blur=1.0)
    rule = FB.Rule(seed=0)
    with torch.no_grad():
        rule.frq.copy_(ck["frq"]); rule.amp.copy_(ck["amp"])
        for k, v in ck["raw"].items(): rule.raw[k].copy_(v)
    cfg = ck["cfg"]
    print("iter", ck.get("iter"), "loss", round(ck.get("loss", -1), 5))
    print({k: round(float(v), 3) for k, v in rule.knobs().items()})
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from train_gecko_brains import seed_state
    gen = torch.Generator().manual_seed(3)
    st = seed_state(N, parts, cfg["fill"], gen)
    shots, mx = [], max(stages)
    with torch.no_grad():
        for i in range(mx + 1):
            if i in stages: shots.append(paint(st.C.numpy()))
            if i < mx: st = FB.step(st, rule, cfg)
    tgt = parts/parts.sum()
    shots.append(paint(tgt))
    row = np.concatenate([np.pad(s, ((1, 1), (1, 1), (0, 0))) for s in shots], 1)
    z = max(1, 640//(N*len(shots)))
    Image.fromarray((row*255).astype(np.uint8)).resize(
        (row.shape[1]*z*2, row.shape[0]*z*2), Image.NEAREST).save(sys.argv[4] if len(sys.argv) > 4 else "grown.png")
    print("stages", stages, "+ target")

main()
