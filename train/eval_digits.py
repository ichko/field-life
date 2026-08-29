"""
A proper accuracy number for a trained world, on digits it has never seen.

    python3 train/eval_digits.py --name dig10 --images 1000

The trainer scores 64 test images at every checkpoint, which is cheap and noisy
-- plus or minus about six points at these rates -- and it picks its own best
checkpoint on that same small sample, so the reported best is optimistically
biased. This runs the saved world over a large held-out sample once, and also
prints the per-class breakdown, because a ten-way accuracy hides whether the
field has learnt ten classes or three of them very well.
"""

import argparse, json, os, sys
import numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import digits as dg, fieldlife as fl
from train_digits import Task, roll

HERE = os.path.dirname(os.path.abspath(__file__))


class Replay:
    """The world as a preset, stepped through the same path index.html would."""

    def __init__(self, cfg, C):
        KR = max(3, min(fl.KMAX, round(max(k["R"] for k in cfg["kernels"][:C]))))
        self.kern = fl.bake_from_config(cfg["kernels"], C, KR, dtype=torch.float32)
        self.mat = torch.tensor(cfg["mat"]).reshape(C, C).float()
        self.f, self.r, self.b = cfg["force"], cfg["repel"], cfg["beta"]

    def rollout(self, rho, steps, kern=None):
        return fl.run(rho, self.kern, self.mat, self.f, self.r, self.b, steps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="dig10")
    ap.add_argument("--best", action="store_true")
    ap.add_argument("--images", type=int, default=1000)
    ap.add_argument("--horizons", default="32,64,128,256")
    ap.add_argument("--classes", type=int, default=10)
    ap.add_argument("--static", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=50)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    run = os.path.join(HERE, "runs", a.name)
    cfg = json.load(open(os.path.join(
        run, "preset-best.json" if a.best else "preset.json")))
    C = cfg["C"]
    t = cfg.get("_task", {})
    task = Task(C, a.classes, cfg["N"], t.get("digit", dg.DIGIT), a.static,
                t.get("ring", dg.RING))
    world = Replay(cfg, C)

    siren = None
    ck = os.path.join(run, "ckpt.pt")
    if os.path.exists(ck):
        st = torch.load(ck, weights_only=False)
        if st.get("siren"):
            siren = dg.SirenSeed(cfg["N"], C - a.static - 1, 32, 2)
            siren.load_state_dict(st["siren"])

    x, y = dg.load("test")
    keep = y < a.classes
    x, y = x[keep][:a.images], y[keep][:a.images]
    hs = [int(v) for v in a.horizons.split(",")]
    rng = np.random.default_rng(1234)

    hits = np.zeros(len(hs))
    per = np.zeros((len(hs), a.classes)); tot = np.zeros(a.classes)
    conf = np.zeros((a.classes, a.classes), dtype=int)
    with torch.no_grad():
        for lo in range(0, len(y), a.chunk):
            seeds, _, lab = task.build(x[lo:lo + a.chunk], y[lo:lo + a.chunk],
                                       rng, siren)
            rho, done = seeds.clone(), 0
            for c in lab.tolist():
                tot[c] += 1
            for j, h in enumerate(hs):
                rho = roll(world, task, rho, h - done)
                done = h
                pred = task.scores(rho).argmax(1)
                ok = (pred == lab).numpy()
                hits[j] += ok.sum()
                for c, o in zip(lab.tolist(), ok):
                    per[j, c] += o
                if j == len(hs) - 1:
                    for c, p in zip(lab.tolist(), pred.tolist()):
                        conf[c, p] += 1

    n = len(y)
    print(f"{a.name}{' (best ckpt)' if a.best else ''}: {n} held-out digits, "
          f"{a.classes} classes, chance {1/a.classes:.3f}\n")
    for j, h in enumerate(hs):
        print(f"  {h:>4} steps   accuracy {hits[j]/n:.3f}")
    print(f"\nper class at {hs[-1]} steps (n in brackets):")
    for c in range(a.classes):
        print(f"  {c}: {per[-1, c]/max(tot[c],1):.3f}  [{int(tot[c])}]")
    print("\nconfusion, rows = truth, cols = predicted:")
    print("     " + " ".join(f"{c:>4}" for c in range(a.classes)))
    for c in range(a.classes):
        print(f"  {c}: " + " ".join(f"{v:>4}" for v in conf[c]))


if __name__ == "__main__":
    main()
