"""
Every channel on its own row, so 'the hidden channels' stop being a rumour.

    python3 train/channels_png.py --name dig10

The composite in progress.png folds all the chemicals into one blue component,
which is fine while they are structured and useless once they are not -- five
uniform channels and one uniform channel look identical there. This draws each
channel separately, per column a rollout length, with each panel normalised by
its own maximum so a faint but structured channel is still visible.
"""

import argparse, json, os, sys
import numpy as np, torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import digits as dg, fieldlife as fl
from train_digits import Task

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="dig10")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--steps", default="0,4,8,16,32,64,128")
    ap.add_argument("--classes", type=int, default=10)
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    torch.set_num_threads(4)

    run = os.path.join(HERE, "runs", a.name)
    cfg = json.load(open(os.path.join(run, "preset.json")))
    C = cfg["C"]
    KR = max(3, min(fl.KMAX, round(max(k["R"] for k in cfg["kernels"][:C]))))
    kern = fl.bake_from_config(cfg["kernels"], C, KR, dtype=torch.float32)
    mat = torch.tensor(cfg["mat"]).reshape(C, C).float()

    task = Task(C, a.classes, cfg["N"], dg.DIGIT)
    x, y = dg.load("test")
    ck = os.path.join(run, "ckpt.pt")
    siren = None
    if os.path.exists(ck):
        st = torch.load(ck, weights_only=False)
        if st.get("siren"):
            siren = dg.SirenSeed(cfg["N"], C - 1, 32, 2)
            siren.load_state_dict(st["siren"])
    seeds, _, lab = task.build(x[a.index:a.index + 1], y[a.index:a.index + 1],
                               np.random.default_rng(7), siren)

    steps = [int(s) for s in a.steps.split(",")]
    frames, rho, done = [], seeds.clone(), 0
    with torch.no_grad():
        for s in steps:
            for _ in range(s - done):
                rho = fl.step(rho, kern, mat, cfg["force"], cfg["repel"],
                              cfg["beta"], 0)
            done = s
            frames.append(rho[0].clone())

    N = cfg["N"]
    rows = []
    for c in range(C):
        row = []
        for f in frames:
            v = f[c].numpy()
            row.append(v / max(v.max(), 1e-9))
        rows.append(np.concatenate(row, axis=1))
    grid = np.concatenate(rows, axis=0)
    img = Image.fromarray((np.clip(grid, 0, 1) * 255).astype(np.uint8))
    out = a.out or os.path.join(run, "channels.png")
    img.resize((img.width * a.scale, img.height * a.scale), Image.NEAREST).save(out)
    print(f"digit is a {int(lab[0])}; rows are channels 0..{C-1} "
          f"(0 = the digit, rest = chemicals), columns are steps {steps}")
    for c in range(C):
        v = frames[-1][c]
        k = max(1, v.numel() // 20)
        top = float(v.flatten().topk(k).values.sum() / v.sum().clamp_min(1e-9))
        print(f"  ch{c}  mass {float(v.sum()):8.1f}   "
              f"top-5% of cells hold {top:.3f} of it"
              f"{'   (uniform)' if top < 0.09 else ''}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
