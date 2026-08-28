"""
How long does the class survive in the field at all?

    python3 train/survival.py --name dig10

This is the measurement that bounds the whole task. A readout can only report
what is still there, so before asking whether the field can learn to classify,
ask how much class information the dynamics has left after T steps -- by fitting
a LINEAR probe directly on the digit channel at each T. The probe is not part of
the model and is not something field-life gets to use; it is an upper bound. If
a linear probe on the raw field cannot tell the digits apart at step 32, then no
arrangement of mass at step 32 encodes the answer, and no rule could have been
trained to produce one.

Ridge regression to one-hot, closed form, so there is nothing to tune and
nothing to blame on optimisation.
"""

import argparse, json, os, sys, time
import numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import digits as dg, fieldlife as fl
from train_digits import Task

HERE = os.path.dirname(os.path.abspath(__file__))


def probe(X, y, Xt, yt, nclass, lam=1e-2):
    X = torch.cat([X, torch.ones(len(X), 1)], 1)
    Xt = torch.cat([Xt, torch.ones(len(Xt), 1)], 1)
    Y = torch.zeros(len(y), nclass); Y[torch.arange(len(y)), y] = 1.0
    A = X.T @ X + lam * len(X) * torch.eye(X.shape[1])
    W = torch.linalg.solve(A, X.T @ Y)
    return float(((Xt @ W).argmax(1) == yt).float().mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="dig10")
    ap.add_argument("--steps", default="0,4,8,16,32,64")
    ap.add_argument("--train", type=int, default=600)
    ap.add_argument("--test", type=int, default=300)
    ap.add_argument("--classes", type=int, default=10)
    ap.add_argument("--channels", type=int, default=6)
    ap.add_argument("--neutral", action="store_true",
                    help="probe the law with the interaction matrix zeroed -- "
                         "pure crowding, which is what an untrained world does")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    cfg = json.load(open(os.path.join(HERE, "runs", a.name, "preset.json")))
    C = cfg["C"]
    KR = max(3, min(fl.KMAX, round(max(k["R"] for k in cfg["kernels"][:C]))))
    kern = fl.bake_from_config(cfg["kernels"], C, KR, dtype=torch.float32)
    mat = torch.tensor(cfg["mat"]).reshape(C, C).float()
    if a.neutral:
        mat = torch.zeros_like(mat)
    force, repel, beta = cfg["force"], cfg["repel"], cfg["beta"]

    task = Task(a.channels, a.classes, cfg["N"], dg.DIGIT)
    x, y = dg.load("train")
    keep = y < a.classes
    x, y = x[keep], y[keep]
    n = a.train + a.test
    rng = np.random.default_rng(0)
    seeds, _, lab = task.build(x[:n], y[:n], rng, None)

    steps = [int(s) for s in a.steps.split(",")]
    print(f"{a.name}{' (matrix zeroed)' if a.neutral else ''}: force {force:.1f} "
          f"repel {repel:.2f} beta {beta:.3f}")
    print(f"linear probe on the digit channel, {a.train} train / {a.test} test, "
          f"{a.classes} classes, chance {1/a.classes:.3f}\n")
    print(f"{'step':>6}  {'probe acc':>10}  {'mass in top 5% of cells':>24}")
    rho, done = seeds.clone(), 0
    for s in steps:
        with torch.no_grad():
            if s > done:
                for _ in range(s - done):
                    rho = fl.step(rho, kern, mat, force, repel, beta, 0)
                done = s
            d = rho[:, 0]
            F = d.reshape(len(d), -1)
            acc = probe(F[:a.train], lab[:a.train], F[a.train:], lab[a.train:],
                        a.classes)
            k = max(1, F.shape[1] // 20)
            top = F.topk(k, dim=1).values.sum(1) / F.sum(1).clamp_min(1e-9)
        print(f"{s:>6}  {acc:>10.3f}  {float(top.mean()):>23.3f}")
    print("\nprobe acc is an UPPER BOUND on any readout: it is a linear "
          "classifier\nwith full access to the field, which field-life does not "
          "have.\nmass in top 5% starts near 1.0 for a sharp digit and falls to "
          "0.05 when uniform.")


if __name__ == "__main__":
    main()
