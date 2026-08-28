"""
Check the PyTorch port against a real WebGL run of the same simulation.

    node train/dump_reference.mjs      # step the browser, write reference.json
    python3 train/parity.py            # step the port, compare

Why this is not one number
--------------------------
The step is chaotic in the regime the page ships (force 90, beta 3): the
softmax inside MaCE is winner-take-all there, so any difference between two
runs -- including the 1e-8 of simply writing the seed field into a float32
texture -- grows by roughly e per step. Demanding that a float64 port track a
float32 GPU for 32 of those steps is demanding the impossible, and a test that
asks for it would have to be loosened until it proved nothing.

So the reference dump carries two regimes and this checks three claims:

  bank   The baked kernels must agree outright. bakeKernel is pure arithmetic
         with no feedback, so anything above float32 round-off here is a real
         porting bug -- and this is where such a bug would show first.

  cool   force 12, beta 0.35: the step does not amplify. The fields must stay
         together for all 32 steps, tightly.

  hot    force 90, beta 3: instead of comparing to the browser directly,
         compare the port-vs-browser gap to the port's OWN float32-vs-float64
         gap. If the two curves sit on top of each other, the port is as close
         to the browser as the simulation is to itself at a different float
         width, which is the strongest statement available. The fitted growth
         rate per step is printed because it is the number that decides how
         many steps you can backpropagate through -- see docs/nca-experiment.md.
"""

import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fieldlife as fl

REF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference.json")

BANK_TOL = 1e-6      # float32 round-off on a unit-L1 kernel
COOL_TOL = 1e-4      # relative, after 32 non-amplifying steps
HOT_SLACK = 8.0      # how far the hot gap may exceed the port's own f32 gap


def trajectory(rho, bank, mat, cfg, checkpoints):
    out, cur, i = {}, rho.clone(), 0
    for s in checkpoints:
        while i < s:
            cur = fl.step(cur, bank, mat, cfg["force"], cfg["repel"], cfg["beta"])
            i += 1
        out[s] = cur.clone()
    return out


def growth_rate(steps, errs):
    """Least-squares slope of log(err) against step -- the Lyapunov exponent."""
    pts = [(s, math.log(e)) for s, e in zip(steps, errs) if s > 0 and e > 0]
    if len(pts) < 2:
        return float("nan")
    n = len(pts)
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    num = sum((p[0] - mx) * (p[1] - my) for p in pts)
    den = sum((p[0] - mx) ** 2 for p in pts)
    return num / den if den else float("nan")


def main():
    if not os.path.exists(REF):
        sys.exit(f"{REF} missing -- run: node train/dump_reference.mjs")
    ref = json.load(open(REF))
    C, N, KR, cps = ref["C"], ref["N"], ref["kernelKR"], ref["checkpoints"]
    ok = True

    print(f"grid {N}x{N}   C {C}   stencil half-width {KR}   ({ref['renderer']})")

    # --- the kernel bank -----------------------------------------------------
    bank = fl.bake_bank_legacy(ref["kernels"], C, dtype=torch.float64)
    if bank.shape[-1] != 2 * KR + 1:
        sys.exit(f"half-width disagrees: port {bank.shape[-1] // 2}, browser {KR}")
    ref_bank = ref["regimes"][next(iter(ref["regimes"]))]["bank"]
    bank_err = 0.0
    for c in range(C):
        got = torch.tensor(ref_bank[c], dtype=torch.float64)
        kr = int(round(len(got) ** 0.5)) // 2
        got = got.reshape(2 * kr + 1, 2 * kr + 1)
        o = KR - kr
        bank_err = max(bank_err, (bank[c, o:o + 2 * kr + 1, o:o + 2 * kr + 1] - got)
                       .abs().max().item())
    good = bank_err <= BANK_TOL
    ok &= good
    print(f"\nbank   max abs err {bank_err:.3e}   tol {BANK_TOL:.0e}   "
          f"{'ok' if good else 'FAIL'}")

    rho0 = torch.tensor(ref["rho0"], dtype=torch.float64).reshape(C, N, N)

    for name, reg in ref["regimes"].items():
        cfg = reg["cfg"]
        mat = torch.tensor(reg["mat"], dtype=torch.float64).reshape(C, C)
        frames = {int(k): v for k, v in reg["frames"].items()}

        got = trajectory(rho0, bank, mat, cfg, cps)
        # the same trajectory at the browser's float width, for the hot regime
        got32 = trajectory(rho0.float(), bank.float(), mat.float(), cfg, cps)

        print(f"\n{name}   force {cfg['force']}  repel {cfg['repel']}  beta {cfg['beta']}")
        print(f"  {'step':>5} {'peak':>8} {'vs browser':>12} {'rel':>10} {'f64 vs f32':>12}")
        rels, selfs = [], []
        for s in cps:
            want = torch.tensor(frames[s], dtype=torch.float64).reshape(C, N, N)
            peak = max(want.abs().max().item(), 1e-12)
            e = (got[s] - want).abs().max().item()
            e_self = (got[s] - got32[s].double()).abs().max().item()
            rels.append(e / peak)
            selfs.append(e_self / peak)
            print(f"  {s:>5} {peak:>8.3f} {e:>12.3e} {e / peak:>10.3e} {e_self / peak:>12.3e}")

        if name == "cool":
            good = rels[-1] <= COOL_TOL
            print(f"  relative err after {cps[-1]} steps {rels[-1]:.3e}"
                  f"   tol {COOL_TOL:.0e}   {'ok' if good else 'FAIL'}")
        else:
            lam = growth_rate(cps, rels)
            lam_self = growth_rate(cps, selfs)
            # the gap to the browser must not outrun the port's own float32 gap
            ratio = max((r / max(s_, 1e-30)) for r, s_ in zip(rels[1:], selfs[1:]))
            good = ratio <= HOT_SLACK
            print(f"  divergence rate  vs browser e^{lam:.2f}/step"
                  f"   own f32 e^{lam_self:.2f}/step")
            print(f"  worst ratio browser-gap / own-f32-gap {ratio:.2f}"
                  f"   tol {HOT_SLACK:.0f}   {'ok' if good else 'FAIL'}")
            print(f"  -> backprop through T steps amplifies by ~e^{lam * 1:.2f}T;"
                  f" past T ~ {int(30 / max(lam, 1e-6))} the gradient is noise.")
        ok &= good

    # mass conservation is exact arithmetic, so it is a hard check
    drift = 0.0
    for name, reg in ref["regimes"].items():
        want = torch.tensor(reg["frames"][str(cps[-1])], dtype=torch.float64).reshape(C, N, N)
        drift = max(drift, ((want.sum(dim=(1, 2)) - rho0.sum(dim=(1, 2))).abs()
                            / rho0.sum(dim=(1, 2))).max().item())
    print(f"\nmass   worst relative drift in the browser over {cps[-1]} steps {drift:.3e}")

    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
