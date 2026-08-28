"""
Compare the ported mip path against the browser run of the same world.

    node train/verify_mip.mjs && python3 train/verify_mip.py

A reach wider than the stencil is served by convolving a halved copy of the
field. That path had never been ported, so a world using it would have been
trained against arithmetic the simulation does not run -- the same class of
mistake as the kernel taper, caught the same way.
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fieldlife as fl

REF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mip_run.json")


def main():
    if not os.path.exists(REF):
        sys.exit("mip_run.json missing -- run: node train/verify_mip.mjs")
    b = json.load(open(REF))
    C, N = b["C"], b["N"]
    cfg = b["cfg"]

    mip, KR = fl.plan_from_config(cfg["kernels"], C, N)
    print(f"plan: port says mip {mip}, stencil {KR}; "
          f"browser used mip {b['mip']}, stencil {b['kernelKR']}")
    if (mip, KR) != (b["mip"], b["kernelKR"]):
        sys.exit("FAIL: the port would have chosen a different plan")

    kern = fl.bake_from_config(cfg["kernels"], C, KR, mip=mip)
    mat = torch.tensor(cfg["mat"], dtype=torch.float64).reshape(C, C)
    rho = torch.tensor(b["rho0"], dtype=torch.float64).reshape(1, C, N, N)
    got = fl.run(rho, kern, mat, cfg["force"], cfg["repel"], cfg["beta"],
                 b["steps"], mip=mip)
    want = torch.tensor(b["rho"], dtype=torch.float64).reshape(1, C, N, N)

    err = (got - want).abs().max().item()
    scale = max(want.abs().max().item(), 1e-12)
    print(f"after {b['steps']} steps: max abs {err:.3e}, relative {err/scale:.3e}")
    m0 = rho.sum(dim=(2, 3))
    print(f"mass drift  port {(got.sum(dim=(2,3))-m0).abs().max().item():.2e}"
          f"   browser {(want.sum(dim=(2,3))-m0).abs().max().item():.2e}")
    ok = err / scale < 5e-3
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
