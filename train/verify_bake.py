"""
Hold index.html's bake against the PyTorch one, kernel by kernel.

    node train/verify_bake.mjs train/runs/add4/preset.json > train/browser_bank.json
    python3 train/verify_bake.py train/runs/add4/preset.json

A trained world is only a result if it is the SAME world in the browser. The
failure mode this guards is silent: an unrecognised field on a term is not
rejected, it is skipped, so an angular kernel loads as the ring you get by
deleting its angle -- and the preset runs, and looks fine, and is a different
simulation. Cosine similarity per channel is the number to read; anything below
0.999 means the export is not faithful.
"""

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fieldlife as fl

HERE = os.path.dirname(os.path.abspath(__file__))
preset = json.load(open(sys.argv[1] if len(sys.argv) > 1
                        else os.path.join(HERE, "runs/add4/preset.json")))
browser = json.load(open(os.path.join(HERE, "browser_bank.json")))

C, KR = preset["C"], browser["KR"]
print(f"{C} channels, shared stencil half-width {KR}, "
      f"reaches {[round(k['R'], 3) for k in preset['kernels'][:C]]}")

ours = fl.bake_from_config(preset["kernels"], C, KR, dtype=torch.float64).numpy()
worst = 1.0
for c in range(C):
    b = browser["banks"][c]
    k = np.array(b["w"], dtype=np.float64)
    n = int(round(len(k) ** 0.5))
    k = k.reshape(n, n)
    if n != 2 * KR + 1:                        # inset a coarser bake, as uploadKernels does
        o = KR - b["kr"]
        pad = np.zeros((2 * KR + 1, 2 * KR + 1))
        pad[o:o + n, o:o + n] = k
        k = pad
    a = ours[c]
    cos = float((a * k).sum() / max(np.linalg.norm(a) * np.linalg.norm(k), 1e-30))
    worst = min(worst, cos)
    print(f"  ch{c}  baked at kr {b['kr']:2d}   cosine {cos:+.6f}   "
          f"max |diff| {np.abs(a - k).max():.2e}")

print(f"\nworst cosine {worst:+.6f}  ->  "
      + ("the browser bakes what was trained"
         if worst > 0.999 else "THE EXPORT IS NOT FAITHFUL"))
