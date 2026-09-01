"""
Hold the browser demo against the PyTorch port, step for step.

    node train/verify_demo.mjs > train/demo_run.json
    python3 train/verify_demo.py

The page takes two shortcuts the trainer does not: the static channels are
convolved once instead of every tick, and the crowding term is one blur of the
summed density instead of one per channel. Both are meant to be exact, and this
is what says whether they are. It also checks mass conservation, which is the
property the whole design rests on.
"""
import json, os, sys
import numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import digits as dg, fieldlife as fl
from train_digits import Task, roll
from eval_digits import Replay

HERE = os.path.dirname(os.path.abspath(__file__))
run = json.load(open(os.path.join(HERE, "demo_run.json")))
data = open(os.path.join(HERE, "..", "staging", "ncpu-data.js")).read()
D = json.loads(data[data.index("{"):data.rindex(";")])

cfg = json.load(open(os.path.join(HERE, "runs", D["name"], "preset-best.json")))
C = cfg["C"]
task = Task(C, D["nClass"], D["grid"], D["digitPx"], D["nStatic"], D["ring"])
world = Replay(cfg, C)

siren = None
ck = os.path.join(HERE, "runs", D["name"], "ckpt.pt")
st = torch.load(ck, weights_only=False)
if st.get("siren"):
    siren = dg.SirenSeed(D["grid"], D["nChem"], 32, 2)
    siren.load_state_dict(st["siren"])

x, y = dg.load("test")
i = run["sample"]
seeds, _, lab = task.build(x[i:i + 1], y[i:i + 1], np.random.default_rng(0), siren)

print(f"sample {i}, a {int(lab[0])}; {len(run['frames'])} checkpoints\n")
print(f"{'step':>5} {'vote max|diff|':>15} {'pointer max|diff|':>18} "
      f"{'argmax':>8} {'mass drift':>12}")
worst = 0.0
rho, done = seeds.clone(), 0
m0 = float(rho[0, task.ptr].sum())
for f in run["frames"]:
    with torch.no_grad():
        rho = roll(world, task, rho, f["step"] - done)
    done = f["step"]
    s_t = task.scores(rho)[0].numpy()
    s_j = np.array(f["scores"])
    p_t = rho[0, task.ptr].numpy().ravel()
    p_j = np.array(f["pointer"])
    dv, dp = np.abs(s_t - s_j).max(), np.abs(p_t - p_j).max()
    worst = max(worst, float(dv))
    agree = "same" if s_t.argmax() == s_j.argmax() else "DIFFER"
    drift = abs(float(np.array(f["mass"])[task.ptr]) - m0) / max(m0, 1e-9)
    print(f"{f['step']:>5} {dv:>15.3e} {dp:>18.3e} {agree:>8} {drift:>12.2e}")
print(f"\nworst vote difference {worst:.3e}  ->  "
      + ("the page runs the trained world" if worst < 1e-4
         else "THE PAGE IS RUNNING SOMETHING ELSE"))
