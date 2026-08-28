"""
Replay a trained world and ask whether it actually does the arithmetic.

    python3 train/eval_adder.py --name add4 --slots 5,9,17 --pairs 512

Two things this does that the trainer's own numbers do not.

**It runs the exported preset, not the trained module.** The kernels are rebuilt
from `preset.json` through `fieldlife.bake_from_config` -- the same path
index.html would take -- so a world that scores well only in the trainer's
parameterisation is caught here. The lizard experiment found exactly that
failure once and it was silent: an unknown field on a term is simply not read.

**It reports accuracy, not loss.** Per-bit and exact-answer rates at a spread of
horizons, at every width asked for, over the whole input space where that is
small enough to enumerate and a large random sample where it is not. The
headline number is the 8-bit one: nine slots, 65536 possible pairs, from a world
that was fitted on five.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import adder as ad
import fieldlife as fl
from train_adder import Task

HERE = os.path.dirname(os.path.abspath(__file__))


class Replay:
    """A world loaded from a preset dict, stepped exactly as index.html would."""

    def __init__(self, cfg, C, dtype=torch.float32):
        self.C = C
        R = [k["R"] for k in cfg["kernels"][:C]]
        self.KR = max(3, min(fl.KMAX, round(max(max(R), 1.0))))
        self.kern = fl.bake_from_config(cfg["kernels"], C, self.KR,
                                        dtype=dtype).to(dtype)
        self.mat = torch.tensor(cfg["mat"], dtype=dtype).reshape(C, C)
        self.force = float(cfg["force"])
        self.repel = float(cfg["repel"])
        self.beta = float(cfg["beta"])

    def rollout(self, rho, steps):
        return fl.run(rho, self.kern, self.mat, self.force, self.repel,
                      self.beta, steps)


def sample_pairs(nbits, count, rng):
    """Every pair when the space is small enough, a random sample when not."""
    lim = 1 << nbits
    if lim * lim <= count:
        return [(x, y) for x in range(lim) for y in range(lim)], True
    return [(int(rng.integers(0, lim)), int(rng.integers(0, lim)))
            for _ in range(count)], False


def baseline(task, pairs, chunk=256):
    """What answering every slot with its own majority bit would score.

    Without this the accuracy columns cannot be read. `and` is 1 in a quarter of
    cases, so a field that has learned nothing and pushes every blob to the zero
    rail scores 0.75 per bit -- which looks like most of the way to solved and is
    none of it. `copy`, `xor` and `add` all sit at 0.5, but only because their
    answers happen to be balanced.
    """
    ones = None
    n = 0
    for lo in range(0, len(pairs), chunk):
        _, _, bits, _ = task.build(pairs[lo:lo + chunk])
        b = bits.sum(0).numpy()
        ones = b if ones is None else ones + b
        n += bits.shape[0]
    p = ones / max(n, 1)
    per_slot = np.maximum(p, 1 - p)
    # exact: every slot right at once, if each is answered by its own majority
    return float(per_slot.mean()), float(np.prod(per_slot))


def evaluate(world, task, pairs, horizons, chunk=64):
    bit = np.zeros(len(horizons))
    exact = np.zeros(len(horizons))
    nbits_seen = 0
    with torch.no_grad():
        for lo in range(0, len(pairs), chunk):
            seeds, _, bits, _ = task.build(pairs[lo:lo + chunk])
            rho, done = seeds.clone(), 0
            nbits_seen += bits.numel()
            for j, h in enumerate(horizons):
                rho = world.rollout(rho, h - done)
                done = h
                ok = task.decode(rho) == bits
                bit[j] += int(ok.sum())
                exact[j] += int(ok.all(dim=1).sum())
    return bit / max(nbits_seen, 1), exact / max(len(pairs), 1)


def strip(path, world, task, pairs, horizons, scale=4):
    """Seed, target, then the rollout at each horizon -- one row per problem."""
    from PIL import Image
    seeds, targets, _, _ = task.build(pairs)
    frames, rho, done = [], seeds.clone(), 0
    with torch.no_grad():
        for h in horizons:
            rho = world.rollout(rho, h - done)
            done = h
            frames.append(rho[:, :3].clone())
    rows = []
    for k in range(len(pairs)):
        tgt = torch.stack([targets[k]] * 3).numpy()
        tiles = [seeds[k, :3].numpy(), tgt] + [f[k].numpy() for f in frames]
        rows.append(np.concatenate(
            [np.clip(t.transpose(1, 2, 0), 0, 1) for t in tiles], axis=1))
    img = Image.fromarray((np.concatenate(rows, axis=0) * 255).astype(np.uint8))
    img.resize((img.width * scale, img.height * scale), Image.NEAREST).save(path)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="add4")
    ap.add_argument("--preset", default=None, help="overrides --name")
    ap.add_argument("--best", action="store_true", help="use preset-best.json")
    ap.add_argument("--op", default=None, help="defaults to the preset's own")
    ap.add_argument("--slots", default="5,9,17")
    ap.add_argument("--pairs", type=int, default=512)
    ap.add_argument("--horizons", default="32,64,128,256,512")
    ap.add_argument("--channels", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--strip", default=None, help="write a picture strip here")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    path = a.preset or os.path.join(
        HERE, "runs", a.name, "preset-best.json" if a.best else "preset.json")
    cfg = json.load(open(path))
    op = a.op or cfg.get("_task", {}).get("op", "add")
    world = Replay(cfg, a.channels)
    horizons = [int(x) for x in a.horizons.split(",")]
    rng = np.random.default_rng(a.seed)

    print(f"{os.path.relpath(path, HERE)}  op {op}  "
          f"force {world.force:.2f}  repel {world.repel:.2f}  "
          f"beta {world.beta:.3f}  stencil {world.KR}"
          + (f"  (best at iteration {cfg['_bestAt']})" if "_bestAt" in cfg else ""))
    trained = cfg.get("_task", {}).get("slots")
    if trained:
        print(f"trained on {trained} slots; anything else below is a width it "
              f"has never seen")
    print()
    head = "  ".join(f"{h:>13d}" for h in horizons)
    print(f"{'width':>16}  {head}   {'majority':>13}")
    print(f"{'':>16}  " + "  ".join(f"{'bit / exact':>13}" for _ in horizons)
          + f"   {'bit / exact':>13}")

    for s in (int(x) for x in a.slots.split(",")):
        task = Task(s, a.channels, op)
        pairs, full = sample_pairs(task.geo.nbits, a.pairs, rng)
        bit, exact = evaluate(world, task, pairs, horizons)
        bb, be = baseline(task, pairs)
        tag = (f"{task.geo.nbits}-bit, {s} slots"
               + ("" if full else "*"))
        print(f"{tag:>16}  " + "  ".join(
            f"{b:6.3f} /{e:6.3f}" for b, e in zip(bit, exact))
            + f"   {bb:6.3f} /{be:6.3f}")
        if a.strip and s == int(a.slots.split(",")[0]):
            print(f"  wrote {strip(a.strip, world, task, pairs[:6], horizons)}")
    print("\n* sampled, not enumerated: the space is larger than --pairs")
    print("majority is the trivial world: answer every slot with its own most "
          "common bit\nand move nothing. Anything at or below it has learned "
          "nothing.")


if __name__ == "__main__":
    main()
