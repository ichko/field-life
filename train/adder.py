"""
The task: an 8-bit adder, encoded so that a mass-conserving field could hold it.

field-life's step (MaCE) only ever MOVES mass, exactly, per channel. Nothing is
created. So "compute" has to mean "transport", and the encoding is most of the
experiment. Three decisions make an adder expressible at all.

**Dual rail.** A bit is not "mass present / mass absent" -- it is one unit of
mass sitting at one of two places. Slot `i` owns two cells: a 0-rail below the
mid line and a 1-rail above it. Then the mass a channel holds is the same for
every input, the answer is purely *which side*, and there is no do-nothing
escape: a blob left in the middle is equally wrong for a 0 and for a 1. With
presence/absence the loss has a trivial minimum -- dump everything and score the
background -- and a mass-conserving rule cannot hit a target whose total mass
depends on the input anyway (1 + 1 = 2 turns two lit bits into one).

**The output is pre-charged, not grown.** Channel 2 starts with one blob per
slot on the mid line, exactly as much mass as the answer costs. Solving the
task is moving each of those blobs to the rail the arithmetic says. That is the
lizard's mass contract (docs/nca-experiment.md §3) with an input-dependent
target: the loss is about arrangement only, and there is no mass term.

**The bit axis is the torus.** A ripple-carry adder is a chain of identical
1-bit full adders with a carry running along it -- which is to say it is a
cellular automaton in the bit index, and field-life's rule is already the same
everywhere. Slots tile the x-axis exactly, so W = nslots * pitch and the
adder is CYCLIC. That sounds wrong and is not: the top slot's two input bits
are pinned to 0, so its carry-out is always 0, so the carry-in that wraps
around into slot 0 is always 0. An `n`-bit adder therefore needs n+1 slots --
the extra one both carries the answer's top bit and terminates the ripple.

The reason to want that is generalisation. A world trained on 5 slots is a
local rule, and the same rule dropped on 9 slots is an 8-bit adder it has never
seen. Width generalisation is the test that separates a learned algorithm from
a memorised table, and no fitted lookup can pass it.

    python3 train/adder.py --show
"""

import argparse
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

PITCH = 6          # cells between neighbouring bit slots
HEIGHT = 24        # rows; the rails use eight of them and the rest is free space
RAIL = 3           # rails sit this far above and below the mid line
BLOB_R = 1.7       # blob radius in cells
BLOB_SOFT = 1.2    # smoothstep width at the blob's rim

# channel roles. 0-2 are read as RGB, so a rollout is directly viewable.
CH_A, CH_B, CH_S = 0, 1, 2


class Geometry:
    """Where the slots and rails are, and the blob stamps for each."""

    def __init__(self, nslots, pitch=PITCH, height=HEIGHT, rail=RAIL):
        self.nslots, self.pitch = nslots, pitch
        self.W, self.H = nslots * pitch, height
        self.ym = height // 2
        self.y1, self.y0 = self.ym - rail, self.ym + rail      # 1-rail above, 0-rail below
        self.xs = [i * pitch + pitch // 2 for i in range(nslots)]
        # (nslots, H, W) stamps: one per slot for each of the three rows
        self.T1 = np.stack([self.blob(x, self.y1) for x in self.xs])
        self.T0 = np.stack([self.blob(x, self.y0) for x in self.xs])
        self.Tm = np.stack([self.blob(x, self.ym) for x in self.xs])

    @property
    def nbits(self):
        """Input width. The top slot is pinned to zero and terminates the ripple."""
        return self.nslots - 1

    def blob(self, cx, cy, r=BLOB_R, soft=BLOB_SOFT):
        """A smoothstep disc of peak 1, wrapped on the torus."""
        y, x = np.mgrid[0:self.H, 0:self.W]
        dx = np.minimum(np.abs(x - cx), self.W - np.abs(x - cx))
        dy = np.minimum(np.abs(y - cy), self.H - np.abs(y - cy))
        d = np.hypot(dx, dy)
        t = np.clip((r - d) / soft, 0.0, 1.0)
        return t * t * (3 - 2 * t)

    def rails(self, bits):
        """(H,W) field holding one blob per slot, on the rail each bit names."""
        b = np.asarray(bits, dtype=np.float64)[:, None, None]
        return (b * self.T1 + (1 - b) * self.T0).sum(0)

    def mid(self):
        """(H,W) field holding one blob per slot, undecided on the mid line."""
        return self.Tm.sum(0)

    def decode(self, field):
        """Read a slot as whichever rail holds more of its blob's mass.

        Correlation against the blob stamp itself, not a box: it is the shape
        the loss asks for, so it is the shape the readout should measure, and it
        falls off smoothly instead of caring where a box edge landed.
        """
        f = np.asarray(field)
        s1 = (self.T1 * f).sum(axis=(1, 2))
        s0 = (self.T0 * f).sum(axis=(1, 2))
        return (s1 > s0).astype(np.int64), s1 - s0


# ------------------------------------------------------------------ the sums

def bits_of(v, n):
    """Little-endian: index 0 is the least significant bit."""
    return [(int(v) >> i) & 1 for i in range(n)]


def value_of(bits):
    return sum(int(b) << i for i, b in enumerate(bits))


def ripple(a_bits, b_bits):
    """The answer, and the carry chain that produces it, on nslots slots.

    Cyclic in the slot index, which is well defined here because the top slot's
    inputs are zero: its carry-out is 0, so the carry that wraps into slot 0 is
    0, and the fixed point is unique and is ordinary addition.
    """
    n = len(a_bits)
    s, c, carry = [0] * n, [0] * n, 0
    for i in range(n):
        c[i] = carry
        t = a_bits[i] + b_bits[i] + carry
        s[i] = t & 1
        carry = t >> 1
    assert carry == 0, "top slot must not carry out; its inputs are pinned to 0"
    return s, c


# A ladder of tasks, in increasing order of what the field has to do. Reporting
# where it breaks says more than a single pass/fail on the adder does.
#
#   copy  transport only: the answer is one input, unchanged. If this fails,
#         nothing about the encoding works and the arithmetic is beside the point.
#   and   one gate per slot, no communication between slots. A threshold on the
#         local density of two channels -- the easiest thing this law could do.
#   or    the same, at the other threshold.
#   xor   still one slot at a time, but NOT a threshold: the answer is high in
#         the middle of the input range and low at both ends, so no single
#         monotone response to local density produces it. It has to come from
#         the dynamics -- one stage computing the AND and a later one subtracting
#         it -- which is exactly the nonlinearity a bare linear affinity lacks.
#   add   xor plus a carry that has to travel along the slot axis, so the answer
#         at slot i is not a function of anything inside slot i.
OPS = ("copy", "and", "or", "xor", "add")


def answer(a_bits, b_bits, op):
    """The bits the sum channel has to end up holding, and the carry chain."""
    if op == "add":
        return ripple(a_bits, b_bits)
    zeros = [0] * len(a_bits)
    if op == "copy":
        return list(a_bits), zeros
    if op == "and":
        return [x & y for x, y in zip(a_bits, b_bits)], zeros
    if op == "or":
        return [x | y for x, y in zip(a_bits, b_bits)], zeros
    if op == "xor":
        return [x ^ y for x, y in zip(a_bits, b_bits)], zeros
    raise ValueError(f"unknown op {op!r}; pick one of {OPS}")


def problem(a, b, geo, op="add"):
    """Input bits and answer bits, padded to the slot count.

    The top slot's inputs are pinned to zero for every op, not only for `add`.
    It is only load-bearing for the adder -- it is what terminates the carry
    ripple -- but keeping it means one geometry serves the whole ladder and the
    widths stay comparable across it.
    """
    n = geo.nslots
    a_bits, b_bits = bits_of(a, n), bits_of(b, n)
    assert a_bits[-1] == 0 and b_bits[-1] == 0, \
        f"a and b must fit in {geo.nbits} bits"
    s_bits, c_bits = answer(a_bits, b_bits, op)
    return a_bits, b_bits, s_bits, c_bits


# ---------------------------------------------------------------- the fields

def seed_field(a_bits, b_bits, C, geo, hidden_mass=1.0):
    """The initial state: inputs on their rails, output and workspace on the mid.

    Every channel gets one blob per slot, so the per-channel mass is the same
    for every input and the whole task is arrangement. The hidden channels are
    pre-charged on the mid line too -- they are the only place a carry can live,
    and a carry needs mass to be made of.
    """
    rho = np.zeros((C, geo.H, geo.W))
    rho[CH_A] = geo.rails(a_bits)
    rho[CH_B] = geo.rails(b_bits)
    rho[CH_S] = geo.mid()
    for c in range(3, C):
        rho[c] = hidden_mass * geo.mid()
    return rho


def target_field(s_bits, geo):
    """What channel 2 has to look like: the answer, on the rails."""
    return geo.rails(s_bits)


# ---------------------------------------------------------------- the dataset

def pairs(nbits, count, rng, exclude=None):
    """Random (a, b) with a, b < 2^nbits, optionally avoiding a held-out set."""
    lim, out, seen = 1 << nbits, [], set(exclude or ())
    while len(out) < count:
        a, b = int(rng.integers(0, lim)), int(rng.integers(0, lim))
        if (a, b) in seen:
            continue
        seen.add((a, b))
        out.append((a, b))
    return out


def split(nbits, n_train, seed=0):
    """A training set of (a,b) pairs, and everything else is the test set.

    At 8 bits there are 65536 pairs and the world has a couple of hundred
    numbers, so held-out accuracy is the only interesting number: nothing here
    has the capacity to memorise a table it could look the answer up in.
    """
    rng = np.random.default_rng(seed)
    tr = pairs(nbits, n_train, rng)
    return tr, set(tr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, default=8)
    ap.add_argument("--channels", type=int, default=6)
    ap.add_argument("--a", type=int, default=0b10110101)
    ap.add_argument("--b", type=int, default=0b01001111)
    ap.add_argument("--op", default="add", choices=list(OPS))
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    geo = Geometry(args.bits + 1)
    a_bits, b_bits, s_bits, c_bits = problem(args.a, args.b, geo, args.op)
    seed = seed_field(a_bits, b_bits, args.channels, geo)
    target = target_field(s_bits, geo)

    print(f"{args.bits}-bit adder on {geo.nslots} slots, "
          f"grid {geo.W} x {geo.H} = {geo.W * geo.H} cells")
    print(f"  rails at y {geo.y1} (one) and y {geo.y0} (zero), mid {geo.ym}; "
          f"slots at x {geo.xs}")
    print(f"  op {args.op}: {args.a} . {args.b} = {value_of(s_bits)}"
          + (f"   (check {args.a + args.b})" if args.op == "add" else ""))
    print(f"  a     {''.join(str(x) for x in a_bits[::-1])}")
    print(f"  b     {''.join(str(x) for x in b_bits[::-1])}")
    print(f"  carry {''.join(str(x) for x in c_bits[::-1])}")
    print(f"  sum   {''.join(str(x) for x in s_bits[::-1])}")
    print(f"  mass per channel: {seed.sum(axis=(1, 2)).round(2).tolist()}")
    print(f"  target mass {target.sum():.2f} vs channel 2 seed mass "
          f"{seed[CH_S].sum():.2f}  (equal by construction)")
    got, margin = geo.decode(target)
    signed = margin * np.where(np.array(s_bits) > 0, 1.0, -1.0)
    print(f"  decoding the target returns it: {list(got) == s_bits}, "
          f"worst margin toward the right rail {signed.min():.3f}")

    if args.show:
        from PIL import Image
        for name, arr in (("adder_seed", seed[:3]), ("adder_target", target)):
            v = arr.transpose(1, 2, 0) if arr.ndim == 3 else np.stack([arr] * 3, -1)
            v = np.clip(v / max(v.max(), 1e-9), 0, 1)
            p = os.path.join(HERE, f"{name}.png")
            Image.fromarray((v * 255).astype(np.uint8)).resize(
                (geo.W * 8, geo.H * 8), Image.NEAREST).save(p)
            print(f"  wrote {p}")


if __name__ == "__main__":
    main()
