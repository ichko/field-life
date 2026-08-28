"""
An 8-bit adder, laid out the way it is written: two columns in, one column out.

    python3 train/adder.py --show

The field is a torus of C channels. **Channel 0 is the only one that carries
data.** Input is written into it and the answer is read out of it. The other
channels hold nothing anyone reads -- they exist to steer channel 0, and they
are pre-charged because they have to be: MaCE conserves mass per channel, so a
channel that starts empty is empty forever and cannot steer anything.

Layout, all in channel 0:

    column A     column B          column OUT
      a7  .        b7  .              s7  .        <- most significant, at the top
      ...          ...                ...
      a0  .        b0  .              s0  .        <- least significant

A bit is a disc of mass present, or nothing. The OUT column starts EMPTY, so
the answer is not rearranged in place -- the mass that spells it has to travel
across the field from the input columns, which is about twenty cells and
therefore at least twenty steps, since MaCE moves mass one cell per step.

There is always enough of it: popcount(a + b) <= popcount(a) + popcount(b) for
every input, so the answer never costs more than the inputs brought. Whatever is
left over has to be parked somewhere outside the read-out band, which is the
only region scored.

The eight bit rows sit in the middle of a taller field with empty space above
and below. That gap is what stops the carry: the rule is the same everywhere and
the field wraps, so without it the carry out of the top bit would come round into
the bottom one. The gap is wider than the kernel reaches, so it does not.
"""

import argparse
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

PITCH_Y = 6        # cells between neighbouring bit rows
COL_A, COL_B, COL_O = 6, 16, 28    # x of the three columns
WIDTH = 40         # so OUT and A are 18 cells apart the other way round the torus
TOP = 8            # y of the most significant row
BLOB_R = 2.6       # disc radius in cells -- under about 2 a disc is a plus sign
BLOB_SOFT = 1.4
READ_HALF = 5      # half-width of the band around OUT that the loss scores

CH_DATA = 0        # the one channel that is written and read


class Geometry:
    def __init__(self, nbits=8, pitch=PITCH_Y, width=WIDTH, top=TOP):
        self.nbits, self.pitch = nbits, pitch
        self.W = width
        # rows in the middle, with a gap to the wrap that is wider than any reach
        self.ys = [top + (nbits - 1 - i) * pitch for i in range(nbits)]
        self.H = self.ys[0] + top + pitch * 2
        self.A = np.stack([self.blob(COL_A, y) for y in self.ys])
        self.B = np.stack([self.blob(COL_B, y) for y in self.ys])
        self.O = np.stack([self.blob(COL_O, y) for y in self.ys])
        self.norm = float((self.O[0] ** 2).sum())          # a full disc's self-overlap
        x = np.arange(self.W)[None, :].repeat(self.H, 0)
        dx = np.minimum(np.abs(x - COL_O), self.W - np.abs(x - COL_O))
        self.mask = (dx <= READ_HALF).astype(np.float64)   # the read-out band

    def blob(self, cx, cy, r=BLOB_R, soft=BLOB_SOFT):
        y, x = np.mgrid[0:self.H, 0:self.W] if hasattr(self, "H") else \
               np.mgrid[0:(self.ys[0] + TOP + self.pitch * 2), 0:self.W]
        dx = np.minimum(np.abs(x - cx), self.W - np.abs(x - cx))
        dy = np.minimum(np.abs(y - cy), y.shape[0] - np.abs(y - cy))
        t = np.clip((r - np.hypot(dx, dy)) / soft, 0.0, 1.0)
        return t * t * (3 - 2 * t)

    def write(self, bits, stamps):
        b = np.asarray(bits, dtype=np.float64)[:, None, None]
        return (b * stamps).sum(0)

    def read(self, field):
        """A row reads 1 when its disc is more than half filled."""
        c = (self.O * np.asarray(field)).sum(axis=(1, 2)) / self.norm
        return (c > 0.5).astype(np.int64), c


def bits_of(v, n):
    return [(int(v) >> i) & 1 for i in range(n)]


def value_of(bits):
    return sum(int(b) << i for i, b in enumerate(bits))


def add(a_bits, b_bits):
    """a + b, dropped to the input width. The carry out of the top bit is lost,
    which is what an n-bit adder without a carry flag does."""
    n, s, carry = len(a_bits), [], 0
    for i in range(n):
        t = a_bits[i] + b_bits[i] + carry
        s.append(t & 1)
        carry = t >> 1
    return s


def problem(a, b, geo):
    a_bits, b_bits = bits_of(a, geo.nbits), bits_of(b, geo.nbits)
    return a_bits, b_bits, add(a_bits, b_bits)


def seed_field(a_bits, b_bits, C, geo, steer=1.0):
    """Inputs written into channel 0; the steering channels pre-charged flat.

    Flat rather than patterned on purpose: a steering channel that starts with
    structure has been told something about the task, and the point is for it to
    find its own. Flat is the one starting state that says nothing -- but it
    cannot be EMPTY, because mass is conserved per channel and empty is a fixed
    point.
    """
    rho = np.zeros((C, geo.H, geo.W))
    rho[CH_DATA] = geo.write(a_bits, geo.A) + geo.write(b_bits, geo.B)
    if C > 1:
        per = steer * geo.nbits * float(geo.O[0].sum()) / (geo.H * geo.W)
        rho[1:] = per
    return rho


def target_field(s_bits, geo):
    return geo.write(s_bits, geo.O)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, default=8)
    ap.add_argument("--channels", type=int, default=6)
    ap.add_argument("--a", type=int, default=0b10110101)
    ap.add_argument("--b", type=int, default=0b01001111)
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    geo = Geometry(args.bits)
    a_bits, b_bits, s_bits = problem(args.a, args.b, geo)
    seed = seed_field(a_bits, b_bits, args.channels, geo)
    target = target_field(s_bits, geo)

    print(f"{args.bits}-bit adder, grid {geo.W} x {geo.H} = {geo.W * geo.H} cells")
    print(f"  columns: A at x {COL_A}, B at x {COL_B}, OUT at x {COL_O}; "
          f"rows at y {geo.ys}")
    print(f"  the answer's mass has to cross {COL_O - COL_A} cells, so no rollout "
          f"shorter than that many steps can work")
    print(f"  {args.a} + {args.b} = {value_of(s_bits)}  (check "
          f"{(args.a + args.b) & ((1 << args.bits) - 1)}, carry out dropped)")
    print(f"  a   {''.join(str(x) for x in a_bits[::-1])}")
    print(f"  b   {''.join(str(x) for x in b_bits[::-1])}")
    print(f"  sum {''.join(str(x) for x in s_bits[::-1])}")
    print(f"  channel 0 seed mass {seed[CH_DATA].sum():.2f}, answer costs "
          f"{target.sum():.2f} -- enough, always")
    got, c = geo.read(target)
    print(f"  reading the target back returns it: {list(got) == s_bits}; "
          f"fill per row {np.round(c, 2).tolist()}")

    if args.show:
        from PIL import Image
        both = np.stack([seed[CH_DATA], target, np.zeros_like(target)], -1)
        both = np.clip(both / max(both.max(), 1e-9), 0, 1)
        p = os.path.join(HERE, "adder_seed.png")
        Image.fromarray((both * 255).astype(np.uint8)).resize(
            (geo.W * 8, geo.H * 8), Image.NEAREST).save(p)
        print(f"  wrote {p}  (red = channel 0 at the seed, green = the answer "
              f"it has to end up holding)")


if __name__ == "__main__":
    main()
