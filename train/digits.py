"""
Spatial classification: the field answers by moving the digit somewhere.

    python3 train/digits.py --show

An MNIST digit is drawn into channel 0. Ten target regions sit in a ring around
it, one per class. The field has answered when the digit's own matter has been
carried into the region for its class -- classification as transport, which is
the one kind of answer a mass-conserving law can give.

    channel 0        the digit. NOT static: it is the matter that has to move,
                     and it is the only channel anything is read from.
    channels 1+      the chemicals. A ball of mass dropped at a random place
                     near the digit, different every time, so the first thing
                     they have to learn is to FIND it.

Two properties make this a better fit for MaCE than the adder was.

**Nothing has to be metered out.** The digit brings its own mass and the answer
is where that mass ends up, so there is no input for which the budget fails --
which is exactly what sank the colour-coded adder, where 1 + 255 needs eight
units of zero-coloured mass and the inputs carry seven.

**The chemicals start in the wrong place on purpose.** Their ball is dropped at
a random offset, so a rule that only works when the reagents happen to be
centred is not a rule that scores. Finding the digit is part of the task.
"""

import argparse
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "mnist.npz")

GRID = 40          # field side; the ring has to fit inside it
DIGIT = 16         # the digit is scaled to this, from MNIST's 28
RING = 14          # how far the class regions sit from the centre
REGION_R = 4.0     # radius of a class region
BALL_R = 3.0       # radius of the chemicals' starting ball
BALL_JITTER = 7.0  # how far from centre that ball may be dropped
NCLASS = 10


def load(split="train"):
    z = np.load(DATA)
    return (z["xtr"], z["ytr"]) if split == "train" else (z["xte"], z["yte"])


def _disc(grid, cx, cy, r, soft=1.4):
    y, x = np.mgrid[0:grid, 0:grid]
    d = np.hypot(x - cx, y - cy)
    t = np.clip((r - d) / soft, 0.0, 1.0)
    return t * t * (3 - 2 * t)


class Geometry:
    """The field, and the ten places an answer can be."""

    def __init__(self, grid=GRID, digit=DIGIT, ring=RING, region_r=REGION_R,
                 nclass=NCLASS):
        self.grid, self.digit, self.nclass = grid, digit, nclass
        self.c = (grid - 1) / 2
        # class k sits at angle 2*pi*k/nclass, measured from straight up and
        # going clockwise, so the classes read round the ring in order
        self.regions = []
        for k in range(nclass):
            th = 2 * np.pi * k / nclass
            self.regions.append(_disc(grid, self.c + ring * np.sin(th),
                                      self.c - ring * np.cos(th), region_r))
        self.regions = np.stack(self.regions)
        # normalised, so a region's score is the FRACTION of the digit inside it
        # and a big region cannot win by being big
        self.w = self.regions / self.regions.sum(axis=(1, 2), keepdims=True)

    def place(self, img):
        """One MNIST digit, scaled and centred in the field, as mass."""
        from PIL import Image
        small = np.asarray(Image.fromarray(img).resize(
            (self.digit, self.digit), Image.BILINEAR), dtype=np.float64) / 255.0
        out = np.zeros((self.grid, self.grid))
        o = (self.grid - self.digit) // 2
        out[o:o + self.digit, o:o + self.digit] = small
        return out

    def ball(self, rng, mass, jitter=BALL_JITTER, r=BALL_R):
        """The chemicals' starting clump, dropped near the digit but not on it."""
        th = rng.random() * 2 * np.pi
        d = jitter * np.sqrt(rng.random())
        b = _disc(self.grid, self.c + d * np.cos(th), self.c + d * np.sin(th), r)
        s = b.sum()
        return b * (mass / s) if s > 0 else b

    def seed(self, img, C, rng, chem=1.0):
        rho = np.zeros((C, self.grid, self.grid))
        rho[0] = self.place(img)
        m = rho[0].sum() * chem
        for c in range(1, C):
            rho[c] = self.ball(rng, m)
        return rho

    def scores(self, field):
        """How much of the digit sits in each class region."""
        return (self.w * np.asarray(field)[None]).sum(axis=(1, 2))

    def target(self, k, mass):
        """Every gram of the digit inside region k, and nothing outside it."""
        t = self.regions[k]
        return t * (mass / t.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channels", type=int, default=6)
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()

    x, y = load("train")
    geo = Geometry()
    rng = np.random.default_rng(0)
    img, lab = x[a.index], int(y[a.index])
    rho = geo.seed(img, a.channels, rng)

    print(f"grid {geo.grid} x {geo.grid} = {geo.grid ** 2} cells, digit scaled "
          f"to {geo.digit}, {geo.nclass} class regions on a ring of radius {RING}")
    print(f"  sample {a.index} is a {lab}; it carries "
          f"{rho[0].sum():.2f} of mass")
    print(f"  chemicals: {a.channels - 1} channels, each a ball of "
          f"{rho[1].sum():.2f} dropped up to {BALL_JITTER} cells off centre")
    print(f"  region scores at the seed: "
          f"{np.round(geo.scores(rho[0]), 4).tolist()}")
    print(f"  a perfect answer scores "
          f"{geo.scores(geo.target(lab, rho[0].sum()))[lab]:.4f} on region {lab}")

    if a.show:
        from PIL import Image
        ring = geo.regions.sum(0)
        v = np.stack([rho[0] / max(rho[0].max(), 1e-9),
                      ring / max(ring.max(), 1e-9),
                      rho[1] / max(rho[1].max(), 1e-9)], -1)
        p = os.path.join(HERE, "digits_seed.png")
        Image.fromarray((np.clip(v, 0, 1) * 255).astype(np.uint8)).resize(
            (geo.grid * 10, geo.grid * 10), Image.NEAREST).save(p)
        print(f"  wrote {p}  (red = the digit, green = the ten class regions, "
              f"blue = where the chemicals landed)")


if __name__ == "__main__":
    main()
