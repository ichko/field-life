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
import math
import os

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "mnist.npz")

GRID = 40          # field side; the ring has to fit inside it
DIGIT = 16         # the digit is scaled to this, from MNIST's 28
RING = 8           # how far the class regions sit from the centre. It was 14,
                   # and mass moves one cell per step, so nothing could be right
                   # before step 14 -- by which point train/survival.py measured
                   # the class as already gone from the field.
REGION_R = 4.0     # radius of a class region
BALL_R = 5.0       # radius of the chemicals' starting ball
# Where the chemicals land. Zero means dead centre, every time.
#
# It was 7, drawn afresh on every reseed, on the reasoning that finding the
# digit should be part of the task. That is a real thing to ask for and it is
# not this experiment's first question: it makes every rollout start from a
# different geometry, so the rule has to be right everywhere before it can score
# anywhere. Fixed placement and a fixed (learned) pattern means one starting
# condition, and whether the field can classify AT ALL gets asked on its own.
# Put the jitter back once something works.
BALL_JITTER = 0.0
NCLASS = 10

# TOTAL chemical mass, as a multiple of the digit's own, split evenly across
# however many chemical channels there are.
#
# Total rather than per-channel, because per-channel silently ties the mass
# budget to the channel count: at 4x each, going from five chemicals to fifteen
# takes the reagents from 20x the digit to 60x, and the experiment changes
# underneath while only the channel count was meant to.
#
# This ratio is the one mass number that matters. A GLOBAL scale is not a free
# parameter of the task, it is a reparameterisation: the affinity is
# force*M.U - repel*sum(N), both U and N are linear in density, so doubling
# every channel is the same as halving force and repel -- and both of those are
# trained. What no rescaling can reach is how much mass the chemicals have
# against the mass they are trying to move. At 1:1 the reagents are the same
# size as the workpiece; the machinery should be bigger than the part.
CHEM = 5.0


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

    def seed(self, img, C, rng, chem=CHEM, n_static=4, pointer=1.0):
        """Static digit copies, then the pointer, then the chemicals.

        Channels 0..n_static-1 all hold the SAME digit and never move. Several
        copies rather than one because a static field enters every channel's
        affinity only through its own convolution, so one copy is one filter
        response and one filter cannot separate ten classes. Four copies with
        four learned kernels is a learned filter bank -- which is what an NCA's
        perception layer is.

        Channel n_static is the POINTER: a compact ball at the centre, and the
        only thing read. It is what the answer is written with, and a ball is
        chosen because a ball survives being pushed across the field where the
        digit's own fine structure does not.
        """
        rho = np.zeros((C, self.grid, self.grid))
        d = self.place(img)
        m = d.sum()
        for c in range(n_static):
            rho[c] = d
        rho[n_static] = self.ball(rng, m * pointer, jitter=0.0)
        nchem = C - n_static - 1
        for c in range(n_static + 1, C):
            rho[c] = self.ball(rng, m * chem / max(nchem, 1))
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


# --------------------------------------------------------- the learned reagent

class SirenSeed(torch.nn.Module):
    """The chemicals' starting pattern, as a coordinate network rather than a ball.

        rho_c(x, y) = softplus( MLP_sin(x, y) )_c  *  window(x, y),  renormalised

    A SIREN: sine activations with the omega0 scaling and the matching
    initialisation, so the field it emits has structure at every scale the grid
    can hold instead of the one blob a hand-written disc has.

    Two properties make this legitimate rather than a way of cheating.

    **It never sees the digit.** The pattern is the same for every input, so it
    cannot smuggle in the answer -- it is the reagent, not the reading. All it
    can learn is what shape of stuff is a good thing to drop next to an unknown
    digit.

    **It is still dropped somewhere random.** The window keeps it a compact clump
    and the clump is rolled to a random offset every time, so finding the digit
    stays part of the task. What is learned is the clump's structure, not where
    it lands.

    The mass is renormalised per channel afterwards, so --chem still means what
    it says and the network cannot win by simply asking for more matter.
    """

    def __init__(self, grid, nchem, hidden=32, layers=2, omega0=30.0,
                 window_r=None, seed=0):
        super().__init__()
        self.grid, self.nchem, self.omega0 = grid, nchem, omega0
        g = torch.Generator().manual_seed(seed + 11)
        dims = [2] + [hidden] * layers
        self.lin = torch.nn.ModuleList(
            [torch.nn.Linear(dims[i], dims[i + 1]) for i in range(layers)])
        self.out = torch.nn.Linear(hidden, nchem)
        with torch.no_grad():
            # SIREN's initialisation: the first layer spans omega0 periods across
            # the input range, the rest are scaled so the pre-activations keep a
            # unit-ish spread through the sines rather than saturating them.
            for i, l in enumerate(self.lin):
                b = 1.0 / dims[i] if i == 0 else math.sqrt(6.0 / dims[i]) / omega0
                l.weight.uniform_(-b, b, generator=g)
                l.bias.uniform_(-b, b, generator=g)
            b = math.sqrt(6.0 / hidden) / omega0
            self.out.weight.uniform_(-b, b, generator=g)
            self.out.bias.zero_()

        c = (grid - 1) / 2
        y, x = torch.meshgrid(torch.arange(grid, dtype=torch.float32),
                              torch.arange(grid, dtype=torch.float32),
                              indexing="ij")
        r = window_r if window_r is not None else BALL_R + 2.0
        d = torch.hypot(x - c, y - c)
        t = ((r - d) / 1.6).clamp(0, 1)
        self.register_buffer("window", t * t * (3 - 2 * t))
        # coordinates normalised so the window spans [-1, 1], which is the range
        # omega0 is calibrated for
        self.register_buffer("coords", torch.stack([(x - c) / r, (y - c) / r], -1))

    def forward(self, mass):
        """(nchem, grid, grid), each channel carrying `mass`, centred."""
        h = self.coords.reshape(-1, 2)
        for l in self.lin:
            h = torch.sin(self.omega0 * l(h))
        v = torch.nn.functional.softplus(self.out(h))
        v = v.T.reshape(self.nchem, self.grid, self.grid) * self.window
        return v * (mass / v.sum(dim=(1, 2), keepdim=True).clamp_min(1e-9))

    def place(self, mass, rng, jitter=BALL_JITTER):
        """The pattern, rolled to a random spot near the centre. Torus, so exact."""
        th = rng.random() * 2 * np.pi
        d = jitter * np.sqrt(rng.random())
        dy, dx = int(round(d * np.sin(th))), int(round(d * np.cos(th)))
        return torch.roll(self(mass), shifts=(dy, dx), dims=(1, 2))
