"""A cute gecko: big head, big eyes, stubby legs, a fat curled tail.

Drawn this way for two reasons that happen to agree. It is what was asked for,
and it is also the easiest thing of its kind for the rule to hold. At forty
cubed a cell is a fortieth of the world and the target is blurred by about a
cell before the fit ever sees it, so anything thinner than three cells is simply
gone -- a realistic lizard spends most of its detail on toes and scales, and
loses all of it. A cartoon spends everything on a handful of big round masses in
a recognisable arrangement, and loses nothing. Cute is, here, the same thing as
legible.

So: a head nearly as big as the body, two eyes that stand well clear of it,
four short legs with round feet, and a tail that starts thick and curls round.
Nothing in it is narrow, and nothing touches that should not.
"""
import numpy as np

from gecko import smin, sd_capsule, sd_ellipsoid

GROUND = -0.200
HEAD = np.array([0.205, 0.062, 0.0], np.float32)
EYE = [np.array([0.290, 0.148, s*0.108], np.float32) for s in (1.0, -1.0)]
EYE_R = np.array([0.080, 0.082, 0.080], np.float32)

# The tail, as a curl in the horizontal plane. Thick where it leaves the body
# and tapering, but never to a thread.
TAIL = [np.array([-0.145, -0.020, 0.000], np.float32),
        np.array([-0.290, -0.005, 0.050], np.float32),
        np.array([-0.370, 0.050, 0.165], np.float32),
        np.array([-0.300, 0.125, 0.265], np.float32),
        np.array([-0.180, 0.155, 0.275], np.float32)]
TAIL_R = [0.088, 0.072, 0.056, 0.040, 0.026]

# Shoulder along the body, how far the foot reaches forward, and out.
LIMBS = [(0.055, 0.085), (-0.095, -0.070)]


def _body(p):
    d = sd_capsule(p, np.array([0.070, 0.000, 0.0], np.float32),
                   np.array([-0.150, -0.015, 0.0], np.float32), 0.135, 0.115)
    return d


def _head(p):
    d = sd_ellipsoid(p, HEAD, np.array([0.172, 0.158, 0.158], np.float32))
    # a blunt snout, and cheeks
    d = smin(d, sd_ellipsoid(p, HEAD + np.array([0.100, -0.030, 0.0], np.float32),
                             np.array([0.105, 0.098, 0.105], np.float32)), 0.050)
    return d


def _eyes(p):
    d = np.full(p.shape[:3], 10.0, np.float32)
    for e in EYE:
        d = np.minimum(d, sd_ellipsoid(p, e, EYE_R))
    return d


def _tail(p):
    d = np.full(p.shape[:3], 10.0, np.float32)
    for i in range(len(TAIL) - 1):
        d = smin(d, sd_capsule(p, TAIL[i], TAIL[i + 1], TAIL_R[i], TAIL_R[i + 1]), 0.035)
    return d


def _legs(p):
    d = np.full(p.shape[:3], 10.0, np.float32)
    for x0, reach in LIMBS:
        for s in (1.0, -1.0):
            hip = np.array([x0, -0.030, s*0.065], np.float32)
            foot = np.array([x0 + reach, GROUND + 0.030, s*0.165], np.float32)
            d = smin(d, sd_capsule(p, hip, foot, 0.062, 0.050), 0.030)
            d = smin(d, sd_ellipsoid(p, foot + np.array([reach*0.35, -0.022, s*0.012], np.float32),
                                     np.array([0.072, 0.040, 0.062], np.float32)), 0.030)
    return d


def _fields(N):
    lin = (np.arange(N) + 0.5)/N - 0.5
    Z, Y, X = np.meshgrid(lin, lin, lin, indexing="ij")
    p = np.stack([X, Y, Z], -1).astype(np.float32)
    return p, dict(body=_body(p), head=_head(p), eyes=_eyes(p),
                   tail=_tail(p), legs=_legs(p))


def _whole(f):
    d = smin(f["body"], f["head"], 0.060)
    d = smin(d, f["tail"], 0.045)
    d = smin(d, f["legs"], 0.040)
    return np.minimum(d, f["eyes"])          # eyes sit proud, they do not blend


def _occ(d, N):
    soft = 1.6/N
    o = np.clip(0.5 - d/soft, 0.0, 1.0)
    return (o*o*(3 - 2*o)).astype(np.float32)


def _masks(p, f, occ):
    """Five pieces that do not overlap and sum to the animal.

    Colours in field life are species, and species separate; three channels
    holding red, green and blue at the same voxel is a picture the rule is built
    not to hold. So each channel is given a region of the animal outright --
    eyes, head, back, belly and legs, tail -- and the picture is put back
    together at the far end by colouring the parts, which is what every other
    world on the page already does.
    """
    N = occ.shape[0]
    soft = 1.6/N
    sel = lambda d: np.clip(0.5 - d/(2.4*soft), 0.0, 1.0)
    smoothstep = lambda u: (lambda t: t*t*(3 - 2*t))(np.clip(u, 0, 1))

    eyes = sel(f["eyes"])
    head = sel(f["head"] - 0.010)
    tail = sel(f["tail"] - 0.010)
    legs = sel(f["legs"] - 0.010)
    # belly: below the body's own axis, which runs level, plus the feet
    belly = smoothstep((-0.012 - p[..., 1])/0.085)
    belly = np.maximum(belly, legs)

    take, left = [], np.ones_like(occ)
    for m in (eyes, head, tail, belly):
        g = np.clip(m, 0, 1)*left
        take.append(g)
        left = np.clip(left - g, 0, 1)
    take.insert(3, left)                       # the back is whatever is left
    return [np.clip(m*occ, 0, 1).astype(np.float32) for m in take]


PAL = [np.array([0.06, 0.05, 0.09], np.float32),    # eyes, near black
       np.array([0.45, 0.76, 0.32], np.float32),    # head, bright green
       np.array([0.96, 0.80, 0.34], np.float32),    # tail, gold
       np.array([0.20, 0.56, 0.28], np.float32),    # back, deeper green
       np.array([0.94, 0.90, 0.72], np.float32)]    # belly and legs, cream


def build(N=64):
    """Returns rgb (N,N,N,3) premultiplied and occupancy (N,N,N), both 0..1."""
    p, f = _fields(N)
    occ = _occ(_whole(f), N)
    parts = _masks(p, f, occ)
    rgb = sum(parts[i][..., None]*PAL[i] for i in range(len(PAL)))
    return np.clip(rgb, 0, 1).astype(np.float32), occ


def build_parts(N=64):
    """(P,N,N,N) disjoint parts summing to the occupancy, and the occupancy."""
    p, f = _fields(N)
    occ = _occ(_whole(f), N)
    return np.stack(_masks(p, f, occ), 0).astype(np.float32), occ


if __name__ == "__main__":
    parts, occ = build_parts(64)
    print("voxels", occ.shape, "filled %.3f" % (occ > 0.5).mean(),
          "mass %.0f" % occ.sum())
    print("parts", parts.shape, "sum err %.4f" % float(np.abs(parts.sum(0) - occ).max()),
          "mass", [round(float(m.sum())) for m in parts])
