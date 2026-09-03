"""A small lizard, as a voxel grid, built out of signed distances.

Shaped the way the emoji is: seen from above, flattened, legs splayed out to the
sides with a bend at the elbow, a long curving tail, a head wider than the neck.
A side-on lizard is mostly a tube at this resolution; a splayed one still reads
as an animal when it is only fifty voxels across, which is the whole point --
the field has to be able to hold it.
"""
import numpy as np

def smin(a, b, k):
    """Polynomial smooth minimum -- unions that blend instead of creasing."""
    h = np.clip(0.5 + 0.5*(b - a)/k, 0.0, 1.0)
    return b*(1 - h) + a*h - k*h*(1 - h)

def sd_capsule(p, a, b, r0, r1=None):
    """Round cone from a to b, radius r0 to r1."""
    a = np.asarray(a, np.float32); b = np.asarray(b, np.float32)
    if r1 is None: r1 = r0
    pa = p - a
    ba = b - a
    h = np.clip((pa @ ba)/max(float(ba @ ba), 1e-9), 0.0, 1.0)
    d = pa - h[..., None]*ba
    return np.sqrt((d*d).sum(-1)) - (r0 + (r1 - r0)*h)

def sd_ellipsoid(p, c, r):
    q = (p - np.asarray(c, np.float32))/np.asarray(r, np.float32)
    k0 = np.sqrt((q*q).sum(-1))
    k1 = np.sqrt(((q/np.asarray(r, np.float32))**2).sum(-1))
    return k0*(k0 - 1.0)/np.maximum(k1, 1e-9)

# The spine, tail tip (t=0) to snout (t=1), lying in the horizontal plane with
# a lazy S in it. y is up, and the animal is flat, so y barely moves.
def spine(t):
    t = np.asarray(t, np.float32)
    x = -0.44 + 0.88*t
    z = 0.10*np.sin(3.1*t + 0.35)*(1.15 - t)
    # The back arches. Tail tip on the ground, a lift over the hips, a dip at
    # the waist and a higher rise at the shoulders, then the neck carries the
    # head up again -- so the animal has a profile and not just a plan.
    y = (-0.150
         + 0.150*np.clip((t - 0.10)/0.22, 0, 1)        # the tail comes up
         + 0.052*np.exp(-((t - 0.50)/0.16)**2)         # hips
         + 0.080*np.exp(-((t - 0.78)/0.15)**2)         # shoulders
         + 0.050*np.clip((t - 0.86)/0.14, 0, 1))       # neck lifts the head
    return np.stack([x, y, z], -1)

def width(t):
    """Half-width across, in the horizontal plane."""
    t = np.asarray(t, np.float32)
    return (0.012
            + 0.050*np.exp(-((t - 0.52)/0.13)**2)      # hips
            + 0.055*np.exp(-((t - 0.75)/0.13)**2)      # chest
            + 0.062*np.exp(-((t - 0.93)/0.075)**2)     # head
            )*np.clip(t*5.0, 0.06, 1.0)

def build(N=64):
    """Returns rgb (N,N,N,3) premultiplied and occupancy (N,N,N), both 0..1."""
    lin = (np.arange(N) + 0.5)/N - 0.5
    Z, Y, X = np.meshgrid(lin, lin, lin, indexing="ij")
    p = np.stack([X, Y, Z], -1).astype(np.float32)
    # A gentle squash in y, so the body is wider than it is tall without being
    # a pancake -- the animal has to read from the side as well as from above.
    flat = p*np.array([1.0, 1.32, 1.0], np.float32)

    ts = np.linspace(0.0, 1.0, 64)
    body = np.full(p.shape[:3], 10.0, np.float32)
    for i in range(len(ts) - 1):
        body = smin(body, sd_capsule(flat, spine(ts[i]), spine(ts[i+1]),
                                     float(width(ts[i])), float(width(ts[i+1]))), 0.030)

    # head: wider than the neck, with a blunt snout
    hc = spine(0.94)
    head = sd_ellipsoid(flat, hc + np.array([0.020, 0.004, 0.0]),
                        np.array([0.105, 0.070, 0.078]))
    body = smin(body, head, 0.030)

    # Four legs, splayed. Elbow out to the side, foot forward of it, then toes
    # -- a lizard's stance, and the thing that makes the silhouette readable.
    legs = np.full(p.shape[:3], 10.0, np.float32)
    ground = -0.185
    for tt, sweep in ((0.76, 0.050), (0.50, -0.042)):
        root = spine(tt)
        for s in (1.0, -1.0):
            sh = root + np.array([0.0, -0.010, s*0.030])
            # elbow held out to the side and still high, foot planted on the
            # ground below it: a sprawling stance, with daylight under the belly
            el = np.array([root[0] + sweep*0.30, root[1] - 0.045, s*0.115])
            ft = np.array([root[0] + sweep*1.15, ground, s*0.150])
            legs = smin(legs, sd_capsule(flat, sh, el, 0.033, 0.025), 0.020)
            legs = smin(legs, sd_capsule(flat, el, ft, 0.025, 0.019), 0.018)
            for ang in (-0.55, 0.0, 0.55):
                tip = ft + np.array([np.cos(ang)*sweep*0.75,
                                     0.004, s*abs(np.sin(ang))*0.050 + s*0.028])
                legs = smin(legs, sd_capsule(flat, ft, tip, 0.017, 0.009), 0.013)
    body = smin(body, legs, 0.026)

    # a low ridge of scales down the back
    for tt in np.linspace(0.30, 0.90, 12):
        h = 0.030 + 0.026*np.sin(np.pi*np.clip((tt - 0.30)/0.60, 0, 1))
        c = spine(tt) + np.array([0.0, float(h), 0.0])
        body = smin(body, sd_ellipsoid(flat, c, np.array([0.024, 0.040, 0.018])), 0.022)

    eyes = np.full(p.shape[:3], 10.0, np.float32)
    for s in (1.0, -1.0):
        e = hc + np.array([0.034, 0.020, s*0.052])
        eyes = np.minimum(eyes, sd_ellipsoid(flat, e, np.array([0.024, 0.030, 0.024])))
    body = np.minimum(body, eyes - 0.005)

    soft = 1.5/N
    occ = np.clip(0.5 - body/soft, 0.0, 1.0)
    occ = (occ*occ*(3 - 2*occ)).astype(np.float32)

    # Colour: green over the back, cream underneath, dark eyes. The line
    # between them follows the spine's own height and the local thickness, so
    # it stays on the flank whether the back is arched or the tail is thin --
    # measured against a fixed height it would ride up over the shoulders and
    # swallow the head.
    t_of = np.clip((p[..., 0] + 0.44)/0.88, 0, 1)
    hh = width(t_of)/1.32
    up = p[..., 1] - spine(t_of)[..., 1]
    belly = np.clip(0.5 - (up + 0.30*hh)/(0.9*hh + 1e-4), 0.0, 1.0)
    belly = belly*belly*(3 - 2*belly)
    band = 0.5 + 0.5*np.sin(t_of*30.0 + 0.6)
    back = np.stack([0.13 + 0.20*band, 0.52 - 0.16*band, 0.16 + 0.12*band], -1)
    bell = np.stack([0.85 + 0.0*band, 0.83 + 0.0*band, 0.55 + 0.0*band], -1)
    rgb = back*(1 - belly[..., None]) + bell*belly[..., None]
    iseye = np.clip(0.5 - eyes/soft, 0.0, 1.0)
    rgb = rgb*(1 - iseye[..., None]) + np.array([0.05, 0.04, 0.08], np.float32)*iseye[..., None]
    return (rgb*occ[..., None]).astype(np.float32), occ

if __name__ == "__main__":
    rgb, occ = build(64)
    print("voxels", occ.shape, "filled %.3f" % (occ > 0.5).mean(),
          "mass %.0f" % occ.sum())


def build_parts(N=64):
    """The same animal, cut into pieces that do not overlap: back, underside,
    head.

    Three channels holding red, green and blue at the same voxel is a picture,
    but it is not something this rule likes to hold -- colours in field life are
    species, and species separate. Asking three of them to sit on top of each
    other everywhere is asking the rule to do the one thing it is built not to
    do, and the fit spends its whole budget losing that argument.

    Parts do not have that problem. Each channel gets a region of the animal
    that is its own, the three sum to the whole, and the picture is put back
    together at the far end by giving each part a colour. Which is what every
    other world on the page already does.
    """
    import numpy as np
    lin = (np.arange(N) + 0.5)/N - 0.5
    Z, Y, X = np.meshgrid(lin, lin, lin, indexing="ij")
    p = np.stack([X, Y, Z], -1).astype(np.float32)
    _, occ = build(N)

    t_of = np.clip((p[..., 0] + 0.44)/0.88, 0, 1)
    hh = width(t_of)/1.32
    up = p[..., 1] - spine(t_of)[..., 1]
    low = np.clip(0.5 - (up + 0.30*hh)/(0.9*hh + 1e-4), 0.0, 1.0)
    low = low*low*(3 - 2*low)
    head = np.clip((t_of - 0.845)/0.06, 0, 1)
    head = head*head*(3 - 2*head)

    parts = np.stack([occ*(1 - low)*(1 - head),      # back and crest
                      occ*low*(1 - head),            # underside, legs, tail
                      occ*head], 0)                  # the head
    return parts.astype(np.float32), occ
