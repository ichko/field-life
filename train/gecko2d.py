"""The gecko as a flat target, cut into four parts that do not overlap.

The volume fit learned this the hard way and wrote it down: colours in field
life are SPECIES, and species separate. Three channels holding red, green and
blue at the same cell is a picture the rule is built not to hold, and a fit
asked for it spends its whole budget losing that argument. So the animal is cut
into regions that are each one species' own, they sum to the whole, and the
picture is put back together at the end by giving each part a colour.

The volume cut them by HEIGHT -- back, underside, head -- which is exactly the
axis a plan view throws away, so this cuts them in the plane instead: head,
torso, limbs, tail. Four parts, four species.
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gecko import build, spine, width


def build2d(N=64, blur=1.0, zoom=0.72):
    # The animal is flat and splayed, so the plan view is the silhouette that
    # still reads as an animal: legs out to the sides, a bend at the elbow, a
    # long curving tail, a head wider than the neck.
    _, occ = build(N)                      # (Z, Y, X)
    plan = occ.max(1)                      # look down the up-axis -> (Z, X)

    lin = (np.arange(N) + 0.5)/N - 0.5
    Z, X = np.meshgrid(lin, lin, indexing="ij")
    t = np.clip((X + 0.44)/0.88, 0, 1)     # tail tip 0, snout 1
    off = np.abs(Z - spine(t)[..., 2])     # how far off the spine, in plan
    hw = np.maximum(width(t)/1.32, 1e-4)

    def smooth(u):
        u = np.clip(u, 0, 1)
        return u*u*(3 - 2*u)
    head = smooth((t - 0.845)/0.06)
    tail = smooth((0.30 - t)/0.08)
    # off the spine by more than the body is wide: that is a limb
    limb = smooth((off - 1.15*hw)/(0.9*hw))
    mid = (1 - head)*(1 - tail)

    parts = np.stack([plan*head,                 # 0  head
                      plan*mid*(1 - limb),       # 1  torso
                      plan*mid*limb,             # 2  limbs
                      plan*tail], 0)             # 3  tail
    # Shrink and centre: at full size the tail runs off the edge and wraps
    # round the torus onto the head, which is not a target anyone can hit.
    if zoom < 1.0:
        parts = np.stack([_fit(q, N, zoom) for q in parts], 0)
        plan = _fit(plan, N, zoom)
    if blur > 0:
        # The rule's smallest feature is about a cell, so a target finer than
        # that is not a hard target but an impossible one.
        parts = gauss(parts, blur)
        plan = gauss(plan[None], blur)[0]
    return parts.astype(np.float32), plan.astype(np.float32)


def _fit(a, N, z):
    """Area-average down by z and paste in the middle of an N x N field."""
    m = max(2, int(round(N*z)))
    src = np.asarray(a, np.float32)
    yi = (np.arange(m) + 0.5)*N/m
    xi = (np.arange(m) + 0.5)*N/m
    y0 = np.clip(yi.astype(int), 0, N - 1); x0 = np.clip(xi.astype(int), 0, N - 1)
    small = src[np.ix_(y0, x0)]
    out = np.zeros((N, N), np.float32)
    o = (N - m)//2
    out[o:o + m, o:o + m] = small
    return out


def gauss(a, s):
    r = max(1, int(np.ceil(3*s)))
    k = np.exp(-0.5*(np.arange(-r, r + 1)/s)**2); k /= k.sum()
    for ax in (-2, -1):
        out = np.zeros_like(a)
        for i, sh in enumerate(range(-r, r + 1)):
            out += k[i]*np.roll(a, sh, ax)
        a = out
    return a


if __name__ == "__main__":
    from PIL import Image
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 96
    parts, plan = build2d(N)
    PAL = np.array([[1, .75, .15], [.25, .85, .35], [.35, .6, 1], [1, .4, .55]], np.float32)
    rgb = np.tensordot(parts, PAL, ([0], [0]))
    img = np.concatenate([np.repeat(plan[..., None], 3, -1), np.clip(rgb, 0, 1)], 1)
    Image.fromarray((img*255).astype(np.uint8)).resize((img.shape[1]*4, img.shape[0]*4),
        Image.NEAREST).save(sys.argv[2] if len(sys.argv) > 2 else "gecko2d.png")
    print("parts", parts.shape, "mass per part",
          [round(float(p.sum()/plan.sum()), 3) for p in parts])
