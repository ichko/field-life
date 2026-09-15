"""The Khronos dragon as a target, from the baked occupancy in dragon128.npz.

A free asset (CC0, glTF-Sample-Assets/DragonAttenuation) rather than something
drawn out of distance fields, and the honest reason it is the second option
rather than the first: nearly all of its interest is at the scale of one voxel.
Coiled round itself, its own folds close up the moment the target is blurred to
what the kernels can hold, and at forty cubed it comes out of that blur as a
lump. It is here because it is worth having, and because at ninety-six cubed and
up -- which is a GPU's problem, not a difficulty in principle -- it is the more
interesting animal by a distance.

The asset carries no colour of its own, so the colour comes off the geometry:
height above the local body, how buried a voxel is, and how far it sits from the
head. The parts are cut the same way.
"""
import os
import numpy as np
import torch
import torch.nn.functional as F

import colour
from bake_mesh import load as load_bake

BAKE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dragon128.npz")


def occupancy(N=64):
    """The bake, resampled to N. Downsampling is what gives it a soft edge."""
    v = load_bake(BAKE)
    if v.shape[0] == N:
        return v
    a = torch.from_numpy(v)[None, None]
    o = F.interpolate(a, size=(N, N, N), mode="trilinear", align_corners=False)
    return np.clip(o[0, 0].numpy()*1.25, 0.0, 1.0).astype(np.float32)


def _readings(occ):
    """Height above the local body, openness, and distance from the head."""
    N = occ.shape[0]
    lin = (np.arange(N) + 0.5)/N - 0.5
    Z, Y, X = np.meshgrid(lin, lin, lin, indexing="ij")
    solid = occ > 0.35

    ys = np.where(solid, Y, np.nan)
    with np.errstate(invalid="ignore"):
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            lo = np.nanmin(ys, axis=1, keepdims=True)
            hi = np.nanmax(ys, axis=1, keepdims=True)
    mid = np.nan_to_num((lo + hi)*0.5)
    span = np.nan_to_num(hi - lo) + 1e-3
    height = np.clip((Y - mid)/(0.6*span), -1, 1).astype(np.float32)

    cav = np.clip(colour.box(occ, max(2, N//26)), 0, 1)
    openness = np.clip(1.0 - (cav - 0.22)/0.50, 0, 1).astype(np.float32)

    idx = np.argwhere(solid)
    com = idx.mean(0)
    hz, hy, hx = idx[np.argmax(((idx - com)**2).sum(1))]
    along = np.sqrt((Z - lin[hz])**2 + (Y - lin[hy])**2 + (X - lin[hx])**2)/1.30
    return height, openness, np.clip(along, 0, 1).astype(np.float32)


def _masks(occ, P=5):
    """P pieces that do not overlap and sum to the animal: bands from the nose
    to the tail.

    Back against belly is the natural cut for something standing on legs, and
    it is the wrong one here -- this animal is coiled, so at most heights both
    its back and its belly are present and the split is a marbling rather than
    a shape. Distance from the head is the parameter that actually follows it.
    The bands are placed at equal shares of the mass, so no channel is handed
    most of the animal and none is handed a sliver; a fit divides its attention
    by what it is scored on, and five very unequal parts are four parts and a
    rounding error.
    """
    _, _, along = _readings(occ)
    w = occ.ravel()
    o = np.argsort(along.ravel())
    cw = np.cumsum(w[o])/max(w.sum(), 1e-9)
    cuts = [float(along.ravel()[o[int(np.searchsorted(cw, (i + 1)/P))]])
            for i in range(P - 1)]
    feather = 0.035

    step = lambda t: (lambda u: u*u*(3 - 2*u))(np.clip(t, 0, 1))
    below = [step((c - along)/feather) for c in cuts] + [np.ones_like(along)]
    parts, prev = [], np.zeros_like(along)
    for b in below:
        parts.append(np.clip(b - prev, 0, 1))
        prev = b
    return [np.clip(m*occ, 0, 1).astype(np.float32) for m in parts]


def build(N=64):
    occ = occupancy(N)
    return colour.colourise(occ), occ


def build_parts(N=64):
    occ = occupancy(N)
    return np.stack(_masks(occ), 0).astype(np.float32), occ


if __name__ == "__main__":
    parts, occ = build_parts(64)
    print("voxels", occ.shape, "filled %.3f" % (occ > 0.5).mean(),
          "mass %.0f" % occ.sum())
    print("parts", parts.shape, "sum err %.4f" % float(np.abs(parts.sum(0) - occ).max()),
          "mass", [round(float(m.sum())) for m in parts])
