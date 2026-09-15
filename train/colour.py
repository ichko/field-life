"""Colour a voxelised mesh from its own shape.

The asset carries no colour -- one flat glass material, no vertex colours and
no texture -- so the variation has to come from the geometry. Three readings of
the form, which between them put colour where an animal has it:

  height   how far above the local body the voxel sits: back against belly.
  cavity   how enclosed it is, from the field blurred wide: crevices go dark,
           ridges and horns catch the light. This is ambient occlusion, used as
           pigment rather than as shading.
  along    distance from the head, so the head reads warm and the tail cools.
"""
import numpy as np

def box(o, r):
    """Separable box blur of half-width r, wrapping is not wanted here."""
    out = o.astype(np.float32)
    for ax in (0, 1, 2):
        c = np.cumsum(np.pad(out, [(r + 1, r) if a == ax else (0, 0)
                                   for a in (0, 1, 2)], mode="edge"), axis=ax)
        sl = [slice(None)]*3
        sl[ax] = slice(2*r + 1, None); hi = c[tuple(sl)]
        sl[ax] = slice(0, -(2*r + 1)); lo = c[tuple(sl)]
        out = (hi - lo)/(2*r + 1)
    return out

def colourise(occ, head=None):
    """occ: (N,N,N) in [0,1], indexed [z,y,x]. Returns premultiplied rgb."""
    N = occ.shape[0]
    lin = (np.arange(N) + 0.5)/N - 0.5
    Z, Y, X = np.meshgrid(lin, lin, lin, indexing="ij")

    # height above the local body: the mid-height of the filled part of this
    # column, which follows a coiled animal where a fixed plane could not
    solid = occ > 0.35
    ys = np.where(solid, Y, np.nan)
    with np.errstate(invalid="ignore"):
        lo = np.nanmin(ys, axis=1, keepdims=True)
        hi = np.nanmax(ys, axis=1, keepdims=True)
    mid = np.nan_to_num((lo + hi)*0.5)
    span = np.nan_to_num(hi - lo) + 1e-3
    height = np.clip((Y - mid)/(0.6*span), -1, 1)

    # A tight blur, so only genuine folds count as buried -- a wide one just
    # reports "inside the animal", which is everywhere and tells you nothing.
    cav = np.clip(box(occ, max(2, N//26)), 0, 1)          # 0 open, 1 buried
    openness = np.clip(1.0 - (cav - 0.22)/0.50, 0, 1)

    if head is None:
        # the head is the filled voxel furthest from the centre of mass
        idx = np.argwhere(solid)
        com = idx.mean(0)
        head = idx[np.argmax(((idx - com)**2).sum(1))]
    hz, hy, hx = [float(v) for v in head]
    d = np.sqrt((Z - lin[int(hz)])**2 + (Y - lin[int(hy)])**2 + (X - lin[int(hx)])**2)
    along = np.clip(d/1.30, 0, 1)                          # 0 at the head

    jade  = np.array([0.10, 0.46, 0.37], np.float32)       # the body
    deep  = np.array([0.03, 0.13, 0.15], np.float32)       # down in the folds
    amber = np.array([0.68, 0.40, 0.09], np.float32)       # underside
    gold  = np.array([0.95, 0.78, 0.34], np.float32)       # crests and horns
    rust  = np.array([0.52, 0.20, 0.10], np.float32)       # towards the tail

    up = np.clip(height, 0, 1)[..., None]
    dn = np.clip(-height, 0, 1)[..., None]
    op = openness[..., None]
    al = along[..., None]

    c = jade*(1 - 0.55*al) + rust*(0.55*al)        # warms towards the tail
    dn = dn*dn                                     # keep gold to the underside
    c = c*(1 - dn*0.85) + amber*(dn*0.85)          # belly
    c = c + (gold - c)*(up*up*op*0.80)             # what sticks out catches gold
    c = c*(0.55 + 0.45*op)                         # what is buried goes dark
    c = c + (deep - c)*np.clip(1 - op*1.5, 0, 1)*0.35
    # scale banding, fine and only where the surface is open
    band = 0.5 + 0.5*np.sin(np.sqrt((X*X + Z*Z))*N*0.9 + Y*N*0.5)
    c = c*(0.90 + 0.20*band[..., None]*op)
    return (np.clip(c, 0, 1)*occ[..., None]).astype(np.float32)
