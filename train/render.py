"""
Draw a field the way index.html draws it.

FS_DRAW gives every channel a hue off a rotating ring and then combines them.
The default -- "Dominant" -- takes the hue from whichever colour dominates a
cell and the brightness from the total mass there, because adding twelve
saturated hues together just washes out to grey. Reproduced here so a rendered
frame and the running simulation look like the same thing.

    hue   = sum_c w_c * palette[c],  w_c = m_c^2      (Soft uses w_c = m_c)
    a     = 1 - exp(-sum_c m_c * expo)
    rgb   = hue / sum_c w_c * a                        + the screen's lift
"""

import numpy as np

# the faint lift FS_DRAW adds so an empty world is not pure black
GROUND = np.array([0.016, 0.024, 0.040])


def hue_ring(t):
    """A fully saturated hue at t in 0..1, exactly as index.html's hueRing."""
    h = ((t % 1) + 1) % 1 * 6
    x = 1 - abs(h % 2 - 1)
    return [[1, x, 0], [x, 1, 0], [0, 1, x], [0, x, 1], [x, 0, 1], [1, 0, x]][int(h) % 6]


def palette(C, name="Spectrum"):
    """makePalette. Only Spectrum is needed here -- it is what a trained world uses."""
    if name != "Spectrum":
        raise ValueError(f"palette {name!r} is not ported; the trained worlds use Spectrum")
    return np.array([hue_ring(i / C) for i in range(C)])


def blend(rho, pal, expo=2.2, mode="dominant", ground=True):
    """Combine every channel into one RGB image. rho is (C, H, W)."""
    m = np.maximum(rho, 0.0)
    total = m.sum(0)
    a = 1.0 - np.exp(-total * expo)

    if mode == "rgb":                       # the first three channels, as they are
        return np.clip(m[:3].transpose(1, 2, 0), 0, 1)
    if mode == "additive":
        add = np.einsum("chw,ck->hwk", m, pal)
        rgb = 1.0 - np.exp(-add * expo)
    elif mode == "winner":
        rgb = pal[m.argmax(0)] * a[..., None]
    else:                                   # dominant (w = m^2), or soft (w = m)
        w = m * m if mode == "dominant" else m
        hue = np.einsum("chw,ck->hwk", w, pal)
        rgb = hue / np.maximum(w.sum(0), 1e-12)[..., None] * a[..., None]

    if ground:
        rgb = rgb + GROUND
    return np.clip(rgb, 0, 1)


def channel(density, colour, expo=2.2):
    """One channel alone: its hue, brightness from its own mass."""
    a = 1.0 - np.exp(-np.maximum(density, 0) * expo)
    return a[..., None] * np.asarray(colour)[None, None, :]
