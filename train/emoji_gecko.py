"""The lizard emoji as a target, cut into four parts that do not overlap.

The flat page's lizard was fitted so a seed disc of the right mass grows into
this exact glyph. This is the same target for the brains rule, and it is cut the
only way that rule allows: colours in field life are SPECIES, and species
separate, so the picture is clustered by colour and each species is given one
cluster to own. The four cluster colours are the palette, so putting the picture
back together at the end is just giving each species its own colour back --
which is what the page already does.
"""
import sys, os
import numpy as np
from PIL import Image, ImageFont, ImageDraw

FONT = "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf"
GLYPH = "\U0001F98E"          # lizard


def render(px=109):
    """NotoColorEmoji is a bitmap font and only speaks one size, 109."""
    f = ImageFont.truetype(FONT, px)
    im = Image.new("RGBA", (px*2, px*2), (0, 0, 0, 0))
    ImageDraw.Draw(im).text((px//2, px//2), GLYPH, font=f, embedded_color=True)
    return im.crop(im.getbbox())


def kmeans(x, k, iters=40, seed=0):
    rng = np.random.default_rng(seed)
    c = x[rng.choice(len(x), k, replace=False)]
    for _ in range(iters):
        lab = np.argmin(((x[:, None] - c[None])**2).sum(-1), 1)
        for j in range(k):
            m = lab == j
            if m.any():
                c[j] = x[m].mean(0)
    return c, lab


def build(N=64, k=4, blur=0.8, zoom=0.78):
    im = render()
    side = max(im.size)
    sq = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    sq.paste(im, ((side - im.size[0])//2, (side - im.size[1])//2))
    m = max(2, int(round(N*zoom)))
    sq = sq.resize((m, m), Image.LANCZOS)
    a = np.asarray(sq).astype(np.float32)/255.0
    canvas = np.zeros((N, N, 4), np.float32)
    o = (N - m)//2
    canvas[o:o + m, o:o + m] = a
    rgb, alpha = canvas[..., :3], canvas[..., 3]

    sel = alpha > 0.35
    cols, lab = kmeans(rgb[sel], k, seed=1)
    # darkest cluster first, so the palette reads in a stable order
    order = np.argsort(cols.sum(1))
    cols = cols[order]
    remap = np.zeros(k, int); remap[order] = np.arange(k)
    parts = np.zeros((k, N, N), np.float32)
    idx = np.where(sel)
    for j, (y, x) in enumerate(zip(*idx)):
        parts[remap[lab[j]], y, x] = alpha[y, x]
    if blur > 0:
        parts = gauss(parts, blur)
    return parts.astype(np.float32), cols.astype(np.float32), (alpha*sel).astype(np.float32)


def gauss(a, s):
    r = max(1, int(np.ceil(3*s)))
    kk = np.exp(-0.5*(np.arange(-r, r + 1)/s)**2); kk /= kk.sum()
    for ax in (-2, -1):
        out = np.zeros_like(a)
        for i, sh in enumerate(range(-r, r + 1)):
            out += kk[i]*np.roll(a, sh, ax)
        a = out
    return a


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    parts, cols, alpha = build(N)
    rgb = np.tensordot(parts, cols, ([0], [0]))
    grid = np.concatenate([np.repeat(alpha[..., None], 3, -1), np.clip(rgb, 0, 1)], 1)
    Image.fromarray((grid*255).astype(np.uint8)).resize(
        (grid.shape[1]*5, grid.shape[0]*5), Image.NEAREST).save(
        sys.argv[2] if len(sys.argv) > 2 else "emoji.png")
    print("parts", parts.shape, "mass frac",
          [round(float(p.sum()/parts.sum()), 3) for p in parts])
    print("palette", [[round(float(c), 3) for c in col] for col in cols])
