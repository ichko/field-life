"""
The target the field has to sculpt, and the mass it is given to do it with.

Growing NCA trains a network to build the lizard emoji from a single seed cell,
and it can do that because its update rule creates matter. MaCE cannot: it only
ever moves mass around, exactly, per channel. So the seed has to arrive already
holding what the picture will cost.

That is the whole trick here. Read the lizard's premultiplied RGB, integrate
each of the three channels, and hand the seed disc exactly that much red, that
much green, that much blue. Solving the task then means transporting the mass
into the right arrangement -- never manufacturing it.

Premultiplied is load-bearing. Density is non-negative and the background must
be empty, so alpha cannot be a separate channel: it has to BE the mass. With
target = rgb * alpha the transparent surround costs nothing, and a loss on the
first three channels alone pins down the silhouette as well as the colour.

The hidden channels are not constrained by the picture, so they are the free
working memory -- the only place the system can keep a signal that is not
supposed to be visible. They get mass too, at least HIDDEN_FLOOR of the mean
visible channel, jittered per run so nothing depends on one lucky allocation.

    python3 train/target.py --show
"""

import argparse
import json
import math
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
EMOJI_FONT = "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf"
EMOJI_SIZE = 109               # NotoColorEmoji is a bitmap font; this is its one size

LIZARD = "\U0001F98E"          # the Growing NCA target
GRID = 64                      # training lattice
SPAN = 40                      # how many cells the animal spans
HIDDEN_FLOOR = 0.35            # least mass a hidden channel gets, vs mean visible


def render_emoji(char=LIZARD, span=SPAN, grid=GRID):
    """Premultiplied RGB of the emoji, centred in a grid x grid field."""
    font = ImageFont.truetype(EMOJI_FONT, EMOJI_SIZE)
    img = Image.new("RGBA", (EMOJI_SIZE * 2, EMOJI_SIZE * 2), (0, 0, 0, 0))
    ImageDraw.Draw(img).text((EMOJI_SIZE // 2, EMOJI_SIZE // 2), char,
                             font=font, embedded_color=True)
    img = img.crop(img.getbbox())
    # square it before scaling, so the animal is not stretched
    side = max(img.size)
    sq = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    sq.paste(img, ((side - img.width) // 2, (side - img.height) // 2))
    small = np.asarray(sq.resize((span, span), Image.LANCZOS), dtype=np.float64) / 255.0

    rgb, a = small[..., :3], small[..., 3:4]
    pre = np.clip(rgb * a, 0.0, 1.0)                 # premultiplied: background is empty

    out = np.zeros((grid, grid, 3))
    o = (grid - span) // 2
    out[o:o + span, o:o + span] = pre
    return out.transpose(2, 0, 1)                     # (3, grid, grid)


def seed_masses(target, C, hidden_floor=HIDDEN_FLOOR, rng=None):
    """Per-channel mass for the seed: the picture's own, then a floor for the rest."""
    rng = rng or np.random.default_rng(0)
    visible = target.sum(axis=(1, 2))
    mean_visible = float(visible.mean())
    hidden = [mean_visible * (hidden_floor + rng.random() * (1.0 - hidden_floor))
              for _ in range(C - 3)]
    return np.array(list(visible) + hidden)


def seed_field(masses, grid=GRID, radius=None, softness=1.5):
    # radius defaults to a tenth of the GRID, which is the wrong thing to scale
    # with once the grid carries padding: a 120-cell animal on a 256 grid gets
    # a 25-cell seed and has to carry mass 73 cells to reach its far end, which
    # sets the BPTT window and therefore the cost of every iteration. Sizing
    # the seed to the animal instead shortens the journey without making the
    # task easier in any way that matters -- the seed is a blob of the right
    # stuff, not a hint about where the stuff goes.
    """A soft disc at the centre carrying exactly `masses` in each channel.

    One disc, not one cell: MaCE moves mass at most one cell per step, so a
    single lit cell would need ~grid/2 steps before the far side of the animal
    could even be reached, and every one of those steps still has to be
    backpropagated through.
    """
    radius = radius or grid * 0.10
    y, x = np.mgrid[0:grid, 0:grid]
    d = np.hypot(y - (grid - 1) / 2, x - (grid - 1) / 2)
    disc = np.clip((radius - d) / softness, 0.0, 1.0)
    disc = disc * disc * (3 - 2 * disc)               # smoothstep, as the seed shader does
    disc = disc / disc.sum()
    return np.stack([m * disc for m in masses])


def budget(target, grid=GRID, cells_per_step=1.0):
    """Fewest steps that could possibly work: mass moves one cell per step."""
    ys, xs = np.nonzero(target.sum(axis=0) > 1e-6)
    if len(ys) == 0:
        return 0
    cy, cx = (grid - 1) / 2, (grid - 1) / 2
    return int(math.ceil(float(np.hypot(ys - cy, xs - cx).max()) / cells_per_step))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channels", type=int, default=6)
    ap.add_argument("--grid", type=int, default=GRID)
    ap.add_argument("--span", type=int, default=SPAN)
    ap.add_argument("--show", action="store_true", help="write target.png and seed.png")
    ap.add_argument("--out", default=os.path.join(HERE, "target.npz"))
    ap.add_argument("--seed-json", default=None,
                    help="also write the seed field as flat JSON, for verify_browser.mjs")
    a = ap.parse_args()

    target = render_emoji(span=a.span, grid=a.grid)
    masses = seed_masses(target, a.channels)
    seed = seed_field(masses, grid=a.grid)

    np.savez(a.out, target=target, seed=seed, masses=masses,
             grid=a.grid, span=a.span, channels=a.channels)

    print(f"target  {target.shape}  premultiplied, background max "
          f"{target[:, :4, :4].max():.2e}")
    print(f"mass per channel:")
    for c, m in enumerate(masses):
        tag = "RGB"[c] if c < 3 else f"hidden {c - 3}"
        print(f"  ch{c}  {tag:<9} {m:9.3f}")
    print(f"seed disc carries {seed.sum(axis=(1, 2))[:3].round(3).tolist()} visible "
          f"(target {target.sum(axis=(1, 2)).round(3).tolist()})")
    print(f"reach: the furthest lit cell is {budget(target, a.grid)} cells from the "
          f"seed, so no run shorter than that many steps can succeed")
    print(f"wrote {a.out}")

    if a.seed_json:
        with open(a.seed_json, "w") as fh:
            json.dump(seed.flatten().tolist(), fh)
        print(f"wrote {a.seed_json}")

    if a.show:
        for name, arr in (("target", target), ("seed", seed[:3])):
            v = np.clip(arr.transpose(1, 2, 0), 0, 1)
            if name == "seed":
                v = v / max(v.max(), 1e-9)
            p = os.path.join(HERE, f"{name}.png")
            Image.fromarray((v * 255).astype(np.uint8)).resize(
                (a.grid * 6, a.grid * 6), Image.NEAREST).save(p)
            print(f"wrote {p}")


if __name__ == "__main__":
    main()
