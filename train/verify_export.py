"""Run the exported rule straight out of the JSON and check it against torch.

The page will read this file and nothing else, so a mistake in the export --
a permutation applied once instead of twice, a stencil written z-fastest where
the page reads x-fastest, a constant folded in the wrong direction -- shows up
as a page that grows something plausible and wrong, which is the hardest kind of
bug to notice. So the arithmetic is done again here from the file alone, in
numpy, and compared against the model it came from.
"""
import argparse, json
import numpy as np
import torch

from field3d import load_field


def step(rho, f):
    """One step, from the JSON alone. rho: (C,N,N,N)."""
    C, N = rho.shape[0], rho.shape[1]
    M = N//2
    half = rho.reshape(C, M, 2, M, 2, M, 2).mean((2, 4, 6))

    # crowd: the channels added up, then one separable blur
    w = np.asarray(f["crowdBlur"], np.float32)
    kb = (len(w) - 1)//2
    crowd = half.sum(0)
    for ax in (0, 1, 2):
        acc = np.zeros_like(crowd)
        for i, wi in enumerate(w):
            acc += np.roll(crowd, i - kb, ax)*wi
        crowd = acc

    # the bank, a group of four channels at a time
    # Channels are in the exported order, so group g holds positions 4g..4g+3.
    # `channels` in the file records where they came from and is not an index
    # into anything here -- reading it as one is the mistake this check caught.
    Kc = np.zeros((C, M, M, M), np.float32)
    for gi, g in enumerate(f["groups"]):
        R = g["radius"]
        wg = np.asarray(g["w"], np.float32).reshape((2*R + 1)**3, 4)
        t = 0
        for dz in range(-R, R + 1):
            for dy in range(-R, R + 1):
                for dx in range(-R, R + 1):
                    sh = np.roll(np.roll(np.roll(half, -dz, 1), -dy, 2), -dx, 3)
                    for j in range(len(g["channels"])):
                        if wg[t, j] != 0.0:
                            Kc[4*gi + j] += sh[4*gi + j]*wg[t, j]
                    t += 1

    mat = np.asarray(f["mat"], np.float32)
    A = f["force"]*np.einsum("cd,dzyx->czyx", mat, Kc) - f["repel"]*crowd[None]

    # trilinear back up to full pitch, the way the page samples it
    # align_corners=False, and clamped at the edge rather than wrapped, because
    # that is what the fit was trained through. The difference is the outermost
    # half cell only, but it is the outermost half cell of every step.
    idx = (np.arange(N) + 0.5)/2.0 - 0.5
    i0 = np.floor(idx).astype(int)
    fr = np.clip(idx - i0, 0, 1).astype(np.float32)
    lo = np.clip(i0, 0, M - 1)
    hi = np.clip(i0 + 1, 0, M - 1)
    up = A
    for ax in (1, 2, 3):
        a0 = np.take(up, lo, axis=ax)
        a1 = np.take(up, hi, axis=ax)
        sh = [1, 1, 1, 1]; sh[ax] = N
        up = a0*(1 - fr.reshape(sh)) + a1*fr.reshape(sh)

    E = np.exp(np.clip(f["beta"]*up, -11.0, 11.0))
    box = lambda x: sum(np.roll(np.roll(np.roll(x, dz, 1), dy, 2), dx, 3)
                        for dz in (-1, 0, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1))
    Z = np.maximum(box(E), 1e-30)
    return E*box(rho/Z)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt"); ap.add_argument("json")
    ap.add_argument("--N", type=int, default=40)
    ap.add_argument("--steps", type=int, default=6)
    a = ap.parse_args()
    f = json.load(open(a.json))
    C, N, perm = f["C"], f["N"], np.asarray(f["order"])

    m = load_field(a.ckpt, a.N)
    m.soft = False
    masses = torch.tensor([f["seedMass"][list(perm).index(c)] for c in range(C)])
    with torch.no_grad():
        ref, _ = m.run(masses, a.steps)
    ref = ref[0].numpy()[perm]                      # into the exported order

    r = np.asarray(f["seedPattern"], np.float32).reshape(C, -1)
    h = f["seedHalf"]; D = 2*h + 1
    seed = np.zeros((C, N, N, N), np.float32)
    c0 = N//2
    seed[:, c0-h:c0+h+1, c0-h:c0+h+1, c0-h:c0+h+1] = r.reshape(C, D, D, D)
    got = seed
    for i in range(a.steps):
        got = step(got, f)

    print("mass per channel, torch vs json:")
    print("  torch", [round(float(v), 2) for v in ref.sum((1, 2, 3))[:6]])
    print("  json ", [round(float(v), 2) for v in got.sum((1, 2, 3))[:6]])
    den = max(float(np.abs(ref).max()), 1e-9)
    print("after %d steps: max abs difference %.3e, relative %.3e"
          % (a.steps, float(np.abs(ref - got).max()),
             float(np.abs(ref - got).max())/den))


if __name__ == "__main__":
    main()
