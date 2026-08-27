"""
field-life's step, in PyTorch.

This is a port of the simulation in index.html, not a reimplementation of the
idea: every constant, every clamp and every normalisation is the one the shader
uses, so a field stepped here and a field stepped in the browser stay on top of
each other. train/parity.py checks that claim against a real WebGL run.

Two things make the port tractable:

  * At the training grid (N <= 96) index.html never builds a mip pyramid --
    `allocate` only pushes a level while min(px,py)>>1 >= 48 -- so `kernelMip`
    is pinned at 0, the convolution runs at full resolution, and the
    Catmull-Rom upsample that follows it is the identity (uNm == uN gives
    t = 0, and cr(a,b,c,d,0) = b).
  * A kernel with three or more lobes, or any angular term, makes
    `separablePlan()` return null, so the browser takes the 2-D stencil path --
    the one ported here. Anything we train therefore runs in the browser
    through exactly this arithmetic.

The step itself, per channel c on a torus:

    U_c    = rho_c  *  K_c                  kernel: zero-mean, unit-L1
    N_c    = rho_c  *  G / sum(G)           G = gaussian, sigma = 0.22 * KR
    A_c    = force * sum_d M[c,d] U_d  -  repel * sum_d N_d
    E      = exp(clamp(beta * A, -20, 20))
    Z      = 3x3 sum of E
    rho'   = E * (3x3 sum of rho/Z)

The last three lines are MaCE, and they conserve mass exactly, per channel:
sum(rho') = sum_kl (rho_kl/Z_kl) * sum_{nb of kl} E = sum_kl rho_kl, because a
cell sits inside its own 3x3 neighbourhood. Mass only ever moves one cell per
step, which is what sets how many steps a target needs.
"""

import math

import torch
import torch.nn.functional as F

KMAX = 15            # kernel half-width in working cells, as in index.html


# --------------------------------------------------------------- torus helpers

def _wrap(x, pad):
    """Circular pad. F.pad's circular mode refuses pads >= the input size."""
    while pad > 0:
        p = min(pad, x.shape[-1] - 1, x.shape[-2] - 1)
        x = F.pad(x, (p, p, p, p), mode="circular")
        pad -= p
    return x


def corr2d(rho, weight):
    """Per-channel toroidal cross-correlation. rho (B,C,H,W) or (C,H,W).

    Cross-correlation, not convolution: the shader reads src at (lx+dx, y+dy)
    against kernel texel (dx+KR, dy+KR), which is what conv2d already does.
    """
    flat = rho.ndim == 3
    x = rho.unsqueeze(0) if flat else rho
    out = F.conv2d(_wrap(x, weight.shape[-1] // 2), weight.unsqueeze(1),
                   groups=x.shape[1])
    return out.squeeze(0) if flat else out


def sum3x3(x):
    """3x3 toroidal sum, per channel."""
    flat = x.ndim == 3
    v = x.unsqueeze(0) if flat else x
    ones = torch.ones(v.shape[1], 1, 3, 3, dtype=v.dtype, device=v.device)
    out = F.conv2d(_wrap(v, 1), ones, groups=v.shape[1])
    return out.squeeze(0) if flat else out


# ------------------------------------------------------------------- the step

def crowd_gaussian(KR, kern, C, dtype=torch.float64, device="cpu"):
    """The tight blur FS_CONV computes beside the interaction integral.

    Weights are exp(-r^2/2rn^2) over the same window as the kernel, normalised
    by their own sum. The shader skips a tap when all four channels of an RGBA
    layer have zero kernel weight there AND g < 1e-4; that drops ~1e-5 of the
    mass off the far rim, and it is reproduced here so parity is exact rather
    than merely close.
    """
    rn = max(1.0, KR * 0.22)
    d = torch.arange(-KR, KR + 1, dtype=dtype, device=device)
    r2 = d[:, None] ** 2 + d[None, :] ** 2
    g = torch.exp(-0.5 * r2 / (rn * rn))
    L = (C + 3) // 4
    out = torch.empty(C, 2 * KR + 1, 2 * KR + 1, dtype=dtype, device=device)
    for l in range(L):
        lo, hi = 4 * l, min(4 * l + 4, C)
        # a layer's texel is vec4; `w == vec4(0.0)` needs all four zero, and a
        # layer short of four channels reads zeros in the unused components.
        zero = (kern[lo:hi] == 0).all(dim=0)
        gl = torch.where(zero & (g < 1e-4), torch.zeros_like(g), g)
        out[lo:hi] = (gl / gl.sum()).expand(hi - lo, -1, -1)
    return out


def step(rho, kern, mat, force, repel, beta):
    """One simulation tick. rho (B,C,H,W) or (C,H,W); kern (C,K,K); mat (C,C).

    mat's row is the feeling channel and its column the felt one, matching
    FS_AFF's texelFetch(uMat, ivec2(d, cc)) against a C-wide R32F texture.
    """
    KR = kern.shape[-1] // 2
    C = kern.shape[0]
    U = corr2d(rho, kern)
    N = corr2d(rho, crowd_gaussian(KR, kern, C, rho.dtype, rho.device))

    # `mat` may be a plain C x C matrix -- the law as it ships, linear in the
    # convolved fields -- or a callable, which is rung 1 of the design doc: the
    # matrix multiply becomes a small per-cell network, which is structurally
    # what an NCA's update MLP is. Everything downstream is untouched, so MaCE
    # still conserves mass exactly either way.
    if callable(mat):
        mixed = mat(U)
    else:
        ax = "cd,dhw->chw" if rho.ndim == 3 else "cd,bdhw->bchw"
        mixed = torch.einsum(ax, mat, U)
    A = force * mixed - repel * N.sum(-3, keepdim=True)

    # +-20, not +-60: share is rho/Z in float32, and a wider span flushes a cold
    # cell's share to zero, which would delete mass. See FS_EXPA.
    E = torch.exp(torch.clamp(beta * A, -20.0, 20.0))
    Z = sum3x3(E)
    S = rho / Z.clamp_min(1e-32)
    return E * sum3x3(S)


def run(rho, kern, mat, force, repel, beta, steps):
    for _ in range(steps):
        rho = step(rho, kern, mat, force, repel, beta)
    return rho


# ------------------------------------------------- baking: the legacy JS path

def _profile(kind, rr, terms, beta_pl):
    """The radial profile v(rr), matching bakeKernel's inner switch."""
    if kind == "disc":
        return torch.ones_like(rr)
    if kind == "pl":
        B = min(0.9, max(0.02, 0.3 if beta_pl is None else beta_pl))
        return torch.where(rr < B, rr / B - 1.0,
                           1.0 - (2.0 * rr - 1.0 - B).abs() / (1.0 - B))
    v = torch.zeros_like(rr)
    for a, r, w in terms:
        v = v + a * torch.exp(-(((rr - r) / w) ** 2))
    return v


def _subsampled(KR, dtype, device):
    """The 3x3 sub-grid bakeKernel samples each cell on: offsets of +-1/3."""
    c = torch.arange(-KR, KR + 1, dtype=dtype, device=device)
    y, x = torch.meshgrid(c, c, indexing="ij")
    o = torch.tensor([-1 / 3, 0.0, 1 / 3], dtype=dtype, device=device)
    px = x[None, None] + o[None, :, None, None]
    py = y[None, None] + o[:, None, None, None]
    return px.expand(3, 3, *x.shape), py.expand(3, 3, *x.shape)


def _fold(a, sym):
    """Symmetry fold, on a grid whose centre is the middle element."""
    if sym == "lateral":
        return (a + a.flip(-1)) / 2
    if sym == "quad":
        return (a + a.flip(-1) + a.flip(-2) + a.flip(-1).flip(-2)) / 4
    if sym == "rot4":
        return (a + a.rot90(1, (-2, -1)) + a.rot90(2, (-2, -1)) + a.rot90(3, (-2, -1))) / 4
    return a


def _finish(v, KR, feather, zero_mean):
    """Feather the rim, zero the mean against the taper, normalise to unit L1.

    The mean is removed as a multiple of the taper rather than as a flat
    constant: a flat subtraction puts weight straight back into the rim cells
    the taper had just brought to zero, leaving a hard circular cliff -- and a
    hard circle on a square lattice is what shows up as square artefacts.
    """
    c = torch.arange(-KR, KR + 1, dtype=v.dtype, device=v.device)
    y, x = torch.meshgrid(c, c, indexing="ij")
    rr = torch.hypot(x, y) / KR
    f = ((1 - rr) / max(1e-3, feather)).clamp(0, 1)
    taper = torch.where(rr > 1, torch.zeros_like(f), f * f * (3 - 2 * f))

    v = torch.where(rr > 1, torch.zeros_like(v), v) * taper
    if zero_mean:
        v = v - (v.sum() / taper.sum().clamp_min(1e-9)) * taper
    l1 = v.abs().sum()
    return torch.where(l1 > 0, v / l1, v)


def bake_kernel_legacy(k, KR, dtype=torch.float64, device="cpu"):
    """Exact port of bakeKernel() in index.html. For parity, not for training.

    `perlin` is deliberately unsupported: it is a seeded integer hash whose JS
    coercions would have to be reproduced bit for bit, and nothing in this
    experiment trains one.
    """
    kind = k.get("type", "radial")
    if kind == "perlin":
        raise NotImplementedError("perlin kernels are not ported; train radial/polar ones")

    wmin = 2.0 / KR
    terms = [(t["a"], t["r"], max(t["w"], wmin)) for t in k.get("terms", [])]

    px, py = _subsampled(KR, dtype, device)
    rr = torch.hypot(px, py) / KR
    v = _profile(kind, rr, terms, k.get("beta"))
    # dividing by 9 rather than by the hit count is what feathers the rim here
    v = torch.where(rr > 1, torch.zeros_like(v), v).sum(dim=(0, 1)) / 9

    v = _fold(v, k.get("sym", "radial"))
    return _finish(v, KR, k.get("feather", 0.1), zero_mean=kind not in ("pl", "disc"))


def bake_bank_legacy(kernels, C, dtype=torch.float64, device="cpu"):
    """uploadKernels(): shared half-width from the widest reach, others inset.

    Note the round() in `kr` -- a channel's reach changes the RESOLUTION it is
    baked at, not a scale on a shared grid. That is exactly the step that makes
    R untrainable, and bake_bank_polar below replaces it.
    """
    Rmax = max(max(k["R"] for k in kernels[:C]), 1.0)
    KR = max(3, min(KMAX, round(Rmax / 1)))      # kernelMip is 0 at these grids
    K = 2 * KR + 1
    out = torch.zeros(C, K, K, dtype=dtype, device=device)
    for c in range(C):
        kr = max(2, round(KR * min(1.0, kernels[c]["R"] / Rmax)))
        small = bake_kernel_legacy(kernels[c], kr, dtype, device)
        o = KR - kr
        out[c, o:o + 2 * kr + 1, o:o + 2 * kr + 1] = small
    return out


# ------------------------------------------------ baking: the trainable path

class PolarKernels(torch.nn.Module):
    """One kernel per channel, as a polar function whose parameters are learned.

        K_c(r, theta) = sum_l  a[c,l] * exp(-((r - mu[c,l]) / w[c,l])^2)
                                      * cos(m[l] * theta + phase[c,l])

    with r = |x| / R_c, then feathered, zero-meaned and normalised to unit L1
    exactly as bakeKernel does -- so `force` keeps meaning the same thing and a
    trained kernel is a legal index.html kernel.

    The angular orders m are FIXED integers (they index a Fourier basis, so
    they cannot be gradient-descended); the amplitude decides whether a lobe is
    used at all. m = 0 is Lenia's radial ring. m = 1 is a signed gradient along
    an axis -- which is what a Sobel filter is, so an NCA's fixed
    identity/Sobel-x/Sobel-y perception is the special case m in {0, 1, 1}.
    Orders above 0 are what let a mass-conserving field break rotational
    symmetry, and an animal is not rotationally symmetric.

    Everything is baked on ONE grid of half-width KR, with R_c entering as a
    continuous radial scale. That is the one place this departs from today's
    uploadKernels(), which rounds R_c to a grid resolution; the departure is
    what makes R differentiable, and the feather is what makes the gradient
    well behaved, because the profile reaches zero smoothly at r = 1 instead of
    being cut there. index.html needs the matching change before a trained
    kernel loads faithfully -- see docs/nca-experiment.md.
    """

    def __init__(self, C, orders=(0, 0, 0, 1, 1, 2), KR=13, feather=0.25,
                 R_init=None, dtype=torch.float32, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        L = len(orders)
        self.C, self.KR, self.feather = C, KR, feather
        self.register_buffer("m", torch.tensor(orders, dtype=dtype))
        # a lobe of order 0 has no meaningful phase (cos(phase) just rescales a)
        self.register_buffer("has_phase", torch.tensor([o != 0 for o in orders]))

        R0 = R_init if R_init is not None else KR * 0.85
        self.logR = torch.nn.Parameter(torch.full((C,), math.log(R0), dtype=dtype))
        self.a = torch.nn.Parameter(torch.randn(C, L, generator=g, dtype=dtype) * 0.5)
        self.mu_raw = torch.nn.Parameter(torch.randn(C, L, generator=g, dtype=dtype) * 0.8)
        self.logw = torch.nn.Parameter(
            torch.full((C, L), math.log(0.25), dtype=dtype)
            + torch.randn(C, L, generator=g, dtype=dtype) * 0.3)
        self.phase = torch.nn.Parameter(
            torch.rand(C, L, generator=g, dtype=dtype) * 2 * math.pi)

    def radii(self):
        """Reach in working cells, kept inside the stencil the browser bakes."""
        return self.logR.exp().clamp(2.0, float(self.KR))

    def forward(self):
        KR, dt, dev = self.KR, self.a.dtype, self.a.device
        px, py = _subsampled(KR, dt, dev)                       # (3,3,K,K)
        rad = torch.hypot(px, py)[None]                         # (1,3,3,K,K)
        theta = torch.atan2(py, px)[None]

        R = self.radii()[:, None, None, None, None]
        rr = rad / R
        # widths below ~2 cells cannot be represented by the stencil and alias
        # into axis-aligned rings; bakeKernel widens them, so do the same.
        w = self.logw.exp().clamp_min(2.0 / KR)[:, :, None, None, None, None]
        mu = torch.sigmoid(self.mu_raw)[:, :, None, None, None, None]
        a = self.a[:, :, None, None, None, None]
        ph = torch.where(self.has_phase[None, :], self.phase,
                         torch.zeros_like(self.phase))[:, :, None, None, None, None]
        m = self.m[None, :, None, None, None, None]

        v = (a * torch.exp(-(((rr[:, None] - mu) / w) ** 2))
             * torch.cos(m * theta[:, None] + ph)).sum(1)       # (C,3,3,K,K)
        v = torch.where(rr > 1, torch.zeros_like(v), v).sum(dim=(1, 2)) / 9

        return torch.stack([_finish(v[c], KR, self.feather, zero_mean=True)
                            for c in range(self.C)])

    def to_config(self):
        """Export as index.html kernel dicts (with the angular fields added)."""
        R = self.radii().detach().tolist()
        mu = torch.sigmoid(self.mu_raw).detach().tolist()
        w = self.logw.exp().clamp_min(2.0 / self.KR).detach().tolist()
        a = self.a.detach().tolist()
        ph = torch.where(self.has_phase[None, :], self.phase,
                         torch.zeros_like(self.phase)).detach().tolist()
        orders = [int(x) for x in self.m.tolist()]
        return [{"type": "radial", "sym": "radial", "R": R[c],
                 "feather": self.feather, "seed": 1, "oct": 3,
                 "terms": [{"a": a[c][l], "r": mu[c][l], "w": w[c][l],
                            "m": orders[l], "phase": ph[c][l]}
                           for l in range(len(orders))]}
                for c in range(self.C)]


def bake_from_config(kernels, C, KR, dtype=torch.float64, device="cpu"):
    """Bake index.html-shaped kernel dicts, honouring angular terms.

    This is the reference for what §7 of docs/nca-experiment.md asks bakeKernel
    to become: `m` and `phase` on a term, and every channel baked on the ONE
    shared grid with `R` as a continuous radial scale rather than resampled to
    its own resolution. Terms without `m`/`phase` default to a plain radial
    lobe, so an existing preset bakes exactly as it does today.

    It is also how a trained preset.json is replayed -- see train/evaluate.py.
    """
    px, py = _subsampled(KR, dtype, device)
    rad, theta = torch.hypot(px, py), torch.atan2(py, px)
    out = []
    for c in range(C):
        k = kernels[c]
        R = max(float(k["R"]), 1e-3)
        rr = rad / R
        v = torch.zeros_like(rr)
        for t in k.get("terms", []):
            w = max(float(t["w"]), 2.0 / KR)      # below ~2 cells the stencil aliases
            v = v + (float(t["a"])
                     * torch.exp(-(((rr - float(t["r"])) / w) ** 2))
                     * torch.cos(float(t.get("m", 0)) * theta + float(t.get("phase", 0.0))))
        v = torch.where(rr > 1, torch.zeros_like(v), v).sum(dim=(0, 1)) / 9
        out.append(_finish(v, KR, k.get("feather", 0.1), zero_mean=True))
    return torch.stack(out)
