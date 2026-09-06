"""The brains rule written again in torch, so it can be differentiated.

A transcription of staging/brains/explore.py, which is itself a transcription of
the page. The one thing that had to change is HOW A LOBE IS READ: the numpy port
gathers four corners by integer index, and an index has no gradient, so the fit
would learn what to think about a reading but never where to look for it. Here a
lobe is grid_sample, which is differentiable in the sample POSITION -- and that
position comes from the velocity, so the gradient reaches the force that made it.

Layout is channel-first throughout, (batch, channel, y, x), because that is what
grid_sample wants and the alternative is a transpose per tap.
"""
import numpy as np
import torch
import torch.nn.functional as TF

K, NS, NC = 8, 4, 10
NIN = 3*K + 5
NOUT = 4 + 1 + K
OFF = [(0, 0), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
_L = [1.0] + [(1.0 if (a == 0 or b == 0) else 2.0**0.5) for a, b in OFF[1:]]
UOFF = [(a/l, b/l) for (a, b), l in zip(OFF, _L)]
ILEN = [1.0/l for l in _L]
BW = [(2.0 if a == 0 else 1.0)*(2.0 if b == 0 else 1.0) for a, b in OFF]


def roll_at(a, off):
    """a[p + off] with p indexed [.., y, x]."""
    return torch.roll(a, shifts=(-off[1], -off[0]), dims=(-2, -1))


def blur(a, s):
    """Separable gaussian, wrapping, over the last two axes."""
    if s < 0.05:
        return a
    r = max(1, int(np.ceil(3*s)))
    t = torch.arange(-r, r + 1, device=a.device, dtype=a.dtype)
    w = torch.exp(-0.5*(t/s)**2); w = w/w.sum()
    out = 0
    for i in range(2*r + 1):
        out = out + w[i]*torch.roll(a, int(t[i].item()), -2)
    a2 = out; out = 0
    for i in range(2*r + 1):
        out = out + w[i]*torch.roll(a2, int(t[i].item()), -1)
    return out


def tap(field, dx, dy, N):
    """Read (C,H,W) at every cell displaced by (dx,dy) cells -- differentiable
    in the displacement, which is the whole point."""
    dev = field.device
    lin = (torch.arange(N, device=dev, dtype=field.dtype) + 0.5)
    gy, gx = torch.meshgrid(lin, lin, indexing="ij")
    # grid_sample wants [-1,1] and does not wrap, so wrap by hand into [0,N)
    X = torch.remainder(gx + dx, N)
    Y = torch.remainder(gy + dy, N)
    grid = torch.stack([X/N*2 - 1, Y/N*2 - 1], -1)[None]
    return TF.grid_sample(field[None], grid, mode="bilinear",
                          padding_mode="border", align_corners=False)[0]


class Rule:
    """Holds the learnable parts. Scalars are unconstrained and squashed, so the
    fit can move them without ever leaving the range the page allows."""

    def __init__(self, seed=0, dev="cpu"):
        g = torch.Generator().manual_seed(seed)
        sc = 1 + 2*torch.rand(NS, NC, 1, generator=g)**2
        self.frq = ((torch.rand(NS, NC, NIN, generator=g)*2 - 1)*sc).to(dev).requires_grad_()
        self.amp = (torch.rand(NS, NC, NOUT, generator=g)*2 - 1).to(dev).requires_grad_()
        # start where the search said the good ones live
        self.raw = {k: torch.tensor(float(v), device=dev).requires_grad_()
                    for k, v in dict(gain=-0.7, beta=1.2, speed=1.1, drag=0.0,
                                     force=-0.5, swerve=-0.3, strafe=-1.2,
                                     moff=0.0, decay=2.7, diff=-1.0).items()}

    def params(self):
        return [self.frq, self.amp] + list(self.raw.values())

    def knobs(self):
        r, sp = self.raw, torch.nn.functional.softplus
        return dict(gain=sp(r["gain"]), beta=sp(r["beta"])*5, speed=sp(r["speed"])*2,
                    drag=torch.sigmoid(r["drag"]), force=sp(r["force"]),
                    swerve=sp(r["swerve"]), strafe=sp(r["strafe"]),
                    moff=sp(r["moff"]), decay=torch.sigmoid(r["decay"]),
                    diff=torch.sigmoid(r["diff"]))


class State:
    def __init__(self, C, V, M=None, F=None):
        N = C.shape[-1]
        self.C = C                                              # (NS,H,W)
        self.V = V                                              # (NS,2,H,W)
        self.M = torch.zeros(K, N, N) if M is None else M       # (K,H,W)
        self.F = torch.zeros(2, N, N) if F is None else F       # (2,H,W)

    def detach(self):
        return State(self.C.detach(), self.V.detach(), self.M.detach(), self.F.detach())


def step(st, rule, cfg):
    N = st.C.shape[-1]
    kn = rule.knobs()
    cnorm = 1.0/cfg["fill"]
    mnorm = (1 - kn["decay"])/(kn["moff"]*cfg["fill"] + 1e-6)
    fnorm = 1 - kn["decay"]
    lod = max(0.0, np.log2(max(cfg["size"], 1.0)) + cfg["feather"]*1.6)
    Mb = blur(st.M, 0.42*(2.0**lod))
    Fb = blur(st.F, 0.42*(2.0**lod))
    ctr = Mb*mnorm - 1.0

    # Softened, not branched. torch.where still evaluates BOTH sides and
    # differentiates both, so a division guarded by a where has an infinite
    # gradient on the branch that was never taken -- which is where the first
    # run's |g| = inf came from. A denominator that is never small has a
    # gradient that is never large.
    sp2 = (st.V*st.V).sum(1)
    fw = st.V/torch.sqrt(sp2 + 1e-4)[:, None]
    sp_ = torch.sqrt(sp2 + 1e-12)
    th = torch.atan2(fw[:, 1], fw[:, 0])
    hs = np.radians(cfg["spread"])*0.5
    lat = torch.stack([-fw[:, 1], fw[:, 0]], 1)

    xs, xms = [], []
    for s in range(NS):
        er = torch.stack([torch.cos(th[s] + hs), torch.sin(th[s] + hs)])
        el = torch.stack([torch.cos(th[s] - hs), torch.sin(th[s] - hs)])
        rg = tap(Mb, er[0]*cfg["dist"], er[1]*cfg["dist"], N)*mnorm - 1.0
        lf = tap(Mb, el[0]*cfg["dist"], el[1]*cfg["dist"], N)*mnorm - 1.0
        fr = tap(Fb, er[0]*cfg["dist"], er[1]*cfg["dist"], N)*fnorm
        fl = tap(Fb, el[0]*cfg["dist"], el[1]*cfg["dist"], N)*fnorm
        one = torch.ones(1, N, N)
        z = torch.zeros(K, N, N)
        flow = torch.stack([(fr*fw[s]).sum(0), (fr*lat[s]).sum(0),
                            (fl*fw[s]).sum(0), (fl*lat[s]).sum(0)])
        flm = torch.stack([flow[2], -flow[3], flow[0], -flow[1]])
        if cfg["shape"] == 0:
            xs.append(torch.cat([rg - lf, z, z, flow, one]))
            xms.append(torch.cat([lf - rg, z, z, flm, one]))
        elif cfg["shape"] == 2:
            xs.append(torch.cat([rg, lf, -ctr, flow, one]))
            xms.append(torch.cat([lf, rg, -ctr, flm, one]))
        else:
            xs.append(torch.cat([rg, lf, z, flow, one]))
            xms.append(torch.cat([lf, rg, z, flm, one]))
    x = torch.stack(xs)                                                # (NS,NIN,H,W)
    xm = torch.stack(xms)

    o = brain(x, rule, kn["gain"])
    if cfg["mirror"]:
        om = brain(xm, rule, kn["gain"])
        o = torch.cat([(o[:, 0:1] + om[:, 0:1])*0.5, (o[:, 1:2] - om[:, 1:2])*0.5,
                       (o[:, 2:3] + om[:, 2:3])*0.5, (o[:, 3:4] - om[:, 3:4])*0.5,
                       (o[:, 4:] + om[:, 4:])*0.5], 1)

    tot = st.C.sum(0)
    here = torch.clamp(st.C*cnorm, max=1.0)[:, None]
    force = (fw*o[:, 0:1]*kn["force"] + lat*o[:, 1:2]*kn["swerve"])*here
    V = cap(st.V*(1 - kn["drag"]) + force)
    strafe = cap((fw*o[:, 2:3] + lat*o[:, 3:4])*kn["strafe"]*here)
    # A GLOBAL shift before the exponential. The shares are E(p)/Z(q) and Z is
    # a sum of E, so multiplying every E by one constant cancels exactly -- but
    # it moves the whole range off the floor, where exp(-40) = 4e-18 divided by
    # a sum of its own kind is what made the gradient explode.
    argE = kn["beta"]*o[:, 4] - cfg["crowd"]*tot[None]*cnorm
    E = torch.exp(torch.clamp(argE - argE.mean().detach(), -30, 30))
    dep = (st.C[:, None]*torch.tanh(o[:, 5:])).sum(0)*(kn["moff"]*cnorm)

    bias = V + strafe
    # The weight a cell uses for offset k, and the one it uses for -k. Taking
    # the second as 1/g is what broke the gradient -- g bottoms out at exp(-30),
    # so its reciprocal is 1e13 and the product with a tiny share overflows
    # float32 on the way back. Its OWN exponential is the same number, bounded,
    # and exactly equal to g at the opposite offset (the offsets are unit and
    # symmetric, so arg flips sign and the length factor is the same), which
    # means conservation survives the clamp as well as the algebra.
    argb = [kn["speed"]*(bias[:, 0]*u0 + bias[:, 1]*u1) for u0, u1 in UOFF]
    g = [torch.exp(torch.clamp(ab, -30, 30))*il for ab, il in zip(argb, ILEN)]
    gr = [torch.exp(torch.clamp(-ab, -30, 30))*il for ab, il in zip(argb, ILEN)]
    z = sum(roll_at(E, off)*g[k] for k, off in enumerate(OFF))
    S = st.C/(z + 1e-20)
    got, acc = 0, 0
    for k, off in enumerate(OFF):
        w = roll_at(S*gr[k], off)
        got = got + w
        acc = acc + w[:, None]*roll_at(V, off)
    C2 = E*got
    V2 = acc/(got[:, None] + 1e-9) + V*(1e-9/(got[:, None] + 1e-9))

    c = st.M + dep
    s9 = sum(w*roll_at(c, off) for w, off in zip(BW, OFF))/16.0
    M2 = torch.clamp(kn["decay"]*(c*(1 - kn["diff"]) + s9*kn["diff"]), 0, 6e4)
    cf = st.F + (st.C[:, None]*V).sum(0)*cnorm
    f9 = sum(w*roll_at(cf, off) for w, off in zip(BW, OFF))/16.0
    F2 = torch.clamp(kn["decay"]*(cf*(1 - kn["diff"]) + f9*kn["diff"]), -6e4, 6e4)
    return State(C2, V2, M2, F2)


def cap(v, m=16.0):
    """Branch-free, for the same reason as everything else here: a where around
    a division differentiates the branch it did not take, and a stalled species
    has L near zero, so m/L is 1e13 and its gradient is worse. Saturating a
    ratio has a gradient of exactly zero where the cap is inactive, which is the
    right answer rather than an enormous wrong one."""
    L = torch.linalg.vector_norm(v, dim=1, keepdim=True)
    return v*torch.clamp(m/(L + 1e-6), max=1.0)


def brain(x, rule, gain):
    """(NS,NIN,H,W) -> (NS,NOUT,H,W). Ten random sinusoids, Fluoddity's shape."""
    NSb, _, H, W = x.shape
    xf = x.reshape(NSb, NIN, H*W)
    ph = torch.bmm(rule.frq, xf)*gain                             # (NS,NC,HW)
    s1, c1 = torch.sin(ph), torch.cos(ph)
    s2, c2 = 2*s1*c1, 1 - 2*s1*s1
    off = 2*torch.arange(NC, dtype=ph.dtype)[None, :]*0.6283 + rule.amp[:, :, NOUT - 1]*np.pi
    bc = torch.stack([torch.cos(off), torch.cos(off*0.7),
                      torch.cos(off*1.3), torch.cos(off*0.5)], -1)
    bs = torch.stack([torch.sin(off), torch.sin(off*0.7),
                      torch.sin(off*1.3), torch.sin(off*0.5)], -1)
    B = [s1*bc[..., 0:1] + c1*bs[..., 0:1], c1*bc[..., 1:2] - s1*bs[..., 1:2],
         s2*bc[..., 2:3] + c2*bs[..., 2:3], c2*bc[..., 3:4] - s2*bs[..., 3:4]]
    o = []
    for i in range(NOUT):
        o.append(torch.bmm(rule.amp[:, :, i:i + 1].transpose(1, 2), B[i % 4]))
    return torch.cat(o, 1).reshape(NSb, NOUT, H, W)
