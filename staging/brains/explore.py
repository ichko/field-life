"""A CPU port of staging/brains.html, faithful enough to search its parameters.

Not a rewrite -- a transcription. Every number below is read off the shader, and
the one check that says so is mass: the species channels are conserved exactly
by construction, so a run whose total drifts means the port is wrong, not the
rule. Perception is the one honest approximation: the page reads a lobe as one
tap off the mip chain, and here it is a gaussian blur at the same radius, then
bilinear-sampled. Regimes transfer; exact pixels do not.
"""
import numpy as np

K, NS, NC = 8, 9, 10
NOUT = 2 + K
OFF = [(0, 0), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]


def mulberry32(seed):
    """The same generator the page uses, so a brain seed means the same thing
    on both sides and a preset actually replays instead of merely rhyming."""
    a = np.uint32(seed)
    M = np.uint32(0xFFFFFFFF)

    def nxt():
        nonlocal a
        a = np.uint32((int(a) + 0x6D2B79F5) & 0xFFFFFFFF)
        t = np.uint32((int(a ^ (a >> np.uint32(15))) * int(np.uint32(1) | a)) & 0xFFFFFFFF)
        t = np.uint32((int(t) + (int(t ^ (t >> np.uint32(7))) * int(np.uint32(61) | t))) & 0xFFFFFFFF) ^ t
        return float(int(np.uint32(t ^ (t >> np.uint32(14))))) / 4294967296.0
    return nxt


def _h32(x):
    old = np.seterr(over="ignore")
    x = x.astype(np.uint32)
    x ^= x >> np.uint32(16); x *= np.uint32(0x7feb352d)
    x ^= x >> np.uint32(15); x *= np.uint32(0x846ca68b)
    x ^= x >> np.uint32(16)
    np.seterr(**old)
    return x


def glsl_rnd(px, py, k, f):
    old = np.seterr(over="ignore")
    """The seed shader's hash, ported exactly, so an initial field replays too."""
    inner = _h32(np.uint32(k) * np.uint32(0xc2b2ae35) ^ np.uint32(f & 0xFFFFFFFF))
    mid = _h32(px.astype(np.uint32) * np.uint32(0x85ebca6b) ^ inner)
    h = _h32(py.astype(np.uint32) * np.uint32(0x9e3779b9) ^ mid)
    np.seterr(**old)
    return h.astype(np.float64) * (1.0 / 4294967296.0) - 0.5


NIN = 3 * K + 5      # three readings a matter channel, the flow, and a bias


def roll_at(a, off):
    """a[p + off], with p indexed as [y, x] and off given as (dx, dy)."""
    return np.roll(np.roll(a, -off[1], axis=0), -off[0], axis=1)


def blur(m, sigma):
    """Separable gaussian over the first two axes, wrapping."""
    if sigma < 0.05:
        return m
    r = max(1, int(np.ceil(3 * sigma)))
    t = np.arange(-r, r + 1)
    w = np.exp(-0.5 * (t / sigma) ** 2)
    w /= w.sum()
    out = np.zeros_like(m)
    for i, s in enumerate(t):
        out += w[i] * np.roll(m, int(s), axis=0)
    m2 = out
    out = np.zeros_like(m)
    for i, s in enumerate(t):
        out += w[i] * np.roll(m2, int(s), axis=1)
    return out


def gather(field, x, y):
    """Bilinear read of (N, N, C) at wrapped float coordinates."""
    N = field.shape[0]
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    fx = (x - x0)[..., None]
    fy = (y - y0)[..., None]
    x0m, y0m = x0 % N, y0 % N
    x1m, y1m = (x0 + 1) % N, (y0 + 1) % N
    f = field.reshape(N * N, -1)
    a = f[y0m * N + x0m]
    b = f[y0m * N + x1m]
    c = f[y1m * N + x0m]
    d = f[y1m * N + x1m]
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


class Brains:
    def __init__(self, P, seed):
        self.P = P
        self.N = N = P["N"]
        self.nin = NIN
        # --- rollBrains: Fluoddity's seeding, drawn in the page's exact order
        r = mulberry32(seed)
        self.frq = np.empty((NS, NC, self.nin), np.float32)
        self.amp = np.empty((NS, NC, NOUT), np.float32)
        for si in range(NS):
            for c in range(NC):
                scale = 1 + 2 * r() ** 2
                for j in range(self.nin):
                    self.frq[si, c, j] = (r() * 2 - 1) * scale
                for j in range(NOUT):
                    self.amp[si, c, j] = r() * 2 - 1
        # the per-centre phase offset, and the four basis shapes' constants
        off = 2 * np.arange(NC) * 0.6283 + self.amp[:, :, NOUT - 1] * np.pi   # (NS, NC)
        self.bcos = np.stack([np.cos(off), np.cos(off * 0.7),
                              np.cos(off * 1.3), np.cos(off * 0.5)], -1)      # (NS,NC,4)
        self.bsin = np.stack([np.sin(off), np.sin(off * 0.7),
                              np.sin(off * 1.3), np.sin(off * 0.5)], -1)
        self.bcos = self.bcos.astype(np.float32)
        self.bsin = self.bsin.astype(np.float32)
        self.frqT = np.ascontiguousarray(self.frq.transpose(0, 2, 1))   # (NS,NIN,NC)
        self.oidx = [[i for i in range(NOUT) if i % 4 == r] for r in range(4)]
        self.reset()

    # ------------------------------------------------------------------ seed
    def reset(self):
        P, N = self.P, self.N
        self.M = np.zeros((N, N, K), np.float32)          # matter: starts empty
        self.F = np.zeros((N, N, 2), np.float32)          # flow: the vector half
        self.C = np.zeros((N, N, NS), np.float32)         # species
        gw, gh, best = 1, NS, 1e9
        for w in range(1, NS + 1):
            h = -(-NS // w)
            sc = (w * h - NS) * 2.5 + abs(w - h)
            if sc < best:
                best, gw, gh = sc, w, h
        rad = P["ball"] * N * 0.5 / max(gw, gh)
        amp = P["fill"] * N * N / max(np.pi * rad * rad * NS, 1)
        yy, xx = np.mgrid[0:N, 0:N]
        fp = np.stack([xx + 0.5, yy + 0.5], -1).astype(np.float32)
        cell = N / np.array([gw, gh], np.float32)
        for s in range(NS):
            mid = (np.array([s % gw, s // gw], np.float32) + 0.5) * cell
            d = fp - mid
            d -= N * np.floor(d / N + 0.5)
            self.C[..., s] = np.where(np.hypot(d[..., 0], d[..., 1]) < rad, amp, 0)
        # each species faces its own way -- the seed shader's own hash
        a = np.stack([(glsl_rnd(xx, yy, 16 + s, P["seed"]) + 0.5) * 2 * np.pi
                      for s in range(NS)], -1).astype(np.float32)
        self.D = np.stack([np.cos(a), np.sin(a)], -1).astype(np.float32)      # (N,N,NS,2)
        yy, xx = np.mgrid[0:N, 0:N]
        self.px = (xx + 0.5).astype(np.float32)
        self.py = (yy + 0.5).astype(np.float32)
        self.mass0 = self.C.sum()

    # ------------------------------------------------------------------ step
    def step(self):
        P, N = self.P, self.N
        cnorm = 1.0 / max(1e-6, P["fill"])
        mnorm = (1 - P["decay"]) / max(1e-6, P["moff"] * P["fill"])
        gain = np.float32(P["gain"] * 2.0 / np.sqrt(self.nin))
        lod = max(0.0, np.log2(max(P["size"], 1.0)) + P["feather"] * 1.6)
        Mb = blur(self.M, 0.42 * (2.0 ** lod))
        Mf = Mb.reshape(N * N, K)
        Fb = blur(self.F, 0.42 * (2.0 ** lod)).reshape(N * N, 2)
        fnorm = np.float32(1 - P["decay"])
        ctr = (Mb * mnorm - 1.0)[:, :, None, :]              # (N,N,1,K), shared

        th = np.arctan2(self.D[..., 1], self.D[..., 0])      # (N,N,NS)
        hs = np.float32(np.radians(P["spread"]) * 0.5)
        rgt = self._tap(Mf, th + hs) * mnorm - 1.0           # (N,N,NS,K)
        lft = self._tap(Mf, th - hs) * mnorm - 1.0
        one = np.ones((N, N, NS, 1), np.float32)
        Z = np.zeros((N, N, NS, K), np.float32)
        ctrb = np.broadcast_to(ctr, (N, N, NS, K))
        # the flow at each lobe, turned into the reading species' own frame
        fwd = np.stack([np.cos(th), np.sin(th)], -1)
        lat = np.stack([-np.sin(th), np.cos(th)], -1)
        fr = self._tap(Fb, th + hs, 2) * fnorm
        fl = self._tap(Fb, th - hs, 2) * fnorm
        F = np.stack([(fr * fwd).sum(-1), (fr * lat).sum(-1),
                      (fl * fwd).sum(-1), (fl * lat).sum(-1)], -1)
        Fm = np.stack([F[..., 2], -F[..., 3], F[..., 0], -F[..., 1]], -1)

        if P["shape"] == 0:
            x = np.concatenate([rgt - lft, Z, Z, F, one], -1)
            xm = np.concatenate([lft - rgt, Z, Z, Fm, one], -1)
        elif P["shape"] == 2:
            x = np.concatenate([rgt, lft, -ctrb, F, one], -1)
            xm = np.concatenate([lft, rgt, -ctrb, Fm, one], -1)
        else:
            x = np.concatenate([rgt, lft, Z, F, one], -1)
            xm = np.concatenate([lft, rgt, Z, Fm, one], -1)

        o = self._brain(x, gain)
        if P["mirror"]:
            om = self._brain(xm, gain)
            o = np.concatenate([(o[..., :1] - om[..., :1]) * 0.5,
                                (o[..., 1:] + om[..., 1:]) * 0.5], -1)

        tot = self.C.sum(-1)
        turn = (np.float32(np.radians(P["turn"])) * np.tanh(o[..., 0])
                * np.minimum(1.0, self.C * cnorm))                       # (N,N,NS)
        E = np.exp(np.clip(P["beta"] * o[..., 1]
                           - P["crowd"] * tot[..., None] * cnorm, -40, 40))
        dep = (self.C[..., None] * np.tanh(o[..., 2:])).sum(2) * np.float32(P["moff"] * cnorm)

        # ---- transport: exp(beta*affinity + speed*heading.offset), per species
        sp = np.float32(P["speed"])
        Dx, Dy = self.D[..., 0], self.D[..., 1]
        z = np.zeros_like(E)
        fwd = [np.exp(sp * (Dx * o0 + Dy * o1)) for o0, o1 in OFF]
        for k, off in enumerate(OFF):
            z += roll_at(E, off) * fwd[k]
        S = self.C / np.maximum(z, 1e-30)
        got = np.zeros_like(E)
        acc = np.zeros_like(self.D)
        for k, off in enumerate(OFF):
            w = roll_at(S / fwd[k], off)          # exp(sp*d.(-off)) = 1/exp(sp*d.off)
            got += w
            acc += w[..., None] * roll_at(self.D, off)
        wt = got
        d = np.where(wt[..., None] > 1e-20, acc / np.maximum(wt, 1e-30)[..., None], self.D)
        L = np.hypot(d[..., 0], d[..., 1])[..., None]
        d = np.where(L > 1e-6, d / np.maximum(L, 1e-12),
                     np.array([1.0, 0.0], np.float32))
        ct, st = np.cos(turn), np.sin(turn)
        self.D = np.stack([d[..., 0] * ct - d[..., 1] * st,
                           d[..., 0] * st + d[..., 1] * ct], -1)
        self.C = E * got

        # ---- the ground, matter and flow alike
        c = self.M + dep
        s9 = sum(roll_at(c, off) for off in OFF) / np.float32(9.0)
        self.M = np.clip(P["decay"] * (c * (1 - P["diff"]) + s9 * P["diff"]),
                         0, 6e4).astype(np.float32)
        cf = self.F + (self.C[..., None] * self.D).sum(2) * np.float32(cnorm)
        f9 = sum(roll_at(cf, off) for off in OFF) / np.float32(9.0)
        self.F = np.clip(P["decay"] * (cf * (1 - P["diff"]) + f9 * P["diff"]),
                         -6e4, 6e4).astype(np.float32)

    def _tap(self, Mf, ang, C=K):
        """Bilinear read of the blurred matter DIST cells along each angle."""
        N, P = self.N, self.P
        x = self.px[..., None] + np.cos(ang) * P["dist"]
        y = self.py[..., None] + np.sin(ang) * P["dist"]
        x0 = np.floor(x); y0 = np.floor(y)
        fx = (x - x0)[..., None]; fy = (y - y0)[..., None]
        x0 = x0.astype(np.int32) % N; y0 = y0.astype(np.int32) % N
        x1 = (x0 + 1) % N; y1 = (y0 + 1) % N
        a = Mf[y0 * N + x0]; b = Mf[y0 * N + x1]
        c = Mf[y1 * N + x0]; e = Mf[y1 * N + x1]   # (N,N,NS,C)
        return (a + (b - a) * fx) + ((c + (e - c) * fx) - (a + (b - a) * fx)) * fy

    def _brain(self, x, gain):
        """(N,N,NS,NIN) -> (N,N,NS,NOUT). Batched gemm, species on the batch axis."""
        N = self.N
        xt = np.ascontiguousarray(x.transpose(2, 0, 1, 3).reshape(NS, N * N, self.nin))
        ph = np.matmul(xt, self.frqT) * gain                  # (NS, M, NC)
        s1 = np.sin(ph); c1 = np.cos(ph)
        s2 = 2 * s1 * c1; c2 = 1 - 2 * s1 * s1                # first harmonic, for free
        bc, bs = self.bcos[:, None, :, :], self.bsin[:, None, :, :]
        B = (s1 * bc[..., 0] + c1 * bs[..., 0],
             c1 * bc[..., 1] - s1 * bs[..., 1],
             s2 * bc[..., 2] + c2 * bs[..., 2],
             c2 * bc[..., 3] - s2 * bs[..., 3])
        o = np.empty((NS, N * N, NOUT), np.float32)
        for r in range(4):
            idx = self.oidx[r]
            o[..., idx] = np.matmul(B[r], self.amp[:, :, idx])
        return o.reshape(NS, N, N, NOUT).transpose(1, 2, 0, 3)
