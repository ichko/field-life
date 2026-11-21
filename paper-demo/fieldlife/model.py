"""Field Life: the model in one file.

A world has C colours on an N x N torus. Colour c is a concentration field rho[c].
One step:
  1. ring integral   U_c = K_c * rho_c            (K_c: wide Gaussian minus tight Gaussian)
  2. affinity        A_c = F sum_d M[c,d] U_d - gamma * sum_d G_d * rho_d
  3. MaCE transport  each cell shares its mass among its 3x3 neighbours in proportion
                     to exp(clip(beta A_c, -20, 20)); mass is conserved exactly.
"""
import base64, json
import numpy as np

CLIP = 20.0


class World:
    """The rule: interaction matrix, one ring kernel per colour, and three scalars."""

    def __init__(self, M, reach, wide, tight, sign=None, force=90.0, repel=6.0, beta=3.0,
                 N=512, density=0.12, jitter=0.0):
        self.M = np.asarray(M, float)
        self.C = len(self.M)
        self.reach = np.asarray(reach, float)      # kernel reach per colour, in cells
        self.wide = np.asarray(wide, float)        # width of the wide Gaussian, fraction of reach
        self.tight = np.asarray(tight, float)      # width of the tight Gaussian, fraction of reach
        self.sign = np.ones(self.C) if sign is None else np.asarray(sign, float)
        self.force, self.repel, self.beta = float(force), float(repel), float(beta)
        self.N, self.density, self.jitter = int(N), float(density), float(jitter)

    # ---- conversion to and from the JSON layout used by the data files and the web page
    @classmethod
    def from_config(cls, cfg):
        ks = cfg["kernels"]
        wide, tight, sign = [], [], []
        for k in ks:
            t = sorted(k["terms"], key=lambda t: -t["w"])
            wide.append(t[0]["w"]); tight.append(t[1]["w"]); sign.append(np.sign(t[0]["a"]))
        return cls(np.reshape(cfg["mat"], (cfg["C"], cfg["C"])), [k["R"] for k in ks], wide, tight, sign,
                   cfg.get("force", 90), cfg.get("repel", 6), cfg.get("beta", 3),
                   cfg.get("N", 512), cfg.get("density", 0.12), cfg.get("noise", 0))

    def to_config(self):
        kernels = [{"R": float(R), "type": "radial", "sym": "radial", "feather": 0.1, "seed": 1, "oct": 3,
                    "terms": [{"a": float(s), "r": 0, "w": float(w)},
                              {"a": float(-s * (w / t) ** 2), "r": 0, "w": float(t)}]}
                   for R, w, t, s in zip(self.reach, self.wide, self.tight, self.sign)]
        return {"N": self.N, "C": self.C, "mat": self.M.ravel().tolist(), "kernels": kernels,
                "force": self.force, "repel": self.repel, "beta": self.beta, "density": self.density,
                "noise": self.jitter, "radMin": float(self.reach.min()), "radMax": float(self.reach.max()),
                "seedMode": "disc", "kterms": 2}


def random_world(rng, C=None, N=512, reach=61.0, force=90.0, repel=6.0, beta=3.0, random_scalars=False):
    """Draw a rule the way the exploration did: matrix uniform in [-1, 1], a ring per colour with
    w_wide in [0.20, 0.50] and w_tight = w_wide / u, u in [1.7, 5.7]."""
    C = int(rng.integers(4, 9)) if C is None else C
    M = rng.uniform(-1, 1, (C, C))
    wide = rng.uniform(0.20, 0.50, C)
    tight = wide / rng.uniform(1.7, 5.7, C)
    if random_scalars:
        reach = float(rng.uniform(22, 110)); force = float(rng.uniform(37, 190))
        repel = float(rng.uniform(2, 13)); beta = float(rng.uniform(1.8, 5.3))
    return World(M, np.full(C, reach), wide, tight, None, force, repel, beta, N)


# ---- kernels
def _line(rad, sigma):
    """One-dimensional Gaussian cut at radius rad, with its tail shifted to zero at the cut."""
    d = np.arange(-rad, rad + 1)
    g = np.maximum(np.exp(-0.5 * d * d / sigma ** 2) - np.exp(-0.5 * rad * rad / sigma ** 2), 0)
    return g / g.sum()


def _spectrum(N, rad, sigma):
    """FFT of the separable 2-D Gaussian, placed on an N x N torus."""
    line = np.zeros(N); line[:2 * rad + 1] = _line(rad, sigma); line = np.roll(line, -rad)
    f = np.fft.fft(line)
    return np.outer(f, f)


class Sim:
    """A running world: the concentration fields and the step function."""

    def __init__(self, world, dtype=np.float64):
        self.w, self.N, self.C = world, world.N, world.C
        self.rho = np.zeros((self.C, self.N, self.N), dtype)
        sig_w = world.wide * world.reach / np.sqrt(2)
        sig_t = world.tight * world.reach / np.sqrt(2)
        rad = int(np.ceil(3.2 * sig_w.max()))
        assert self.N >= 2 * rad + 1, f"grid of {self.N} cells is too small for a kernel of radius {rad}; use N >= {2 * rad + 1}"
        self.K, self.G = [], []
        for c in range(self.C):
            Fw, Ft = _spectrum(self.N, rad, sig_w[c]), _spectrum(self.N, rad, sig_t[c])
            r = np.arange(0.25, 6 * sig_w[c], 0.25)      # L1 norm of the continuous ring, for scaling
            l1 = np.sum(np.abs(np.exp(-0.5 * r * r / sig_w[c] ** 2) / sig_w[c] ** 2
                               - np.exp(-0.5 * r * r / sig_t[c] ** 2) / sig_t[c] ** 2) * r * 0.25)
            self.K.append((Fw - Ft) * world.sign[c] / l1)
            self.G.append(Ft)
        self.rng = np.random.default_rng(0)

    # ---- initial states
    def seed_disc(self, rng=None):
        """A disc of noisy mass, radius 0.2 N, in the centre of an empty world."""
        rng = rng or self.rng
        N = self.N
        y, x = np.mgrid[0:N, 0:N] + 0.5
        d = np.hypot(x - N / 2, y - N / 2)
        e = np.clip((0.2 * N - d) / max(1.5, 0.012 * N), 0, 1); mask = e * e * (3 - 2 * e)
        for c in range(self.C):
            n = 0.6 * _value_noise(rng, N, 6) + 0.4 * _value_noise(rng, N, 14)
            self.rho[c] = self.w.density * 4 * mask * (0.6 + 0.8 * n)
        return self

    def seed_noise(self, rng=None):
        """Patches of noise over the whole world, scaled so each colour carries density * N^2 mass."""
        rng = rng or self.rng
        for c in range(self.C):
            v = 0.55 * _value_noise(rng, self.N, 4) + 0.3 * _value_noise(rng, self.N, 9) + 0.15 * _value_noise(rng, self.N, 18)
            thr = 0.42 + 0.14 * rng.random()
            v = np.maximum(0, v - thr) / (1 - thr)
            v = v * v
            self.rho[c] = v * (self.w.density * self.N * self.N / max(v.sum(), 1e-9))
        return self

    def seed_pattern(self, pattern):
        """Place a saved body (the `seedPattern` of a glider) in the centre of an empty world."""
        p = pattern
        raw = np.frombuffer(base64.b64decode(p["data"]), np.uint8).astype(float)
        raw = raw.reshape(p["C"], p["h"], p["w"]) ** 2 * (p.get("scale", 1) / 255 ** 2)
        z = max(1, int(round(p.get("cells", 1))))
        raw = raw.repeat(z, 1).repeat(z, 2)
        C, h, w = min(self.C, p["C"]), raw.shape[1], raw.shape[2]
        y0, x0 = (self.N - h) // 2, (self.N - w) // 2
        self.rho[:] = 0
        ys, xs = (np.arange(h) + y0) % self.N, (np.arange(w) + x0) % self.N
        self.rho[:C][:, ys[:, None], xs[None, :]] = raw[:C]
        return self

    # ---- one step
    def step(self):
        w = self.w
        R = np.fft.fft2(self.rho)
        U = np.real(np.fft.ifft2(R * np.array(self.K)))
        crowd = np.real(np.fft.ifft2((R * np.array(self.G)).sum(0)))
        A = w.force * np.einsum("cd,dyx->cyx", w.M, U) - w.repel * crowd[None]
        X = w.beta * A
        if w.jitter > 0:
            X = X + w.jitter * self.rng.uniform(-1, 1, X.shape)
        E = np.exp(np.clip(X, -CLIP, CLIP))
        Z = _sum9(E)                        # total weight of each cell's neighbourhood
        self.rho = E * _sum9(self.rho / Z)  # what each cell receives from its nine neighbours
        return self

    def run(self, steps):
        for _ in range(steps):
            self.step()
        return self

    def mass(self):
        return self.rho.sum((1, 2))


def _sum9(a):
    return sum(np.roll(a, (dy, dx), (-2, -1)) for dy in (-1, 0, 1) for dx in (-1, 0, 1))


def _value_noise(rng, N, freq):
    """Tileable value noise in [0, 1] with `freq` cells of noise across the world."""
    g = rng.random((freq, freq))
    t = (np.arange(N) + 0.5) / N * freq
    i = np.floor(t).astype(int); f = t - i; f = f * f * (3 - 2 * f)
    i0, i1 = i % freq, (i + 1) % freq
    a = g[np.ix_(i0, i0)] * (1 - f)[:, None] + g[np.ix_(i1, i0)] * f[:, None]
    b = g[np.ix_(i0, i1)] * (1 - f)[:, None] + g[np.ix_(i1, i1)] * f[:, None]
    return a * (1 - f)[None, :] + b * f[None, :]


# ---- data files
def load_setups(path):
    """Read a worlds or gliders file: a list of {name, tags, cfg} records."""
    with open(path) as f:
        return json.load(f)["setups"]


def find_setup(setups, key):
    """Find a record by index, by name, or by the paper's W<n> / B<n> label."""
    if isinstance(key, int):
        return setups[key]
    if key[:1] in "WB" and key[1:].isdigit():
        return setups[int(key[1:])] if key[0] == "B" else setups[int(key[1:]) - 1]   # worlds are numbered from W1, gliders from B0
    return next(s for s in setups if s["name"] == key)
