"""The volume page's rule, written again in torch so it can be differentiated.

Every step here is the step staging/volume.html takes, in the same order and
with the same arithmetic, with one part generalised: the kernel. The page's
kernel is a difference of two Gaussians, which is radially symmetric, and a
bank of radially symmetric kernels has only radially symmetric fixed points --
it cannot hold anything shaped like an animal. So a kernel here is a sum of
Gaussians that have been MOVED OFF CENTRE:

    K_c = sum_t  a[c,t] * G_sigma(t) ( rho_c )  sampled at  p + o[c,t]

which is still separable -- the blurs are shared by every channel and every
term, and a displacement is free, it is just where you read the result. A
handful of displaced blobs of two or three widths will describe a thoroughly
lopsided kernel, and the whole bank costs three separable passes per width
plus one gather of T taps.

The state is the field rho: C channels on an N^3 torus. Mass per channel is
conserved by construction, which is the property the whole thing is built on,
so the seed has to arrive already holding what the picture will cost.
"""
import math
import torch
import torch.nn.functional as F


def gauss1d(sigma, K, device, dtype):
    """A feathered Gaussian, cut and slid down to zero at the rim, as the page
    does it: a hard cut at the edge of the window prints the window's own shape
    onto the field."""
    x = torch.arange(-K, K + 1, device=device, dtype=dtype)
    w = torch.exp(-0.5*(x/sigma)**2) - math.exp(-0.5*(K/sigma)**2)
    w = torch.clamp(w, min=0.0)
    return w/w.sum()


def blur3(x, w, axis):
    """Wrapped 1-D convolution of (B,C,D,H,W) along one spatial axis."""
    K = (w.numel() - 1)//2
    C = x.shape[1]
    pad = [0, 0, 0, 0, 0, 0]
    pad[2*(2 - axis)] = K
    pad[2*(2 - axis) + 1] = K
    x = F.pad(x, pad, mode="circular")
    shape = [1, 1, 1, 1, 1]
    shape[2 + axis] = w.numel()
    k = w.view(shape).expand(C, 1, *shape[2:])
    return F.conv3d(x, k, groups=C)


def shift_all(x, off, pad, base_grid):
    """Sample every channel of x at its own displacement, in one gather.

    x is (1,C,D,H,W) and off is (C,3) in cells, ordered (z,y,x). Channels go
    into the batch dimension so grid_sample can give each one a different
    offset; the volume is padded round the outside first, which is what makes
    the sample wrap the way the torus does.
    """
    C = x.shape[1]
    xp = F.pad(x, (pad,)*6, mode="circular").permute(1, 0, 2, 3, 4)   # (C,1,...)
    D = xp.shape[-1]
    # grid_sample wants normalised coordinates in x,y,z order and the offset is
    # in cells, so it scales by 2/size; the base grid already sits where the
    # unpadded volume does.
    o = (off.flip(-1)*(2.0/(D - 1))).view(C, 1, 1, 1, 3)
    return F.grid_sample(xp, base_grid + o, mode="bilinear",
                         padding_mode="zeros", align_corners=True).permute(1, 0, 2, 3, 4)


class Field3D(torch.nn.Module):
    """C channels, S blur widths, T displaced terms per channel."""

    def __init__(self, C=8, S=3, T=6, N=48, seedR=3.5, device="cpu", dtype=torch.float32):
        super().__init__()
        self.C, self.S, self.T, self.N = C, S, T, N
        self.soft = True
        self.register_buffer('dscale', torch.tensor(1.0, dtype=dtype))
        self.device, self.dtype = device, dtype
        g = torch.Generator().manual_seed(7)

        # Widths, in HALF-pitch cells: the neighbourhood integrals are taken on
        # a grid with half the spacing, as on the page, because a blur over a
        # dozen cells does not notice and it is eight times less work.
        self.log_sig = torch.nn.Parameter(torch.log(torch.tensor(
            [0.7, 1.2, 2.0, 3.2, 5.0][:S], dtype=dtype)))
        # which width each term reads
        self.term_sig = [t % S for t in range(T)]

        self.amp = torch.nn.Parameter(torch.randn(C, T, generator=g).to(dtype)*0.45)
        self.off = torch.nn.Parameter(torch.randn(C, T, 3, generator=g).to(dtype)*1.1)
        self.mat = torch.nn.Parameter(torch.randn(C, C, generator=g).to(dtype)*0.5)
        self.log_force = torch.nn.Parameter(torch.tensor(math.log(25.0), dtype=dtype))
        self.repel = torch.nn.Parameter(torch.tensor(0.3, dtype=dtype))
        self.log_beta = torch.nn.Parameter(torch.tensor(math.log(1.0), dtype=dtype))

        # The seed: a small ball, with a free amount of each channel in it and a
        # free pattern inside it. Small, but not one cell -- there has to be
        # somewhere for an orientation to live.
        self.seedR = seedR
        r = int(math.ceil(seedR)) + 1
        self.seed_half = r
        self.seed_raw = torch.nn.Parameter(torch.randn(C, 2*r + 1, 2*r + 1, 2*r + 1,
                                                       generator=g).to(dtype)*0.3)
        self.seed_mass = torch.nn.Parameter(torch.zeros(C, dtype=dtype))
        zz = torch.arange(-r, r + 1, dtype=dtype)
        d = torch.sqrt(zz.view(-1,1,1)**2 + zz.view(1,-1,1)**2 + zz.view(1,1,-1)**2)
        e = torch.clamp((seedR - d)/1.2, 0, 1)
        self.register_buffer("seed_ball", (e*e*(3 - 2*e)))

        # The sampling grid for the displaced reads, built once. It addresses
        # the unpadded volume inside a padded one, in grid_sample's normalised
        # x,y,z coordinates; a displacement is then just an offset added to it.
        M = N//2
        self.gpad = 6
        P = M + 2*self.gpad
        u = (torch.arange(M, dtype=dtype) + self.gpad)*(2.0/(P - 1)) - 1.0
        gz, gy, gx = torch.meshgrid(u, u, u, indexing="ij")
        self.register_buffer("base_grid",
                             torch.stack([gx, gy, gz], -1).unsqueeze(0))   # (1,M,M,M,3)

    # ---------------------------------------------------------------- seeding
    def seed(self, masses):
        """masses: (C,) total mass to place. Returns rho (1,C,N,N,N)."""
        N, C, r = self.N, self.C, self.seed_half
        pat = torch.nn.functional.softplus(self.seed_raw)*self.seed_ball
        pat = pat/pat.sum((1, 2, 3), keepdim=True).clamp_min(1e-9)
        pat = pat*masses.view(C, 1, 1, 1)
        rho = torch.zeros(1, C, N, N, N, device=self.device, dtype=self.dtype)
        c0 = N//2
        rho[0, :, c0-r:c0+r+1, c0-r:c0+r+1, c0-r:c0+r+1] = pat
        return rho

    # ------------------------------------------------------------- one step
    def step(self, rho):
        C, N = self.C, self.N
        half = F.avg_pool3d(rho, 2)                        # the half-pitch copy
        sig = torch.exp(self.log_sig)

        bank = []
        for s in range(self.S):
            K = max(2, int(math.ceil(4.0*float(sig[s].detach()))))
            K = min(K, N//4 - 1)
            w = gauss1d(sig[s], K, rho.device, rho.dtype)
            b = blur3(blur3(blur3(half, w, 0), w, 1), w, 2)
            bank.append(b)

        # the kernel: displaced blobs, summed per channel. One gather per term
        # over every channel at once -- the displacement is the only thing that
        # differs between channels, and grid_sample will take it per batch item.
        # The lobe weights are normalised to sum to one in absolute value, as
        # the flat page normalises each baked kernel: without it the kernel's
        # overall scale and the force in front of it are the same number twice,
        # and the fit spends its time trading one against the other.
        amp = self.amp/self.amp.abs().sum(1, keepdim=True).clamp_min(1e-6)
        Kc = 0.0
        for t in range(self.T):
            g = shift_all(bank[self.term_sig[t]], self.off[:, t, :],
                          self.gpad, self.base_grid)
            Kc = Kc + g*amp[:, t].view(1, C, 1, 1, 1)

        crowd = bank[0].sum(1, keepdim=True)               # tightest blur, all colours
        # Both terms are divided by the field's mean density before force and
        # repel are applied, so those two read the same whatever the object
        # weighs or however large the grid is. In three dimensions an animal is
        # about a hundredth of the cube, where on a sheet it is a good fraction
        # of it, and without this the same force means something a hundred
        # times different. The constant folds back into force on the way out.
        A = torch.exp(self.log_force)*torch.einsum("cd,bdzyx->bczyx", self.mat, Kc)*self.dscale \
            - self.repel*crowd*self.dscale

        A = F.interpolate(A, size=(N, N, N), mode="trilinear", align_corners=False)
        # The page clamps the exponent hard at +-11, which is right for
        # running and wrong for fitting: a step that saturates would hand back
        # no gradient at all. Squashing to the same bound keeps the range and
        # keeps the slope; the fitted model is checked against the hard clamp
        # afterwards, and stays well inside it.
        bA = torch.exp(self.log_beta)*A
        E = torch.exp(11.0*torch.tanh(bA/11.0) if self.soft else torch.clamp(bA, -11.0, 11.0))

        ones = torch.ones(C, 1, 3, 3, 3, device=rho.device, dtype=rho.dtype)
        def box27(x):
            x = F.pad(x, (1, 1, 1, 1, 1, 1), mode="circular")
            return F.conv3d(x, ones, groups=C)

        Z = box27(E)
        S_ = rho/Z.clamp_min(1e-30)
        return E*box27(S_)

    def run(self, masses, steps, keep=()):
        # mass per cell, averaged over the cube: the yardstick the affinity uses
        self.dscale.fill_(float(self.N**3)/float(masses.sum().detach()))
        rho = self.seed(masses)
        out = {}
        for i in range(1, steps + 1):
            rho = self.step(rho)
            if i in keep:
                out[i] = rho
        return rho, out
