"""Kernels drawn by a small network, over the offset vector, per channel.

The flat page writes a kernel as a sum of lobes, each with a radius, a width,
an angular order m and a phase: `a * exp(-((r-r0)/w)^2) * cos(m*theta + phase)`.
The angular order is the part that matters — it is the only thing in the whole
bank that can tell one direction from another, and a rotationally symmetric bank
has only rotationally symmetric fixed points.

`cos(m*theta + phase)` does not generalise to three dimensions on its own: there
is no single angle. What does generalise is the thing the phase was standing in
for — a DIRECTION. Take a learned unit axis v, project the offset onto it,
p = u . v, and Chebyshev polynomials of that projection are exactly cos(m*angle
to v). So an angular order about a learned axis is the 3D form of the flat
page's order-and-phase, and it is richer: the axis can point anywhere, where a
phase could only slide round one circle.

The other half of the flat page's lobe is the RING: `exp(-((r - r0)/w)^2)`, a
shell of attraction at one radius with nothing at the others. A plain MLP over
r has to work to make one of those -- a tanh network is smooth and a ring is not
-- so the radius goes in as a Fourier basis, cos and sin of a few multiples of
pi*r. Each of those is already a set of concentric shells across the reach, and
a weighted sum of them is any ring pattern the fit wants, sharp ones included.
Rings for the radial part, Chebyshev for the angular part: between them they
span what the flat page writes by hand, and the network only has to choose.

Those features, the raw offset, and a per-channel embedding go into a small MLP,
which is evaluated at every offset in the stencil to draw that channel's kernel.
A CPPN over the neighbourhood, in other words, informed by rotations of the
offset vector. It is baked once per step into a dense stencil and convolved; the
gradient flows back through the bake into the network.

Big and small in one bank: each channel carries its own reach, so the same
stencil holds a kernel that looks a couple of cells out and one that looks
across a third of the world -- which is what the flat page's radius bank is for.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class KernelCPPN(nn.Module):
    def __init__(self, C, K=7, axes=4, orders=3, rings=5, emb=10, hidden=48, seed=5):
        super().__init__()
        self.C, self.K, self.A, self.MO, self.R = C, K, axes, orders, rings
        g = torch.Generator().manual_seed(seed)

        # the offsets the stencil covers, in cells, once
        d = torch.arange(-K, K + 1, dtype=torch.float32)
        gz, gy, gx = torch.meshgrid(d, d, d, indexing="ij")
        self.register_buffer("off", torch.stack([gz, gy, gx], -1).reshape(-1, 3))

        self.axis = nn.Parameter(torch.randn(axes, 3, generator=g))
        self.embed = nn.Parameter(torch.randn(C, emb, generator=g)*1.4)
        # Reach per channel, as a fraction of the stencil. Spread the starting
        # values right across the range so the bank begins with big kernels and
        # small ones rather than C copies of the same middling one -- at K=7
        # that is a couple of cells at one end and thirteen at the other, which
        # is the span the flat page's radius bank covers.
        self.log_reach = nn.Parameter(torch.linspace(-2.6, 2.6, C))

        nin = 2 + 3 + axes*orders + 2*rings + emb
        self.net = nn.Sequential(
            nn.Linear(nin, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh())
        # The readout is PER CHANNEL. One shared head with only an embedding to
        # tell the channels apart starts the bank nearly degenerate -- eight
        # kernels that are all the same kernel -- and a matrix cannot undo that;
        # the fit has to spend its early iterations inventing diversity it could
        # have been given. Independent heads over shared features cost C*hidden
        # parameters and start the bank varied.
        self.head = nn.Parameter(torch.randn(C, hidden, generator=g)/math.sqrt(hidden))
        self.head_b = nn.Parameter(torch.zeros(C))

        # the rim taper, and the mask, in units of the stencil
        r = self.off.norm(dim=-1)/K
        self.register_buffer("rr", r)

    def reach(self):
        """Each channel's radius as a fraction of the stencil, 0.10 to 1."""
        return 0.10 + 0.90*torch.sigmoid(self.log_reach)

    def bake(self):
        """Draw every channel's kernel. Returns (C,1,2K+1,2K+1,2K+1)."""
        C, K = self.C, self.K
        P = self.off.shape[0]
        reach = self.reach()                                   # (C,)
        # u is the offset in units of THIS channel's reach, so a channel with a
        # small reach sees the same network input at a smaller distance
        u = self.off.unsqueeze(0)/(K*reach.view(C, 1, 1))      # (C,P,3)
        r = u.norm(dim=-1)                                     # (C,P)

        v = F.normalize(self.axis, dim=-1)                     # (A,3)
        p = torch.einsum("cpk,ak->cpa", u, v).clamp(-1.0, 1.0)
        # Chebyshev of the projection is cos(m * angle to that axis): the flat
        # page's angular order, about an axis that can point anywhere.
        cheb = [p, 2*p*p - 1.0]
        while len(cheb) < self.MO:
            cheb.append(2*p*cheb[-1] - cheb[-2])
        ang = torch.stack(cheb[:self.MO], -1).reshape(C, P, self.A*self.MO)

        # Concentric shells across the reach: the radial half of a lobe, as a
        # basis rather than as one hand-placed Gaussian.
        k = torch.arange(1, self.R + 1, device=r.device, dtype=r.dtype)
        kr = math.pi*r.unsqueeze(-1)*k
        ring = torch.cat([torch.cos(kr), torch.sin(kr)], -1)

        e = self.embed.unsqueeze(1).expand(C, P, -1)
        x = torch.cat([r.unsqueeze(-1), (r*r).unsqueeze(-1), u, ang, ring, e], -1)
        h = self.net(x.reshape(C*P, -1)).reshape(C, P, -1)
        w = torch.einsum("cph,ch->cp", h, self.head) + self.head_b.unsqueeze(1)

        # Outside its own reach a channel's kernel is zero; at the rim it is
        # feathered, because a hard cut at the edge of the window prints the
        # window's own shape onto the field.
        t = ((1.0 - r)/0.22).clamp(0, 1)
        taper = t*t*(3 - 2*t)
        w = w*taper
        # Zero mean, but subtracted as a multiple of the taper rather than as a
        # flat constant -- a constant would undo the feather and put a hard
        # circular cliff back at the rim. Then normalised, so the kernel's
        # overall scale and the force in front of it are not the same number
        # twice. Both exactly as the flat page bakes its own.
        c = (w.sum(1)/taper.sum(1).clamp_min(1e-6)).unsqueeze(1)
        w = w - c*taper
        w = w/w.abs().sum(1, keepdim=True).clamp_min(1e-6)
        return w.reshape(C, 1, 2*K + 1, 2*K + 1, 2*K + 1)

    def prep(self, k, M):
        """The bank, transformed once and ready to multiply.

        The kernel does not change inside an unroll either, so its transform is
        taken here rather than forty times over. Returns the spatial stencil
        unchanged when it is too big for the volume, and the direct path picks
        that up by asking whether what it was handed is complex.
        """
        K, C, S = self.K, self.C, 2*self.K + 1
        if S > M:
            return k
        kf = k.new_zeros(C, M, M, M)
        kf[:, :S, :S, :S] = k[:, 0].flip(-1).flip(-2).flip(-3)
        return torch.fft.rfftn(torch.roll(kf, (-K, -K, -K), (-3, -2, -1)),
                               dim=(-3, -2, -1))

    def forward(self, x, k=None):
        """x: (B,C,D,H,W) at half pitch. Returns the kernel integrals.

        `k` is what `prep` handed back: the bank drawn and transformed once.
        The bank does not change inside an unroll, and drawing and transforming
        it cost more than using it, so both happen once per iteration and the
        result is handed down; the gradient reaches the network through the one
        bake either way.

        Through an FFT, because the world is a torus and so the wrapped
        convolution the page wants is exactly the one a transform gives for
        nothing. A dense fifteen-cubed stencil against a twenty-cubed volume is
        three thousand multiplies a voxel done directly and a handful done this
        way, and the difference is the difference between a kernel bank you can
        afford to differentiate through forty times a step and one you cannot.
        """
        M = x.shape[-1]
        if k is None:
            k = self.prep(self.bake(), M)
        if not k.is_complex():          # the stencil did not fit: do it directly
            return F.conv3d(F.pad(x, (self.K,)*6, mode="circular"), k,
                            groups=self.C)
        dims = (-3, -2, -1)
        return torch.fft.irfftn(torch.fft.rfftn(x, dim=dims)*k.unsqueeze(0),
                                s=(M, M, M), dim=dims)
