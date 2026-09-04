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

Those projections, the radius, and a per-channel embedding go into a small MLP,
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
    def __init__(self, C, K=7, axes=4, orders=3, emb=10, hidden=48, seed=5):
        super().__init__()
        self.C, self.K, self.A, self.MO = C, K, axes, orders
        g = torch.Generator().manual_seed(seed)

        # the offsets the stencil covers, in cells, once
        d = torch.arange(-K, K + 1, dtype=torch.float32)
        gz, gy, gx = torch.meshgrid(d, d, d, indexing="ij")
        self.register_buffer("off", torch.stack([gz, gy, gx], -1).reshape(-1, 3))

        self.axis = nn.Parameter(torch.randn(axes, 3, generator=g))
        self.embed = nn.Parameter(torch.randn(C, emb, generator=g)*1.4)
        # Reach per channel, as a fraction of the stencil. Spread the starting
        # values across the range so the bank begins with big kernels and small
        # ones rather than C copies of the same middling one.
        start = torch.linspace(-1.4, 1.4, C)
        self.log_reach = nn.Parameter(start)

        nin = 2 + 3 + axes*orders + emb
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
        """Each channel's radius as a fraction of the stencil, 0.22 to 1."""
        return 0.22 + 0.78*torch.sigmoid(self.log_reach)

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

        e = self.embed.unsqueeze(1).expand(C, P, -1)
        x = torch.cat([r.unsqueeze(-1), (r*r).unsqueeze(-1), u, ang, e], -1)
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

    def forward(self, x):
        """x: (1,C,D,H,W) at half pitch. Returns the kernel integrals."""
        k = self.bake()
        K = self.K
        x = F.pad(x, (K,)*6, mode="circular")
        return F.conv3d(x, k, groups=self.C)
