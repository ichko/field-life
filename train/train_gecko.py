"""Fit the 3D rule so that a small seed grows into an animal.

The same trick the flat page's lizard was made with: unroll the simulation from
a seed for a fixed number of steps, compare the leading channels against a
picture, and push the error back through every step into the rule itself -- the
matrix, the kernels, the three scalars, and what the seed is made of.

Which animal is `--target`, and it is any module with a `build_parts(N)` that
hands back a stack of pieces that do not overlap and sum to the whole. How many
pieces is up to the target; the fit reads it off the stack and scores that many
channels, so a five-part animal simply needs a couple more channels than a
three-part one.

Two things make it harder here than on a sheet. Mass is conserved, so the seed
has to arrive holding exactly what the finished animal weighs; that is not a
loss term, it is arithmetic, and the visible channels' seed masses are simply
set to the target's. And the error is compared at several late times rather than
one, because a shape the field passes through on its way somewhere else is not
a shape you can look at.
"""
import argparse, importlib, math, os, time
import torch
import torch.nn.functional as F

from field3d import Field3D, load_field


def soften(x, s):
    """A Gaussian blur, wrapping, in cells."""
    if s <= 0: return x
    K = max(1, int(math.ceil(3*s)))
    xs = torch.arange(-K, K + 1, dtype=x.dtype, device=x.device)
    w = torch.exp(-0.5*(xs/s)**2); w = w/w.sum()
    for ax in (0, 1, 2):
        pad = [0]*6; pad[2*(2 - ax)] = K; pad[2*(2 - ax) + 1] = K
        sh = [1, 1, 1, 1, 1]; sh[2 + ax] = w.numel()
        x = F.conv3d(F.pad(x, pad, mode="circular"),
                     w.view(sh).expand(x.shape[1], 1, *sh[2:]), groups=x.shape[1])
    return x


def target_field(mod, N, blur=0.0):
    """The picture to grow, softened to something the field can actually hold.

    The kernels here are Gaussians a cell or two wide, so the smallest thing the
    rule has a fixed point for is about that size; a target with detail finer
    than that is not a hard target, it is an impossible one, and asking for it
    spends the whole fit on a compromise that satisfies nothing. Blurred by
    about a cell the animal still reads as an animal -- legs, tail, head -- and
    is inside what the rule can do.
    """
    parts, occ = mod.build_parts(N)
    t = torch.from_numpy(parts).unsqueeze(0).contiguous()
    o = torch.from_numpy(occ).unsqueeze(0).unsqueeze(0).contiguous()
    return soften(t, blur), soften(o, blur)


def pyramid_loss(x, y, m, levels=3):
    """Compare at full pitch and at two coarser ones, with the animal weighted
    up against the empty cube around it.

    Both halves matter. The coarse levels carry WHERE the mass has to go, which
    is what a fine loss is worst at saying -- two blobs a few cells apart look
    equally wrong however far apart they are. And the weight map is what stops
    the fit walking into the answer that a target which is ninety-nine per cent
    empty makes so tempting: spread everything out thin and score well on the
    empty part. Mass is conserved, so that answer cannot lose the mass, it can
    only put it nowhere; weighting the occupied cells up makes nowhere expensive.
    """
    loss = 0.0
    w = 1.0
    for i in range(levels):
        loss = loss + w*((m*(x - y)**2).mean()/m.mean())
        if i < levels - 1:
            x, y, m = F.avg_pool3d(x, 2), F.avg_pool3d(y, 2), F.avg_pool3d(m, 2)
            w *= 2.0
    return loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="cute",
                    help="which animal: beast, gecko, or dragon")
    ap.add_argument("--N", type=int, default=40)
    ap.add_argument("--hidden", type=int, default=0,
                    help="channels beyond the target's parts. Zero means every "
                         "chemical is a piece of the animal and nothing else exists")
    ap.add_argument("--T", type=int, default=8)
    ap.add_argument("--S", type=int, default=3)
    ap.add_argument("--steps", type=int, default=48)
    ap.add_argument("--warm", type=int, default=20, help="steps at the start of training")
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seedR", type=float, default=2.6)
    ap.add_argument("--hold", type=int, default=8, help="also match this many steps later")
    ap.add_argument("--out", default="gecko_fit.json")
    ap.add_argument("--resume", default="")
    ap.add_argument("--wsil", type=float, default=2.5,
                    help="how much the silhouette outweighs the colour")
    ap.add_argument("--blur", type=float, default=0.9,
                    help="soften the target to what the kernels can hold, in cells")
    ap.add_argument("--pool", type=int, default=0,
                    help="keep this many grown states and restart from them")
    ap.add_argument("--chunk", type=int, default=14, help="steps run from a pool state")
    ap.add_argument("--fresh", type=float, default=0.25,
                    help="how often to start from the seed, at the end of the run")
    ap.add_argument("--fresh0", type=float, default=0.85,
                    help="and at the start of it")
    ap.add_argument("--wfresh", type=float, default=1.6,
                    help="how much a run from the seed outweighs a restart")
    ap.add_argument("--batch", type=int, default=4, help="pool states run at once")
    ap.add_argument("--wout", type=float, default=6.0,
                    help="penalty on chemical sitting outside the animal")
    ap.add_argument("--recycle", type=int, default=14, help="retire a pool state after this many visits")
    ap.add_argument("--kernel", default="gauss", choices=("gauss", "cppn"),
                    help="displaced Gaussians, or a network over the offset vector")
    ap.add_argument("--K", type=int, default=7, help="stencil half-width, in half-pitch cells")
    ap.add_argument("--axes", type=int, default=4, help="learned axes for the angular orders")
    ap.add_argument("--orders", type=int, default=3, help="angular orders about each axis")
    ap.add_argument("--device", default="cpu", help="cpu, or cuda")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(3)

    dev = torch.device(a.device)
    tgt, occ = target_field(importlib.import_module(a.target), a.N, a.blur)
    tgt, occ = tgt.to(dev), occ.to(dev)
    P = tgt.shape[1]                                      # how many parts, and so
    C = P + a.hidden                                      # how many channels
    vis_mass = tgt.sum((0, 2, 3, 4))                      # what the picture weighs
    wmap = 1.0 + 14.0*occ                                 # the animal against the void
    # Where the animal is not. Nothing is allowed to live here: no chemical
    # standing outside the body holding its shape up from a distance, which a
    # fit will happily invent if nothing stops it and which means the thing you
    # see is not the thing that is there. Dilated by a cell, because the edge
    # has to be allowed to breathe.
    outside = 1.0 - torch.clamp(soften(occ, 1.0)*3.0, 0.0, 1.0)
    tot = float(vis_mass.sum())
    print("target:", a.target, "parts:", P, "channels:", C,
          "mass each:", [round(float(v), 1) for v in vis_mass])

    # Resuming rebuilds the model the checkpoint came from rather than the one
    # the flags describe. The two disagreeing is not a thing anyone notices at
    # the call site -- the flags that made a run are the first thing forgotten
    # -- and it fails minutes in, as a shape mismatch.
    if a.resume and os.path.exists(a.resume):
        m = load_field(a.resume, a.N, map_location=dev).to(dev)
        if m.C != C:
            raise SystemExit("%s has %d channels, this target wants %d"
                             % (a.resume, m.C, C))
        print("resumed from", a.resume, "S", m.S, "T", m.T, "seedR", m.seedR)
    else:
        m = Field3D(C=C, S=a.S, T=a.T, N=a.N, seedR=a.seedR,
                    kernel=a.kernel, K=a.K, axes=a.axes, orders=a.orders).to(dev)
    # hidden channels start with about as much mass as a visible one. The
    # inverse of softplus overflows the moment its argument is large, which for
    # a mass of a few hundred it certainly is, so it is taken the safe way.
    def inv_softplus(x):
        return torch.where(x > 20.0, x, torch.log(torch.expm1(x.clamp(max=20.0))))
    if not a.resume:
        with torch.no_grad():
            m.seed_mass.copy_(inv_softplus(torch.full((C,), float(vis_mass.mean()))))

    opt = torch.optim.Adam(m.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.iters, eta_min=a.lr*0.08)

    # The pool. A rule that is only ever run from the seed learns to ARRIVE at
    # the target and not to STAY there, which is why lengthening the unroll
    # keeps throwing the loss back up: nothing has ever asked the pattern to
    # stop. So keep a bag of states the rule has already made, start from one of
    # those as often as from the seed, and score where it gets to. A state that
    # is already the animal is then scored on whether it is still the animal a
    # dozen steps later, which is the only way a fixed point gets learned.
    pool = None
    if a.pool > 0:
        pool = torch.zeros(a.pool, C, a.N, a.N, a.N, device=dev)
        pool_age = torch.zeros(a.pool, dtype=torch.long)

    best = float("inf")
    t0 = time.time()
    rng = torch.Generator().manual_seed(11)
    for it in range(a.iters):
        masses = F.softplus(m.seed_mass).clone()
        masses = torch.cat([vis_mass, masses[P:]])        # visible mass is not free

        wt, mode = 1.0, "run "
        if pool is None:
            frac = min(1.0, it/(0.55*a.iters))
            steps = int(round(a.warm + (a.steps - a.warm)*frac))
            keep = tuple(sorted({steps, steps + a.hold//2, steps + a.hold}))
            rho, snaps = m.run(masses, keep[-1], keep=keep)
        else:
            m.dscale.fill_(float(a.N**3)/float(masses.sum().detach()))
            live = (pool_age > 0).nonzero().flatten()
            # Hold the fresh fraction high at the start and let it fall. Only
            # the run from the seed ever has to BUILD anything; a restart only
            # has to keep what is already there, and a rule asked to hold before
            # it can make will hold the simplest thing it can find -- which is
            # exactly what happened the last time this was fixed at a fifth.
            pf = a.fresh0 + (a.fresh - a.fresh0)*min(1.0, it/max(1, a.iters - 1))
            fresh = len(live) == 0 or float(torch.rand(1, generator=rng)) < pf
            if fresh:
                rho, slots, steps, wt = m.seed(masses), None, a.warm, a.wfresh
                mode = "seed"
            else:
                # Several restarts at once. They all run the same number of
                # steps, so they go through the unroll together as a batch, and
                # the gradient per second is multiplied for nearly nothing.
                pick = live[torch.randperm(len(live), generator=rng)[:a.batch]]
                rho, slots, steps = pool[pick].clone(), pick, a.chunk
                mode = "pool"
            keep = tuple(sorted({max(1, steps - a.hold//2), steps}))
            snaps, kbank = {}, m.bank()
            for i in range(1, steps + 1):
                rho = m.step(rho, kbank)
                if i in keep: snaps[i] = rho
            with torch.no_grad():
                # age 0 means the slot is empty. A slot visited too many times
                # is retired rather than kept for ever, so the pool never drifts
                # away from states the seed can actually reach.
                if slots is None:
                    j = int(torch.randint(a.pool, (1,), generator=rng))
                    pool[j] = rho[0].detach()
                    pool_age[j] = 1
                else:
                    pool[slots] = rho.detach()
                    pool_age[slots] += 1
                    pool_age[slots[pool_age[slots] > a.recycle]] = 0

        # Shape first, colour second. Left to itself the fit spends its early
        # effort deciding where red sits against where green sits -- which comes
        # out as rainbow fringing on a body that is not a body yet. Scoring the
        # silhouette, the three channels added together, separately and heavily
        # says: agree on where the animal is, then argue about its colour.
        loss = 0.0
        for k in keep:
            w = 1.0 if k == keep[-1] else 0.6
            r = snaps[k]
            vis = r[:, :P]
            # the fraction of everything the field is carrying that has ended up
            # outside the animal, which should be none of it
            spill = (r*outside).sum()/(r.shape[0]*tot)
            loss = loss + w*(a.wsil*pyramid_loss(vis.sum(1, keepdim=True),
                                                 tgt.sum(1, keepdim=True), wmap)
                             + pyramid_loss(vis, tgt, wmap)
                             + a.wout*spill)
        loss = wt*loss/len(keep)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(m.parameters(), 2.0)
        opt.step()
        sched.step()

        if not torch.isfinite(loss):
            print("diverged at", it); break
        if float(loss) < best:
            best = float(loss)
            torch.save(m.state_dict(), a.out.replace(".json", ".pt"))
        # With a pool the loss alternates between easy iterations from the seed
        # and hard ones from a grown state, so the lowest score is not the best
        # rule -- it is the luckiest draw. Keep the latest as well, and judge it
        # by running it.
        if it % 50 == 0:
            torch.save(m.state_dict(), a.out.replace(".json", "_last.pt"))
        if it % 10 == 0 or it == a.iters - 1:
            with torch.no_grad():
                rk = snaps[keep[-1]]
                err = float(F.mse_loss(rk[:, :P], tgt))
                sp = float((rk*outside).sum()/(rk.shape[0]*tot))
            print(f"{it:4d} {mode} {steps:3d} "
                  f"loss {float(loss):.5f} best {best:.5f} mse {err:.5f} "
                  f"out {sp:.3f} |g| {float(gn):.2f} "
                  f"{(time.time()-t0)/max(it,1):.1f}s/it", flush=True)
    print("done, best", best, "in %.1f min" % ((time.time() - t0)/60))


if __name__ == "__main__":
    main()
