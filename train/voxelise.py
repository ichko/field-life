"""Solid voxelisation by ray parity: for every column of the grid, find where
the surface crosses it and fill between alternate crossings."""
import numpy as np

def voxelise(pos, tri, N=64, margin=0.06, up=1, length=0):
    p = pos.copy()
    # put the long axis along x, the up axis along y, the rest on z
    order = [length, up, [a for a in (0,1,2) if a not in (length, up)][0]]
    p = p[:, order]
    lo, hi = p.min(0), p.max(0)
    c, s = (lo + hi)/2, (hi - lo).max()
    p = (p - c)/s*(1 - 2*margin)                       # into -0.5..0.5, centred
    v = p[tri]                                         # (T,3,3)

    lin = (np.arange(N) + 0.5)/N - 0.5
    occ = np.zeros((N, N, N), np.float32)              # [z, y, x]
    hits = [[[] for _ in range(N)] for _ in range(N)]  # per (y, x) column, along z

    a, b, cc = v[:, 0], v[:, 1], v[:, 2]
    xmin = np.minimum(np.minimum(a[:,0], b[:,0]), cc[:,0])
    xmax = np.maximum(np.maximum(a[:,0], b[:,0]), cc[:,0])
    ymin = np.minimum(np.minimum(a[:,1], b[:,1]), cc[:,1])
    ymax = np.maximum(np.maximum(a[:,1], b[:,1]), cc[:,1])
    i0 = np.clip(np.ceil((xmin + 0.5)*N - 0.5).astype(int), 0, N-1)
    i1 = np.clip(np.floor((xmax + 0.5)*N - 0.5).astype(int), 0, N-1)
    j0 = np.clip(np.ceil((ymin + 0.5)*N - 0.5).astype(int), 0, N-1)
    j1 = np.clip(np.floor((ymax + 0.5)*N - 0.5).astype(int), 0, N-1)

    for t in range(len(v)):
        if i1[t] < i0[t] or j1[t] < j0[t]: continue
        A, B, C = v[t]
        X, Y = np.meshgrid(lin[i0[t]:i1[t]+1], lin[j0[t]:j1[t]+1], indexing='xy')
        d = ((B[1]-C[1])*(A[0]-C[0]) + (C[0]-B[0])*(A[1]-C[1]))
        if abs(d) < 1e-12: continue
        l1 = ((B[1]-C[1])*(X-C[0]) + (C[0]-B[0])*(Y-C[1]))/d
        l2 = ((C[1]-A[1])*(X-C[0]) + (A[0]-C[0])*(Y-C[1]))/d
        l3 = 1 - l1 - l2
        m = (l1 >= 0) & (l2 >= 0) & (l3 >= 0)
        if not m.any(): continue
        z = l1*A[2] + l2*B[2] + l3*C[2]
        jj, ii = np.nonzero(m)
        for k in range(len(ii)):
            hits[j0[t] + jj[k]][i0[t] + ii[k]].append(z[jj[k], ii[k]])

    for jy in range(N):
        for ix in range(N):
            h = hits[jy][ix]
            if len(h) < 2: continue
            h = np.sort(np.array(h))
            for k in range(0, len(h) - 1, 2):
                k0 = int(np.ceil((h[k] + 0.5)*N - 0.5))
                k1 = int(np.floor((h[k+1] + 0.5)*N - 0.5))
                if k1 >= k0: occ[max(k0,0):min(k1+1,N), jy, ix] = 1.0
    return occ

def smooth(occ, passes=1):
    """One pass of a 3^3 box, so the target has a soft edge to aim at."""
    o = occ.astype(np.float32)
    for _ in range(passes):
        s = np.zeros_like(o)
        for dz in (-1,0,1):
            for dy in (-1,0,1):
                for dx in (-1,0,1):
                    s += np.roll(np.roll(np.roll(o, dz, 0), dy, 1), dx, 2)
        o = s/27.0
    return np.clip(o*1.35, 0, 1)
