"""Views of a voxel volume: front-to-back, with a real normal off the field."""
import numpy as np
from PIL import Image

def rotmat(yaw, pitch):
    cy, sy, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
    return (np.array([[1,0,0],[0,cp,-sp],[0,sp,cp]])
            @ np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]]))

def sample(vol, q):
    N = vol.shape[0]
    g = (q + 0.5)*N - 0.5
    i0 = np.floor(g).astype(np.int32)
    f = (g - i0).astype(np.float32)
    out = np.zeros(q.shape[:-1] + (vol.shape[-1],), np.float32)
    for dz in (0, 1):
        for dy in (0, 1):
            for dx in (0, 1):
                idx = i0 + np.array([dx, dy, dz])
                ok = np.all((idx >= 0) & (idx < N), -1)
                w = ((f[...,0] if dx else 1-f[...,0])
                     *(f[...,1] if dy else 1-f[...,1])
                     *(f[...,2] if dz else 1-f[...,2]))*ok
                ic = np.clip(idx, 0, N-1)
                out += vol[ic[...,2], ic[...,1], ic[...,0]]*w[..., None]
    return out

def render(rgb, occ, yaw=0.6, pitch=0.32, W=300, steps=190, sigma=55.0):
    N = occ.shape[0]
    gz, gy, gx = np.gradient(occ.astype(np.float32))
    vol = np.concatenate([rgb, occ[..., None], -gx[..., None],
                          -gy[..., None], -gz[..., None]], -1).astype(np.float32)
    R = rotmat(yaw, pitch)
    u = (np.arange(W) + 0.5)/W - 0.5
    U, V = np.meshgrid(u*1.30, -u*1.30, indexing="xy")
    acc = np.zeros((W, W, 3), np.float32)
    trans = np.ones((W, W), np.float32)
    dt, key = 1.8/steps, np.array([0.45, 0.72, 0.53], np.float32)
    key = key/np.linalg.norm(key)
    for s in range(steps):
        cam = np.stack([U, V, np.full_like(U, -0.9 + s*dt)], -1)
        v = sample(vol, cam @ R)
        a = 1.0 - np.exp(-v[..., 3]*sigma*dt)
        n = v[..., 4:7]
        nl = np.sqrt((n*n).sum(-1, keepdims=True)) + 1e-6
        lam = np.clip((n/nl) @ key, 0, 1)
        alb = v[..., :3]/np.maximum(v[..., 3:4], 1e-4)
        acc += (trans*a)[..., None]*alb*(0.30 + 1.15*lam[..., None])
        trans *= 1 - a
    bg = (np.linspace(0.09, 0.15, W)[:, None, None]
          *np.ones((1, W, 3), np.float32))*np.array([0.9, 1.0, 1.15], np.float32)
    return (np.clip(acc + trans[..., None]*bg, 0, 1)**(1/2.2)*255).astype(np.uint8)

def sheet(rgb, occ, path, views, W=300, labels=None):
    tiles = [render(rgb, occ, y, p, W) for y, p in views]
    n = len(tiles)
    out = np.full((W, W*n + 6*(n - 1), 3), 26, np.uint8)
    for i, t in enumerate(tiles):
        out[:, i*(W + 6):i*(W + 6) + W] = t
    Image.fromarray(out).save(path)
    return path

if __name__ == "__main__":
    import sys, lizard
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    rgb, occ = lizard.build(N)
    sheet(rgb, occ, "lizard-views.png",
          [(0.0, 1.45), (0.0, 0.05), (0.75, 0.55), (2.4, 0.35)])
    print("wrote lizard-views.png")
