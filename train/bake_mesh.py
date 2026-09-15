"""Bake a mesh into a packed occupancy volume the repo can carry.

The asset itself is six megabytes and lives elsewhere; what the fit needs is a
filled volume, which at 128 cubed and one bit a voxel is a couple of hundred
kilobytes and compresses to far less, because an animal is a few per cent of
its own bounding box. Everything softer than that -- the antialiased edge, the
colour, the parts -- is derived on the way out, so the same bake serves any
resolution the fit wants to run at.

    python3 bake_mesh.py DragonAttenuation.glb Dragon dragon128.npz --N 128
"""
import argparse
import numpy as np

import glb
import voxelise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("mesh"); ap.add_argument("out")
    ap.add_argument("--N", type=int, default=128)
    ap.add_argument("--up", type=int, default=1, help="which axis of the asset is up")
    ap.add_argument("--length", type=int, default=0, help="which is the long one")
    a = ap.parse_args()

    pos, tri = glb.load(a.src, a.mesh)
    print("verts", pos.shape[0], "tris", tri.shape[0])
    occ = voxelise.voxelise(pos, tri, N=a.N, up=a.up, length=a.length)
    print("filled %.3f of the cube" % occ.mean())
    np.savez_compressed(a.out, bits=np.packbits(occ.astype(bool).ravel()),
                        N=np.int32(a.N))
    print("wrote", a.out)


def load(path):
    """Unpack a bake back into a binary volume, indexed [z, y, x]."""
    z = np.load(path)
    N = int(z["N"])
    return np.unpackbits(z["bits"])[:N**3].reshape(N, N, N).astype(np.float32)


if __name__ == "__main__":
    main()
