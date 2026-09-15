"""Just enough glTF to get one mesh's triangles out of a .glb."""
import json, struct
import numpy as np

CT = {5120:'i1', 5121:'u1', 5122:'i2', 5123:'u2', 5125:'u4', 5126:'f4'}
NC = {'SCALAR':1, 'VEC2':2, 'VEC3':3, 'VEC4':4}

def load(path, mesh_name=None):
    b = open(path, 'rb').read()
    _, _, ln = struct.unpack_from('<III', b, 0)
    off, chunks = 12, []
    while off < ln:
        clen, ctype = struct.unpack_from('<II', b, off)
        chunks.append((ctype, off + 8, clen)); off += 8 + clen
    j = json.loads(b[chunks[0][1]:chunks[0][1] + chunks[0][2]].decode('utf8'))
    binoff = chunks[1][1]

    def acc(i):
        a = j['accessors'][i]
        bv = j['bufferViews'][a['bufferView']]
        dt = np.dtype(CT[a['componentType']]).newbyteorder('<')
        n = NC[a['type']]
        start = binoff + bv.get('byteOffset', 0) + a.get('byteOffset', 0)
        stride = bv.get('byteStride') or dt.itemsize*n
        raw = np.frombuffer(b, dtype=np.uint8,
                            offset=start, count=stride*(a['count'] - 1) + dt.itemsize*n)
        out = np.zeros((a['count'], n), dt)
        for k in range(a['count']):
            out[k] = np.frombuffer(raw, dt, n, k*stride)
        return out

    for m in j['meshes']:
        if mesh_name and m.get('name') != mesh_name: continue
        p = m['primitives'][0]
        pos = acc(p['attributes']['POSITION']).astype(np.float64)
        idx = acc(p['indices']).astype(np.int64).ravel()
        return pos, idx.reshape(-1, 3)
    raise SystemExit('mesh not found')

if __name__ == '__main__':
    import sys
    pos, tri = load(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
    print('verts', pos.shape, 'tris', tri.shape)
    print('bbox', pos.min(0), pos.max(0))
