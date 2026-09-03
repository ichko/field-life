import sys, warnings, numpy as np, torch, torch.nn.functional as F
warnings.filterwarnings('ignore'); sys.path.insert(0, '.')
import gecko, view
from field3d import Field3D
from PIL import Image
torch.set_num_threads(2)
ck, at = sys.argv[1], [int(v) for v in sys.argv[2].split(',')]
N, C, S, T = (int(v) for v in sys.argv[3:7])
m = Field3D(C=C, S=S, T=T, N=N); m.load_state_dict(torch.load(ck, map_location='cpu'))
if len(sys.argv) > 8 and sys.argv[8] == 'hard': m.soft = False
parts, occ = gecko.build_parts(N)
vis = torch.from_numpy(parts).unsqueeze(0).sum((0,2,3,4))
masses = torch.cat([vis, F.softplus(m.seed_mass.detach())[3:]])
with torch.no_grad(): _, sn = m.run(masses, max(at), keep=tuple(at))
pal = np.array([[0.16,0.55,0.20],[0.86,0.83,0.55],[0.95,0.62,0.20]], np.float32)
W, tiles = 250, []
for k in at:
    p = sn[k][0,:3].clamp(min=0).numpy()
    tiles.append((np.einsum('pzyx,pc->zyxc', p, pal), np.clip(p.sum(0), 0, 1)))
n = len(at)
imgs = [view.render(r,o,0.0,1.45,W) for r,o in tiles]+[view.render(r,o,0.7,0.32,W) for r,o in tiles]
sheet = np.full((2*W+6, n*W+6*(n-1), 3), 26, np.uint8)
for i in range(n):
    sheet[:W, i*(W+6):i*(W+6)+W] = imgs[i]; sheet[W+6:, i*(W+6):i*(W+6)+W] = imgs[n+i]
Image.fromarray(sheet).save(sys.argv[7])
print('peak %.2f' % float(sn[at[-1]][:, :3].max()),
      'mass', [round(float(x),1) for x in sn[at[-1]][0,:3].sum((1,2,3))])
