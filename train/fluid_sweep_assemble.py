"""assemble.py select <n> <out.json>   -- score the sweep, pick n diverse worlds
   assemble.py build <capdir> <outdir> <gallery.html> -- gifs, contact sheet, gallery"""
import json, glob, math, os, sys
import numpy as np
from PIL import Image, ImageDraw

def load():
    rows = []
    for f in sorted(glob.glob('sweep/score-*.json')):
        rows += json.load(open(f))
    return rows

def score(r):
    if r['bad'] or not all(math.isfinite(r[k]) for k in ('cv', 'purity', 'act', 'occ', 'maxr', 'edge')):
        return 0.0
    # alive late: something still moves; structured: not flat, not one spike;
    # legible: fluids stay somewhat apart, and there are interfaces.
    alive = math.tanh(r['act']/0.25)
    struct = min(r['cv'], 1.0) * (1.0 if r['maxr'] < 40 else 0.2)
    legible = r['purity']**0.5
    edges = math.tanh(r['edge']/0.15)
    room = 1.0 if 0.03 < r['occ'] < 0.98 else 0.3
    # calm but structured is still worth a look; dead and flat is not
    return (0.25 + 0.75*alive)*struct*legible*edges*room

def select(n, out):
    rows = load()
    for r in rows: r['score'] = score(r)
    rows.sort(key=lambda r: -r['score'])
    print(f"{len(rows)} worlds scored; top scores:", [round(r['score'], 3) for r in rows[:8]])
    pool = rows[:max(n*3, 40)]
    def feat(r): return np.array([r['cv'], r['purity'], math.tanh(r['act']/0.25), r['edge']*3, r['occ'],
                                  r['g']/2.6, r['range']/14, r['spread'], r['seep'], r['seedMode']*0.5])
    F = np.array([feat(r) for r in pool]); F = (F - F.mean(0))/(F.std(0) + 1e-9)
    chosen = [0]
    while len(chosen) < min(n, len(pool)):
        d = np.min([np.linalg.norm(F - F[c], axis=1) for c in chosen], axis=0)
        # farthest point, but weighted by score so the picks stay good
        d = d * np.array([r['score'] for r in pool])**0.5
        d[chosen] = -1
        chosen.append(int(np.argmax(d)))
    sel = [pool[i] for i in chosen]
    sel.sort(key=lambda r: -r['score'])
    json.dump(sel, open(out, 'w'))
    for r in sel: print(r['idx'], r['kind'], round(r['score'], 3), {k: round(r[k], 2) for k in ('cv', 'purity', 'act', 'edge', 'occ')})

def frames_of(d):
    fs = sorted(glob.glob(os.path.join(d, 'f*.png')))
    return [Image.open(f).convert('RGB') for f in fs]

def save_gif(frames, path, ms=70):
    # one shared palette, so tiles do not flicker between frames
    sheet = Image.new('RGB', (frames[0].width, frames[0].height*min(len(frames), 8)))
    for i, f in enumerate(frames[::max(1, len(frames)//8)][:8]): sheet.paste(f, (0, i*frames[0].height))
    pal = sheet.quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    q = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    q[0].save(path, save_all=True, append_images=q[1:], duration=ms, loop=0, optimize=False, disposal=1)

def build(capdir, outdir, gallery):
    os.makedirs(outdir, exist_ok=True)
    dirs = sorted(glob.glob(os.path.join(capdir, '*/')))
    worlds = []
    for d in dirs:
        p = json.load(open(os.path.join(d, 'params.json')))
        fr = frames_of(d)[8:]                      # the seed's first frames are bland
        if not fr: continue
        name = f"{p['idx']:04d}"
        save_gif([f.resize((192, 192), Image.NEAREST) for f in fr], os.path.join(outdir, name + '.gif'))
        worlds.append((p, fr))
        print('gif', name, len(fr), 'frames')
    worlds.sort(key=lambda w: -w[0].get('score', 0))
    # contact sheet: a grid of the lot, tiles at 160
    T, cols = 128, math.ceil(math.sqrt(len(worlds)))
    rows_ = math.ceil(len(worlds)/cols)
    nf = min(len(w[1]) for w in worlds)
    sheet = []
    for i in range(nf):
        im = Image.new('RGB', (cols*T, rows_*T), (4, 8, 11))
        for k, (p, fr) in enumerate(worlds):
            im.paste(fr[i].resize((T, T), Image.NEAREST), ((k % cols)*T, (k//cols)*T))
        sheet.append(im)
    save_gif(sheet, os.path.join(outdir, 'sheet.gif'), ms=80)
    print('sheet', cols, 'x', rows_, nf, 'frames')
    # the gallery
    def link(p):
        q = {k: p[k] for k in ('M', 'seed', 'g', 'range', 'stiff', 'gamma', 'dt', 'drag', 'visc', 'couple',
                                'vmax', 'spread', 'seep', 'seedMode', 'amt', 'rad', 'fill')}
        q['N'] = 256
        import urllib.parse
        return 'fluid.html#p=' + urllib.parse.quote(json.dumps(q, separators=(',', ':')))
    def mat(p):
        cells = ''
        for c in range(4):
            for d in range(4):
                v = p['M'][c*4 + d]; a = 0.12 + 0.88*min(1, abs(v))
                col = f"rgba(84,196,122,{a:.2f})" if v >= 0 else f"rgba(226,84,96,{a:.2f})"
                cells += f'<i style="background:{col}"></i>'
        return f'<span class="m">{cells}</span>'
    tiles = ''
    for p, fr in worlds:
        name = f"{p['idx']:04d}"
        info = (f"{p['kind']} &middot; g {p['g']} &middot; range {p['range']} &middot; stiff {p['stiff']} "
                f"&middot; &gamma; {p['gamma']} &middot; dt {p['dt']} &middot; drag {p['drag']} &middot; visc {p['visc']} "
                f"&middot; couple {p['couple']} &middot; spread {p['spread']} &middot; seep {p['seep']} "
                f"&middot; {'blobs' if p['seedMode'] == 0 else 'noise'}")
        tiles += (f'<a class="t" href="{link(p)}" title="open in the simulation">'
                  f'<img src="fluid-sweep/{name}.gif" loading="lazy" width="192" height="192">'
                  f'<div class="c">{mat(p)}<span class="s">{p.get("score", 0):.2f}</span></div>'
                  f'<div class="i">{info}</div></a>\n')
    html = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fluid_sweep_gallery.html')).read().replace('{{TILES}}', tiles).replace('{{COUNT}}', str(len(worlds)))
    open(gallery, 'w').write(html)
    print('gallery', gallery, len(worlds))

if __name__ == '__main__':
    if sys.argv[1] == 'select': select(int(sys.argv[2]), sys.argv[3])
    else: build(sys.argv[2], sys.argv[3], sys.argv[4])
