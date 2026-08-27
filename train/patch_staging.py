"""
Apply the four changes of docs/nca-experiment.md §7 to a copy of the page.

    python3 train/patch_staging.py staging/index.html

Why a script and not a committed diff: the simulation is one 3400-line file
that is edited often, and these changes have to be re-applied on top of
whatever it has become. Replaying twenty surgical replacements onto the
current file keeps other people's work; copying a patched file over it does
not -- which is exactly the mistake this script exists to stop. Every
replacement asserts it matched exactly once, so a moved anchor is a loud
failure rather than a silent no-op.

Idempotent: running it on an already-patched file exits without touching it.

  1  bakeKernel honours a lobe's angular order m and phase
  2  separablePlan refuses angular kernels
  3  uploadKernels bakes on the shared grid, reach as a radial scale
  4  FS_DRAW gains blend 4, channels 0-2 straight to R, G, B
  5  a Rate control, capping how often the world advances
  6  FS_DRAW gains blend 5: channels 0-2 as themselves, the rest screened
     in behind them

plus seedFromMasses, so a preset can carry the mass its picture costs, and a
square lattice for worlds trained on one.
"""

import sys

EDITS = [
    # ---- 1. bakeKernel honours m and phase -------------------------------
    ('''  const terms = (k.terms || []).map(t => ({ a: t.a, r: t.r, w: Math.max(t.w, wmin) }));''',
     '''  const terms = (k.terms || []).map(t => ({ a: t.a, r: t.r, w: Math.max(t.w, wmin),
                                            m: t.m || 0, phase: t.phase || 0 }));'''),

    ('''          }else{
            for(const t of terms)
              v += t.a*Math.exp(-(((rr - t.r)/t.w)**2));
          }''',
     '''          }else{
            // A lobe may carry an angular order m and a phase. m = 0 is the
            // radial ring this has always drawn -- Lenia's shape. m = 1 is a
            // signed gradient along an axis, which is exactly what a Sobel
            // filter is, so an NCA's identity/Sobel-x/Sobel-y perception is
            // the case m in {0, 1, 1}. Higher orders add finer angular
            // structure. Orders above zero are also the only way a kernel can
            // tell one direction from another, and a rotationally symmetric
            // bank therefore has only rotationally symmetric fixed points --
            // it cannot hold anything shaped like an animal.
            // Absent fields mean m = 0 and phase = 0, so every kernel written
            // before this bakes to exactly what it did.
            const th = Math.atan2(py, px);
            for(const t of terms)
              v += t.a*Math.exp(-(((rr - t.r)/t.w)**2))
                 * (t.m ? Math.cos(t.m*th + t.phase) : 1.0);
          }'''),

    # ---- 2. separablePlan refuses angular kernels -------------------------
    ('''    if(k.type !== "radial" || k.sym !== "radial") return null;''',
     '''    if(k.type !== "radial" || k.sym !== "radial") return null;
    // The separable path rebuilds a kernel from two 1-D blurs, which can only
    // ever produce a radial shape. An angular lobe run down it would silently
    // come out as a plain difference of gaussians.
    if((k.terms || []).some(t => t.m)) return null;'''),

    # ---- 3. reach as a radial scale on one shared grid --------------------
    ('''function bakeKernel(k, KR){
  const K = 2*KR + 1, out = new Float32Array(K*K);''',
     '''// KR is the grid this is baked on; RR is the kernel's REACH on that grid, in
// cells. They used to be the same number, which meant a short-reach colour was
// baked on a smaller grid and pasted into the middle -- so its reach was a
// resolution, quantised by a round(), and it lost detail for no reason. Baking
// every colour on the shared grid and letting the reach be a plain radial
// scale is both sharper and continuous, and continuity is what lets a reach be
// fitted rather than only chosen.
function bakeKernel(k, KR, RR){
  const K = 2*KR + 1, out = new Float32Array(K*K);
  RR = RR || KR;'''),

    ('''  const wmin = 2.0/KR;''',
     '''  // A lobe of relative width w spans w*RR cells, so the width the stencil can
  // no longer resolve is 2/RR -- set by the kernel's reach, not by the grid it
  // is baked on. Capped, so a very short reach cannot demand a width wider
  // than the disc itself.
  const wmin = Math.min(2.0/RR, 0.7);'''),

    ('''          const rr = Math.hypot(px, py)/KR;''',
     '''          const rr = Math.hypot(px, py)/RR;'''),

    ('''      const rr = Math.hypot(x, y)/KR, i = (y + KR)*K + (x + KR);''',
     '''      const rr = Math.hypot(x, y)/RR, i = (y + KR)*K + (x + KR);'''),

    ('''    const kr = Math.max(2, Math.round(kernelKR*Math.min(1, k.R/Rmax)));
    const small = bakeKernel(k, kr), SK = 2*kr + 1;
    const layer = c >> 2, comp = c & 3;
    for(let y = 0; y < SK; y++)
      for(let x = 0; x < SK; x++){
        const gy = y - kr + kernelKR, gx = x - kr + kernelKR;
        data[((layer*K + gy)*K + gx)*4 + comp] = small[y*SK + x];
      }''',
     '''    const rr = Math.max(2, kernelKR*Math.min(1, k.R/Rmax));
    const full = bakeKernel(k, kernelKR, rr);
    const layer = c >> 2, comp = c & 3;
    for(let y = 0; y < K; y++)
      for(let x = 0; x < K; x++)
        data[((layer*K + y)*K + x)*4 + comp] = full[y*K + x];'''),

    ('''  const k = bakeKernel(kernels[c], KR);''',
     '''  const k = bakeKernel(kernels[c], KR, KR);'''),

    # ---- 4. a raw RGB read-out -------------------------------------------
    ('''  if(uSmooth < 0.5) f = floor(f + 0.5);''',
     '''  if(uSmooth < 0.5) f = floor(f + 0.5);
  // Blend 4: the first three channels ARE red, green and blue. No palette, no
  // tone map, no lifted background -- what is on screen is the density itself,
  // which is the only way to compare it against a picture.
  if(uBlend == 4){
    vec4 v0 = mix(mix(cell(0,i0.x,i0.y),   cell(0,i0.x+1,i0.y),   f.x),
                  mix(cell(0,i0.x,i0.y+1), cell(0,i0.x+1,i0.y+1), f.x), f.y);
    oC = vec4(clamp(v0.rgb*uExp, 0.0, 1.0), 1.0);
    return;
  }'''),

    ('''      <option value="2">Additive</option><option value="3">Winner</option>''',
     '''      <option value="2">Additive</option><option value="3">Winner</option>
      <option value="4">RGB</option>'''),

    # ---- a square lattice, and a seed carrying prescribed mass ------------
    ('''function sizeGrid(){
  const w = cv.clientWidth || cv.width || 1, ht = cv.clientHeight || cv.height || 1;''',
     '''function sizeGrid(){
  // A world trained on a square torus has to run on one: the lattice normally
  // follows the canvas aspect, and a wide canvas would stretch it.
  if(S.square){ S.NX = S.NY = S.N; return; }
  const w = cv.clientWidth || cv.width || 1, ht = cv.clientHeight || cv.height || 1;'''),

    ('''const seed = () => fillField(false);
const seedDisc = () => fillField(true);''',
     '''const seed = () => fillField(false);
const seedDisc = () => fillField(true);

// Seed a soft central disc holding an EXACT mass in each channel. The update
// only MOVES mass, so a trained world has to be handed the mass its picture
// costs -- the red that the lizard's red adds up to, and so on -- and the
// whole of the task is then where to put it. A disc rather than one lit cell
// because mass travels one cell per step.
function seedFromMasses(masses){
  if(!T.rhoA) return;
  const R = Math.min(S.NX, S.NY)*0.10, soft = 1.5;
  const w = S.NX*S.L, px = new Float32Array(w*S.NY*4);
  const disc = new Float64Array(S.NX*S.NY);
  let sum = 0;
  for(let y = 0; y < S.NY; y++){
    for(let x = 0; x < S.NX; x++){
      const d = Math.hypot(y - (S.NY - 1)/2, x - (S.NX - 1)/2);
      let e = Math.min(1, Math.max(0, (R - d)/soft));
      e = e*e*(3 - 2*e);
      disc[y*S.NX + x] = e; sum += e;
    }
  }
  for(let c = 0; c < S.C; c++){
    const per = (masses[c] || 0)/Math.max(sum, 1e-9);
    for(let y = 0; y < S.NY; y++)
      for(let x = 0; x < S.NX; x++)
        px[(y*w + (c >> 2)*S.NX + x)*4 + (c & 3)] = per*disc[y*S.NX + x];
  }
  gl.bindTexture(gl.TEXTURE_2D, T.rhoA);
  gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, w, S.NY, gl.RGBA, gl.FLOAT, px);
  S.tick = 0; mass0 = null;
}'''),

    ('''function reseed(){ (S.seedMode === "disc" ? seedDisc : seed)(); }''',
     '''function reseed(){
  if(S.seedMode === "masses" && S.seedMasses) return seedFromMasses(S.seedMasses);
  (S.seedMode === "disc" ? seedDisc : seed)();
}'''),

    # ---- carry the new fields through save/load ---------------------------
    ('''"cfreq","cdepth"];''', '''"cfreq","cdepth","square"];'''),

    ('''  const resize = (c.C !== S.C || c.N !== S.N);''',
     '''  // A change of lattice shape is a resize too: a world pinned square needs new
  // textures even when its channel count and size are unchanged.
  const resize = (c.C !== S.C || c.N !== S.N
                  || !!c.square !== !!S.square);'''),

    ('''  S.seed = c.seed ?? S.seed;
  palette = makePalette(S.C, S.palette);''',
     '''  S.seed = c.seed ?? S.seed;
  if(Array.isArray(c.seedMasses)) S.seedMasses = c.seedMasses.slice();
  palette = makePalette(S.C, S.palette);'''),

    # ---- a button, and the world loaded on open --------------------------
    ('''      <button id="reclear" title="Clear every color out of the world">Empty</button>''',
     '''      <button id="reclear" title="Clear every color out of the world">Empty</button>
      <button id="lizard" title="Load the trained world and grow it from its seed">Lizard</button>'''),

    ('''  $("reclear").addEventListener("click", () => $("clear").click());''',
     '''  $("reclear").addEventListener("click", () => $("clear").click());
  $("lizard").addEventListener("click", () => loadLizard());'''),

    ('''  requestAnimationFrame(frame);
}

addEventListener("error", e => {''',
     '''  loadLizard().catch(() => {});     // the trained world, if it is sitting beside us

  requestAnimationFrame(frame);
}

// Load the trained preset and put its seed in the middle. The grid is pinned
// square because that is the torus it was trained on, and the view is switched
// to the raw RGB read-out because the first three channels ARE the picture.
async function loadLizard(){
  const res = await fetch("lizard.json", { cache: "no-cache" });
  if(!res.ok) throw new Error("no lizard.json");
  const cfg = await res.json();
  cfg.square = true;
  cfg.blend = 5;
  cfg.expo = 1;
  cfg.seedMode = "masses";
  if(!applyConfig(cfg)) throw new Error("lizard.json was rejected");
  allocate();                       // the square lattice is a new lattice
  reseed();
  S.holdUntil = performance.now() + 400;
  S.running = true;
  const btn = document.getElementById("play");
  if(btn) btn.textContent = "Pause";
}

addEventListener("error", e => {'''),

    # ---- Rate: how often the world advances ----------------------------
    ('    <div class="row"><label class="lab" for="inter">Law</label><select id="inter" title="Built-in interaction shape, when no kernel is designed">',
     '    <div class="row"><label class="lab" for="fps">Rate</label><input id="fps" title="How many times a second the world advances. 60 runs as fast as the display will go; turn it down to watch a pattern form. Drawing is unaffected, so panning and painting stay smooth however slow this is." type="range" min="1" max="60" step="1" value="60"><span class="val" id="fpsv">60</span></div>\n    <div class="row"><label class="lab" for="inter">Law</label><select id="inter" title="Built-in interaction shape, when no kernel is designed">'),

    ('let lastT = performance.now(), fpsAcc = 0, fpsN = 0, mass0 = null, painting = null;',
     'let lastT = performance.now(), fpsAcc = 0, fpsN = 0, mass0 = null, painting = null;\nlet lastStepT = 0;                    // when the world last advanced, for the Rate cap'),

    ('  if(S.running && performance.now() >= S.holdUntil)\n    for(let i = 0; i < S.steps; i++) step();\n  render();',
     '  // Rate caps how often the world ADVANCES, not how often it is drawn. Gating\n  // the draw instead would make panning and painting judder at low rates, and\n  // the point is to watch a pattern form, not to watch the page struggle.\n  const tNow = performance.now();\n  const due = S.fps >= 60 || tNow - lastStepT >= 1000/Math.max(S.fps, 0.1);\n  if(S.running && tNow >= S.holdUntil && due){\n    lastStepT = tNow;\n    for(let i = 0; i < S.steps; i++) step();\n  }\n  render();'),

    ('  slider("steps", "steps");',
     '  slider("steps", "steps");\n  slider("fps", "fps");'),

    ('"cfreq","cdepth","square"];',
     '"cfreq","cdepth","square","fps"];'),

    # ---- blend 5: the picture over its own scaffold --------------------
    ('  vec3 hue = vec3(0.0), add = vec3(0.0), best = vec3(0.0);\n  float wsum = 0.0, total = 0.0, top = -1.0;\n  for(int l = 0; l < uL; l++){\n    vec4 v = mix(mix(cell(l,i0.x,i0.y),   cell(l,i0.x+1,i0.y),   f.x),\n                 mix(cell(l,i0.x,i0.y+1), cell(l,i0.x+1,i0.y+1), f.x), f.y);\n    for(int k = 0; k < 4; k++){\n      if(l*4 + k >= uC) break;\n      float m = max(v[k], 0.0);\n      vec3 col = uPal[l*4 + k];\n      float w = uBlend == 1 ? m : m*m;\n      total += m; wsum += w; hue += w*col; add += m*col;\n      if(m > top){ top = m; best = col; }\n    }\n  }\n  float a = 1.0 - exp(-total*uExp);\n  vec3 rgb = uBlend == 2 ? vec3(1.0) - exp(-add*uExp)\n           : uBlend == 3 ? best*a\n           : (hue/max(wsum, 1e-12))*a;',
     '  vec3 hue = vec3(0.0), add = vec3(0.0), best = vec3(0.0);\n  // ...and the same sums again over the channels PAST the first three, plus\n  // the first three kept raw, so blend 5 can put one on top of the other.\n  vec3 hid = vec3(0.0), vis = vec3(0.0);\n  float wsum = 0.0, total = 0.0, top = -1.0, hidW = 0.0, hidT = 0.0;\n  for(int l = 0; l < uL; l++){\n    vec4 v = mix(mix(cell(l,i0.x,i0.y),   cell(l,i0.x+1,i0.y),   f.x),\n                 mix(cell(l,i0.x,i0.y+1), cell(l,i0.x+1,i0.y+1), f.x), f.y);\n    for(int k = 0; k < 4; k++){\n      int c = l*4 + k;\n      if(c >= uC) break;\n      float m = max(v[k], 0.0);\n      vec3 col = uPal[c];\n      float w = uBlend == 1 ? m : m*m;\n      total += m; wsum += w; hue += w*col; add += m*col;\n      if(m > top){ top = m; best = col; }\n      if(c == 0) vis.r = m;\n      else if(c == 1) vis.g = m;\n      else if(c == 2) vis.b = m;\n      else { hidW += m*m; hid += m*m*col; hidT += m; }\n    }\n  }\n  float a = 1.0 - exp(-total*uExp);\n  // Blend 5. The first three channels are the picture and are drawn as\n  // themselves; everything else is drawn as it would be under Dominant and\n  // screened in underneath at a fraction of its strength. Read as RGB alone a\n  // world looks like the whole of its state, which it is not -- the shape sits\n  // inside a much larger scaffold of hidden mass. Blended flat, that scaffold\n  // drowns it. This keeps the picture legible and what holds it up visible.\n  // Screen rather than addition: two bright layers added clip to white and\n  // lose both, where screen keeps the brighter of the two readable.\n  vec3 back = (hid/max(hidW, 1e-12))*(1.0 - exp(-hidT*uExp));\n  vec3 over = vec3(1.0) - (vec3(1.0) - clamp(vis*uExp, 0.0, 1.0))\n                        *(vec3(1.0) - back*0.42);\n  vec3 rgb = uBlend == 5 ? over\n           : uBlend == 2 ? vec3(1.0) - exp(-add*uExp)\n           : uBlend == 3 ? best*a\n           : (hue/max(wsum, 1e-12))*a;'),

    ('      <option value="4">RGB</option>',
     '      <option value="4">RGB</option>\n      <option value="5">RGB first</option>'),
]


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "staging/index.html"
    src = open(path).read()
    if "loadLizard" in src:
        print(f"{path} is already patched; nothing to do")
        return 0
    for i, (old, new) in enumerate(EDITS, 1):
        got = src.count(old)
        if got != 1:
            sys.exit(f"edit {i} matched {got} times, expected 1 -- the file has "
                     f"moved under this patch:\n  {old.strip().splitlines()[0][:90]}")
        src = src.replace(old, new)
    open(path, "w").write(src)
    print(f"applied {len(EDITS)} edits to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
