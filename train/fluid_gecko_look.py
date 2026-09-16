"""Roll the trained rule out from its disc and lay the frames in a strip:
    python3 train/fluid_gecko_look.py staging/fluid-gecko.json out.png [steps...]"""
import json, sys, os
import numpy as np, jax, jax.numpy as jnp
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fluid_gecko as fg

def paint(rho):                                     # (4,H,W) -> (H,W,3), row 0 at the top
    im = np.clip(np.einsum("chw,cd->hwd", np.asarray(rho[:4]), fg.COL), 0, 1)
    return (im[::-1]*255).astype(np.uint8)

def main():
    j = json.load(open(sys.argv[1])); out = sys.argv[2]
    marks = [int(x) for x in sys.argv[3:]] or [0, 16, 32, 48, 64, 96, 128, 192, 256, 400]
    params = {k: jnp.asarray(np.array(j[k], np.float32)) for k in ("w1", "b1", "w2", "b2")}
    disc_w = jnp.asarray(fg.disc(j["grid"], j["disc"]))
    rho = jnp.asarray(j["masses"])[:, None, None]*disc_w[None]
    s = (rho, jnp.zeros_like(rho), jnp.zeros_like(rho))
    T = fg.target_fields(os.path.join(os.path.dirname(os.path.abspath(__file__)), "gecko.png"), 40, j["grid"])
    frames, t = [paint(T)], 0
    step = jax.jit(lambda s: fg.step(params, s, j["phys"]))
    for m in marks:
        while t < m: s = step(s); t += 1
        frames.append(paint(s[0]))
        err = float(np.mean((np.asarray(s[0][:4]) - T)**2))
        print(f"step {m:4d}  mse {err:.5f}  mass {float(s[0][:4].sum()):.2f}  hidden {float(s[0][4:].sum()):.2f}")
    scale = 3
    W = j["grid"]*scale
    strip = Image.new("RGB", (W*len(frames), W))
    for i, f in enumerate(frames):
        strip.paste(Image.fromarray(f).resize((W, W), Image.NEAREST), (i*W, 0))
    strip.save(out); print("wrote", out)

if __name__ == "__main__":
    main()
