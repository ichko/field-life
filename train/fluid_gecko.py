"""Train the fluid page's rule to grow a gecko from a disc, and keep it.

The simulation is the one in staging/fluid.html, step for step: densities
and velocities for C fluids, an equation of state on the total, a velocity
pushed by the pressure gradient, and the MaCE move -- nine shares per fluid
per cell, a Gaussian sat on the velocity leaned towards lower pressure,
normalised at the source. What is learned is what the matrix used to do: a
small network reads each cell (density, its Sobel gradients and Laplacian,
velocity) and answers with a potential and a push per fluid. Mass is
conserved by construction, so the seed disc must hold exactly what the
picture costs, fluid by fluid; four hidden fluids ride along with masses the
training picks.

    python3 train/fluid_gecko.py            # trains, writes staging/fluid-gecko.json
    python3 train/fluid_gecko.py --iters 200 --out /tmp/x.json
"""
import argparse, functools, json, os, time
import numpy as np
# One CPU device per batch element: XLA's elementwise work is single-threaded
# per device, and nearly all of a step is elementwise.
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")
import jax, jax.numpy as jnp
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# The page's four colours, and what a gecko is made of in them.
COL = np.array([[1.00, 0.36, 0.22], [0.42, 0.90, 0.46], [0.30, 0.58, 1.00], [1.00, 0.84, 0.30]], np.float32)

PHYS = dict(dt=0.15, drag=0.05, visc=0.10, couple=0.05, spread=0.35, seep=0.5,
            stiff=0.5, gamma=2.0, vmax=0.9)
NV, NHID = 4, 4            # visible fluids, hidden fluids
C = NV + NHID
NIN, NH, NOUT = 6*C, 64, 3*C   # features: rho, sobel x, sobel y, laplacian, vx, vy -> U, Fx, Fy

# ------------------------------------------------------------------ target
def target_fields(png, size, grid):
    """Gecko PNG -> (NV, grid, grid) densities the page's paint view shows as
    the picture. Per pixel, non-negative least squares of premultiplied RGB
    against the four colours, by projected gradient (it is tiny)."""
    im = Image.open(png).convert("RGBA").resize((size, size), Image.LANCZOS)
    a = np.asarray(im, np.float32)/255.0
    rgb = a[..., :3]*a[..., 3:4]                      # on black
    rgb = rgb[::-1]                                   # row 0 at the bottom, like the GPU
    P = rgb.reshape(-1, 3)
    A = COL                                            # (4,3): rgb = rho @ A
    rho = np.zeros((P.shape[0], NV), np.float32)
    L = np.linalg.norm(A @ A.T, 2)
    for _ in range(600):
        g = (rho @ A - P) @ A.T + 1e-3*rho
        rho = np.maximum(rho - g/L, 0.0)
    T = np.zeros((NV, grid, grid), np.float32)
    o = (grid - size)//2
    T[:, o:o + size, o:o + size] = rho.reshape(size, size, NV).transpose(2, 0, 1)
    return T

# ------------------------------------------------------------------ physics
OFF = [(0, 0), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]

def shift(x, ox, oy):
    """x at (y+oy, x+ox) brought to (y, x): a read from the neighbour."""
    return jnp.roll(x, (-oy, -ox), axis=(-2, -1))

def conv3(x, k):
    """3x3 kernel k[dy+1][dx+1] on the torus, all channels alike."""
    out = 0.0
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            w = k[dy + 1][dx + 1]
            if w: out = out + w*shift(x, dx, dy)
    return out

SOBX = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
SOBY = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
LAP = [[1, 2, 1], [2, -12, 2], [1, 2, 1]]
BIN = [[1, 2, 1], [2, 4, 2], [1, 2, 1]]

def blur5(x):
    """The page's separable 5-tap binomial, run once."""
    k = jnp.array([1, 4, 6, 4, 1], jnp.float32)/16
    x = sum(k[i + 2]*jnp.roll(x, -i, axis=-1) for i in range(-2, 3))
    x = sum(k[i + 2]*jnp.roll(x, -i, axis=-2) for i in range(-2, 3))
    return x

def net(params, rho, vx, vy):
    f = jnp.concatenate([rho, conv3(rho, SOBX)/8, conv3(rho, SOBY)/8, conv3(rho, LAP)/16, vx, vy], 0)
    h = jax.nn.relu(jnp.einsum("ij,jhw->ihw", params["w1"], f) + params["b1"][:, None, None])
    o = jnp.einsum("ij,jhw->ihw", params["w2"], h) + params["b2"][:, None, None]
    return o[:C], o[C:2*C], o[2*C:]

def step(params, state, ph=PHYS):
    rho, vx, vy = state
    r1 = blur5(rho)
    tot1 = jnp.maximum(r1.sum(0), 0.0)
    press = ph["stiff"]*tot1**ph["gamma"]
    Un, Fx, Fy = net(params, rho, vx, vy)
    U = press[None] + Un
    gx = 0.5*(shift(U, 1, 0) - shift(U, -1, 0))
    gy = 0.5*(shift(U, 0, 1) - shift(U, 0, -1))
    # The coupling and the viscosity average the velocities the step STARTED
    # with, in the cell and around it: that is all a single pass on the GPU
    # can see, and the page is a single pass.
    vx0, vy0 = vx, vy
    vx = vx + ph["dt"]*(Fx - gx)
    vy = vy + ph["dt"]*(Fy - gy)
    tot = rho.sum(0)
    if ph["couple"] > 0:
        mx = (rho*vx0).sum(0)/(tot + 1e-9); my = (rho*vy0).sum(0)/(tot + 1e-9)
        vx = vx + ph["couple"]*(mx[None] - vx); vy = vy + ph["couple"]*(my[None] - vy)
    if ph["visc"] > 0:
        sw = conv3(rho, BIN); has = (sw > 1e-9).astype(jnp.float32); sw = jnp.maximum(sw, 1e-9)
        vx = vx + ph["visc"]*has*(conv3(rho*vx0, BIN)/sw - vx)
        vy = vy + ph["visc"]*has*(conv3(rho*vy0, BIN)/sw - vy)
    vx = vx*(1 - ph["drag"]); vy = vy*(1 - ph["drag"])
    # the epsilon is inside the root: sqrt has no gradient at zero, and every
    # velocity starts there
    sp = jnp.sqrt(vx*vx + vy*vy + 1e-12)
    sc = jnp.minimum(1.0, ph["vmax"]/sp)
    vx = vx*sc; vy = vy*sc
    # the move: nine affinities a fluid a cell, shares by softmax at the source
    inv = 1.0/(ph["spread"]**2)
    logits = jnp.stack([(ox*vx + oy*vy - 0.5*(ox*ox + oy*oy))*inv - ph["seep"]*(shift(U, ox, oy) - U)
                        for ox, oy in OFF], 0)
    logits = jnp.clip(logits, -60, 60)
    w = jax.nn.softmax(logits, axis=0)
    rho2 = 0.0; mx2 = 0.0; my2 = 0.0
    for k, (ox, oy) in enumerate(OFF):
        got = w[k]*rho
        rho2 = rho2 + shift(got, -ox, -oy)          # what leaves p for p+o arrives at p+o
        mx2 = mx2 + shift(got*vx, -ox, -oy)
        my2 = my2 + shift(got*vy, -ox, -oy)
    # eps well above float32 underflow: its square appears in the backward pass
    inv2 = 1.0/(rho2 + 1e-12)
    return (rho2, mx2*inv2, my2*inv2)

def rollout(params, state, n):
    def body(s, _):
        s = jax.checkpoint(step)(params, s)
        return s, None
    s, _ = jax.lax.scan(body, state, None, length=n)
    return s

# ------------------------------------------------------------------ seed
def disc(grid, r):
    y, x = np.mgrid[0:grid, 0:grid].astype(np.float32) + 0.5 - grid/2
    d = np.sqrt(x*x + y*y)
    m = np.clip((r + 0.5 - d), 0, 1)                  # a soft edge one cell wide
    return m/m.sum()

def seed_state(params, disc_w, masses):
    """The disc, holding exactly the target's mass per visible fluid, and the
    learned amount of each hidden one."""
    hid = jax.nn.softplus(params["hidden"])
    m = jnp.concatenate([masses, hid])
    rho = m[:, None, None]*disc_w[None]
    return (rho, jnp.zeros_like(rho), jnp.zeros_like(rho))

# ------------------------------------------------------------------ train
def init_params(key):
    k1, k2 = jax.random.split(key)
    return {"w1": jax.random.normal(k1, (NH, NIN))*np.sqrt(2/NIN),
            "b1": jnp.zeros(NH),
            "w2": jnp.zeros((NOUT, NH)),               # starts as the plain fluid
            "b2": jnp.zeros(NOUT),
            "hidden": jnp.full((NHID,), 0.5)}

def export(params, T, grid, r, path):
    masses = T.reshape(NV, -1).sum(1)
    out = {
        "what": "the fluid page's rule, trained to grow a gecko from a disc",
        "C": C, "visible": NV, "hidden": NHID, "nin": NIN, "nh": NH, "nout": NOUT,
        "grid": grid, "disc": r, "phys": PHYS,
        "masses": [float(v) for v in masses] + [float(v) for v in jax.nn.softplus(params["hidden"])],
        "features": "rho, sobel_x/8, sobel_y/8, laplacian/16, vx, vy per fluid; relu; U, Fx, Fy per fluid",
        "w1": np.asarray(params["w1"]).round(6).tolist(), "b1": np.asarray(params["b1"]).round(6).tolist(),
        "w2": np.asarray(params["w2"]).round(6).tolist(), "b2": np.asarray(params["b2"]).round(6).tolist(),
    }
    json.dump(out, open(path, "w"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--png", default=os.path.join(HERE, "gecko.png"))
    ap.add_argument("--out", default=os.path.join(ROOT, "staging", "fluid-gecko.json"))
    ap.add_argument("--ckpt", default=os.path.join(HERE, "fluid-gecko.npz"))
    ap.add_argument("--grid", type=int, default=64)
    ap.add_argument("--size", type=int, default=40)
    ap.add_argument("--disc", type=float, default=9.0)
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--pool", type=int, default=256)
    ap.add_argument("--steps", type=int, nargs=2, default=[64, 96])
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    T = target_fields(args.png, args.size, args.grid)
    masses = jnp.asarray(T.reshape(NV, -1).sum(1))
    disc_w = jnp.asarray(disc(args.grid, args.disc))
    print("target mass per fluid", np.asarray(masses).round(1), "disc cells", float((disc_w > 0).sum()))

    key = jax.random.PRNGKey(0)
    params = init_params(key)
    if args.resume and os.path.exists(args.ckpt):
        z = np.load(args.ckpt); params = {k: jnp.asarray(z[k]) for k in params}
        print("resumed", args.ckpt)

    seed0 = seed_state(params, disc_w, masses)
    # the pool lives on the host; pmap shards a batch across the devices itself
    pool = jax.tree_util.tree_map(lambda x: np.repeat(np.asarray(x)[None], args.pool, 0), seed0)

    Tj = jnp.asarray(T)
    ndev = jax.local_device_count()
    assert args.batch == ndev, f"batch must equal the device count ({ndev})"
    def loss_one(params, state, reseed, n):
        # the seed is rebuilt here, inside the gradient, so the hidden masses
        # it holds are trained along with everything else
        seed_now = seed_state(params, disc_w, masses)
        state = jax.tree_util.tree_map(lambda x, s: jnp.where(reseed, s, x), state, seed_now)
        out = rollout(params, state, n)
        l = jnp.mean((out[0][:NV] - Tj)**2)
        return l, out

    @functools.partial(jax.pmap, axis_name="b", in_axes=(None, 0, 0, None), static_broadcasted_argnums=3)
    def grads(params, state, reseed, n):
        (l, out), g = jax.value_and_grad(loss_one, has_aux=True)(params, state, reseed, n)
        return jax.lax.pmean(g, "b"), l, out

    import optax
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(args.lr))
    opt_state = opt.init(params)

    @jax.jit
    def apply(params, opt_state, g):
        # NCA's normalised gradient: the scale of the loss is not the scale of the step
        g = jax.tree_util.tree_map(lambda x: x/(jnp.linalg.norm(x) + 1e-8), g)
        upd, opt_state = opt.update(g, opt_state, params)
        return optax.apply_updates(params, upd), opt_state

    def train_step(params, opt_state, states, worst, n):
        reseed = np.arange(args.batch) == worst
        g, per, out = grads(params, states, reseed, n)
        g = jax.tree_util.tree_map(lambda x: x[0], g)          # the same on every device
        params, opt_state = apply(params, opt_state, g)
        return params, opt_state, per.mean(), out, per

    rng = np.random.default_rng(0)
    t0 = time.time(); best = 1e9
    for it in range(args.iters):
        idx = rng.choice(args.pool, args.batch, replace=False)
        states = jax.tree_util.tree_map(lambda x: x[idx], pool)
        # the worst of the batch starts over from the seed, so the seed is
        # never forgotten and the rest learn to hold what they have
        pre = np.mean((states[0][:, :NV] - T[None])**2, axis=(1, 2, 3))
        worst = int(np.argmax(pre))
        # a few lengths only, so the jit compiles a few times, not once per length
        n = int(rng.choice(np.linspace(args.steps[0], args.steps[1], 3).round().astype(int)))
        params, opt_state, loss, out, per = train_step(params, opt_state, states, worst, n)
        for p_, o in zip(pool, out): p_[idx] = np.asarray(o)
        loss = float(loss)
        if it % 20 == 0 or it < 4 or it == args.iters - 1:
            print(f"{it:5d}  loss {loss:.5f}  steps {n}  hidden {np.asarray(jax.nn.softplus(params['hidden'])).round(2)}  "
                  f"{(time.time() - t0)/(it + 1):.2f}s/it", flush=True)
        if it % 100 == 99 or it == args.iters - 1:
            np.savez(args.ckpt, **{k: np.asarray(v) for k, v in params.items()})
            if loss < best:
                best = loss; export(params, T, args.grid, args.disc, args.out)
                print("   exported", args.out, "loss", round(loss, 5), flush=True)

if __name__ == "__main__":
    main()
