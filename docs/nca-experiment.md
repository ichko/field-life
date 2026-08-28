# Sculpting the lizard: field-life as a trainable NCA

**Status:** design + parity harness landed. No training yet.

The question is whether field-life can be trained rather than sampled: read the
first three channels as RGB, leave the rest as hidden state, and ask the field
to arrange itself into the lizard emoji that Growing Neural Cellular Automata
builds. What follows is what the simulation actually does, what has to change,
what is trainable, and the two measurements that already constrain the design.

---

## 1. The law we are training

One tick, per channel `c`, on a torus:

```
U_c  = rho_c  *  K_c                        kernel: zero-mean, unit-L1
N_c  = rho_c  *  G / sum(G)                 G: gaussian, sigma = 0.22 * KR
A_c  = force * sum_d M[c,d] U_d  -  repel * sum_d N_d
E    = exp(clamp(beta * A, -20, 20))
Z    = 3x3 sum of E
rho' = E * (3x3 sum of rho/Z)
```

The last three lines are MaCE, and they are the reason this is an unusual thing
to train. **Mass is conserved exactly, per channel**: every cell hands out all
of what it holds, split across its 3×3 neighbourhood in proportion to `E`, and
`sum(rho') = sum(rho)` falls out because a cell lies inside its own
neighbourhood. Two consequences run through every decision below:

- **Nothing can be created.** Growing NCA grows a lizard from one cell because
  its update rule manufactures matter. This one cannot.
- **Mass moves at most one cell per step.** That sets a hard floor on how many
  steps any target needs, measured in §4.

Note also that the affinity is **linear** in the convolved fields — one matrix
multiply per cell, no hidden layer. That, not the kernels, is the real capacity
ceiling, and §6 is about where to break it.

At the training grid this is simpler than the shipped code looks. `allocate()`
only builds a mip level while `min(px,py)>>1 >= 48`, so at N ≤ 96 the pyramid
is empty, `kernelMip` is pinned at 0, the convolution runs at full resolution,
and the Catmull-Rom upsample after it is the identity. And any kernel with
three or more lobes makes `separablePlan()` return `null`, so the browser takes
the 2-D stencil path. **Whatever we train runs in the browser through exactly
the arithmetic we differentiate.**

## 2. Reading three channels as RGB

Density is non-negative and the background must be empty, so alpha cannot be
its own channel — it has to *be* the mass. The target is therefore
**premultiplied**: `target = rgb * alpha`. The transparent surround then costs
nothing, and an L2 loss on channels 0–2 alone pins down the silhouette as well
as the colour. `train/target.py` renders it from `NotoColorEmoji`: 40 cells of
lizard centred in a 64×64 field, background max exactly `0.0`.

Channels 3+ are unconstrained by the picture. They are the working memory — the
only place the system can carry a signal that is not supposed to be visible.

## 3. The mass contract

The seed cannot be one lit cell, and it cannot be neutral. It has to arrive
already holding what the picture will cost:

| channel | role | mass |
|---|---|---|
| 0 | R | 74.6 |
| 1 | G | 317.3 |
| 2 | B | 126.6 |
| 3–5 | hidden | ≥ 0.35 × mean visible (≈ 60.5), jittered per run |

The visible three are the target's own channel integrals, so **the loss is
purely about arrangement** — the totals are correct by construction and there is
no mass term to balance. The hidden three are free working budget; they are
jittered so no result depends on one lucky allocation.

This does hand the model the target's colour histogram, and that should be said
plainly in any write-up: the task is *sculpting*, not *growing*. What is not
handed over is every spatial fact — where the tail goes, that the eyes are dark,
that the dorsal stripe is orange — and that is the whole of the difficulty.

The seed is a soft disc, not a cell, for the reason in §4.

## 4. The measurement that constrains everything

`train/parity.py` steps the real WebGL simulation in headless Chromium and the
PyTorch port side by side. Three results, all reproducible with
`node train/dump_reference.mjs && python3 train/parity.py`:

**The port is exact.** The baked kernel bank agrees to `3.2e-9`. In a
non-amplifying regime (force 12, beta 0.35) the fields stay together to
`4.9e-7` relative over 32 steps.

**The shipped regime is chaotic.** At force 90, beta 3 the port-vs-browser gap
grows at `e^0.40` per step — but the port's *own* float32-vs-float64 gap grows
at `e^0.44` per step. The port is as close to the browser as the browser is to
itself at a different float width; the divergence is the simulation, not the
port. MaCE's softmax is winner-take-all there, and it amplifies the `1e-8` of
merely writing the seed into a float32 texture.

**Mass survives float32.** Worst relative drift over 32 browser steps: `9.2e-7`.

The chaos number is the design constraint. Backpropagating through `T` steps
amplifies gradients by roughly `e^0.40T`. Meanwhile the furthest lit cell of the
lizard sits **26 cells** from a small central seed, and mass moves one cell per
step, so `T >= 26` is a floor. At `T = 26` that is `e^10`; at `T = 64`, `e^26`.

So the central tension is: **transport needs many steps, and many steps destroy
the gradient.** Three levers, in order of preference:

1. **Truncated BPTT over a sample pool**, exactly Growing NCA's persistence
   trick. Backpropagate through a window of `W = 16` steps (`e^6.4`, tractable)
   starting from *pooled* states, and chain windows through the pool to reach
   arbitrarily long rollouts. This caps the horizon independently of `T`.
2. **Seed wider.** A disc of radius ~6 leaves 26 cells to cross; a disc that
   already spans the animal's extent leaves ~6. The seed is "a blob of the right
   stuff", not the answer, so this is legitimate — but it is a dial to report,
   not to hide.
3. **Anneal `beta`.** Train cool, where the step does not amplify, and raise the
   exponent as the shape settles. `force` and `beta` are themselves trainable,
   so the optimiser can wander into chaos on its own — the run should measure
   the divergence rate online and penalise it, not just hope.

## 5. The kernels — the actual question

They are *already* polar functions. Today a kernel is
`{R, feather, terms: [{a, r, w}]}` baked as

```
K(rr) = sum_i  a_i * exp(-((rr - r_i)/w_i)^2)          rr = |x| / R
```

then feathered at the rim, zero-meaned against the taper and normalised to unit
L1. So "generate the kernels from a polar function and train its parameters" is
not a new mechanism — it is the mechanism, with `a_i, r_i, w_i, R` promoted from
sampled to learned. Two changes make that work.

**Add the angle.** Give each lobe a fixed integer angular order and a learned
phase:

```
K(r, theta) = sum_l  a_l * exp(-((r - mu_l)/w_l)^2) * cos(m_l * theta + phi_l)
```

`m` cannot be gradient-descended — it indexes a Fourier basis — so the orders
are fixed per lobe and the amplitude decides whether a lobe is used. This is
worth doing because of what the orders *are*:

- `m = 0` is a radial ring. That is Lenia, and it is everything the sim can
  express today.
- `m = 1` is a signed gradient along an axis. **That is a Sobel filter** — so an
  NCA's fixed identity / Sobel-x / Sobel-y perception is exactly the special
  case `m ∈ {0, 1, 1}`, and the two lineages meet here.
- `m ≥ 1` is what lets a mass-conserving field break rotational symmetry. An
  animal is not rotationally symmetric, and a bank of `m = 0` kernels has no
  way to prefer "tail that way".

A sensible starting bank is `orders = (0, 0, 0, 1, 1, 2)` per channel: three
Lenia rings, an oriented pair, and one quadrupole.

**Make `R` differentiable.** Today `uploadKernels` gives a channel a *grid
resolution* proportional to its reach — `kr = max(2, round(KR * R_c / Rmax))` —
and bakes it there. That `round()` is a step function, so `R` cannot be trained
through it, and small-`R` channels also lose resolution for no reason.

Instead, bake every channel on the one shared grid of half-width `KR` and let
`R_c` enter as a continuous radial scale, `rr = |x| / R_c`. Then `dK/dR` exists,
and — the nice part — **the feather is what makes that gradient well behaved**,
because the profile reaches zero smoothly at `rr = 1` instead of being cut
there. The existing taper was written to stop hard circles from printing square
artefacts; it turns out to be the same thing that makes reach learnable.

Both are already implemented on the torch side as `PolarKernels`
(`train/fieldlife.py`), which emits `index.html`-shaped kernel dicts.

**Dropped for training:** the `sym` fold (angular orders subsume it, and the
choice is discrete) and `perlin` kernels (a seeded integer hash, nothing trains
one). `pl` and `disc` stay ported for parity only.

## 6. How much capacity — measured

This section predicted that the linear affinity was the capacity ceiling and
that rung 1 was where a lizard would appear. Both halves are wrong, and the
run that settles it is `train/runs/rung1b`.

**Rung 0 — the law as shipped.** `M`, the per-channel polar kernels, and
`force`, `repel`, `beta`. 543 numbers at C=12. No new shaders; loads into the
browser. This is what is deployed.

**Rung 1 — a hidden layer where the matrix was.** `A = W2·tanh(W1·U + b1)`,
C→48→C, applied per cell. 1743 numbers. Structurally an NCA's update MLP, and
one new shader in `FS_AFF`.

Distance from the target, from a fresh seed, at the best checkpoint each run
reached:

            h32     h64    h128    h256    h512
  rung 1   0.0026  0.0034  0.0035  0.0038  0.0058    1743 numbers
  rung 0   0.0019  0.0022  0.0025  0.0028  0.0039     543 numbers

**The matrix wins on every horizon, by 1.44x overall, with a third of the
parameters.** Rung 1 went 1600 iterations without improving on 0.00460 against
rung 0's 0.00300.

(h16 is left out of that ratio: rung 1 trained with `--loss-from 28` so step 16
was never scored for it, while rung 0 used 16. Including it gives 1.51x, which
flatters the conclusion.)

The caveat: rung 0 had 13800 iterations and rung 1 had 6800. This is a plateau,
not a proven convergence. It is worth reporting because rung 0 at 6800 was
still improving substantially and rung 1 was not.

**Why, plausibly.** The two runs fail differently. Rung 1 learned to BUILD
quickly — h32 fell fourfold inside 1400 iterations, reaching rung 0's converged
value — and to HOLD slowly, h512 taking another 4000 to come down at all.
Holding a pattern for five hundred steps is a stability problem more than an
expressiveness one, and a linear map has bounded, predictable long-run
behaviour where a nonlinearity does not. The network buys the ability to say
more per cell and pays for it in what happens after three hundred steps of
saying it.

So the capacity that mattered was never in the per-cell function. It was in the
kernels: §5's angular orders are the difference between a lizard and a ring,
and no amount of per-cell machinery substitutes for them.

**Rung 2 — a full per-cell MLP** over `[rho, U, N]`. Not run. On this evidence
it would make the stability problem worse, not better.

## 7. What `index.html` needs — measured, not argued

An earlier draft of this section said the minimum diff was zero, and that a
trained bank could be dropped into the shipped simulation as a preset. That is
true only of kernels the shipped `bakeKernel` can bake, and it does not hold
for anything this experiment actually learns.

`train/verify_browser.mjs` loads a preset into index.html in headless Chromium
and runs it. Doing that with polar3's bank, against the same bank baked as the
trainer meant it:

    kernel bank, max abs difference     0.034   on unit-L1 kernels
    per-channel cosine similarity       +0.58 +0.55 +0.00 +0.12 +0.07 +0.41
                                        +0.12 +0.25 -0.11 +0.23 -0.11 +0.70
    loss after 64 steps, trained        0.0052
    loss after 64 steps, browser today  0.0973      18.7x worse

Two channels come out orthogonal to what was trained and two come out
anticorrelated. The picture is a lizard through the trained kernels and a set
of concentric rings through the shipped ones.

And it fails **silently**. `applyConfig` accepts the preset, `validConfig`
passes, the simulation runs. An unknown field on a term is simply not read, so
every angular lobe is baked as a plain radial one and the preset loads as a
different, rotationally symmetric world. Nothing anywhere reports a problem.

So change 1 below is not optional for any result worth having, and the
zero-diff option and the working option are the same choice, not two:

1. **`bakeKernel`** — honour `m` and `phase` on a term:
   `v += t.a * exp(-((rr - t.r)/t.w)^2) * cos(t.m * theta + t.phase)`, with
   `theta = atan2(py, px)`. `fl.bake_from_config` in `train/fieldlife.py` is
   the reference implementation. Absent fields default to `m = 0, phase = 0`,
   so every existing preset stays byte-identical. **Required.**
2. **`separablePlan()`** — bail out when any term has `m !== 0`. Required
   alongside 1, and for the same reason: without it an angular kernel takes the
   separable path and is run as a radial difference-of-gaussians. (polar3 has
   enough lobes to miss that path anyway, but nothing guarantees that in
   general.)
3. **`uploadKernels` / `bakeKernel`** — bake at the shared `kernelKR` with
   `R_c` as a radial scale, dropping the `round()` resampling. Buys a trainable
   reach, and removes resolution loss on short-reach channels. Worth doing
   independently.
4. **`FS_DRAW`** — a "Raw RGB" blend mapping channels 0-2 straight to R, G, B
   with no palette, tone-map or lifted background. Convenience only.

If a genuinely zero-diff result is wanted, it has to be an all-`m = 0` bank --
and §6's ablation is that those cannot do the task at all. Choose one.

Rung 1 needs its own change on top: `FS_AFF` computes a matrix multiply, and a
per-cell network is not one.

## 8. Protocol

- Grid 64×64, C = 6, seed as §3, target as §2.
- Loss: L2 on channels 0–2 against the premultiplied target, evaluated at a
  rollout length sampled from a range, so the pattern has to *persist* rather
  than pass through the right arrangement once.
- Sample pool as in Growing NCA, with periodic re-seeding; damage recovery as a
  later probe, not a first goal.
- Adam, per-parameter gradient normalisation, `W = 16` step BPTT windows.
- Report per rung: final L2, whether the pattern persists to 4× the training
  horizon, the measured divergence rate, and the learned kernel bank as an
  image.

**What counts as a result.** A lizard is the headline and may not arrive. These
are worth reporting whether or not it does:

- the rung at which a recognisable silhouette appears (capacity ablation);
- whether trained kernels use `m > 0`, and how much worse an `m = 0`-only bank
  is — that is the Lenia-vs-NCA question stated quantitatively;
- whether a mass-conserving CA can hold *any* prescribed asymmetric pattern
  stably, which as far as I know is not established either way.

## 9. What exists

```
train/fieldlife.py        the step and both kernel bakers, in torch
train/dump_reference.mjs  steps the real WebGL sim in headless Chromium
train/parity.py           port vs browser; the numbers in §4
train/target.py           the lizard, the seed, and the mass budget
```

Not yet written: the trainer, the sample pool, and the JSON export back into a
preset.

Run the harness with:

```
npm install --no-save playwright
pip install torch numpy pillow
node train/dump_reference.mjs && python3 train/parity.py
python3 train/target.py --show
```
