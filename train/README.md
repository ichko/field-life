# Growing a gecko

Fitting the volume page's rule so that a small seed grows into a 3D animal, the
way the flat page's lizard was fitted to an emoji.

| | |
|---|---|
| `gecko.py` | the target: a lizard drawn out of signed distances, and the same animal cut into three parts that do not overlap |
| `field3d.py` | the page's step written again in torch, so it can be differentiated |
| `train_gecko.py` | the fit: unroll from a seed, compare, push the error back |
| `export_fit.py` | writes the fitted rule out in the units the page reads |
| `render_fit.py`, `look.py`, `view.py` | look at what grew |
| `sheet.py` | contact sheets: `stages` compares every attempt against the target, `growth` follows one from its seed |

```
pip install torch numpy pillow
python3 train_gecko.py --iters 900 --N 40 --C 8 --S 5 --T 12 \
                       --steps 44 --warm 24 --lr 5e-3 --blur 0.9 --out geckoP.json
python3 look.py geckoP.pt 18,32,44 40 8 5 12 out.png
python3 export_fit.py geckoP.pt --N 40 --C 8 --S 5 --T 12 --out gecko-fit.json
```

## What the rule had to grow to allow this

The page's kernel is a difference of two Gaussians. That is radially symmetric,
and a bank of radially symmetric kernels has only radially symmetric fixed
points — it cannot hold anything shaped like an animal. So a kernel here is a
sum of Gaussians **moved off centre**:

    K_c = sum_t  a[c,t] * G_sigma(t)( rho_c )  read at  p + o[c,t]

which is still separable. The blurs are shared by every channel and every term,
and a displacement is free — it is only where you read the result. A handful of
displaced blobs of a few widths describes a thoroughly lopsided kernel, and the
bank costs three separable passes per width plus one gather of T taps. Nothing
else about the step changes, and mass is still conserved to the digit.

## Three things the fit taught, each of which cost a run

**Force is not a detail.** Below about ten the field simply diffuses, the animal
never forms, and the gradient through twenty steps dies with it. Above it the
field holds structure and the gradient survives the whole unroll. The affinity
is divided by the mean density here so force means the same thing whatever the
animal weighs — in three dimensions an animal is about a hundredth of the cube
where on a sheet it is a good fraction of it.

**The target has to be reachable.** The kernels are Gaussians a cell or two
wide, so the smallest thing the rule has a fixed point for is about that size. A
target with finer detail is not a hard target, it is an impossible one, and
asking for it buys a compromise that satisfies nothing. Blurred by about a cell
the gecko still reads as a gecko — legs, tail, head — and is inside what the
rule can do.

**Colours are species, and species separate.** Three channels carrying red,
green and blue at the same voxel is a picture the rule is built not to hold, and
a fit asked for it spends its whole budget losing that argument — visibly, as
rainbow fringes on a body that never becomes a body. So the animal is cut into
parts that do not overlap: back, underside, head. They sum to the whole, each
channel gets a region of its own, and the picture is put back together at the
far end by giving each part a colour, which is what every other world on the
page already does.

## Where it has got to

The fit brings the three parts into the right arrangement — a gold head at one
end, the green back along the middle, the pale underside around them. It is not
a gecko yet: no legs, no tail, no silhouette.

Two things stood between here and one. The second is now fixed.

**It needs far more training than one sitting.** Fits of this kind normally run
for tens of thousands of steps; this has had a couple of thousand, at three to
seven seconds each on four CPU cores. That is the whole of the remaining gap.

**The pattern was not a stable fixed point.** The first runs lengthened the
unroll as they went and the loss jumped every time they did: a rule tuned to
look right at step 32 did not hold at step 36, because it had never been asked
to stop. Run it past where it was trained and the animal kept going, into
something else.

`--pool` is the fix, and it works. Keep a bag of states the rule has already
made, start from one of those as often as from the seed, and score where it
gets to fourteen steps later. A state that is already the animal is then scored
on whether it is still the animal afterwards, which is the only way a fixed
point gets learned. The measure that matters is the loss on those pool
restarts, and over the run it falls steadily:

    pool restarts   1- 15   mean 0.101
                   16- 30   mean 0.098
                   31- 45   mean 0.092
                   46- 60   mean 0.078
                   61- 75   mean 0.057

and the picture stops moving: run from the seed, the shape at step 20, step 40
and step 70 is the same shape. Before the pool it was three different ones.

The cost is that what it holds is *simpler* than what it reached before —
persistence pressure likes smooth attractors, and it found one before it found
a gecko.

And here is the part worth knowing before spending a night on this: **the extra
training did not buy the shape back.** A thousand more pool iterations took the
restart loss from 0.078 to 0.057, and the picture at the end of them is the
picture at the start — the same slab, held more precisely. The improvement went
into polishing the attractor it had already found, not into leaving it. So
"just run it longer" is not, on this evidence, the whole answer.

What that points at is the balance between the two pressures rather than the
amount of either. Once the pool is filling, nearly every iteration is a short
restart from a state that is already smooth, and the long run from the seed --
the only one that ever has to *build* anything -- is a fifth of the batch and
gets a fifth of the gradient. Things to try, in order: hold the fresh fraction
high and decay it slowly rather than fixing it at a fifth; weight the fresh
iterations up in the loss; and keep the silhouette term climbing while the pool
term holds, so persistence is bought without paying for it in shape.

```
python3 train_gecko.py --iters 2400 --N 40 --C 8 --S 5 --T 12 \
    --pool 16 --warm 30 --chunk 14 --lr 3e-3 --wsil 2.0 --blur 0.9 \
    --resume geckoP_best.pt --out geckoQ.json
```

## Running it on a GPU

Everything here is device-agnostic now — the model asks its own parameters where
they live rather than remembering a string, so `.to("cuda")` works:

```
python3 train_gecko.py --device cuda --iters 20000 --N 48 --C 8 \
    --kernel cppn --K 7 --steps 44 --warm 30 --lr 3e-3 --wsil 2.0 --blur 0.9 \
    --resume geckoK2.pt --out geckoG.json
python3 look.py geckoG.pt 20,34,48 48 8 3 8 out.png
```

`geckoK2.pt` and `geckoP_best.pt` are committed so a run can pick up where the
CPU left off rather than start over: the first is the CPPN bank at 170
iterations, the second the displaced-Gaussian fit it is being measured against.

What the hardware is worth here is not subtle. A step forward and back is about
330 ms on four CPU cores at 40 cubed with eight channels; the same thing is a
few tens of milliseconds on a current card, and 48 or 64 cubed becomes
affordable at the same time. This whole page's worth of conclusions was drawn
from runs of a few hundred iterations. Fits of this kind normally get tens of
thousands, and the one thing every result here points at is that the fit has not
had enough of them.

The next thing to add once there is a GPU to test it on is a **batch**. The pool
restarts are batch-one, which wastes most of a card; running sixteen pool states
at once is nearly free there and multiplies the gradient signal per second again.
It is not written yet because it cannot be tested here.

## Not on the page yet

Nothing has been added to `staging/volume.html` for this, deliberately. The page
would need the displaced-Gaussian bank, a gather pass, and a seed that is a
small pattern rather than a ball — a contained change, but not one worth making
to a page that works until there is a gecko to put in it.

## The brains rule, and why it will not learn a gecko by gradient

`gecko2d.py`, `emoji_gecko.py`, `field_brains.py`, `train_gecko_brains.py` and
`train_emoji_gecko.py` are the same exercise aimed at `staging/brains.html`: the
lizard emoji as a target, the rule written again in torch, a fit that unrolls
from a seed and pushes the error back. It does not work, and the reason is worth
more than the attempt was.

**The gradient through this rule explodes, at every setting it has.** Gradient
norm against how many steps carry one, after a fixed warm-up:

    beta speed drag |     t=1       t=8      t=16      t=32
     7.3   2.8  0.50   2.4e-4   9.9e+07       inf       nan
     2.0   1.5  0.64   1.9e-4   6.0e+07   1.1e+19       nan
     1.0   0.8  0.80   8.7e-5   3.7e+04   6.8e+15       nan
     0.5   0.4  0.90   2.7e-5   1.5e+04   7.0e+13       nan
     0.2   0.2  0.95   8.2e-6   1.4e+03   9.7e+11       nan

Beta turned down thirty-sixfold, transport nearly off and damping at 0.95, and
it still passes float32 before twenty steps. This is not a regime that was
missed, it is the shape of the rule: MaCE divides by a normaliser that is itself
a function of the state, and that division sits inside every step.

**And a one-step gradient cannot do morphogenesis.** Growing a lizard from a
disc means putting a tail somewhere because a head is somewhere else, which is
credit assigned across many steps. Growing NCA backprops through sixty-four to
ninety-six of them, and its update is a small residual on purpose, which is
exactly what keeps that possible. One step can only learn a local move.

Measured, both halves hold. A gradient through four steps does not descend at
all -- 0.0250 to 0.0256 over 320 iterations -- and the only length that does
descend is one, because it is the only one whose gradient stays under the
clipping threshold. Fitted at one step the loss falls a long way and the picture
is a disc: silhouette overlap 0.64 against the emoji, which is about what a
green disc of the right area scores.

Two blind alleys, both of which looked like the answer:

*The seed being round.* It was not, but the seed WAS a problem: it drew a fresh
random velocity field every iteration, so every run broke the rotational
symmetry differently and the fit averaged over all of them. The average of a
lizard over every angle is a disc. Fixed to one coherent heading per species,
identical every time -- and it still makes a disc, for the reason above.

*Numerical faults.* Three real ones, none of them the cause. `torch.where`
differentiates the branch it did not take, so a division guarded by one has an
infinite gradient -- this was in `cap`, and a stalled species trips it
constantly. The reverse transport weight cannot be taken as `1/g` when `g`
bottoms out at `exp(-30)`. The affinity needs a global shift before its
exponential. With all three fixed the gradient is finite, and still explodes.

What would work, in order of how much of the rule it changes: a gradient-free
search over the 1680 weights, which does not care that the rule is chaotic and
does care that there are 1680 of them; or a residual, damped update in place of
the conservative transport, which is to say a different rule.

The emoji target and its four-colour cut are kept and are reusable: colours in
field life are species and species separate, so the glyph is clustered by colour
and each species is given one cluster to own, with the cluster colours as the
palette.
