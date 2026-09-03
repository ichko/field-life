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

## Where it has got to, and what is left

The fit brings the three parts into the right arrangement — a gold head at one
end, the green back along the middle, the pale underside around them — and stops
there. It is not a gecko yet: no legs, no tail, no silhouette.

Two things stand between here and one, and neither is a mystery.

**It needs far more training than one sitting.** Fits of this kind normally run
for tens of thousands of steps; this has had a few hundred, at seven seconds
each on four CPU cores.

**The pattern is not a stable fixed point yet.** The curriculum lengthens the
unroll as it goes, and each time it does, the loss jumps: a rule tuned to look
right at step 32 does not hold at step 36, because it was never asked to stop.
The standard fix is a pool — keep a bag of states the rule has already produced,
restart from those as often as from the seed, and score them too, which is what
teaches a pattern to stay rather than merely to arrive. That is the next thing
to write.

Nothing has been added to `staging/volume.html` for this yet, deliberately. The
page would need the displaced-Gaussian bank, a gather pass, and a seed that is a
small pattern rather than a ball — a contained change, but not one worth making
to a page that works until there is a gecko to put in it.
