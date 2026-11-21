# Field Life: paper demo and code

Code, data and interactive demo for the paper *Field Life: Particle Life as Interacting Chemical Fields* (IEEE SSCI 2027). Live demo: https://www.izhv.eu/field-life/paper-demo/. Later work on Field Life is in the [main demo](https://www.izhv.eu/field-life/) and the rest of this repository.

Run the commands below from this folder.

Field Life is a cellular automaton with the interaction dynamics of Particle Life. Each colour is a
channel of the automaton, and each cell holds the concentration of that colour. In one step, each
colour is convolved with a ring-shaped kernel, the interaction matrix turns the results into an
affinity for each colour, and the mass of each colour moves towards higher affinity with the
mass-conserving MaCE update.

## Contents

| path | what it is |
|---|---|
| `fieldlife/model.py` | the model: rule, kernels, initial states, one step (NumPy, one file) |
| `fieldlife/render.py` | concentrations to an image, and a command that renders rollouts |
| `fieldlife/explore.py` | reproducible random exploration: sample rules, measure, keep, tag |
| `fieldlife/symmetry.py` | the symmetrisation test of the seven gliders |
| `index.html` | interactive simulation (WebGL2) with presets, to sample and watch rules |
| `data/worlds.json` | the 120 worlds W1 to W120 found by the exploration, with their tags |
| `data/gliders.json` | the seven gliders B0 to B6, each with its rule and its starting body |

Requirements: Python 3.9+, NumPy, Pillow (`pip install -r requirements.txt`). The page needs a
browser with WebGL2.

## Quick start

Render a world from a disc of noisy mass at steps 0, 100, 300, 1000 and 2000 (Fig. 4 of the paper):

    python -m fieldlife.render data/worlds.json W1 --steps 0,100,300,1000,2000 --out w1.png

Render a glider alone in an empty world (Fig. 5):

    python -m fieldlife.render data/gliders.json B3 --steps 0,60,120,180,240,300 --out b3.png

Run the exploration: draw 24 random rules with seed 0, run each for 300 steps, keep the ones that
form structure and move it, tag them, and write them in the layout of `data/worlds.json`:

    python -m fieldlife.explore --n 24 --seed 0 --out results

The same seed gives the same worlds. One sample takes about a minute at 512 x 512 cells.

Watch rules in the browser: open `index.html`. Served over http, the page loads all 120 worlds and
7 gliders as presets:

    python -m http.server 8000      # then open http://localhost:8000/index.html

Opened as a file, it has the 12 worlds and 3 gliders shown in the paper built in. A link such as `index.html#W118` or `index.html#B3` opens that preset; otherwise the page opens
on a preset chosen at random, and *Random preset* picks another. *Sample rule*
draws a random matrix and a random ring per colour, the sliders change the three scalars, and the
JSON box exports the current rule in the layout the Python code reads.

## Using the model from Python

```python
import numpy as np
from fieldlife.model import World, Sim, load_setups, find_setup, random_world
from fieldlife.render import to_image

setup = find_setup(load_setups("data/worlds.json"), "W39")
sim = Sim(World.from_config(setup["cfg"])).seed_disc(np.random.default_rng(0))
sim.run(1000)
to_image(sim.rho).save("w39.png")
print(sim.mass())                     # mass per colour; conserved to rounding

sim = Sim(random_world(np.random.default_rng(1), C=6)).seed_noise()
```

A `World` is the interaction matrix `M` (C x C), a reach and two widths per colour (the standard
deviations of the wide and tight Gaussians are `width * reach / sqrt(2)`), and the scalars `force`,
`repel` (the crowding weight) and `beta` (the sharpness). `Sim.step()` is the rule as written in the
paper: potentials by FFT convolution, affinity, then the MaCE transport over the 3 x 3 neighbourhood
with the exponent clipped to [-20, 20]. Mass is conserved to about 1e-14 in float64.

The NumPy step and the WebGL2 step in `index.html` agree to about 1e-6 per cell; after 300 steps of
glider B3 the two place its centre of mass within 1e-3 cells of each other.

## The exploration

`explore.py` draws rules the way the catalogue was built: matrix entries uniform in [-1, 1], a ring
per colour with the wide width uniform in [0.20, 0.50] and the tight width equal to the wide width
divided by a factor uniform in [1.7, 5.7]; by default the scalars stay at force 90, crowding 6,
sharpness 3 and reach 61 (`--random-scalars` draws them too). Each sample runs from noise and is
measured every 10 steps:

- *activity*: fraction of the mass that moves in one step;
- *speed*: displacement per step of the mass that moves, from the best-matching shift of 32-cell patches;
- *alignment*: how much those displacements point the same way;
- *size*: typical extent of the connected structures, in cells;
- *occupied*, *haze*, *purity*, *anisotropy*: the fraction of cells holding structure, the mass in a
  faint halo, how single-coloured the cells are, and how striped a colour is.

A sample is kept when it moves, forms structures larger than the kernel core, and moves them
coherently. Tags come from thresholds on these numbers; the thresholds are written in `explore.py`.
The catalogue in `data/worlds.json` was built with an earlier browser-based version of this
pipeline whose thresholds were set by hand while inspecting samples, so the tags of a catalogued
world and the tags this script assigns to it can differ; the rules and the behaviours are the same.

## Data layout

Both data files are `{"setups": [{"name", "tags", "cfg"}, ...]}`. A `cfg` holds `N`, `C`, `mat`
(row-major C x C), `kernels` (one per colour: `R` is the reach, `terms[0].w` and `terms[1].w` the
wide and tight widths), `force`, `repel`, `beta`, `density` and `noise` (the jitter, 0 unless
stated). A glider's `cfg` also carries `seedPattern`, its starting body. `World.from_config` and
`World.to_config` read and write this layout, and the page imports and exports it.

## The counted run of the paper

The 48 rules of the paper's exploration section are four runs of the sampler, seeds 0 to 3, 12 rules each, from noise, 300 steps on a 512 x 512 grid:

    for s in 0 1 2 3; do python -m fieldlife.explore --n 12 --seed $s --out run_s$s; done

Add `--symmetric` to replace every drawn matrix by its symmetric part (M + M^T) / 2. The symmetrisation test of the gliders is `python -m fieldlife.symmetry`, which runs each glider alone under its own matrix, under the symmetric part, and under the symmetric part with a fresh random antisymmetric part, and prints the speed of its centre of mass.
