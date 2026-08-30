# Field Life

Particle life as a grid of concentrations, running on the GPU in a browser:
[www.izhv.eu/field-life](https://www.izhv.eu/field-life)

Particle life is a swarm of coloured dots that attract and repel each other by
colour. Field Life keeps the interaction matrix and throws away the dots. Each
colour becomes a scalar field over a grid — one number per cell, saying how much
of that colour sits there — and the whole grid is updated every frame.

## How it works

One step has two halves.

First, each colour's field is **convolved with its own kernel**. A kernel gives
every cell within a fixed radius — a few dozen cells — a weight, and that weight
depends only on how far the cell is from the middle. Two cells the same distance
out always get the same weight, so the whole kernel is really one curve read off
against distance. Convolving means: for every cell in the grid, add up that
colour's field over all the cells around it inside the radius, each one
multiplied by the weight for its distance. The number that comes out says how
much of that colour is nearby, counted the way the kernel says to count it. The
default curve is negative near zero distance and positive out at a ring, so the
sum is largest when the colour forms a hollow shell at the ring's distance
instead of a solid blob.

The **interaction matrix** then mixes the colours. Entry `M[c][d]` is how
strongly colour `c` is pulled towards the amount of colour `d` that its kernel
just measured; negative entries push away instead. Adding those up over every
`d`, multiplying by a global strength, and subtracting an amount proportional to
how crowded the cell already is leaves one number per cell per colour. Call it
`A`: how much that colour wants to be in that cell.

Second, **the mass moves**. Each cell splits its own contents among its nine
neighbours, itself included, giving each a share proportional to `exp(beta*A)`
at that neighbour, with the nine shares scaled to sum to exactly one. Every unit
that leaves a cell arrives at another, so the total per colour never changes for
the life of the run. The simulation only moves mass around; it cannot create or
destroy it.

The matrix need not be symmetric, so `M[c][d]` and `M[d][c]` can disagree: red
can chase green while green flees red. That asymmetry is where the chasing,
orbiting and gliding come from.

## The worlds and the creatures

A world is a matrix, a kernel per colour, and a dozen scalars. Most random ones
do nothing worth watching, so the 120 in `worlds.json` were found by sampling
that space in a headless browser and scoring what each one did — how fast it
moved, how big its structures were, whether the motion was coherent. They are
the shelf on the left under the field, and `worlds.html` is a gallery of them
with an animation each.

The creatures on the right shelf were found inside those worlds. Something
compact that survives on its own is cut out of a running world, dropped alone
into an empty one, and kept only if its mass stays gathered around its own
centre rather than hazing out over the grid. What holds is then taken out of the
empty world it has been living in — centred on itself, so there are no cut marks
— and saved as the preset's starting pattern. Loading one starts an empty world
with the creature in the middle. The rule is the parent world's, untouched;
only the opening state differs.

The lizard is different: its kernels and matrix were fitted so that a seed disc
of the right mass grows into the lizard emoji. Mass is conserved here, so the
seed has to arrive already holding what the picture costs.

## What is in here

| | |
|---|---|
| `index.html` | the whole simulation and its interface, in one file. WebGL2, no build step, no dependencies |
| `worlds.json`, `worlds/` | the worlds shelf: 120 worlds with a thumbnail and an animation each, plus the lizard |
| `gliders.json`, `gliders/` | the creatures shelf: seven of them, same shape |
| `worlds.html` | a gallery of the 120 worlds, sortable, with the preset for each |
| `presets.json` | eighteen setups made by hand rather than found |
| `*-presets.json` | the same worlds and creatures as importable preset files |
| `particle_life.py` | the dots version this started from, and the gif it makes |
| `staging/` | a copy of the page for trying things that are not ready |

## Running it

It is a static page. Opening `index.html` from disk works, but the shelves are
fetched, so they only appear when it is served:

```
python3 -m http.server 8000
```

The simulation needs WebGL2 with `EXT_color_buffer_float` — it renders into
32-bit float textures — and it will say so plainly if the GPU cannot.

## Built on

- **MaCE** — the transport rule used here: each cell hands its mass to its nine
  neighbours in proportion to `exp(beta*A)`, so mass is conserved by
  construction. Papadopoulos, V. & Guichard, E. (2025).
  [arXiv:2507.12306](https://arxiv.org/abs/2507.12306)
- **Particle Life** — the interaction matrix and short-range repulsion, in its
  original swarm-of-dots form. Mohr, T., [particle-life.com](https://particle-life.com),
  after Ventrella, J., [Clusters](http://ventrella.com/Clusters/)
- **Lenia** — kernels convolved with a field, fed through a growth function.
  Chan, B. W.-C. (2019). Complex Systems 28(3), 251–286.
  [arXiv:1812.05433](https://arxiv.org/abs/1812.05433)
- **Flow-Lenia** — the nearest relative: per-channel affinity, an anti-crowding
  term and mass conservation on a grid, via reintegration tracking rather than a
  softmax. Plantec, E. et al. (2023).
  [arXiv:2212.07906](https://arxiv.org/abs/2212.07906)
- **Non-reciprocal Cahn–Hilliard** — the same asymmetric-matrix idea as continuum
  physics, where it produces traveling waves and chasing phases. Saha, S.,
  Agudo-Canalejo, J. & Golestanian, R. (2020). Phys. Rev. X 10, 041009.
  [arXiv:2005.07101](https://arxiv.org/abs/2005.07101)

The idea came after Alex Rowe showed me [cip-life](https://github.com/aprowe/cip-life),
his take on particle life, which initialises the interaction matrix as a
continuous CPPN-generated field; then a discussion with Mundy Reimer and Sean
Hardy about continuity and the concept of a field.

## Cite

```
@misc{fieldlife,
author = {Zhechev, Iliya},
title  = {Field Life: particle life as a grid of concentrations},
year   = {2026},
url    = {https://www.izhv.eu/field-life}
}
```
