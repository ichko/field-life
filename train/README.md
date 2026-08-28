# train/

Groundwork for training field-life to sculpt a target image. The design is in
[`docs/nca-experiment.md`](../docs/nca-experiment.md); this is the code it
refers to.

| file | what it does |
|---|---|
| `fieldlife.py` | the simulation step in PyTorch, plus both kernel bakers: `bake_bank_legacy` (an exact port of `bakeKernel`, for parity) and `PolarKernels` (differentiable, with angular orders) |
| `dump_reference.mjs` | steps the real WebGL simulation in headless Chromium and writes `reference.json` |
| `parity.py` | checks the port against that dump, in two regimes |
| `target.py` | renders the lizard premultiplied, allocates the seed's per-channel mass |
| `train.py` | the trainer; `--orders` sets the kernels' angular orders, `--hidden` switches to rung 1 |
| `compare.py` | every run's loss and horizon curve in one table; safe to call mid-run |
| `evaluate.py` | replays a trained `preset.json` and reports how well it holds the target |
| `verify_browser.mjs` | loads a `preset.json` into the real `index.html` and dumps what it does |

Generated files -- `reference.json`, `browser_run.json`, `seed_field.json`,
`target.npz`, the PNGs and each run's `ckpt.pt` -- are not tracked. Every one
of them is reproducible from the commands below.

## Setup

```
npm install --no-save playwright      # the browsers are already on the image
pip install torch numpy pillow
```

`dump_reference.mjs` points at the pre-installed Chromium at
`/opt/pw-browsers/chromium-1194/chrome-linux/chrome` rather than downloading
one; if that path moves, `CHROME` at the top of the file is the only thing to
change.

## Run

```
node train/dump_reference.mjs && python3 train/parity.py
python3 train/target.py --show
```

Parity prints three things: the kernel bank must agree outright, the cool
regime must stay together for 32 steps, and the hot regime must diverge no
faster than the port diverges from its own float32 self. See §4 of the design
doc for why the last one is phrased that way.
