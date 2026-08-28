# Can field-life compute? An 8-bit adder in a mass-conserving field

**Status:** encoding, trainer and browser support landed and checked. Training
in progress; §7 holds the numbers as they arrive and says which are final.

**A note on provenance.** This was asked for as a port of the experiment at
`izhv.eu/ncpu`, and that page is blocked by this session's egress policy, so it
has not been read. Everything below is designed from the task name -- an 8-bit
adder -- and from what `docs/nca-experiment.md` established about this
simulation. Where the reference design made a different choice, the difference
will be in the encoding of §2, which is the load-bearing part.

---

## 1. The question

`docs/nca-experiment.md` asked whether field-life could be trained to hold a
*picture*. It could: a mass-conserving cellular automaton, given the lizard's
own per-channel mass in a seed disc, learns to arrange it into the lizard.

This asks something different. A picture is one fixed target. A computation is a
*function*: the field is handed an input it has never seen and has to arrive at
the answer for that input, and be right about it. Nothing about sculpting one
shape implies anything about that.

The specific function is 8-bit addition, and it is a good choice for this
substrate for a reason that is not obvious until it is said: **a ripple-carry
adder is a cellular automaton.** It is a chain of identical one-bit full adders
with a carry running along it -- one local rule, repeated, with a signal
travelling. field-life's rule is already the same in every cell. The two
structures are the same shape, and the question is whether gradient descent can
find the rule.

## 2. What mass conservation forbids

One tick of MaCE moves mass; it never makes any. Per channel, exactly:

```
rho' = E * (3x3 sum of rho/Z)        Z = 3x3 sum of E,  E = exp(clamp(beta*A, -20, 20))
```

Three consequences decide the encoding, and getting them wrong makes the task
unsolvable rather than merely hard.

**A bit cannot be "mass present or absent".** Under that encoding the total mass
of the answer depends on the input -- `1 + 1 = 2` turns two lit bits into one --
and a mass-conserving rule cannot hit a target whose mass it was not handed. It
is also badly shaped for learning: an all-dark output is reachable by dumping
everything somewhere unscored, which is a local minimum sitting right next to
the initial condition.

**So a bit is one unit of mass at one of two places.** Slot `i` owns a 0-rail
below a mid line and a 1-rail above it. Every input costs the same mass, the
answer is purely *which side*, and there is no do-nothing escape -- a blob left
on the mid line is equally wrong for a 0 and for a 1. Dual rail is what digital
logic does when it needs a bit to be a *presence* rather than a *level*, and it
is what this substrate needs for the same reason.

**The output is pre-charged, not grown.** Channel 2 starts with one blob per
slot on the mid line, holding exactly the mass the answer costs. Solving the
task is moving each blob to the rail the arithmetic names. That is the lizard's
mass contract with an input-dependent target: the loss is about arrangement
only, there is no mass term to balance, and the totals are right by
construction.

The hidden channels are pre-charged the same way. They are the only place a
carry can live, and a carry has to be made of something.

| channel | role |
|---|---|
| 0 | A, dual rail, one blob per slot — read as red |
| 1 | B, dual rail — read as green |
| 2 | the answer, pre-charged on the mid line — read as blue, and the only channel scored |
| 3–5 | workspace, pre-charged on the mid line; nothing constrains them |

## 3. The bit axis is the torus

Slots tile the x-axis exactly: `W = nslots * pitch`, so the adder is **cyclic**.
That sounds wrong and is not. The top slot's two input bits are pinned to zero,
so its carry-out is always zero, so the carry that wraps around into slot 0 is
always zero -- the fixed point is unique and is ordinary addition. An `n`-bit
adder needs `n+1` slots: the extra one carries the answer's top bit and
terminates the ripple.

The reason to want this is the whole experiment. With no boundary there is
nothing width-specific for the rule to latch onto, so **a world fitted on five
slots can be dropped on nine and asked to be an 8-bit adder it has never seen,
or on seventeen and asked to be a 16-bit one.** Width generalisation is the test
that separates a learned algorithm from a fitted table, and it is not a test a
table can pass by getting lucky.

It also makes the experiment affordable. Five slots is a 30x24 grid.

## 4. The ladder

A single pass/fail on the adder says almost nothing about *where* a
mass-conserving field stops being able to compute. So the same encoding, the
same trainer and the same world are run against five tasks in increasing order
of what they demand:

| rung | the answer at slot i | what it needs |
|---|---|---|
| `copy` | `a` | transport, driven by one channel. If this fails the encoding is wrong and arithmetic is beside the point. |
| `and` | `a & b` | one gate per slot, no communication between slots: a threshold on the local density of two channels. |
| `or` | `a \| b` | the same at the other threshold. |
| `xor` | `a ^ b` | still one slot at a time, but **not** a threshold -- the answer is high in the middle of the input range and low at both ends, so no monotone response to local density produces it. |
| `add` | `a ^ b ^ carry` | `xor`, plus a carry that has to travel along the slot axis, so slot `i`'s answer is not a function of anything inside slot `i`. |

`xor` is where the interesting boundary should be. The affinity as shipped is
**linear** in the convolved fields -- one matrix multiply per cell -- and a
linear functional of local density cannot separate `{01, 10}` from `{00, 11}` in
one step. What is not obvious is whether it has to: the step as a whole is not
linear in `rho`, because `rho` enters both the exponent and the mass being
transported, and a rollout is many steps deep. So `xor` is a real question about
the law, not a foregone conclusion, and it is the one rung whose answer is
worth having on its own.

## 5. What is trained

Rung 0 of `docs/nca-experiment.md`, unchanged: the interaction matrix, one polar
kernel per channel, and `force`, `repel`, `beta`. About 190 numbers, against
65536 possible input pairs at 8 bits. Nothing here has the capacity to memorise
a table, which is why held-out accuracy is the only number worth printing.

The kernels carry **angular orders**. A lobe of order `m = 0` is a ring, which is
every kernel this simulation has ever sampled; `m = 1` is a signed gradient
along an axis, which is what a Sobel filter is. A bank of rings cannot prefer a
direction, and a carry has to travel one way along the slot axis. §8 is what
`index.html` needed before a bank like that would load.

Reaches are **pinned to the stencil** rather than learned, and that is a
portability decision, not a modelling one: `uploadKernels` bakes a short-reach
channel at a coarser grid rather than as a radial scale on the shared one, so a
learned reach exports to a different kernel than the one trained. Pinning makes
the two bakers agree to float precision (§8), and costs little -- `mu` and `w`
still decide where inside the reach a lobe's weight sits.

## 6. Protocol

- 5 slots (a 4-bit adder) on a 30x24 torus, C = 6, stencil half-width 7 --
  a reach of about one slot, so a carry has to be *carried*, not seen.
- 75% of the width's input pairs train; the rest are held out.
- Loss: L2 on channel 2 against the answer's arrangement, scored throughout a
  16-step backprop window, from a Growing-NCA sample pool.
- **A frame is scored on its absolute age, not its position in the window.** A
  carry has to ripple, and the field cannot know slot 4's answer before slot 3
  has decided, so charging for the arrangement at step 2 asks the impossible.
  With a pool those are different numbers and the lizard's trainer measured the
  wrong one.
- The pool starts young. A pool of long-lived states is a refinement, not a
  bootstrap -- that is the lizard experiment's own lesson, and hands a world
  that cannot yet do the task a diet of rotted states to repair.
- Adam, per-parameter gradient normalisation, the divergence-rate penalty.
- Reported every checkpoint: per-bit and exact-answer accuracy at six rollout
  lengths out to 256 steps, held-out accuracy, and accuracy at 9 and 17 slots.

**What counts as a result**, whether or not the adder lands:

- the rung at which the ladder breaks, which is a statement about what a linear
  affinity can compute and is worth having on its own;
- whether a world fitted at one width is correct at another, which is the
  algorithm-versus-table question stated so that it can be answered;
- whether an answer, once reached, is *held* -- a fixed point of the dynamics
  rather than a configuration passed through on the way somewhere else.
