"""
Every adder run's ladder rung and its numbers, in one table.

    python3 train/compare_adder.py

Reads each run's log.csv, so it is safe to call mid-run. What it prints is the
accuracy the run reached, not its loss: L2 against the target arrangement is
what trains, but a world can halve it and still answer nothing correctly, so
the column that decides anything is the fraction of input pairs answered
exactly on inputs it was never trained on.
"""

import csv
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def read(run):
    rows = list(csv.DictReader(open(os.path.join(run, "log.csv"))))
    return [r for r in rows if r.get("iter")]


def main():
    runs = sorted(glob.glob(os.path.join(HERE, "runs", "*", "log.csv")))
    print(f"{'run':<12} {'op':<5} {'it':>7} {'loss':>8} "
          f"{'bit':>6} {'exact':>6} {'held-out':>9} {'s9':>11} {'s17':>11}  best")
    for logp in runs:
        run = os.path.dirname(logp)
        rows = read(run)
        if not rows or "bit_test" not in rows[0]:
            continue                      # a lizard run, not an adder one
        name = os.path.basename(run)
        cfg = {}
        for f in ("preset-best.json", "preset.json"):
            p = os.path.join(run, f)
            if os.path.exists(p):
                cfg = json.load(open(p)).get("_task", {})
                break
        last = rows[-1]
        # the row that scored best on held-out exact answers, which is what
        # preset-best.json was written from
        best = max(rows, key=lambda r: float(r["exact_test"] or 0))
        s9 = [c for c in last if c.endswith("_s9")]
        s17 = [c for c in last if c.endswith("_s17")]

        def wide(row, cols):
            if len(cols) < 2:
                return "     -     "
            b, e = row.get(cols[0]), row.get(cols[1])
            try:
                return f"{float(b):.2f} / {float(e):.2f}"
            except (TypeError, ValueError):
                return "     -     "

        h = [c for c in last if c.startswith("bit") and c[3:].isdigit()]
        e = [c for c in last if c.startswith("ex") and c[2:].isdigit()]
        print(f"{name:<12} {cfg.get('op', '?'):<5} {last['iter']:>7} "
              f"{float(last['loss']):>8.5f} "
              f"{float(last[h[-1]]):>6.2f} {float(last[e[-1]]):>6.2f} "
              f"{float(last['bit_test']):>4.2f}/{float(last['exact_test']):<4.2f} "
              f"{wide(last, sorted(s9)):>11} {wide(last, sorted(s17)):>11}  "
              f"exact {float(best['exact_test']):.2f} at it {best['iter']}")
    print("\nheld-out / s9 / s17 are bit / exact-answer rates. s9 and s17 are "
          "widths\nno run was trained at: five slots fitted, nine and seventeen "
          "asked.")


if __name__ == "__main__":
    main()
