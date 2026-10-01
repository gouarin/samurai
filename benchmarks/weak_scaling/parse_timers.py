#!/usr/bin/env python3
"""Parse the --timers tables written by run_weak_scaling.sh.

Writes <results_dir>/timers.csv (one row per log and timer) and prints, for each
version and process grid, the median over the repetitions of the max over the
ranks of a few timers, and the weak scaling efficiency T(1 rank) / T(P ranks).

Usage: parse_timers.py results_dir [timer ...]
"""

import csv
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ANSI = re.compile(r"\x1b\[[0-9;]*m")
LOG = re.compile(r"(?P<label>.+)_(?P<npx>\d+)x(?P<npy>\d+)_rep(?P<rep>\d+)\.log$")
# "<tree><name>  <min> [r] <max> [r] <ave> <pct>% [<pct>%] <calls>"
ROW = re.compile(
    r"^(?P<tree>[|`+\- ]*)(?P<name>.*?)\s+(?P<min>[\d.]+)\s+\[\s*\d+\]\s+(?P<max>[\d.]+)\s+\[\s*\d+\]\s+(?P<ave>[\d.]+)\s+[\d.]+%(?:\s+[\d.]+%)?\s+(?P<calls>\d+)\s*$"
)

DEFAULT_TIMERS = [
    "total runtime",
    "mesh adaptation",
    "mesh adaptation/mesh update/mesh construction",
    "mesh adaptation/mesh update/make_graduation",
    "mesh adaptation/ghost update",
    "ghost update",
    "convection operator",
]


def parse_log(path):
    """Return {timer path: (min, max, ave, calls)}; nested timers are joined with '/'."""
    timers = {}
    stack = []
    in_table = False
    for raw in path.read_text(errors="replace").splitlines():
        line = ANSI.sub("", raw).rstrip()
        if line.startswith("Timer"):
            in_table = True
            continue
        if not in_table:
            continue
        if line.startswith("total runtime (ave)"):
            break
        m = ROW.match(line)
        if not m:
            continue
        depth = len(m["tree"]) // 4
        name = m["name"].strip()
        stack = stack[:depth] + [name]
        # every timer is nested in "total runtime": leave it out of the paths
        key = "/".join(stack[1:]) if depth > 0 else name
        timers[key] = (float(m["min"]), float(m["max"]), float(m["ave"]), int(m["calls"]))
    return timers


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    results = Path(sys.argv[1])
    wanted = sys.argv[2:] or DEFAULT_TIMERS

    rows = []
    for log in sorted(results.glob("*.log")):
        m = LOG.match(log.name)
        if not m:
            continue
        timers = parse_log(log)
        if not timers:
            print(f"warning: no timers in {log.name} (failed run?)", file=sys.stderr)
            continue
        for key, (tmin, tmax, tave, calls) in timers.items():
            rows.append(
                {
                    "label": m["label"],
                    "npx": int(m["npx"]),
                    "npy": int(m["npy"]),
                    "ranks": int(m["npx"]) * int(m["npy"]),
                    "rep": int(m["rep"]),
                    "timer": key,
                    "min": tmin,
                    "max": tmax,
                    "ave": tave,
                    "calls": calls,
                }
            )

    with open(results / "timers.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "npx", "npy", "ranks", "rep", "timer", "min", "max", "ave", "calls"])
        writer.writeheader()
        writer.writerows(rows)

    samples = defaultdict(list)
    for r in rows:
        samples[(r["label"], r["npx"], r["npy"], r["timer"])].append(r["max"])

    labels = sorted({r["label"] for r in rows})
    grids = sorted({(r["npx"], r["npy"]) for r in rows}, key=lambda g: (g[0] * g[1], g))
    for label in labels:
        print(f"\n== {label} (median over repetitions of the max over ranks, seconds)")
        print(f"{'timer':52s}" + "".join(f"{f'{x}x{y}':>10s}" for x, y in grids))
        for timer in wanted:
            line = f"{timer:52s}"
            for x, y in grids:
                v = samples.get((label, x, y, timer))
                line += f"{statistics.median(v):10.3f}" if v else f"{'-':>10s}"
            print(line)
        ref = samples.get((label, 1, 1, "total runtime"))
        if ref:
            t1 = statistics.median(ref)
            line = f"{'efficiency T(1)/T(P)':52s}"
            for x, y in grids:
                v = samples.get((label, x, y, "total runtime"))
                line += f"{t1 / statistics.median(v):10.2f}" if v else f"{'-':>10s}"
            print(line)
    print(f"\nall timers: {results / 'timers.csv'}")


if __name__ == "__main__":
    main()
