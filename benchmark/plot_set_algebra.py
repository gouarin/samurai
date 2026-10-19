#!/usr/bin/env python3
# Copyright 2018-2025 the samurai's authors
# SPDX-License-Identifier:  BSD-3-Clause

"""Plot the time per interval of the set algebra benchmarks.

Reads the JSON output of one or more benchmark runs
(--benchmark_out=run.json --benchmark_out_format=json) and writes one PNG per
benchmark family (a site or synthetic benchmark in one dimension, such as
BM_Site_Tag<3>): the time per emitted interval (the `t/itv` counter) against
the number of intervals or the level, one curve per backend and per set of
the other arguments, and, when the run has two backends, the ratio of their
times. Needs matplotlib.

    python benchmark/plot_set_algebra.py run.json --out plots
    python benchmark/plot_set_algebra.py run.json --x level --filter 'Site_Tag'
    python benchmark/plot_set_algebra.py before.json:old after.json:new --ratio new old

The first argument of a benchmark is its level (the max level of a mesh), so
the curves of a sweep are the benchmarks that differ only by it. A curve
needs --min-points points (2 by default): the meshes outside the sweep are
single points and are left out unless --min-points 1.
"""

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

GEOMETRIES = {0: "box", 1: "hole", 2: "periodic"}
# The categorical palette, in the order backends are assigned to it.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
LINESTYLES = ["-", "--", ":", "-."]
MARKERS = ["o", "s", "^", "D"]
X_LABELS = {
    "itv": "intervals emitted per iteration",
    "in_itv": "intervals of the input set",
    "leaf_itv": "intervals of the leaves of the mesh",
    "level": "level (max level of the mesh)",
}


def parse_name(name):
    """Split 'backend/BM_X<d>/a/b/c' into (backend, family, args).

    A benchmark without a backend prefix (the synthetic ones) gets the
    backend ''. Aggregate suffixes (_median, ...) are expected to be gone.
    """
    match = re.match(r"^(?:(?P<backend>[^/<]+)/)?(?P<family>[^/]+?<[^>]*>|[^/]+)(?P<args>(?:/-?\d+)*)$", name)
    if match is None:
        return None
    args = tuple(int(a) for a in match.group("args").split("/")[1:])
    return match.group("backend") or "", match.group("family"), args


def describe_args(family, args):
    """The arguments other than the level, in words where they are known."""
    if family.startswith(("BM_Site", "BM_Mode")) and len(args) in (3, 4):
        words = [f"{args[0]} balls", f"radius {args[1]}", GEOMETRIES.get(args[2], f"geometry {args[2]}")]
        if len(args) == 4:
            words.append(f"{args[3]} ranks")
        return ", ".join(words)
    return "/".join(["L", *(str(a) for a in args)]) if args else ""


def load(paths):
    """Points by (family, rest of the args, backend): {level: record}.

    `path:label` prefixes the backends of that file with `label/`, to compare
    two runs of the same backend. A median aggregate, when the run has one,
    replaces the repetitions; otherwise the repetitions are averaged.
    """
    points = defaultdict(dict)
    for spec in paths:
        path, label = spec, ""
        if ":" in spec and not Path(spec).exists():
            path, _, label = spec.rpartition(":")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        runs = defaultdict(list)
        medians = {}
        for b in data["benchmarks"]:
            if b.get("error_occurred"):
                print(f"{path}: {b['name']}: {b.get('error_message', 'error')}, left out", file=sys.stderr)
                continue
            if "t/itv" not in b:
                continue
            run_name = b.get("run_name", b["name"])
            if b.get("run_type") == "aggregate":
                if b.get("aggregate_name") == "median":
                    medians[run_name] = b
            else:
                runs[run_name].append(b)
        for run_name, records in runs.items():
            parsed = parse_name(run_name)
            if parsed is None or not parsed[2]:
                continue
            backend, family, args = parsed
            if label:
                backend = f"{label}/{backend}" if backend else label
            if run_name in medians:
                record = medians[run_name]
            else:
                record = {key: sum(r[key] for r in records) / len(records) for key in records[0] if isinstance(records[0][key], (int, float))}
            points[(family, args[1:], backend)][args[0]] = record
    return points


def x_values(curve, x):
    levels = sorted(curve)
    if x == "level":
        return levels, [float(level) for level in levels]
    return levels, [curve[level].get(x, math.nan) for level in levels]


def family_x(curves, x):
    """The x of a family: `x`, or the level when `x` is the same at every
    point of a curve (a site whose output does not grow with the mesh)."""
    if x == "level":
        return x
    for curve in curves.values():
        values = {curve[level].get(x) for level in curve}
        if None in values or len(values) < 2:
            return "level"
    return x


def plain_log_ticks(axis):
    """Labels a log axis with plain numbers, at 1, 2 and 5 times each power
    of ten, instead of powers of ten alone."""
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    axis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_formatter(NullFormatter())


def integer_or_log_x(ax, x):
    """Levels on integer ticks, counts of intervals on a log scale."""
    if x == "level":
        from matplotlib.ticker import MaxNLocator

        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    else:
        ax.set_xscale("log")


def plot_family(plt, family, curves, colors, args):
    """curves: {(rest of the args, backend): {level: record}}."""
    x = family_x(curves, args.x)
    backends = sorted({backend for _, backend in curves}, key=list(colors).index)
    rests = sorted({rest for rest, _ in curves})
    ratio = args.ratio if args.ratio else (backends[1], backends[0]) if len(backends) == 2 else None
    pairs = []
    if ratio:
        pairs = [rest for rest in rests if (rest, ratio[0]) in curves and (rest, ratio[1]) in curves]

    ncols = 2 if pairs else 1
    fig, axes = plt.subplots(1, ncols, figsize=(6.4 * ncols, 4.8), squeeze=False, layout="constrained")
    ax = axes[0][0]
    handles = []
    for (rest, backend), curve in sorted(curves.items(), key=lambda item: (rests.index(item[0][0]), backends.index(item[0][1]))):
        k = rests.index(rest)
        _, xs = x_values(curve, x)
        ys = [curve[level]["t/itv"] * 1e9 for level in sorted(curve)]
        label = " ".join(part for part in (backend, describe_args(family, rest)) if part)
        (line,) = ax.plot(
            xs,
            ys,
            color=colors[backend],
            linestyle=LINESTYLES[k % len(LINESTYLES)],
            marker=MARKERS[k % len(MARKERS)],
            markersize=5,
            linewidth=2,
            label=label,
        )
        handles.append(line)
    ax.set_yscale("log")
    integer_or_log_x(ax, x)
    plain_log_ticks(ax.yaxis)
    ax.set_xlabel(X_LABELS.get(x, x))
    ax.set_ylabel("time per emitted interval (ns)")
    ax.set_title("time per interval")
    ax.grid(True, which="major", color="#d0d0d0", linewidth=0.6)
    ax.grid(True, which="minor", color="#ececec", linewidth=0.4)

    if pairs:
        rax = axes[0][1]
        for rest in pairs:
            k = rests.index(rest)
            num, den = curves[(rest, ratio[0])], curves[(rest, ratio[1])]
            levels = sorted(set(num) & set(den))
            _, xs = x_values({level: den[level] for level in levels}, x)
            ys = [num[level]["t/itv"] / den[level]["t/itv"] for level in levels]
            rax.plot(
                xs,
                ys,
                color="#3a3a38",
                linestyle=LINESTYLES[k % len(LINESTYLES)],
                marker=MARKERS[k % len(MARKERS)],
                markersize=5,
                linewidth=2,
                label=describe_args(family, rest) or None,
            )
        rax.axhline(1.0, color="#8a8a86", linewidth=1)
        integer_or_log_x(rax, x)
        rax.set_ylim(bottom=0)
        rax.set_xlabel(X_LABELS.get(x, x))
        rax.set_ylabel(f"{ratio[0]} / {ratio[1]}")
        rax.set_title(f"time ratio {ratio[0]} / {ratio[1]}")
        rax.grid(True, which="major", color="#d0d0d0", linewidth=0.6)
        rax.grid(True, which="minor", color="#ececec", linewidth=0.4)
        if len(pairs) > 1:
            rax.legend(fontsize="small", frameon=False)

    fig.suptitle(family, fontsize="large")
    fig.legend(handles=handles, loc="outside lower center", ncols=min(len(handles), 2 * ncols), frameon=False, fontsize="small")
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", help="JSON outputs of the benchmarks, as path or path:label")
    parser.add_argument("--out", default="set_algebra_plots", help="directory of the figures (default: %(default)s)")
    parser.add_argument("--x", choices=sorted(X_LABELS), default="itv", help="horizontal axis (default: %(default)s)")
    parser.add_argument("--filter", default="", help="regular expression on the family names")
    parser.add_argument("--ratio", nargs=2, metavar=("NUM", "DEN"), help="backends of the ratio (default: the two backends of the run)")
    parser.add_argument("--min-points", type=int, default=2, help="points a curve needs to be drawn (default: %(default)s)")
    parser.add_argument("--format", default="png", help="format of the figures (default: %(default)s)")
    args = parser.parse_args()

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        sys.exit("plot_set_algebra.py needs matplotlib (micromamba install matplotlib-base)")

    points = load(args.runs)
    families = defaultdict(dict)
    for (family, rest, backend), curve in points.items():
        if len(curve) >= args.min_points and re.search(args.filter, family):
            families[family][(rest, backend)] = curve
    if not families:
        sys.exit("no curve to draw: check --filter and --min-points")

    backends = sorted({backend for curves in families.values() for _, backend in curves}, key=lambda b: (b != "static", b))
    if len(backends) > len(COLORS):
        sys.exit(f"{len(backends)} backends, at most {len(COLORS)} can be told apart")
    colors = dict(zip(backends, COLORS))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for family, curves in sorted(families.items()):
        fig = plot_family(plt, family, curves, colors, args)
        name = re.sub(r"[^A-Za-z0-9_]+", "_", family).strip("_")
        path = out / f"{name}.{args.format}"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        print(path)


if __name__ == "__main__":
    main()
