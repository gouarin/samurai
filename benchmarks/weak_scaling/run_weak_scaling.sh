#!/usr/bin/env bash
# Weak scaling runs of burgers_os_2d_mpi: every rank solves the same periodic
# problem on its own box, so the ideal time is constant with the number of ranks.
#
# Usage:
#   BINARIES="main=/path/to/main/build study=/path/to/study/build" \
#   GRIDS="1x1 2x2 3x3 4x4 6x6 8x8 12x12 16x16" REPS=3 \
#   LAUNCHER="mpiexec -n" ./run_weak_scaling.sh results_dir
#
# BINARIES  label=build_dir pairs; the demo is build_dir/demos/FiniteVolume/finite-volume-burgers-os-2d-mpi
# GRIDS     process grids npx x npy (the global domain is npx x npy boxes)
# REPS      repetitions of each run; the versions are interleaved inside a repetition
# LAUNCHER  command that takes the number of ranks as its next argument
#           ("mpiexec -n", "srun -n", "mpirun -np"...)
# DEMO_ARGS extra arguments of the demo (defaults: levels 4 to 10, Tf 0.5)
set -euo pipefail

out=${1:?usage: $0 results_dir}
: "${BINARIES:?set BINARIES to label=build_dir pairs}"
GRIDS=${GRIDS:-"1x1 2x2 3x3 4x4"}
REPS=${REPS:-3}
LAUNCHER=${LAUNCHER:-"mpiexec -n"}
DEMO_ARGS=${DEMO_ARGS:-""}

mkdir -p "$out"
{
    echo "date: $(date -Iseconds)"
    echo "host: $(hostname)"
    echo "launcher: $LAUNCHER"
    echo "grids: $GRIDS"
    echo "reps: $REPS"
    echo "demo args: $DEMO_ARGS"
    for pair in $BINARIES; do
        echo "binary ${pair%%=*}: ${pair#*=}"
    done
} > "$out/run_info.txt"

for rep in $(seq 1 "$REPS"); do
    for grid in $GRIDS; do
        npx=${grid%x*}
        npy=${grid#*x}
        n=$((npx * npy))
        for pair in $BINARIES; do
            label=${pair%%=*}
            exe="${pair#*=}/demos/FiniteVolume/finite-volume-burgers-os-2d-mpi"
            log="$out/${label}_${npx}x${npy}_rep${rep}.log"
            echo "[$(date +%H:%M:%S)] $label ${npx}x${npy} rep $rep"
            # check_diff (built into the demo) aborts if the subdomain meshes differ
            # shellcheck disable=SC2086
            $LAUNCHER "$n" "$exe" --npx "$npx" --npy "$npy" --no-output --timers $DEMO_ARGS > "$log" 2>&1 \
                || echo "  FAILED (exit code $?), see $log"
        done
    done
done
echo "done: python3 $(dirname "$0")/parse_timers.py $out"
