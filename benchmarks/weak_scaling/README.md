# Weak scaling of burgers_os_2d_mpi

Every rank solves the same periodic Burgers problem on its own box (the global
domain is `npx x npy` boxes), so the ideal time does not depend on the number of
ranks. The demo checks at every time step that all the subdomain meshes are
identical and aborts otherwise.

## Build the two versions

```bash
# reference: upstream main
git worktree add ../samurai-main upstream/main
# study: main + #477, #478, #510, #511, #512, #513, #514
git worktree add ../samurai-study weak-scaling-study

for w in ../samurai-main ../samurai-study; do
    cmake -S $w -B $w/build -DCMAKE_BUILD_TYPE=Release -DWITH_MPI=ON -DBUILD_DEMOS=ON
    cmake --build $w/build --target finite-volume-burgers-os-2d-mpi -j
done
```

## Run

```bash
BINARIES="main=../samurai-main/build study=../samurai-study/build" \
GRIDS="1x1 2x2 3x3 4x4 6x6 8x8" REPS=3 LAUNCHER="mpiexec -n" \
./run_weak_scaling.sh results-$(hostname)
```

- `LAUNCHER`: the command that takes the number of ranks next (`"mpiexec -n"`,
  `"srun -n"`, `"mpirun -np"`). Under Slurm, run the script inside an allocation
  large enough for the biggest grid.
- `GRIDS`: square grids keep the global domain square; `8x1` / `1x8` grids show
  the costs that depend on the shape of the global domain.
- `DEMO_ARGS`: extra demo arguments, e.g. `"--max-level 11"` for more work per rank.
- One rank per core; pin the ranks if the launcher does not (e.g. `--bind-to core`).

## Analyse

```bash
python3 parse_timers.py results-$(hostname)
```

It prints, for each version and grid, the median over the repetitions of the max
over the ranks of the main timers and the efficiency `T(1x1) / T(npx x npy)`, and
writes every timer to `results-.../timers.csv`.
