#!/bin/bash
# usage: bench.sh <case> <reps> <variant>...
# Interleaved runs of one case over several build dirs (variant = suffix of build-<variant>).
# Prints, per run: operator timer, total runtime, wall time, load, and the comparison of the
# solution with the first run of the first variant (h5diff without tolerance).
source "$(dirname "$0")/env.sh"
case=$1; N=$2; shift 2
case $case in
  burgers) bin=demos/FiniteVolume/finite-volume-burgers; args="--min-level 4 --max-level 8 --nfiles 1 --timers"; sol=burgers_2D.h5; timer="convection operator";;
  linconv) bin=demos/FiniteVolume/finite-volume-linear-convection; args="--min-level 5 --max-level ${LINCONV_L:-8} --nfiles 1 --timers"; sol=linear_convection_2D.h5; timer="convection";;
  heat)    bin=demos/FiniteVolume/finite-volume-heat; args="--explicit --min-level 4 --max-level ${HEAT_L:-9} --Tf ${HEAT_TF:-0.4} --save-final-state-only --timers"; sol=heat_2D.h5; timer="diffusion";;
  dmr)     bin=exp/euler/euler_2d; args="--min-level 3 --max-level ${DMR_L:-7} --order 2 --nfiles 1 --timers"; sol=double_mach_reflection_hllc.h5; timer="muscl operator";;
  *) echo "unknown case"; exit 1;;
esac
ref=""
for r in $(seq 1 $N); do for v in "$@"; do
  o=$W/runs/$case/$v-$r; rm -rf $o; mkdir -p $o
  load=$(cut -d' ' -f1 /proc/loadavg)
  t0=$(date +%s.%N)
  (cd $o && $W/build-$v/$bin $args --path $o > log 2>&1); rc=$?
  wall=$(echo "$(date +%s.%N) - $t0" | bc)
  sed -i 's/\x1b\[[0-9;]*m//g' $o/log
  op=$(grep -E -- "[+\`]-- .*$timer" $o/log | head -1 | sed 's/.*[a-z)] \+\([0-9.]\+\) \+[0-9.]\+%.*/\1/')
  tot=$(grep "^total runtime" $o/log | head -1 | awk '{print $3}')
  if [ -z "$ref" ]; then ref=$o; same=reference
  elif h5diff -q $o/$sol $ref/$sol; then same=identical
  else same=DIFFERENT; fi
  printf "%-8s %-10s r%d rc=%d op=%-8s total=%-8s wall=%-7.2f load=%s %s\n" $case $v $r $rc "$op" "$tot" $wall $load $same
  [ "$o" != "$ref" ] && [ "$same" = identical ] && rm -f $o/*.h5 $o/*.xdmf
done; done
exit 0
