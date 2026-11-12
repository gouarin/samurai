#!/bin/bash
# usage: probe.sh <variant>... Compiles every probe with the flags of build-<variant> and prints ok/FAIL.
source "$(dirname "$0")/../env.sh"
N=$(grep -c "PROBE ==" $W/exp/probe/user_code_probe.cpp)
for v in "$@"; do
  flags=$(grep -o "^ *\"command\": \".*burgers.cpp" $W/build-$v/compile_commands.json | head -1 | sed 's/.*c++ //; s/ -o .*//; s/-DSAMURAI_GIT_SHA=[^ ]* //')
  line="$v:"
  for p in $(seq 1 $N); do
    if c++ $flags -fsyntax-only -w -DPROBE=$p $W/exp/probe/user_code_probe.cpp > $W/exp/probe/out-$v-$p.log 2>&1; then line="$line $p=ok"; rm $W/exp/probe/out-$v-$p.log; else line="$line $p=FAIL"; fi
  done
  echo "$line"
done
