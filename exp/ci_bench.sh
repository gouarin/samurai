#!/bin/bash
# Builds flux_bench and euler_2d in several variants and times them on the current machine.
# Run from an activated environment providing g++ and, optionally, clang++.
W=$(cd "$(dirname "$0")/.." && pwd)
# conda activation exports its own optimisation flags; the variants must control them.
unset CXXFLAGS CFLAGS CPPFLAGS LDFLAGS DEBUG_CXXFLAGS DEBUG_CFLAGS DEBUG_CPPFLAGS
case $(uname -m) in
  x86_64) NATIVE="-march=native";;
  *) NATIVE="-mcpu=native";;
esac
PREFIX=$CONDA_PREFIX

echo "## machine"
uname -m
lscpu | grep -E "Model name|Vendor ID|Flags|Features|^CPU\(s\)" | cut -c1-600
g++ --version | head -1
command -v clang++ > /dev/null && clang++ --version | head -1

built=""
# variant <name> <compiler> <eager ON|OFF> <force inline ON|OFF> <extra flags>
variant()
{
    local name=$1 cxx=$2 eager=$3 inl=$4 flags=$5
    command -v $cxx > /dev/null || { echo "skip $name: no $cxx"; return; }
    local container=xtensor
    [ $eager = ON ] && container=array
    if cmake -S $W -B $W/build-$name -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=$PREFIX \
        -DCMAKE_CXX_COMPILER=$cxx -DCMAKE_CXX_FLAGS="$flags" -DSAMURAI_ENABLE_CACHE=OFF \
        -DSAMURAI_FLUX_CONTAINER=$container -DSAMURAI_EXP_EAGER_ARRAY=$eager -DEXP_FORCE_INLINE=$inl > $W/build-$name.cfg.log 2>&1 \
        && ninja -C $W/build-$name flux_bench euler_2d > $W/build-$name.log 2>&1; then
        built="$built $name"
        # packed double arithmetic in the whole flux_bench binary, as a trace of the vectorizer
        local packed=$(objdump -d $W/build-$name/exp/euler/flux_bench | grep -cE "\sv?(add|sub|mul|div)pd\s|\sf(add|sub|mul|div)\s+v[0-9]+\.2d")
        echo "built $name (flags: '$flags'), packed FP instructions: $packed"
    else
        echo "FAILED $name"; tail -30 $W/build-$name.cfg.log $W/build-$name.log
    fi
}

echo "## builds"
variant gcc-base                  g++     OFF OFF ""
variant gcc-B                     g++     ON  OFF ""
variant gcc-Binl                  g++     ON  ON  ""
variant gcc-Binl-noloopvec        g++     ON  ON  "-fno-tree-loop-vectorize"
variant gcc-Binl-native           g++     ON  ON  "$NATIVE"
variant gcc-Binl-native-noloopvec g++     ON  ON  "$NATIVE -fno-tree-loop-vectorize"
variant clang-B                   clang++ ON  OFF ""
variant clang-Binl                clang++ ON  ON  ""
variant clang-Binl-native         clang++ ON  ON  "$NATIVE"

echo "## flux_bench 4096 400 7, 3 interleaved runs"
for r in 1 2 3; do for v in $built; do
    $W/build-$v/exp/euler/flux_bench 4096 400 7 | sed "s/^/$v r$r /"
done; done

echo "## DMR level 3-7, 3 interleaved runs"
bash $W/exp/bench.sh dmr 3 $built
