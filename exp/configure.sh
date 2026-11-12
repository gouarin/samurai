#!/bin/bash
# usage: configure.sh <build-name> [cmake args]. Configures $W/build-<name> with GCC from samurai-env.
source "$(dirname "$0")/env.sh"
name=$1; shift
cmake -S $W -B $W/build-$name -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=$E \
  -DCMAKE_CXX_COMPILER=$E/bin/c++ -DCMAKE_C_COMPILER=$E/bin/cc \
  -DCMAKE_AR=$E/bin/aarch64-conda-linux-gnu-ar -DCMAKE_RANLIB=$E/bin/aarch64-conda-linux-gnu-ranlib \
  -DBUILD_TESTS=ON -DBUILD_DEMOS=ON -DWITH_PETSC=ON -DSAMURAI_ENABLE_CACHE=OFF \
  -DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=$W/../perf-weno5-per-component/build/_deps/googletest-src "$@"
