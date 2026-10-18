// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

// Price of the dynamic set algebra (include/samurai/subset/dynamic/) against
// the static one: the call sites of bench_set_call_sites with both backends,
// on one 2D and one 3D mesh of bench::mesh_cases. Each site builds its
// expressions inside the timed loop, as the library does, so the dynamic
// figure includes the construction of their heap-allocated nodes. The ratio
// dynamic/BM_Site_X / static/BM_Site_X is the price of the runtime tree.

#include <cstddef>
#include <string_view>

#include <benchmark/benchmark.h>

#include "call_site_benchmarks.hpp"
#include "set_algebra_backend.hpp"
#include "set_algebra_dynamic_backend.hpp"
#include "site_inputs.hpp"

namespace
{
    // The most fronts at the default max level, stencil radius 1.
    template <std::size_t dim>
    void mesh_args(benchmark::internal::Benchmark* b)
    {
        b->Args({static_cast<long>(bench::default_max_level<dim>()), (dim == 2) ? 16 : 8, 1});
        b->Unit(benchmark::kMicrosecond);
    }

    // Without the prediction ghosts of the mesh construction: on 2026-10-18,
    // samurai::dyn::expand returned other cells than the static expansion
    // there, and took 290 s per iteration in 3D against 1 to 2 ms.
    bool keep(std::string_view site)
    {
        return site.find("MeshPredictionGhosts") == std::string_view::npos;
    }

    template <class B>
    bool register_backend()
    {
        bench::register_call_sites<B, 2>(mesh_args<2>, keep);
        bench::register_call_sites<B, 3>(mesh_args<3>, keep);
        return true;
    }

    const bool registered_static  = register_backend<bench::SamuraiStatic>();
    const bool registered_dynamic = register_backend<bench::SamuraiDynamic>();
} // namespace

BENCHMARK_MAIN();
