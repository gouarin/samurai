// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

// Price of the dynamic set algebra (include/samurai/subset/dynamic/) against
// the static one: the call sites of bench_set_call_sites with both backends,
// on the main 2D and 3D meshes of bench::mesh_cases, on each geometry, and
// on their partitions into 4 subdomains. Each
// site builds its expressions inside the timed loop, as the library does, so
// the dynamic figure includes the construction of their heap-allocated
// nodes. The ratio dynamic/BM_Site_X / static/BM_Site_X is the price of the
// runtime tree.

#include <algorithm>
#include <array>
#include <cstddef>
#include <string_view>
#include <vector>

#include <benchmark/benchmark.h>

#include "call_site_benchmarks.hpp"
#include "set_algebra_backend.hpp"
#include "set_algebra_dynamic_backend.hpp"
#include "site_inputs.hpp"

namespace
{
    // The most fronts at the default max level, stencil radius 1, on the
    // three geometries; with the sweep, the meshes of bench::sweep_cases.
    template <std::size_t dim>
    std::vector<bench::MeshCase> mesh_cases()
    {
        constexpr std::size_t nballs    = (dim == 2) ? 16 : 8;
        constexpr std::size_t max_level = bench::default_max_level<dim>();
        std::vector<bench::MeshCase> cases{
            {max_level, nballs, 1},
            {max_level - 2, nballs, 1, bench::Geometry::hole},
            {max_level, nballs, 1, bench::Geometry::periodic}
        };
        if constexpr (bench::sweep_enabled)
        {
            for (const auto& c : bench::sweep_cases<dim>())
            {
                if (c.max_level != max_level)
                {
                    cases.push_back(c);
                }
            }
        }
        return cases;
    }

    // Without the sites the dynamic algebra gets wrong. On 2026-10-18:
    // - the prediction ghosts of the mesh construction, alone
    //   (MeshPredictionGhosts) or within MRMesh::update_sub_mesh_impl
    //   (MeshSubMesh): samurai::dyn::expand of a set projection returned
    //   other cells than the static expansion, took 290 s per iteration in
    //   3D against 1 to 2 ms, and failed the assertion n_traversers == 1 of
    //   Projection<DynamicSetAdaptor> in a build with assertions;
    // - the contraction along every axis (Mode_Contract): samurai::dyn::contract
    //   returned other cells than the static one in 3D, which matches
    //   domain_boundary(mesh, level);
    // - the subdomain exchange of a partition in 3D (SubdomainExchange): in
    //   outer_subdomain_corner, the layer band
    //   difference(translate(domain, e_z), translate(domain, 0)), where both
    //   translations read the same node self(domain), came out empty along
    //   z, where the static algebra and the library find the boundary layer
    //   (16 cells at level 2 on the 2-rank partition); with a node per
    //   translation it is right.
    bool keep(std::string_view site, std::size_t dim)
    {
        constexpr std::array<std::string_view, 4> excluded{"BM_Site_MeshPredictionGhosts",
                                                           "BM_SiteCached_MeshPredictionGhosts",
                                                           "BM_Site_MeshSubMesh",
                                                           "BM_Mode_Contract"};
        if (dim == 3 && site == "BM_Site_SubdomainExchange")
        {
            return false;
        }
        return std::find(excluded.begin(), excluded.end(), site) == excluded.end();
    }

    // The partitions of the same meshes into 4 subdomains, in the box and
    // periodic.
    template <std::size_t dim>
    std::vector<bench::MeshCase> partition_cases()
    {
        std::vector<bench::MeshCase> cases;
        for (const auto& c : bench::partition_cases<dim>())
        {
            if (c.ranks == 4)
            {
                cases.push_back(c);
            }
        }
        return cases;
    }

    template <class B>
    bool register_backend()
    {
        bench::register_call_sites<B, 2>(mesh_cases<2>(), keep);
        bench::register_call_sites<B, 3>(mesh_cases<3>(), keep);
        bench::register_partition_sites<B, 2>(partition_cases<2>(), keep);
        bench::register_partition_sites<B, 3>(partition_cases<3>(), keep);
        return true;
    }

    const bool registered_static  = register_backend<bench::SamuraiStatic>();
    const bool registered_dynamic = register_backend<bench::SamuraiDynamic>();
} // namespace

BENCHMARK_MAIN();
