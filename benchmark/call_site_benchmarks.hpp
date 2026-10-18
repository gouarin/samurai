// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

// The benchmarks of the call sites (set_call_sites.hpp) and of the ways to
// consume a set (set_modes.hpp), for one backend of the set algebra
// (set_algebra_backend.hpp). register_call_sites<B, dim> registers them as
// <B::name>/BM_Site_X<dim> and <B::name>/BM_Mode_X<dim>, so that
// --benchmark_filter='^<name>/' selects the backend; register_partition_sites
// registers those of an MPI rank on the partitions of a mesh.
//
// Args: (max_level, number of balls, stencil radius, geometry), see
// bench::mesh_cases, and the number of ranks for a partition, see
// bench::partition_cases.

#pragma once

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <benchmark/benchmark.h>

#include "set_algebra_backend.hpp"
#include "set_algebra_fixtures.hpp"
#include "set_call_sites.hpp"
#include "set_modes.hpp"
#include "site_inputs.hpp"

namespace bench
{
    namespace call_sites
    {
        // The callback of a site only counts: the time is that of the set
        // algebra in the library function, to set against BM_SiteOp_X or
        // BM_Op_* of bench_set_library_ops.
        template <class B, class Site>
        void run(benchmark::State& state, Site&& site)
        {
            state.counters["sets"] = static_cast<double>(bench::run_site<B>(state, std::forward<Site>(site)));
        }

        template <class B, std::size_t dim>
        void MeshPredictionGhosts(benchmark::State& state)
        {
            const auto& in = backend_inputs<B, dim>(state);
            run<B>(state,
                   [&](auto&& f)
                   {
                       sites::mesh_prediction_ghosts<B>(in, sites::projected_subdomain<B>(in), f);
                   });
        }

        // The Cached variants read the coarse domain or subdomain the mesh
        // already stores instead of coarsening the finest one at every use:
        // the gap with the site is what caching coarse sets would save.
        template <class B, std::size_t dim>
        void CachedMeshPredictionGhosts(benchmark::State& state)
        {
            const auto& in = backend_inputs<B, dim>(state);
            run<B>(state,
                   [&](auto&& f)
                   {
                       sites::mesh_prediction_ghosts<B>(in, sites::cached_subdomain(in), f);
                   });
        }

        template <class B, std::size_t dim>
        void Corners(benchmark::State& state)
        {
            const auto& in = backend_inputs<B, dim>(state);
            run<B>(state,
                   [&](auto&& f)
                   {
                       sites::corners<B>(in, f);
                   });
        }

        template <class B, std::size_t dim>
        void GhostProjection(benchmark::State& state)
        {
            const auto& in = backend_inputs<B, dim>(state);
            run<B>(state,
                   [&](auto&& f)
                   {
                       sites::ghost_projection<B>(in, f);
                   });
        }

        template <class B, std::size_t dim>
        void GhostPrediction(benchmark::State& state)
        {
            const auto& in = backend_inputs<B, dim>(state);
            run<B>(state,
                   [&](auto&& f)
                   {
                       sites::ghost_prediction<B>(in, f);
                   });
        }

        template <class B, std::size_t dim>
        void OuterGhosts(benchmark::State& state)
        {
            const auto& in = backend_inputs<B, dim>(state);
            run<B>(state,
                   [&](auto&& f)
                   {
                       sites::outer_ghosts<B>(in, sites::projected_domain<B>(in), f);
                   });
        }

        template <class B, std::size_t dim>
        void CachedOuterGhosts(benchmark::State& state)
        {
            const auto& in = backend_inputs<B, dim>(state);
            run<B>(state,
                   [&](auto&& f)
                   {
                       sites::outer_ghosts<B>(in, sites::cached_domain(in), f);
                   });
        }

        template <class B, std::size_t dim>
        void PeriodicGhosts(benchmark::State& state)
        {
            const auto& in = backend_inputs<B, dim>(state);
            run<B>(state,
                   [&](auto&& f)
                   {
                       sites::periodic_ghosts<B>(in, f);
                   });
        }

        template <class B, std::size_t dim>
        void Detail(benchmark::State& state)
        {
            const auto& in = backend_inputs<B, dim>(state);
            run<B>(state,
                   [&](auto&& f)
                   {
                       sites::detail<B>(in, f);
                   });
        }

        template <class B, std::size_t dim>
        void Tag(benchmark::State& state)
        {
            const auto& in = backend_inputs<B, dim>(state);
            run<B>(state,
                   [&](auto&& f)
                   {
                       sites::tag<B>(in, f);
                   });
        }

        template <class B, std::size_t dim>
        void Interfaces(benchmark::State& state)
        {
            const auto& in = backend_inputs<B, dim>(state);
            run<B>(state,
                   [&](auto&& f)
                   {
                       sites::interfaces<B>(in, f);
                   });
        }

        template <class B, std::size_t dim>
        void FieldTransfer(benchmark::State& state)
        {
            const auto& in     = backend_inputs<B, dim>(state);
            const auto& new_in = backend_inputs<B, dim>(state, true);
            run<B>(state,
                   [&](auto&& f)
                   {
                       sites::field_transfer<B>(in, new_in, f);
                   });
        }

        // The sites that build level cell arrays: the time is that of
        // building them, `itv` and `cells` count what they keep.
        template <class B, class Site>
        void run_build(benchmark::State& state, Site&& site)
        {
            state.counters["sets"] = static_cast<double>(bench::run_build_site<B>(state, std::forward<Site>(site)));
        }

        template <class B, std::size_t dim>
        void MeshSubMesh(benchmark::State& state)
        {
            const auto& in = backend_inputs<B, dim>(state);
            run_build<B>(state,
                         [&](auto&& emit)
                         {
                             sites::mesh_sub_mesh<B>(in, emit);
                         });
        }

        template <class B, std::size_t dim>
        void ConstructUnion(benchmark::State& state)
        {
            const auto& in = backend_inputs<B, dim>(state);
            run_build<B>(state,
                         [&](auto&& emit)
                         {
                             sites::construct_union<B>(in, emit);
                         });
        }

        template <class B, std::size_t dim>
        void BuildPyramid(benchmark::State& state)
        {
            const auto& in = backend_inputs<B, dim>(state);
            run_build<B>(state,
                         [&](auto&& emit)
                         {
                             sites::build_pyramid<B>(in, emit);
                         });
        }

        template <class B, std::size_t dim>
        void CellArrayFromTag(benchmark::State& state)
        {
            const auto& in = backend_inputs<B, dim>(state);
            run_build<B>(state,
                         [&](auto&& emit)
                         {
                             sites::cell_array_from_tag<B>(in, emit);
                         });
        }

        template <class B, std::size_t dim>
        void AmrSubMesh(benchmark::State& state)
        {
            const auto& in = backend_inputs<B, dim>(state);
            run_build<B>(state,
                         [&](auto&& emit)
                         {
                             sites::amr_sub_mesh<B>(in, emit);
                         });
        }

        // The regions are built once, when the boundary condition is.
        template <class B, std::size_t dim>
        void BcCells(benchmark::State& state)
        {
            const auto& in     = backend_inputs<B, dim>(state);
            const auto regions = sites::bc_regions<B>(in);
            run<B>(state,
                   [&](auto&& f)
                   {
                       sites::bc_cells<B>(in, regions, f);
                   });
        }

        template <class B, std::size_t dim>
        void BcCorners(benchmark::State& state)
        {
            const auto& in = backend_inputs<B, dim>(state);
            run<B>(state,
                   [&](auto&& f)
                   {
                       sites::bc_corners<B>(in, f);
                   });
        }

        template <class B, std::size_t dim>
        void BcFurtherGhosts(benchmark::State& state)
        {
            const auto& in = backend_inputs<B, dim>(state);
            run<B>(state,
                   [&](auto&& f)
                   {
                       sites::bc_further_ghosts<B>(in, f);
                   });
        }

        template <class B, std::size_t dim>
        void PerCellChildren(benchmark::State& state)
        {
            const auto& in = backend_inputs<B, dim>(state);
            run<B>(state,
                   [&](auto&& f)
                   {
                       sites::per_cell_children<B>(in, f);
                   });
        }

        // The ghosts without an owner are found outside the timed loop.
        template <class B, std::size_t dim>
        void OwnershipPerCell(benchmark::State& state)
        {
            const auto& in    = backend_inputs<B, dim>(state);
            const auto ghosts = sites::ownership_ghosts<B>(in);
            run<B>(state,
                   [&](auto&& f)
                   {
                       sites::ownership_per_cell<B>(in, ghosts, f);
                   });
        }

        // The sites on a partition run every rank in turn: the time is that
        // of all the ranks, as if one process ran them all.
        template <class B, std::size_t dim>
        auto partition_neighbours(const std::vector<backend_inputs_t<B, dim>>& ranks)
        {
            std::vector<sites::neighbours_t<dim, typename backend_inputs_t<B, dim>::ca_t>> neighbours;
            for (std::size_t r = 0; r < ranks.size(); ++r)
            {
                neighbours.push_back(sites::neighbours_of(ranks, r));
            }
            return neighbours;
        }

        template <class B, std::size_t dim>
        void PartitionFindNeighbourhood(benchmark::State& state)
        {
            const auto& ranks      = backend_partition<B, dim>(state);
            state.counters["sets"] = static_cast<double>(
                bench::run_empty_site<B>(state,
                                         [&](auto&& f)
                                         {
                                             for (std::size_t r = 0; r < ranks.size(); ++r)
                                             {
                                                 auto found = sites::find_neighbourhood<B>(ranks, r, f);
                                                 benchmark::DoNotOptimize(found);
                                             }
                                         }));
        }

        template <class B, std::size_t dim>
        void PartitionConstructSubdomain(benchmark::State& state)
        {
            const auto& ranks = backend_partition<B, dim>(state);
            run_build<B>(state,
                         [&](auto&& emit)
                         {
                             for (const auto& in : ranks)
                             {
                                 sites::construct_subdomain<B>(in, emit);
                             }
                         });
        }

        template <class B, std::size_t dim>
        void PartitionSubdomainExchange(benchmark::State& state)
        {
            const auto& ranks     = backend_partition<B, dim>(state);
            const auto neighbours = partition_neighbours<B, dim>(ranks);
            run<B>(state,
                   [&](auto&& f)
                   {
                       for (std::size_t r = 0; r < ranks.size(); ++r)
                       {
                           sites::subdomain_exchange<B>(ranks[r], neighbours[r], f);
                       }
                   });
        }

        template <class B, std::size_t dim>
        void PartitionMeshSubMesh(benchmark::State& state)
        {
            const auto& ranks     = backend_partition<B, dim>(state);
            const auto neighbours = partition_neighbours<B, dim>(ranks);
            run_build<B>(state,
                         [&](auto&& emit)
                         {
                             for (std::size_t r = 0; r < ranks.size(); ++r)
                             {
                                 sites::mesh_sub_mesh<B>(ranks[r], neighbours[r], emit);
                             }
                         });
        }

        template <class B, std::size_t dim>
        void PartitionInterfaces(benchmark::State& state)
        {
            const auto& ranks     = backend_partition<B, dim>(state);
            const auto neighbours = partition_neighbours<B, dim>(ranks);
            run<B>(state,
                   [&](auto&& f)
                   {
                       for (std::size_t r = 0; r < ranks.size(); ++r)
                       {
                           sites::interfaces<B>(ranks[r], neighbours[r], f);
                       }
                   });
        }

        template <class B, std::size_t dim>
        void PartitionPeriodicGhosts(benchmark::State& state)
        {
            const auto& ranks     = backend_partition<B, dim>(state);
            const auto neighbours = partition_neighbours<B, dim>(ranks);
            run<B>(state,
                   [&](auto&& f)
                   {
                       for (std::size_t r = 0; r < ranks.size(); ++r)
                       {
                           sites::periodic_ghosts<B>(ranks[r], neighbours[r], f);
                       }
                   });
        }

        // Every rank peels towards each of its neighbours, from the partition
        // as it is (see sites::diffusion_peel).
        template <class B, std::size_t dim>
        void PartitionDiffusionPeel(benchmark::State& state)
        {
            const auto& ranks     = backend_partition<B, dim>(state);
            const auto neighbours = partition_neighbours<B, dim>(ranks);
            std::vector<double> loads;
            for (const auto& in : ranks)
            {
                loads.push_back(sites::peel_load<B>(in));
            }
            run_build<B>(state,
                         [&](auto&& emit)
                         {
                             for (std::size_t r = 0; r < ranks.size(); ++r)
                             {
                                 for (const auto* neighbour : neighbours[r])
                                 {
                                     sites::diffusion_peel<B>(ranks[r], *neighbour, loads[r], emit);
                                 }
                             }
                         });
        }

        // The modes of set_modes.hpp, one benchmark per function.
        template <class B, std::size_t dim, auto Mode>
        void TraverseMode(benchmark::State& state)
        {
            const auto& in = backend_inputs<B, dim>(state);
            run<B>(state,
                   [&](auto&& f)
                   {
                       Mode(in, f);
                   });
        }

        template <class B, std::size_t dim, auto Mode>
        void BuildMode(benchmark::State& state)
        {
            const auto& in = backend_inputs<B, dim>(state);
            run_build<B>(state,
                         [&](auto&& emit)
                         {
                             Mode(in, emit);
                         });
        }

        template <class B, std::size_t dim, auto Mode>
        void EmptyMode(benchmark::State& state)
        {
            const auto& in         = backend_inputs<B, dim>(state);
            state.counters["sets"] = static_cast<double>(bench::run_empty_site<B>(state,
                                                                                  [&](auto&& f)
                                                                                  {
                                                                                      Mode(in, f);
                                                                                  }));
        }
    } // namespace call_sites

    // Registers the sites for which keep(name, dim) holds, all of them by
    // default, on the meshes of `cases`; the periodic ghosts only on the
    // periodic meshes, the others have none.
    template <class B, std::size_t dim, class Keep = bool (*)(std::string_view, std::size_t)>
    void register_call_sites(
        const std::vector<MeshCase>& cases,
        Keep&& keep =
            [](std::string_view, std::size_t)
        {
            return true;
        })
    {
        std::vector<MeshCase> periodic_cases;
        std::copy_if(cases.begin(),
                     cases.end(),
                     std::back_inserter(periodic_cases),
                     [](const MeshCase& c)
                     {
                         return c.geometry == Geometry::periodic;
                     });
        const auto add = [&](const char* site, void (*bm)(benchmark::State&), bool periodic_only = false)
        {
            const auto& site_cases = periodic_only ? periodic_cases : cases;
            if (!keep(site, dim) || site_cases.empty())
            {
                return;
            }
            const std::string name = std::string(B::name) + "/" + site + "<" + std::to_string(dim) + ">";
            mesh_args(benchmark::RegisterBenchmark(name, bm), site_cases);
        };
        add("BM_Site_MeshPredictionGhosts", call_sites::MeshPredictionGhosts<B, dim>);
        add("BM_SiteCached_MeshPredictionGhosts", call_sites::CachedMeshPredictionGhosts<B, dim>);
        add("BM_Site_Corners", call_sites::Corners<B, dim>);
        add("BM_Site_GhostProjection", call_sites::GhostProjection<B, dim>);
        add("BM_Site_GhostPrediction", call_sites::GhostPrediction<B, dim>);
        add("BM_Site_OuterGhosts", call_sites::OuterGhosts<B, dim>);
        add("BM_SiteCached_OuterGhosts", call_sites::CachedOuterGhosts<B, dim>);
        add("BM_Site_PeriodicGhosts", call_sites::PeriodicGhosts<B, dim>, true);
        add("BM_Site_Detail", call_sites::Detail<B, dim>);
        add("BM_Site_Tag", call_sites::Tag<B, dim>);
        add("BM_Site_FieldTransfer", call_sites::FieldTransfer<B, dim>);
        add("BM_Site_Interfaces", call_sites::Interfaces<B, dim>);
        add("BM_Site_MeshSubMesh", call_sites::MeshSubMesh<B, dim>);
        add("BM_Site_ConstructUnion", call_sites::ConstructUnion<B, dim>);
        add("BM_Site_BuildPyramid", call_sites::BuildPyramid<B, dim>);
        add("BM_Site_CellArrayFromTag", call_sites::CellArrayFromTag<B, dim>);
        add("BM_Site_AmrSubMesh", call_sites::AmrSubMesh<B, dim>);
        add("BM_Site_BcCells", call_sites::BcCells<B, dim>);
        add("BM_Site_BcCorners", call_sites::BcCorners<B, dim>);
        add("BM_Site_BcFurtherGhosts", call_sites::BcFurtherGhosts<B, dim>);
        add("BM_Site_PerCellChildren", call_sites::PerCellChildren<B, dim>);
        add("BM_Site_OwnershipPerCell", call_sites::OwnershipPerCell<B, dim>);

        using in_t = backend_inputs_t<B, dim>;
        // A function template cannot be a template argument: each mode is
        // wrapped in a lambda without capture.
#define BENCH_MODE(KIND, NAME, FUNCTION)                      \
    add("BM_Mode_" NAME,                                      \
        call_sites::KIND##Mode<B,                             \
                               dim,                           \
                               [](const in_t& in, auto& f)    \
                               {                              \
                                   modes::FUNCTION<B>(in, f); \
                               }>)
        BENCH_MODE(Traverse, "Traverse", traverse_proj_cells);
        BENCH_MODE(Build, "Materialize", materialize_proj_cells);
        BENCH_MODE(Build, "Build", build_proj_cells);
        BENCH_MODE(Empty, "EmptyHit", empty_hit);
        BENCH_MODE(Empty, "EmptyMiss", empty_miss);
        BENCH_MODE(Traverse, "Contract", contract);
        BENCH_MODE(Traverse, "ContractMasked", contract_masked);
        BENCH_MODE(Traverse, "ExpandUniform", expand_uniform);
        BENCH_MODE(Traverse, "ExpandAxes", expand_axes);
        BENCH_MODE(Traverse, "ExpandMasked", expand_masked);
        BENCH_MODE(Traverse, "OnMixedLevels", on_mixed_levels);
        BENCH_MODE(Traverse, "OnSameLevel", on_same_level);
        BENCH_MODE(Traverse, "NestedOn", nested_on);
        BENCH_MODE(Traverse, "FlatOn", flat_on);
        BENCH_MODE(Traverse, "IntersectionSelfOn", intersection_self_on);
        BENCH_MODE(Traverse, "SelfOn", self_on);
#undef BENCH_MODE
    }

    // Registers the sites of an MPI rank on the partitions of `cases`
    // (partition_cases), as <B::name>/BM_Site_X<dim>/<args>/<ranks>: those
    // that only run with MPI, and the variants with the neighbours of the
    // sites that have one. The periodic ghosts only on the periodic
    // partitions.
    template <class B, std::size_t dim, class Keep = bool (*)(std::string_view, std::size_t)>
    void register_partition_sites(
        const std::vector<MeshCase>& cases,
        Keep&& keep =
            [](std::string_view, std::size_t)
        {
            return true;
        })
    {
        std::vector<MeshCase> periodic_cases;
        std::copy_if(cases.begin(),
                     cases.end(),
                     std::back_inserter(periodic_cases),
                     [](const MeshCase& c)
                     {
                         return c.geometry == Geometry::periodic;
                     });
        const auto add = [&](const char* site, void (*bm)(benchmark::State&), bool periodic_only = false)
        {
            const auto& site_cases = periodic_only ? periodic_cases : cases;
            if (!keep(site, dim) || site_cases.empty())
            {
                return;
            }
            const std::string name = std::string(B::name) + "/" + site + "<" + std::to_string(dim) + ">";
            mesh_args(benchmark::RegisterBenchmark(name, bm), site_cases);
        };
        add("BM_Site_FindNeighbourhood", call_sites::PartitionFindNeighbourhood<B, dim>);
        add("BM_Site_ConstructSubdomain", call_sites::PartitionConstructSubdomain<B, dim>);
        add("BM_Site_SubdomainExchange", call_sites::PartitionSubdomainExchange<B, dim>);
        add("BM_Site_MeshSubMesh", call_sites::PartitionMeshSubMesh<B, dim>);
        add("BM_Site_Interfaces", call_sites::PartitionInterfaces<B, dim>);
        add("BM_Site_PeriodicGhosts", call_sites::PartitionPeriodicGhosts<B, dim>, true);
        add("BM_Site_DiffusionPeel", call_sites::PartitionDiffusionPeel<B, dim>);
    }
} // namespace bench
