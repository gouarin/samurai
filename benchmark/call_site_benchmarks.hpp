// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

// The benchmarks of the call sites (set_call_sites.hpp), for one backend of
// the set algebra (set_algebra_backend.hpp). register_call_sites<B, dim>
// registers them as <B::name>/BM_Site_X<dim>, so that
// --benchmark_filter='^<name>/' selects the backend.
//
// Args: (max_level, number of balls, stencil radius), see bench::mesh_cases.

#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <utility>

#include <benchmark/benchmark.h>

#include "set_algebra_backend.hpp"
#include "set_algebra_fixtures.hpp"
#include "set_call_sites.hpp"
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
    } // namespace call_sites

    // `args` gives the meshes, as mesh_args<dim> does. Registers the sites
    // for which keep(name) holds, all of them by default.
    template <class B, std::size_t dim, class Args, class Keep = bool (*)(std::string_view)>
    void register_call_sites(
        Args&& args,
        Keep&& keep =
            [](std::string_view)
        {
            return true;
        })
    {
        const auto add = [&](const char* site, void (*bm)(benchmark::State&))
        {
            if (!keep(site))
            {
                return;
            }
            const std::string name = std::string(B::name) + "/" + site + "<" + std::to_string(dim) + ">";
            benchmark::RegisterBenchmark(name, bm)->Apply(args);
        };
        add("BM_Site_MeshPredictionGhosts", call_sites::MeshPredictionGhosts<B, dim>);
        add("BM_SiteCached_MeshPredictionGhosts", call_sites::CachedMeshPredictionGhosts<B, dim>);
        add("BM_Site_Corners", call_sites::Corners<B, dim>);
        add("BM_Site_GhostProjection", call_sites::GhostProjection<B, dim>);
        add("BM_Site_GhostPrediction", call_sites::GhostPrediction<B, dim>);
        add("BM_Site_OuterGhosts", call_sites::OuterGhosts<B, dim>);
        add("BM_SiteCached_OuterGhosts", call_sites::CachedOuterGhosts<B, dim>);
        add("BM_Site_Detail", call_sites::Detail<B, dim>);
        add("BM_Site_Tag", call_sites::Tag<B, dim>);
        add("BM_Site_FieldTransfer", call_sites::FieldTransfer<B, dim>);
        add("BM_Site_Interfaces", call_sites::Interfaces<B, dim>);
    }
} // namespace bench
