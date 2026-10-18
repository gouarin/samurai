// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

// MR-adapted meshes refined around the surfaces of balls, and the copy of
// their sets into bench::SiteInputs. Needs the mesh, the fields, the boundary
// conditions and the MR adaptation: only generate_set_algebra_inputs and
// bench_set_library_ops include it.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <benchmark/benchmark.h>

#include <xtensor/generators/xbuilder.hpp>

#include <samurai/bc.hpp>
#include <samurai/box.hpp>
#include <samurai/domain_builder.hpp>
#include <samurai/field.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/stencil.hpp>

#include "set_algebra_fixtures.hpp"
#include "site_inputs.hpp"

namespace bench
{
    // disable_minimal_ghost_width keeps the ghost width at stencil_radius: by
    // default the mesh raises the max stencil radius to 2 (see SET_ALGEBRA.md).
    template <std::size_t dim>
    auto mr_config(std::size_t max_level, int stencil_radius)
    {
        return samurai::mesh_config<dim>().min_level(2).max_level(max_level).max_stencil_radius(stencil_radius).disable_minimal_ghost_width();
    }

    template <std::size_t dim>
    using mr_mesh_t = decltype(samurai::mra::make_mesh(std::declval<samurai::Box<double, dim>>(), mr_config<dim>(4, 1)));

    template <std::size_t dim>
    using mr_field_t = decltype(samurai::make_scalar_field<double>("u", std::declval<mr_mesh_t<dim>&>()));

    // The mesh adapted to this indicator is refined to the finest level along
    // every ball surface. `shift` moves the balls along the diagonal.
    template <std::size_t dim>
    auto balls_indicator(std::size_t nballs, double shift)
    {
        return [centers = ball_centers<dim>(nballs, 0.25, 0.75),
                radius  = 0.2 / std::pow(static_cast<double>(nballs), 1. / static_cast<double>(dim)),
                shift](const auto& x)
        {
            for (const auto& c : centers)
            {
                double r2 = 0;
                for (std::size_t d = 0; d < dim; ++d)
                {
                    const double dx = x[d] - c[d] - shift;
                    r2 += dx * dx;
                }
                if (r2 <= radius * radius)
                {
                    return 1.;
                }
            }
            return 0.;
        };
    }

    constexpr double mr_epsilon = 1e-3;

    // How far the fronts move in one time step.
    template <class Mesh>
    double step_shift(const Mesh& mesh)
    {
        return 4. * mesh.min_cell_length();
    }

    // The mesh of c before adaptation, on its geometry: the unit box, the
    // unit box with a hole as demos/FiniteVolume/linear_convection_obstacle
    // builds it, or the periodic unit box.
    template <std::size_t dim>
    mr_mesh_t<dim> make_unit_mesh(const MeshCase& c)
    {
        auto config = mr_config<dim>(c.max_level, c.stencil_radius);
        const samurai::Box<double, dim> unit_box(xt::zeros<double>({dim}), xt::ones<double>({dim}));
        switch (c.geometry)
        {
            case Geometry::box:
                break;
            case Geometry::periodic:
                config.periodic(true);
                break;
            case Geometry::hole:
            {
                samurai::DomainBuilder<dim> domain(unit_box);
                domain.remove(samurai::Box<double, dim>(xt::ones<double>({dim}) * 0.5, xt::ones<double>({dim}) * 0.75));
                return samurai::mra::make_mesh(domain, config);
            }
        }
        return samurai::mra::make_mesh(unit_box, config);
    }

    template <std::size_t dim>
    struct AdaptedCase
    {
        mr_mesh_t<dim> mesh;
        mr_field_t<dim> u;

        AdaptedCase(const MeshCase& c, bool moved)
            : mesh(make_unit_mesh<dim>(c))
            , u(samurai::make_scalar_field<double>("u", mesh, balls_indicator<dim>(c.nballs, moved ? step_shift(mesh) : 0.)))
        {
            samurai::make_bc<samurai::Dirichlet<1>>(u, 0.);
            samurai::make_MRAdapt(u)(samurai::mra_config().epsilon(mr_epsilon));
        }

        AdaptedCase(const AdaptedCase&)            = delete;
        AdaptedCase& operator=(const AdaptedCase&) = delete;
    };

    // Built once per binary and shared by its benchmarks. `moved` gives the
    // mesh of the next time step.
    template <std::size_t dim>
    AdaptedCase<dim>& adapted_case(const MeshCase& c, bool moved = false)
    {
        static std::map<std::tuple<mesh_key_t, bool>, std::unique_ptr<AdaptedCase<dim>>> cache;
        auto& entry = cache[{c.key(), moved}];
        if (!entry)
        {
            entry = std::make_unique<AdaptedCase<dim>>(c, moved);
        }
        return *entry;
    }

    template <class Mesh>
    void report_mesh(benchmark::State& state, const Mesh& mesh)
    {
        report_mesh(state, mesh[Mesh::mesh_id_t::cells]);
    }

    template <std::size_t dim>
    SiteInputs<dim> make_site_inputs(const mr_mesh_t<dim>& mesh)
    {
        using mesh_t    = mr_mesh_t<dim>;
        using mesh_id_t = typename mesh_t::mesh_id_t;
        static_assert(std::is_same_v<typename mesh_t::ca_type, typename SiteInputs<dim>::ca_t>);
        static_assert(mesh_t::config_t::prediction_stencil_radius == prediction_stencil_radius);

        SiteInputs<dim> in;
        in.min_level          = mesh.min_level();
        in.max_level          = mesh.max_level();
        in.graduation_width   = mesh.graduation_width();
        in.max_stencil_radius = mesh.max_stencil_radius();
        in.domain_is_box      = mesh.domain().is_box();
        in.periodic           = mesh.periodicity();
        in.cells              = mesh[mesh_id_t::cells];
        in.cells_and_ghosts   = mesh[mesh_id_t::cells_and_ghosts];
        in.proj_cells         = mesh[mesh_id_t::proj_cells];
        in.reference          = mesh[mesh_id_t::reference];
        in.union_             = mesh.get_union();
        in.domain_pyramid     = mesh.domain_pyramid();
        for (std::size_t level = 0; level <= mesh.max_level(); ++level)
        {
            in.subdomain_pyramid[level] = mesh.subdomain(level);
        }
        samurai::for_each_diagonal_direction<dim>(
            [&](const auto& direction)
            {
                in.corners.push_back(mesh.corner(direction));
            });
        set_domain_bounds(in);
        return in;
    }

    // The sets update_cell_array_from_tag (algorithm/graduation.hpp) builds
    // from the tags of `cells` before its set expression: a copy of its loop
    // over the cells, which fills them cell by cell.
    template <std::size_t dim, class Tag>
    void tag_sets(SiteInputs<dim>& in, const typename mr_mesh_t<dim>::ca_type& old_ca, const Tag& tag)
    {
        using ca_type          = typename mr_mesh_t<dim>::ca_type;
        using interval_t       = typename ca_type::interval_t;
        using value_t          = typename interval_t::value_t;
        using size_type        = unsigned int;
        using unsigned_value_t = std::make_unsigned_t<value_t>;
        using coord_type       = typename ca_type::lca_type::coord_type;

        const auto& mesh = tag.mesh();

        ca_type ca_add_m;
        ca_type ca_remove_m;
        ca_type ca_add_p;
        ca_type ca_remove_p;

        std::vector<interval_t> add_p_interval;
        std::vector<coord_type> add_p_inner_stencil;
        std::vector<std::size_t> add_p_idx;

        for (std::size_t level = old_ca.min_level(); level < old_ca.max_level() + 1; ++level)
        {
            const auto begin = old_ca[level].cbegin();
            const auto end   = old_ca[level].cend();
            for (auto it = begin; it != end; ++it)
            {
                const auto& x_interval = *it;
                const auto& yz         = it.index();
                const bool is_yz_even  = dim == 1 or xt::all(xt::equal(yz % 2, 0));

                for (value_t x = x_interval.start; x < x_interval.end; ++x)
                {
                    const size_type itag         = static_cast<size_type>(x_interval.index) + static_cast<unsigned_value_t>(x);
                    const bool refine            = tag[itag] & static_cast<std::uint8_t>(samurai::CellFlag::refine);
                    const bool coarsenAndNotKeep = tag[itag] & static_cast<std::uint8_t>(samurai::CellFlag::coarsen)
                                               and not(tag[itag] & static_cast<std::uint8_t>(samurai::CellFlag::keep));
                    if (refine and level < mesh.max_level())
                    {
                        ca_remove_p[level].add_point_back(x, yz);
                        if constexpr (dim == 1)
                        {
                            ca_add_p[level + 1].add_interval_back({2 * x, 2 * x + 2}, {});
                        }
                        else
                        {
                            samurai::nestedLoop<dim - 1, 0, dim - 2>(0,
                                                                     2,
                                                                     [&](const auto& inner_stencil)
                                                                     {
                                                                         add_p_interval.push_back({2 * x, 2 * x + 2});
                                                                         if constexpr (dim > 2)
                                                                         {
                                                                             add_p_inner_stencil.emplace_back(2 * yz + inner_stencil);
                                                                             add_p_idx.push_back(add_p_interval.size() - 1);
                                                                         }
                                                                     });
                        }
                    }
                    else if (coarsenAndNotKeep and level > mesh.min_level())
                    {
                        if (x % 2 == 0 and is_yz_even)
                        {
                            ca_add_m[level - 1].add_point_back(x >> 1, yz >> 1);
                        }
                        ca_remove_m[level].add_point_back(x, yz);
                    }
                }
                if (dim != 1 and (it + 1 == end or (it + 1).index()[dim - 2] != yz[dim - 2]))
                {
                    samurai::add_list_of_interval_back(add_p_interval, coord_type(2 * yz), add_p_inner_stencil, add_p_idx, ca_add_p[level + 1]);
                    add_p_interval.clear();
                    add_p_inner_stencil.clear();
                    add_p_idx.clear();
                }
            }
        }
        in.tag_add_m    = ca_add_m;
        in.tag_remove_m = ca_remove_m;
        in.tag_add_p    = ca_add_p;
        in.tag_remove_p = ca_remove_p;
    }

    // What Adapt::harten (mr/adapt.hpp) passes to make_graduation at the first
    // iteration of the adaptation of `adapted` to the fronts shifted by
    // `shift`: update_cell_array_from_tag(cells, tag), before graduation, in
    // in.ungraded_cells, and the sets it builds from the tags (tag_sets). A
    // copy of harten up to that call, for the configuration of AdaptedCase:
    // sequential, no relative detail, boundary not kept refined.
    template <std::size_t dim>
    void add_next_step(SiteInputs<dim>& in, const mr_mesh_t<dim>& adapted, std::size_t nballs, double shift)
    {
        using mesh_id_t = typename mr_mesh_t<dim>::mesh_id_t;

        auto mesh = adapted;
        auto u    = samurai::make_scalar_field<double>("u", mesh, balls_indicator<dim>(nballs, shift));
        samurai::make_bc<samurai::Dirichlet<1>>(u, 0.);
        auto detail = samurai::make_scalar_field<double>("detail", mesh, 0.);
        auto tag    = samurai::make_scalar_field<std::uint8_t>("tag", mesh, std::uint8_t{0});

        const std::size_t min_level = mesh.min_level();
        const std::size_t max_level = mesh.max_level();
        const auto cfg              = samurai::mra_config().epsilon(mr_epsilon);

        samurai::for_each_cell(mesh[mesh_id_t::cells],
                               [&](const auto& cell)
                               {
                                   tag[cell] = static_cast<std::uint8_t>(samurai::CellFlag::keep);
                               });
        samurai::update_ghost_mr(u);

        for (std::size_t level = (min_level > 0) ? min_level - 1 : 0; level < max_level; ++level)
        {
            auto ghosts_below_cells = samurai::intersection(
                                          mesh[mesh_id_t::all_cells][level],
                                          samurai::union_(mesh[mesh_id_t::cells][level + 1], mesh[mesh_id_t::cells][level + 2]))
                                          .on(level);
            ghosts_below_cells.apply_op(samurai::compute_detail(detail, u));
        }

        for (std::size_t level = std::max<std::size_t>(min_level, 1); level <= max_level; ++level)
        {
            const double eps_l = cfg.epsilon() / static_cast<double>(1 << (dim * (max_level - level)));
            auto subset_1      = samurai::intersection(mesh[mesh_id_t::cells][level], mesh[mesh_id_t::all_cells][level - 1]).on(level - 1);
            subset_1.apply_op(samurai::mr_criteria(detail, tag, eps_l, cfg.regularity() + dim));
            samurai::update_tag_subdomains(level, tag, true);
        }

        for (std::size_t level = max_level; level > 0; --level)
        {
            auto keep_subset = samurai::intersection(mesh[mesh_id_t::cells][level], mesh[mesh_id_t::all_cells][level - 1]).on(level - 1);
            samurai::update_tag_periodic(level, tag);
            samurai::update_tag_subdomains(level, tag);
            keep_subset.apply_op(samurai::maximum(tag));
        }

        in.ungraded_cells = samurai::update_cell_array_from_tag(mesh[mesh_id_t::cells], tag);
        tag_sets(in, mesh[mesh_id_t::cells], tag);
    }
} // namespace bench
