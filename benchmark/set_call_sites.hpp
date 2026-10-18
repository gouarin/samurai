// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

// The call sites: the set expressions of the library functions samurai runs
// at every time step, copied from the source cited above each and written
// against the backend B (set_algebra_backend.hpp) through A = Algebra<B>:
// samurai::f(...) becomes A::f(...) and s.on(l) becomes A::on(s, l). The
// operands are read from bench::SiteInputs instead of the mesh. Each site
// passes every set it evaluates to f(level, set).

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>

#include <xtensor/views/xview.hpp>

#include <samurai/stencil.hpp>

#include "set_algebra_backend.hpp"
#include "site_inputs.hpp"

namespace bench::sites
{
    // MRMesh::update_sub_mesh_impl (mr/mesh.hpp): the ghosts the prediction
    // needs one and two levels below each cell, then the projection cells.
    template <class B, std::size_t dim, class CA, class SubdomainAt, class F>
    void mesh_prediction_ghosts(const SiteInputs<dim, CA>& in, SubdomainAt&& subdomain_at, F&& f)
    {
        using A              = Algebra<B>;
        constexpr int pred_r = prediction_stencil_radius;
        const auto& cells    = in.cells;
        const auto& cag      = in.cells_and_ghosts;
        for (std::size_t level = cells.min_level(); level <= cells.max_level(); ++level)
        {
            if (level < 2 || A::empty(cells[level]))
            {
                continue;
            }
            f(level - 2, A::expand(A::on(cells[level], level - 2), pred_r));
            f(level - 1, A::intersection(A::expand(A::on(cag[level], level - 1), pred_r), A::expand(subdomain_at(level - 1), pred_r)));
        }
        for (std::size_t level = in.reference.min_level(); level <= in.reference.max_level(); ++level)
        {
            f(level, A::intersection(in.reference[level], in.union_[level]));
        }
    }

    // What the library does.
    template <class B, std::size_t dim, class CA>
    auto projected_subdomain(const SiteInputs<dim, CA>& in)
    {
        return [&in](std::size_t level)
        {
            return Algebra<B>::on(in.subdomain(), level);
        };
    }

    // Not what the library does: the pyramid the mesh already stores.
    template <std::size_t dim, class CA>
    auto cached_subdomain(const SiteInputs<dim, CA>& in)
    {
        return [&in](std::size_t level) -> const auto&
        {
            return in.subdomain(level);
        };
    }

    // Mesh_base::construct_corners (mesh.hpp). Runs at every mesh
    // construction although its result only depends on the domain.
    template <class B, std::size_t dim, class CA, class F>
    void corners(const SiteInputs<dim, CA>& in, F&& f)
    {
        using A = Algebra<B>;
        if constexpr (dim > 1)
        {
            using direction_t      = samurai::DirectionVector<dim>;
            const auto& domain_lca = in.domain();
            samurai::for_each_diagonal_direction<dim>(
                [&](const auto& direction)
                {
                    std::array<direction_t, dim> shifts;
                    std::size_t n = 0;
                    for (std::size_t i = 0; i < dim; ++i)
                    {
                        if (direction[i] != 0)
                        {
                            shifts[n].fill(0);
                            shifts[n][i] = -direction[i];
                            ++n;
                        }
                    }
                    if (n == 2)
                    {
                        f(in.max_level,
                          A::difference(domain_lca, A::union_(A::translate(domain_lca, shifts[0]), A::translate(domain_lca, shifts[1]))));
                    }
                    else if constexpr (dim > 2)
                    {
                        f(in.max_level,
                          A::difference(domain_lca,
                                        A::union_(A::translate(domain_lca, shifts[0]),
                                                  A::translate(domain_lca, shifts[1]),
                                                  A::translate(domain_lca, shifts[2]))));
                    }
                });
        }
    }

    // update_ghost_mr (algorithm/update_ghost_mr.hpp): projection of the
    // ghosts, bottom-up.
    template <class B, std::size_t dim, class CA, class F>
    void ghost_projection(const SiteInputs<dim, CA>& in, F&& f)
    {
        using A = Algebra<B>;
        for (std::size_t level = in.max_level; level >= 1; --level)
        {
            f(level - 1, A::on(A::intersection(in.reference[level], in.proj_cells[level - 1]), level - 1));
        }
    }

    // update_ghost_mr: prediction of the ghosts, top-down.
    template <class B, std::size_t dim, class CA, class F>
    void ghost_prediction(const SiteInputs<dim, CA>& in, F&& f)
    {
        using A = Algebra<B>;
        for (std::size_t level = 1; level <= in.max_level; ++level)
        {
            auto pred_ghosts = A::difference(in.all_cells()[level], A::union_(in.cells[level], in.proj_cells[level]));
            f(level, A::on(A::intersection(pred_ghosts, in.subdomain(level), in.all_cells()[level - 1]), level));
        }
    }

    // domain_boundary(mesh, level, direction) (boundary.hpp), which calls
    // boundary_layer with a layer width of 1.
    template <class B, std::size_t dim, class CA, class Vector>
    auto domain_boundary(const SiteInputs<dim, CA>& in, std::size_t level, const Vector& direction)
    {
        using A                       = Algebra<B>;
        const std::size_t layer_width = 1;
        const auto& cells             = in.cells[level];
        return A::difference(cells, A::translate(A::on(in.domain(), level), -layer_width * direction));
    }

    // domain_boundary_outer_layer(mesh, level, direction, layer_width)
    // (boundary.hpp), which fills a LevelCellList: the builder of the
    // backend. Without the origin point and scaling factor of the mesh: they
    // play no part in the set algebra.
    template <class B, std::size_t dim, class CA>
    auto domain_boundary_outer_layer(const SiteInputs<dim, CA>& in,
                                     std::size_t level,
                                     const samurai::DirectionVector<dim>& direction,
                                     int layer_width)
    {
        using A = Algebra<B>;

        auto domain = A::lift(in.domain(level));

        auto outer_boundary = A::template builder<dim>(level);

        auto inner_boundary = domain_boundary<B>(in, level, direction);
        for (int layer = 1; layer <= layer_width; ++layer)
        {
            auto outer_layer = A::difference(A::translate(inner_boundary, layer * direction), domain);
            A::traverse(outer_layer,
                        [&](const auto& i, const auto& index)
                        {
                            outer_boundary.add_interval(i, index);
                        });
        }
        return outer_boundary.build();
    }

    // update_outer_ghosts (algorithm/update_outer_ghost.hpp) without the
    // boundary condition itself: project_corner_below, project_bc and
    // predict_bc, with one layer of boundary ghosts (Dirichlet). predict_bc
    // builds a LevelCellArray (domain_boundary_outer_layer) on every call.
    template <class B, std::size_t dim, class CA, class DomainAt, class F>
    void outer_ghosts(const SiteInputs<dim, CA>& in, DomainAt&& domain_at, F&& f)
    {
        using A                        = Algebra<B>;
        constexpr int n_bc_ghosts      = 1;
        const std::size_t lowest_level = in.min_level > 0 ? in.min_level - 1 : 0;
        for (std::size_t level = in.max_level + 1; level-- > lowest_level;)
        {
            if constexpr (dim > 1)
            {
                if (level >= in.min_level)
                {
                    samurai::for_each_diagonal_direction<dim>(
                        [&](const auto& direction)
                        {
                            for (std::size_t delta_l = 1; delta_l <= 2 && delta_l <= level; ++delta_l)
                            {
                                auto fine_inner_corner = A::on(in.corner(direction), level);
                                auto fine_outer_corner = A::intersection(A::translate(fine_inner_corner, direction), in.reference[level]);
                                f(level - delta_l, A::intersection(A::on(fine_outer_corner, level - delta_l), in.reference[level - delta_l]));
                            }
                        });
                }
            }
            samurai::for_each_cartesian_direction<dim>(
                [&](auto, const auto& direction)
                {
                    if (level < in.max_level)
                    {
                        auto outside_layer = A::difference(A::translate(in.union_[level], direction), domain_at(level));
                        f(level, A::on(A::intersection(outside_layer, in.reference[level]), level));
                    }
                    if (level < in.max_level && level >= in.min_level)
                    {
                        auto bc_ghosts = domain_boundary_outer_layer<B>(in, level, direction, n_bc_ghosts);
                        f(level + 1, A::on(A::intersection(bc_ghosts, in.reference[level + 1]), level + 1));
                    }
                });
        }
    }

    // What the library does.
    template <class B, std::size_t dim, class CA>
    auto projected_domain(const SiteInputs<dim, CA>& in)
    {
        return [&in](std::size_t level)
        {
            return Algebra<B>::on(in.domain(), level);
        };
    }

    // Not what the library does: the pyramid the mesh already stores.
    template <std::size_t dim, class CA>
    auto cached_domain(const SiteInputs<dim, CA>& in)
    {
        return [&in](std::size_t level) -> const auto&
        {
            return in.domain(level);
        };
    }

    // Adapt::harten (mr/adapt.hpp): detail computation. At the last level,
    // cells[level + 2] is cells[max_level + 1], as in the library: an empty
    // level past the mesh, which CellArray stores (it holds max_size + 1
    // levels), so the read is safe.
    template <class B, std::size_t dim, class CA, class F>
    void detail(const SiteInputs<dim, CA>& in, F&& f)
    {
        using A                     = Algebra<B>;
        const std::size_t min_level = in.min_level;
        for (std::size_t level = (min_level > 0) ? min_level - 1 : 0; level < in.max_level; ++level)
        {
            f(level, A::on(A::intersection(in.all_cells()[level], A::union_(in.cells[level + 1], in.cells[level + 2])), level));
        }
    }

    // Adapt::harten: tagging (mr_criteria) and tag finalization (maximum),
    // which evaluate the same set.
    template <class B, std::size_t dim, class CA, class F>
    void tag(const SiteInputs<dim, CA>& in, F&& f)
    {
        using A = Algebra<B>;
        for (std::size_t level = std::max<std::size_t>(in.min_level, 1); level <= in.max_level; ++level)
        {
            f(level - 1, A::on(A::intersection(in.cells[level], in.all_cells()[level - 1]), level - 1));
        }
    }

    // for_each_interface (interface.hpp): faces between cells of the same
    // level, then between a cell and a finer one.
    template <class B, std::size_t dim, class CA, class F>
    void interfaces(const SiteInputs<dim, CA>& in, F&& f)
    {
        using A               = Algebra<B>;
        const auto& cells     = in.cells;
        const auto directions = samurai::positive_cartesian_directions<dim>();
        for (std::size_t d = 0; d < dim; ++d)
        {
            const samurai::DirectionVector<dim> direction = xt::view(directions, d);
            for (std::size_t level = cells.min_level(); level <= cells.max_level(); ++level)
            {
                f(level, A::intersection(cells[level], A::translate(cells[level], -direction)));
                if (level < cells.max_level())
                {
                    f(level + 1, A::on(A::intersection(cells[level], A::translate(cells[level + 1], -direction)), level + 1));
                    f(level + 1, A::on(A::intersection(cells[level], A::translate(cells[level + 1], direction)), level + 1));
                }
            }
        }
    }

    // update_fields (algorithm/update_fields.hpp): copy of the unchanged
    // cells, then coarsening and refinement between the old and new meshes.
    template <class B, std::size_t dim, class CA, class F>
    void field_transfer(const SiteInputs<dim, CA>& in, const SiteInputs<dim, CA>& new_in, F&& f)
    {
        using A = Algebra<B>;
        for (std::size_t level = in.min_level; level <= in.max_level; ++level)
        {
            f(level, A::intersection(in.reference[level], new_in.cells[level]));
        }
        for (std::size_t level = in.min_level + 1; level <= in.max_level; ++level)
        {
            f(level - 1, A::on(A::intersection(in.cells[level], new_in.cells[level - 1]), level - 1));
            f(level - 1, A::on(A::intersection(new_in.cells[level], in.cells[level - 1]), level - 1));
        }
    }
} // namespace bench::sites
