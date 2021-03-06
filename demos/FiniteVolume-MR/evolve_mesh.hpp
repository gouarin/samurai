// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <samurai/samurai.hpp>
#include <samurai/mr/mesh.hpp>
#include "criteria.hpp"


template <class Config>
bool evolve_mesh(samurai::Field<Config> &u, double eps, std::size_t ite)
{
    constexpr auto dim = Config::dim;
    constexpr auto max_refinement_level = Config::max_refinement_level;

    using interval_t = typename Config::interval_t;
    auto mesh = u.mesh();
    std::size_t min_level = mesh.min_level(), max_level = mesh.max_level();
    samurai::Field<Config> detail{"detail", mesh};
    samurai::Field<Config, int> tag{"tag", mesh};

    tag.array().fill(0);

    mesh.for_each_cell([&](auto &cell) {
        tag[cell] = static_cast<int>(samurai::CellFlag::keep);
    });

    // ARE WE SURE THEY DO EXACTLY WHAT WE WANT???
    samurai::mr_projection(u);
    samurai::mr_prediction(u);

    u.update_bc();
    // {

    //     std::stringstream s;
    //     s << "debugproj_"<<ite;
    //     auto h5file = samurai::Hdf5(s.str().data());
    //     h5file.add_field_by_level(mesh, u);

    // }

    xt::xtensor_fixed<double, xt::xshape<max_refinement_level + 1>>max_detail;
    max_detail.fill(std::numeric_limits<double>::min());



    // What are the data it uses at min_level - 1 ???
    for (std::size_t level = min_level - 1; level < max_level - ite; ++level)   {
        auto subset = intersection(mesh[samurai::MeshType::all_cells][level],
                                   mesh[samurai::MeshType::cells][level + 1])
                                .on(level);
        subset.apply_op(compute_detail(detail, u),
                               compute_max_detail(detail, max_detail));
    }



    // std::stringstream s;
    // s << "evolve_mesh_"<<ite;
    // auto h5file = samurai::Hdf5(s.str().data());
    // h5file.add_mesh(mesh);
    // h5file.add_field(detail);
    // h5file.add_field(u);


    // AGAIN I DONT KNOW WHAT min_level - 1 is
    for (std::size_t level = min_level; level <= max_level - ite; ++level)
    {
        //int exponent = dim * (level - max_level + 1);

        int exponent = dim * (level - max_level);

        auto eps_l = std::pow(2, exponent) * eps;

        // COMPRESSION

        auto subset_1 = samurai::intersection(mesh[samurai::MeshType::cells][level],
                                         mesh[samurai::MeshType::all_cells][level-1])
                                        .on(level-1);


        // This operations flags the cells to coarsen
        subset_1.apply_op(to_coarsen_mr(detail, max_detail, tag, eps_l, min_level));

        auto subset_2 = intersection(mesh[samurai::MeshType::cells][level],
                                     mesh[samurai::MeshType::cells][level]);
        auto subset_3 = intersection(mesh[samurai::MeshType::cells_and_ghosts][level],
                                     mesh[samurai::MeshType::cells_and_ghosts][level]);

        subset_2.apply_op(samurai::enlarge(tag, samurai::CellFlag::keep));
        subset_3.apply_op(samurai::tag_to_keep(tag));

        auto subset_4 = samurai::intersection(mesh[samurai::MeshType::cells][level],
                                           mesh[samurai::MeshType::cells][level])
                       .on(level);

        subset_4.apply_op(to_refine_mr(detail, max_detail, tag, 8 * eps_l, max_level));

    }

    //h5file.add_field(tag);


    // FROM NOW ON LOIC HAS TO EXPLAIN

    for (std::size_t level = max_level; level > min_level; --level)
    {

        auto subset_1 = intersection(mesh[samurai::MeshType::cells][level],
                                     mesh[samurai::MeshType::cells][level]);

        subset_1.apply_op(extend(tag));



        auto keep_subset =
            intersection(mesh[samurai::MeshType::cells][level],
                         mesh[samurai::MeshType::all_cells][level - 1])
                .on(level - 1);
        keep_subset.apply_op(maximum(tag));
        xt::xtensor_fixed<int, xt::xshape<dim>> stencil;
        for (std::size_t d = 0; d < dim; ++d)
        {
            for (std::size_t d1 = 0; d1 < dim; ++d1)
                stencil[d1] = 0;
            for (int s = -1; s <= 1; ++s)
            {
                if (s != 0)
                {
                    stencil[d] = s;
                    auto subset =
                        intersection(
                            mesh[samurai::MeshType::cells][level],
                            translate(
                                mesh[samurai::MeshType::cells][level - 1], stencil))
                            .on(level - 1);
                    subset.apply_op(balance_2to1(tag, stencil));

                    auto subset_bis =
                            intersection(translate(mesh[samurai::MeshType::cells][level], stencil),
                                                   mesh[samurai::MeshType::cells][level-1])
                                .on(level);

                    subset_bis.apply_op(make_graduation(tag));
                }
            }
        }
    }

    samurai::CellList<Config> cell_list;
    for (std::size_t level = min_level; level <= max_level; ++level)
    {
        auto level_cell_array = mesh[samurai::MeshType::cells][level];
        if (!level_cell_array.empty())
        {
            level_cell_array.for_each_interval_in_x([&](auto const &index_yz, auto const &interval) {
                for (int i = interval.start; i < interval.end; ++i)
                {
                    if (tag.array()[i + interval.index] & static_cast<int>(samurai::CellFlag::refine))
                    {
                        samurai::static_nested_loop<dim - 1, 0, 2>(
                            [&](auto stencil) {
                                auto index = 2 * index_yz + stencil;
                                cell_list[level + 1][index].add_point(2 * i);
                                cell_list[level + 1][index].add_point(2 * i + 1);
                            });
                    }
                    else if (tag.array()[i + interval.index] & static_cast<int>(samurai::CellFlag::keep))
                        {
                            cell_list[level][index_yz].add_point(i);
                        }
                        else
                        {
                            cell_list[level-1][index_yz>>1].add_point(i>>1);
                        }
                    }
                });
        }
    }

    samurai::Mesh<Config> new_mesh{cell_list, mesh.initial_mesh(),
                            min_level, max_level};

    if (new_mesh == mesh)
        return true;


    samurai::Field<Config> new_u{u.name(), new_mesh, u.bc()};

    for (std::size_t level = min_level; level <= max_level; ++level)
    {
        auto subset = samurai::intersection(mesh[samurai::MeshType::all_cells][level],
                                   new_mesh[samurai::MeshType::cells][level]);
        subset.apply_op(copy(new_u, u));
    }

    for (std::size_t level = min_level; level < max_level; ++level)
    {
        auto level_cell_array = mesh[samurai::MeshType::cells][level];

        if (!level_cell_array.empty())
        {
            level_cell_array.for_each_interval_in_x(
                [&](auto const &index_yz, auto const &interval) {
                    for (int i = interval.start; i < interval.end; ++i)
                    {
                        if (tag.array()[i + interval.index] &
                            static_cast<int>(samurai::CellFlag::refine))
                        {
                            samurai::compute_new_u(level, interval_t{i, i+1}, index_yz, u, new_u);
                        }
                    }
                });
        }
    }




    u.mesh_ptr()->swap(new_mesh);
    std::swap(u.array(), new_u.array());

    return false;
 }