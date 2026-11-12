// Copyright 2025 the samurai team
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <samurai/field.hpp>
#include <samurai/io/hdf5.hpp>

#include "variables.hpp"

void save(const std::string& path, const std::string& filename, const auto& field, auto eos)
{
    static constexpr std::size_t dim = std::decay_t<decltype(field)>::dim;
    auto& mesh                       = field.mesh();
    auto rho                         = samurai::make_scalar_field<double>("rho", field.mesh());
    auto pressure                    = samurai::make_scalar_field<double>("pressure", field.mesh());
    auto velocity                    = samurai::make_vector_field<double, dim>("velocity", field.mesh());

    samurai::for_each_cell(mesh,
                           [&](auto& cell)
                           {
                               auto prim      = cons2prim<dim>(field[cell], eos);
                               rho[cell]      = prim.rho;
                               pressure[cell] = prim.p;
                               for (std::size_t d = 0; d < dim; ++d)
                               {
                                   velocity[cell][d] = prim.v[d];
                               }
                           });

    samurai::save(path, filename, mesh, rho, pressure, velocity);
}
