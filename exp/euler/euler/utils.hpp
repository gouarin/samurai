// Copyright 2025 the samurai team
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <type_traits>

#ifdef SAMURAI_WITH_MPI
#include <boost/mpi.hpp>
namespace mpi = boost::mpi;
#endif

#include <samurai/algorithm.hpp>

#include "variables.hpp"

// The largest wave speed in the mesh, taken as the largest of |v_d| + c over
// the cells AND over the directions, which is what the time step is built on.
//
// NOTE that is the one-dimensional bound. An unsplit scheme sums the flux
// divergence of every direction into one update, and its stability limit is
// nearer CFL / dim than CFL: at --cfl 0.4 in three dimensions, --time-integrator
// ssprk2 is already above it, and sum_d (|v_d| + c) would be the honest
// denominator there. A directional sweep is a one-dimensional problem and is
// not concerned, which is one more reason for Strang to be the default.
auto get_max_lambda(const auto& u, auto eos)
{
    static constexpr std::size_t dim = std::decay_t<decltype(u)>::dim;
    double res                       = 0.;

    const auto& mesh = u.mesh();

    samurai::for_each_cell(mesh,
                           [&](const auto& cell)
                           {
                               auto prim = cons2prim<dim>(u[cell], eos);

                               auto c = eos.c(prim.rho, prim.p);
                               for (std::size_t d = 0; d < dim; ++d)
                               {
                                   res = std::max(std::abs(prim.v[d]) + c, res);
                               }
                           });
#ifdef SAMURAI_WITH_MPI
    double global_res;
    mpi::communicator world;
    mpi::all_reduce(world, res, global_res, mpi::maximum<double>());
    return global_res;
#else
    return res;
#endif
}

void check_positive_density(const auto& u)
{
    static constexpr std::size_t dim = std::decay_t<decltype(u)>::dim;

    const auto& mesh = u.mesh();

    samurai::for_each_cell(mesh,
                           [&](const auto& cell)
                           {
                               double rho = u[cell][EulerLayout<dim>::rho];
                               if (rho <= 0.)
                               {
                                   throw std::runtime_error("Negative density detected");
                               }
                           });
}

void check_positive_pressure(const auto& u, auto eos)
{
    static constexpr std::size_t dim = std::decay_t<decltype(u)>::dim;

    const auto& mesh = u.mesh();

    samurai::for_each_cell(mesh,
                           [&](const auto& cell)
                           {
                               double rho   = u[cell][EulerLayout<dim>::rho];
                               double e     = u[cell][EulerLayout<dim>::rhoE] / rho;
                               double norm2 = 0.;
                               for (std::size_t d = 0; d < dim; ++d)
                               {
                                   double v = u[cell][EulerLayout<dim>::mom(d)] / rho;
                                   norm2 += v * v;
                               }
                               double p = eos.p(rho, e - 0.5 * norm2);
                               if (p <= 0.)
                               {
                                   throw std::runtime_error("Negative pressure detected");
                               }
                           });
}

void check(const auto& u, auto eos)
{
    check_positive_density(u);
    check_positive_pressure(u, eos);
}
