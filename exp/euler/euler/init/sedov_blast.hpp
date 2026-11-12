// Copyright 2025 the samurai team
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <cmath>
#include <numbers>

#include <samurai/box.hpp>

#include "../bc.hpp"
#include "../eos.hpp"
#include "../variables.hpp"
#include "registry.hpp"

// =============================================================================
//  Sedov blast wave
// -----------------------------------------------------------------------------
//  A fixed amount of energy is deposited in a small ball at the centre of an
//  ambient gas at rest, driving a strong self-similar shock. The exact solution
//  is radially symmetric, which makes it the canonical stress test for an AMR
//  solver: any anisotropy in adaptation, prediction or flux reconstruction shows
//  up immediately as a broken symmetry.
//
//  The definition is dimension agnostic — only the volume over which the energy
//  is spread changes — so the same case serves euler_1d, euler_2d and euler_3d.
//
//  Reference solution
//  ------------------
//      L.I. Sedov, "Similarity and Dimensional Methods in Mechanics", Academic
//      Press, New York, 1959, chapter IV.
//      J.R. Kamm, "Evaluation of the Sedov-von Neumann-Taylor Blast Wave
//      Solution", report LA-UR-00-6055, Los Alamos National Laboratory, 2000.
//      J.R. Kamm, F.X. Timmes, "On Efficient Generation of Numerically Robust
//      Sedov Solutions", report LA-UR-07-2849, Los Alamos National Laboratory,
//      2007.
//
//  The similarity solution puts the shock at
//
//      r_s(t) = ( E t^2 / (alpha rho_0) )^(1/(nu+2)),
//
//  nu = 1, 2, 3 being the geometry and alpha the dimensionless energy of the
//  solution, a number that depends on nu and gamma alone. Integrating the
//  similarity equations for gamma = 1.4 gives
//
//      alpha = 1.077485 (planar, energy counted on both sides of the plane)
//              0.984074 (cylindrical)
//              0.851072 (spherical)
//
//  python/sedov_exact.py computes these constants and the radius they give,
//  and the validation suite measures the computed shock against it.
// =============================================================================

namespace test_case::sedov_blast
{
    inline constexpr double rho_ambient = 1.0;  // ambient density
    inline constexpr double p_ambient   = 1e-5; // ambient pressure (very small)
    inline constexpr double r_blast     = 0.1;  // blast radius

    // Blast energy, one per geometry: the standard cases of the Kamm report,
    // which fix E so that the shock sits at r = 0.5, 0.75 and 1 at t = 1 for
    // gamma = 1.4 and rho_0 = 1. The spherical value is the one clawpack and
    // lanl/HARD use as well, and equals alpha, spherical being the geometry
    // whose shock is normalised to r = 1.
    //
    // The planar value is twice Kamm's 0.0673185 because his planar blast
    // releases its energy on one side of the plane, while the deposit below
    // straddles the origin and feeds two shocks. Doubling it gives each shock
    // his energy, and puts both where his solution puts its own, at 0.5.
    template <std::size_t dim>
    constexpr double blast_energy()
    {
        static_assert(dim >= 1 && dim <= 3, "no blast energy tabulated for this dimension");

        if constexpr (dim == 1)
        {
            return 0.134637; // = 2 x 0.0673185, see above
        }
        else if constexpr (dim == 2)
        {
            return 0.311357;
        }
        else
        {
            return 0.851072;
        }
    }

    // Measure of the region the energy is deposited in: a segment in 1D, a disk
    // in 2D, a ball in 3D. Spelled out for every dimension rather than left to a
    // trailing else, which would silently hand a new dimension the 3D answer.
    template <std::size_t dim>
    constexpr double blast_volume()
    {
        static_assert(dim >= 1 && dim <= 3, "no blast volume for this dimension");

        if constexpr (dim == 1)
        {
            return 2. * r_blast;
        }
        else if constexpr (dim == 2)
        {
            return std::numbers::pi * r_blast * r_blast;
        }
        else
        {
            return 4. / 3. * std::numbers::pi * r_blast * r_blast * r_blast;
        }
    }

    template <class Field>
    void init_fn(Field& u, const typename Field::cell_t& cell, EOS::IdealGas eos)
    {
        static constexpr std::size_t dim = Field::dim;
        using EulerConsVar               = EulerLayout<dim>;

        const auto x = cell.center();

        // Domain is centred on the origin, so the blast centre is the origin.
        double r2 = 0.;
        for (std::size_t d = 0; d < dim; ++d)
        {
            r2 += x[d] * x[d];
        }

        const double p = (r2 < r_blast * r_blast) ? (eos.gamma - 1.0) * blast_energy<dim>() / blast_volume<dim>() // blast zone
                                                  : p_ambient;                                                    // ambient zone

        u[cell][EulerConsVar::rho]  = rho_ambient;
        u[cell][EulerConsVar::rhoE] = rho_ambient * eos.e(rho_ambient, p);
        for (std::size_t d = 0; d < dim; ++d)
        {
            u[cell][EulerConsVar::mom(d)] = 0.; // gas initially at rest
        }
    }

    template <class Field>
    void bc_fn(Field& u, double& /*t*/, EOS::IdealGas /*eos*/)
    {
        bc::outflow(u);
    }

    template <std::size_t dim>
    auto box_fn()
    {
        xt::xtensor_fixed<double, xt::xshape<dim>> min_corner;
        xt::xtensor_fixed<double, xt::xshape<dim>> max_corner;
        min_corner.fill(-1.);
        max_corner.fill(1.);

        return samurai::Box<double, dim>(min_corner, max_corner);
    }

    template <class Field>
    test_case::TestCase<Field> definition()
    {
        return {.box = &box_fn<Field::dim>, .init = &init_fn<Field>, .bc = &bc_fn<Field>, .eos = EOS::ideal_gas(1.4)};
    }
}

REGISTER_TEST_CASE(sedov_blast, test_case::sedov_blast, 1, 2, 3)
