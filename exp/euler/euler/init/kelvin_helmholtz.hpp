// Copyright 2025 the samurai team
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <cmath>
#include <numbers>

#include <samurai/box.hpp>

#include "../bc.hpp"

#include "../variables.hpp"
#include "registry.hpp"

// =============================================================================
//  Kelvin-Helmholtz instability
// -----------------------------------------------------------------------------
//  Two horizontal layers in shear. A small vertical-velocity perturbation
//  localized at the two interfaces grows into the characteristic rolled-up
//  billows.
//
//  The layer states, the shear and the seeded perturbation follow the widely
//  reproduced set-up of
//
//      V. Springel, "E pur si muove: Galilean-invariant cosmological
//      hydrodynamical simulations on a moving mesh", Mon. Not. R. Astron. Soc.
//      401 (2) (2010) 791-851,
//      https://doi.org/10.1111/j.1365-2966.2009.15715.x
//
//  with two departures from it, which matter as soon as a figure here is
//  compared to a published one:
//
//    - gamma is 1.4 here against 5/3 there, so that the case shares the
//      equation of state of most other cases in this repository;
//    - the boundaries are outflow rather than periodic. The registry carries
//      periodicity per axis and blast_periodic uses it, so this departure could
//      be closed; it has not been, because closing it moves the reference this
//      case is held to and tests nothing the periodic blast does not.
//
//  The interface is a discontinuity, which makes the growth of the billows
//  depend on the resolution rather than converge to one answer:
//
//      C.P. McNally, W. Lyra, J.-C. Passy, "A well-posed Kelvin-Helmholtz
//      instability test and comparison", Astrophys. J. Suppl. Ser. 201 (2)
//      (2012) 18, https://doi.org/10.1088/0067-0049/201/2/18
//
//  So the case is qualitative: it shows the scheme developing the instability
//  and the adaptation following it. No number should be read off it. What earns
//  it a place in the non-regression suite is being a moving, structured,
//  adapted solution.
// =============================================================================

namespace test_case::kelvin_helmholtz
{
    using field_t = config<2>::field_t;

    constexpr double pi = std::numbers::pi;

    // Layer states
    inline constexpr double rho_in  = 2.0; // inner layer (0.25 < y < 0.75)
    inline constexpr double rho_out = 1.0; // outer layers
    inline constexpr double v_shear = 0.5; // horizontal shear velocity (+/-)
    inline constexpr double p0      = 2.5; // uniform pressure

    // Perturbation
    inline constexpr double amp   = 0.1;  // amplitude of the seeded vertical velocity
    inline constexpr double sigma = 0.05; // interface thickness of the seed
    inline constexpr int mode     = 2;    // number of billows (wavenumber = 2*mode)

    inline void init_fn(field_t& u, const typename field_t::cell_t& cell, EOS::IdealGas eos)
    {
        const auto c   = cell.center();
        const double x = c[0];
        const double y = c[1];

        const bool inner = (y > 0.25 && y < 0.75);
        const double rho = inner ? rho_in : rho_out;
        const double vx  = inner ? v_shear : -v_shear;

        const double seed = std::exp(-(y - 0.25) * (y - 0.25) / (2 * sigma * sigma))
                          + std::exp(-(y - 0.75) * (y - 0.75) / (2 * sigma * sigma));
        const double vy   = amp * std::sin(2 * mode * pi * x) * seed;

        PrimState<2> state{
            rho,
            p0,
            xt::xtensor_fixed<double, xt::xshape<2>>{vx, vy}
        };
        u[cell] = prim2cons<2>(state, eos);
    }

    inline void bc_fn(field_t& u, double& /*t*/, EOS::IdealGas /*eos*/)
    {
        bc::outflow(u);
    }

    template <std::size_t dim>
    auto box_fn()
    {
        xt::xtensor_fixed<double, xt::xshape<dim>> min_corner = {0., 0.};
        xt::xtensor_fixed<double, xt::xshape<dim>> max_corner = {1., 1.};
        return samurai::Box<double, dim>(min_corner, max_corner);
    }

    template <class Field>
    test_case::TestCase<Field> definition()
    {
        static_assert(Field::dim == 2, "this test case is two-dimensional");
        return {.box = &box_fn<2>, .init = &init_fn, .bc = &bc_fn, .eos = EOS::ideal_gas(1.4)};
    }
}

REGISTER_TEST_CASE(kelvin_helmholtz, test_case::kelvin_helmholtz, 2)
