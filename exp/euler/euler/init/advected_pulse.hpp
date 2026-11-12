// Copyright 2025 the samurai team
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <cmath>

#include <samurai/box.hpp>

#include "../bc.hpp"
#include "../variables.hpp"
#include "registry.hpp"

// =============================================================================
//  Advected density pulse
// -----------------------------------------------------------------------------
//  A smooth bump of density carried by a uniform flow, at uniform pressure.
//  With p and u constant the Euler system collapses onto
//
//      rho_t + u rho_x = 0,
//
//  so the exact solution is the initial profile translated by u t, undeformed.
//  The bump travels on a contact wave, which carries no pressure signal.
//
//  It is the one-dimensional counterpart of the isentropic vortex, and exists
//  for the same reason: an order can only be measured on a solution that is
//  smooth and known. The vortex is two-dimensional by nature, and one
//  dimension is where the Hancock predictor is expected to be second order.
//
//  The pulse is narrow enough, and started far enough from the boundaries, that
//  it never reaches them: the imposed ambient state at either end is the exact
//  solution there to better than 1e-12 for the whole run.
// =============================================================================

namespace test_case::advected_pulse
{
    inline constexpr double rho_ambient = 1.0;  // density away from the pulse
    inline constexpr double amplitude   = 0.5;  // density added at the centre
    inline constexpr double sigma       = 0.04; // width of the bump
    inline constexpr double x0          = 0.3;  // where it starts
    inline constexpr double p0          = 1.0;  // uniform pressure
    inline constexpr double u0          = 1.0;  // uniform velocity, along x

    // The profile at time t, which is the profile at t = 0 translated by u0 t.
    inline double density(double x, double t)
    {
        const double s = x - x0 - u0 * t;
        return rho_ambient + amplitude * std::exp(-0.5 * s * s / (sigma * sigma));
    }

    template <std::size_t dim>
    PrimState<dim> ambient_state()
    {
        PrimState<dim> state{rho_ambient, p0, {}};
        state.v.fill(0.);
        state.v[0] = u0;
        return state;
    }

    template <class Field>
    void init_fn(Field& u, const typename Field::cell_t& cell, EOS::IdealGas eos)
    {
        static constexpr std::size_t dim = Field::dim;

        auto state = ambient_state<dim>();
        state.rho  = density(cell.center(0), 0.);

        u[cell] = prim2cons<dim>(state, eos);
    }

    template <class Field>
    void bc_fn(Field& u, double& /*t*/, EOS::IdealGas eos)
    {
        bc::imposed(u, ambient_state<Field::dim>(), eos);
    }

    template <std::size_t dim>
    auto box_fn()
    {
        xt::xtensor_fixed<double, xt::xshape<dim>> min_corner;
        xt::xtensor_fixed<double, xt::xshape<dim>> max_corner;
        min_corner.fill(0.);
        max_corner.fill(1.);

        return samurai::Box<double, dim>(min_corner, max_corner);
    }

    template <class Field>
    test_case::TestCase<Field> definition()
    {
        return {.box = &box_fn<Field::dim>, .init = &init_fn<Field>, .bc = &bc_fn<Field>, .eos = EOS::ideal_gas(1.4)};
    }
}

REGISTER_TEST_CASE(advected_pulse, test_case::advected_pulse, 1, 2, 3)
