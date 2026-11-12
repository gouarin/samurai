// Copyright 2025 the samurai team
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <samurai/box.hpp>

#include "../bc.hpp"
#include "../variables.hpp"
#include "registry.hpp"

// =============================================================================
//  Free-stream preservation (mesh adaptation sanity check)
// -----------------------------------------------------------------------------
//  A uniform flow (constant density, pressure and velocity) is an exact, trivial
//  solution of the Euler equations: it must stay strictly uniform for all time.
//
//  On an adapted (multiresolution) mesh this is NOT automatic: any inconsistency
//  in the prediction / projection operators or in the flux reconstruction at
//  level interfaces breaks the uniform state and shows up immediately as a
//  non-zero L1/Linf error. This makes it the cheapest, sharpest test to validate
//  the AMR machinery (free-stream preservation).
//
//  The exact solution at any time is the same constant state, so the error is
//  obtained by comparing to the initial uniform values.
//
//  No paper to compare this against: free-stream preservation is a property of
//  the discretisation, so it is checked against machine precision rather than
//  against a published figure.
// =============================================================================

namespace test_case::free_stream
{
    inline constexpr double rho = 1.; // density
    inline constexpr double p   = 1.; // pressure
    inline constexpr double v   = 1.; // velocity, same on every axis (non-zero on purpose)

    template <std::size_t dim>
    auto uniform_state()
    {
        PrimState<dim> state{rho, p, {}};
        state.v.fill(v);
        return state;
    }

    template <class Field>
    void init_fn(Field& u, const typename Field::cell_t& cell, EOS::IdealGas eos)
    {
        u[cell] = prim2cons<Field::dim>(uniform_state<Field::dim>(), eos);
    }

    template <class Field>
    void bc_fn(Field& u, double& /*t*/, EOS::IdealGas eos)
    {
        // Impose the exact uniform state on every boundary.
        bc::imposed(u, uniform_state<Field::dim>(), eos);
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

REGISTER_TEST_CASE(free_stream, test_case::free_stream, 1, 2, 3)
