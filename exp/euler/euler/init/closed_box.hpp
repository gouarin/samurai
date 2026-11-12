// Copyright 2025 the samurai team
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <samurai/box.hpp>

#include "../bc.hpp"
#include "../variables.hpp"
#include "registry.hpp"

// =============================================================================
//  Uniform flow in a closed box
// -----------------------------------------------------------------------------
//  A uniform gas moving diagonally, enclosed by solid walls. It is not a steady
//  state: the flow piles up against the downstream walls and reflects, so the
//  solution develops real structure within a few time steps.
//
//  The case exists for what the walls guarantee rather than for the flow. No
//  mass and no energy cross a reflecting boundary, so both totals stay constant
//  to round-off whatever the scheme does inside. That makes it the cheapest
//  check that a flux is conservative, which is the first thing a reconstruction
//  can quietly break, and it is the only case that exercises the Reflective
//  boundary condition.
//
//  Like the free stream, this case has no publication behind it: what it checks
//  is a conservation identity of the scheme, held to round-off.
// =============================================================================

namespace test_case::closed_box
{
    inline constexpr double rho = 1.; // density
    inline constexpr double p   = 1.; // pressure
    inline constexpr double v   = 1.; // velocity, same on every axis

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
    void bc_fn(Field& u, double& /*t*/, EOS::IdealGas /*eos*/)
    {
        bc::wall(u);
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

REGISTER_TEST_CASE(closed_box, test_case::closed_box, 1, 2, 3)
