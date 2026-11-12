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
//  Sod shock tube
// -----------------------------------------------------------------------------
//  The classical 1D Riemann problem, in two layouts that share their states:
//
//    - `sod_x`, along the x axis, which is the tube as the article this
//      repository reproduces runs it (its fig. 7) and the one to hold against
//      the exact solution, no projection in the way. It is dimension agnostic:
//      with the velocity zero and the discontinuity normal to x, the same
//      definition serves euler_1d, euler_2d and euler_3d.
//
//    - `sod`, laid out along a diagonal of a 2D domain so that the solution is
//      not aligned with the mesh. Any directional bias in the scheme or in the
//      adaptation shows up as a distortion of what should stay a planar wave.
//      The rotation is ours, not Sod's, and not the article's.
//
//  The states are those of
//
//      G.A. Sod, "A survey of several finite difference methods for systems of
//      nonlinear hyperbolic conservation laws", J. Comput. Phys. 27 (1) (1978)
//      1-31, https://doi.org/10.1016/0021-9991(78)90023-2
//
//  reprinted as test 1 of Table 4.1 of
//
//      E.F. Toro, "Riemann Solvers and Numerical Methods for Fluid Dynamics.
//      A Practical Introduction", 3rd ed., Springer, 2009,
//      https://doi.org/10.1007/b79761
//
//  whose exact solution python/exact_riemann.py computes: p* = 0.30313,
//  u* = 0.92745.
// =============================================================================

namespace test_case::sod
{
    using field_t = config<2>::field_t;

    // The two states of the tube. Written once, in any dimension: both layouts
    // below are the same Riemann problem and must not be able to drift apart.
    template <std::size_t dim>
    PrimState<dim> left_state()
    {
        PrimState<dim> state{1., 1., {}};
        state.v.fill(0.);
        return state;
    }

    template <std::size_t dim>
    PrimState<dim> right_state()
    {
        PrimState<dim> state{0.125, 0.1, {}};
        state.v.fill(0.);
        return state;
    }

    // Where the diaphragm sits, the same fraction of the domain in both layouts.
    inline constexpr double x_diaphragm = 0.5;

    // The diagonal of `sod`, and the line the diaphragm sits on.
    inline const double theta = std::numbers::pi / 4.;

    inline const double Rdx = std::sin(theta);
    inline const double Rdy = std::cos(theta);
    inline const double k   = x_diaphragm / Rdy;
    inline const double x0  = x_diaphragm + k * Rdx;

    inline void init_fn(field_t& u, const typename field_t::cell_t& cell, EOS::IdealGas eos)
    {
        const auto x = cell.center();

        const double y_theta = (x0 - x[0]) * Rdy / Rdx;

        if (x[1] < y_theta)
        {
            u[cell] = prim2cons<2>(left_state<2>(), eos);
        }
        else
        {
            u[cell] = prim2cons<2>(right_state<2>(), eos);
        }
    }

    inline void bc_fn(field_t& u, double& /*t*/, EOS::IdealGas /*eos*/)
    {
        bc::outflow(u);
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
        static_assert(Field::dim == 2, "this test case is two-dimensional");
        return {.box = &box_fn<2>, .init = &init_fn, .bc = &bc_fn, .eos = EOS::ideal_gas(1.4)};
    }
}

// The axis-aligned tube, which is the one the article runs and the one the
// exact solution is compared against. It reuses everything above but the
// geometry of the diaphragm.
namespace test_case::sod_x
{
    template <class Field>
    void init_fn(Field& u, const typename Field::cell_t& cell, EOS::IdealGas eos)
    {
        static constexpr std::size_t dim = Field::dim;

        if (cell.center(0) < sod::x_diaphragm)
        {
            u[cell] = prim2cons<dim>(sod::left_state<dim>(), eos);
        }
        else
        {
            u[cell] = prim2cons<dim>(sod::right_state<dim>(), eos);
        }
    }

    template <class Field>
    void bc_fn(Field& u, double& /*t*/, EOS::IdealGas /*eos*/)
    {
        bc::outflow(u);
    }

    template <class Field>
    test_case::TestCase<Field> definition()
    {
        return {.box = &sod::box_fn<Field::dim>, .init = &init_fn<Field>, .bc = &bc_fn<Field>, .eos = EOS::ideal_gas(1.4)};
    }
}

REGISTER_TEST_CASE(sod, test_case::sod, 2)
REGISTER_TEST_CASE(sod_x, test_case::sod_x, 1, 2, 3)
