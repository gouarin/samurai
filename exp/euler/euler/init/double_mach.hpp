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
//  Double Mach reflection
// -----------------------------------------------------------------------------
//  A Mach 10 shock hits a 30 degree ramp. The ramp is represented by a reflecting
//  wall along the bottom boundary starting at x0, the shock entering upstream of
//  it. The top boundary follows the analytic shock position, which is why the
//  boundary condition needs the current time.
//
//  The set-up is the one of
//
//      P. Woodward, P. Colella, "The numerical simulation of two-dimensional
//      fluid flow with strong shocks", J. Comput. Phys. 54 (1) (1984) 115-173,
//      https://doi.org/10.1016/0021-9991(84)90142-6
//
//  section IVa: domain [0,4] x [0,1], wall from x0 = 1/6, final time 0.2.
//
//  The post-shock state below follows from the Rankine-Hugoniot conditions for
//  a Mach 10 shock running into (rho, p) = (1.4, 1), a gas whose sound speed is
//  therefore 1: density 1.4 (2.4 M^2)/(0.4 M^2 + 2) = 8, pressure
//  (2.8 M^2 - 0.4)/2.4 = 116.5, speed (1 - 1.4/8) x 10 = 8.25.
// =============================================================================

namespace test_case::double_mach_reflection
{
    using field_t = config<2>::field_t;

    inline const double alpha = std::numbers::pi / 3.;
    inline const double x0    = 1. / 6;

    inline const PrimState<2> left_state{
        8.,
        116.5,
        xt::xtensor_fixed<double, xt::xshape<2>>{8.25 * std::sin(alpha), -8.25 * std::cos(alpha)}
    };

    inline const PrimState<2> right_state{
        1.4,
        1.,
        xt::xtensor_fixed<double, xt::xshape<2>>{0., 0.}
    };

    inline void init_fn(field_t& u, const typename field_t::cell_t& cell, EOS::IdealGas eos)
    {
        const auto x = cell.center();

        if (x[0] < x0 + x[1] / std::tan(alpha))
        {
            u[cell] = prim2cons<2>(left_state, eos);
        }
        else
        {
            u[cell] = prim2cons<2>(right_state, eos);
        }
    }

    // The bottom boundary is two conditions in one: upstream of the ramp foot
    // the post-shock state flows in, downstream of it the boundary is a wall.
    // Expressing that as two conditions restricted by coordinates is what
    // samurai's CoordsRegion is for, but that path throws on this non-square
    // domain, so the two live in one condition here.
    //
    // It is a condition rather than a value function because a value function
    // is handed one cell and produces one value, which every ghost layer then
    // repeats. A wall is a mirror: the k-th ghost reflects the k-th cell
    // inside, so with a single value the second layer of a MUSCL stencil would
    // carry the reflection of the first cell instead of the second. The
    // reconstruction would then read a zero slope in the ghost, and the wall
    // would fall back to first order along the stretch where the jet runs.
    template <std::size_t StencilSize, class Field>
    struct RampBottomImpl : public samurai::Bc<Field>
    {
        INIT_BC(RampBottomImpl, StencilSize)

        apply_function_t get_apply_function(constant_stencil_size_t, const direction_t&) const override
        {
            return [](Field& u, const stencil_cells_t& cells, const value_t& inflow)
            {
                static constexpr std::size_t ghost0 = StencilSize / 2;

                if (cells[ghost0 - 1].center(0) < x0)
                {
                    for (std::size_t i = ghost0; i < StencilSize; ++i)
                    {
                        u[cells[i]] = inflow;
                    }
                    return;
                }

                for (std::size_t k = 0; k < ghost0; ++k)
                {
                    const auto& inside = cells[ghost0 - 1 - k];
                    const auto& ghost  = cells[ghost0 + k];

                    u[ghost]                                  = u[inside];
                    u[ghost][EulerLayout<Field::dim>::mom(1)] = -u[inside][EulerLayout<Field::dim>::mom(1)];
                }
            };
        }
    };

    template <std::size_t StencilSize = 2>
    struct RampBottom
    {
        template <class Field>
        using impl_t = RampBottomImpl<StencilSize, Field>;
    };

    inline void bc_fn(field_t& u, double& t, EOS::IdealGas eos)
    {
        static constexpr std::size_t dim = field_t::dim;

        const xt::xtensor_fixed<int, xt::xshape<dim>> bottom = {0, -1};
        const xt::xtensor_fixed<int, xt::xshape<dim>> top    = {0, 1};
        const xt::xtensor_fixed<int, xt::xshape<dim>> right  = {1, 0};
        const xt::xtensor_fixed<int, xt::xshape<dim>> left   = {-1, 0};

        // Bottom: the post-shock state upstream of the ramp foot, a wall
        // downstream of it, at the width the scheme reads.
        const auto inflow = prim2cons<2>(left_state, eos);
        if (bc::wide())
        {
            samurai::make_bc<RampBottom<4>>(u, inflow[0], inflow[1], inflow[2], inflow[3])->on(bottom);
        }
        else
        {
            samurai::make_bc<RampBottom<2>>(u, inflow[0], inflow[1], inflow[2], inflow[3])->on(bottom);
        }

        // Top: follows the analytic shock position, hence the dependence on t.
        bc::imposed(u,
                    [&t, eos](const auto&, const auto& cell, const auto&)
                    {
                        const double x1 = x0 + 10 * t / std::sin(alpha) + 1 / std::tan(alpha);
                        if (cell.center(0) < x1)
                        {
                            return prim2cons<2>(left_state, eos);
                        }
                        else
                        {
                            return prim2cons<2>(right_state, eos);
                        }
                    })
            ->on(top);

        // Right: outflow. Left: the incoming post-shock state.
        bc::outflow(u)->on(right);

        bc::imposed(u, left_state, eos)->on(left);
    }

    template <std::size_t dim>
    auto box_fn()
    {
        xt::xtensor_fixed<double, xt::xshape<dim>> min_corner = {0., 0.};
        xt::xtensor_fixed<double, xt::xshape<dim>> max_corner = {4., 1.};

        return samurai::Box<double, dim>(min_corner, max_corner);
    }

    template <class Field>
    test_case::TestCase<Field> definition()
    {
        static_assert(Field::dim == 2, "this test case is two-dimensional");
        return {.box = &box_fn<2>, .init = &init_fn, .bc = &bc_fn, .eos = EOS::ideal_gas(1.4)};
    }
}

REGISTER_TEST_CASE(double_mach_reflection, test_case::double_mach_reflection, 2)
