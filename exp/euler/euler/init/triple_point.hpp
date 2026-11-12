// Copyright 2025 the samurai team
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <samurai/box.hpp>

#include "../bc.hpp"
#include "../variables.hpp"
#include "registry.hpp"

// =============================================================================
//  Triple point, single gamma
// -----------------------------------------------------------------------------
//  Three regions meet at one point. The left one is at high pressure and drives
//  a shock to the right; the two right-hand regions differ in density, so the
//  shock crosses them at different speeds, the contact between them shears, and
//  a Kelvin-Helmholtz roll-up grows around the triple point. It is a standard
//  problem of the multimaterial ALE literature,
//
//      S. Galera, P.-H. Maire, J. Breil, "A two-dimensional unstructured
//      cell-centered multi-material ALE scheme using VOF interface
//      reconstruction", J. Comput. Phys. 229 (16) (2010) 5755-5787,
//      https://doi.org/10.1016/j.jcp.2010.04.019
//
//  and is section 6.1.2 of the article this repository reproduces.
//
//  THIS IS NOT THE CASE OF THE ARTICLE, and no figure here should be held
//  against its fig. 12. There, the three regions carry two gases, gamma = 1.5 in
//  regions 1 and 3 and gamma = 1.4 in region 2, and the run is made with the
//  five-equation two-phase model. A monofluid solver has one gamma to give, so
//  this case gives 1.5, the value of the two regions that share a gas and of the
//  single-material version of the problem in the ALE literature. Region 2 is
//  then the same gas as the others rather than a second one, the contact across
//  y = 1.5 is a pure density jump, and the interface no longer has a surface
//  tension-free two-phase model behind it.
//
//  What survives the simplification is the geometry and the instability, which
//  is what makes the case worth having. It is the longest run in the
//  repository, on the most strongly adapted mesh: three shocks and a shear
//  layer that the multiresolution has to follow long enough for a defect in the
//  adaptation to show.
//
//  Domain [0,7] x [0,3], t_f = 2.0, walls all around, everything at rest:
//
//      region 1  [0,1[ x [0,3[      rho = 1,     P = 1
//      region 2  [1,7[ x [0,1.5[    rho = 1,     P = 0.1
//      region 3  [1,7[ x [1.5,3[    rho = 0.125, P = 0.1
// =============================================================================

namespace test_case::triple_point_single_gamma
{
    using field_t = config<2>::field_t;

    inline constexpr double x_wall = 1.0; // between region 1 and the other two
    inline constexpr double y_wall = 1.5; // between regions 2 and 3

    inline const PrimState<2> region_1{
        1.,
        1.,
        xt::xtensor_fixed<double, xt::xshape<2>>{0., 0.}
    };

    inline const PrimState<2> region_2{
        1.,
        0.1,
        xt::xtensor_fixed<double, xt::xshape<2>>{0., 0.}
    };

    inline const PrimState<2> region_3{
        0.125,
        0.1,
        xt::xtensor_fixed<double, xt::xshape<2>>{0., 0.}
    };

    inline void init_fn(field_t& u, const typename field_t::cell_t& cell, EOS::IdealGas eos)
    {
        const auto x = cell.center();

        if (x[0] < x_wall)
        {
            u[cell] = prim2cons<2>(region_1, eos);
        }
        else if (x[1] < y_wall)
        {
            u[cell] = prim2cons<2>(region_2, eos);
        }
        else
        {
            u[cell] = prim2cons<2>(region_3, eos);
        }
    }

    // Solid walls. The problem is posed in a closed box: the shock reflects off
    // the right-hand wall well after t_f, and the top and bottom walls are what
    // keeps the roll-up two-dimensional.
    inline void bc_fn(field_t& u, double& /*t*/, EOS::IdealGas /*eos*/)
    {
        bc::wall(u);
    }

    template <std::size_t dim>
    auto box_fn()
    {
        xt::xtensor_fixed<double, xt::xshape<dim>> min_corner = {0., 0.};
        xt::xtensor_fixed<double, xt::xshape<dim>> max_corner = {7., 3.};

        return samurai::Box<double, dim>(min_corner, max_corner);
    }

    template <class Field>
    test_case::TestCase<Field> definition()
    {
        static_assert(Field::dim == 2, "this test case is two-dimensional");
        return {.box = &box_fn<2>, .init = &init_fn, .bc = &bc_fn, .eos = EOS::ideal_gas(1.5)};
    }
}

REGISTER_TEST_CASE(triple_point_single_gamma, test_case::triple_point_single_gamma, 2)
