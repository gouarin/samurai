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
//  Isentropic Euler vortex (smooth, analytic test case)
// -----------------------------------------------------------------------------
//  An isentropic vortex superimposed on a uniform mean flow. It is an exact
//  solution of the Euler equations: the vortex is advected by the mean flow
//  without deformation, so the exact solution at time t is the initial condition
//  translated by the mean velocity. Being C-infinity (no shock, no contact), the
//  L1/L2/Linf errors decay at the *design* order of the scheme, which makes this
//  case the reference for measuring convergence rates; the moving structure also
//  exercises the mesh adaptation (the refined region must track the vortex).
//
//  The case goes back to
//      C.-W. Shu, "Essentially non-oscillatory and weighted essentially
//      non-oscillatory schemes for hyperbolic conservation laws", in: Advanced
//      Numerical Approximation of Nonlinear Hyperbolic Equations, Lecture Notes
//      in Mathematics 1697, Springer, 1998, pp. 325-432,
//      https://doi.org/10.1007/BFb0096355
//
//  but the many variants of it in circulation differ in their scaling, so the
//  notation and formulas here follow the survey that sorts them out:
//      S. C. Spiegel, H. T. Huynh, J. R. DeBonis, "A Survey of the Isentropic
//      Euler Vortex Problem using High-Order Methods", AIAA Paper 2015-2444,
//      NASA Glenn Research Center, 2015 (NTRS 20150018403),
//      https://doi.org/10.2514/6.2015-2444.
//  Equation/section numbers below refer to that paper. The parameter set is the
//  "Shu" row of Table 1, in the paper's (sound-speed) non-dimensionalization:
//  rho_inf = 1, a_inf = 1, T_inf = 1, R_gas = 1, so p_inf = 1/gamma.
//
//  Everything that depends on gamma is derived from the equation of state, so
//  the exact solution stays exact if `--gamma` is changed. Note that the Python
//  post-processing (python/error_analysis.py) assumes gamma = 1.4.
//
//  Periodicity is emulated by imposing the exact (time-dependent) solution at the
//  boundaries, so no change to the mesh / BC framework is required.
// =============================================================================

namespace test_case::isentropic_vortex
{
    using field_t = config<2>::field_t;
    using std::numbers::pi;

    // --- "Shu" parameter row of Table 1 ------------------------------------
    inline const double alpha       = pi / 4.; // angle of attack (mean-flow direction)
    inline constexpr double rho_inf = 1.;      // free-stream density
    inline constexpr double R       = 1.;      // characteristic length scale (eq. 21)
    inline constexpr double sigma   = 1.;      // Gaussian standard deviation (eq. 21)
    inline constexpr double L       = 5.;      // half domain length: domain [-L, L]^2 (sec. IV.A.2)
    inline constexpr double x0      = 0.;      // initial vortex center x (eq. 24)
    inline constexpr double y0      = 0.;      // initial vortex center y (eq. 24)

    // free-stream Mach number (Table 1) and vortex strength (eq. 20)
    inline double M_inf(EOS::IdealGas eos)
    {
        return std::sqrt(2. / eos.gamma);
    }

    inline double beta(EOS::IdealGas eos)
    {
        return M_inf(eos) * 5. * std::sqrt(2.) / (4. * pi) * std::exp(0.5);
    }

    // Exact primitive state at point (x, y) and time t. Implements eqs. (20)-(24)
    // of the reference paper (periodic, nearest-image evaluation).
    inline PrimState<2> exact_state(double x, double y, double t, EOS::IdealGas eos)
    {
        constexpr double domain_length = 2. * L; // periodic length in each direction

        // free-stream velocity components (eq. 23): M_inf * (cos alpha, sin alpha)
        const double v_x_inf = M_inf(eos) * std::cos(alpha);
        const double v_y_inf = M_inf(eos) * std::sin(alpha);

        // moving vortex center (eq. 24), wrapped into [-L, L]
        const double x_c = x0 + v_x_inf * t;
        const double y_c = y0 + v_y_inf * t;

        double x_bar = x - x_c;
        double y_bar = y - y_c;
        x_bar -= domain_length * std::round(x_bar / domain_length);
        y_bar -= domain_length * std::round(y_bar / domain_length);

        // Gaussian (eqs. 20-21) and perturbations (eq. 22)
        const double f     = -1. / (2. * sigma * sigma) * ((x_bar / R) * (x_bar / R) + (y_bar / R) * (y_bar / R));
        const double Omega = beta(eos) * std::exp(f);

        const double delta_v_x = -(y_bar / R) * Omega;
        const double delta_v_y = +(x_bar / R) * Omega;
        const double delta_T   = -(eos.gamma - 1.) / 2. * Omega * Omega;

        // initial primitive variables (eq. 23), with the isentropic relations
        const double rho = rho_inf * std::pow(1. + delta_T, 1. / (eos.gamma - 1.));
        const double p   = 1. / eos.gamma * std::pow(1. + delta_T, eos.gamma / (eos.gamma - 1.));
        const double v_x = v_x_inf + delta_v_x;
        const double v_y = v_y_inf + delta_v_y;

        return PrimState<2>{
            rho,
            p,
            xt::xtensor_fixed<double, xt::xshape<2>>{v_x, v_y}
        };
    }

    inline void init_fn(field_t& u, const typename field_t::cell_t& cell, EOS::IdealGas eos)
    {
        const auto x = cell.center();
        u[cell]      = prim2cons<2>(exact_state(x[0], x[1], 0., eos), eos);
    }

    inline void bc_fn(field_t& u, double& t, EOS::IdealGas eos)
    {
        // Impose the exact (time-dependent) solution on every boundary.
        // `t` is the simulation time, captured by reference on purpose: it must
        // be the current one every time the boundary is applied.
        auto exact_bc = [&t, eos](const auto&, const auto& cell, const auto&)
        {
            const auto x = cell.center();
            return prim2cons<2>(exact_state(x[0], x[1], t, eos), eos);
        };

        for (const auto& dir : {
                 xt::xtensor_fixed<int, xt::xshape<2>>{-1, 0 },
                 xt::xtensor_fixed<int, xt::xshape<2>>{1,  0 },
                 xt::xtensor_fixed<int, xt::xshape<2>>{0,  -1},
                 xt::xtensor_fixed<int, xt::xshape<2>>{0,  1 }
        })
        {
            bc::imposed(u, exact_bc)->on(dir);
        }
    }

    template <std::size_t dim>
    auto box_fn()
    {
        xt::xtensor_fixed<double, xt::xshape<dim>> min_corner = {-L, -L};
        xt::xtensor_fixed<double, xt::xshape<dim>> max_corner = {L, L};

        return samurai::Box<double, dim>(min_corner, max_corner);
    }

    template <class Field>
    test_case::TestCase<Field> definition()
    {
        static_assert(Field::dim == 2, "this test case is two-dimensional");
        return {.box = &box_fn<2>, .init = &init_fn, .bc = &bc_fn, .eos = EOS::ideal_gas(1.4)};
    }
}

REGISTER_TEST_CASE(isentropic_vortex, test_case::isentropic_vortex, 2)
