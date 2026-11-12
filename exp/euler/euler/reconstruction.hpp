// Copyright 2025 the samurai team
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include "variables.hpp"

// =============================================================================
//  MUSCL reconstruction
// -----------------------------------------------------------------------------
//  A first-order scheme hands the Riemann solver the two cell averages. A
//  second-order one hands it the two values the solution takes ON the interface,
//  obtained by giving each cell a slope and evaluating it at the face.
//
//  The slope cannot be the centred difference: at a discontinuity that
//  overshoots, and the overshoot becomes an oscillation that no amount of
//  resolution removes. A limiter compares the two one-sided differences and
//  returns zero whenever they disagree in sign, which is what makes the
//  reconstruction total-variation diminishing:
//
//      none     centred, unlimited. Second order, oscillates at shocks. Kept
//               because a smooth test case (the vortex) is where an unlimited
//               slope is both admissible and the cleanest measure of the order.
//      minmod   the smaller of the two differences, and the most diffusive of
//               the three.
//      vanleer  harmonic mean, smooth in its arguments.
//      moncen   monotonized central, twice the one-sided slopes capped by the
//               centred one. The sharpest of the three, and the default.
//
//  Slopes are taken on the primitive variables. On the conservative ones a
//  limited slope of the total energy says nothing useful about the pressure,
//  which is the quantity that has to stay positive.
// =============================================================================

enum class SlopeLimiter
{
    none,
    minmod,
    vanleer,
    moncen
};

inline SlopeLimiter slope_limiter_from_name(const std::string& name)
{
    if (name == "none")
    {
        return SlopeLimiter::none;
    }
    if (name == "minmod")
    {
        return SlopeLimiter::minmod;
    }
    if (name == "vanleer")
    {
        return SlopeLimiter::vanleer;
    }
    if (name == "moncen")
    {
        return SlopeLimiter::moncen;
    }
    throw std::runtime_error("Unknown slope limiter: " + name);
}

// The slope over one cell width, from the two one-sided differences
// dm = w_i - w_{i-1} and dp = w_{i+1} - w_i.
template <SlopeLimiter limiter>
double limited_slope(double dm, double dp)
{
    if constexpr (limiter == SlopeLimiter::none)
    {
        return 0.5 * (dm + dp);
    }
    else
    {
        if (dm * dp <= 0.)
        {
            return 0.;
        }

        if constexpr (limiter == SlopeLimiter::minmod)
        {
            return std::abs(dm) < std::abs(dp) ? dm : dp;
        }
        else if constexpr (limiter == SlopeLimiter::vanleer)
        {
            return 2. * dm * dp / (dm + dp);
        }
        else // moncen
        {
            const double centred = 0.5 * (dm + dp);
            const double slope   = std::min({2. * std::abs(dm), 2. * std::abs(dp), std::abs(centred)});
            return std::copysign(slope, centred);
        }
    }
}

// The same, component by component, on a packed primitive state.
template <SlopeLimiter limiter, std::size_t Dim, class Array>
ConsArray<Dim> limited_slope(const Array& dm, const Array& dp)
{
    ConsArray<Dim> slope;
    for (std::size_t i = 0; i < EulerLayout<Dim>::size; ++i)
    {
        slope[i] = limited_slope<limiter>(dm[i], dp[i]);
    }
    return slope;
}

// =============================================================================
//  Hancock predictor
// -----------------------------------------------------------------------------
//  The reconstruction above is second order in space at time t^n. Evaluating
//  the flux with it and stepping in time with explicit Euler would still be
//  first order in time. The Hancock predictor buys the missing order without a
//  second stage: each face value is advanced half a step with the primitive
//  form of the equations,
//
//      W_t + A_d(W) W_x = 0,
//
//  the slope standing in for W_x. A_d applied to a slope is the function below:
//
//      (A dW)_rho = u_d drho + rho du_d
//      (A dW)_u_i = u_d du_i + dp / rho   (i == d only)
//      (A dW)_p   = u_d dp + rho c^2 du_d
//
//  Only the normal direction appears. The transverse terms of an unsplit
//  second-order scheme need the neighbours across the face, and a flux stencil
//  is a line, so they are out of reach here. That is why the order is measured
//  in each dimension, and why SSP-RK2 sits alongside as the integrator that
//  reaches second order in all of them.
// =============================================================================

template <std::size_t d, std::size_t Dim, class Array, class Eos>
ConsArray<Dim> primitive_jacobian_times(const PrimState<Dim>& prim, const Array& slope, Eos eos)
{
    using EulerConsVar = EulerLayout<Dim>;

    const auto c = eos.c(prim.rho, prim.p);

    const double drho = slope[EulerConsVar::rho];
    const double dp   = slope[EulerConsVar::rhoE];
    const double dud  = slope[EulerConsVar::mom(d)];

    ConsArray<Dim> out;
    out[EulerConsVar::rho]  = prim.v[d] * drho + prim.rho * dud;
    out[EulerConsVar::rhoE] = prim.v[d] * dp + prim.rho * c * c * dud;
    for (std::size_t i = 0; i < Dim; ++i)
    {
        out[EulerConsVar::mom(i)] = prim.v[d] * slope[EulerConsVar::mom(i)];
    }
    out[EulerConsVar::mom(d)] += dp / prim.rho;

    return out;
}

// Runtime choice of limiter. The switch sits outside the component loop and
// costs one branch per slope, against one Riemann solve per interface: the
// limiter stays a command line option without being a template parameter of
// every scheme.
template <std::size_t Dim, class Array>
ConsArray<Dim> limited_slope(const Array& dm, const Array& dp, SlopeLimiter limiter)
{
    switch (limiter)
    {
        case SlopeLimiter::none:
            return limited_slope<SlopeLimiter::none, Dim>(dm, dp);
        case SlopeLimiter::minmod:
            return limited_slope<SlopeLimiter::minmod, Dim>(dm, dp);
        case SlopeLimiter::vanleer:
            return limited_slope<SlopeLimiter::vanleer, Dim>(dm, dp);
        default:
            return limited_slope<SlopeLimiter::moncen, Dim>(dm, dp);
    }
}
