// Copyright 2025 the samurai team
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <array>
#include <memory>
#include <stdexcept>
#include <string>

#include <samurai/field.hpp>

// =============================================================================
//  Time integration
// -----------------------------------------------------------------------------
//  Three integrators:
//
//      euler   one explicit step. First order on its own; with the Hancock
//              predictor in the flux it is the MUSCL-Hancock scheme, second
//              order and one flux evaluation per step.
//      ssprk2  two stages averaged (Heun). Second order in time whatever the
//              flux does, at two flux evaluations per step.
//      strang  one directional sweep at a time, X(dt/2) Y(dt) X(dt/2) in two
//              dimensions, symmetric so that the splitting error cancels to
//              second order. Every sweep is genuinely one-dimensional, which is
//              the one setting where the Hancock predictor is complete. In one
//              dimension it is the single Hancock step, to the bit.
//
//  Measured: on the advected pulse, in one dimension, MUSCL-Hancock converges
//  at order 2.01 and is about twice as accurate as SSP-RK2 at equal
//  resolution, for half the work. On the isentropic vortex, in two dimensions,
//  it drops to order 1.05, against 2.13 for SSP-RK2 and 2.23 for Strang.
//
//  The Hancock predictor advances a face value with the equations taken normal
//  to that face, and in more than one dimension the transverse terms it leaves
//  out are of the same order as the ones it keeps. Recovering them needs the
//  cells across the face, and a flux stencil is a line. Lowering the CFL
//  shrinks the missing term and the order climbs back towards 2, reaching 1.77
//  at CFL 0.05, so the reconstruction itself is sound.
//
//  Strang buys the missing terms back rather than working around them: a sweep
//  along x has no transverse direction, so the normal-only predictor is the
//  right one for it. Counted in flux passes it is the cheaper of the two,
//  2*dim - 1 against 2*dim, but each sweep pays for a full ghost update, and
//  measured wall time comes out even in two dimensions and about 15% against
//  it in three. What it wins is accuracy: on the vortex its L1 error is 1.8
//  times smaller than SSP-RK2 at every resolution measured. It also brings a
//  splitting error of its own, which is second order and symmetric.
//
//  Hence `auto`: explicit Euler at order 1, Strang at order 2. In one dimension
//  Strang is the single Hancock step, so `auto` is the right scheme there too,
//  and SSP-RK2 stays available as the counter-check that owes nothing to a
//  predictor or to a sweep order.
// =============================================================================

enum class TimeIntegrator
{
    euler,
    ssprk2,
    strang
};

// What `--time-integrator auto` means: the integrator that gives the scheme the
// order it is being asked for, and the most accurate of those that do.
inline std::string default_time_integrator(std::size_t order)
{
    return order == 1 ? "euler" : "strang";
}

inline TimeIntegrator time_integrator_from_name(const std::string& name)
{
    if (name == "euler")
    {
        return TimeIntegrator::euler;
    }
    if (name == "ssprk2")
    {
        return TimeIntegrator::ssprk2;
    }
    if (name == "strang")
    {
        return TimeIntegrator::strang;
    }
    throw std::runtime_error("Unknown time integrator: " + name);
}

// One step, from u to u.
//
// `work1` and `work2` are scratch fields on the same mesh, carrying the same
// boundary conditions as u: the second stage of SSP-RK2 evaluates the scheme on
// an intermediate state, and a scheme evaluation fills ghosts, which needs the
// boundary conditions to be there.
//
// `sweeps` holds the same scheme restricted to one direction each, used by
// Strang and ignored by the others. `dt_in_flux` is what the Hancock predictor
// reads: a sweep advances the face values by its own step, not by the step of
// the whole iteration.
template <class Field, class Scheme>
void advance(Field& u,
             Field& work1,
             Field& work2,
             Scheme& scheme,
             std::array<Scheme, Field::dim>& sweeps,
             const std::shared_ptr<double>& dt_in_flux,
             double dt,
             TimeIntegrator integrator)
{
    work1.resize();
    *dt_in_flux = dt;

    if (integrator == TimeIntegrator::euler)
    {
        work1 = u - dt * scheme(u);
        samurai::swap(u, work1);
        return;
    }

    if (integrator == TimeIntegrator::strang)
    {
        auto sweep = [&](std::size_t d, double tau)
        {
            *dt_in_flux = tau;
            work1.resize();
            work1 = u - tau * sweeps[d](u);
            samurai::swap(u, work1);
        };

        // Half a step along every direction but the last, a whole one along the
        // last, then the halves again in reverse. In one dimension this is a
        // single full sweep, and the two loops are empty.
        for (std::size_t d = 0; d + 1 < Field::dim; ++d)
        {
            sweep(d, 0.5 * dt);
        }
        sweep(Field::dim - 1, dt);
        for (std::size_t d = Field::dim - 1; d-- > 0;)
        {
            sweep(d, 0.5 * dt);
        }
        return;
    }

    work2.resize();

    work1 = u - dt * scheme(u);          // u^(1)
    work2 = work1 - dt * scheme(work1);  // u^(2), one more step from there
    work1 = 0.5 * (u + work2);           // the average of the two is second order

    samurai::swap(u, work1);
}
