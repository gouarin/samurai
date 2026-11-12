// Copyright 2025 the samurai team
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <cassert>
#include <memory>

#include <samurai/schemes/fv.hpp>

#include "../reconstruction.hpp"
#include "../variables.hpp"
#include "riemann.hpp"

// =============================================================================
//  Second-order scheme: MUSCL reconstruction, optionally with a Hancock step
// -----------------------------------------------------------------------------
//  Four cells per interface instead of two. Numbering them 0 to 3, the flux is
//  computed between cells 1 and 2, and cells 0 and 3 are there to give each of
//  those two a slope:
//
//      |  0  |  1  |  2  |  3  |
//                  ^
//                  the interface this flux crosses
//
//  Each cell is given a limited slope, the reconstruction is evaluated on the
//  face from either side, and the Riemann solver is handed those two values
//  rather than the cell averages.
//
//  With `hancock` the two face values are then advanced half a time step, which
//  makes a single explicit Euler update second order in time as well. Without
//  it, the flux is a pure spatial reconstruction and the time loop is expected
//  to supply the second order itself, through SSP-RK2.
//
//  The reconstruction can produce a state that is not admissible: a negative
//  pressure in a near-vacuum, or a negative density under an unlimited slope.
//  A face where that happens falls back to the cell average, which makes the
//  flux first order at that one face.
// =============================================================================

struct MusclOptions
{
    SlopeLimiter limiter = SlopeLimiter::moncen;

    // Advance the face values half a step before solving the Riemann problem.
    bool hancock = true;

    // The time step the Hancock predictor uses. The flux function is built once
    // and called with a different dt at every iteration, so it reads the value
    // through this pointer rather than capturing it.
    std::shared_ptr<const double> dt;

    // Restrict the scheme to one direction, for a directional sweep. Every
    // other direction is left without a flux function, which samurai skips
    // entirely, so a sweep costs one pass and not dim of them. -1 keeps all of
    // them, which is the unsplit scheme.
    int direction = -1;
};

template <riemann::Solver solver, class Field, class Eos>
auto make_muscl_scheme(Eos eos, const MusclOptions& options)
{
    // The predictor reads the time step through the pointer; without one it
    // would read through null at the first interface of the first iteration.
    assert((!options.hancock || options.dt) && "the Hancock predictor needs a time step to read");

    static constexpr std::size_t dim          = Field::dim;
    static constexpr std::size_t stencil_size = 4;

    using cfg = samurai::FluxConfig<samurai::SchemeType::NonLinear, stencil_size, Field, Field>;

    samurai::FluxDefinition<cfg> muscl;

    samurai::static_for<0, dim>::apply( // for each positive Cartesian direction 'd'
        [&](auto _d)
        {
            static constexpr std::size_t d = _d();

            if (options.direction >= 0 && d != static_cast<std::size_t>(options.direction))
            {
                return;
            }

            muscl[d].cons_flux_function =
                [eos, options](samurai::FluxValue<cfg>& flux, const samurai::StencilData<cfg>& data, const samurai::StencilValues<cfg>& field)
            {
                const auto w0 = pack<dim>(cons2prim<dim>(field[0], eos));
                const auto w1 = pack<dim>(cons2prim<dim>(field[1], eos));
                const auto w2 = pack<dim>(cons2prim<dim>(field[2], eos));
                const auto w3 = pack<dim>(cons2prim<dim>(field[3], eos));

                const auto slopeL = limited_slope<dim>(w1 - w0, w2 - w1, options.limiter);
                const auto slopeR = limited_slope<dim>(w2 - w1, w3 - w2, options.limiter);

                ConsArray<dim> wL = w1 + 0.5 * slopeL;
                ConsArray<dim> wR = w2 - 0.5 * slopeR;

                if (options.hancock)
                {
                    const double half = 0.5 * (*options.dt) / data.cell_length;
                    wL -= half * primitive_jacobian_times<d>(unpack<dim>(w1), slopeL, eos);
                    wR -= half * primitive_jacobian_times<d>(unpack<dim>(w2), slopeR, eos);
                }

                auto admissible = [](const ConsArray<dim>& reconstructed, const ConsArray<dim>& average)
                {
                    const bool ok = reconstructed[EulerLayout<dim>::rho] > 0. && reconstructed[EulerLayout<dim>::rhoE] > 0.;
                    return unpack<dim>(ok ? reconstructed : average);
                };

                const auto primL = admissible(wL, w1);
                const auto primR = admissible(wR, w2);

                flux = riemann::solve<solver, d>(primL, prim2cons<dim>(primL, eos), primR, prim2cons<dim>(primR, eos), eos);
            };
        });

    auto scheme = make_flux_based_scheme(muscl);
    scheme.set_name("muscl");

    return scheme;
}
