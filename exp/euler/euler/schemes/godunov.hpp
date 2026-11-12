// Copyright 2025 the samurai team
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <samurai/schemes/fv.hpp>

#include "../variables.hpp"
#include "riemann.hpp"

// =============================================================================
//  First-order Godunov scheme
// -----------------------------------------------------------------------------
//  The flux across an interface is the Riemann flux between the two cell
//  averages it separates. Two cells per interface, hence a stencil of 2, and an
//  update that is first order in space.
//
//  Paired with the explicit Euler step of the time loop, it is what `--order 1`
//  runs, and the reference the second-order scheme is compared against.
// =============================================================================

// `direction` restricts the scheme to one direction, for a directional sweep;
// -1 keeps all of them.
template <riemann::Solver solver, class Field, class Eos>
auto make_godunov_scheme(Eos eos, int direction = -1)
{
    static constexpr std::size_t dim          = Field::dim;
    static constexpr std::size_t stencil_size = 2;

    using cfg = samurai::FluxConfig<samurai::SchemeType::NonLinear, stencil_size, Field, Field>;

    samurai::FluxDefinition<cfg> godunov;

    samurai::static_for<0, dim>::apply( // for each positive Cartesian direction 'd'
        [&](auto _d)
        {
            static constexpr std::size_t d = _d();

            if (direction >= 0 && d != static_cast<std::size_t>(direction))
            {
                return;
            }

            godunov[d].cons_flux_function =
                [eos](samurai::FluxValue<cfg>& flux, const samurai::StencilData<cfg>& /*data*/, const samurai::StencilValues<cfg>& field)
            {
                static constexpr std::size_t left  = 0;
                static constexpr std::size_t right = 1;

                const auto& qL = field[left];
                const auto& qR = field[right];

                flux = riemann::solve<solver, d>(cons2prim<dim>(qL, eos), qL, cons2prim<dim>(qR, eos), qR, eos);
            };
        });

    auto scheme = make_flux_based_scheme(godunov);
    scheme.set_name("godunov");

    return scheme;
}
