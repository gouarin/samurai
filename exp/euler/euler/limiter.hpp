// Copyright 2025 the samurai team
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <samurai/algorithm.hpp>

#include "eos.hpp"
#include "variables.hpp"

// =============================================================================
//  Positivity limiter
// -----------------------------------------------------------------------------
//  The Euler system is only well posed for strictly positive density and
//  pressure: the sound speed c = sqrt(gamma (p + pi_inf) / rho) - and hence the
//  time step - becomes NaN as soon as one of them turns negative.
//
//  On an adapted mesh the multiresolution prediction reconstructs fine cells by
//  a polynomial extrapolation of the coarse ones. Across a steep interface (a
//  blast front against a near-vacuum ambient state, for instance) that
//  extrapolation can overshoot and produce a conservative state with negative
//  density or pressure, even though every input state was admissible. First-order
//  finite-volume updates are positivity preserving on a uniform grid but lose
//  that guarantee at level jumps for the same reason.
//
//  This limiter projects any inadmissible cell back onto the admissible set with
//  the minimal change: the density is clamped to rho_min and, keeping the
//  momentum untouched, the total energy is reset so that the pressure equals
//  p_min. Cells that are already admissible are left strictly unchanged.
//
//  It returns how many cells it had to touch. A floor that never fires can be
//  removed; one that fires away from level interfaces points at the scheme
//  rather than at the prediction. The projection is not conservative, so a run
//  where it fires is a run whose mass and energy have moved by that much.
// =============================================================================

template <class Field, class Eos>
std::size_t limit_positivity(Field& u, Eos eos, double rho_min = 1e-10, double p_min = 1e-10)
{
    static constexpr std::size_t dim = std::decay_t<Field>::dim;
    using EulerConsVar               = EulerLayout<dim>;

    auto& mesh = u.mesh();

    std::size_t limited = 0;

    samurai::for_each_cell(mesh,
                           [&](const auto& cell)
                           {
                               bool touched = false;

                               // Density floor.
                               double rho = u[cell][EulerConsVar::rho];
                               if (rho < rho_min)
                               {
                                   rho                        = rho_min;
                                   u[cell][EulerConsVar::rho] = rho_min;
                                   touched                    = true;
                               }

                               // Pressure floor: keep the momentum, adjust the total energy.
                               double kinetic = 0.;
                               for (std::size_t d = 0; d < dim; ++d)
                               {
                                   const double mom = u[cell][EulerConsVar::mom(d)];
                                   kinetic += 0.5 * mom * mom / rho;
                               }
                               const double e = (u[cell][EulerConsVar::rhoE] - kinetic) / rho;
                               const double p = eos.p(rho, e);
                               if (p < p_min)
                               {
                                   u[cell][EulerConsVar::rhoE] = rho * eos.e(rho, p_min) + kinetic;
                                   touched                     = true;
                               }

                               limited += touched ? 1 : 0;
                           });

    return limited;
}
