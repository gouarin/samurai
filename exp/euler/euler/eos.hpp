// Copyright 2025 the samurai team
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <cmath>

// =============================================================================
//  Equations of state
// -----------------------------------------------------------------------------
//  The coefficients are *data*, not compile-time constants: the test cases do
//  not all share the same gamma (1.4 for air, 5/3 for the blast wave, 1.645 for
//  an air-helium mixture), and a two-phase model needs two gases alive at once
//  in the same computation. An instance is built in main() and passed down to
//  every routine that needs thermodynamics.
//
//  There are two of them on purpose. Everything monofluid runs on IdealGas,
//  whose state law carries no reference pressure nor energy: pushing the two
//  extra, always-zero coefficients of a stiffened gas through the flux kernel
//  costs about 5% of the time to solution (measured on a uniform level-8 Riemann
//  problem). Routines are templated on the equation of state rather than taking
//  one concrete type, so the monofluid path never sees pi_inf.
// =============================================================================

namespace EOS
{
    // Ideal gas:  p = (gamma - 1) rho e
    struct IdealGas
    {
        double gamma = 1.4;

        auto p(const auto& rho, const auto& e) const
        {
            return (gamma - 1.0) * rho * e;
        }

        auto c(const auto& rho, const auto& p) const
        {
            return std::sqrt(gamma * p / rho);
        }

        auto e(const auto& rho, const auto& p) const
        {
            return p / ((gamma - 1.0) * rho);
        }
    };

    // Stiffened gas:  p = (gamma - 1) rho (e - q_inf) - gamma pi_inf
    //
    // Models a liquid phase: pi_inf is a reference pressure, of the order of a
    // GPa for water. No monofluid test case needs it. It is the state law the
    // five-equation two-phase model will be written against, which is why it
    // stays here rather than being reinvented then.
    struct StiffenedGas
    {
        double gamma  = 1.4;
        double pi_inf = 0.;
        double q_inf  = 0.;

        auto p(const auto& rho, const auto& e) const
        {
            return (gamma - 1.0) * rho * (e - q_inf) - gamma * pi_inf;
        }

        auto c(const auto& rho, const auto& p) const
        {
            return std::sqrt(gamma * (p + pi_inf) / rho);
        }

        auto e(const auto& rho, const auto& p) const
        {
            return (p + gamma * pi_inf) / ((gamma - 1.0) * rho) + q_inf;
        }
    };

    constexpr IdealGas ideal_gas(double gamma)
    {
        return {gamma};
    }

    inline constexpr IdealGas air = ideal_gas(1.4);
} // namespace EOS
