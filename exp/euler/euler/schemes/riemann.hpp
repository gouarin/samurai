// Copyright 2025 the samurai team
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <algorithm>
#include <cmath>

#include "../variables.hpp"
#include "flux.hpp"

// =============================================================================
//  Riemann solvers
// -----------------------------------------------------------------------------
//  Each solver answers the same question: given the two states an interface
//  separates, what flux crosses it. None of them knows where those states came
//  from. A first-order scheme hands over the two cell averages, a MUSCL scheme
//  hands over two reconstructed face values, and the solver is the same.
//
//  Every solver takes each state twice, as primitives and as the matching
//  conservative vector. Both are needed (the wave speeds come from the
//  primitives, the jumps from the conservative variables) and the caller
//  usually has both already, so converting again here would be waste as well
//  as a source of round-off.
// =============================================================================

namespace riemann
{
    template <std::size_t d, std::size_t Dim, class ConsL, class ConsR, class Eos>
    ConsArray<Dim> rusanov(const PrimState<Dim>& primL, const ConsL& qL, const PrimState<Dim>& primR, const ConsR& qR, Eos eos)
    {
        const auto cL = eos.c(primL.rho, primL.p);
        const auto cR = eos.c(primR.rho, primR.p);

        const auto lambda = std::max(std::abs(primL.v[d]) + cL, std::abs(primR.v[d]) + cR);

        return 0.5 * (compute_flux<d>(primL, eos) + compute_flux<d>(primR, eos) - lambda * (qR - qL));
    }

    template <std::size_t d, std::size_t Dim, class ConsL, class ConsR, class Eos>
    ConsArray<Dim> hll(const PrimState<Dim>& primL, const ConsL& qL, const PrimState<Dim>& primR, const ConsR& qR, Eos eos)
    {
        const auto cL = eos.c(primL.rho, primL.p);
        const auto cR = eos.c(primR.rho, primR.p);

        const double sL = std::min(primL.v[d] - cL, primR.v[d] - cR);
        const double sR = std::max(primL.v[d] + cL, primR.v[d] + cR);

        if (sL >= 0)
        {
            return compute_flux<d>(primL, eos);
        }
        if (sR <= 0)
        {
            return compute_flux<d>(primR, eos);
        }
        return (sR * compute_flux<d>(primL, eos) - sL * compute_flux<d>(primR, eos) + sL * sR * (qR - qL)) / (sR - sL);
    }

    // The state between the contact and the wave of speed s, for HLLC.
    template <std::size_t d, std::size_t Dim, class Eos>
    ConsArray<Dim> star_state(const PrimState<Dim>& prim, double s, double s_star, Eos eos)
    {
        using EulerConsVar = EulerLayout<Dim>;
        ConsArray<Dim> q_star;

        auto rho_star = prim.rho * (s - prim.v[d]) / (s - s_star);

        q_star[EulerConsVar::rho] = rho_star;

        auto e                     = eos.e(prim.rho, prim.p);
        q_star[EulerConsVar::rhoE] = rho_star * (e + (s_star - prim.v[d]) * (s_star + prim.p / (prim.rho * (s - prim.v[d]))));

        for (std::size_t i = 0; i < Dim; ++i)
        {
            q_star[EulerConsVar::mom(i)] = rho_star * prim.v[i];
            q_star[EulerConsVar::rhoE] += 0.5 * rho_star * prim.v[i] * prim.v[i];
        }
        q_star[EulerConsVar::mom(d)] = rho_star * s_star;

        return q_star;
    }

    template <std::size_t d, std::size_t Dim, class ConsL, class ConsR, class Eos>
    ConsArray<Dim> hllc(const PrimState<Dim>& primL, const ConsL& qL, const PrimState<Dim>& primR, const ConsR& qR, Eos eos)
    {
        const auto cL = eos.c(primL.rho, primL.p);
        const auto cR = eos.c(primR.rho, primR.p);

        const double sL = std::min(primL.v[d] - cL, primR.v[d] - cR);
        const double sR = std::max(primL.v[d] + cL, primR.v[d] + cR);
        const double sM = (primL.rho * primL.v[d] * (sL - primL.v[d]) - primL.p - primR.rho * primR.v[d] * (sR - primR.v[d]) + primR.p)
                        / (primL.rho * (sL - primL.v[d]) - primR.rho * (sR - primR.v[d]));

        if (sL >= 0)
        {
            return compute_flux<d>(primL, eos);
        }
        if (sL < 0 && sM >= 0)
        {
            return compute_flux<d>(primL, eos) + sL * (star_state<d>(primL, sL, sM, eos) - qL);
        }
        if (sM < 0 && sR >= 0)
        {
            return compute_flux<d>(primR, eos) + sR * (star_state<d>(primR, sR, sM, eos) - qR);
        }
        return compute_flux<d>(primR, eos);
    }

    // Which solver a name stands for. The three take the same arguments, so the
    // schemes below are written once and instantiated three times.
    enum class Solver
    {
        rusanov,
        hll,
        hllc
    };

    inline Solver from_name(const std::string& name)
    {
        if (name == "rusanov")
        {
            return Solver::rusanov;
        }
        if (name == "hll")
        {
            return Solver::hll;
        }
        if (name == "hllc")
        {
            return Solver::hllc;
        }
        throw std::runtime_error("Unknown scheme: " + name);
    }

    template <Solver s, std::size_t d, std::size_t Dim, class ConsL, class ConsR, class Eos>
    ConsArray<Dim> solve(const PrimState<Dim>& primL, const ConsL& qL, const PrimState<Dim>& primR, const ConsR& qR, Eos eos)
    {
        if constexpr (s == Solver::rusanov)
        {
            return rusanov<d>(primL, qL, primR, qR, eos);
        }
        else if constexpr (s == Solver::hll)
        {
            return hll<d>(primL, qL, primR, qR, eos);
        }
        else
        {
            return hllc<d>(primL, qL, primR, qR, eos);
        }
    }
}
