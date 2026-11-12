// Copyright 2025 the samurai team
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include "../variables.hpp"

template <std::size_t d, std::size_t Dim, class Eos>
auto compute_flux(const PrimState<Dim>& prim, Eos eos)
{
    using EulerConsVar = EulerLayout<Dim>;

    auto flux = xt::xtensor_fixed<double, xt::xshape<EulerConsVar::size>>{};

    flux[EulerConsVar::rho]  = prim.rho * prim.v[d];
    auto e                   = eos.e(prim.rho, prim.p);
    flux[EulerConsVar::rhoE] = (prim.rho * e + prim.p) * prim.v[d];
    for (std::size_t i = 0; i < Dim; ++i)
    {
        flux[EulerConsVar::mom(i)] = prim.rho * prim.v[i] * prim.v[d];
        flux[EulerConsVar::rhoE] += 0.5 * prim.rho * prim.v[i] * prim.v[i] * prim.v[d];
    }
    flux[EulerConsVar::mom(d)] += prim.p;
    return flux;
}