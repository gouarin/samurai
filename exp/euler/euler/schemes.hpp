// Copyright 2025 the samurai team
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <string>

#include "eos.hpp"
#include "schemes/godunov.hpp"
#include "schemes/muscl.hpp"

// =============================================================================
//  Choosing a scheme
// -----------------------------------------------------------------------------
//  Two things are chosen separately and named separately on the command line:
//
//      --scheme      which Riemann solver settles an interface
//      --order       whether the solver is handed cell averages (1) or values
//                    reconstructed on the face (2)
//
//  The Riemann solver is a compile-time choice inside each builder, so the
//  three of them produce the same scheme type and the dispatch below returns
//  one. The order cannot be hidden the same way: a wider stencil is a different
//  type, which is why the two orders are built by two functions and the time
//  loop is handed both.
// =============================================================================

template <class Field, class Eos>
auto make_first_order_scheme(const std::string& name, Eos eos, int direction = -1)
{
    switch (riemann::from_name(name))
    {
        case riemann::Solver::rusanov:
            return make_godunov_scheme<riemann::Solver::rusanov, Field>(eos, direction);
        case riemann::Solver::hll:
            return make_godunov_scheme<riemann::Solver::hll, Field>(eos, direction);
        default:
            return make_godunov_scheme<riemann::Solver::hllc, Field>(eos, direction);
    }
}

template <class Field, class Eos>
auto make_second_order_scheme(const std::string& name, Eos eos, const MusclOptions& options)
{
    switch (riemann::from_name(name))
    {
        case riemann::Solver::rusanov:
            return make_muscl_scheme<riemann::Solver::rusanov, Field>(eos, options);
        case riemann::Solver::hll:
            return make_muscl_scheme<riemann::Solver::hll, Field>(eos, options);
        default:
            return make_muscl_scheme<riemann::Solver::hllc, Field>(eos, options);
    }
}
