// Copyright 2025 the samurai team
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <array>
#include <cstddef>

#include <samurai/box.hpp>

#include "../bc.hpp"
#include "../eos.hpp"
#include "../variables.hpp"
#include "registry.hpp"

// =============================================================================
//  Blast wave in a periodic box
// -----------------------------------------------------------------------------
//  A high pressure sphere in a slightly denser ambient gas, in a box with no
//  boundary at all. The blast drives a spherical shock outwards, the shock
//  leaves through one face and comes back through the opposite one, and what
//  the run then shows is the scheme's treatment of shocks crossing each other.
//
//  It is the initial condition of the weak scaling study of the article this
//  repository reproduces (its eq. 5):
//
//      (rho, u, v, P) = (1.0, 0, 0, 10.0)  if r <= 0.15
//                       (1.2, 0, 0,  0.1)  if r >  0.15
//
//  with gamma = 5/3, periodic boundaries and t_f = 0.2. The article replicates
//  the sphere once per MPI rank so that the work per rank stays constant as the
//  machine grows; one sphere in one box is that same problem seen by one rank,
//  which is what a single node can say anything about. The replicated version
//  is a scaling study, not a solution, and belongs with the performance work.
//
//  Two things here exist nowhere else in the repository, which is most of what
//  earns the case its place in the suite. It is the only one that runs on a
//  gamma other than 1.4, so it is what says that the equation of state really
//  is data. And it is the only periodic one: mass and energy have nowhere to go
//  and must be conserved to round-off, which tests/test_invariants.py asserts.
//  A closed box conserves them too, but through a wall condition rather than
//  through the periodic ghost update, and those are different code.
// =============================================================================

namespace test_case::blast_periodic
{
    inline constexpr double radius  = 0.15;
    inline constexpr double rho_in  = 1.0;
    inline constexpr double p_in    = 10.0;
    inline constexpr double rho_out = 1.2;
    inline constexpr double p_out   = 0.1;

    inline constexpr double gamma = 5. / 3.;

    // The box is the unit one and the sphere sits at its centre, as it does in
    // each of the trees the article replicates it over.
    inline constexpr double center = 0.5;

    template <class Field>
    void init_fn(Field& u, const typename Field::cell_t& cell, EOS::IdealGas eos)
    {
        static constexpr std::size_t dim = Field::dim;

        const auto x = cell.center();

        double r2 = 0.;
        for (std::size_t d = 0; d < dim; ++d)
        {
            r2 += (x[d] - center) * (x[d] - center);
        }

        const bool inside = r2 <= radius * radius;

        PrimState<dim> state{inside ? rho_in : rho_out, inside ? p_in : p_out, {}};
        state.v.fill(0.);

        u[cell] = prim2cons<dim>(state, eos);
    }

    // Nothing to do: every face of the box is periodic, so there is no boundary
    // to give a condition to. The ghost cells are filled by samurai's periodic
    // update instead, which the mesh configuration below turns on.
    template <class Field>
    void bc_fn(Field& /*u*/, double& /*t*/, EOS::IdealGas /*eos*/)
    {
    }

    template <std::size_t dim>
    auto box_fn()
    {
        xt::xtensor_fixed<double, xt::xshape<dim>> min_corner;
        xt::xtensor_fixed<double, xt::xshape<dim>> max_corner;
        min_corner.fill(0.);
        max_corner.fill(1.);

        return samurai::Box<double, dim>(min_corner, max_corner);
    }

    template <class Field>
    test_case::TestCase<Field> definition()
    {
        std::array<bool, Field::dim> periodic;
        periodic.fill(true);

        return {.box = &box_fn<Field::dim>, .init = &init_fn<Field>, .bc = &bc_fn<Field>, .eos = EOS::ideal_gas(gamma), .periodic = periodic};
    }
}

REGISTER_TEST_CASE(blast_periodic, test_case::blast_periodic, 1, 2, 3)
