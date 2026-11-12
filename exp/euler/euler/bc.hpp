// Copyright 2025 the samurai team
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <cstddef>
#include <utility>

#include <samurai/bc.hpp>

#include "variables.hpp"

// =============================================================================
//  Boundary conditions for the Euler system
// -----------------------------------------------------------------------------
//  samurai ships Dirichlet, Neumann and polynomial extrapolation. The Euler
//  equations need two more that depend on the *meaning* of the components, so
//  they live here rather than in the library:
//
//    - Imposed    a state written straight into the ghost cell, either constant
//                 or given by a function of the cell. This is how a supersonic
//                 inflow and a time-dependent boundary are expressed.
//    - Reflective a solid wall: the ghost cell mirrors its interior neighbour
//                 with the normal momentum reversed, so the wall-normal mass
//                 flux vanishes.
//
//  The free helpers at the bottom build the boundary conditions the test cases
//  actually ask for, in any dimension, so that a `bc_fn` is one line. They apply
//  to every boundary by default; chain `->on(direction)` to restrict one to a
//  single face, spelling the direction as a vector at the call site:
//
//      bc::outflow(u)->on(right);
//      bc::imposed(u, inflow_state, eos)->on(left);
//
//  Both conditions are written for any even stencil size: a stencil of 2 fills
//  one layer of ghost cells, which is what the first-order flux reads, and a
//  stencil of 4 fills two, which is what the MUSCL reconstruction reads. The
//  helpers below pick the width from `bc::ghost_layers`, set once at startup
//  from the order of the scheme.
// =============================================================================

namespace detail
{
    // In a boundary stencil of size N the cells run from the inside out: the
    // last N/2 are the ghosts to fill, cells[N/2 - 1] is the cell just inside
    // the boundary, and cells[N/2 - 1 - k] is k cells further in.
    inline constexpr std::size_t first_ghost(std::size_t stencil_size)
    {
        return stencil_size / 2;
    }
}

template <std::size_t StencilSize, class Field>
struct ImposedImpl : public samurai::Bc<Field>
{
    INIT_BC(ImposedImpl, StencilSize)

    apply_function_t get_apply_function(constant_stencil_size_t, const direction_t&) const override
    {
        return [](Field& u, const stencil_cells_t& cells, const value_t& value)
        {
            for (std::size_t i = detail::first_ghost(StencilSize); i < StencilSize; ++i)
            {
                u[cells[i]] = value;
            }
        };
    }
};

template <std::size_t StencilSize, class Field>
struct ReflectiveImpl : public samurai::Bc<Field>
{
    INIT_BC(ReflectiveImpl, StencilSize)

    apply_function_t get_apply_function(constant_stencil_size_t, const direction_t& direction) const override
    {
        // The stencil is aligned with the boundary normal, so exactly one
        // component of `direction` is non-zero: that is the axis to mirror.
        std::size_t normal = 0;
        for (std::size_t d = 0; d < Field::dim; ++d)
        {
            if (direction[d] != 0)
            {
                normal = d;
                break;
            }
        }

        return [normal](Field& u, const stencil_cells_t& cells, const value_t&)
        {
            // The wall is a mirror: the k-th ghost copies the k-th cell inside,
            // with the normal momentum reversed, so the wall-normal mass flux
            // vanishes.
            static constexpr std::size_t ghost0 = detail::first_ghost(StencilSize);

            for (std::size_t k = 0; k < ghost0; ++k)
            {
                const auto& inside = cells[ghost0 - 1 - k];
                const auto& ghost  = cells[ghost0 + k];

                u[ghost]                                       = u[inside];
                u[ghost][EulerLayout<Field::dim>::mom(normal)] = -u[inside][EulerLayout<Field::dim>::mom(normal)];
            }
        };
    }
};

// Zeroth-order extrapolation: every ghost repeats the boundary cell. This is
// what samurai's Neumann<1> does with a zero derivative, widened to more than
// one layer.
template <std::size_t StencilSize, class Field>
struct OutflowImpl : public samurai::Bc<Field>
{
    INIT_BC(OutflowImpl, StencilSize)

    apply_function_t get_apply_function(constant_stencil_size_t, const direction_t&) const override
    {
        return [](Field& u, const stencil_cells_t& cells, const value_t&)
        {
            static constexpr std::size_t ghost0 = detail::first_ghost(StencilSize);

            for (std::size_t i = ghost0; i < StencilSize; ++i)
            {
                u[cells[i]] = u[cells[ghost0 - 1]];
            }
        };
    }
};

// samurai's make_bc takes a type exposing impl_t<Field>, so each condition gets
// the same thin wrapper its Neumann does.
template <std::size_t StencilSize = 2>
struct Imposed
{
    template <class Field>
    using impl_t = ImposedImpl<StencilSize, Field>;
};

template <std::size_t StencilSize = 2>
struct Reflective
{
    template <class Field>
    using impl_t = ReflectiveImpl<StencilSize, Field>;
};

template <std::size_t StencilSize = 2>
struct Outflow
{
    template <class Field>
    using impl_t = OutflowImpl<StencilSize, Field>;
};

namespace bc
{
    // How many layers of ghost cells the boundary conditions have to fill: one
    // for the first-order flux, two for the MUSCL reconstruction. Set once at
    // startup from the order of the scheme, and read by every helper below, so
    // that the `bc_fn` of a test case states what the boundary is and not how
    // wide the stencil reading it will be.
    inline std::size_t& ghost_layers()
    {
        static std::size_t layers = 1;
        return layers;
    }

    inline bool wide()
    {
        return ghost_layers() > 1;
    }

    // Outflow (non-reflecting to first order): the boundary cell repeated into
    // every ghost. The one-layer case still goes through samurai's Neumann,
    // which leaves a first-order run unchanged to the last bit.
    template <class Field>
    auto outflow(Field& u)
    {
        if (wide())
        {
            return samurai::make_bc<Outflow<4>>(u);
        }

        return [&]<std::size_t... I>(std::index_sequence<I...>)
        {
            return samurai::make_bc<samurai::Neumann<1>>(u, (static_cast<void>(I), 0.)...);
        }(std::make_index_sequence<Field::n_comp>{});
    }

    // Solid wall on every boundary.
    template <class Field>
    auto wall(Field& u)
    {
        return wide() ? samurai::make_bc<Reflective<4>>(u) : samurai::make_bc<Reflective<2>>(u);
    }

    // An imposed state given by a function of the boundary cell. This is how a
    // time-dependent boundary is written: the case captures whatever it needs,
    // typically the current simulation time, by reference.
    //
    // NOTE that time is the one at the start of the iteration. A stage of
    // SSP-RK2 and a sweep of Strang both re-read it, so both see t^n where the
    // one wants t^n + dt and the other t^n + dt/2. Nothing measurable comes of
    // it today, the only moving boundary in the repository being the analytic
    // shock of the double Mach, whose position over one step moves by less than
    // the error the scheme makes anyway. It is a ceiling on the order that a
    // genuinely unsteady boundary would hit.
    template <class Field>
    auto imposed(Field& u, const typename samurai::FunctionBc<Field>::function_t& f)
    {
        return wide() ? samurai::make_bc<Imposed<4>>(u, f) : samurai::make_bc<Imposed<2>>(u, f);
    }

    // A uniform state imposed on every boundary.
    template <class Field, class Eos>
    auto imposed(Field& u, const PrimState<Field::dim>& state, Eos eos)
    {
        const auto cons = prim2cons<Field::dim>(state, eos);

        return [&]<std::size_t... I>(std::index_sequence<I...>)
        {
            return wide() ? samurai::make_bc<Imposed<4>>(u, cons[I]...) : samurai::make_bc<Imposed<2>>(u, cons[I]...);
        }(std::make_index_sequence<Field::n_comp>{});
    }
}
