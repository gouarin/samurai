// Copyright 2025 the samurai team
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>

#include <CLI/CLI.hpp>
#include <samurai/box.hpp>

#include "../bc.hpp"
#include "../variables.hpp"
#include "registry.hpp"

// =============================================================================
//  Lax & Liu two-dimensional Riemann problems
// -----------------------------------------------------------------------------
//  The unit square is split into four quadrants, each holding a uniform state.
//  The nineteen admissible combinations are classified in
//
//      P.D. Lax, X.-D. Liu, "Solution of two-dimensional Riemann problems of gas
//      dynamics by positive schemes", SIAM J. Sci. Comput. 19 (2) (1998)
//      319-340, https://doi.org/10.1137/S1064827595291819
//
//  and are tabulated, with this numbering, in
//
//      A. Kurganov, E. Tadmor, "Solution of two-dimensional Riemann problems
//      for gas dynamics without Riemann problem solvers", Numer. Methods
//      Partial Differ. Equ. 18 (5) (2002) 584-608,
//      https://doi.org/10.1002/num.10025
//
//  All nineteen are here, picked by `--riemann-config`, rather than one case per
//  configuration: they differ by twenty numbers and nothing else, and a case per
//  configuration is twenty more numbers to copy each time one is wanted. The
//  article this repository reproduces (eq. 4) takes configuration 3 as its
//  reference case for performance, which is why it is the default.
//
//  Quadrants are numbered counter-clockwise from the upper right, as Lax & Liu
//  number them:
//      q[0] : x >= x0, y >= y0        q[1] : x <  x0, y >= y0
//      q[2] : x <  x0, y <  y0        q[3] : x >= x0, y <  y0
//
//  Where the interfaces sit is not part of the classification. Lax & Liu and
//  Kurganov & Tadmor classify four states and put them in the four quadrants of
//  the unit square, meeting at its centre. The default here is x0 = y0 = 0.8,
//  the position the article this repository reproduces uses for its
//  configuration 3 (its eq. 4): with t_f = 0.8 the waves then fill the domain
//  without reaching its boundary, where the outflow condition would pollute
//  them.
//
//  That default suits configurations 3 and 4 and misleads for the other
//  seventeen, whose structure is born 0.2 from the upper right corner and
//  reaches it well before the final times the published figures use. Anyone
//  comparing against Kurganov & Tadmor rather than against the article wants
//  `--riemann-interface 0.5`.
//
//  Three dimensions
//  ----------------
//  The same case serves euler_3d for configuration 3 only, which is the one the
//  article extends (its fig. 9, equivalent resolution 512^3, t_f = 0.8). The
//  eight octant states are those of `getRiemannConfig3d` in the reference code,
//  and they follow one rule: an octant is given velocity 1.206 along each
//  direction in which it sits on the *low* side of the interface, and the state
//  that goes with the number of such directions, (1.5, 1.5) for none,
//  (0.5323, 0.3) for one, (0.138, 0.029) for two or three. Restricted to w = 0
//  that is exactly configuration 3, which is what makes this extension the
//  natural one rather than one of several.
//
//  Octants are numbered as the quadrants above on the upper slab, then the same
//  four on the lower one:
//      o[0..3] : z >= z0, in quadrant order
//      o[4..7] : z <  z0, in quadrant order
//
//  Where the reference code differs from Lax & Liu
//  -----------------------------------------------
//  The table below is Kurganov & Tadmor's, cross-checked against two independent
//  implementations. The reference implementation of the article disagrees with
//  it in three places, and the table here follows Lax & Liu:
//
//    - configuration 6 has p = 1 in all four quadrants; the reference code puts
//      0.5 in the upper left one;
//    - configuration 10 has v = -0.4297 in the lower right quadrant, not
//      -0.4259, which is the figure belonging to configuration 9;
//    - configuration 19 is the one below; the reference code repeats
//      configuration 1 in its place.
//
//  Only configurations 3 and 4 are used in the article, and neither is affected.
//
//  Configurations 2, 3, 4, 7, 8 and 12 are invariant under the reflection about
//  the diagonal (x,y,u,v) -> (y,x,v,u), so their solution must be too.
//  tests/test_validation.py asserts that, and a single mistyped digit in one
//  quadrant of any of the six is enough to break it.
// =============================================================================

namespace test_case::lax_liu
{
    using Quadrants = std::array<PrimState<2>, 4>;
    using Octants   = std::array<PrimState<3>, 8>;

    // Which configuration runs, and where its interfaces sit on the unit box, in
    // every direction. `--riemann-config` and `--riemann-interface` write into
    // these, so both are read when a cell is initialised and not before.
    inline int& selected_config()
    {
        static int number = 3;
        return number;
    }

    inline double& selected_interface()
    {
        static double x0 = 0.8;
        return x0;
    }

    // (rho, p, (u, v)) per quadrant, configurations 1 to 19 in order.
    inline const std::array<Quadrants, 19> configurations = {
        Quadrants{PrimState<2>{1., 1., {0., 0.}},
                  PrimState<2>{0.5197, 0.4, {-0.7259, 0.}},
                  PrimState<2>{0.1072, 0.0439, {-0.7259, -1.4045}},
                  PrimState<2>{0.2579, 0.15, {0., -1.4045}}  },

        Quadrants{PrimState<2>{1., 1., {0., 0.}},
                  PrimState<2>{0.5197, 0.4, {-0.7259, 0.}},
                  PrimState<2>{1., 1., {-0.7259, -0.7259}},
                  PrimState<2>{0.5197, 0.4, {0., -0.7259}}   },

        Quadrants{PrimState<2>{1.5, 1.5, {0., 0.}},
                  PrimState<2>{0.5323, 0.3, {1.206, 0.}},
                  PrimState<2>{0.138, 0.029, {1.206, 1.206}},
                  PrimState<2>{0.5323, 0.3, {0., 1.206}}     },

        Quadrants{PrimState<2>{1.1, 1.1, {0., 0.}},
                  PrimState<2>{0.5065, 0.35, {0.8939, 0.}},
                  PrimState<2>{1.1, 1.1, {0.8939, 0.8939}},
                  PrimState<2>{0.5065, 0.35, {0., 0.8939}}   },

        Quadrants{PrimState<2>{1., 1., {-0.75, -0.5}},
                  PrimState<2>{2., 1., {-0.75, 0.5}},
                  PrimState<2>{1., 1., {0.75, 0.5}},
                  PrimState<2>{3., 1., {0.75, -0.5}}         },

        Quadrants{PrimState<2>{1., 1., {0.75, -0.5}},
                  PrimState<2>{2., 1., {0.75, 0.5}},
                  PrimState<2>{1., 1., {-0.75, 0.5}},
                  PrimState<2>{3., 1., {-0.75, -0.5}}        },

        Quadrants{PrimState<2>{1., 1., {0.1, 0.1}},
                  PrimState<2>{0.5197, 0.4, {-0.6259, 0.1}},
                  PrimState<2>{0.8, 0.4, {0.1, 0.1}},
                  PrimState<2>{0.5197, 0.4, {0.1, -0.6259}}  },

        Quadrants{PrimState<2>{0.5197, 0.4, {0.1, 0.1}},
                  PrimState<2>{1., 1., {-0.6259, 0.1}},
                  PrimState<2>{0.8, 1., {0.1, 0.1}},
                  PrimState<2>{1., 1., {0.1, -0.6259}}       },

        Quadrants{PrimState<2>{1., 1., {0., 0.3}},
                  PrimState<2>{2., 1., {0., -0.3}},
                  PrimState<2>{1.039, 0.4, {0., -0.8133}},
                  PrimState<2>{0.5197, 0.4, {0., -0.4259}}   },

        Quadrants{PrimState<2>{1., 1., {0., 0.4297}},
                  PrimState<2>{0.5, 1., {0., 0.6076}},
                  PrimState<2>{0.2281, 0.3333, {0., -0.6076}},
                  PrimState<2>{0.4562, 0.3333, {0., -0.4297}}},

        Quadrants{PrimState<2>{1., 1., {0.1, 0.}},
                  PrimState<2>{0.5313, 0.4, {0.8276, 0.}},
                  PrimState<2>{0.8, 0.4, {0.1, 0.}},
                  PrimState<2>{0.5313, 0.4, {0.1, 0.7276}}   },

        Quadrants{PrimState<2>{0.5313, 0.4, {0., 0.}},
                  PrimState<2>{1., 1., {0.7276, 0.}},
                  PrimState<2>{0.8, 1., {0., 0.}},
                  PrimState<2>{1., 1., {0., 0.7276}}         },

        Quadrants{PrimState<2>{1., 1., {0., -0.3}},
                  PrimState<2>{2., 1., {0., 0.3}},
                  PrimState<2>{1.0625, 0.4, {0., 0.8145}},
                  PrimState<2>{0.5313, 0.4, {0., 0.4276}}    },

        Quadrants{PrimState<2>{2., 8., {0., -0.5606}},
                  PrimState<2>{1., 8., {0., -1.2172}},
                  PrimState<2>{0.4736, 2.6667, {0., 1.2172}},
                  PrimState<2>{0.9474, 2.6667, {0., 1.1606}} },

        Quadrants{PrimState<2>{1., 1., {0.1, -0.3}},
                  PrimState<2>{0.5197, 0.4, {-0.6259, -0.3}},
                  PrimState<2>{0.8, 0.4, {0.1, -0.3}},
                  PrimState<2>{0.5313, 0.4, {0.1, 0.4276}}   },

        Quadrants{PrimState<2>{0.5313, 0.4, {0.1, 0.1}},
                  PrimState<2>{1.0222, 1., {-0.6179, 0.1}},
                  PrimState<2>{0.8, 1., {0.1, 0.1}},
                  PrimState<2>{1., 1., {0.1, 0.8276}}        },

        Quadrants{PrimState<2>{1., 1., {0., -0.4}},
                  PrimState<2>{2., 1., {0., -0.3}},
                  PrimState<2>{1.0625, 0.4, {0., 0.2145}},
                  PrimState<2>{0.5197, 0.4, {0., -1.1259}}   },

        Quadrants{PrimState<2>{1., 1., {0., 1.}},
                  PrimState<2>{2., 1., {0., -0.3}},
                  PrimState<2>{1.0625, 0.4, {0., 0.2145}},
                  PrimState<2>{0.5197, 0.4, {0., 0.2741}}    },

        Quadrants{PrimState<2>{1., 1., {0., 0.3}},
                  PrimState<2>{2., 1., {0., -0.3}},
                  PrimState<2>{1.0625, 0.4, {0., 0.2145}},
                  PrimState<2>{0.5197, 0.4, {0., -0.4259}}   },
    };

    // Configuration 3 on eight octants, the three-dimensional case of the
    // article. Reading down the column of velocities shows the rule: 1.206 in
    // every direction the octant is on the low side of.
    inline const Octants octants_config_3 = {
        PrimState<3>{1.5,    1.5,   {0., 0., 0.}         }, // z >= z0, upper right
        PrimState<3>{0.5323, 0.3,   {1.206, 0., 0.}      }, //          upper left
        PrimState<3>{0.138,  0.029, {1.206, 1.206, 0.}   }, //          lower left
        PrimState<3>{0.5323, 0.3,   {0., 1.206, 0.}      }, //          lower right
        PrimState<3>{0.5323, 0.3,   {0., 0., 1.206}      }, // z <  z0, upper right
        PrimState<3>{0.138,  0.029, {1.206, 0., 1.206}   }, //          upper left
        PrimState<3>{0.138,  0.029, {1.206, 1.206, 1.206}}, //          lower left
        PrimState<3>{0.138,  0.029, {0., 1.206, 1.206}   }, //          lower right
    };

    // The octant states of a configuration. Configuration 3 is the only one the
    // article extends to three dimensions and the only one tabulated here, and
    // `--riemann-config` is restricted to it in euler_3d: reaching the throw
    // means a configuration was added to that restriction and not to this table.
    inline const Octants& octants(int configuration)
    {
        if (configuration != 3)
        {
            throw std::runtime_error("Lax & Liu configuration " + std::to_string(configuration) + " has no three-dimensional states");
        }
        return octants_config_3;
    }

    // Which of the four quadrants, or of the eight octants, a point falls in.
    template <std::size_t dim>
    std::size_t region(const auto& x)
    {
        const double x0 = selected_interface();

        const bool right = x[0] >= x0;
        const bool above = x[1] >= x0;

        const std::size_t quadrant = right ? (above ? 0 : 3) : (above ? 1 : 2);

        if constexpr (dim == 2)
        {
            return quadrant;
        }
        else
        {
            return quadrant + (x[2] >= x0 ? 0 : 4);
        }
    }

    template <class Field>
    void init_fn(Field& u, const typename Field::cell_t& cell, EOS::IdealGas eos)
    {
        static constexpr std::size_t dim = Field::dim;

        const auto x = cell.center();

        if constexpr (dim == 2)
        {
            const auto& quadrants = configurations[static_cast<std::size_t>(selected_config()) - 1];
            u[cell]               = prim2cons<2>(quadrants[region<2>(x)], eos);
        }
        else
        {
            u[cell] = prim2cons<3>(octants(selected_config())[region<3>(x)], eos);
        }
    }

    template <class Field>
    void bc_fn(Field& u, double& /*t*/, EOS::IdealGas /*eos*/)
    {
        bc::outflow(u);
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

    // The option is declared by the binary it belongs to, so the values it
    // accepts are those that dimension actually has: nineteen configurations in
    // two dimensions, only the one the article extends in three. A number the
    // case cannot honour is then a parse error, not a surprise at run time.
    template <std::size_t dim>
    void add_options(CLI::App& app)
    {
        auto* option = app.add_option("--riemann-config", selected_config(), "Lax & Liu configuration of the lax_liu test case")
                           ->capture_default_str()
                           ->group("Test case parameters");

        if constexpr (dim == 2)
        {
            option->check(CLI::Range(1, 19));
        }
        else
        {
            option->check(CLI::IsMember({3}));
        }

        app.add_option("--riemann-interface",
                       selected_interface(),
                       "Where the quadrants of the lax_liu test case meet (0.5 is the published convention)")
            ->capture_default_str()
            ->check(CLI::Range(0., 1.))
            ->group("Test case parameters");
    }

    template <class Field>
    test_case::TestCase<Field> definition()
    {
        static_assert(Field::dim == 2 || Field::dim == 3, "this test case is two- or three-dimensional");

        return {.box     = &box_fn<Field::dim>,
                .init    = &init_fn<Field>,
                .bc      = &bc_fn<Field>,
                .eos     = EOS::ideal_gas(1.4),
                .options = &add_options<Field::dim>};
    }
}

REGISTER_TEST_CASE(lax_liu, test_case::lax_liu, 2, 3)
