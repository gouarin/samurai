// Copyright 2025 the samurai team
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <samurai/numeric/prediction.hpp>
#include <samurai/operators_base.hpp>

#include "variables.hpp"

namespace detail
{
    template <typename Tuple1, typename Tuple2, typename F, std::size_t... Is>
    constexpr void zip_apply_impl(F&& f, Tuple1&& t1, Tuple2&& t2, std::index_sequence<Is...>)
    {
        (f(std::get<Is>(std::forward<Tuple1>(t1)), std::get<Is>(std::forward<Tuple2>(t2))), ...);
    }

    template <typename Tuple1, typename Tuple2, typename F>
    constexpr void zip_apply(F&& f, Tuple1&& t1, Tuple2&& t2)
    {
        constexpr std::size_t size1 = std::tuple_size_v<std::remove_reference_t<Tuple1>>;
        constexpr std::size_t size2 = std::tuple_size_v<std::remove_reference_t<Tuple2>>;

        static_assert(size1 == size2, "Tuples must have the same size");

        zip_apply_impl(std::forward<F>(f), std::forward<Tuple1>(t1), std::forward<Tuple2>(t2), std::make_index_sequence<size1>{});
    }

    template <std::size_t Dim>
    consteval auto cube_children()
    {
        std::array<std::array<int, Dim>, (1u << Dim)> children{};
        for (std::size_t i = 0; i < (1u << Dim); ++i)
        {
            for (std::size_t d = 0; d < Dim; ++d)
            {
                children[i][d] = (i >> d) & 1; // bit d
            }
        }
        return children;
    }

    template <std::size_t Dim>
    constexpr auto tail_as_xt(const std::array<int, Dim>& corner)
    {
        xt::xtensor_fixed<int, xt::xshape<Dim - 1>> t;
        for (std::size_t d = 1; d < Dim; ++d)
        {
            t[d - 1] = corner[d];
        }
        return t;
    }
}

template <std::size_t dim, class TInterval>
class Euler_prediction_op : public samurai::field_operator_base<dim, TInterval>
{
  public:

    INIT_OPERATOR(Euler_prediction_op)

    inline void operator()(samurai::Dim<dim>, auto& dest, const auto& src, auto eos) const
    {
        using EulerConsVar = EulerLayout<dim>;
        using field_t      = std::decay_t<decltype(src)>;

        constexpr std::size_t pred_order = field_t::mesh_t::config_t::prediction_stencil_radius;

        //
        // Step 1/4 — Compute dest values using the default prediction of order pred_order
        //
        samurai::prediction<pred_order, true>(dest, src)(level, i, index);

        if constexpr (!std::decay_t<decltype(src)>::is_scalar)
        {
            if (src.name() == "euler")
            {
                auto i_f     = i << 1;
                i_f.step     = 2;
                auto index_f = index << 1;

                //
                // Step 2/4 — Compute mask for negative density
                //
                const auto mask_rho = std::apply(
                    [&](const auto&... child)
                    {
                        return ((dest(EulerConsVar::rho, level + 1, i_f + child[0], index_f + detail::tail_as_xt(child)) < 0.0) || ...);
                    },
                    detail::cube_children<dim>());

                //
                // Step 3/4 — Compute the pressure and the mask for negative pressure
                //
                std::array<xt::xtensor<double, 1>, (1u << dim)> pressure;
                pressure.fill(xt::empty<double>({i.size()}));

                auto compute_pressure = [&](auto& p, const auto& child)
                {
                    auto xt_child = detail::tail_as_xt(child);
                    auto rho      = dest(EulerConsVar::rho, level + 1, i_f + child[0], index_f + xt_child);
                    auto e        = xt::eval(dest(EulerConsVar::rhoE, level + 1, i_f + child[0], index_f + xt_child)
                                             / dest(EulerConsVar::rho, level + 1, i_f + child[0], index_f + xt_child));

                    for (std::size_t d = 0; d < dim; ++d)
                    {
                        auto v_d = dest(EulerConsVar::mom(d), level + 1, i_f + child[0], index_f + xt_child)
                                 / dest(EulerConsVar::rho, level + 1, i_f + child[0], index_f + xt_child);
                        e -= 0.5 * v_d * v_d;
                    }
                    p = eos.p(rho, e);
                };

                detail::zip_apply(compute_pressure, pressure, detail::cube_children<dim>());

                const auto mask_p = std::apply(
                    [&](auto&... p)
                    {
                        return ((p < 0.0) || ...);
                    },
                    pressure);

                //
                // Step 4/4 — Apply prediction of order 0 on masked cells
                //
                samurai::apply_on_masked(mask_rho || mask_p,
                                         [&](auto& ie)
                                         {
                                             std::apply(
                                                 [&](const auto&... child)
                                                 {
                                                     ((xt::view(dest(level + 1, i_f + child[0], index_f + detail::tail_as_xt(child)),
                                                                ie) = xt::view(src(level, i, index), ie)),
                                                      ...);
                                                 },
                                                 detail::cube_children<dim>());
                                         });
            }
        }
    }
};
