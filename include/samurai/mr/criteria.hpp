// Copyright 2018-2024 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include "../cell_flag.hpp"
#include "../operators_base.hpp"

namespace samurai
{

    template <std::size_t dim, class TInterval>
    class to_coarsen_mr_op : public field_operator_base<dim, TInterval>
    {
      public:

        INIT_OPERATOR(to_coarsen_mr_op)

        template <class T1, class T2>
        // inline void operator()(Dim<1>, const T1& detail, const T3&
        // max_detail, T2 &tag, double eps, std::size_t min_lev) const
        inline void operator()(Dim<1>, const T1& detail, T2& tag, double eps, std::size_t min_lev) const

        {
            using namespace math;
            constexpr auto size    = T1::size;
            std::size_t fine_level = level + 1;

            if (fine_level > min_lev)
            {
                // auto maxd = xt::view(max_detail, level);

                if constexpr (size == 1)
                {
                    // auto mask = abs(detail(level, 2*i))/maxd < eps;
                    xt::xtensor<bool, 1> mask = abs(detail(fine_level, 2 * i)) < eps; // NO normalization

                    tag(fine_level, 2 * i)     = mask * static_cast<int>(CellFlag::coarsen) + !mask * tag(fine_level, 2 * i);
                    tag(fine_level, 2 * i + 1) = mask * static_cast<int>(CellFlag::coarsen) + !mask * tag(fine_level, 2 * i + 1);
                }
                else
                {
                    // auto mask = xt::sum((abs(detail(level, 2*i))/maxd <
                    // eps), {1}) > (size-1);
                    constexpr std::size_t axis = T1::is_soa ? 0 : 1;

                    xt::xtensor<bool, 1> mask = sum<axis>((abs(detail(fine_level, 2 * i)) < eps)) > (size - 1); // No normalization

                    tag(fine_level, 2 * i)     = mask * static_cast<int>(CellFlag::coarsen) + !mask * tag(fine_level, 2 * i);
                    tag(fine_level, 2 * i + 1) = mask * static_cast<int>(CellFlag::coarsen) + !mask * tag(fine_level, 2 * i + 1);
                }
            }
        }

        template <class T1, class T2>
        inline void operator()(Dim<2>, const T1& detail, T2& tag, double eps, std::size_t min_lev) const
        {
            using namespace math;

            constexpr auto size    = T1::size;
            std::size_t fine_level = level + 1;

            if (fine_level > min_lev)
            {
                if constexpr (size == 1)
                {
                    xt::xtensor<bool, 1> mask = (abs(detail(fine_level, 2 * i, 2 * j)) < eps)
                                             && (abs(detail(fine_level, 2 * i + 1, 2 * j)) < eps)
                                             && (abs(detail(fine_level, 2 * i, 2 * j + 1)) < eps)
                                             && (abs(detail(fine_level, 2 * i + 1, 2 * j + 1)) < eps);

                    static_nested_loop<dim - 1, 0, 2>(
                        [&](auto stencil)
                        {
                            for (int ii = 0; ii < 2; ++ii)
                            {
                                tag(fine_level, 2 * i + ii, 2 * index + stencil) = mask * static_cast<int>(CellFlag::coarsen)
                                                                                 + !mask * tag(fine_level, 2 * i + ii, 2 * index + stencil);
                            }
                        });
                    // xt::masked_view(tag(fine_level, 2 * i, 2 * j), mask)         = static_cast<int>(CellFlag::coarsen);
                    // xt::masked_view(tag(fine_level, 2 * i + 1, 2 * j), mask)     = static_cast<int>(CellFlag::coarsen);
                    // xt::masked_view(tag(fine_level, 2 * i, 2 * j + 1), mask)     = static_cast<int>(CellFlag::coarsen);
                    // xt::masked_view(tag(fine_level, 2 * i + 1, 2 * j + 1), mask) = static_cast<int>(CellFlag::coarsen);
                }
                else
                {
                    // auto mask = xt::sum((abs(detail(level, 2*i  ,
                    // 2*j))/maxd < eps) &&
                    //                     (abs(detail(level, 2*i+1,
                    //                     2*j))/maxd < eps) &&
                    //                     (abs(detail(level, 2*i  ,
                    //                     2*j+1))/maxd < eps) &&
                    //                     (abs(detail(level, 2*i+1,
                    //                     2*j+1))/maxd < eps), {1}) > (size-1);
                    constexpr std::size_t axis = T1::is_soa ? 0 : 1;

                    xt::xtensor<bool, 1> mask = sum<axis>((abs(detail(fine_level, 2 * i, 2 * j)) < eps)
                                                          && (abs(detail(fine_level, 2 * i + 1, 2 * j)) < eps)
                                                          && (abs(detail(fine_level, 2 * i, 2 * j + 1)) < eps)
                                                          && (abs(detail(fine_level, 2 * i + 1, 2 * j + 1)) < eps))
                                              > (size - 1);

                    static_nested_loop<dim - 1, 0, 2>(
                        [&](auto stencil)
                        {
                            for (int ii = 0; ii < 2; ++ii)
                            {
                                tag(fine_level, 2 * i + ii, 2 * index + stencil) = mask * static_cast<int>(CellFlag::coarsen)
                                                                                 + !mask * tag(fine_level, 2 * i + ii, 2 * index + stencil);
                            }
                        });

                    // xt::masked_view(tag(fine_level, 2 * i, 2 * j), mask)         = static_cast<int>(CellFlag::coarsen);
                    // xt::masked_view(tag(fine_level, 2 * i + 1, 2 * j), mask)     = static_cast<int>(CellFlag::coarsen);
                    // xt::masked_view(tag(fine_level, 2 * i, 2 * j + 1), mask)     = static_cast<int>(CellFlag::coarsen);
                    // xt::masked_view(tag(fine_level, 2 * i + 1, 2 * j + 1), mask) = static_cast<int>(CellFlag::coarsen);
                }
            }
        }

        template <class T1, class T2>
        inline void operator()(Dim<3>, const T1& detail, T2& tag, double eps, std::size_t min_lev) const
        {
            using namespace math;

            constexpr auto size    = T1::size;
            std::size_t fine_level = level + 1;

            if (fine_level > min_lev)
            {
                // auto maxd = xt::view(max_detail, level);

                if constexpr (size == 1)
                {
                    // auto mask = (abs(detail(level, 2*i  ,   2*j))/maxd <
                    // eps) and
                    //             (abs(detail(level, 2*i+1,   2*j))/maxd <
                    //             eps) and (abs(detail(level, 2*i  ,
                    //             2*j+1))/maxd < eps) and
                    //             (abs(detail(level, 2*i+1, 2*j+1))/maxd <
                    //             eps);
                    xt::xtensor<bool, 1> mask = (abs(detail(fine_level, 2 * i, 2 * j, 2 * k)) < eps)
                                             && (abs(detail(fine_level, 2 * i + 1, 2 * j, 2 * k)) < eps)
                                             && (abs(detail(fine_level, 2 * i, 2 * j + 1, 2 * k)) < eps)
                                             && (abs(detail(fine_level, 2 * i + 1, 2 * j + 1, 2 * k)) < eps)
                                             && (abs(detail(fine_level, 2 * i, 2 * j, 2 * k + 1)) < eps)
                                             && (abs(detail(fine_level, 2 * i + 1, 2 * j, 2 * k + 1)) < eps)
                                             && (abs(detail(fine_level, 2 * i, 2 * j + 1, 2 * k + 1)) < eps)
                                             && (abs(detail(fine_level, 2 * i + 1, 2 * j + 1, 2 * k + 1)) < eps);

                    tag(fine_level, 2 * i, 2 * j, 2 * k) = mask * static_cast<int>(CellFlag::coarsen)
                                                         + !mask * tag(fine_level, 2 * i, 2 * j, 2 * k);
                    tag(fine_level, 2 * i + 1, 2 * j, 2 * k) = mask * static_cast<int>(CellFlag::coarsen)
                                                             + !mask * tag(fine_level, 2 * i + 1, 2 * j, 2 * k);
                    tag(fine_level, 2 * i, 2 * j + 1, 2 * k) = mask * static_cast<int>(CellFlag::coarsen)
                                                             + !mask * tag(fine_level, 2 * i, 2 * j + 1, 2 * k);
                    tag(fine_level, 2 * i + 1, 2 * j + 1, 2 * k) = mask * static_cast<int>(CellFlag::coarsen)
                                                                 + !mask * tag(fine_level, 2 * i + 1, 2 * j + 1, 2 * k);
                    tag(fine_level, 2 * i, 2 * j, 2 * k + 1) = mask * static_cast<int>(CellFlag::coarsen)
                                                             + !mask * tag(fine_level, 2 * i, 2 * j, 2 * k + 1);
                    tag(fine_level, 2 * i + 1, 2 * j, 2 * k + 1) = mask * static_cast<int>(CellFlag::coarsen)
                                                                 + !mask * tag(fine_level, 2 * i + 1, 2 * j, 2 * k + 1);
                    tag(fine_level, 2 * i, 2 * j + 1, 2 * k + 1) = mask * static_cast<int>(CellFlag::coarsen)
                                                                 + !mask * tag(fine_level, 2 * i, 2 * j + 1, 2 * k + 1);
                    tag(fine_level, 2 * i + 1, 2 * j + 1, 2 * k + 1) = mask * static_cast<int>(CellFlag::coarsen)
                                                                     + !mask * tag(fine_level, 2 * i + 1, 2 * j + 1, 2 * k + 1);
                }
                else
                {
                    // auto mask = xt::sum((abs(detail(level, 2*i  ,
                    // 2*j))/maxd < eps) and
                    //                     (abs(detail(level, 2*i+1,
                    //                     2*j))/maxd < eps) and
                    //                     (abs(detail(level, 2*i  ,
                    //                     2*j+1))/maxd < eps) and
                    //                     (abs(detail(level, 2*i+1,
                    //                     2*j+1))/maxd < eps), {1}) > (size-1);

                    constexpr std::size_t axis = T1::is_soa ? 0 : 1;

                    xt::xtensor<bool, 1> mask = sum<axis>((abs(detail(fine_level, 2 * i, 2 * j, 2 * k)) < eps)
                                                          && (abs(detail(fine_level, 2 * i + 1, 2 * j, 2 * k)) < eps)
                                                          && (abs(detail(fine_level, 2 * i, 2 * j + 1, 2 * k)) < eps)
                                                          && (abs(detail(fine_level, 2 * i + 1, 2 * j + 1, 2 * k)) < eps)
                                                          && (abs(detail(fine_level, 2 * i, 2 * j, 2 * k + 1)) < eps)
                                                          && (abs(detail(fine_level, 2 * i + 1, 2 * j, 2 * k + 1)) < eps)
                                                          && (abs(detail(fine_level, 2 * i, 2 * j + 1, 2 * k + 1)) < eps)
                                                          && (abs(detail(fine_level, 2 * i + 1, 2 * j + 1, 2 * k + 1)) < eps))
                                              > (size - 1);

                    tag(fine_level, 2 * i, 2 * j, 2 * k) = mask * static_cast<int>(CellFlag::coarsen)
                                                         + !mask * tag(fine_level, 2 * i, 2 * j, 2 * k);
                    tag(fine_level, 2 * i + 1, 2 * j, 2 * k) = mask * static_cast<int>(CellFlag::coarsen)
                                                             + !mask * tag(fine_level, 2 * i + 1, 2 * j, 2 * k);
                    tag(fine_level, 2 * i, 2 * j + 1, 2 * k) = mask * static_cast<int>(CellFlag::coarsen)
                                                             + !mask * tag(fine_level, 2 * i, 2 * j + 1, 2 * k);
                    tag(fine_level, 2 * i + 1, 2 * j + 1, 2 * k) = mask * static_cast<int>(CellFlag::coarsen)
                                                                 + !mask * tag(fine_level, 2 * i + 1, 2 * j + 1, 2 * k);
                    tag(fine_level, 2 * i, 2 * j, 2 * k + 1) = mask * static_cast<int>(CellFlag::coarsen)
                                                             + !mask * tag(fine_level, 2 * i, 2 * j, 2 * k + 1);
                    tag(fine_level, 2 * i + 1, 2 * j, 2 * k + 1) = mask * static_cast<int>(CellFlag::coarsen)
                                                                 + !mask * tag(fine_level, 2 * i + 1, 2 * j, 2 * k + 1);
                    tag(fine_level, 2 * i, 2 * j + 1, 2 * k + 1) = mask * static_cast<int>(CellFlag::coarsen)
                                                                 + !mask * tag(fine_level, 2 * i, 2 * j + 1, 2 * k + 1);
                    tag(fine_level, 2 * i + 1, 2 * j + 1, 2 * k + 1) = mask * static_cast<int>(CellFlag::coarsen)
                                                                     + !mask * tag(fine_level, 2 * i + 1, 2 * j + 1, 2 * k + 1);
                }
            }
        }
    };

    template <class... CT>
    inline auto to_coarsen_mr(CT&&... e)
    {
        return make_field_operator_function<to_coarsen_mr_op>(std::forward<CT>(e)...);
    }

    template <std::size_t dim, class TInterval>
    class to_refine_mr_op : public field_operator_base<dim, TInterval>
    {
      public:

        INIT_OPERATOR(to_refine_mr_op)

        template <std::size_t size, bool is_soa, class T1>
        inline xt::xtensor<bool, 1> get_mask(const T1& detail_view, double eps) const
        {
            using namespace math;

            if constexpr (size == 1)
            {
                return eval(abs(detail_view) > eps); // No normalization
            }
            else
            {
                constexpr std::size_t axis = is_soa ? 0 : 1;
                return eval(sum<axis>(abs(detail_view) > eps) > 0);
            }
        }

        template <class T1, class T2>
        inline void operator()(Dim<dim>, const T1& detail, T2& tag, double eps, std::size_t max_level) const
        {
            using namespace math;
            constexpr auto size    = T1::size;
            std::size_t fine_level = level + 1;

            auto mask_ghost = get_mask<size, T1::is_soa>(detail(fine_level - 1, i, index), eps / (1 << dim));

            static_nested_loop<dim - 1, 0, 2>(
                [&](auto stencil)
                {
                    tag(fine_level, 2 * i, 2 * index + stencil) |= mask_ghost * static_cast<int>(CellFlag::keep);
                    tag(fine_level, 2 * i + 1, 2 * index + stencil) |= mask_ghost * static_cast<int>(CellFlag::keep);
                });

            if (fine_level < max_level)
            {
                static_nested_loop<dim - 1, 0, 2>(
                    [&](auto stencil)
                    {
                        for (int ii = 0; ii < 2; ++ii)
                        {
                            auto mask = get_mask<size, T1::is_soa>(detail(fine_level, 2 * i + ii, 2 * index + stencil), eps);

                            tag(fine_level, 2 * i + ii, 2 * index + stencil) = mask * static_cast<int>(CellFlag::refine)
                                                                             + !mask * tag(fine_level, 2 * i + ii, 2 * index + stencil);
                        }
                    });
            }
        }
    };

    template <class... CT>
    inline auto to_refine_mr(CT&&... e)
    {
        return make_field_operator_function<to_refine_mr_op>(std::forward<CT>(e)...);
    }

    template <std::size_t dim, class TInterval>
    class max_detail_mr_op : public field_operator_base<dim, TInterval>
    {
      public:

        INIT_OPERATOR(max_detail_mr_op)

        template <class T1>
        inline void operator()(Dim<2>, const T1& detail, double& max_detail) const
        {
            auto ii = 2 * i;
            ii.step = 1;

            max_detail = std::max(max_detail,
                                  xt::amax(xt::maximum(abs(detail(level + 1, ii, 2 * j)), abs(detail(level + 1, ii, 2 * j + 1))))[0]);
        }
    };

    template <class... CT>
    inline auto max_detail_mr(CT&&... e)
    {
        return make_field_operator_function<max_detail_mr_op>(std::forward<CT>(e)...);
    }
} // namespace samurai
