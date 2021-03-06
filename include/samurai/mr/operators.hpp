// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xmasked_view.hpp>

#include "../operators_base.hpp"
#include "cell_flag.hpp"
#include "prediction.hpp"
namespace samurai
{
    /***********************
     * projection operator *
     ***********************/

    template<class TInterval>
    class projection_op_: public field_operator_base<TInterval>
    {
    public:
        INIT_OPERATOR(projection_op_)

        template<class T>
        inline void operator()(Dim<1>, T& field) const
        {
            field(level, i) = .5 * (field(level + 1, 2 * i) + field(level + 1, 2 * i + 1));
        }

        template<class T>
        inline void operator()(Dim<2>, T& field) const
        {
            field(level, i, j) = .25 * (field(level + 1, 2 * i, 2 * j) +
                                        field(level + 1, 2 * i, 2 * j + 1) +
                                        field(level + 1, 2 * i + 1, 2 * j) +
                                        field(level + 1, 2 * i + 1, 2 * j + 1));
        }

        template<class T>
        inline void operator()(Dim<3>, T& field) const
        {
            field(level, i, j, k) = .125 * (field(level - 1, 2 * i, 2 * j, 2 * k) +
                                            field(level - 1, 2 * i + 1, 2 * j, 2 * k) +
                                            field(level - 1, 2 * i, 2 * j + 1, 2 * k) +
                                            field(level - 1, 2 * i + 1, 2 * j + 1, 2 * k) +
                                            field(level - 1, 2 * i, 2 * j + 1, 2 * k + 1) +
                                            field(level - 1, 2 * i + 1, 2 * j + 1, 2 * k + 1));
        }
    };

    template<class T>
    inline auto projection(T&& field)
    {
        return make_field_operator_function<projection_op_>(std::forward<T>(field));
    }

    /***********************
     * prediction operator *
     ***********************/

    template<class TInterval>
    class prediction_op : public field_operator_base<TInterval> {
      public:
        INIT_OPERATOR(prediction_op)

        template<class T>
        inline void operator()(Dim<1>, T &field) const
        {
            auto qs_i = Qs_i<1>(field, level - 1, i >> 1);

            auto even_i = i.even_elements();
            if (even_i.is_valid())
            {
                auto coarse_even_i = even_i >> 1;
                auto dec_even = (i.start & 1) ? 1 : 0;
                field(level, even_i) = field(level - 1, coarse_even_i)
                                     + xt::view(qs_i, xt::range(dec_even, qs_i.shape()[0]));
            }

            auto odd_i = i.odd_elements();
            if (odd_i.is_valid())
            {
                auto coarse_odd_i = odd_i >> 1;
                auto dec_odd = (i.end & 1) ? 1 : 0;
                field(level, odd_i) = field(level - 1, coarse_odd_i)
                                    - xt::view(qs_i, xt::range(0, safe_subs<int>(qs_i.shape()[0], dec_odd)));
            }
        }

        template<class T>
        inline void operator()(Dim<2>, T &field) const
        {
            auto qs_i = Qs_i<1>(field, level - 1, i >> 1, j >> 1);
            auto qs_j = Qs_j<1>(field, level - 1, i >> 1, j >> 1);
            auto qs_ij = Qs_ij<1>(field, level - 1, i >> 1, j >> 1);

            if (j & 1)
            {
                auto even_i = i.even_elements();
                if (even_i.is_valid())
                {
                    auto coarse_even_i = even_i >> 1;
                    auto dec_even = (i.start & 1) ? 1 : 0;
                    field(level, even_i, j) = field(level - 1, coarse_even_i, j >> 1)
                                            + xt::view(qs_i, xt::range(dec_even, qs_i.shape()[0]))
                                            - xt::view(qs_j, xt::range(dec_even, qs_j.shape()[0]))
                                            + xt::view(qs_ij, xt::range(dec_even, qs_ij.shape()[0]));
                }

                auto odd_i = i.odd_elements();
                if (odd_i.is_valid())
                {
                    auto coarse_odd_i = odd_i >> 1;
                    auto dec_odd = (i.end & 1) ? 1 : 0;
                    field(level, odd_i, j) = field(level - 1, coarse_odd_i, j >> 1)
                                           - xt::view(qs_i, xt::range(0, safe_subs<int>(qs_i.shape()[0], dec_odd)))
                                           - xt::view(qs_j, xt::range(0, safe_subs<int>(qs_j.shape()[0], dec_odd)))
                                           - xt::view(qs_ij, xt::range(0, safe_subs<int>(qs_ij.shape()[0], dec_odd)));
                }
            }
            else
            {
                auto even_i = i.even_elements();
                if (even_i.is_valid())
                {
                    auto coarse_even_i = even_i >> 1;
                    auto dec_even = (i.start & 1) ? 1 : 0;
                    field(level, even_i, j) = field(level - 1, coarse_even_i, j >> 1)
                                            + xt::view(qs_i, xt::range(dec_even, qs_i.shape()[0]))
                                            + xt::view(qs_j, xt::range(dec_even, qs_j.shape()[0]))
                                            - xt::view(qs_ij, xt::range(dec_even, qs_ij.shape()[0]));
                }

                auto odd_i = i.odd_elements();
                if (odd_i.is_valid())
                {
                    auto coarse_odd_i = odd_i >> 1;
                    auto dec_odd = (i.end & 1) ? 1 : 0;
                    field(level, odd_i, j) = field(level - 1, coarse_odd_i, j >> 1)
                                           - xt::view(qs_i, xt::range(0, safe_subs<int>(qs_i.shape()[0], dec_odd)))
                                           + xt::view(qs_j, xt::range(0, safe_subs<int>(qs_j.shape()[0], dec_odd)))
                                           + xt::view(qs_ij, xt::range(0, safe_subs<int>(qs_ij.shape()[0], dec_odd)));
                }
            }
        }
    };

    template<class T>
    inline auto prediction(T &&field)
    {
        return make_field_operator_function<prediction_op>(
            std::forward<T>(field));
    }

    template<class interval_t, class coord_index_t, class field_t>
    inline void compute_prediction_impl(
        const std::size_t level, const interval_t i,
        const xt::xtensor_fixed<coord_index_t, xt::xshape<0>> &index_yz,
        const field_t &field, field_t &new_field)
    {
        auto ii = i << 1;
        ii.step = 2;

        auto qs_i = Qs_i<1>(field, level, i);

        new_field(level + 1, ii) = field(level, i) + qs_i;

        new_field(level + 1, ii + 1) = field(level, i) - qs_i;
    }

    template<class interval_t, class coord_index_t, class field_t>
    inline void compute_prediction_impl(
        const std::size_t level, const interval_t i,
        const xt::xtensor_fixed<coord_index_t, xt::xshape<1>> &index_yz,
        const field_t &field, field_t &new_field)
    {
        auto ii = i << 1;
        ii.step = 2;

        auto j = index_yz[0];
        auto jj = j << 1;

        auto qs_i = Qs_i<1>(field, level, i, j);
        auto qs_j = Qs_j<1>(field, level, i, j);
        auto qs_ij = Qs_ij<1>(field, level, i, j);

        new_field(level + 1, ii, jj) = field(level, i, j) + qs_i + qs_j - qs_ij;
        new_field(level + 1, ii + 1, jj) = field(level, i, j) - qs_i + qs_j + qs_ij;
        new_field(level + 1, ii, jj + 1) = field(level, i, j) + qs_i - qs_j + qs_ij;
        new_field(level + 1, ii + 1, jj + 1) = field(level, i, j) - qs_i - qs_j - qs_ij;
    }

    template<class interval_t, class coord_index_t, class field_t,
             std::size_t dim>
    inline void compute_prediction(
        const std::size_t level, const interval_t i,
        const xt::xtensor_fixed<coord_index_t, xt::xshape<dim>> &index_yz,
        const field_t &field, field_t &new_field)
    {
        compute_prediction_impl(level, i, index_yz, field, new_field);
    }
    /********************
     * maximum operator *
     ********************/

    template<class TInterval>
    class maximum_op : public field_operator_base<TInterval> {
      public:
        INIT_OPERATOR(maximum_op)

        template<class T>
        inline void operator()(Dim<1>, T &field) const
        {
            xt::xtensor<bool, 1> mask =
                (field(level + 1, 2 * i) & static_cast<int>(CellFlag::keep)) |
                (field(level + 1, 2 * i + 1) &
                 static_cast<int>(CellFlag::keep));

            xt::masked_view(field(level + 1, 2 * i), mask) |=
                static_cast<int>(CellFlag::keep);
            xt::masked_view(field(level + 1, 2 * i + 1), mask) |=
                static_cast<int>(CellFlag::keep);

            xt::masked_view(field(level, i), mask) |=
                static_cast<int>(CellFlag::keep);

            mask = (field(level + 1, 2 * i) &
                    static_cast<int>(CellFlag::coarsen)) &
                   (field(level + 1, 2 * i + 1) &
                    static_cast<int>(CellFlag::coarsen));

            xt::masked_view(field(level + 1, 2 * i), !mask) &=
                ~static_cast<int>(CellFlag::coarsen);
            xt::masked_view(field(level + 1, 2 * i + 1), !mask) &=
                ~static_cast<int>(CellFlag::coarsen);
            xt::masked_view(field(level, i), mask) |=
                static_cast<int>(CellFlag::keep);
        }

        template<class T>
        inline void operator()(Dim<2>, T &field) const
        {
            xt::xtensor<bool, 1> mask =
                (field(level + 1, 2 * i, 2 * j) &
                 static_cast<int>(CellFlag::keep)) |
                (field(level + 1, 2 * i + 1, 2 * j) &
                 static_cast<int>(CellFlag::keep)) |
                (field(level + 1, 2 * i, 2 * j + 1) &
                 static_cast<int>(CellFlag::keep)) |
                (field(level + 1, 2 * i + 1, 2 * j + 1) &
                 static_cast<int>(CellFlag::keep));

            xt::masked_view(field(level + 1, 2 * i, 2 * j), mask) |=
                static_cast<int>(CellFlag::keep);
            xt::masked_view(field(level + 1, 2 * i + 1, 2 * j), mask) |=
                static_cast<int>(CellFlag::keep);
            xt::masked_view(field(level + 1, 2 * i, 2 * j + 1), mask) |=
                static_cast<int>(CellFlag::keep);
            xt::masked_view(field(level + 1, 2 * i + 1, 2 * j + 1), mask) |=
                static_cast<int>(CellFlag::keep);

            xt::masked_view(field(level, i, j), mask) |=
                static_cast<int>(CellFlag::keep);

            mask = (field(level + 1, 2 * i, 2 * j) &
                    static_cast<int>(CellFlag::coarsen)) &
                   (field(level + 1, 2 * i + 1, 2 * j) &
                    static_cast<int>(CellFlag::coarsen)) &
                   (field(level + 1, 2 * i, 2 * j + 1) &
                    static_cast<int>(CellFlag::coarsen)) &
                   (field(level + 1, 2 * i + 1, 2 * j + 1) &
                    static_cast<int>(CellFlag::coarsen));

            xt::masked_view(field(level + 1, 2 * i, 2 * j), !mask) &=
                ~static_cast<int>(CellFlag::coarsen);
            xt::masked_view(field(level + 1, 2 * i + 1, 2 * j), !mask) &=
                ~static_cast<int>(CellFlag::coarsen);
            xt::masked_view(field(level + 1, 2 * i, 2 * j + 1), !mask) &=
                ~static_cast<int>(CellFlag::coarsen);
            xt::masked_view(field(level + 1, 2 * i + 1, 2 * j + 1), !mask) &=
                ~static_cast<int>(CellFlag::coarsen);
            xt::masked_view(field(level, i, j), mask) |=
                static_cast<int>(CellFlag::keep);
        }
    };

    template<class T>
    inline auto maximum(T &&field)
    {
        return make_field_operator_function<maximum_op>(std::forward<T>(field));
    }

    /*****************
     * copy operator *
     *****************/

    template<class TInterval>
    class copy_op : public field_operator_base<TInterval> {
      public:
        INIT_OPERATOR(copy_op)

        template<class T>
        inline void operator()(Dim<1>, T &dest, const T &src) const
        {
            dest(level, i) = src(level, i);
        }

        template<class T>
        inline void operator()(Dim<2>, T &dest, const T &src) const
        {
            dest(level, i, j) = src(level, i, j);
        }

        template<class T>
        inline void operator()(Dim<3>, T &dest, const T &src) const
        {
            dest(level, i, j, k) = src(level, i, j, k);
        }
    };

    template<class T>
    inline auto copy(T &&dest, T &&src)
    {
        return make_field_operator_function<copy_op>(std::forward<T>(dest),
                                                     std::forward<T>(src));
    }

    /*****************
     * copy operator *
     *****************/

    template<class TInterval>
    class balance_2to1_op : public field_operator_base<TInterval> {
      public:
        INIT_OPERATOR(balance_2to1_op)

        template<class T, class stencil_t>
        inline void operator()(Dim<1>, T &cell_flag, const stencil_t &stencil) const
        {
            cell_flag(level, i - stencil[0]) |= (cell_flag(level, i) & static_cast<int>(samurai::CellFlag::keep));
        }

        template<class T, class stencil_t>
        inline void operator()(Dim<2>, T &cell_flag, const stencil_t &stencil) const
        {
            cell_flag(level, i - stencil[0], j - stencil[1]) |= (cell_flag(level, i, j) & static_cast<int>(samurai::CellFlag::keep));
        }

        template<class T, class stencil_t>
        inline void operator()(Dim<3>, T &cell_flag, const stencil_t &stencil) const
        {
            cell_flag(level, i - stencil[0], j - stencil[1], k - stencil[2]) |= (cell_flag(level, i, j, k) & static_cast<int>(samurai::CellFlag::keep));
        }
    };

    template<class T, class stencil_t>
    inline auto balance_2to1(T &&cell_flag, stencil_t &&stencil)
    {
        return make_field_operator_function<balance_2to1_op>(
            std::forward<T>(cell_flag), std::forward<stencil_t>(stencil));
    }

    /***************************
     * compute detail operator *
     ***************************/

    template<class TInterval>
    class compute_detail_op : public field_operator_base<TInterval> {
      public:
        INIT_OPERATOR(compute_detail_op)

        template<class T>
        inline void operator()(Dim<1>, T &detail, const T &field) const
        {
            auto qs_i = xt::eval(Qs_i<1>(field, level, i));

            detail(level + 1, 2 * i) =
                field(level + 1, 2 * i) - (field(level, i) + qs_i);

            detail(level + 1, 2 * i + 1) =
                field(level + 1, 2 * i + 1) - (field(level, i) - qs_i);
        }

        template<class T>
        inline void operator()(Dim<2>, T &detail, const T &field) const
        {
            auto qs_i = Qs_i<1>(field, level, i, j);
            auto qs_j = Qs_j<1>(field, level, i, j);
            auto qs_ij = Qs_ij<1>(field, level, i, j);

            detail(level + 1, 2 * i, 2 * j) =
                field(level + 1, 2 * i, 2 * j) -
                (field(level, i, j) + qs_i + qs_j - qs_ij);

            detail(level + 1, 2 * i + 1, 2 * j) =
                field(level + 1, 2 * i + 1, 2 * j) -
                (field(level, i, j) - qs_i + qs_j + qs_ij);

            detail(level + 1, 2 * i, 2 * j + 1) =
                field(level + 1, 2 * i, 2 * j + 1) -
                (field(level, i, j) + qs_i - qs_j + qs_ij);

            detail(level + 1, 2 * i + 1, 2 * j + 1) =
                field(level + 1, 2 * i + 1, 2 * j + 1) -
                (field(level, i, j) - qs_i - qs_j - qs_ij);



            // This is what is done by Bihari and Harten 1999
            // // It seems the good choice.
            // detail(level + 1, 2 * i, 2 * j) =
            //     field(level + 1, 2 * i, 2 * j) -
            //     (field(level, i, j) - qs_i - qs_j + qs_ij);

            // detail(level + 1, 2 * i + 1, 2 * j) =
            //     field(level + 1, 2 * i + 1, 2 * j) -
            //     (field(level, i, j) + qs_i - qs_j - qs_ij);

            // detail(level + 1, 2 * i, 2 * j + 1) =
            //     field(level + 1, 2 * i, 2 * j + 1) -
            //     (field(level, i, j) - qs_i + qs_j - qs_ij);

            // detail(level + 1, 2 * i + 1, 2 * j + 1) =
            //     field(level + 1, 2 * i + 1, 2 * j + 1) -
            //     (field(level, i, j) + qs_i + qs_j + qs_ij);


        }
    };

    template<class T>
    inline auto compute_detail(T &&detail, T &&field)
    {
        return make_field_operator_function<compute_detail_op>(
            std::forward<T>(detail), std::forward<T>(field));
    }

    /*******************************
     * compute max detail operator *
     *******************************/

    template<class TInterval>
    class compute_max_detail_op : public field_operator_base<TInterval> {
      public:
        INIT_OPERATOR(compute_max_detail_op)

        template<class T, class U>
        inline void operator()(Dim<1>, const U &detail, T &max_detail) const
        {
            auto ii = 2 * i;
            ii.step = 1;
            auto max_view = xt::view(max_detail, level + 1);

            max_view = xt::maximum(max_view,
                                   xt::amax(xt::abs(detail(level + 1, ii)),
                                            {0})
                                  );
        }

        template<class T, class U>
        inline void operator()(Dim<2>, const U &detail, T &max_detail) const
        {
            auto ii = 2 * i;
            ii.step = 1;
            auto max_view = xt::view(max_detail, level + 1);

            max_view = xt::maximum(max_view,
                                   xt::amax(xt::maximum(xt::abs(detail(level + 1, ii, 2 * j)),
                                                        xt::abs(detail(level + 1, ii, 2 * j + 1))),
                                            {0})
                                  );
        }
    };

    template<class T, class U>
    inline auto compute_max_detail(U &&detail, T &&max_detail)
    {
        return make_field_operator_function<compute_max_detail_op>(
            std::forward<U>(detail), std::forward<T>(max_detail));
    }

    /*******************************
     * compute max detail operator *
     *******************************/

    template<class TInterval>
    class compute_max_detail_op_ : public field_operator_base<TInterval> {
      public:
        INIT_OPERATOR(compute_max_detail_op_)

        template<class T, class U>
        inline void operator()(Dim<1>, const U &detail, T &max_detail) const
        {
            max_detail[level] = std::max(
                max_detail[level], xt::amax(xt::abs(detail(level, i)))[0]);
        }

        template<class T, class U>
        inline void operator()(Dim<2>, const U &detail, T &max_detail) const
        {
            max_detail[level] = std::max(
                max_detail[level], xt::amax(xt::abs(detail(level, i, j)))[0]);
        }
    };

    template<class T, class U>
    inline auto compute_max_detail_(U &&detail, T &&max_detail)
    {
        return make_field_operator_function<compute_max_detail_op_>(
            std::forward<U>(detail), std::forward<T>(max_detail));
    }

    /***********************
     * to_coarsen operator *
     ***********************/

    template<class TInterval>
    class to_coarsen_op : public field_operator_base<TInterval> {
      public:
        INIT_OPERATOR(to_coarsen_op)

        template<class T, class U, class V>
        inline void operator()(Dim<1>, T &keep, const U &detail,
                        double eps) const
        {
            auto mask = xt::abs(detail(level + 1, 2 * i)) < eps;
            // auto mask = (.5 *
            //              (xt::abs(detail(level + 1, 2 * i)) +
            //               xt::abs(detail(level + 1, 2 * i + 1))) /
            //              max_detail[level + 1]) < eps;

            for (coord_index_t ii = 0; ii < 2; ++ii)
            {
                xt::masked_view(keep(level + 1, 2 * i + ii), mask) =
                    static_cast<int>(CellFlag::coarsen);
            }
        }

        template<class T, class U, class V>
        inline void operator()(Dim<2>, T &keep, const U &detail,
                        double eps) const
        {
            auto mask = xt::abs(detail(level + 1, 2 * i, 2 * j)) < eps;

            // auto mask = (0.25 *
            //              (xt::abs(detail(level + 1, 2 * i, 2 * j)) +
            //               xt::abs(detail(level + 1, 2 * i + 1, 2 * j)) +
            //               xt::abs(detail(level + 1, 2 * i, 2 * j + 1)) +
            //               xt::abs(detail(level + 1, 2 * i + 1, 2 * j + 1))) /
            //              max_detail[level + 1]) < eps;

            for (coord_index_t jj = 0; jj < 2; ++jj)
            {
                for (coord_index_t ii = 0; ii < 2; ++ii)
                {
                    xt::masked_view(keep(level + 1, 2 * i + ii, 2 * j + jj),
                                    mask) = static_cast<int>(CellFlag::coarsen);
                }
            }
        }
    };

    template<class... CT>
    inline auto to_coarsen(CT &&... e)
    {
        return make_field_operator_function<to_coarsen_op>(
            std::forward<CT>(e)...);
    }

    /*************************
     * refine_ghost operator *
     *************************/

    template<class TInterval>
    class refine_ghost_op : public field_operator_base<TInterval> {
      public:
        INIT_OPERATOR(refine_ghost_op)

        template<class T>
        inline void operator()(Dim<1>, T &flag) const
        {
            auto mask = flag(level + 1, i) & static_cast<int>(CellFlag::keep);
            xt::masked_view(flag(level, i / 2), mask) =
                static_cast<int>(CellFlag::refine);
        }

        template<class T>
        inline void operator()(Dim<2>, T &flag) const
        {
            auto mask =
                flag(level + 1, i, j) & static_cast<int>(CellFlag::keep);
            xt::masked_view(flag(level, i / 2, j / 2), mask) =
                static_cast<int>(CellFlag::refine);
        }

        template<class T>
        inline void operator()(Dim<3>, T &flag) const
        {
            auto mask =
                flag(level + 1, i, j, k) & static_cast<int>(CellFlag::keep);
            xt::masked_view(flag(level, i / 2, j / 2, k / 2), mask) =
                static_cast<int>(CellFlag::refine);
        }
    };

    template<class... CT>
    inline auto refine_ghost(CT &&... e)
    {
        return make_field_operator_function<refine_ghost_op>(
            std::forward<CT>(e)...);
    }

    /********************
     * enlarge operator *
     ********************/

    template<class TInterval>
    class enlarge_op : public field_operator_base<TInterval> {
      public:
        INIT_OPERATOR(enlarge_op)

        template<class T>
        inline void operator()(Dim<1>, T &cell_flag) const
        {
            auto keep_mask = cell_flag(level, i) & static_cast<int>(CellFlag::keep);

            for (int ii = -1; ii < 2; ++ii)
            {
                xt::masked_view(cell_flag(level, i + ii), keep_mask) |= static_cast<int>(CellFlag::enlarge);
            }
        }

        template<class T>
        inline void operator()(Dim<2>, T &cell_flag) const
        {
            auto keep_mask = cell_flag(level, i, j) & static_cast<int>(CellFlag::keep);

            for (int jj = -1; jj < 2; ++jj)
            {
                for (int ii = -1; ii < 2; ++ii)
                {
                    xt::masked_view(cell_flag(level, i + ii, j + jj), keep_mask) |= static_cast<int>(CellFlag::enlarge);
                }
            }
        }
    };

    template<class... CT>
    inline auto enlarge(CT &&... e)
    {
        return make_field_operator_function<enlarge_op>(std::forward<CT>(e)...);
    }

    /*******************************
     * keep_around_refine operator *
     *******************************/

    template<class TInterval>
    class keep_around_refine_op : public field_operator_base<TInterval> {
      public:
        INIT_OPERATOR(keep_around_refine_op)

        template<class T>
        inline void operator()(Dim<1>, T &cell_flag) const
        {
            auto refine_mask = cell_flag(level, i) & static_cast<int>(CellFlag::refine);

            for (int ii = -1; ii < 2; ++ii)
            {
                xt::masked_view(cell_flag(level, i + ii), refine_mask) |= static_cast<int>(CellFlag::keep);
            }
        }

        template<class T>
        inline void operator()(Dim<2>, T &cell_flag) const
        {
            auto refine_mask = cell_flag(level, i, j) & static_cast<int>(CellFlag::refine);

            for (int jj = -1; jj < 2; ++jj)
            {
                for (int ii = -1; ii < 2; ++ii)
                {
                    xt::masked_view(cell_flag(level, i + ii, j + jj), refine_mask) |= static_cast<int>(CellFlag::keep);
                }
            }
        }
    };

    template<class... CT>
    inline auto keep_around_refine(CT &&... e)
    {
        return make_field_operator_function<keep_around_refine_op>(std::forward<CT>(e)...);
    }

    /***********************
     * to_refine operator *
     ***********************/

    template<class TInterval>
    class to_refine_op : public field_operator_base<TInterval> {
      public:
        INIT_OPERATOR(to_refine_op)

        template<class T, class U, class V>
        inline void operator()(Dim<1>, T &refine, const U &detail, const V &max_detail,
                        std::size_t max_level, double eps) const
        {
            if (level < max_level)
            {
                auto mask = xt::abs(detail(level, i)) > eps;
                xt::masked_view(refine(level, i), mask) =
                    static_cast<int>(CellFlag::refine);
            }
        }

        template<class T, class U, class V>
        inline void operator()(Dim<2>, T &refine, const U &detail, const V &max_detail,
                        std::size_t max_level, double eps) const
        {
            if (level < max_level)
            {
                auto mask = xt::abs(detail(level, i, j)) > eps;
                xt::masked_view(refine(level, i, j), mask) =
                    static_cast<int>(CellFlag::refine);
            }
        }
    };

    template<class... CT>
    inline auto to_refine(CT &&... e)
    {
        return make_field_operator_function<to_refine_op>(
            std::forward<CT>(e)...);
    }

    /***********************
     * tag_to_keep operator *
     ***********************/

    template<class TInterval>
    class tag_to_keep_op : public field_operator_base<TInterval> {
      public:
        INIT_OPERATOR(tag_to_keep_op)

        template<class T>
        inline void operator()(Dim<1>, T &cell_flag) const
        {
            auto mask =
                cell_flag(level, i) & static_cast<int>(CellFlag::enlarge);
            xt::masked_view(cell_flag(level, i), mask) |=
                static_cast<int>(CellFlag::keep);
        }

        template<class T>
        inline void operator()(Dim<2>, T &cell_flag) const
        {
            auto mask =
                cell_flag(level, i, j) & static_cast<int>(CellFlag::enlarge);
            xt::masked_view(cell_flag(level, i, j), mask) |=
                static_cast<int>(CellFlag::keep);
        }
    };

    template<class... CT>
    inline auto tag_to_keep(CT &&... e)
    {
        return make_field_operator_function<tag_to_keep_op>(
            std::forward<CT>(e)...);
    }

    /***********************
     * apply_expr operator *
     ***********************/

    template<class TInterval>
    class apply_expr_op : public field_operator_base<TInterval> {
      public:
        INIT_OPERATOR(apply_expr_op)

        template<class T, class E>
        inline void operator()(Dim<1>, T &field, const field_expression<E> &e) const
        {
            field(level, i) = e.derived_cast()(level, i);
        }

        template<class T, class E>
        inline void operator()(Dim<2>, T &field, const field_expression<E> &e) const
        {
            field(level, i, j) = e.derived_cast()(level, i, j);
        }

        template<class T, class E>
        inline void operator()(Dim<3>, T &field, const field_expression<E> &e) const
        {
            field(level, i, j, k) = e.derived_cast()(level, i, j, k);
        }
    };

    template<class... CT>
    inline auto apply_expr(CT &&... e)
    {
        return make_field_operator_function<apply_expr_op>(
            std::forward<CT>(e)...);
    }

    /*******************
     * extend operator *
     *******************/
    template<class TInterval>
    class extend_op : public field_operator_base<TInterval> {
        public:
        INIT_OPERATOR(extend_op)

        template<class T>
        inline void operator()(Dim<1>, T &tag) const
        {
            auto refine_mask =
                tag(level, i) & static_cast<int>(samurai::CellFlag::refine);


            int added_cells = 1; // 1 by default

            for (int ii = -added_cells; ii < added_cells + 1; ++ii)
            {
                xt::masked_view(tag(level, i + ii), refine_mask) |= static_cast<int>(samurai::CellFlag::keep);
            }
        }

        template<class T>
        inline void operator()(Dim<2>, T &tag) const
        {
            auto refine_mask = tag(level, i, j) & static_cast<int>(samurai::CellFlag::refine);

            for (int jj = -1; jj < 2; ++jj)
            {
                for (int ii = -1; ii < 2; ++ii)
                {
                    xt::masked_view(tag(level, i + ii, j + jj), refine_mask) |= static_cast<int>(samurai::CellFlag::keep);
                }
            }
        }
    };

    template<class... CT>
    inline auto extend(CT &&... e)
    {
        return make_field_operator_function<extend_op>(
            std::forward<CT>(e)...);
    }
    /****************************
     * make_graduation operator *
     ****************************/

    template<class TInterval>
    class make_graduation_op : public field_operator_base<TInterval> {
        public:
        INIT_OPERATOR(make_graduation_op)

        template<class T>
        inline void operator()(Dim<1>, T &tag) const
        {
            auto i_even = i.even_elements();
            if (i_even.is_valid())
            {
                auto mask = tag(level, i_even) & static_cast<int>(CellFlag::keep);
                xt::masked_view(tag(level-1, i_even>>1), mask) |= static_cast<int>(CellFlag::refine);
            }

            auto i_odd = i.odd_elements();
            if (i_odd.is_valid())
            {
                auto mask = tag(level, i_odd) & static_cast<int>(CellFlag::keep);
                xt::masked_view(tag(level-1, i_odd>>1), mask) |= static_cast<int>(CellFlag::refine);
            }
        }

        template<class T>
        inline void operator()(Dim<2>, T &tag) const
        {
            auto i_even = i.even_elements();
            if (i_even.is_valid())
            {
                auto mask = tag(level, i_even, j) & static_cast<int>(CellFlag::keep);
                xt::masked_view(tag(level-1, i_even>>1, j>>1), mask) |= static_cast<int>(CellFlag::refine);
            }

            auto i_odd = i.odd_elements();
            if (i_odd.is_valid())
            {
                auto mask = tag(level, i_odd, j) & static_cast<int>(CellFlag::keep);
                xt::masked_view(tag(level-1, i_odd>>1, j>>1), mask) |= static_cast<int>(CellFlag::refine);
            }
        }
    };

    template<class... CT>
    inline auto make_graduation(CT &&... e)
    {
        return make_field_operator_function<make_graduation_op>(
            std::forward<CT>(e)...);
    }
} // namespace samurai