#pragma once


#include <mure/operators_base.hpp>

namespace mure
{

    template<class TInterval>
    class to_coarsen_mr_op : public field_operator_base<TInterval> {
        public:
        INIT_OPERATOR(to_coarsen_mr_op)

        template<class T1, class T2, class T3>
        inline void operator()(Dim<1>, const T1& detail, const T3&  max_detail, T2 &tag, double eps, std::size_t min_lev) const
        {
            constexpr auto size = T1::size;

            if (level > min_lev)
            {
                auto maxd = xt::view(max_detail, level);

                if (size == 1)
                {
                    auto mask = xt::abs(detail(level, 2*i))/maxd < eps;

                    xt::masked_view(tag(level, 2*i), mask) = static_cast<int>(mure::CellFlag::coarsen);
                    xt::masked_view(tag(level, 2*i + 1), mask) = static_cast<int>(mure::CellFlag::coarsen);
                }
                else
                {
                    auto mask = xt::sum((xt::abs(detail(level, 2*i))/maxd < eps), {1}) > (size-1);

                    xt::masked_view(tag(level, 2*i), mask) = static_cast<int>(mure::CellFlag::coarsen);
                    xt::masked_view(tag(level, 2*i+1), mask) = static_cast<int>(mure::CellFlag::coarsen);
                }
            }            
        }

        template<class T1, class T2, class T3>
        inline void operator()(Dim<2>, const T1& detail, const T3&  max_detail, T2 &tag, double eps, std::size_t min_lev) const
        {
            constexpr auto size = T1::size;

            if (level > min_lev)
            {
                auto maxd = xt::view(max_detail, level);

                if (size == 1)
                {
                    auto mask = (xt::abs(detail(level, 2*i  ,   2*j))/maxd < eps) and
                                (xt::abs(detail(level, 2*i+1,   2*j))/maxd < eps) and
                                (xt::abs(detail(level, 2*i  , 2*j+1))/maxd < eps) and
                                (xt::abs(detail(level, 2*i+1, 2*j+1))/maxd < eps);

                    xt::masked_view(tag(level, 2*i  ,   2*j), mask) = static_cast<int>(mure::CellFlag::coarsen);
                    xt::masked_view(tag(level, 2*i+1,   2*j), mask) = static_cast<int>(mure::CellFlag::coarsen);
                    xt::masked_view(tag(level, 2*i  , 2*j+1), mask) = static_cast<int>(mure::CellFlag::coarsen);
                    xt::masked_view(tag(level, 2*i+1, 2*j+1), mask) = static_cast<int>(mure::CellFlag::coarsen);
                }
                else
                {
                    auto mask = xt::sum((xt::abs(detail(level, 2*i  ,   2*j))/maxd < eps) and
                                        (xt::abs(detail(level, 2*i+1,   2*j))/maxd < eps) and
                                        (xt::abs(detail(level, 2*i  , 2*j+1))/maxd < eps) and
                                        (xt::abs(detail(level, 2*i+1, 2*j+1))/maxd < eps), {1}) > (size-1);

                    xt::masked_view(tag(level, 2*i  ,   2*j), mask) = static_cast<int>(mure::CellFlag::coarsen);
                    xt::masked_view(tag(level, 2*i+1,   2*j), mask) = static_cast<int>(mure::CellFlag::coarsen);
                    xt::masked_view(tag(level, 2*i  , 2*j+1), mask) = static_cast<int>(mure::CellFlag::coarsen);
                    xt::masked_view(tag(level, 2*i+1, 2*j+1), mask) = static_cast<int>(mure::CellFlag::coarsen);
                }
            }            
            /*
            auto mask = 0.25 * (xt::abs(detail(level, 2*i  ,   2*j)) +
                        xt::abs(detail(level, 2*i+1,   2*j)) +
                        xt::abs(detail(level, 2*i  , 2*j+1)) +
                        xt::abs(detail(level, 2*i+1, 2*j+1))) < eps;
            */          

            // CAVEAT : THIS CONTROL IS NECESSARY...MANY PROBLEMS WITHOUT
        //     if (level > min_lev) {
        //         xt::masked_view(tag(level, 2*i  ,   2*j), mask) = static_cast<int>(mure::CellFlag::coarsen);
        //         xt::masked_view(tag(level, 2*i+1,   2*j), mask) = static_cast<int>(mure::CellFlag::coarsen);
        //         xt::masked_view(tag(level, 2*i  , 2*j+1), mask) = static_cast<int>(mure::CellFlag::coarsen);
        //         xt::masked_view(tag(level, 2*i+1, 2*j+1), mask) = static_cast<int>(mure::CellFlag::coarsen);
        //    }
        }
    };



    template<class... CT>
    inline auto to_coarsen_mr(CT &&... e)
    {
        return make_field_operator_function<to_coarsen_mr_op>(
            std::forward<CT>(e)...);
    }


    // Uses the details in the way suggested in
    // the paper by Bihari and Harten [1997]
    template<class TInterval>
    class to_coarsen_mr_BH_op : public field_operator_base<TInterval> {
        public:
        INIT_OPERATOR(to_coarsen_mr_BH_op)

        template<class T1, class T2, class T3>
        inline void operator()(Dim<2>, const T1& detail, const T3&  max_detail, T2 &tag, double eps, std::size_t min_lev) const
        {
            constexpr auto size = T1::size;

            // CAVEAT : THIS CONTROL IS NECESSARY...MANY PROBLEMS WITHOUT
            if (level > min_lev)
            {
                const double C_fourth_term = 2.0;
                auto maxd = xt::view(max_detail, level);

                if (size == 1)
                {
                    auto mask = (xt::abs(.25 * ( detail(level, 2*i, 2*j)-detail(level, 2*i+1, 2*j)-detail(level, 2*i, 2*j+1)+detail(level, 2*i+1, 2*j+1)) / maxd * C_fourth_term) < eps) and
                                (xt::abs(.25 * (-detail(level, 2*i, 2*j)+detail(level, 2*i+1, 2*j)-detail(level, 2*i, 2*j+1)+detail(level, 2*i+1, 2*j+1)) / maxd))                < eps and  
                                (xt::abs(.25 * (-detail(level, 2*i, 2*j)-detail(level, 2*i+1, 2*j)+detail(level, 2*i, 2*j+1)+detail(level, 2*i+1, 2*j+1)) / maxd))                < eps;

                    xt::masked_view(tag(level, 2*i  ,   2*j), mask) = static_cast<int>(mure::CellFlag::coarsen);
                    xt::masked_view(tag(level, 2*i+1,   2*j), mask) = static_cast<int>(mure::CellFlag::coarsen);
                    xt::masked_view(tag(level, 2*i  , 2*j+1), mask) = static_cast<int>(mure::CellFlag::coarsen);
                    xt::masked_view(tag(level, 2*i+1, 2*j+1), mask) = static_cast<int>(mure::CellFlag::coarsen);
                }
                else
                {
                    auto mask = xt::sum((xt::abs(.25 * ( detail(level, 2*i, 2*j)-detail(level, 2*i+1, 2*j)-detail(level, 2*i, 2*j+1)+detail(level, 2*i+1, 2*j+1)) / maxd * C_fourth_term) < eps) and
                                        (xt::abs(.25 * (-detail(level, 2*i, 2*j)+detail(level, 2*i+1, 2*j)-detail(level, 2*i, 2*j+1)+detail(level, 2*i+1, 2*j+1)) / maxd))                < eps and  
                                        (xt::abs(.25 * (-detail(level, 2*i, 2*j)-detail(level, 2*i+1, 2*j)+detail(level, 2*i, 2*j+1)+detail(level, 2*i+1, 2*j+1)) / maxd))                < eps, {1}) > (size-1);

                    xt::masked_view(tag(level, 2*i  ,   2*j), mask) = static_cast<int>(mure::CellFlag::coarsen);
                    xt::masked_view(tag(level, 2*i+1,   2*j), mask) = static_cast<int>(mure::CellFlag::coarsen);
                    xt::masked_view(tag(level, 2*i  , 2*j+1), mask) = static_cast<int>(mure::CellFlag::coarsen);
                    xt::masked_view(tag(level, 2*i+1, 2*j+1), mask) = static_cast<int>(mure::CellFlag::coarsen);
                }
           }
        }
    };

    template<class... CT>
    inline auto to_coarsen_mr_BH(CT &&... e)
    {
        return make_field_operator_function<to_coarsen_mr_BH_op>(
            std::forward<CT>(e)...);
    }

    
    template<class TInterval>
    class to_refine_mr_op : public field_operator_base<TInterval> {
        public:
        INIT_OPERATOR(to_refine_mr_op)

        template<class T1, class T2, class T3>
        inline void operator()(Dim<1>, const T1& detail, const T3& max_detail, 
                               T2 &tag, double eps, std::size_t max_level) const
        {
            constexpr auto size = T1::size;

            if (level < max_level)
            {
                auto maxd = xt::view(max_detail, level);
                if (size == 1)
                {
                    auto mask = ((xt::abs(detail(level, 2*i))/maxd) > eps) or 
                                ((xt::abs(detail(level, 2*i+1))/maxd) > eps);

                    xt::masked_view(tag(level, 2*i  ), mask) = static_cast<int>(mure::CellFlag::refine);
                    xt::masked_view(tag(level, 2*i+1), mask) = static_cast<int>(mure::CellFlag::refine);
                }
                else
                {
                    auto mask = xt::sum(((xt::abs(detail(level, 2*i  ))/maxd) > eps) or 
                                        ((xt::abs(detail(level, 2*i+1))/maxd) > eps), {1}) > 0;

                    xt::masked_view(tag(level, 2*i  ), mask) = static_cast<int>(mure::CellFlag::refine);
                    xt::masked_view(tag(level, 2*i+1), mask) = static_cast<int>(mure::CellFlag::refine);
                }
            }        
        }

        template<class T1, class T2, class T3>
        inline void operator()(Dim<2>, const T1& detail, const T3& max_detail, 
                               T2 &tag, double eps, std::size_t max_level) const
        {
            constexpr auto size = T1::size;

            if (level < max_level)
            {
                auto maxd = xt::view(max_detail, level);
                if (size == 1)
                {
                    auto mask = ((xt::abs(detail(level, 2*i  , 2*j  ))/maxd) > eps) or 
                                ((xt::abs(detail(level, 2*i+1, 2*j  ))/maxd) > eps) or
                                ((xt::abs(detail(level, 2*i  , 2*j+1))/maxd) > eps) or
                                ((xt::abs(detail(level, 2*i+1, 2*j+1))/maxd) > eps);

                    xt::masked_view(tag(level, 2*i  , 2*j  ), mask) = static_cast<int>(mure::CellFlag::refine);
                    xt::masked_view(tag(level, 2*i+1, 2*j  ), mask) = static_cast<int>(mure::CellFlag::refine);
                    xt::masked_view(tag(level, 2*i  , 2*j+1), mask) = static_cast<int>(mure::CellFlag::refine);
                    xt::masked_view(tag(level, 2*i+1, 2*j+1), mask) = static_cast<int>(mure::CellFlag::refine);
                }
                else
                {
                    auto mask = xt::sum(((xt::abs(detail(level, 2*i  , 2*j  ))/maxd) > eps) or 
                                        ((xt::abs(detail(level, 2*i+1, 2*j  ))/maxd) > eps) or
                                        ((xt::abs(detail(level, 2*i  , 2*j+1))/maxd) > eps) or
                                        ((xt::abs(detail(level, 2*i+1, 2*j+1))/maxd) > eps), {1}) > 0;

                    xt::masked_view(tag(level, 2*i  , 2*j  ), mask) = static_cast<int>(mure::CellFlag::refine);
                    xt::masked_view(tag(level, 2*i+1, 2*j  ), mask) = static_cast<int>(mure::CellFlag::refine);
                    xt::masked_view(tag(level, 2*i  , 2*j+1), mask) = static_cast<int>(mure::CellFlag::refine);
                    xt::masked_view(tag(level, 2*i+1, 2*j+1), mask) = static_cast<int>(mure::CellFlag::refine);
                }
            }        
        }
    };

    template<class... CT>
    inline auto to_refine_mr(CT &&... e)
    {
        return make_field_operator_function<to_refine_mr_op>(
            std::forward<CT>(e)...);
    }

    
    template<class TInterval>
    class to_refine_mr_BH_op : public field_operator_base<TInterval> {
        public:
        INIT_OPERATOR(to_refine_mr_BH_op)

        template<class T1, class T2, class T3>
        inline void operator()(Dim<2>, const T1& detail, const T3& max_detail, 
                               T2 &tag, double eps, std::size_t max_level) const
        {
            constexpr auto size = T1::size;

            if (level < max_level)  {

                const double C_fourth_term = 2.0;
                auto maxd = xt::view(max_detail, level);

                if (size == 1)
                {
                    auto mask = (xt::abs(.25 * ( detail(level, 2*i, 2*j)-detail(level, 2*i+1, 2*j)-detail(level, 2*i, 2*j+1)+detail(level, 2*i+1, 2*j+1)) / maxd * C_fourth_term) > eps) or
                                (xt::abs(.25 * (-detail(level, 2*i, 2*j)+detail(level, 2*i+1, 2*j)-detail(level, 2*i, 2*j+1)+detail(level, 2*i+1, 2*j+1)) / maxd))                > eps  or  
                                (xt::abs(.25 * (-detail(level, 2*i, 2*j)-detail(level, 2*i+1, 2*j)+detail(level, 2*i, 2*j+1)+detail(level, 2*i+1, 2*j+1)) / maxd))                > eps;


                    xt::masked_view(tag(level, 2*i  , 2*j  ), mask) = static_cast<int>(mure::CellFlag::refine);
                    xt::masked_view(tag(level, 2*i+1, 2*j  ), mask) = static_cast<int>(mure::CellFlag::refine);
                    xt::masked_view(tag(level, 2*i  , 2*j+1), mask) = static_cast<int>(mure::CellFlag::refine);
                    xt::masked_view(tag(level, 2*i+1, 2*j+1), mask) = static_cast<int>(mure::CellFlag::refine);
                }
                else
                {
                    auto mask = xt::sum((xt::abs(.25 * ( detail(level, 2*i, 2*j)-detail(level, 2*i+1, 2*j)-detail(level, 2*i, 2*j+1)+detail(level, 2*i+1, 2*j+1)) / maxd * C_fourth_term) > eps) or
                                        (xt::abs(.25 * (-detail(level, 2*i, 2*j)+detail(level, 2*i+1, 2*j)-detail(level, 2*i, 2*j+1)+detail(level, 2*i+1, 2*j+1)) / maxd))                > eps  or  
                                        (xt::abs(.25 * (-detail(level, 2*i, 2*j)-detail(level, 2*i+1, 2*j)+detail(level, 2*i, 2*j+1)+detail(level, 2*i+1, 2*j+1)) / maxd))                > eps, {1}) > 0;


                    xt::masked_view(tag(level, 2*i  , 2*j  ), mask) = static_cast<int>(mure::CellFlag::refine);
                    xt::masked_view(tag(level, 2*i+1, 2*j  ), mask) = static_cast<int>(mure::CellFlag::refine);
                    xt::masked_view(tag(level, 2*i  , 2*j+1), mask) = static_cast<int>(mure::CellFlag::refine);
                    xt::masked_view(tag(level, 2*i+1, 2*j+1), mask) = static_cast<int>(mure::CellFlag::refine);                    
                }
            }        
        }
    };

    template<class... CT>
    inline auto to_refine_mr_BH(CT &&... e)
    {
        return make_field_operator_function<to_refine_mr_BH_op>(
            std::forward<CT>(e)...);
    }



    template<class TInterval>
    class max_detail_mr_op : public field_operator_base<TInterval> {
        public:
        INIT_OPERATOR(max_detail_mr_op)

        template<class T1>
        inline void operator()(Dim<2>, const T1& detail, double & max_detail) const
        {
            auto ii = 2 * i;
            ii.step = 1;

            max_detail = std::max(max_detail,
                                  xt::amax(xt::maximum(xt::abs(detail(level + 1, ii, 2 * j)),
                                                       xt::abs(detail(level + 1, ii, 2 * j + 1))))[0]
                                 );
        }
    };

    template<class... CT>
    inline auto max_detail_mr(CT &&... e)
    {
        return make_field_operator_function<max_detail_mr_op>(
            std::forward<CT>(e)...);
    }


}