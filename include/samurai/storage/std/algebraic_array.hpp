#pragma once
#include "../collapsable.hpp"
#include <array>
#include <cassert>
#include <cmath>
#include <ostream>
#include <type_traits>
#include <xtensor/core/xexpression.hpp>
#if defined(SAMURAI_FIELD_CONTAINER_EIGEN3)
#include "../eigen/eigen.hpp"
#endif

namespace samurai
{
    /**
     * Simple array with algebraic operations.
     * All the operations are eager and return by value.
     */
    template <class T, std::size_t N>
    struct StdArrayWrapper
    {
        using value_type     = T;
        using size_type      = std::size_t;
        using iterator       = typename std::array<T, N>::iterator;
        using const_iterator = typename std::array<T, N>::const_iterator;

      private:

        std::array<T, N> _a;

        // Copies an xtensor expression holding N values, whatever its shape.
        template <class E>
        void copy_from(const xt::xexpression<E>& e)
        {
            const auto& d = e.derived_cast();
            if (d.dimension() == 1)
            {
                for (std::size_t i = 0; i < N; ++i)
                {
                    _a[i] = d(i);
                }
            }
            else
            {
                auto it = d.begin();
                for (std::size_t i = 0; i < N; ++i, ++it)
                {
                    _a[i] = *it;
                }
            }
        }

      public:

        StdArrayWrapper()
        {
        }

        explicit StdArrayWrapper(const std::array<T, N>& a)
            : _a(a)
        {
        }

        explicit StdArrayWrapper(std::array<T, N>&& a)
            : _a(std::move(a))
        {
        }

        template <class T2>
            requires std::is_convertible_v<T2, T>
        StdArrayWrapper(std::initializer_list<T2> list)
        {
            assert(list.size() == N);
            auto it = list.begin();
            for (std::size_t i = 0; i < N; ++i)
            {
                _a[i] = static_cast<T>(*it);
                it++;
            }
        }

        // Any xtensor expression: xfunction, xview, xtensor_fixed...
        template <class E>
        StdArrayWrapper(const xt::xexpression<E>& e) // cppcheck-suppress noExplicitConstructor
        {
            copy_from(e);
        }

#if defined(SAMURAI_FIELD_CONTAINER_EIGEN3)
        template <class T2>
            requires(!std::is_arithmetic_v<T2> && !xt::is_xexpression<T2>::value)
        StdArrayWrapper(const T2& eigen) // cppcheck-suppress noExplicitConstructor
        {
            for (std::size_t i = 0; i < N; ++i)
            {
                this->_a[i] = eigen(static_cast<Eigen::Index>(i));
            }
        }
#endif

        StdArrayWrapper(T value) // cppcheck-suppress noExplicitConstructor
        {
            _a.fill(value);
        }

        auto& array()
        {
            return _a;
        }

        const auto& array() const
        {
            return _a;
        }

        // Number of components, as in xtensor and Eigen containers.
        constexpr std::size_t size() const
        {
            return N;
        }

        auto begin()
        {
            return _a.begin();
        }

        auto begin() const
        {
            return _a.begin();
        }

        auto end()
        {
            return _a.end();
        }

        auto end() const
        {
            return _a.end();
        }

        T* data()
        {
            return _a.data();
        }

        const T* data() const
        {
            return _a.data();
        }

        T& operator()(std::size_t i)
        {
            return _a[i];
        }

        const T& operator()(std::size_t i) const
        {
            return _a[i];
        }

        T& operator[](std::size_t i)
        {
            return _a[i];
        }

        const T& operator[](std::size_t i) const
        {
            return _a[i];
        }

        template <class E>
        StdArrayWrapper& operator=(const xt::xexpression<E>& e)
        {
            copy_from(e);
            return *this;
        }

        auto& operator+=(const StdArrayWrapper<T, N>& other)
        {
            for (std::size_t i = 0; i < N; ++i)
            {
                _a[i] += other._a[i];
            }
            return *this;
        }

        auto& operator*=(const StdArrayWrapper<T, N>& other)
        {
            for (std::size_t i = 0; i < N; ++i)
            {
                _a[i] *= other._a[i];
            }
            return *this;
        }

        auto& operator-=(const StdArrayWrapper<T, N>& other)
        {
            for (std::size_t i = 0; i < N; ++i)
            {
                _a[i] -= other._a[i];
            }
            return *this;
        }

        auto& operator/=(const StdArrayWrapper<T, N>& other)
        {
            for (std::size_t i = 0; i < N; ++i)
            {
                _a[i] /= other._a[i];
            }
            return *this;
        }

        template <class E>
        auto& operator+=(const xt::xexpression<E>& e)
        {
            return *this += StdArrayWrapper(e);
        }

        template <class E>
        auto& operator-=(const xt::xexpression<E>& e)
        {
            return *this -= StdArrayWrapper(e);
        }

        template <class E>
        auto& operator*=(const xt::xexpression<E>& e)
        {
            return *this *= StdArrayWrapper(e);
        }

        template <class E>
        auto& operator/=(const xt::xexpression<E>& e)
        {
            return *this /= StdArrayWrapper(e);
        }

        template <class NumberType, typename std::enable_if_t<std::is_arithmetic_v<NumberType>, bool> = true>
        auto& operator+=(NumberType scalar)
        {
            for (std::size_t i = 0; i < N; ++i)
            {
                _a[i] += scalar;
            }
            return *this;
        }

        template <class NumberType, typename std::enable_if_t<std::is_arithmetic_v<NumberType>, bool> = true>
        auto& operator*=(NumberType scalar)
        {
            for (std::size_t i = 0; i < N; ++i)
            {
                _a[i] *= scalar;
            }
            return *this;
        }

        template <class NumberType, typename std::enable_if_t<std::is_arithmetic_v<NumberType>, bool> = true>
        auto& operator-=(NumberType scalar)
        {
            for (std::size_t i = 0; i < N; ++i)
            {
                _a[i] -= scalar;
            }
            return *this;
        }

        template <class NumberType, typename std::enable_if_t<std::is_arithmetic_v<NumberType>, bool> = true>
        auto& operator/=(NumberType scalar)
        {
            for (std::size_t i = 0; i < N; ++i)
            {
                _a[i] /= scalar;
            }
            return *this;
        }

        void fill(T value)
        {
            _a.fill(value);
        }
    };

    template <class>
    struct is_std_array_wrapper : std::false_type
    {
    };

    template <class T, std::size_t N>
    struct is_std_array_wrapper<StdArrayWrapper<T, N>> : std::true_type
    {
    };

    template <class T>
    constexpr bool is_std_array_wrapper_v = is_std_array_wrapper<std::decay_t<T>>::value;

    // What stands for a scalar on the other side of a binary operation.
    template <class S>
    concept array_scalar = std::is_arithmetic_v<S>;

    //------------------//
    // array (op) array //
    //------------------//

    template <class T, std::size_t size>
    auto operator+(const StdArrayWrapper<T, size>& a, const StdArrayWrapper<T, size>& b)
    {
        StdArrayWrapper<T, size> c(a);
        c += b;
        return c;
    }

    template <class T, std::size_t size>
    auto operator-(const StdArrayWrapper<T, size>& a, const StdArrayWrapper<T, size>& b)
    {
        StdArrayWrapper<T, size> c(a);
        c -= b;
        return c;
    }

    template <class T, std::size_t size>
    auto operator*(const StdArrayWrapper<T, size>& a, const StdArrayWrapper<T, size>& b)
    {
        StdArrayWrapper<T, size> c(a);
        c *= b;
        return c;
    }

    template <class T, std::size_t size>
    auto operator/(const StdArrayWrapper<T, size>& a, const StdArrayWrapper<T, size>& b)
    {
        StdArrayWrapper<T, size> c(a);
        c /= b;
        return c;
    }

    template <class T, std::size_t size>
    auto operator-(const StdArrayWrapper<T, size>& a)
    {
        StdArrayWrapper<T, size> b;
        for (std::size_t i = 0; i < size; ++i)
        {
            b[i] = -a[i];
        }
        return b;
    }

    //-------------------//
    // array (op) scalar //
    //-------------------//

    template <array_scalar NumberType, class T, std::size_t size>
    auto operator*(NumberType scalar, const StdArrayWrapper<T, size>& a)
    {
        StdArrayWrapper<T, size> b(a);
        b *= scalar;
        return b;
    }

    template <class T, std::size_t size, array_scalar NumberType>
    auto operator*(const StdArrayWrapper<T, size>& a, NumberType scalar)
    {
        return scalar * a;
    }

    template <class T, std::size_t size, array_scalar NumberType>
    auto operator/(const StdArrayWrapper<T, size>& a, NumberType scalar)
    {
        StdArrayWrapper<T, size> b(a);
        b /= scalar;
        return b;
    }

    template <class T, std::size_t size, array_scalar NumberType>
    auto operator/(NumberType scalar, const StdArrayWrapper<T, size>& a)
    {
        StdArrayWrapper<T, size> b;
        for (std::size_t i = 0; i < size; ++i)
        {
            b[i] = scalar / a[i];
        }
        return b;
    }

    template <array_scalar NumberType, class T, std::size_t size>
    auto operator+(NumberType scalar, const StdArrayWrapper<T, size>& a)
    {
        StdArrayWrapper<T, size> b(a);
        b += scalar;
        return b;
    }

    template <class T, std::size_t size, array_scalar NumberType>
    auto operator+(const StdArrayWrapper<T, size>& a, NumberType scalar)
    {
        return scalar + a;
    }

    template <class T, std::size_t size, array_scalar NumberType>
    auto operator-(const StdArrayWrapper<T, size>& a, NumberType scalar)
    {
        StdArrayWrapper<T, size> b(a);
        b -= scalar;
        return b;
    }

    template <array_scalar NumberType, class T, std::size_t size>
    auto operator-(NumberType scalar, const StdArrayWrapper<T, size>& a)
    {
        StdArrayWrapper<T, size> b;
        for (std::size_t i = 0; i < size; ++i)
        {
            b[i] = scalar - a[i];
        }
        return b;
    }

    //-------------------------------//
    // array (op) xtensor expression //
    //-------------------------------//
    // The expression is evaluated: the result is an array, not a lazy expression.

// Forwarding references and a constraint: with the same signature as the lazy operators
// of xtensor, the constraint is what makes overload resolution pick these ones.
#define SAMURAI_ARRAY_XT_OP(OP)                                                          \
    template <class A, class E>                                                          \
        requires is_std_array_wrapper_v<A> && xt::is_xexpression<std::decay_t<E>>::value \
    auto operator OP(A&& a, E&& e)                                                       \
    {                                                                                    \
        return a OP std::decay_t<A>(e);                                                  \
    }                                                                                    \
    template <class E, class A>                                                          \
        requires xt::is_xexpression<std::decay_t<E>>::value && is_std_array_wrapper_v<A> \
    auto operator OP(E&& e, A&& a)                                                       \
    {                                                                                    \
        return std::decay_t<A>(e) OP a;                                                  \
    }

    SAMURAI_ARRAY_XT_OP(+)
    SAMURAI_ARRAY_XT_OP(-)
    SAMURAI_ARRAY_XT_OP(*)
    SAMURAI_ARRAY_XT_OP(/)
#undef SAMURAI_ARRAY_XT_OP

    //-----------//
    // Functions //
    //-----------//

    template <class T, std::size_t size, class NumberType>
    auto pow(const StdArrayWrapper<T, size>& a, NumberType exponent)
    {
        StdArrayWrapper<T, size> b;
        for (std::size_t i = 0; i < size; ++i)
        {
            b[i] = std::pow(a[i], exponent);
        }
        return b;
    }

    template <class NumberType1,
              class NumberType2,
              typename std::enable_if_t<std::is_arithmetic_v<NumberType1> && std::is_arithmetic_v<NumberType2>, bool> = true>
    SAMURAI_INLINE auto pow(NumberType1 base, NumberType2 exponent)
    {
        return std::pow(base, exponent);
    }

    template <class T, std::size_t size>
    void fill(StdArrayWrapper<T, size>& array, T value)
    {
        array.fill(value);
    }

    namespace math
    {
        template <class T, std::size_t size>
        T sum(const StdArrayWrapper<T, size>& a)
        {
            T s = 0;
            for (std::size_t i = 0; i < size; ++i)
            {
                s += a[i];
            }
            return s;
        }
    }

    template <class T, std::size_t size>
    bool operator==(const StdArrayWrapper<T, size>& a, const StdArrayWrapper<T, size>& b)
    {
        return a.array() == b.array();
    }

    // Already evaluated: nothing to do.
    template <class T, std::size_t size>
    auto eval(const StdArrayWrapper<T, size>& a)
    {
        return a;
    }

    template <class T, std::size_t size>
    bool compare(const StdArrayWrapper<T, size>& a, const StdArrayWrapper<T, size>& b)
    {
        return a == b;
    }

    template <class T, std::size_t size, class E>
    bool compare(const StdArrayWrapper<T, size>& a, const xt::xexpression<E>& e)
    {
        return e.derived_cast().size() == size && a == StdArrayWrapper<T, size>(e);
    }

    template <class E, class T, std::size_t size>
    bool compare(const xt::xexpression<E>& e, const StdArrayWrapper<T, size>& a)
    {
        return compare(a, e);
    }

    template <class T, std::size_t size>
    std::ostream& operator<<(std::ostream& out, const StdArrayWrapper<T, size>& a)
    {
        out << "{";
        for (std::size_t i = 0; i < size; ++i)
        {
            out << (i == 0 ? "" : ", ") << a[i];
        }
        return out << "}";
    }

    template <class value_type, std::size_t size, bool can_collapse>
    using collapsable_algebraic_std_array = CollapsableArray<StdArrayWrapper<value_type, size>, value_type, size, can_collapse>;

} // end namespace samurai
