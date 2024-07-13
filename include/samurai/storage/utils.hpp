// Copyright 2018-2024 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

namespace samurai
{
    template <class T>
    struct strided_range_t
    {
        T start;
        T end;
        T step = 1;
    };

    template <class T>
    struct contiguous_range_t
    {
        T start;
        T end;
    };

    template <class T, class T1>
    strided_range_t<T> sam_range(T start, T end, T1 step)
    {
        return {start, end, step};
    }

    template <class T>
    contiguous_range_t<T> sam_range(T start, T end)
    {
        return {start, end};
    }
}
