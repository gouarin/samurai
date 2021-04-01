#include <array>
#include <vector>
#include <experimental/random>
#include <benchmark/benchmark.h>

#include <xtensor/xfixed.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xnoalias.hpp>

#include <samurai/algorithm.hpp>
#include <samurai/cell_array.hpp>
#include <samurai/field.hpp>

static void BM_std_vector_lap_1D(benchmark::State& state)
{
    std::size_t size = state.range(0);
    std::vector<double> u1(size), u2(size);

    for (auto _ : state)
    {
        for(std::size_t i=1; i<size-1; ++i)
        {
            u2[i] = u1[i-1] - 2*u1[i] + u1[i+1];
        }
    }
    state.SetComplexityN(state.range(0));
}

static void BM_std_vector_lap_2D(benchmark::State& state)
{
    std::size_t size = state.range(0);
    std::vector<double> u1(size*size), u2(size*size);

    for (auto _ : state)
    {
        for(std::size_t j=1; j<size-1; ++j)
        {
            for(std::size_t i=1; i<size-1; ++i)
            {
                u2[i + j*size] = (                      u1[i + (j-1)*size]
                                 + u1[i-1 + j*size] - 2*u1[i + j*size] + u1[i+1 + j*size]
                                                      + u1[i + (j+1)*size]
                                 );
            }
        }
    }
    state.SetComplexityN(state.range(0));
}

static void BM_std_vector_lap_3D(benchmark::State& state)
{
    std::size_t size = state.range(0);
    std::size_t size2 = size*size;
    std::vector<double> u1(size*size*size), u2(size*size*size);

    for (auto _ : state)
    {
        for(std::size_t k=1; k<size-1; ++k)
        {
            for(std::size_t j=1; j<size-1; ++j)
            {
                for(std::size_t i=1; i<size-1; ++i)
                {
                    u2[i + j*size + k*size2] = (    u1[i + j*size + (k-1)*size2] + u1[i + (j-1)*size + k*size2]
                                    + u1[i-1 + j*size + k*size2] - 2*u1[i + j*size + k*size2] + u1[i+1 + j*size + k*size2]
                                                        + u1[i + (j+1)*size + k*size2] + u1[i + j*size + (k+1)*size2]
                                    );
                }
            }
        }
    }
    state.SetComplexityN(state.range(0));
}

static void BM_cellarray_lap_1D(benchmark::State& state)
{
    int size = state.range(0);
    samurai::Box<int, 1> box({0}, {size});
    samurai::CellArray<1> ca;
    std::size_t level = 0;

    ca[level] = {level, box};

    auto u1 = samurai::make_field<double, 1>("u1", ca);
    auto u2 = samurai::make_field<double, 1>("u2", ca);

    samurai::Interval<int> i = {1, size-1};
    for (auto _ : state)
    {
        xt::noalias(u2(level, i)) = u1(level, i-1) - 2*u1(level, i) + u1(level, i+1);
    }
    state.SetComplexityN(state.range(0));
}

static void BM_cellarray_lap_2D(benchmark::State& state)
{
    int size = state.range(0);
    samurai::Box<int, 2> box({0, 0}, {size, size});
    samurai::CellArray<2> ca;
    std::size_t level = 0;

    ca[level] = {level, box};

    auto u1 = samurai::make_field<double, 1>("u1", ca);
    auto u2 = samurai::make_field<double, 1>("u2", ca);

    samurai::Interval<int> i = {1, size-1};
    for (auto _ : state)
    {
        for(int j=1; j<size-1; ++j)
        {
            xt::noalias(u2(level, i, j)) =                       u1(level, i, j-1)
                            + u1(level, i-1, j) - 2*u1(level, i, j) + u1(level, i+1, j)
                                                  + u1(level, i, j+1);
        }
    }
    state.SetComplexityN(state.range(0));
}


static void BM_cellarray_lap_3D(benchmark::State& state)
{
    int size = state.range(0);
    samurai::Box<int, 3> box({0, 0, 0}, {size, size, size});
    samurai::CellArray<3> ca;
    std::size_t level = 0;

    ca[level] = {level, box};

    auto u1 = samurai::make_field<double, 1>("u1", ca);
    auto u2 = samurai::make_field<double, 1>("u2", ca);

    samurai::Interval<int> i = {1, size-1};
    for (auto _ : state)
    {
        for(int k=1; k<size-1; ++k)
        {
            for(int j=1; j<size-1; ++j)
             {
                xt::noalias(u2(level, i, j, k)) =  u1(level, i, j, k-1) + u1(level, i, j-1, k)
                            + u1(level, i-1, j, k) - 2*u1(level, i, j, k) + u1(level, i+1, j, k)
                                                  + u1(level, i, j+1, k) + + u1(level, i, j, k+1);
             }
        }
    }
    state.SetComplexityN(state.range(0));
}

static void BM_xtensor_with_step_lap_1D(benchmark::State& state)
{
    std::size_t size = state.range(0);
    xt::xtensor<double, 1> u1 = xt::zeros<double>({size});
    xt::xtensor<double, 1> u2 = xt::zeros<double>({size});

    auto r0 = xt::range(1, size-1, 1);
    auto rm1 = xt::range(0, size-2, 1);
    auto rp1 = xt::range(2, size, 1);

    for (auto _ : state)
    {
        xt::noalias(xt::view(u2, r0)) = xt::view(u1, rm1)
                                           - 2*xt::view(u1, r0)
                                           + xt::view(u1, rp1);
    }
    state.SetComplexityN(state.range(0));
}

static void BM_xtensor_without_step_lap_1D(benchmark::State& state)
{
    std::size_t size = state.range(0);
    xt::xtensor<double, 1> u1 = xt::zeros<double>({size});
    xt::xtensor<double, 1> u2 = xt::zeros<double>({size});

    auto r0 = xt::range(1, size-1);
    auto rm1 = xt::range(0, size-2);
    auto rp1 = xt::range(2, size);

    for (auto _ : state)
    {
        xt::noalias(xt::view(u2, r0)) = xt::view(u1, rm1)
                                           - 2*xt::view(u1, r0)
                                           + xt::view(u1, rp1);
    }
    state.SetComplexityN(state.range(0));
}

static void BM_xtensor_with_step_lap_2D(benchmark::State& state)
{
    std::size_t size = state.range(0);
    xt::xtensor<double, 2> u1 = xt::zeros<double>({size, size});
    xt::xtensor<double, 2> u2 = xt::zeros<double>({size, size});

    auto r0 = xt::range(1, size-1, 1);
    auto rm1 = xt::range(0, size-2, 1);
    auto rp1 = xt::range(2, size, 1);

    for (auto _ : state)
    {
        xt::noalias(xt::view(u2, r0, r0)) =
           + xt::view(u1, rm1, r0)
           + xt::view(u1, r0, rm1)
           - 2*xt::view(u1, r0, r0)
           + xt::view(u1, rp1, r0)
           + xt::view(u1, r0, rp1);
    }
    state.SetComplexityN(state.range(0));
}

static void BM_xtensor_without_step_lap_2D(benchmark::State& state)
{
    std::size_t size = state.range(0);
    xt::xtensor<double, 2> u1 = xt::zeros<double>({size, size});
    xt::xtensor<double, 2> u2 = xt::zeros<double>({size, size});

    auto r0 = xt::range(1, size-1);
    auto rm1 = xt::range(0, size-2);
    auto rp1 = xt::range(2, size);

    for (auto _ : state)
    {
        xt::noalias(xt::view(u2, r0, r0)) =
           + xt::view(u1, rm1, r0)
           + xt::view(u1, r0, rm1)
           - 2*xt::view(u1, r0, r0)
           + xt::view(u1, rp1, r0)
           + xt::view(u1, r0, rp1);
    }
    state.SetComplexityN(state.range(0));
}

static void BM_xtensor_with_step_lap_3D(benchmark::State& state)
{
    std::size_t size = state.range(0);
    xt::xtensor<double, 3> u1 = xt::zeros<double>({size, size, size});
    xt::xtensor<double, 3> u2 = xt::zeros<double>({size, size, size});

    auto r0 = xt::range(1, size-1, 1);
    auto rm1 = xt::range(0, size-2, 1);
    auto rp1 = xt::range(2, size, 1);

    for (auto _ : state)
    {
        xt::noalias(xt::view(u2, r0, r0, r0)) =
           + xt::view(u1, rm1, r0, r0)
           + xt::view(u1, r0, rm1, r0)
           + xt::view(u1, r0, r0, rm1)
           - 2*xt::view(u1, r0, r0, r0)
           + xt::view(u1, rp1, r0, r0)
           + xt::view(u1, r0, rp1, r0)
           + xt::view(u1, r0, r0, rp1);
    }
    state.SetComplexityN(state.range(0));
}

static void BM_xtensor_without_step_lap_3D(benchmark::State& state)
{
    std::size_t size = state.range(0);
    xt::xtensor<double, 3> u1 = xt::zeros<double>({size, size, size});
    xt::xtensor<double, 3> u2 = xt::zeros<double>({size, size, size});

    auto r0 = xt::range(1, size-1);
    auto rm1 = xt::range(0, size-2);
    auto rp1 = xt::range(2, size);

    for (auto _ : state)
    {
        xt::noalias(xt::view(u2, r0, r0, r0)) =
           + xt::view(u1, rm1, r0, r0)
           + xt::view(u1, r0, rm1, r0)
           + xt::view(u1, r0, r0, rm1)
           - 2*xt::view(u1, r0, r0, r0)
           + xt::view(u1, rp1, r0, r0)
           + xt::view(u1, r0, rp1, r0)
           + xt::view(u1, r0, r0, rp1);
    }
    state.SetComplexityN(state.range(0));
}

BENCHMARK(BM_std_vector_lap_1D)->RangeMultiplier(2)->Ranges({{1<<14, 1<<20}})->Complexity(benchmark::oN);
BENCHMARK(BM_cellarray_lap_1D)->RangeMultiplier(2)->Ranges({{1<<14, 1<<20}})->Complexity(benchmark::oN);
BENCHMARK(BM_xtensor_with_step_lap_1D)->RangeMultiplier(2)->Ranges({{1<<14, 1<<20}})->Complexity(benchmark::oN);
BENCHMARK(BM_xtensor_without_step_lap_1D)->RangeMultiplier(2)->Ranges({{1<<14, 1<<20}})->Complexity(benchmark::oN);
BENCHMARK(BM_std_vector_lap_2D)->RangeMultiplier(2)->Ranges({{1<<7, 1<<10}})->Complexity(benchmark::oNSquared);
BENCHMARK(BM_cellarray_lap_2D)->RangeMultiplier(2)->Ranges({{1<<7, 1<<10}})->Complexity(benchmark::oNSquared);
BENCHMARK(BM_xtensor_with_step_lap_2D)->RangeMultiplier(2)->Ranges({{1<<7, 1<<10}})->Complexity(benchmark::oNSquared);
BENCHMARK(BM_xtensor_without_step_lap_2D)->RangeMultiplier(2)->Ranges({{1<<7, 1<<10}})->Complexity(benchmark::oNSquared);
BENCHMARK(BM_std_vector_lap_3D)->RangeMultiplier(2)->Ranges({{1<<5, 1<<9}})->Complexity(benchmark::oNCubed);
BENCHMARK(BM_cellarray_lap_3D)->RangeMultiplier(2)->Ranges({{1<<5, 1<<9}})->Complexity(benchmark::oNCubed);
BENCHMARK(BM_xtensor_with_step_lap_3D)->RangeMultiplier(2)->Ranges({{1<<5, 1<<9}})->Complexity(benchmark::oNCubed);
BENCHMARK(BM_xtensor_without_step_lap_3D)->RangeMultiplier(2)->Ranges({{1<<5, 1<<9}})->Complexity(benchmark::oNCubed);
