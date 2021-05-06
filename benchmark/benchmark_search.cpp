#include <array>
#include <experimental/random>
#include <benchmark/benchmark.h>

#include <fmt/format.h>

#include <xtensor/xfixed.hpp>
#include <xtensor/xrandom.hpp>

#include <samurai/algorithm.hpp>
#include <samurai/cell_list.hpp>
#include <samurai/cell_array.hpp>
#include <samurai/static_algorithm.hpp>

template <std::size_t dim>
auto generate_mesh(int bound, std::size_t start_level, std::size_t max_level)
{
    samurai::Box<int, dim> box({-bound<<start_level, -bound<<start_level, -bound<<start_level},
                               { bound<<start_level,  bound<<start_level,  bound<<start_level});
    samurai::CellArray<dim> ca;

    ca[start_level] = {start_level, box};

    for(std::size_t ite = 0; ite < max_level - start_level; ++ite)
    {
        samurai::CellList<dim> cl;

        samurai::for_each_interval(ca, [&](std::size_t level, const auto& interval, const auto& index)
        {
            auto choice = xt::random::choice(xt::xtensor_fixed<bool, xt::xshape<2>>{true, false}, interval.size());
            for(int i = interval.start, ic = 0; i<interval.end; ++i, ++ic)
            {
                if (choice[ic])
                {
                  samurai::static_nested_loop<dim - 1, 0, 2>([&](auto stencil)
                  {
                      auto new_index = 2 * index + stencil;
                      cl[level + 1][new_index].add_interval({2*i, 2*i+2});
                  });
                }
                else
                {
                    cl[level][index].add_point(i);
                }
            }
        });

        ca = {cl, true};
    }

    return ca;
}

template <std::size_t dim_, int bound>
class MyFixture : public ::benchmark::Fixture
{
  public:
    static constexpr std::size_t dim = dim_;
    static constexpr std::size_t min_level = 1;
    static constexpr std::size_t max_level = 10;

    MyFixture()
    {
      mesh = generate_mesh<dim_>(bound, min_level, max_level);
    }

    void bench(benchmark::State& state)
    {
        if (state.range(0) != 0)
        {
            std::size_t found = 0;
            for (auto _ : state)
            {
                for(std::size_t s = 0; s < state.range(0); ++s)
                {
                    auto level = std::experimental::randint(min_level, max_level);
                    std::array<int, dim> coord;
                    for(auto& c: coord)
                    {
                        c = std::experimental::randint(-bound<<level, (bound<<level) - 1);
                    }
                    auto out = samurai::find(mesh[level], coord);
                    if  (out != -1)
                    {
                        found++;
                    }
                }
            }

            for(std::size_t d=1; d<dim; ++d)
            {
                std::size_t min_size = std::numeric_limits<std::size_t>::max();
                std::size_t max_size = std::numeric_limits<std::size_t>::min();
                for (std::size_t level = min_level; level <= max_level; ++level)
                {
                    if (!mesh[level].empty())
                    {
                        auto offsets = mesh[level].offsets(d);
                        std::adjacent_difference(offsets.begin(), offsets.end(), offsets.begin());
                        auto minmax = std::minmax_element(offsets.cbegin()+1, offsets.cend());

                        min_size = std::min(min_size, *minmax.first);
                        max_size = std::max(max_size, *minmax.second);
                    }
                }
                state.counters[fmt::format("min[{}]", d)] = min_size;
                state.counters[fmt::format("max[{}]", d)] = max_size;
            }

            state.counters["nb cells"] = mesh.nb_cells();
            state.counters["percent found"] = static_cast<double>(found)/state.iterations()/state.range(0);
        }
    }

    samurai::CellArray<dim_> mesh;
};

BENCHMARK_TEMPLATE_DEFINE_F(MyFixture, Search_1D, 1, 1000)(benchmark::State& state){bench(state);}
BENCHMARK_REGISTER_F(MyFixture, Search_1D)->DenseRange(1, 9, 1)->DenseRange(10, 100, 10);

BENCHMARK_TEMPLATE_DEFINE_F(MyFixture, Search_2D, 2, 10)(benchmark::State& state){bench(state);}
BENCHMARK_REGISTER_F(MyFixture, Search_2D)->DenseRange(1, 9, 1)->DenseRange(10, 100, 10);

BENCHMARK_TEMPLATE_DEFINE_F(MyFixture, Search_3D, 3, 1)(benchmark::State& state){bench(state);}
BENCHMARK_REGISTER_F(MyFixture, Search_3D)->DenseRange(1, 9, 1)->DenseRange(10, 100, 10);
