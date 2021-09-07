#include <vector>
#include <chrono>
#include <benchmark/benchmark.h>

#include <xtensor/xnoalias.hpp>
#include <xtensor/xfixed.hpp>
#include <xtensor/xrandom.hpp>

#include <samurai/box.hpp>
#include <samurai/amr/mesh.hpp>

#include <samurai/field.hpp>
#include <samurai/uniform_mesh.hpp>

#include <samurai/algorithm.hpp>
#include <samurai/cell_list.hpp>
#include <samurai/cell_array.hpp>
#include <samurai/static_algorithm.hpp>
#include <samurai/hdf5.hpp>

static void BM_std_vector_neigh_2D(benchmark::State& state)
{
    std::size_t size = state.range(0);
    std::size_t size2 = (size+2)*(size+2);
    std::vector<double> u1(size2), u2(size2);

    for (auto _ : state)
    {
        for(std::size_t j=1; j<size-1; ++j)
        {
            for(std::size_t i=1; i<size-1; ++i)
            {
                u2[i + j*size] = u1[  i +     j*size]
                               + u1[i-1 +     j*size]
                               + u1[i+1 +     j*size]
                               + u1[  i + (j-1)*size]
                               + u1[  i + (j+1)*size];
            }
        }
    }
    state.counters["nb cells"] = size*size;
    state.SetComplexityN(state.range(0));
}

static void BM_samurai_uniform_neigh_2D(benchmark::State& state)
{
    std::size_t size = state.range(0);
    constexpr std::size_t dim = 2;
    using Config = samurai::UniformConfig<dim, 2>;
    using mesh_id_t = typename samurai::UniformMesh<Config>::mesh_id_t;

    samurai::Box<double, dim> b{{0, 0}, {size, size}};
    // samurai::Box<double, dim> b{{0, 0}, {size, 1}};
    samurai::UniformMesh<Config> mesh(b, 0);

    auto u1 = samurai::make_field<double, 1>("u1", mesh);
    auto u2 = samurai::make_field<double, 1>("u2", mesh);

    for (auto _ : state)
    {
        samurai::for_each_interval(mesh[mesh_id_t::cells], [&](std::size_t level, const auto& i, const auto& index)
        {
            auto j = index[0];

            xt::noalias(u2(level, i, j)) = u1(level, i, j)
                                         + u1(level, i+1, j)
                                         + u1(level, i-1, j)
                                         + u1(level, i, j+1)
                                         + u1(level, i, j-1);
        });
    }
    state.SetComplexityN(state.range(0));
    state.counters["nb cells"] = mesh.nb_cells(mesh_id_t::cells);

}

template <std::size_t dim>
auto generate_mesh(std::size_t block_size, std::size_t start_level, std::size_t max_level)
{
    using namespace xt::placeholders;
    using Config = samurai::amr::Config<dim>;
    using mesh_t = samurai::amr::Mesh<Config>;
    using mesh_id_t = typename mesh_t::mesh_id_t;
    using cl_type = typename mesh_t::cl_type;

    samurai::Box<double, 2> b{{0, 0}, {1, 1}};
    mesh_t mesh(b, start_level, start_level, max_level);

    cl_type cl;

    std::size_t start = 0;
    std::size_t end = block_size;
    for (std::size_t level = start_level, dl=0; level <= max_level; ++level, ++dl)
    {
        for(int j=0; j<(1<<dl)*block_size; ++j)
        // for(int j=0; j<1; ++j)
        {
            cl[level][{j}].add_interval({start, end});
        }
        start = 2*end;
        end = start + block_size;
    }

    start = -block_size;
    end = 0;
    for (std::size_t level = start_level + 1, dl=1; level <= max_level; ++level, ++dl)
    {
        for(int j=0; j<(1<<dl)*block_size; ++j)
        // for(int j=0; j<1; ++j)
        {
            cl[level][{j}].add_interval({start, end});
        }
        end = 2*start;
        start = end - block_size;
    }

    mesh = {cl, mesh.min_level(), mesh.max_level()};
    return mesh;
}

static void BM_samurai_adapted_neigh_2D(benchmark::State& state)
{
    std::size_t block_size = state.range(0);
    constexpr std::size_t dim = 2;

    using Config = samurai::amr::Config<dim>;
    using mesh_t = samurai::amr::Mesh<Config>;
    using mesh_id_t = typename mesh_t::mesh_id_t;

    static constexpr std::size_t min_level = 1;
    static constexpr std::size_t max_level = 8;

    auto mesh = generate_mesh<dim>(block_size, min_level, max_level);

    auto u1 = samurai::make_field<double, 1>("u1", mesh);
    auto u2 = samurai::make_field<double, 1>("u2", mesh);

    for (auto _ : state)
    {
        samurai::for_each_interval(mesh[mesh_id_t::cells], [&](std::size_t level, const auto& i, const auto& index)
        {
            auto j = index[0];

            xt::noalias(u2(level, i, j)) = u1(level, i, j)
                                         + u1(level, i+1, j)
                                         + u1(level, i-1, j)
                                         + u1(level, i, j+1)
                                         + u1(level, i, j-1);
        });
    }
    // state.SetComplexityN(state.range(0));
    state.counters["nb cells"] = mesh.nb_cells(mesh_id_t::cells);

}

template <std::size_t dim_,  std::size_t block_size>
class MyFixture : public ::benchmark::Fixture
{
  public:
    static constexpr std::size_t dim = dim_;
    using Config = samurai::amr::Config<dim>;
    using mesh_t = samurai::amr::Mesh<Config>;
    using mesh_id_t = typename mesh_t::mesh_id_t;

    static constexpr std::size_t min_level = 4;
    static constexpr std::size_t max_level = 8;

    MyFixture()
    {
      mesh = generate_mesh<dim_>(block_size, min_level, max_level);
    //   samurai::save("mesh", mesh);
    //   mesh.clean();
    }

    void bench(benchmark::State& state)
    {
        auto u1 = samurai::make_field<double, 1>("u1", mesh);
        auto u2 = samurai::make_field<double, 1>("u2", mesh);

        for (auto _ : state)
        {
            samurai::for_each_interval(mesh[mesh_id_t::cells], [&](std::size_t level, const auto& i, const auto& index)
            {
                auto j = index[0];

                xt::noalias(u2(level, i, j)) = u1(level, i, j)
                                             + u1(level, i+1, j)
                                             + u1(level, i-1, j)
                                             + u1(level, i, j+1)
                                             + u1(level, i, j-1);
            });
        }
        state.counters["nb cells"] = mesh.nb_cells(mesh_id_t::cells);
        state.counters["nb all cells"] = mesh.nb_cells();
        state.counters["nb cells min level"] = mesh[mesh_id_t::cells][min_level].nb_cells();
        state.counters["nb cells max level"] = mesh[mesh_id_t::cells][max_level].nb_cells();
    }

    mesh_t mesh;
};

BENCHMARK(BM_std_vector_neigh_2D)->RangeMultiplier(2)->Ranges({{1<<5, 1<<9}})->Complexity(benchmark::oNSquared);
BENCHMARK(BM_samurai_uniform_neigh_2D)->RangeMultiplier(2)->Ranges({{1<<5, 1<<9}})->Complexity(benchmark::oNSquared);
BENCHMARK(BM_samurai_adapted_neigh_2D)->RangeMultiplier(2)->Ranges({{1<<2, 1<<5}});


// BENCHMARK(BM_std_vector_neigh_2D)->RangeMultiplier(2)->Ranges({{1<<5, 1<<9}})->Complexity(benchmark::oNSquared);
// BENCHMARK(BM_samurai_uniform_neigh_2D)->RangeMultiplier(2)->Ranges({{1<<10, 1<<17}})->Complexity(benchmark::oNSquared);
// BENCHMARK(BM_samurai_adapted_neigh_2D)->RangeMultiplier(2)->Ranges({{1<<2, 1<<12}});

// BENCHMARK_TEMPLATE_DEFINE_F(MyFixture, adapted_neigh_2D, 2, 1024)(benchmark::State& state){bench(state);}
// // BENCHMARK_REGISTER_F(MyFixture, adapted_neigh_2D)->DenseRange(1, 10, 1);
// BENCHMARK_REGISTER_F(MyFixture, adapted_neigh_2D);