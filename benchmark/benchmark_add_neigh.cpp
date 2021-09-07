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

static void BM_std_vector_neigh_3D(benchmark::State& state)
{
    std::size_t size = state.range(0);
    std::size_t size2 = size*size;
    std::size_t size3 = size*size*size;
    std::vector<double> u1(size3), u2(size3);

    for (auto _ : state)
    {
        for(std::size_t k=1; k<size-1; ++k)
        {
            for(std::size_t j=1; j<size-1; ++j)
            {
                for(std::size_t i=1; i<size-1; ++i)
                {
                    u2[i + j*size + k*size2] = u1[i + j*size + (k-1)*size2]
                                             + u1[i + (j-1)*size + k*size2]
                                             + u1[i-1 + j*size + k*size2]
                                             + u1[i + j*size + k*size2]
                                             + u1[i+1 + j*size + k*size2]
                                             + u1[i + (j+1)*size + k*size2]
                                             + u1[i + j*size + (k+1)*size2];
                }
            }
        }
    }
    state.SetComplexityN(state.range(0));
}

static void BM_samurai_uniform_neigh_3D(benchmark::State& state)
{
    std::size_t size = state.range(0);
    constexpr std::size_t dim = 3;
    using Config = samurai::UniformConfig<dim, 2>;
    using mesh_id_t = typename samurai::UniformMesh<Config>::mesh_id_t;

    samurai::Box<double, 3> b{{0, 0, 0}, {size, size, size}};
    samurai::UniformMesh<Config> mesh(b, 0);

    auto u1 = samurai::make_field<double, 1>("u1", mesh);
    auto u2 = samurai::make_field<double, 1>("u2", mesh);

    for (auto _ : state)
    {
        samurai::for_each_interval(mesh[mesh_id_t::cells], [&](std::size_t level, const auto& i, const auto& index)
        {
            auto j = index[0];
            auto k = index[1];

            xt::noalias(u2(level, i, j, k)) = u1(level, i, j, k)
                                            + u1(level, i+1, j, k)
                                            + u1(level, i-1, j, k)
                                            + u1(level, i, j+1, k)
                                            + u1(level, i, j-1, k)
                                            + u1(level, i, j, k+1)
                                            + u1(level, i, j, k-1);
        });
    }
    state.SetComplexityN(state.range(0));
}

static void BM_samurai_adapted_neigh_3D(benchmark::State& state)
{
    std::size_t size = state.range(0);
    constexpr std::size_t dim = 3;
    using Config = samurai::UniformConfig<dim, 2>;
    using mesh_id_t = typename samurai::UniformMesh<Config>::mesh_id_t;

    samurai::Box<double, 3> b{{0, 0, 0}, {size, size, size}};
    samurai::UniformMesh<Config> mesh(b, 0);

    auto u1 = samurai::make_field<double, 1>("u1", mesh);
    auto u2 = samurai::make_field<double, 1>("u2", mesh);

    for (auto _ : state)
    {
        samurai::for_each_interval(mesh[mesh_id_t::cells], [&](std::size_t level, const auto& i, const auto& index)
        {
            auto j = index[0];
            auto k = index[1];

            xt::noalias(u2(level, i, j, k)) = u1(level, i, j, k)
                                            + u1(level, i+1, j, k)
                                            + u1(level, i-1, j, k)
                                            + u1(level, i, j+1, k)
                                            + u1(level, i, j-1, k)
                                            + u1(level, i, j, k+1)
                                            + u1(level, i, j, k-1);
        });
    }
    state.SetComplexityN(state.range(0));
}

template <std::size_t dim>
auto generate_mesh(int bound, std::size_t start_level, std::size_t max_level)
{
    using namespace xt::placeholders;
    using Config = samurai::amr::Config<dim>;
    using mesh_t = samurai::amr::Mesh<Config>;
    using mesh_id_t = typename mesh_t::mesh_id_t;
    using cl_type = typename mesh_t::cl_type;

    double eps = 1e-2;
    samurai::Box<double, dim> box({-bound, -bound, -bound},
                                  { bound,  bound,  bound});
    mesh_t mesh(box, start_level, start_level, max_level);

    for(std::size_t ite = 0; ite < max_level - start_level; ++ite)
    {
        cl_type cl;

        samurai::for_each_cell(mesh[mesh_id_t::cells], [&](const auto& cell)
        {
            auto x = cell.center(0);
            auto y = cell.center(1);
            auto z = cell.center(2);
            auto level = cell.level;
            auto i = xt::view(cell.indices, 0)[0];
            auto index  = xt::view(cell.indices, xt::range(1, _));

            // if ((-0.5 <= x && x <= 0.5)
            //  && (-0.5 <= y && y <= 0.5)
            //  && (-0.5 <= z && z <= 0.5))
            // {
            //     samurai::static_nested_loop<dim - 1, 0, 2>([&](auto stencil)
            //     {
            //         auto new_index = 2 * index + stencil;
            //         cl[level + 1][new_index].add_interval({2*i, 2*i+2});
            //     });
            // }
            // else
            // {
            //     cl[level][index].add_point(i);
            // }

            if (x*x + y*y + z*z <= 1.25*(0.5*0.5)
            &&  x*x + y*y + z*z >= 0.75*(0.5*0.5))
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
        });

        mesh = {cl, mesh.min_level(), mesh.max_level()};
    }

    // for(std::size_t ite = 0; ite < max_level - start_level; ++ite)
    // {
    //     cl_type cl;

    //     samurai::for_each_interval(mesh[mesh_id_t::cells], [&](std::size_t level, const auto& interval, const auto& index)
    //     {
    //         auto choice = xt::random::choice(xt::xtensor_fixed<bool, xt::xshape<2>>{true, false}, interval.size());
    //         for(int i = interval.start, ic = 0; i<interval.end; ++i, ++ic)
    //         {
    //             if (choice[ic])
    //             {
    //               samurai::static_nested_loop<dim - 1, 0, 2>([&](auto stencil)
    //               {
    //                   auto new_index = 2 * index + stencil;
    //                   cl[level + 1][new_index].add_interval({2*i, 2*i+2});
    //               });
    //             }
    //             else
    //             {
    //                 cl[level][index].add_point(i);
    //             }
    //         }
    //     });

    //     mesh = {cl, mesh.min_level(), mesh.max_level()};
    // }

    return mesh;
}

template <std::size_t dim_, int bound>
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
      mesh = generate_mesh<dim_>(bound, min_level, max_level);
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
                auto k = index[1];

                xt::noalias(u2(level, i, j, k)) = u1(level, i, j, k)
                                                + u1(level, i+1, j, k)
                                                + u1(level, i-1, j, k)
                                                + u1(level, i, j+1, k)
                                                + u1(level, i, j-1, k)
                                                + u1(level, i, j, k+1)
                                                + u1(level, i, j, k-1);
            });

            // std::size_t level = min_level;
            // samurai::for_each_interval(mesh[mesh_id_t::cells][level], [&](std::size_t level, const auto& i, const auto& index)
            // {
            //     auto j = index[0];
            //     auto k = index[1];

            //     xt::noalias(u2(level, i, j, k)) = u1(level, i, j, k)
            //                                     + u1(level, i+1, j, k)
            //                                     + u1(level, i-1, j, k)
            //                                     + u1(level, i, j+1, k)
            //                                     + u1(level, i, j-1, k)
            //                                     + u1(level, i, j, k+1)
            //                                     + u1(level, i, j, k-1);
            // });
        }
        state.counters["nb cells"] = mesh.nb_cells(mesh_id_t::cells);
        state.counters["nb all cells"] = mesh.nb_cells();
        state.counters["nb cells min level"] = mesh[mesh_id_t::cells][min_level].nb_cells();
        state.counters["nb cells max level"] = mesh[mesh_id_t::cells][max_level].nb_cells();
    }

    mesh_t mesh;
};

BENCHMARK(BM_std_vector_neigh_3D)->RangeMultiplier(2)->Ranges({{1<<5, 1<<9}})->Complexity(benchmark::oNCubed);
BENCHMARK(BM_samurai_uniform_neigh_3D)->RangeMultiplier(2)->Ranges({{1<<5, 1<<9}})->Complexity(benchmark::oNCubed);
BENCHMARK_TEMPLATE_DEFINE_F(MyFixture, adapted_neigh_3D, 3, 1)(benchmark::State& state){bench(state);}
BENCHMARK_REGISTER_F(MyFixture, adapted_neigh_3D)->DenseRange(1, 10, 1);