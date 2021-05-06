#include <chrono>
#include <samurai/cell_array.hpp>
#include <samurai/box.hpp>
#include <samurai/field.hpp>

#include <xtensor/xnoalias.hpp>

/// Timer used in tic & toc
auto tic_timer = std::chrono::high_resolution_clock::now();

/// Launching the timer
void tic()
{
    tic_timer = std::chrono::high_resolution_clock::now();
}

/// Stopping the timer and returning the duration in seconds
double toc()
{
    const auto toc_timer = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> time_span = toc_timer - tic_timer;
    return time_span.count();
}

int main()
{
    constexpr std::size_t dim = 2;
    constexpr std::size_t start_level = 11;
    samurai::Box<double, dim> box = {{0, 0}, {1, 1}};

    samurai::CellArray<dim> ca;

    ca[start_level] = {start_level, box};

    auto u1 = samurai::make_field<double, 1>("u1", ca);
    auto u2 = samurai::make_field<double, 1>("u2", ca);

    std::size_t nrep = 1000;
    xt::xtensor<double, 1> times = xt::zeros<double>({nrep});
    for(std::size_t irep=0; irep<nrep; ++irep)
    {
        tic();
        samurai::for_each_interval(ca, [&](std::size_t level, auto& i, auto index)
        {
            auto j = index[0];
            xt::noalias(u2(level, i, j)) = 3*u1(level, i, j) + 2;
        });
        times[irep] = toc();
    }

    std::cout << xt::mean(times) << std::endl;
    return 0;
}