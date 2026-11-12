// Copyright 2025 the samurai team
// SPDX-License-Identifier:  BSD-3-Clause

// Times the MUSCL + HLLC flux function of make_muscl_scheme alone, outside any mesh traversal.
// The function is called through its std::function, as samurai calls it, on a fixed array of
// random stencils. Prints the time per flux in ns (minimum over the repetitions) and a checksum.
//
// usage: flux_bench [n_stencils] [passes] [repetitions]

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <random>
#include <vector>

#include <samurai/field.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/schemes/fv.hpp>

#include "euler/config.hpp"
#include "euler/schemes/muscl.hpp"

static constexpr std::size_t dim = 2;
using field_t                    = config<dim>::field_t;
using cfg                        = samurai::FluxConfig<samurai::SchemeType::NonLinear, 4, field_t, field_t>;
using stencil_t                  = samurai::StencilValues<cfg>;
using layout                     = EulerLayout<dim>;

// Which branch of hllc the two middle cells select, from their averages (the reconstruction can move a few).
template <std::size_t d>
int hllc_branch(const PrimState<dim>& l, const PrimState<dim>& r, EOS::IdealGas eos)
{
    const double cl = eos.c(l.rho, l.p);
    const double cr = eos.c(r.rho, r.p);
    const double sl = std::min(l.v[d] - cl, r.v[d] - cr);
    const double sr = std::max(l.v[d] + cl, r.v[d] + cr);
    const double sm = (l.rho * l.v[d] * (sl - l.v[d]) - l.p - r.rho * r.v[d] * (sr - r.v[d]) + r.p)
                    / (l.rho * (sl - l.v[d]) - r.rho * (sr - r.v[d]));
    return sl >= 0 ? 0 : sm >= 0 ? 1 : sr >= 0 ? 2 : 3;
}

struct Stencil
{
    stencil_t values;
    int branch;
};

// A smooth profile over the four cells, with a jump between the two middle ones one time out of five.
// The normal Mach number is uniform in [-2.5, 2.5], so the four branches of hllc are all taken.
template <std::size_t d>
std::vector<Stencil> make_stencils(std::size_t n, EOS::IdealGas eos, unsigned seed)
{
    std::mt19937_64 gen(seed);
    std::uniform_real_distribution<double> u01(0., 1.);
    auto in = [&](double a, double b)
    {
        return a + (b - a) * u01(gen);
    };

    std::vector<Stencil> out(n);
    for (auto& s : out)
    {
        PrimState<dim> base;
        base.rho         = in(0.5, 8.);
        base.p           = in(0.5, 100.);
        const double c   = eos.c(base.rho, base.p);
        base.v[d]        = in(-2.5, 2.5) * c;
        base.v[1 - d]    = in(-1., 1.) * c;
        const double eps = in(-0.05, 0.05);
        const bool jump  = u01(gen) < 0.2;
        const double jr  = in(0.5, 2.);
        const double jp  = in(0.5, 2.);

        std::array<PrimState<dim>, 4> prim;
        for (std::size_t k = 0; k < 4; ++k)
        {
            const double f = 1. + eps * (static_cast<double>(k) - 1.5);
            prim[k]        = base;
            prim[k].rho *= f;
            prim[k].p *= f;
            prim[k].v[d] *= f;
            if (jump && k >= 2)
            {
                prim[k].rho *= jr;
                prim[k].p *= jp;
            }
            const auto q = prim2cons<dim>(prim[k], eos);
            for (std::size_t i = 0; i < layout::size; ++i)
            {
                s.values[k][i] = q[i];
            }
        }
        s.branch = hllc_branch<d>(prim[1], prim[2], eos);
    }
    return out;
}

template <class Flux>
__attribute__((noinline)) double
run(const Flux& flux_function, const samurai::StencilData<cfg>& data, const std::vector<Stencil>& stencils, std::size_t passes, double& checksum)
{
    samurai::FluxValue<cfg> flux;
    double sum    = 0.;
    const auto t0 = std::chrono::steady_clock::now();
    for (std::size_t p = 0; p < passes; ++p)
    {
        for (const auto& s : stencils)
        {
            flux_function(flux, data, s.values);
            sum += flux[0] + flux[1] + flux[2] + flux[3];
        }
    }
    const std::chrono::duration<double, std::nano> dt = std::chrono::steady_clock::now() - t0;
    checksum                                          = sum;
    return dt.count() / static_cast<double>(passes * stencils.size());
}

template <std::size_t d, class Scheme>
void bench(Scheme& scheme, EOS::IdealGas eos, std::size_t n, std::size_t passes, std::size_t repetitions)
{
    samurai::StencilCells<cfg> cells;
    samurai::StencilData<cfg> data(cells);
    data.cell_length = 1. / 128;

    auto stencils = make_stencils<d>(n, eos, 1234 + static_cast<unsigned>(d));
    std::array<std::size_t, 4> count{};
    for (const auto& s : stencils)
    {
        ++count[static_cast<std::size_t>(s.branch)];
    }

    const auto& flux_function = scheme.flux_definition()[d].cons_flux_function;
    for (int sorted = 0; sorted < 2; ++sorted)
    {
        if (sorted)
        {
            std::stable_sort(stencils.begin(),
                             stencils.end(),
                             [](const auto& a, const auto& b)
                             {
                                 return a.branch < b.branch;
                             });
        }
        double best = 1e300, checksum = 0.;
        for (std::size_t r = 0; r < repetitions; ++r)
        {
            best = std::min(best, run(flux_function, data, stencils, passes, checksum));
        }
        std::printf("d=%zu %-8s ns/flux=%7.2f  checksum=%.17g  branches=%zu/%zu/%zu/%zu\n",
                    d,
                    sorted ? "sorted" : "shuffled",
                    best,
                    checksum,
                    count[0],
                    count[1],
                    count[2],
                    count[3]);
    }
}

int main(int argc, char* argv[])
{
    const std::size_t n           = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 4096;
    const std::size_t passes      = argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 400;
    const std::size_t repetitions = argc > 3 ? std::strtoull(argv[3], nullptr, 10) : 7;

    const EOS::IdealGas eos = EOS::air;
    // Same options as the DMR runs: moncen limiter, Hancock predictor.
    MusclOptions options{.limiter = SlopeLimiter::moncen, .hancock = true, .dt = std::make_shared<const double>(0.4 / 128 / 60.), .direction = -1};
    auto scheme = make_muscl_scheme<riemann::Solver::hllc, field_t>(eos, options);

    bench<0>(scheme, eos, n, passes, repetitions);
    bench<1>(scheme, eos, n, passes, repetitions);
    return 0;
}
