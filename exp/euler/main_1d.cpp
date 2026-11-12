// Copyright 2025 the samurai team
// SPDX-License-Identifier:  BSD-3-Clause

#include <cassert>
#include <memory>
#include <samurai/algorithm/update.hpp>
#include <samurai/field.hpp>
#include <samurai/io/hdf5.hpp>
#include <samurai/io/restart.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/samurai.hpp>

#include "euler/config.hpp"
#include "euler/eos.hpp"
#include "euler/init/cases.hpp"
#include "euler/metrics.hpp"
#include "euler/save.hpp"
#include "euler/reconstruction.hpp"
#include "euler/schemes.hpp"
#include "euler/time_stepping.hpp"
#include "euler/utils.hpp"
#include "euler/variables.hpp"

int main(int argc, char* argv[])
{
    constexpr std::size_t dim = 1;

    using field_t = config<dim>::field_t;

    auto& app = samurai::initialize("Euler equations solver (1D)", argc, argv);

    double Tf  = .15;
    double cfl = 0.4;
    double t   = 0.;
    std::string restart_file;
    std::size_t order = 1;
    std::string slope_limiter   = "moncen";
    std::string time_integrator = "auto";
    std::string scheme    = "hll";
    std::string test_case = "double_rarefaction";
    double gamma          = 0.; // only used when --gamma is given

    // Output parameters
    fs::path path = "results";
    std::string filename;
    std::size_t nfiles = 1;
    std::string metrics_file;

    auto available = test_case::TestCaseRegistry<field_t>::instance().available_test_cases();

    app.add_option("--cfl", cfl, "The CFL")->capture_default_str()->group("Simulation parameters");
    app.add_option("--Ti", t, "Initial time")->capture_default_str()->group("Simulation parameters");
    app.add_option("--Tf", Tf, "Final time")->capture_default_str()->group("Simulation parameters");
    app.add_option("--scheme", scheme, "Finite volume scheme")
        ->capture_default_str()
        ->check(CLI::IsMember({"rusanov", "hll", "hllc"}))
        ->group("Simulation parameters");
    app.add_option("--order", order, "Order of the scheme in space: 1 for cell averages, 2 for a MUSCL reconstruction")
        ->capture_default_str()
        ->check(CLI::IsMember({1, 2}))
        ->group("Simulation parameters");
    app.add_option("--slope-limiter", slope_limiter, "Slope limiter of the MUSCL reconstruction")
        ->capture_default_str()
        ->check(CLI::IsMember({"none", "minmod", "vanleer", "moncen"}))
        ->group("Simulation parameters");
    app.add_option("--time-integrator", time_integrator, "Time integration")
        ->capture_default_str()
        ->check(CLI::IsMember({"auto", "euler", "ssprk2", "strang"}))
        ->group("Simulation parameters");
    app.add_option("--test-case", test_case, "Test case")->capture_default_str()->check(CLI::IsMember(available))->group("Simulation parameters");
    auto* gamma_opt = app.add_option("--gamma", gamma, "Ratio of specific heats (defaults to the value of the test case)")
                          ->group("Simulation parameters");
    app.add_option("--restart-file", restart_file, "Restart file")->capture_default_str()->group("Simulation parameters");
    app.add_option("--path", path, "Output path")->capture_default_str()->group("Output");
    app.add_option("--filename", filename, "File name prefix (defaults to <test-case>_<scheme>)")->group("Output");
    app.add_option("--nfiles", nfiles, "Number of output files")->capture_default_str()->group("Output");
    app.add_option("--metrics-file", metrics_file, "Write the performance metrics of the run, as JSON, to this file")->group("Output");

    // The cases that take a parameter of their own declare it here, before the
    // parse. All of them do, not only the selected one: which case runs is
    // itself decided by the parse.
    test_case::TestCaseRegistry<field_t>::instance().add_options(app);

    SAMURAI_PARSE(argc, argv);

    std::cout << "Samurai version: " << SAMURAI_VERSION << std::endl; // Print Samurai version info

    const auto& selected = test_case::TestCaseRegistry<field_t>::instance().get(test_case);

    // The test case carries the gas it was designed for; --gamma overrides it.
    EOS::IdealGas eos = selected.eos;
    if (gamma_opt->count() > 0)
    {
        eos.gamma = gamma;
    }
    std::cout << "Using gamma = " << eos.gamma << std::endl;

    // The default integrator follows the order: explicit Euler is all a
    // first-order flux can use, and SSP-RK2 is the one that reaches second
    // order in every dimension. Naming one explicitly always wins.
    if (time_integrator == "auto")
    {
        time_integrator = default_time_integrator(order);
    }
    const auto integrator = time_integrator_from_name(time_integrator);

    if (filename.empty())
    {
        filename = fmt::format("{}_{}", test_case, scheme);
    }

    // Initialize the mesh
    auto box = selected.box();

    auto config = samurai::mesh_config<dim>().min_level(8).max_level(8).max_stencil_size(4).disable_minimal_ghost_width();
    config.periodic(selected.periodic);
    config.parse_args();

    auto mesh = samurai::mra::make_empty_mesh(config);
    auto u    = samurai::make_vector_field<double, 2 + dim>("euler", mesh);

    if (restart_file.empty())
    {
        mesh = samurai::mra::make_mesh(box, config);
        u.resize();
        samurai::for_each_cell(mesh,
                               [&](auto& cell)
                               {
                                   selected.init(u, cell, eos);
                               });
    }
    else
    {
        samurai::load(restart_file, mesh, u);
    }

    std::cout << config.min_level() << " " << config.max_level() << std::endl;

    // A MUSCL reconstruction reads two layers of ghost cells where the
    // first-order flux reads one, so the boundary conditions are built to the
    // width the scheme asks for.
    bc::ghost_layers() = order;
    selected.bc(u, t, eos);

    auto unp1 = samurai::make_vector_field<double, 2 + dim>("euler", mesh);
    auto unp2 = samurai::make_vector_field<double, 2 + dim>("euler", mesh);

    // SSP-RK2 evaluates the scheme on an intermediate state, and evaluating a
    // scheme fills the ghost cells: the scratch fields need the same boundary
    // conditions as the solution itself.
    unp1.copy_bc_from(u);
    unp2.copy_bc_from(u);

    double dx            = mesh.cell_length(config.max_level());
    const double dt_save = Tf / static_cast<double>(nfiles);
    std::size_t nsave    = 1;
    std::size_t nt       = 0;

    save(path.string(), fmt::format("{}_init", filename), u, eos);
    // The conservative state, which is what --restart-file reloads. save()
    // writes primitives for post-processing and cannot be read back.
    samurai::dump(path, fmt::format("{}_restart_init", filename), mesh, u);

    std::cout << fmt::format("Using scheme: {}, order {}, {} in time", scheme, order, time_integrator) << std::endl;

    // Built once, called with a different time step at every iteration: the
    // Hancock predictor reads the current one through this.
    auto dt_for_flux = std::make_shared<double>(0.);

    // The Hancock predictor belongs to the single-step integrators: with SSP-RK2
    // the second order comes from the stages instead, and tracing as well would
    // count it twice.
    MusclOptions muscl_options{.limiter = slope_limiter_from_name(slope_limiter),
                               .hancock = integrator != TimeIntegrator::ssprk2,
                               .dt      = dt_for_flux};

    // Both orders are built, and the one the time loop uses is chosen per step.
    // They are different types, a wider stencil being a different scheme.
    auto first_order  = make_first_order_scheme<decltype(u)>(scheme, eos);
    auto second_order = make_second_order_scheme<decltype(u)>(scheme, eos, muscl_options);

    // The same two, restricted to one direction each, for Strang.
    auto directional = [&](auto&& make_one)
    {
        return [&]<std::size_t... D>(std::index_sequence<D...>)
        {
            return std::array{make_one(static_cast<int>(D))...};
        }(std::make_index_sequence<dim>{});
    };

    auto first_order_sweeps = directional(
        [&](int d)
        {
            return make_first_order_scheme<decltype(u)>(scheme, eos, d);
        });
    auto second_order_sweeps = directional(
        [&](int d)
        {
            auto options      = muscl_options;
            options.direction = d;
            return make_second_order_scheme<decltype(u)>(scheme, eos, options);
        });

    auto MRadaptation = samurai::make_MRAdapt(u);
    auto mra_config   = samurai::mra_config().relative_detail(true);

    // The three numbers the article reports per run, measured here rather than
    // reconstructed afterwards from a log: see euler/metrics.hpp.
    Metrics metrics(mesh);
    metrics.start(mesh);

    while (t != Tf)
    {
        metrics.adapt(
            [&]
            {
                MRadaptation(mra_config);
            });

        double dt = cfl * dx / get_max_lambda(u, eos);
        t += dt;

        if (std::isnan(t))
        {
            std::cerr << "Error: Time became NaN, stopping simulation" << std::endl;
            break;
        }

        if (t > Tf)
        {
            dt += Tf - t;
            t = Tf;
        }

        std::cout << fmt::format("iteration {}: t = {}, dt = {}", nt++, t, dt) << std::endl;

        metrics.step(mesh);

        if (order == 1)
        {
            advance(u, unp1, unp2, first_order, first_order_sweeps, dt_for_flux, dt, integrator);
        }
        else
        {
            advance(u, unp1, unp2, second_order, second_order_sweeps, dt_for_flux, dt, integrator);
        }

        if (t >= static_cast<double>(nsave + 1) * dt_save || t == Tf)
        {
            const std::string suffix = (nfiles != 1) ? fmt::format("_ite_{}", nsave++) : "";
            // Writing is not part of the time to solution: how many files a run
            // produces is a choice of the person running it.
            metrics.output(
                [&]
                {
                    save(path.string(), fmt::format("{}{}", filename, suffix), u, eos);
                    samurai::dump(path, fmt::format("{}_restart{}", filename, suffix), mesh, u);
                });
        }
    }

    metrics.stop(mesh);
    metrics.report(metrics_file);

    samurai::finalize();
    return 0;
}
