// Copyright 2025 the samurai team
// SPDX-License-Identifier:  BSD-3-Clause

#include <numbers>

#include <cassert>
#include <memory>
#include <samurai/algorithm/update.hpp>
#include <samurai/field.hpp>
#include <samurai/io/hdf5.hpp>
#include <samurai/io/restart.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/samurai.hpp>
#include <samurai/timers.hpp>

#include "euler/config.hpp"
#include "euler/eos.hpp"
#include "euler/init/cases.hpp"
#include "euler/metrics.hpp"
#include "euler/prediction.hpp"
#include "euler/save.hpp"
#include "euler/reconstruction.hpp"
#include "euler/schemes.hpp"
#include "euler/time_stepping.hpp"
#include "euler/utils.hpp"
#include "euler/variables.hpp"

template <class Field>
void init_bc(Field& u, double& t, const std::string& test_case_name, auto eos)
{
    auto& registry  = test_case::TestCaseRegistry<Field>::instance();
    auto& test_case = registry.get(test_case_name);
    test_case.bc(u, t, eos);
}

template <class Field>
void init_sol(Field& u, auto& config, int jump, auto& mra_config, const std::string& test_case_name, auto eos)
{
    samurai::ScopedTimer timer("initialization");
    static constexpr std::size_t dim = Field::dim;
    using mesh_t                     = typename Field::mesh_t;
    using cl_type                    = typename mesh_t::cl_type;

    auto& registry  = test_case::TestCaseRegistry<Field>::instance();
    auto& test_case = registry.get(test_case_name);

    auto& mesh = u.mesh();
    u.resize();
    samurai::for_each_cell(mesh,
                           [&](auto& cell)
                           {
                               test_case.init(u, cell, eos);
                           });

    std::cout << "Refining to level " << mesh.max_level() << std::endl;
    // NOTE this deliberately uses the DEFAULT prediction, not the positivity
    // preserving one used in the time loop. main_3d.cpp does the opposite and
    // says so explicitly. Left as is so that this commit changes no result;
    // to be reconciled in lot 1.
    auto MRadaptation = samurai::make_MRAdapt(u);
    MRadaptation(mra_config);

    while (jump > 0)
    {
        cl_type cl;
        for_each_interval(mesh,
                          [&](std::size_t level, const auto& i, const auto& index)
                          {
                              samurai::static_nested_loop<dim - 1, 0, 2>(
                                  [&](const auto& stencil)
                                  {
                                      auto new_index = 2 * index + stencil;
                                      for (auto ii = i.start; ii < i.end; ++ii)
                                      {
                                          cl[level + 1][new_index].add_interval(i << 1);
                                      }
                                  });
                          });
        config.max_level()++;
        mesh = {cl, config};

        std::cout << "Refining to level " << mesh.max_level() << std::endl;
        u.resize();
        samurai::for_each_cell(mesh,
                               [&](auto& cell)
                               {
                                   test_case.init(u, cell, eos);
                               });
        MRadaptation(mra_config);
        jump--;
    }
}

int main(int argc, char* argv[])
{
    constexpr std::size_t dim = 2;
    std::size_t default_level = 10;

    using field_t = config<dim>::field_t;

    auto& app = samurai::initialize("Euler equations solver (2D)", argc, argv);

    double Tf  = .25;
    double cfl = 0.4;
    double t   = 0.;
    std::string restart_file;
    std::size_t order = 1;
    std::string slope_limiter   = "moncen";
    std::string time_integrator = "auto";
    std::string scheme    = "hllc";
    std::string test_case = "double_mach_reflection";
    double gamma          = 0.; // only used when --gamma is given

    bool check_positivity = false;

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
    app.add_flag("--check-positivity", check_positivity, "Check positivity of density and pressure at each iteration")
        ->group("Simulation parameters");
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
    config.disable_args_parse();

    auto mesh = samurai::mra::make_empty_mesh(config);
    auto u    = samurai::make_vector_field<double, 2 + dim>("euler", mesh);

    // The operator function stores what it is handed; pass a prvalue so that it
    // owns its own copy of the equation of state rather than a reference to this
    // closure's member.
    auto prediction_fn = [eos](auto& new_field, const auto& old_field)
    {
        return make_field_operator_function<Euler_prediction_op>(new_field, old_field, EOS::IdealGas{eos});
    };

    auto MRadaptation = samurai::make_MRAdapt(prediction_fn, u);
    auto mra_config   = samurai::mra_config().relative_detail(true);

    if (restart_file.empty())
    {
        int jump      = 0;
        default_level = std::max(config.min_level(), default_level);
        if (config.min_level() != config.max_level())
        {
            jump = static_cast<int>(config.max_level() - default_level);
            if (jump > 0)
            {
                config.max_level() = default_level;
            }
        }

        std::cout << "jump = " << jump << " min-level = " << config.min_level() << " max-level = " << config.max_level() << std::endl;
        mesh = samurai::mra::make_mesh(box, config);
        init_sol(u, config, jump, mra_config, test_case, eos);
        std::cout << "Mesh initialized with " << mesh.nb_cells() << " cells." << std::endl;
    }
    else
    {
        samurai::load(restart_file, mesh, u);
    }
    // A MUSCL reconstruction reads two layers of ghost cells where the
    // first-order flux reads one, so the boundary conditions are built to the
    // width the scheme asks for.
    bc::ghost_layers() = order;
    init_bc(u, t, test_case, eos);

    auto unp1 = samurai::make_vector_field<double, 2 + dim>("euler", mesh);
    auto unp2 = samurai::make_vector_field<double, 2 + dim>("euler", mesh);

    // SSP-RK2 evaluates the scheme on an intermediate state, and evaluating a
    // scheme fills the ghost cells: the scratch fields need the same boundary
    // conditions as the solution itself.
    unp1.copy_bc_from(u);
    unp2.copy_bc_from(u);

    double dx            = mesh.cell_length(config.max_level());
    const double dt_save = Tf / static_cast<double>(nfiles);
    std::size_t nsave    = 0;
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

    // The three numbers the article reports per run, measured here rather than
    // reconstructed afterwards from a log: see euler/metrics.hpp.
    Metrics metrics(mesh);

    samurai::times::timers.start("TimeLoop");
    metrics.start(mesh);
    bool done = false;
    while (!done)
    {
        double dt = cfl * dx / get_max_lambda(u, eos);

        metrics.adapt(
            [&]
            {
                MRadaptation(mra_config);
            });

        if (check_positivity)
        {
            check(u, eos);
        }

        if (std::isnan(t))
        {
            std::cerr << "Error: Time became NaN, stopping simulation" << std::endl;
            break;
        }

        if (t + dt > Tf)
        {
            dt   = Tf - t;
            done = true;
        }
        std::cout << fmt::format("iteration {}: t = {}, dt = {}", nt++, t, dt) << "\r";

        metrics.step(mesh);

        if (order == 1)
        {
            advance(u, unp1, unp2, first_order, first_order_sweeps, dt_for_flux, dt, integrator);
        }
        else
        {
            advance(u, unp1, unp2, second_order, second_order_sweeps, dt_for_flux, dt, integrator);
        }

        t += dt;

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
    samurai::times::timers.stop("TimeLoop");

    metrics.report(metrics_file);

    samurai::finalize();
    return 0;
}
