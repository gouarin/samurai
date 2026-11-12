// Copyright 2025 the samurai team
// SPDX-License-Identifier:  BSD-3-Clause

#include <cassert>
#include <samurai/algorithm/update.hpp>
#include <samurai/field.hpp>
#include <samurai/io/hdf5.hpp>
#include <samurai/io/restart.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/samurai.hpp>

#include "euler/eos.hpp"
#include "euler/prediction.hpp"
#include "euler/schemes.hpp"
#include "euler/utils.hpp"
#include "euler/variables.hpp"

double rhoL = 1.;
double pL   = 0.4;
double vL   = -2.;

double rhoR = 1.;
double pR   = 0.4;
double vR   = 2.;

void init(auto& u, auto eos)
{
    static constexpr std::size_t dim = std::decay_t<decltype(u)>::dim;
    using EulerConsVar               = EulerLayout<dim>;

    auto& mesh = u.mesh();

    u.resize();
    auto set_conserved = [eos](auto&& u, double rho, double p, double v)
    {
        u[EulerConsVar::rho] = rho;
        double norm2         = 0.;
        for (std::size_t d = 0; d < dim; ++d)
        {
            u[EulerConsVar::mom(d)] = rho * v;
            norm2 += v * v;
        }
        u[EulerConsVar::rhoE] = rho * (eos.e(rho, p) + 0.5 * norm2);
    };

    samurai::for_each_cell(mesh,
                           [&](auto& cell)
                           {
                               auto x = cell.center();

                               if (x[0] < 0.5)
                               {
                                   set_conserved(u[cell], rhoL, pL, vL);
                               }
                               else
                               {
                                   set_conserved(u[cell], rhoR, pR, vR);
                               }
                           });
}

void update_p(auto& p, auto& u, auto eos)
{
    static constexpr std::size_t dim = std::decay_t<decltype(u)>::dim;
    using EulerConsVar               = EulerLayout<dim>;

    auto& mesh = u.mesh();
    samurai::for_each_cell(mesh,
                           [&](const auto& cell)
                           {
                               double rho = u[cell][EulerConsVar::rho];
                               double v   = u[cell][EulerConsVar::mom(0)] / rho;
                               p[cell]    = eos.p(rho, (u[cell][EulerConsVar::rhoE] / rho) - 0.5 * v * v);
                           });
}

bool check_positivity(const auto& p)
{
    bool positive = true;

    auto& mesh = p.mesh();

    samurai::for_each_cell(mesh,
                           [&](const auto& cell)
                           {
                               if (p[cell] < 0.)
                               {
                                   positive = false;
                               }
                           });
    return positive;
}

int main(int argc, char* argv[])
{
    constexpr std::size_t dim = 1;
    using Config              = samurai::MRConfig<dim>;

    const EOS::IdealGas eos = EOS::air;

    auto& app = samurai::initialize("Euler equations solver", argc, argv);

    // Simulation parameters
    xt::xtensor_fixed<double, xt::xshape<dim>> min_corner = {0.};
    xt::xtensor_fixed<double, xt::xshape<dim>> max_corner = {1.};

    double Tf  = .15;
    double cfl = 0.45;
    double t   = 0.;
    std::string restart_file;
    std::string scheme = "hll";

    // Output parameters
    fs::path path        = fs::current_path();
    std::string filename = fmt::format("euler_{}d", dim);
    std::size_t nfiles   = 1;

    app.add_option("--min-corner", min_corner, "The min corner of the box")->capture_default_str()->group("Simulation parameters");
    app.add_option("--max-corner", max_corner, "The max corner of the box")->capture_default_str()->group("Simulation parameters");
    app.add_option("--cfl", cfl, "The CFL")->capture_default_str()->group("Simulation parameters");
    app.add_option("--Ti", t, "Initial time")->capture_default_str()->group("Simulation parameters");
    app.add_option("--Tf", Tf, "Final time")->capture_default_str()->group("Simulation parameters");
    app.add_option("--scheme", scheme, "Finite volume scheme")
        ->capture_default_str()
        ->check(CLI::IsMember({"rusanov", "hll", "hllc"}))
        ->group("Simulation parameters");
    app.add_option("--restart-file", restart_file, "Restart file")->capture_default_str()->group("Simulation parameters");
    app.add_option("--path", path, "Output path")->capture_default_str()->group("Output");
    app.add_option("--filename", filename, "File name prefix")->capture_default_str()->group("Output");
    app.add_option("--nfiles", nfiles, "Number of output files")->capture_default_str()->group("Output");

    SAMURAI_PARSE(argc, argv);
    std::cout << "Samurai version: " << SAMURAI_VERSION << std::endl; // Print Samurai version info

    // Initialize the mesh
    const samurai::Box<double, dim> box(min_corner, max_corner);
    auto config = samurai::mesh_config<dim>().min_level(8).max_level(8).max_stencil_size(2).disable_minimal_ghost_width();
    config.parse_args();

    auto mesh = samurai::mra::make_empty_mesh(config);
    auto u    = samurai::make_vector_field<double, 2 + dim>("euler", mesh);

    if (restart_file.empty())
    {
        mesh = samurai::mra::make_mesh(box, config);
        init(u, eos);
    }
    else
    {
        samurai::load(restart_file, mesh, u);
    }

    const xt::xtensor_fixed<int, xt::xshape<1>> left  = {-1};
    const xt::xtensor_fixed<int, xt::xshape<1>> right = {1};

    samurai::make_bc<samurai::Dirichlet<1>>(u, rhoL, rhoL * (eos.e(rhoL, pL) + 0.5 * vL * vL), rhoL * vL)->on(left);
    samurai::make_bc<samurai::Dirichlet<1>>(u, rhoR, rhoR * (eos.e(rhoR, pR) + 0.5 * vR * vR), rhoR * vR)->on(right);

    auto unp1 = samurai::make_vector_field<double, 2 + dim>("euler", mesh);

    double dx            = mesh.cell_length(config.max_level());
    const double dt_save = Tf / static_cast<double>(nfiles);
    std::size_t nsave    = 1;
    std::size_t nt       = 0;

    samurai::save("results", fmt::format("{}_{}_init", filename, scheme), mesh, u);

    std::cout << "Using scheme: " << scheme << std::endl;
    // This demo stays first order: it is here for the user-defined prediction
    // operator, not for the scheme.
    auto fv_scheme = make_first_order_scheme<decltype(u)>(scheme, eos);

    auto prediction_fn = [&](auto& new_field, const auto& old_field)
    {
        return make_field_operator_function<Euler_prediction_op>(new_field, old_field, eos);
    };

    auto p = samurai::make_scalar_field<double>("p", mesh);

    // auto MRadaptation = samurai::make_MRAdapt(p);
    auto MRadaptation = samurai::make_MRAdapt(prediction_fn, p);
    auto mra_config   = samurai::mra_config().relative_detail(true);

    while (t != Tf)
    {
        update_p(p, u, eos);

        MRadaptation(mra_config, u);

        double dt = cfl * dx / get_max_lambda(u, eos);
        t += dt;

        if (std::isnan(t))
        {
            std::cerr << "Error: Time became NaN, stopping simulation" << std::endl;
            break;
        }

        if (check_positivity(p) == false)
        {
            std::cerr << "Error: Negative pressure detected, stopping simulation" << std::endl;
            break;
        }

        if (t > Tf)
        {
            dt += Tf - t;
            t = Tf;
        }

        std::cout << fmt::format("iteration {}: t = {}, dt = {}", nt++, t, dt) << std::endl;

        unp1.resize();
        unp1 = u - dt * fv_scheme(u);

        samurai::swap(u, unp1);

        if (t >= static_cast<double>(nsave + 1) * dt_save || t == Tf)
        {
            const std::string suffix = (nfiles != 1) ? fmt::format("_ite_{}", nsave++) : "";
            samurai::save("results", fmt::format("{}_{}{}", filename, scheme, suffix), mesh, u, p);
        }
    }

    samurai::finalize();
    return 0;
}
