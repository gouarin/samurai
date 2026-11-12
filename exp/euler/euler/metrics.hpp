// Copyright 2025 the samurai team
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <chrono>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>

#include <fmt/format.h>

#ifdef SAMURAI_WITH_MPI
#include <boost/mpi.hpp>
namespace mpi = boost::mpi;
#endif

// =============================================================================
//  Performance metrics
// -----------------------------------------------------------------------------
//  The article this repository reproduces is first of all a performance paper,
//  and it reports three numbers per run. They are cheap to measure and useless
//  to guess, so the solver measures them itself rather than leaving them to be
//  reconstructed from a log:
//
//      sparsity index   cells of the adapted mesh over the cells a uniform mesh
//                       at max-level would hold, as a percentage. 100% is a mesh
//                       that never coarsened. Reported at the initial and at the
//                       final time, as the article does: the mesh of a Riemann
//                       problem fills up as the waves spread, and one number
//                       taken at one end would flatter or damn it.
//
//      Mcu/s            millions of cell updates per second, a cell update being
//                       one cell advanced by one time step. Counted as one per
//                       cell per time step whatever the integrator does
//                       internally: SSP-RK2 evaluates the flux twice and Strang
//                       sweeps 2*dim - 1 times, and counting those would make a
//                       more expensive integrator look faster. It is the same
//                       convention as the article, whose scheme is the
//                       single-pass MUSCL-Hancock.
//
//      time to solution the time loop alone. Initialization and output are
//                       measured too but kept out of it: nfiles is a choice of
//                       the person running, not a property of the solver.
//
//  Plus the fraction of the time to solution spent adapting the mesh, which is
//  what the comparison of a multiresolution against a gradient criterion is
//  really about.
//
//  All of it is accumulated on the local subdomain and reduced once, at print
//  time: cells and updates are summed over the ranks, times are taken at their
//  maximum, which is the rank that everyone else waits for.
// =============================================================================

class Metrics
{
  public:

    // The denominator of the sparsity index is fixed once, from the domain at
    // max-level: it is the mesh the run would have used without adaptation. The
    // levels and the dimension travel with it so that a metrics file says which
    // run it describes without the command line that produced it.
    template <class Mesh>
    explicit Metrics(const Mesh& mesh)
        : m_uniform_cells(mesh.domain().nb_cells())
        , m_dim(Mesh::dim)
        , m_min_level(mesh.min_level())
        , m_max_level(mesh.max_level())
    {
    }

    // Starts the time to solution, and records the mesh the run starts from.
    template <class Mesh>
    void start(const Mesh& mesh)
    {
        m_initial_cells = local_cells(mesh);
        m_start         = clock::now();
    }

    // One time step over the cells the mesh carries, whatever the integrator
    // does inside: one cell advanced by one time step is one cell update.
    template <class Mesh>
    void step(const Mesh& mesh)
    {
        m_cell_updates += local_cells(mesh);
        ++m_steps;
    }

    // The mesh adaptation, timed apart. It is part of the time to solution.
    template <class Fn>
    void adapt(Fn&& adaptation)
    {
        const auto begin = clock::now();
        adaptation();
        m_adapt_time += seconds_since(begin);
    }

    // Writing a file, timed apart and subtracted from the time to solution.
    template <class Fn>
    void output(Fn&& write)
    {
        const auto begin = clock::now();
        write();
        m_output_time += seconds_since(begin);
    }

    // Stops the time to solution and records the mesh the run ends on.
    template <class Mesh>
    void stop(const Mesh& mesh)
    {
        m_run_time    = seconds_since(m_start) - m_output_time;
        m_final_cells = local_cells(mesh);
    }

    // Everything above, reduced over the ranks. Call once, after stop().
    struct Summary
    {
        std::size_t dim             = 0;
        std::size_t min_level       = 0;
        std::size_t max_level       = 0;
        std::uint64_t uniform_cells = 0;
        std::uint64_t initial_cells = 0;
        std::uint64_t final_cells   = 0;
        std::uint64_t cell_updates  = 0;
        std::uint64_t steps         = 0;
        double run_time             = 0.;
        double adapt_time           = 0.;
        double output_time          = 0.;

        double initial_sparsity() const
        {
            return sparsity(initial_cells);
        }

        double final_sparsity() const
        {
            return sparsity(final_cells);
        }

        // Millions of cell updates per second, averaged over the whole run.
        double mcu_per_second() const
        {
            return run_time > 0. ? static_cast<double>(cell_updates) / run_time * 1e-6 : 0.;
        }

        double adapt_fraction() const
        {
            return run_time > 0. ? 100. * adapt_time / run_time : 0.;
        }

      private:

        double sparsity(std::uint64_t cells) const
        {
            return uniform_cells > 0 ? 100. * static_cast<double>(cells) / static_cast<double>(uniform_cells) : 0.;
        }
    };

    // Prints the block below and, when given a name, writes the same numbers as
    // JSON. Collective: the reduction is inside, so every rank calls it and only
    // the root one writes.
    void report(const std::string& filename = "") const
    {
        const auto s = summary();
        if (!is_root())
        {
            return;
        }
        print(s);
        write(s, filename);
    }

    // The reduced numbers, for a caller that wants them rather than a report.
    // Collective as well.
    Summary summary() const
    {
        Summary s{.dim           = m_dim,
                  .min_level     = m_min_level,
                  .max_level     = m_max_level,
                  .uniform_cells = m_uniform_cells,
                  .initial_cells = m_initial_cells,
                  .final_cells   = m_final_cells,
                  .cell_updates  = m_cell_updates,
                  .steps         = m_steps,
                  .run_time      = m_run_time,
                  .adapt_time    = m_adapt_time,
                  .output_time   = m_output_time};
#ifdef SAMURAI_WITH_MPI
        mpi::communicator world;
        // The uniform mesh is the whole domain and is already global; the rest
        // is what this rank carried.
        s.initial_cells = mpi::all_reduce(world, s.initial_cells, std::plus<std::uint64_t>());
        s.final_cells   = mpi::all_reduce(world, s.final_cells, std::plus<std::uint64_t>());
        s.cell_updates  = mpi::all_reduce(world, s.cell_updates, std::plus<std::uint64_t>());
        s.run_time      = mpi::all_reduce(world, s.run_time, mpi::maximum<double>());
        s.adapt_time    = mpi::all_reduce(world, s.adapt_time, mpi::maximum<double>());
        s.output_time   = mpi::all_reduce(world, s.output_time, mpi::maximum<double>());
#endif
        return s;
    }

  private:

    // The block printed at the end of a run, in the conventions above.
    static void print(const Summary& s)
    {
        std::cout << "\nperformance" << std::endl;
        std::cout << fmt::format("  cells            {} -> {} of {} uniform", s.initial_cells, s.final_cells, s.uniform_cells) << std::endl;
        std::cout << fmt::format("  sparsity index   {:.2f}% -> {:.2f}%", s.initial_sparsity(), s.final_sparsity()) << std::endl;
        std::cout << fmt::format("  cell updates     {} over {} time steps", s.cell_updates, s.steps) << std::endl;
        std::cout << fmt::format("  time to solution {:.2f} s, {:.1f}% of it adapting the mesh", s.run_time, s.adapt_fraction()) << std::endl;
        std::cout << fmt::format("  throughput       {:.2f} Mcu/s", s.mcu_per_second()) << std::endl;
    }

    // The same numbers as JSON, for whoever builds a table out of several runs.
    static void write(const Summary& s, const std::string& filename)
    {
        if (filename.empty())
        {
            return;
        }

        std::ofstream out(filename);
        if (!out)
        {
            throw std::runtime_error("cannot write the metrics file " + filename);
        }

        out << fmt::format("{{\n"
                           "  \"dim\": {},\n"
                           "  \"min_level\": {},\n"
                           "  \"max_level\": {},\n"
                           "  \"uniform_cells\": {},\n"
                           "  \"initial_cells\": {},\n"
                           "  \"final_cells\": {},\n"
                           "  \"initial_sparsity\": {:.6e},\n"
                           "  \"final_sparsity\": {:.6e},\n"
                           "  \"cell_updates\": {},\n"
                           "  \"steps\": {},\n"
                           "  \"run_time\": {:.6e},\n"
                           "  \"adapt_time\": {:.6e},\n"
                           "  \"output_time\": {:.6e},\n"
                           "  \"adapt_fraction\": {:.6e},\n"
                           "  \"mcu_per_second\": {:.6e}\n"
                           "}}\n",
                           s.dim,
                           s.min_level,
                           s.max_level,
                           s.uniform_cells,
                           s.initial_cells,
                           s.final_cells,
                           s.initial_sparsity(),
                           s.final_sparsity(),
                           s.cell_updates,
                           s.steps,
                           s.run_time,
                           s.adapt_time,
                           s.output_time,
                           s.adapt_fraction(),
                           s.mcu_per_second());
    }

    using clock = std::chrono::steady_clock;

    // What this rank holds, the leaves only: ghosts are not cells of the mesh.
    template <class Mesh>
    static std::uint64_t local_cells(const Mesh& mesh)
    {
        return mesh.nb_cells(Mesh::mesh_id_t::cells);
    }

    static double seconds_since(const clock::time_point& begin)
    {
        return std::chrono::duration<double>(clock::now() - begin).count();
    }

    static bool is_root()
    {
#ifdef SAMURAI_WITH_MPI
        return mpi::communicator().rank() == 0;
#else
        return true;
#endif
    }

    std::uint64_t m_uniform_cells = 0;
    std::size_t m_dim             = 0;
    std::size_t m_min_level       = 0;
    std::size_t m_max_level       = 0;
    std::uint64_t m_initial_cells = 0;
    std::uint64_t m_final_cells   = 0;
    std::uint64_t m_cell_updates  = 0;
    std::uint64_t m_steps         = 0;

    clock::time_point m_start{};
    double m_run_time    = 0.;
    double m_adapt_time  = 0.;
    double m_output_time = 0.;
};
