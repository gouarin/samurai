#pragma once
#include "fv/cell_based_scheme_assembly.hpp"
#include "fv/flux_based_scheme_assembly.hpp"
#include "fv/operator_sum_assembly.hpp"
#include "utils.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <petsc.h>

namespace samurai
{
    namespace petsc
    {
        /**
         * Solver for a set of independent local non-linear systems, one per cell
         * (cell-based scheme with stencil size 1).
         *
         * Each local system scheme(x) = rhs is solved with a hand-rolled dense Newton
         * method. The residual scheme(x) and the Jacobian d(scheme)/dx are evaluated
         * through the scheme callbacks ('local_scheme_function' and
         * 'local_jacobian_function'), and the n_comp x n_comp linear system of each
         * Newton iteration is solved directly (Gaussian elimination, partial pivoting).
         *
         * Note: a previous implementation used one PETSc SNES per cell. Because PETSc
         * re-queries its options database on every SNESSolve, that approach spent most
         * of its time in PetscOptionsFindPair when sweeping over the (many, tiny) local
         * systems. The dense Newton avoids that machinery entirely while producing the
         * same solution.
         */
        template <class Scheme>
        class NonLinearLocalSolvers
        {
            using scheme_t      = Scheme;
            using cfg_t         = typename scheme_t::cfg_t;
            using field_t       = typename scheme_t::field_t;
            using mesh_t        = typename field_t::mesh_t;
            using field_value_t = typename field_t::value_type;
            using cell_t        = Cell<mesh_t::dim, typename mesh_t::interval_t>;

            static constexpr std::size_t n_comp = field_t::n_comp;

          protected:

            field_t* m_unknown = nullptr;
            scheme_t m_scheme;
            bool m_is_set_up = false;

            // One worker per thread for the scheme value scheme(x) and the Jacobian d(scheme)/dx.
            std::vector<SchemeValue<cfg_t>> m_f_scheme_value;
            std::vector<JacobianMatrix<cfg_t>> m_J_coeffs;

            std::size_t m_n_threads = 1;

          public:

            // Newton stopping criteria (same defaults as the PETSc SNES that this solver replaces).
            // Convergence is reached when the residual is small,
            //     ||scheme(x) - rhs|| <= newton_atol + newton_rtol * ||initial residual||,
            // or when the Newton step becomes negligible,
            //     ||dx|| <= newton_stol * ||x||.
            // The step criterion is what lets cells already at equilibrium (tiny initial
            // residual, hence an unreachable relative tolerance) converge in one iteration.
            field_value_t newton_rtol         = 1e-8;
            field_value_t newton_atol         = 1e-50;
            field_value_t newton_stol         = 1e-8;
            std::size_t newton_max_iterations = 50;

            explicit NonLinearLocalSolvers(const scheme_t& scheme)
                : m_scheme(scheme)
            {
                if (!m_scheme.scheme_definition().local_scheme_function)
                {
                    std::cerr << "The scheme function 'local_scheme_function' of operator '" << scheme.name()
                              << "' has not been implemented." << std::endl;
                    assert(false && "Undefined 'local_scheme_function'");
                    exit(EXIT_FAILURE);
                }
                if (!m_scheme.scheme_definition().local_jacobian_function)
                {
                    std::cerr << "The function 'local_jacobian_function' of operator '" << scheme.name() << "' has not been implemented."
                              << std::endl;
                    assert(false && "Undefined 'local_jacobian_function'");
                    exit(EXIT_FAILURE);
                }

#ifdef SAMURAI_WITH_OPENMP
                m_n_threads = static_cast<std::size_t>(omp_get_max_threads());
#else
                m_n_threads = 1;
#endif
                m_f_scheme_value.resize(m_n_threads);
                m_J_coeffs.resize(m_n_threads);
            }

            NonLinearLocalSolvers& operator=(const NonLinearLocalSolvers& other)
            {
                if (this != &other)
                {
                    this->m_unknown = other.m_unknown;
                }
                return *this;
            }

            NonLinearLocalSolvers& operator=(NonLinearLocalSolvers&& other)
            {
                if (this != &other)
                {
                    this->m_unknown = other.m_unknown;
                    other.m_unknown = nullptr;
                }
                return *this;
            }

            auto& scheme()
            {
                return m_scheme;
            }

            void set_unknown(field_t& u)
            {
                m_unknown = &u;
            }

            field_t& unknown()
            {
                return *m_unknown;
            }

            bool is_set_up()
            {
                return m_is_set_up;
            }

            void setup()
            {
                if (is_set_up())
                {
                    return;
                }

                if (!m_unknown)
                {
                    std::cerr << "Undefined unknown for this non-linear system. Please set the unknowns using the instruction '[solver].set_unknown(u);'."
                              << std::endl;
                    assert(false && "Undefined unknown");
                    exit(EXIT_FAILURE);
                }

                m_is_set_up = true;
            }

            void reset()
            {
                m_is_set_up = false;
            }

            void set_scheme(const scheme_t& s)
            {
                m_scheme = s;
            }

            void solve(field_t& rhs)
            {
                if (!m_is_set_up)
                {
                    this->setup();
                }

                times::timers.start("non-linear local solves");

#ifdef SAMURAI_WITH_OPENMP
                static constexpr Run run_type = Run::Parallel;
#else
                static constexpr Run run_type = Run::Sequential;
#endif
                for_each_cell<run_type>(unknown().mesh(),
                                        [&](auto& cell)
                                        {
#ifdef SAMURAI_WITH_OPENMP
                                            std::size_t thread_num = static_cast<std::size_t>(omp_get_thread_num());
#else
                                            std::size_t thread_num = 0;
#endif
                                            newton_local_solve(cell, rhs, thread_num);
                                        });

                times::timers.stop("non-linear local solves");
            }

            void solve(field_t& unknown, field_t& rhs)
            {
                set_unknown(unknown);
                solve(rhs);
            }

          private:

            // Solve the local non-linear system scheme(x) = rhs(cell) by Newton's method,
            // starting from the current value of the unknown in 'cell'.
            void newton_local_solve(const cell_t& cell, field_t& rhs, std::size_t thread_num)
            {
                auto& f_value = m_f_scheme_value[thread_num]; // worker: scheme(x)
                auto& jac     = m_J_coeffs[thread_num];       // worker: d(scheme)/dx

                std::array<field_value_t, n_comp> x; // current iterate (local initial guess)
                std::array<field_value_t, n_comp> b; // local right-hand side
                for (std::size_t i = 0; i < n_comp; ++i)
                {
                    x[i] = unknown()[cell](i);
                    b[i] = rhs[cell](i);
                }

                field_value_t norm0 = 0;
                bool converged      = false;
                for (std::size_t iter = 0; iter <= newton_max_iterations; ++iter)
                {
                    LocalField<field_t> x_field(cell, x.data());

                    // residual r = scheme(x) - b
                    m_scheme.scheme_definition().local_scheme_function(f_value, cell, x_field);
                    std::array<field_value_t, n_comp> r;
                    field_value_t norm = 0;
                    for (std::size_t i = 0; i < n_comp; ++i)
                    {
                        r[i] = f_value(i) - b[i];
                        norm += r[i] * r[i];
                    }
                    norm = std::sqrt(norm);
                    if (iter == 0)
                    {
                        norm0 = norm;
                    }
                    if (norm <= newton_atol + newton_rtol * norm0)
                    {
                        converged = true;
                        break;
                    }

                    // Jacobian J = d(scheme)/dx, then Newton update x -= J^{-1} r
                    m_scheme.scheme_definition().local_jacobian_function(jac, cell, x_field);
                    std::array<field_value_t, n_comp> dx;
                    dense_solve(jac, r, dx);

                    field_value_t step_norm2 = 0;
                    field_value_t x_norm2    = 0;
                    for (std::size_t i = 0; i < n_comp; ++i)
                    {
                        x[i] -= dx[i];
                        step_norm2 += dx[i] * dx[i];
                        x_norm2 += x[i] * x[i];
                    }
                    if (std::sqrt(step_norm2) <= newton_stol * std::sqrt(x_norm2))
                    {
                        converged = true;
                        break;
                    }
                }

                if (!converged)
                {
                    std::cerr << "Divergence of the local non-linear solver: Newton did not converge in " << newton_max_iterations
                              << " iterations." << std::endl;
                    assert(false && "Divergence of the local non-linear solver");
                    exit(EXIT_FAILURE);
                }

                for (std::size_t i = 0; i < n_comp; ++i)
                {
                    unknown()[cell][i] = x[i];
                }
            }

            // Solve the dense system J sol = rhs (Gaussian elimination with partial pivoting).
            static void dense_solve(const JacobianMatrix<cfg_t>& J,
                                    const std::array<field_value_t, n_comp>& rhs,
                                    std::array<field_value_t, n_comp>& sol)
            {
                std::array<std::array<field_value_t, n_comp>, n_comp> a;
                std::array<field_value_t, n_comp> b = rhs;
                for (std::size_t i = 0; i < n_comp; ++i)
                {
                    for (std::size_t j = 0; j < n_comp; ++j)
                    {
                        a[i][j] = J(i, j);
                    }
                }

                for (std::size_t k = 0; k < n_comp; ++k)
                {
                    std::size_t piv = k;
                    for (std::size_t i = k + 1; i < n_comp; ++i)
                    {
                        if (std::abs(a[i][k]) > std::abs(a[piv][k]))
                        {
                            piv = i;
                        }
                    }
                    if (piv != k)
                    {
                        std::swap(a[piv], a[k]);
                        std::swap(b[piv], b[k]);
                    }
                    assert(a[k][k] != 0 && "Singular Jacobian in the local non-linear solver");
                    for (std::size_t i = k + 1; i < n_comp; ++i)
                    {
                        const field_value_t factor = a[i][k] / a[k][k];
                        for (std::size_t j = k; j < n_comp; ++j)
                        {
                            a[i][j] -= factor * a[k][j];
                        }
                        b[i] -= factor * b[k];
                    }
                }

                for (std::size_t k = n_comp; k-- > 0;)
                {
                    field_value_t s = b[k];
                    for (std::size_t j = k + 1; j < n_comp; ++j)
                    {
                        s -= a[k][j] * sol[j];
                    }
                    sol[k] = s / a[k][k];
                }
            }
        };

    } // end namespace petsc
} // end namespace samurai
