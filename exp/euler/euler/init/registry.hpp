// Copyright 2025 the samurai team
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <array>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include <CLI/CLI.hpp>
#include <samurai/box.hpp>

#include "../config.hpp"
#include "../eos.hpp"

// =============================================================================
//  Test case registry
// -----------------------------------------------------------------------------
//  A test case is the four things main() needs in order to set a simulation up
//  and that main() cannot guess: where the domain is, what the initial state is,
//  how the boundaries behave, and which gas is being modelled. A case that comes
//  in variants, such as the nineteen configurations of a two-dimensional Riemann
//  problem, adds a fifth: the command line option that picks one.
//
//  The registry is templated on the *field* rather than fixed to 2D, so the same
//  case can serve euler_2d and euler_3d when its definition is dimension
//  agnostic (Sedov, a free stream, a Riemann problem). Cases that only make
//  sense in one dimension simply register for that one.
//
//  It is templated on the equation of state as well. Unlike the flux kernel,
//  which is generic, the registry has to type-erase the initial state and the
//  boundary conditions behind a std::function, so their signature must name one
//  concrete state law. Every monofluid case is an ideal gas, hence the default;
//  a two-phase model would instantiate its own registry on StiffenedGas.
//
//  A test case header is self-sufficient: it exposes `definition<Field>()` and
//  registers itself with the macro at the bottom of this file, so adding
//  a case means adding one file and one #include to cases.hpp, never editing a
//  list somewhere else. Self-registration needs a concrete field type, so the
//  case states the dimensions it is written for as the trailing arguments of the
//  macro; a dimension agnostic one passes 2 and 3 and is available to both
//  binaries. The unused instantiation that costs euler_2d measures at about a
//  second of compile time, which is not a reason to give the property up.
// =============================================================================

namespace test_case
{
    template <class Field>
    using BoxFunc = std::function<samurai::Box<double, Field::dim>()>;

    template <class Field, class Eos>
    using InitFunc = std::function<void(Field&, const typename Field::cell_t&, Eos)>;

    template <class Field, class Eos>
    using BCFunc = std::function<void(Field&, double&, Eos)>;

    // Command line options a case adds for itself. A parameterised case cannot
    // read its parameter at registration, which happens before main() parses
    // anything, so it hands over a function that declares the option and keeps
    // the variable the parser writes into.
    using OptionsFunc = std::function<void(CLI::App&)>;

    template <class Field, class Eos = EOS::IdealGas>
    struct TestCase
    {
        BoxFunc<Field> box;
        InitFunc<Field, Eos> init;
        BCFunc<Field, Eos> bc;

        // Default gas for this case. `--gamma` on the command line overrides it.
        Eos eos = {};

        // Periodicity per axis. Set on the mesh, not on the field: samurai wraps
        // the ghost update rather than attaching a boundary condition.
        std::array<bool, Field::dim> periodic = {};

        // Options this case adds to the command line, if it takes any.
        OptionsFunc options = nullptr;
    };

    template <class Field, class Eos = EOS::IdealGas>
    class TestCaseRegistry
    {
      public:

        using test_case_t = TestCase<Field, Eos>;

        static TestCaseRegistry& instance()
        {
            static TestCaseRegistry registry;
            return registry;
        }

        void register_test_case(const std::string& name, test_case_t test_case)
        {
            test_cases_[name] = std::move(test_case);
        }

        const test_case_t& get(const std::string& name) const
        {
            auto it = test_cases_.find(name);
            if (it == test_cases_.end())
            {
                throw std::runtime_error("Test case '" + name + "' not found");
            }
            return it->second;
        }

        // Declares the options of every registered case into main()'s parser.
        //
        // Every case, not only the selected one: --test-case is itself an option
        // and is not known until the parse is over. So two cases must not ask
        // for the same option name, and the convention that keeps them apart is
        // to name the option after the case. A collision is caught here rather
        // than left to CLI11, which knows the name of the option but not which
        // cases are fighting over it.
        void add_options(CLI::App& app) const
        {
            for (const auto& [name, test_case] : test_cases_)
            {
                if (!test_case.options)
                {
                    continue;
                }

                try
                {
                    test_case.options(app);
                }
                catch (const CLI::OptionAlreadyAdded& e)
                {
                    throw std::runtime_error("test case '" + name + "' declares an option another case already declared (" + e.what()
                                             + "); name a case option after its case");
                }
            }
        }

        std::vector<std::string> available_test_cases() const
        {
            std::vector<std::string> names;
            names.reserve(test_cases_.size());
            for (const auto& [name, _] : test_cases_)
            {
                names.push_back(name);
            }
            return names;
        }

      private:

        std::map<std::string, test_case_t> test_cases_;
    };

    // Registers the case under `name` for each of the dimensions given.
    // Self-registration has to name a concrete field type, so the dimensions are
    // compile-time values; `make` hands back the definition for whichever field
    // type it is asked for, which is what lets one call serve several of them.
    template <std::size_t... Dims, class MakeDefinition>
    bool register_for_dims(const std::string& name, MakeDefinition make)
    {
        (TestCaseRegistry<typename config<Dims>::field_t>::instance().register_test_case(
             name,
             make.template operator()<typename config<Dims>::field_t>()),
         ...);
        return true;
    }
}

// Put this at the bottom of a test case header:
//
//     REGISTER_TEST_CASE(sedov_blast, test_case::sedov_blast, 2, 3)
//     REGISTER_TEST_CASE(sod, test_case::sod, 2)
//
// NAME is the string `--test-case` accepts, NS the namespace holding the case's
// `definition<Field>()`, and the rest is the list of dimensions the case is
// written for. The list is passed as trailing arguments rather than as `{2, 3}`
// because braces do not protect commas from macro argument splitting.
#define REGISTER_TEST_CASE(NAME, NS, ...)                                                                              \
    namespace                                                                                                          \
    {                                                                                                                  \
        const bool registered_##NAME = ::test_case::register_for_dims<__VA_ARGS__>(#NAME,                              \
                                                                                   []<class Field>()                   \
                                                                                   {                                   \
                                                                                       return NS::definition<Field>(); \
                                                                                   });                                 \
    }
