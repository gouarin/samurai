cmake_minimum_required(VERSION 3.11)

if (CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    project(samurai-benchmark)

    find_package(samurai REQUIRED CONFIG)
    set(SAMURAI_INCLUDE_DIR ${samurai_INCLUDE_DIRS})
endif ()

message(STATUS "Forcing benchmark build type to Release")
set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)

include(CheckCXXCompilerFlag)

string(TOUPPER "${CMAKE_BUILD_TYPE}" U_CMAKE_BUILD_TYPE)

include(addGoogleTest)
include(addGoogleBench)

find_package(Threads)

# Self-sufficient benchmarks: one source file, one executable, each file
# providing its own main (BENCHMARK_MAIN() or, where samurai::initialize()/
# finalize() is required, a hand-written one), linking only samurai and
# Google Benchmark. To add a benchmark, add its source file to this list;
# samurai_add_benchmark() below turns benchmark_<name>.cpp into the
# bench_<name> executable.
set(SAMURAI_BENCHMARK_SOURCES
    benchmark_celllist_construction.cpp
    benchmark_detail_computation.cpp
    benchmark_dynamic_set.cpp
    benchmark_fv_scheme.cpp
    benchmark_search.cpp
    benchmark_set_call_sites.cpp
    benchmark_set_graduation.cpp
    benchmark_set_library_ops.cpp
    benchmark_set_mesh_shape.cpp
    benchmark_set_operators.cpp
    benchmark_set_projection.cpp
    benchmark_update_fields.cpp
    benchmark_update_ghost.cpp
)

function(samurai_add_benchmark source)
    get_filename_component(name ${source} NAME_WE)
    string(REPLACE "benchmark_" "bench_" target ${name})
    add_executable(${target} ${source})
    target_link_libraries(${target} samurai benchmark::benchmark)
endfunction()

foreach(source IN LISTS SAMURAI_BENCHMARK_SOURCES)
    samurai_add_benchmark(${source})
endforeach()

# bench_set_call_sites, bench_dynamic_set and bench_set_graduation read the
# sets of the MR-adapted meshes from files, so they compile with the interval
# structures and the set algebra alone. generate_set_algebra_inputs writes the files. The build runs
# it once, when the directory has no stamp, and not when the library changes,
# so the inputs stay the same while the set algebra does. With
# SAMURAI_BENCHMARK_GENERATE_SET_ALGEBRA_INPUTS=OFF building these binaries
# no longer builds the generator, which is then run by hand.
# SAMURAI_SET_ALGEBRA_INPUTS in the environment overrides the directory at run
# time.
option(SAMURAI_BENCHMARK_GENERATE_SET_ALGEBRA_INPUTS "Generate the set algebra inputs when building the benchmarks that read them" ON)
set(SAMURAI_SET_ALGEBRA_INPUTS_DIR ${CMAKE_CURRENT_BINARY_DIR}/set_algebra_inputs)
add_executable(generate_set_algebra_inputs generate_set_algebra_inputs.cpp)
target_link_libraries(generate_set_algebra_inputs samurai benchmark::benchmark)
set(SAMURAI_SET_ALGEBRA_INPUT_READERS bench_set_call_sites bench_dynamic_set bench_set_graduation)
foreach(target generate_set_algebra_inputs ${SAMURAI_SET_ALGEBRA_INPUT_READERS})
    target_compile_definitions(${target} PRIVATE SAMURAI_SET_ALGEBRA_INPUTS_DIR="${SAMURAI_SET_ALGEBRA_INPUTS_DIR}")
endforeach()
if(SAMURAI_BENCHMARK_GENERATE_SET_ALGEBRA_INPUTS)
    # Naming the generator in COMMAND, not in DEPENDS, builds it first without
    # running it again each time it is rebuilt.
    add_custom_command(
        OUTPUT ${SAMURAI_SET_ALGEBRA_INPUTS_DIR}/generated.stamp
        COMMAND generate_set_algebra_inputs
        COMMAND ${CMAKE_COMMAND} -E touch ${SAMURAI_SET_ALGEBRA_INPUTS_DIR}/generated.stamp
        COMMENT "Generating the set algebra inputs in ${SAMURAI_SET_ALGEBRA_INPUTS_DIR}"
    )
    add_custom_target(set_algebra_inputs DEPENDS ${SAMURAI_SET_ALGEBRA_INPUTS_DIR}/generated.stamp)
    foreach(target IN LISTS SAMURAI_SET_ALGEBRA_INPUT_READERS)
        add_dependencies(${target} set_algebra_inputs)
    endforeach()
endif()

# Load balancing micro-benchmarks: MPI-only, its own MPI-aware main(), and
# optional extra links (ParMETIS / PT-Scotch), so it is built outside the
# generic loop above.
if(WITH_MPI)
    add_executable(bench_load_balancing benchmark_load_balancing.cpp)
    target_include_directories(bench_load_balancing PRIVATE ${SAMURAI_INCLUDE_DIR})
    target_link_libraries(bench_load_balancing samurai benchmark::benchmark Threads::Threads)

    if(SAMURAI_WITH_PARMETIS)
        target_compile_definitions(bench_load_balancing PRIVATE SAMURAI_WITH_PARMETIS)
        target_link_libraries(bench_load_balancing ParMETIS::ParMETIS ParMETIS::Metis)
    endif()

    if(SAMURAI_WITH_PTSCOTCH)
        target_compile_definitions(bench_load_balancing PRIVATE SAMURAI_WITH_PTSCOTCH)
        target_link_libraries(bench_load_balancing PTScotch::PTScotch PTScotch::Scotch)
    endif()
endif()
