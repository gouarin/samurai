cmake_minimum_required(VERSION 3.11)

if (CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    project(samurai-benchmark)

    find_package(samurai REQUIRED CONFIG)
    set(SAMURAI_INCLUDE_DIR ${samurai_INCLUDE_DIRS})
endif ()

include(CheckCXXCompilerFlag)

include(addGoogleTest)
include(addGoogleBench)

find_package(Threads)

set(SAMURAI_BENCHMARKS
    benchmark_celllist_construction.cpp
    benchmark_laplacian.cpp
    benchmark_search.cpp
    benchmark_set.cpp
    main.cpp
)

add_executable(bench_samurai ${SAMURAI_BENCHMARKS})
target_link_libraries(bench_samurai samurai benchmark::benchmark xtensor::optimize xtensor::use_xsimd)
