find_package(OpenMP)
add_executable(test_openmp test_openmp.cpp)
target_link_libraries(test_openmp samurai xtensor::optimize OpenMP::OpenMP_CXX)
