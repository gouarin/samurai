# Copy of samurai-euler (2D case only), built against this tree for the eager array experiment.
add_executable(euler_2d main_2d.cpp)
target_link_libraries(euler_2d PRIVATE samurai CLI11::CLI11)
target_compile_definitions(euler_2d PRIVATE SAMURAI_VERSION="exp-eager-array")
