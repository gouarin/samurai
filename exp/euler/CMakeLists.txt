# Copy of samurai-euler, built against this tree for the eager array experiment.
foreach(name 1d 2d 3d user_pred_1d)
  add_executable(euler_${name} main_${name}.cpp)
  target_link_libraries(euler_${name} PRIVATE samurai CLI11::CLI11)
  target_compile_definitions(euler_${name} PRIVATE SAMURAI_VERSION="exp-eager-array")
endforeach()
