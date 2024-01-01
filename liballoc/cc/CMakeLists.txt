find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})


# add subdirectories
# == dp ===
add_subdirectory(dp)
