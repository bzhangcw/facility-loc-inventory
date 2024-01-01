# third party
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

add_executable(dp_main main.cpp sol.h sol.cpp dp.h dp.cpp tail.cpp tail.h action.cpp action.h state.cpp state.h problem_queue.cpp problem_queue.h action.h)
target_link_libraries(dp_main PRIVATE nlohmann_json::nlohmann_json)

add_executable(dp_test_io
        test_io.cpp tail.cpp tail.h action.cpp action.h state.cpp state.h problem_queue.cpp problem_queue.h
)
target_link_libraries(dp_test_io PRIVATE nlohmann_json::nlohmann_json cppdp
)
add_executable(dp_test_sample
        test_sample.cpp
)
target_link_libraries(dp_test_sample nlohmann_json::nlohmann_json cppdp)

add_library(cppdp SHARED
        sol.h sol.cpp
        dp.h dp.cpp
        tail.cpp tail.h
        action.cpp action.h
        state.cpp state.h
        problem_queue.cpp problem_queue.h
        )
target_link_libraries(cppdp PRIVATE nlohmann_json::nlohmann_json)

set(CPPREPAIRDP_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/cpp/dp")