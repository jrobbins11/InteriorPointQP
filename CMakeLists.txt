cmake_minimum_required(VERSION 3.15...3.27)

# project name
project(InteriorPointQP
        VERSION 0.0.0
        LANGUAGES CXX
)
set(TARGET_VAR InteriorPointQP)

# testing
include(CTest)

# compile commands
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Eigen dependency
include(FetchContent)

find_package(Eigen3 3.3 QUIET NO_MODULE)
if (NOT Eigen3_FOUND)
    message(STATUS "Eigen3 not found, fetching 5.0.0 via FetchContent...")
    FetchContent_Declare(
            eigen_default
            GIT_REPOSITORY "https://gitlab.com/libeigen/eigen.git"
            GIT_TAG        "5.0.0"
    )
    FetchContent_MakeAvailable(eigen_default)
endif()

if (NOT TARGET Eigen3::Eigen)
    message(FATAL_ERROR "Failed to fetch Eigen dependency.")
endif()

# source files
set(SOURCE_FILES
    src/InteriorPointQP.cpp
)

# build
add_library(${TARGET_VAR} STATIC ${SOURCE_FILES})

# link dependencies
target_link_libraries(${TARGET_VAR} PUBLIC Eigen3::Eigen)

# include directories
target_include_directories(${TARGET_VAR} PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

# C++ standard
target_compile_features(${TARGET_VAR}
        PUBLIC cxx_std_17
)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# debug flags
if (MSVC)
    target_compile_options(${TARGET_VAR} PRIVATE /bigobj /W3)
else()
    target_compile_options(${TARGET_VAR} PRIVATE -Wall -Wextra -Wpedantic)
endif()

# testing
if (BUILD_TESTING)
    find_package(GTest CONFIG QUIET)
    if (NOT GTest_FOUND)
        message(STATUS "GTest not found, fetching v1.14.0 via FetchContent...")
        FetchContent_Declare(
            googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG v1.14.0
        )
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
        set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(googletest)
    endif()
    enable_testing()
    add_subdirectory(test)
endif()