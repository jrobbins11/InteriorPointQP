cmake_minimum_required(VERSION 3.15)
project(zonoopt_unit_tests)

include(GoogleTest)

set(TEST_NAME interior_point_qp_test)
set(SOURCE "interior_point_qp_tests.cpp")
add_executable(${TEST_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE})
target_link_libraries(${TEST_NAME} PRIVATE InteriorPointQP GTest::gtest_main)
gtest_discover_tests(${TEST_NAME})