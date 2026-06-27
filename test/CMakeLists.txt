cmake_minimum_required(VERSION 3.15)
project(zonoopt_unit_tests)

include(GoogleTest)

set(TEST_NAME mehrotra_qp_test)
set(SOURCE "mehrotra_qp_tests.cpp")
add_executable(${TEST_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE})
target_link_libraries(${TEST_NAME} PRIVATE MehrotraQP GTest::gtest_main)
gtest_discover_tests(${TEST_NAME})