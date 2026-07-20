# SPDX-License-Identifier: MIT
add_executable(psfi_tests
    test_main.cpp
    test_impulse_response_field.cpp
    test_dimensions.cpp
    test_source_target.cpp
    test_moments.cpp
    test_low_rank.cpp
    test_partition.cpp
    test_block_low_rank.cpp
    test_rbf.cpp
    test_kernel_evaluator.cpp
    test_kernel_low_rank.cpp)
target_link_libraries(psfi_tests PRIVATE psfi::psfi)
target_include_directories(psfi_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty)
target_compile_options(psfi_tests PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall;-Wextra>)
add_test(NAME psfi_tests COMMAND psfi_tests)
