# SPDX-License-Identifier: MIT
set(ELLIPSOID_PSF_EXAMPLE_NAMES
    frog_compression
    frog_kernel)

foreach(example_name ${ELLIPSOID_PSF_EXAMPLE_NAMES})
    add_executable(example_${example_name} ${example_name}.cpp)
    target_link_libraries(example_${example_name} PRIVATE ellipsoid_psf::ellipsoid_psf)
    set_target_properties(example_${example_name} PROPERTIES OUTPUT_NAME ${example_name})
    target_compile_options(example_${example_name} PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall;-Wextra>)
endforeach()
