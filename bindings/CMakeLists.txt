# SPDX-License-Identifier: MIT
set(PYBIND11_FINDPYTHON ON)

# Inside a pip build, scikit-build-core provides pybind11 via find_package;
# otherwise fall back to a pinned download.
find_package(pybind11 2.12 CONFIG QUIET)
if(NOT pybind11_FOUND)
    include(FetchContent)
    FetchContent_Declare(pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.13.6
        GIT_SHALLOW TRUE)
    FetchContent_MakeAvailable(pybind11)
endif()

pybind11_add_module(ellipsoid_psf_python ellipsoid_psf_bindings.cpp)
target_link_libraries(ellipsoid_psf_python PRIVATE ellipsoid_psf::ellipsoid_psf)
set_target_properties(ellipsoid_psf_python PROPERTIES OUTPUT_NAME ellipsoid_psf)

if(SKBUILD)
    install(TARGETS ellipsoid_psf_python LIBRARY DESTINATION .)
endif()
