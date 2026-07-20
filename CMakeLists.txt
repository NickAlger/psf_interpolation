# SPDX-License-Identifier: MIT
cmake_minimum_required(VERSION 3.16)

# Version is single-sourced from the public header
# (see include/ellipsoid_psf/ellipsoid_psf.hpp).
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/include/ellipsoid_psf/ellipsoid_psf.hpp" _ellipsoid_psf_hpp)
string(REGEX MATCH "ELLIPSOID_PSF_VERSION_MAJOR ([0-9]+)" _ "${_ellipsoid_psf_hpp}")
set(_ellipsoid_psf_major "${CMAKE_MATCH_1}")
string(REGEX MATCH "ELLIPSOID_PSF_VERSION_MINOR ([0-9]+)" _ "${_ellipsoid_psf_hpp}")
set(_ellipsoid_psf_minor "${CMAKE_MATCH_1}")
string(REGEX MATCH "ELLIPSOID_PSF_VERSION_PATCH ([0-9]+)" _ "${_ellipsoid_psf_hpp}")
set(_ellipsoid_psf_patch "${CMAKE_MATCH_1}")
project(ellipsoid_psf
    VERSION ${_ellipsoid_psf_major}.${_ellipsoid_psf_minor}.${_ellipsoid_psf_patch}
    LANGUAGES CXX)

add_library(ellipsoid_psf INTERFACE)
add_library(ellipsoid_psf::ellipsoid_psf ALIAS ellipsoid_psf)
target_compile_features(ellipsoid_psf INTERFACE cxx_std_17)
target_include_directories(ellipsoid_psf INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

# ellipsoid_tree (which brings Eigen and Threads): use an installed copy if available,
# otherwise download a pinned release. For local development against a
# checkout, configure with -DFETCHCONTENT_SOURCE_DIR_ELLIPSOID_TREE=/path/to/ellipsoid_tree.
set(_ellipsoid_psf_ellipsoid_tree_vendored FALSE)
find_package(ellipsoid_tree 0.2 QUIET CONFIG)
if(NOT TARGET ellipsoid_tree::ellipsoid_tree)
    set(_ellipsoid_psf_ellipsoid_tree_vendored TRUE)
    option(ELLIPSOID_PSF_FETCH_ELLIPSOID_TREE "Download ellipsoid_tree if no installed copy is found" ON)
    if(NOT ELLIPSOID_PSF_FETCH_ELLIPSOID_TREE)
        message(FATAL_ERROR
            "ellipsoid_psf requires ellipsoid_tree. Install it (https://github.com/NickAlger/ellipsoid_tree) "
            "or enable ELLIPSOID_PSF_FETCH_ELLIPSOID_TREE.")
    endif()
    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
        cmake_policy(SET CMP0135 NEW)
    endif()
    include(FetchContent)
    # Single-sourced so the URL and the status message cannot drift apart.
    set(_ellipsoid_tree_pinned_version 0.2.0)
    FetchContent_Declare(ellipsoid_tree
        URL https://github.com/NickAlger/ellipsoid_tree/archive/refs/tags/v${_ellipsoid_tree_pinned_version}.tar.gz
        URL_HASH SHA256=24e66d161a85a9a1f98cb3838b0bb14edeba98d7c7d1dfb98678bdd61852ad66)
    FetchContent_MakeAvailable(ellipsoid_tree)
    message(STATUS "ellipsoid_psf: using downloaded ellipsoid_tree v${_ellipsoid_tree_pinned_version} (${ellipsoid_tree_SOURCE_DIR})")
endif()
target_link_libraries(ellipsoid_psf INTERFACE ellipsoid_tree::ellipsoid_tree)

if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    option(ELLIPSOID_PSF_BUILD_TESTS "Build ellipsoid_psf tests" ON)
    option(ELLIPSOID_PSF_BUILD_EXAMPLES "Build ellipsoid_psf examples" ON)
else()
    option(ELLIPSOID_PSF_BUILD_TESTS "Build ellipsoid_psf tests" OFF)
    option(ELLIPSOID_PSF_BUILD_EXAMPLES "Build ellipsoid_psf examples" OFF)
endif()

if(ELLIPSOID_PSF_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(ELLIPSOID_PSF_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

option(ELLIPSOID_PSF_BUILD_PYTHON
    "Build the ellipsoid_psf Python bindings (pybind11 via FetchContent)" OFF)
if(ELLIPSOID_PSF_BUILD_PYTHON)
    add_subdirectory(bindings)
endif()

# ------------------------- Installation / CMake package export -------------------------
# Default ON when built standalone against an *installed* ellipsoid_tree; OFF as a
# subproject, OFF inside pip builds (scikit-build-core would otherwise install
# the headers into the wheel), and OFF when ellipsoid_tree came from the FetchContent
# fallback — the exported ellipsoid_psfConfig.cmake does find_dependency(ellipsoid_tree),
# which only works if consumers can find an installed ellipsoid_tree.
if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR AND NOT SKBUILD AND NOT _ellipsoid_psf_ellipsoid_tree_vendored)
    set(ELLIPSOID_PSF_INSTALL_DEFAULT ON)
else()
    set(ELLIPSOID_PSF_INSTALL_DEFAULT OFF)
endif()
option(ELLIPSOID_PSF_INSTALL
    "Generate install and CMake package export rules" ${ELLIPSOID_PSF_INSTALL_DEFAULT})

if(ELLIPSOID_PSF_INSTALL AND _ellipsoid_psf_ellipsoid_tree_vendored)
    message(FATAL_ERROR
        "ELLIPSOID_PSF_INSTALL requires an installed ellipsoid_tree (found via find_package); the "
        "FetchContent fallback cannot be exported. Install ellipsoid_tree first, then reconfigure.")
endif()

if(ELLIPSOID_PSF_INSTALL)
    include(GNUInstallDirs)
    include(CMakePackageConfigHelpers)

    install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
    install(TARGETS ellipsoid_psf EXPORT ellipsoid_psfTargets)
    install(EXPORT ellipsoid_psfTargets
        FILE ellipsoid_psfTargets.cmake
        NAMESPACE ellipsoid_psf::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ellipsoid_psf)

    configure_package_config_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/ellipsoid_psfConfig.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/ellipsoid_psfConfig.cmake
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ellipsoid_psf)
    write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/ellipsoid_psfConfigVersion.cmake
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion
        ARCH_INDEPENDENT)
    install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/ellipsoid_psfConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/ellipsoid_psfConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ellipsoid_psf)
    install(FILES LICENSE DESTINATION ${CMAKE_INSTALL_DOCDIR})
endif()
