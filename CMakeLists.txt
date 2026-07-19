# SPDX-License-Identifier: MIT
cmake_minimum_required(VERSION 3.16)

# Version is single-sourced from the public header (see include/psfi/psfi.hpp).
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/include/psfi/psfi.hpp" _psfi_hpp)
string(REGEX MATCH "PSFI_VERSION_MAJOR ([0-9]+)" _ "${_psfi_hpp}")
set(_psfi_major "${CMAKE_MATCH_1}")
string(REGEX MATCH "PSFI_VERSION_MINOR ([0-9]+)" _ "${_psfi_hpp}")
set(_psfi_minor "${CMAKE_MATCH_1}")
string(REGEX MATCH "PSFI_VERSION_PATCH ([0-9]+)" _ "${_psfi_hpp}")
set(_psfi_patch "${CMAKE_MATCH_1}")
project(psfi VERSION ${_psfi_major}.${_psfi_minor}.${_psfi_patch} LANGUAGES CXX)

add_library(psfi INTERFACE)
add_library(psfi::psfi ALIAS psfi)
target_compile_features(psfi INTERFACE cxx_std_17)
target_include_directories(psfi INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

# etree (which brings Eigen and Threads): use an installed copy if available,
# otherwise download a pinned release. For local development against a
# checkout, configure with -DFETCHCONTENT_SOURCE_DIR_ETREE=/path/to/ellipsoid_tree.
set(_psfi_etree_vendored FALSE)
find_package(etree 0.1 QUIET CONFIG)
if(NOT TARGET etree::etree)
    set(_psfi_etree_vendored TRUE)
    option(PSFI_FETCH_ETREE "Download etree if no installed copy is found" ON)
    if(NOT PSFI_FETCH_ETREE)
        message(FATAL_ERROR
            "psfi requires etree. Install it (https://github.com/NickAlger/ellipsoid_tree) "
            "or enable PSFI_FETCH_ETREE.")
    endif()
    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
        cmake_policy(SET CMP0135 NEW)
    endif()
    include(FetchContent)
    FetchContent_Declare(etree
        URL https://github.com/NickAlger/ellipsoid_tree/archive/refs/tags/v0.1.0.tar.gz
        URL_HASH SHA256=938db0e4ade40281e2d0be39911927734f814d5ef333ab75be750f595514c411)
    FetchContent_MakeAvailable(etree)
    message(STATUS "psfi: using downloaded etree v0.1.0 (${etree_SOURCE_DIR})")
endif()
target_link_libraries(psfi INTERFACE etree::etree)

if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    option(PSFI_BUILD_TESTS "Build psfi tests" ON)
else()
    option(PSFI_BUILD_TESTS "Build psfi tests" OFF)
endif()

if(PSFI_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

option(PSFI_BUILD_PYTHON "Build the psfi Python bindings (pybind11 via FetchContent)" OFF)
if(PSFI_BUILD_PYTHON)
    add_subdirectory(bindings)
endif()

# ------------------------- Installation / CMake package export -------------------------
# Default ON when built standalone against an *installed* etree; OFF as a
# subproject, OFF inside pip builds (scikit-build-core would otherwise install
# the headers into the wheel), and OFF when etree came from the FetchContent
# fallback — the exported psfiConfig.cmake does find_dependency(etree), which
# only works if consumers can find an installed etree.
if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR AND NOT SKBUILD AND NOT _psfi_etree_vendored)
    set(PSFI_INSTALL_DEFAULT ON)
else()
    set(PSFI_INSTALL_DEFAULT OFF)
endif()
option(PSFI_INSTALL "Generate install and CMake package export rules" ${PSFI_INSTALL_DEFAULT})

if(PSFI_INSTALL AND _psfi_etree_vendored)
    message(FATAL_ERROR
        "PSFI_INSTALL requires an installed etree (found via find_package); the "
        "FetchContent fallback cannot be exported. Install etree first, then reconfigure.")
endif()

if(PSFI_INSTALL)
    include(GNUInstallDirs)
    include(CMakePackageConfigHelpers)

    install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
    install(TARGETS psfi EXPORT psfiTargets)
    install(EXPORT psfiTargets
        FILE psfiTargets.cmake
        NAMESPACE psfi::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/psfi)

    configure_package_config_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/psfiConfig.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/psfiConfig.cmake
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/psfi)
    write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/psfiConfigVersion.cmake
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion
        ARCH_INDEPENDENT)
    install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/psfiConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/psfiConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/psfi)
    install(FILES LICENSE DESTINATION ${CMAKE_INSTALL_DOCDIR})
endif()
