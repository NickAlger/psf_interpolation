#pragma once
// SPDX-License-Identifier: MIT

/// @file
/// @brief Umbrella header — includes the whole psfi public API.
///
/// psfi: point spread function interpolation. Evaluate an integral kernel
/// Phi(y, x) anywhere by interpolating translated (or affinely transported)
/// impulse responses sampled at scattered points. Header-only C++17; depends
/// on Eigen and etree. https://github.com/NickAlger/psf_interpolation
///
/// The method comes from N. Alger, T. Hartland, N. Petra, O. Ghattas,
/// "Point spread function approximation of high-rank Hessians with locally
/// supported non-negative integral kernels", SISC 46(3), 2024, A1658-A1689.

// Single source of truth for the version. CMakeLists.txt parses these macros
// to set the project version, and a CI check keeps pyproject.toml in sync;
// PSFI_VERSION is the composed "MAJOR.MINOR.PATCH" string.
#define PSFI_VERSION_MAJOR 0
#define PSFI_VERSION_MINOR 1
#define PSFI_VERSION_PATCH 0
#define PSFI_STRINGIZE_IMPL(x) #x
#define PSFI_STRINGIZE(x)      PSFI_STRINGIZE_IMPL(x)
#define PSFI_VERSION                        \
    PSFI_STRINGIZE(PSFI_VERSION_MAJOR) "." \
    PSFI_STRINGIZE(PSFI_VERSION_MINOR) "." \
    PSFI_STRINGIZE(PSFI_VERSION_PATCH)

// Umbrella header. Public headers are included here as they land:
#include "psfi/config.hpp"                 // Frame, Scaling, Support, EvalConfig
#include "psfi/impulse_response_field.hpp" // sampled impulse responses + per-neighbor predictions
