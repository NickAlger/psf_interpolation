#pragma once
// SPDX-License-Identifier: MIT

/// @file
/// @brief Umbrella header — includes the whole ellipsoid_psf public API.
///
/// ellipsoid_psf: point spread function interpolation with ellipsoidal
/// support. Evaluate an integral kernel Phi(y, x) anywhere by interpolating
/// translated (or affinely transported) impulse responses sampled at
/// scattered points, each supported within an ellipsoid. Header-only C++17;
/// depends on Eigen and ellipsoid_tree. https://github.com/NickAlger/ellipsoid_psf
///
/// The method comes from N. Alger, T. Hartland, N. Petra, O. Ghattas,
/// "Point spread function approximation of high-rank Hessians with locally
/// supported non-negative integral kernels", SISC 46(3), 2024, A1658-A1689.

// Single source of truth for the version. CMakeLists.txt parses these macros
// to set the project version, and a CI check keeps pyproject.toml in sync;
// ELLIPSOID_PSF_VERSION is the composed "MAJOR.MINOR.PATCH" string.
#define ELLIPSOID_PSF_VERSION_MAJOR 0
#define ELLIPSOID_PSF_VERSION_MINOR 1
#define ELLIPSOID_PSF_VERSION_PATCH 0
#define ELLIPSOID_PSF_STRINGIZE_IMPL(x) #x
#define ELLIPSOID_PSF_STRINGIZE(x)      ELLIPSOID_PSF_STRINGIZE_IMPL(x)
#define ELLIPSOID_PSF_VERSION                                \
    ELLIPSOID_PSF_STRINGIZE(ELLIPSOID_PSF_VERSION_MAJOR) "." \
    ELLIPSOID_PSF_STRINGIZE(ELLIPSOID_PSF_VERSION_MINOR) "." \
    ELLIPSOID_PSF_STRINGIZE(ELLIPSOID_PSF_VERSION_PATCH)

// Umbrella header. Public headers are included here as they land:
#include "ellipsoid_psf/config.hpp"                 // Frame, Scaling, Support, EvalConfig
#include "ellipsoid_psf/moments.hpp"                // covariance hygiene: SPD eigenvalue clamping
#include "ellipsoid_psf/low_rank.hpp"               // truncated SVD, ACA, randomized SVD, recompression
#include "ellipsoid_psf/impulse_response_field.hpp" // sampled impulse responses + per-neighbor predictions
#include "ellipsoid_psf/rbf.hpp"                    // RBF interpolation of scattered data
#include "ellipsoid_psf/kernel_evaluator.hpp"       // Phi(y, x) entries and threaded blocks
#include "ellipsoid_psf/kernel_low_rank.hpp"        // global low rank of the kernel matrix (dense-SVD / ACA)
#include "ellipsoid_psf/partition.hpp"              // source partitioning + per-block target sets
#include "ellipsoid_psf/block_low_rank.hpp"         // the BRLR format: container, builder, apply/applyT
