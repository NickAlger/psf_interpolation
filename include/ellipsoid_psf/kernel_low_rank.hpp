#pragma once
// SPDX-License-Identifier: MIT
// Part of ellipsoid_psf — https://github.com/NickAlger/ellipsoid_psf

/// @file
/// @brief Global low-rank approximation of the kernel matrix
/// [ Phi(yy.col(ii), xx.col(jj)) ]_{ii,jj} over given target/source points.
///
/// This is THE adapter between the kernel language (source/target) and the
/// generic matrix layer (low_rank.hpp): the fixed mapping, stated here once
/// and used everywhere, is
///
///     target axis = matrix rows,  source axis = matrix columns,
///
/// matching KernelEvaluator::block(yy, xx), whose result has shape
/// (num_targets, num_sources). In the returned factors, U rows correspond to
/// the target points yy and V rows to the source points xx.
///
/// Two pathways (selected by CompressionMethod):
/// - dense_svd: assemble the dense kernel matrix, Frobenius-truncate its SVD.
///   Exact error reporting; cost num_targets * num_sources entry
///   evaluations — the small-problem / reference path.
/// - aca: adaptive cross approximation; touches only O(rank * (num_targets
///   + num_sources)) entries — the production path at scale. Sampled matrix
///   rows are kernel evaluations at one target against all sources; sampled
///   columns are one source against all targets (a kernel "impulse
///   response" slice, the cheap direction once the column-major evaluation
///   lands).
///
/// The operator/transpose dictionary lives in the block-low-rank header;
/// here the result is just a factored kernel matrix.

#include <algorithm>
#include <stdexcept>

#include <Eigen/Dense>

#include "ellipsoid_psf/kernel_evaluator.hpp"
#include "ellipsoid_psf/low_rank.hpp"

namespace ellipsoid_psf {

/// How to compress a kernel matrix (globally, or one block of a partition).
enum class CompressionMethod
{
    automatic, ///< dense_svd when min(num_targets, num_sources) <= dense_min_dim, else aca
    dense_svd, ///< assemble dense, truncated SVD (exact, superlinear cost)
    aca,       ///< cross approximation (samples O(rank * (num_targets + num_sources)) entries)
};

/// Options for kernel_low_rank(). The ACA knobs mirror ACAOptions (with
/// recompression always on); rtol is passed separately alongside the points.
struct KernelLowRankOptions
{
    CompressionMethod method = CompressionMethod::automatic;
    int max_rank = -1;      ///< rank cap for both pathways; < 0 means no cap
    int dense_min_dim = 128;///< automatic-rule threshold on min(num_targets, num_sources)
    int num_threads = 0;    ///< threads for dense assembly and ACA slice evaluation (<= 0: all cores)
    double aca_safety_factor = 0.25;
    double recompress_safety_factor = 0.75;
    int required_consecutive_successes = 10;
    unsigned int seed = 0;
};

/// Result of kernel_low_rank(): factors plus diagnostics. As with ACAResult,
/// check hit_max_rank — a binding rank cap is never silent.
struct KernelLowRankResult
{
    LowRank factors;              ///< U rows = target points, V rows = source points
    bool used_aca = false;        ///< which pathway ran
    bool converged = true;        ///< tolerance met (dense_svd: exact; aca: sampled estimate)
    bool hit_max_rank = false;    ///< max_rank bound before the tolerance was met
    double relerr_estimate = 0.0; ///< relative Frobenius error: exact for dense_svd, estimate for aca
};

/// Low-rank approximation of the kernel matrix over the given points:
/// factors with U (num_targets, r), V (num_sources, r) and
/// U V^T ~ [ Phi(yy.col(ii), xx.col(jj)) ], to relative Frobenius tolerance
/// rtol. yy is (dim_target, num_targets), xx (dim_source, num_sources) —
/// arbitrary coordinates, typically dof locations. Deterministic for a given
/// options.seed. Thread-safe (const evaluator; parallelism internal).
inline KernelLowRankResult kernel_low_rank( const KernelEvaluator& kernel,
                                            const Eigen::Ref<const Eigen::MatrixXd>& yy,
                                            const Eigen::Ref<const Eigen::MatrixXd>& xx,
                                            double rtol,
                                            const KernelLowRankOptions& options = {} )
{
    if ( yy.rows() != kernel.dim_target() || xx.rows() != kernel.dim_source() )
    {
        throw std::invalid_argument("ellipsoid_psf::kernel_low_rank: yy must have dim_target rows and xx "
                                    "dim_source rows (points are columns)");
    }
    if ( !( rtol >= 0.0 ) )
    {
        throw std::invalid_argument("ellipsoid_psf::kernel_low_rank: rtol must be >= 0");
    }
    if ( options.dense_min_dim < 0 )
    {
        throw std::invalid_argument("ellipsoid_psf::kernel_low_rank: dense_min_dim must be >= 0");
    }
    const int num_targets = static_cast<int>(yy.cols());
    const int num_sources = static_cast<int>(xx.cols());

    KernelLowRankResult result;
    result.factors = LowRank{ Eigen::MatrixXd(num_targets, 0), Eigen::MatrixXd(num_sources, 0) };
    if ( num_targets == 0 || num_sources == 0 )
    {
        return result;
    }

    const bool use_aca =
        ( options.method == CompressionMethod::aca )
        || ( options.method == CompressionMethod::automatic
             && std::min(num_targets, num_sources) > options.dense_min_dim );

    if ( use_aca )
    {
        auto get_row = [&]( int ii ) -> Eigen::VectorXd
        {
            return kernel.block(yy.col(ii), xx, options.num_threads).row(0).transpose();
        };
        auto get_col = [&]( int jj ) -> Eigen::VectorXd
        {
            return kernel.block(yy, xx.col(jj), options.num_threads).col(0);
        };
        ACAOptions aca_options;
        aca_options.max_rank = options.max_rank;
        aca_options.aca_safety_factor = options.aca_safety_factor;
        aca_options.recompress_safety_factor = options.recompress_safety_factor;
        aca_options.required_consecutive_successes = options.required_consecutive_successes;
        aca_options.seed = options.seed;
        ACAResult aca_result = aca(get_row, get_col, num_targets, num_sources, rtol, aca_options);
        result.factors = std::move(aca_result.factors);
        result.used_aca = true;
        result.converged = aca_result.converged;
        result.hit_max_rank = aca_result.hit_max_rank;
        result.relerr_estimate = aca_result.relerr_estimate;
    }
    else
    {
        const Eigen::MatrixXd A = kernel.block(yy, xx, options.num_threads);
        result.factors = truncated_svd(A, rtol, options.max_rank);
        const double norm_A = A.norm();
        result.relerr_estimate =
            ( norm_A > 0.0 ) ? ( A - result.factors.to_dense() ).norm() / norm_A : 0.0;
        // The truncation rule guarantees the tolerance unless max_rank cut
        // it short; the epsilon absorbs recomputation rounding at the
        // boundary.
        result.converged = ( result.relerr_estimate <= rtol * ( 1.0 + 1e-9 ) + 1e-14 );
        result.hit_max_rank = !result.converged;
    }
    return result;
}

} // end namespace ellipsoid_psf
