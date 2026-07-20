#pragma once
// SPDX-License-Identifier: MIT
// Part of ellipsoid_psf — https://github.com/NickAlger/ellipsoid_psf

/// @file
/// @brief BlockLowRank: the source-partitioned compressed kernel format
/// (BRLR), and its builder.
///
/// The kernel matrix's SOURCE axis is partitioned into disjoint index sets;
/// each block holds the kernel restricted to (its target set, its source
/// set), where the target set comes from the support oracles and every
/// entry outside it is exactly zero — so the block structure itself is
/// lossless w.r.t. the approximate kernel, and all error comes from the
/// per-block compression. Per block the storage is either a factor pair
/// (target_factor * source_factor^T) or the verbatim dense block, whichever
/// is smaller at the break-even rank r (|S| + |T|) < |S| |T|.
///
/// The operator/transpose dictionary, stated once, here. The impulse
/// response field samples columns of the kernel of whatever operator OP was
/// probed, at the source points. Therefore:
///
///     apply  ~ OP   (in the nodal kernel sense — the paper's mass wrapping
///                    A = M Phi M stays downstream, eq. 3.5),
///     applyT ~ OP^T.
///
/// If the batches came from forward applies of A: apply ~ A. If they came
/// from A^T: apply ~ A^T, and A itself is applyT. In symmetric mode the
/// matrix is approximately symmetric and consumers may form
/// (apply + applyT)/2. There is deliberately no "row"/"column" language in
/// this API — that vocabulary caused a silent transpose in this format's
/// ancestry.
///
/// The container is pure linear algebra: global ids + factors + axis sizes,
/// no geometry, no meshes, no evaluator reference. This matches the
/// distributed-format design (per-subdomain row/column index vectors):
/// distributing later means dealing blocks to ranks and wiring two
/// scatters, with no format change.

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "etree/detail/parallel_for.hpp"

#include "ellipsoid_psf/kernel_evaluator.hpp"
#include "ellipsoid_psf/kernel_low_rank.hpp"
#include "ellipsoid_psf/low_rank.hpp"
#include "ellipsoid_psf/partition.hpp"

namespace ellipsoid_psf {

/// The source-partitioned compressed kernel matrix. Sources not covered by
/// any block correspond to identically-zero kernel columns; targets not in
/// any block's set receive zero from apply.
class BlockLowRank
{
public:
    /// One block: the kernel restricted to (target_ids, source_ids).
    /// factored: block ~ target_factor (|T|, r) * source_factor^T (r, |S|).
    /// dense: block = target_factor (|T|, |S|) verbatim, source_factor empty.
    struct Block
    {
        std::vector<int> source_ids;   ///< disjoint across blocks; order fixes factor rows
        std::vector<int> target_ids;   ///< strictly increasing
        bool factored = true;
        Eigen::MatrixXd target_factor;
        Eigen::MatrixXd source_factor;

        /// Stored rank: factor columns when factored, min(|T|, |S|) as an
        /// upper bound when dense.
        int rank() const
        {
            return factored ? static_cast<int>(target_factor.cols())
                            : static_cast<int>(std::min(target_factor.rows(),
                                                        target_factor.cols()));
        }
        /// Stored matrix entries (the quantity the break-even rule minimizes).
        long long storage_entries() const
        {
            return static_cast<long long>(target_factor.size())
                 + static_cast<long long>(source_factor.size());
        }
    };

    /// An empty 0 x 0 matrix.
    BlockLowRank() = default;

    /// Validates everything the applies rely on: ids in range, source ids
    /// disjoint across blocks (and duplicate-free), target ids strictly
    /// increasing per block, factor shapes consistent with the storage mode.
    BlockLowRank( int num_sources, int num_targets, std::vector<Block> blocks )
        : num_sources_(num_sources), num_targets_(num_targets), blocks_(std::move(blocks))
    {
        if ( num_sources_ < 0 || num_targets_ < 0 )
        {
            throw std::invalid_argument("ellipsoid_psf::BlockLowRank: axis sizes must be >= 0");
        }
        std::vector<bool> source_seen(num_sources_, false);
        for ( size_t bb = 0; bb < blocks_.size(); ++bb )
        {
            const Block& blk = blocks_[bb];
            const std::string where = "ellipsoid_psf::BlockLowRank: block " + std::to_string(bb);
            for ( int jj : blk.source_ids )
            {
                if ( jj < 0 || jj >= num_sources_ )
                {
                    throw std::invalid_argument(where + ": source id " + std::to_string(jj)
                                                + " out of range");
                }
                if ( source_seen[jj] )
                {
                    throw std::invalid_argument(where + ": source id " + std::to_string(jj)
                                                + " appears twice (blocks must partition the "
                                                "source axis)");
                }
                source_seen[jj] = true;
            }
            for ( size_t tt = 0; tt < blk.target_ids.size(); ++tt )
            {
                if ( blk.target_ids[tt] < 0 || blk.target_ids[tt] >= num_targets_ )
                {
                    throw std::invalid_argument(where + ": target id "
                                                + std::to_string(blk.target_ids[tt])
                                                + " out of range");
                }
                if ( tt > 0 && blk.target_ids[tt - 1] >= blk.target_ids[tt] )
                {
                    throw std::invalid_argument(where + ": target ids must be strictly "
                                                "increasing");
                }
            }
            const int nt = static_cast<int>(blk.target_ids.size());
            const int ns = static_cast<int>(blk.source_ids.size());
            if ( blk.factored )
            {
                if ( blk.target_factor.rows() != nt || blk.source_factor.rows() != ns
                     || blk.target_factor.cols() != blk.source_factor.cols() )
                {
                    throw std::invalid_argument(where + ": factored storage needs "
                                                "target_factor (|T|, r) and source_factor "
                                                "(|S|, r)");
                }
            }
            else
            {
                if ( blk.target_factor.rows() != nt || blk.target_factor.cols() != ns
                     || blk.source_factor.size() != 0 )
                {
                    throw std::invalid_argument(where + ": dense storage needs target_factor "
                                                "(|T|, |S|) and an empty source_factor");
                }
            }
        }
    }

    int num_sources() const { return num_sources_; }
    int num_targets() const { return num_targets_; }
    int num_blocks() const  { return static_cast<int>(blocks_.size()); }
    const std::vector<Block>& blocks() const { return blocks_; }

    /// Total stored entries across blocks.
    long long storage_entries() const
    {
        long long total = 0;
        for ( const Block& blk : blocks_ )
        {
            total += blk.storage_entries();
        }
        return total;
    }

    /// Integrates against the source axis: U is (num_sources, k) — one
    /// column per input vector of source-point values — and the result is
    /// (num_targets, k). If the probed operator was OP, this approximates
    /// OP in the nodal kernel sense (see the file header dictionary).
    Eigen::MatrixXd apply( const Eigen::Ref<const Eigen::MatrixXd>& U ) const
    {
        if ( U.rows() != num_sources_ )
        {
            throw std::invalid_argument("ellipsoid_psf::BlockLowRank::apply: input must have num_sources "
                                        "rows");
        }
        Eigen::MatrixXd out = Eigen::MatrixXd::Zero(num_targets_, U.cols());
        for ( const Block& blk : blocks_ )
        {
            const int ns = static_cast<int>(blk.source_ids.size());
            const int nt = static_cast<int>(blk.target_ids.size());
            if ( ns == 0 || nt == 0 || ( blk.factored && blk.target_factor.cols() == 0 ) )
            {
                continue;
            }
            Eigen::MatrixXd Us(ns, U.cols());
            for ( int ss = 0; ss < ns; ++ss )
            {
                Us.row(ss) = U.row(blk.source_ids[ss]);
            }
            const Eigen::MatrixXd P = blk.factored
                ? ( blk.target_factor * ( blk.source_factor.transpose() * Us ) ).eval()
                : ( blk.target_factor * Us ).eval();
            for ( int tt = 0; tt < nt; ++tt )
            {
                out.row(blk.target_ids[tt]) += P.row(tt); // targets overlap: accumulate
            }
        }
        return out;
    }

    /// The transpose action: V is (num_targets, k), the result
    /// (num_sources, k) — approximately OP^T per the dictionary. Sources
    /// outside every block receive zero.
    Eigen::MatrixXd applyT( const Eigen::Ref<const Eigen::MatrixXd>& V ) const
    {
        if ( V.rows() != num_targets_ )
        {
            throw std::invalid_argument("ellipsoid_psf::BlockLowRank::applyT: input must have num_targets "
                                        "rows");
        }
        Eigen::MatrixXd out = Eigen::MatrixXd::Zero(num_sources_, V.cols());
        for ( const Block& blk : blocks_ )
        {
            const int ns = static_cast<int>(blk.source_ids.size());
            const int nt = static_cast<int>(blk.target_ids.size());
            if ( ns == 0 || nt == 0 || ( blk.factored && blk.target_factor.cols() == 0 ) )
            {
                continue;
            }
            Eigen::MatrixXd Vt(nt, V.cols());
            for ( int tt = 0; tt < nt; ++tt )
            {
                Vt.row(tt) = V.row(blk.target_ids[tt]);
            }
            const Eigen::MatrixXd Q = blk.factored
                ? ( blk.source_factor * ( blk.target_factor.transpose() * Vt ) ).eval()
                : ( blk.target_factor.transpose() * Vt ).eval();
            for ( int ss = 0; ss < ns; ++ss )
            {
                out.row(blk.source_ids[ss]) = Q.row(ss); // sources disjoint: assign
            }
        }
        return out;
    }

    /// The dense (num_targets, num_sources) matrix — for testing and small
    /// problems.
    Eigen::MatrixXd to_dense() const
    {
        Eigen::MatrixXd out = Eigen::MatrixXd::Zero(num_targets_, num_sources_);
        for ( const Block& blk : blocks_ )
        {
            const Eigen::MatrixXd D = blk.factored
                ? ( blk.target_factor * blk.source_factor.transpose() ).eval()
                : blk.target_factor;
            for ( size_t ss = 0; ss < blk.source_ids.size(); ++ss )
            {
                for ( size_t tt = 0; tt < blk.target_ids.size(); ++tt )
                {
                    out(blk.target_ids[tt], blk.source_ids[ss]) = D(tt, ss);
                }
            }
        }
        return out;
    }

private:
    int num_sources_ = 0;
    int num_targets_ = 0;
    std::vector<Block> blocks_;
};

/// Per-block construction diagnostics (indices parallel the partition).
struct BlockBuildInfo
{
    bool   used_aca = false;
    bool   converged = true;
    bool   hit_max_rank = false;
    double relerr_estimate = 0.0; ///< exact for dense-path blocks, ACA estimate otherwise
};

/// block_low_rank() result: the matrix plus diagnostics. As everywhere:
/// a binding rank cap is never silent — check any_hit_max_rank.
struct BlockLowRankBuildResult
{
    BlockLowRank matrix;
    std::vector<BlockBuildInfo> block_info;
    bool all_converged = true;
    bool any_hit_max_rank = false;
};

/// Builds the BRLR approximation of the kernel matrix over the given
/// points: target sets from the support oracles (lossless block sparsity),
/// then per-block compression per the options (dense-SVD or ACA, automatic
/// by dense_min_dim), then the smaller of factored and verbatim-dense
/// storage per block. When the dense path's storage rule picks dense, the
/// stored block is the EXACT kernel block (better than its truncation).
/// Blocks are compressed in parallel (options.num_threads; each block's
/// own evaluation is single-threaded), with per-block seeds
/// options.seed + block index, so results are deterministic regardless of
/// scheduling. rtol is the per-block relative Frobenius tolerance; since
/// blocks partition the source axis and the excluded entries are exactly
/// zero, the whole-matrix relative Frobenius error is also at most rtol.
inline BlockLowRankBuildResult
block_low_rank( const KernelEvaluator& kernel,
                const Eigen::Ref<const Eigen::MatrixXd>& yy, // (dim_target, num_targets)
                const Eigen::Ref<const Eigen::MatrixXd>& xx, // (dim_source, num_sources)
                const std::vector<std::vector<int>>& source_partition,
                double rtol,
                const KernelLowRankOptions& options = {} )
{
    if ( yy.rows() != kernel.dim_target() || xx.rows() != kernel.dim_source() )
    {
        throw std::invalid_argument("ellipsoid_psf::block_low_rank: yy must have dim_target rows and xx "
                                    "dim_source rows (points are columns)");
    }
    if ( !( rtol >= 0.0 ) )
    {
        throw std::invalid_argument("ellipsoid_psf::block_low_rank: rtol must be >= 0");
    }
    const int num_targets = static_cast<int>(yy.cols());
    const int num_sources = static_cast<int>(xx.cols());
    const int num_parts = static_cast<int>(source_partition.size());

    const std::vector<std::vector<int>> target_sets =
        block_target_sets(kernel, yy, xx, source_partition, options.num_threads);

    BlockLowRankBuildResult result;
    std::vector<BlockLowRank::Block> blocks(num_parts);
    result.block_info.resize(num_parts);

    etree::detail::parallel_for(0, num_parts,
        [&]( std::ptrdiff_t aa, std::ptrdiff_t bb )
        {
            for ( std::ptrdiff_t pp = aa; pp < bb; ++pp )
            {
                BlockLowRank::Block& blk = blocks[pp];
                BlockBuildInfo& info = result.block_info[pp];
                blk.source_ids = source_partition[pp];
                blk.target_ids = target_sets[pp];
                const int ns = static_cast<int>(blk.source_ids.size());
                const int nt = static_cast<int>(blk.target_ids.size());
                if ( ns == 0 || nt == 0 )
                {
                    blk.factored = true; // rank-0: an identically-zero block
                    blk.target_factor.resize(nt, 0);
                    blk.source_factor.resize(ns, 0);
                    continue;
                }

                Eigen::MatrixXd yy_blk(yy.rows(), nt);
                for ( int tt = 0; tt < nt; ++tt )
                {
                    yy_blk.col(tt) = yy.col(blk.target_ids[tt]);
                }
                Eigen::MatrixXd xx_blk(xx.rows(), ns);
                for ( int ss = 0; ss < ns; ++ss )
                {
                    xx_blk.col(ss) = xx.col(blk.source_ids[ss]);
                }

                const bool use_aca =
                    ( options.method == CompressionMethod::aca )
                    || ( options.method == CompressionMethod::automatic
                         && std::min(ns, nt) > options.dense_min_dim );
                const long long dense_entries = static_cast<long long>(ns) * nt;

                if ( use_aca )
                {
                    auto get_row = [&]( int ii ) -> Eigen::VectorXd
                    {
                        return kernel.block(yy_blk.col(ii), xx_blk, 1).row(0).transpose();
                    };
                    auto get_col = [&]( int jj ) -> Eigen::VectorXd
                    {
                        return kernel.block(yy_blk, xx_blk.col(jj), 1).col(0);
                    };
                    ACAOptions aca_options;
                    aca_options.max_rank = options.max_rank;
                    aca_options.aca_safety_factor = options.aca_safety_factor;
                    aca_options.recompress_safety_factor = options.recompress_safety_factor;
                    aca_options.required_consecutive_successes =
                        options.required_consecutive_successes;
                    aca_options.seed = options.seed + static_cast<unsigned int>(pp);
                    ACAResult res = aca(get_row, get_col, nt, ns, rtol, aca_options);
                    info.used_aca = true;
                    info.converged = res.converged;
                    info.hit_max_rank = res.hit_max_rank;
                    info.relerr_estimate = res.relerr_estimate;
                    const long long r = res.factors.rank();
                    if ( r * ( ns + nt ) < dense_entries )
                    {
                        blk.factored = true;
                        blk.target_factor = std::move(res.factors.U);
                        blk.source_factor = std::move(res.factors.V);
                    }
                    else
                    {
                        blk.factored = false;
                        blk.target_factor = res.factors.to_dense();
                        blk.source_factor.resize(0, 0);
                    }
                }
                else
                {
                    const Eigen::MatrixXd A = kernel.block(yy_blk, xx_blk, 1);
                    LowRank F = truncated_svd(A, rtol, options.max_rank);
                    const long long r = F.rank();
                    if ( r * ( ns + nt ) < dense_entries )
                    {
                        blk.factored = true;
                        const double norm_A = A.norm();
                        info.relerr_estimate =
                            ( norm_A > 0.0 ) ? ( A - F.to_dense() ).norm() / norm_A : 0.0;
                        info.converged =
                            ( info.relerr_estimate <= rtol * ( 1.0 + 1e-9 ) + 1e-14 );
                        info.hit_max_rank = !info.converged;
                        blk.target_factor = std::move(F.U);
                        blk.source_factor = std::move(F.V);
                    }
                    else
                    {
                        // Storing dense: keep the exact block, not its
                        // truncation.
                        blk.factored = false;
                        blk.target_factor = A;
                        blk.source_factor.resize(0, 0);
                        info.relerr_estimate = 0.0;
                        info.converged = true;
                        info.hit_max_rank = false;
                    }
                }
            }
        }, options.num_threads);

    for ( const BlockBuildInfo& info : result.block_info )
    {
        result.all_converged = result.all_converged && info.converged;
        result.any_hit_max_rank = result.any_hit_max_rank || info.hit_max_rank;
    }
    result.matrix = BlockLowRank(num_sources, num_targets, std::move(blocks));
    return result;
}

/// Global low rank from a block low rank matrix — the (e) -> (f) hop of the
/// pipeline: randomized SVD driven by the (cheap, BLAS-3) block applies
/// instead of kernel entry sampling. Returns factors with U rows = targets,
/// V rows = sources, like kernel_low_rank. At scale this beats global ACA
/// on the kernel: once the blocks are built, no further kernel evaluations
/// happen at all. Deterministic for a given options.seed.
inline LowRank randomized_svd( const BlockLowRank& B,
                               int max_rank,
                               const RSVDOptions& options = {} )
{
    auto apply = [&B]( const Eigen::Ref<const Eigen::MatrixXd>& X ) -> Eigen::MatrixXd
    {
        return B.apply(X);
    };
    auto apply_transpose = [&B]( const Eigen::Ref<const Eigen::MatrixXd>& Y ) -> Eigen::MatrixXd
    {
        return B.applyT(Y);
    };
    return randomized_svd(apply, apply_transpose, B.num_targets(), B.num_sources(),
                          max_rank, options);
}

} // end namespace ellipsoid_psf
