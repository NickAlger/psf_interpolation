#pragma once
// SPDX-License-Identifier: MIT
// Part of ellipsoid_psf — https://github.com/NickAlger/ellipsoid_psf

/// @file
/// @brief Source-domain partitioning and per-block target sets — the
/// geometric layer of the block low rank format.
///
/// A partition is plain data: disjoint index sets over the source points.
/// recursive_bisection_partition is the shipped default constructor; any
/// other partitioner (graph-based, ellipsoid-width-balanced, ...) can hand
/// the same shape of data to the downstream builders. The cutoff is an
/// experimentation knob: production intuition is that a block should span a
/// few local support-ellipsoid widths — much smaller and the blocks are
/// merely sparse (rank ~ block size, nothing to compress), much larger and
/// the target sets balloon.
///
/// block_target_sets computes, for each part, the target points where
/// kernel entries from that part's sources can be nonzero — the
/// Omega_{T,i} of the block low rank format. It uses the support oracles
/// (KernelEvaluator::target_support, and source_support in symmetric mode),
/// so the sets are exact w.r.t. the approximate kernel: every entry outside
/// them is exactly zero, and block sparsity built on them is lossless.

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "etree/detail/parallel_for.hpp"
#include "etree/geometry.hpp"
#include "etree/object_tree.hpp"

#include "ellipsoid_psf/kernel_evaluator.hpp"

namespace ellipsoid_psf {

namespace detail {

inline void recursive_bisection_helper( int start,
                                        int stop,
                                        int depth,
                                        int max_part_size,
                                        const Eigen::Ref<const Eigen::MatrixXd>& points,
                                        std::vector<int>& inds,
                                        std::vector<std::vector<int>>& parts )
{
    const int count = stop - start;
    if ( count <= max_part_size )
    {
        std::vector<int> part(inds.begin() + start, inds.begin() + stop);
        std::sort(part.begin(), part.end()); // deterministic output order
        parts.push_back(std::move(part));
        return;
    }
    const int axis = depth % static_cast<int>(points.rows());
    const int mid = start + count / 2;
    // Median split via nth_element (O(count), vs a full sort per level).
    // The index tie-break makes the comparator a strict total order, so
    // part membership is identical across standard libraries.
    std::nth_element(inds.begin() + start, inds.begin() + mid, inds.begin() + stop,
                     [&]( int a, int b )
                     {
                         const double pa = points(axis, a);
                         const double pb = points(axis, b);
                         return ( pa < pb ) || ( pa == pb && a < b );
                     });
    recursive_bisection_helper(start, mid, depth + 1, max_part_size, points, inds, parts);
    recursive_bisection_helper(mid, stop, depth + 1, max_part_size, points, inds, parts);
}

} // end namespace detail

/// Partitions {0, ..., num_points - 1} into spatially coherent parts of at
/// most max_part_size points each, by recursive median bisection along
/// cycling coordinate axes (points are columns). Sibling parts differ in
/// size by at most one; each part's indices are sorted ascending; the
/// result is deterministic across platforms. Zero points give zero parts.
inline std::vector<std::vector<int>>
recursive_bisection_partition( const Eigen::Ref<const Eigen::MatrixXd>& points,
                               int max_part_size )
{
    if ( max_part_size < 1 )
    {
        throw std::invalid_argument("ellipsoid_psf::recursive_bisection_partition: max_part_size must be "
                                    ">= 1");
    }
    const int n = static_cast<int>(points.cols());
    if ( n == 0 )
    {
        return {};
    }
    if ( points.rows() < 1 )
    {
        throw std::invalid_argument("ellipsoid_psf::recursive_bisection_partition: points must have at "
                                    "least one row (the spatial dimension)");
    }
    std::vector<int> inds(n);
    for ( int ii = 0; ii < n; ++ii )
    {
        inds[ii] = ii;
    }
    std::vector<std::vector<int>> parts;
    detail::recursive_bisection_helper(0, n, 0, max_part_size, points, inds, parts);
    return parts;
}

/// For each part of a source partition, the indices of the target points
/// (columns of yy) where kernel entries against that part's sources can be
/// nonzero — every entry at an excluded target is exactly zero, by the
/// support oracles. Cols-only mode uses target_support alone; symmetric
/// mode adds the adjoint piece (targets whose source_support contains a
/// part source). With Support::none there is no compact support and every
/// part gets all targets. Each returned set is sorted ascending.
///
/// source_partition must hold disjoint indices into the columns of xx
/// (indices out of range throw; disjointness is the caller's contract).
/// Sources outside every part are simply not queried.
inline std::vector<std::vector<int>>
block_target_sets( const KernelEvaluator& kernel,
                   const Eigen::Ref<const Eigen::MatrixXd>& yy, // (dim_target, num_targets)
                   const Eigen::Ref<const Eigen::MatrixXd>& xx, // (dim_source, num_sources)
                   const std::vector<std::vector<int>>& source_partition,
                   int num_threads = 0 )
{
    if ( yy.rows() != kernel.dim_target() || xx.rows() != kernel.dim_source() )
    {
        throw std::invalid_argument("ellipsoid_psf::block_target_sets: yy must have dim_target rows and xx "
                                    "dim_source rows (points are columns)");
    }
    const int num_targets = static_cast<int>(yy.cols());
    const int num_sources = static_cast<int>(xx.cols());
    const int num_parts = static_cast<int>(source_partition.size());

    std::vector<int> part_of_source(num_sources, -1);
    for ( int pp = 0; pp < num_parts; ++pp )
    {
        for ( int jj : source_partition[pp] )
        {
            if ( jj < 0 || jj >= num_sources )
            {
                throw std::invalid_argument("ellipsoid_psf::block_target_sets: partition index "
                                            + std::to_string(jj) + " is out of range");
            }
            part_of_source[jj] = pp;
        }
    }

    std::vector<std::vector<int>> out(num_parts);
    if ( num_targets == 0 || num_parts == 0 )
    {
        return out;
    }

    if ( kernel.config().support == Support::none )
    {
        // No compact support: every part is full-width.
        std::vector<int> all(num_targets);
        for ( int tt = 0; tt < num_targets; ++tt )
        {
            all[tt] = tt;
        }
        for ( int pp = 0; pp < num_parts; ++pp )
        {
            if ( !source_partition[pp].empty() )
            {
                out[pp] = all;
            }
        }
        return out;
    }

    // Forward piece: each queried source's support ellipsoids (in target
    // space), tagged with its part; one tree, one batched point query.
    {
        std::vector<std::vector<etree::Ellipsoid>> per_source(num_sources);
        etree::detail::parallel_for(0, num_sources,
            [&]( std::ptrdiff_t aa, std::ptrdiff_t bb )
            {
                for ( std::ptrdiff_t jj = aa; jj < bb; ++jj )
                {
                    if ( part_of_source[jj] >= 0 )
                    {
                        per_source[jj] = kernel.target_support(xx.col(jj));
                    }
                }
            }, num_threads);

        std::vector<etree::Ellipsoid> ellipsoids;
        std::vector<int> owner_part;
        for ( int jj = 0; jj < num_sources; ++jj )
        {
            for ( etree::Ellipsoid& e : per_source[jj] )
            {
                ellipsoids.push_back(std::move(e));
                owner_part.push_back(part_of_source[jj]);
            }
        }
        if ( !ellipsoids.empty() )
        {
            const etree::EllipsoidTree tree(std::move(ellipsoids), 1.0, num_threads);
            const std::vector<std::vector<int>> hits =
                tree.point_collisions_batch(yy, num_threads);
            for ( int tt = 0; tt < num_targets; ++tt )
            {
                for ( int ee : hits[tt] )
                {
                    out[owner_part[ee]].push_back(tt);
                }
            }
        }
    }

    // Adjoint piece (symmetric mode): each target's source_support
    // ellipsoids (in source space), tagged with the target; sources landing
    // inside pull that target into their part's set.
    if ( kernel.symmetric() )
    {
        std::vector<std::vector<etree::Ellipsoid>> per_target(num_targets);
        etree::detail::parallel_for(0, num_targets,
            [&]( std::ptrdiff_t aa, std::ptrdiff_t bb )
            {
                for ( std::ptrdiff_t tt = aa; tt < bb; ++tt )
                {
                    per_target[tt] = kernel.source_support(yy.col(tt));
                }
            }, num_threads);

        std::vector<etree::Ellipsoid> ellipsoids;
        std::vector<int> owner_target;
        for ( int tt = 0; tt < num_targets; ++tt )
        {
            for ( etree::Ellipsoid& e : per_target[tt] )
            {
                ellipsoids.push_back(std::move(e));
                owner_target.push_back(tt);
            }
        }
        if ( !ellipsoids.empty() )
        {
            const etree::EllipsoidTree tree(std::move(ellipsoids), 1.0, num_threads);
            const std::vector<std::vector<int>> hits =
                tree.point_collisions_batch(xx, num_threads);
            for ( int jj = 0; jj < num_sources; ++jj )
            {
                if ( part_of_source[jj] < 0 )
                {
                    continue;
                }
                for ( int ee : hits[jj] )
                {
                    out[part_of_source[jj]].push_back(owner_target[ee]);
                }
            }
        }
    }

    for ( std::vector<int>& set : out )
    {
        std::sort(set.begin(), set.end());
        set.erase(std::unique(set.begin(), set.end()), set.end());
    }
    return out;
}

} // end namespace ellipsoid_psf
