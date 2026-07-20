// SPDX-License-Identifier: MIT
#include "doctest/doctest.h"

#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

#include <Eigen/Dense>

#include "ellipsoid_psf/ellipsoid_psf.hpp"

using namespace ellipsoid_psf;

namespace {

// Structured triangulation of [0, 1]^2 (same as test_kernel_evaluator).
void make_grid_mesh( int n, Eigen::MatrixXd& vertices, Eigen::MatrixXi& cells )
{
    const int nv = ( n + 1 ) * ( n + 1 );
    vertices.resize(2, nv);
    for ( int jj = 0; jj <= n; ++jj )
    {
        for ( int ii = 0; ii <= n; ++ii )
        {
            const int v = jj * ( n + 1 ) + ii;
            vertices(0, v) = static_cast<double>(ii) / n;
            vertices(1, v) = static_cast<double>(jj) / n;
        }
    }
    cells.resize(3, 2 * n * n);
    int c = 0;
    for ( int jj = 0; jj < n; ++jj )
    {
        for ( int ii = 0; ii < n; ++ii )
        {
            const int v00 = jj * ( n + 1 ) + ii;
            const int v10 = v00 + 1;
            const int v01 = v00 + ( n + 1 );
            const int v11 = v01 + 1;
            cells.col(c++) = Eigen::Vector3i(v00, v10, v11);
            cells.col(c++) = Eigen::Vector3i(v00, v11, v01);
        }
    }
}

// A gated field with full moment data: six single-sample batches carrying
// Gaussian bumps; the Sigma field varies strongly with position so that
// "y in x's ellipsoid" and "x in y's ellipsoid" genuinely differ (that
// asymmetry is what the symmetric-mode support tests need).
std::shared_ptr<ImpulseResponseField> make_moment_field()
{
    Eigen::MatrixXd vertices;
    Eigen::MatrixXi cells;
    make_grid_mesh(8, vertices, cells);
    const int nv = static_cast<int>(vertices.cols());
    auto F = std::make_shared<ImpulseResponseField>(vertices, cells, /*batches_normalized=*/false);

    const Eigen::Vector2d offset(0.02, -0.01);
    const std::vector<Eigen::Vector2d> pts = {{0.25, 0.30}, {0.60, 0.35}, {0.40, 0.60},
                                              {0.70, 0.70}, {0.30, 0.75}, {0.55, 0.55}};
    for ( int b = 0; b < static_cast<int>(pts.size()); ++b )
    {
        Eigen::MatrixXd P(2, 1);
        P.col(0) = pts[b];
        const Eigen::Vector2d mu = pts[b] + offset;
        Eigen::VectorXd psi(nv);
        for ( int v = 0; v < nv; ++v )
        {
            psi(v) = std::exp(-( vertices.col(v) - mu ).squaredNorm() / ( 0.12 * 0.12 ));
        }
        const double sx = 0.06 + 0.01 * b;
        std::vector<Eigen::MatrixXd> Sigma = { Eigen::Vector2d(sx * sx, 0.05 * 0.05).asDiagonal() };
        F->add_batch(P, psi, Eigen::VectorXd::Constant(1, 1.0),
                     Eigen::MatrixXd(mu), Sigma);
    }

    Eigen::MatrixXd field_mu(2, nv);
    Eigen::MatrixXd field_Sigma(4, nv);
    for ( int v = 0; v < nv; ++v )
    {
        field_mu.col(v) = vertices.col(v) + offset;
        const double sx = 0.06 + 0.08 * vertices(0, v);
        const double sy = 0.05 + 0.04 * vertices(1, v);
        const Eigen::Matrix2d S = Eigen::Vector2d(sx * sx, sy * sy).asDiagonal();
        field_Sigma.col(v) = Eigen::Map<const Eigen::VectorXd>(S.data(), 4);
    }
    F->set_moment_fields(Eigen::VectorXd::Ones(nv), field_mu, field_Sigma);
    return F;
}

EvalConfig gated_config( Frame frame, int num_neighbors = 3, double tau = 2.0 )
{
    EvalConfig cfg;
    cfg.frame         = frame;
    cfg.scaling       = Scaling::none;
    cfg.support       = Support::ellipsoid;
    cfg.tau           = tau;
    cfg.num_neighbors = num_neighbors;
    return cfg;
}

bool inside_any( const Eigen::Vector2d& y, const std::vector<ellipsoid_tree::Ellipsoid>& ells )
{
    for ( const ellipsoid_tree::Ellipsoid& e : ells )
    {
        const Eigen::VectorXd d = y - e.mu;
        if ( d.dot(e.Sigma.llt().solve(d)) <= 1.0 )
        {
            return true;
        }
    }
    return false;
}

Eigen::MatrixXd square_grid( int m, double lo, double hi )
{
    Eigen::MatrixXd pts(2, m * m);
    for ( int jj = 0; jj < m; ++jj )
    {
        for ( int ii = 0; ii < m; ++ii )
        {
            pts(0, jj * m + ii) = lo + ( hi - lo ) * ii / ( m - 1 );
            pts(1, jj * m + ii) = lo + ( hi - lo ) * jj / ( m - 1 );
        }
    }
    return pts;
}

} // end anonymous namespace


TEST_CASE("recursive_bisection_partition: sizes, cover, determinism")
{
    // Points on a line: the median split is exactly the halves.
    Eigen::MatrixXd line(1, 100);
    for ( int ii = 0; ii < 100; ++ii )
    {
        line(0, ii) = ii;
    }
    const auto halves = recursive_bisection_partition(line, 50);
    REQUIRE(halves.size() == 2);
    for ( int ii = 0; ii < 50; ++ii )
    {
        CHECK(halves[0][ii] == ii);
        CHECK(halves[1][ii] == 50 + ii);
    }

    // 2D: sizes at most the cutoff, disjoint cover, deterministic.
    const Eigen::MatrixXd pts = square_grid(10, 0.0, 1.0); // 100 points
    const auto parts = recursive_bisection_partition(pts, 30);
    CHECK(parts.size() == 4); // 100 -> 50 -> 25 <= 30
    std::vector<int> seen(100, 0);
    for ( const auto& part : parts )
    {
        CHECK(static_cast<int>(part.size()) <= 30);
        for ( size_t ii = 1; ii < part.size(); ++ii )
        {
            CHECK(part[ii - 1] < part[ii]); // sorted ascending
        }
        for ( int ii : part )
        {
            seen[ii] += 1;
        }
    }
    for ( int count : seen )
    {
        CHECK(count == 1);
    }
    CHECK(recursive_bisection_partition(pts, 30) == parts); // deterministic

    // Degenerate cases.
    CHECK(recursive_bisection_partition(pts, 100).size() == 1);
    CHECK(recursive_bisection_partition(Eigen::MatrixXd(2, 0), 10).empty());
    CHECK_THROWS_AS(recursive_bisection_partition(pts, 0), std::invalid_argument);
}


TEST_CASE("support_ellipsoids: predictions vanish outside the returned union")
{
    auto F = make_moment_field();
    const Eigen::MatrixXd xq = square_grid(3, 0.25, 0.75);
    const Eigen::MatrixXd yq = square_grid(9, 0.0, 1.0);

    for ( Frame frame : { Frame::identity, Frame::translation,
                          Frame::mean_translation, Frame::whitened_affine } )
    {
        const EvalConfig cfg = gated_config(frame);
        int outside_pairs = 0;
        int violations = 0;
        for ( int qq = 0; qq < xq.cols(); ++qq )
        {
            const std::vector<ellipsoid_tree::Ellipsoid> ells = F->support_ellipsoids(xq.col(qq), cfg);
            CHECK(static_cast<int>(ells.size()) == ( frame == Frame::whitened_affine ? 1 : 3 ));
            for ( int tt = 0; tt < yq.cols(); ++tt )
            {
                if ( inside_any(yq.col(tt), ells) )
                {
                    continue;
                }
                ++outside_pairs;
                for ( const Prediction& p : F->predictions(yq.col(tt), xq.col(qq), cfg) )
                {
                    if ( p.value != 0.0 )
                    {
                        ++violations;
                    }
                }
            }
        }
        CHECK(violations == 0);
        CHECK(outside_pairs > 0); // the invariant was actually exercised
    }
}


TEST_CASE("support_ellipsoids: emptiness and validation")
{
    auto F = make_moment_field();
    const Eigen::Vector2d far_outside(2.0, 2.0);

    // Configurations needing field values at x: outside the mesh the column
    // is zero and the cover is empty.
    CHECK(F->support_ellipsoids(far_outside, gated_config(Frame::mean_translation)).empty());
    EvalConfig volume_cfg = gated_config(Frame::translation);
    volume_cfg.scaling = Scaling::volume;
    CHECK(F->support_ellipsoids(far_outside, volume_cfg).empty());

    // No field values needed: samples still predict at x outside the mesh.
    CHECK(F->support_ellipsoids(far_outside, gated_config(Frame::identity)).size() == 3);
    CHECK(F->support_ellipsoids(far_outside, gated_config(Frame::translation)).size() == 3);

    // No gate, no compact support.
    EvalConfig ungated = gated_config(Frame::translation);
    ungated.support = Support::none;
    CHECK_THROWS_AS(F->support_ellipsoids(Eigen::Vector2d(0.5, 0.5), ungated),
                    std::invalid_argument);

    CHECK_THROWS_AS(F->support_ellipsoids(Eigen::Vector3d(0.5, 0.5, 0.5),
                                          gated_config(Frame::identity)),
                    std::invalid_argument);
}


TEST_CASE("block_target_sets: excluded entries are exactly zero (cols-only)")
{
    auto F = make_moment_field();
    const EvalConfig cfg = gated_config(Frame::mean_translation);
    const KernelEvaluator K(F, nullptr, cfg);

    Eigen::MatrixXd vertices;
    Eigen::MatrixXi cells;
    make_grid_mesh(8, vertices, cells);
    const Eigen::MatrixXd& yy = vertices;              // 81 targets
    const Eigen::MatrixXd xx = square_grid(4, 0.15, 0.85); // 16 sources
    const auto partition = recursive_bisection_partition(xx, 4);
    REQUIRE(partition.size() == 4);

    const auto sets = block_target_sets(K, yy, xx, partition);
    REQUIRE(sets.size() == partition.size());

    const Eigen::MatrixXd B = K.block(yy, xx);
    for ( size_t pp = 0; pp < partition.size(); ++pp )
    {
        CHECK(!sets[pp].empty());
        std::vector<bool> in_set(yy.cols(), false);
        for ( int tt : sets[pp] )
        {
            in_set[tt] = true;
        }
        int violations = 0;
        int excluded = 0;
        for ( int tt = 0; tt < yy.cols(); ++tt )
        {
            if ( in_set[tt] )
            {
                continue;
            }
            ++excluded;
            for ( int jj : partition[pp] )
            {
                if ( B(tt, jj) != 0.0 )
                {
                    ++violations;
                }
            }
        }
        CHECK(violations == 0);
        CHECK(excluded > 0); // the sparsity is genuine, not all-targets
    }

    // Out-of-range partition indices are caught.
    CHECK_THROWS_AS(block_target_sets(K, yy, xx, { { 99 } }), std::invalid_argument);
}


TEST_CASE("block_target_sets: Support::none gives full-width blocks")
{
    Eigen::MatrixXd vertices;
    Eigen::MatrixXi cells;
    make_grid_mesh(8, vertices, cells);
    auto F = std::make_shared<ImpulseResponseField>(vertices, cells, /*batches_normalized=*/false);
    Eigen::MatrixXd P(2, 1);
    P.col(0) = Eigen::Vector2d(0.5, 0.5);
    F->add_batch(P, Eigen::VectorXd::Ones(vertices.cols()), Eigen::VectorXd(),
                 Eigen::MatrixXd(), {});
    EvalConfig cfg;
    cfg.frame   = Frame::translation;
    cfg.scaling = Scaling::none;
    cfg.support = Support::none;
    const KernelEvaluator K(F, nullptr, cfg);

    const Eigen::MatrixXd xx = square_grid(3, 0.3, 0.7);
    const auto sets = block_target_sets(K, vertices, xx, { { 0, 1, 2 }, { 3, 4 }, {} });
    CHECK(static_cast<int>(sets[0].size()) == vertices.cols());
    CHECK(static_cast<int>(sets[1].size()) == vertices.cols());
    CHECK(sets[2].empty()); // an empty part stays empty
}


TEST_CASE("block_target_sets: symmetric mode needs the adjoint piece")
{
    auto F = make_moment_field();
    const EvalConfig cfg = gated_config(Frame::whitened_affine);
    const KernelEvaluator K_sym(F, F, cfg);
    const KernelEvaluator K_cols(F, nullptr, cfg);

    Eigen::MatrixXd vertices;
    Eigen::MatrixXi cells;
    make_grid_mesh(8, vertices, cells);
    const Eigen::MatrixXd& yy = vertices;
    const Eigen::MatrixXd xx = square_grid(4, 0.15, 0.85);
    const auto partition = recursive_bisection_partition(xx, 4);

    const auto sets_sym  = block_target_sets(K_sym, yy, xx, partition);
    const auto sets_cols = block_target_sets(K_cols, yy, xx, partition);

    // The invariant holds for the symmetric kernel with the symmetric sets.
    const Eigen::MatrixXd B = K_sym.block(yy, xx);
    int violations = 0;
    for ( size_t pp = 0; pp < partition.size(); ++pp )
    {
        std::vector<bool> in_set(yy.cols(), false);
        for ( int tt : sets_sym[pp] )
        {
            in_set[tt] = true;
        }
        for ( int tt = 0; tt < yy.cols(); ++tt )
        {
            if ( in_set[tt] )
            {
                continue;
            }
            for ( int jj : partition[pp] )
            {
                if ( B(tt, jj) != 0.0 )
                {
                    ++violations;
                }
            }
        }
    }
    CHECK(violations == 0);

    // ... and the cols-only sets would NOT be enough: the row field makes
    // entries nonzero at targets the forward ellipsoids miss (the Sigma
    // field grows with position, so big ellipsoids at far targets contain
    // sources whose own small ellipsoids do not reach back).
    bool adjoint_needed = false;
    for ( size_t pp = 0; pp < partition.size() && !adjoint_needed; ++pp )
    {
        std::vector<bool> in_cols(yy.cols(), false);
        for ( int tt : sets_cols[pp] )
        {
            in_cols[tt] = true;
        }
        for ( int tt = 0; tt < yy.cols() && !adjoint_needed; ++tt )
        {
            if ( in_cols[tt] )
            {
                continue;
            }
            for ( int jj : partition[pp] )
            {
                if ( B(tt, jj) != 0.0 )
                {
                    adjoint_needed = true;
                    break;
                }
            }
        }
    }
    CHECK(adjoint_needed);

    // source_support is symmetric-mode-only.
    CHECK_THROWS_AS(K_cols.source_support(Eigen::Vector2d(0.5, 0.5)), std::logic_error);
    CHECK(!K_sym.source_support(Eigen::Vector2d(0.5, 0.5)).empty());
}
