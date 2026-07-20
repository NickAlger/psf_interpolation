// SPDX-License-Identifier: MIT
#include "doctest/doctest.h"

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

Eigen::VectorXd bump_vertex_values( double cx, double cy, double w, const Eigen::MatrixXd& vertices )
{
    Eigen::VectorXd out(vertices.cols());
    for ( int v = 0; v < vertices.cols(); ++v )
    {
        const double dx = vertices(0, v) - cx;
        const double dy = vertices(1, v) - cy;
        out(v) = std::exp(-( dx * dx + dy * dy ) / ( w * w ));
    }
    return out;
}

const Eigen::VectorXd kNoV;
const Eigen::MatrixXd kNoMu;
const std::vector<Eigen::MatrixXd> kNoSigma;

// A smooth kernel: single-sample batches carrying Gaussian bumps, translation
// frame, no moments needed. The bump width varies per sample, so the kernel
// is genuinely x-dependent (not a pure displacement kernel), and
// num_neighbors = all samples keeps the neighbor set constant so the kernel
// is smooth in x (a changing kNN set introduces kinks and slows the
// singular-value decay to nothing at tight tolerances).
KernelEvaluator make_kernel()
{
    Eigen::MatrixXd vertices;
    Eigen::MatrixXi cells;
    make_grid_mesh(8, vertices, cells);
    auto F = std::make_shared<ImpulseResponseField>(vertices, cells, /*batches_normalized=*/false);
    const std::vector<Eigen::Vector2d> pts = {{0.30, 0.40}, {0.55, 0.50}, {0.45, 0.65},
                                              {0.62, 0.35}, {0.38, 0.58}};
    for ( int b = 0; b < 5; ++b )
    {
        Eigen::MatrixXd P(2, 1);
        P.col(0) = pts[b];
        F->add_batch(P, bump_vertex_values(pts[b](0), pts[b](1), 0.15 + 0.03 * b, vertices),
                     kNoV, kNoMu, kNoSigma);
    }
    EvalConfig cfg;
    cfg.frame         = Frame::translation;
    cfg.scaling       = Scaling::none;
    cfg.support       = Support::none;
    cfg.num_neighbors = 5;
    return KernelEvaluator(std::move(F), nullptr, cfg);
}

// Evaluation points: an m x m grid over [lo, hi]^2. The windows below are
// chosen so every transported point y - x + x_i stays inside the mesh:
// leaving it would EXCLUDE the sample (Support::none semantics), and those
// exclusion jumps make the kernel matrix nearly full rank at any tolerance.
// Even without jumps the CG1 mesh interpolation gives only algebraic
// singular decay, so compression is asserted at tolerances representative
// of actual use (the interpolated kernel itself carries ~1e-1..1e-2 error),
// while tight-tolerance cases check correctness only.
Eigen::MatrixXd window_grid( int m, double lo, double hi )
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


TEST_CASE("kernel_low_rank: dense_svd pathway matches block() to tolerance")
{
    const KernelEvaluator K = make_kernel();
    const Eigen::MatrixXd yy = window_grid(9, 0.35, 0.65); // 81 targets
    const Eigen::MatrixXd xx = window_grid(6, 0.40, 0.60); // 36 sources

    const Eigen::MatrixXd A = K.block(yy, xx);
    REQUIRE(A.norm() > 0.0);

    // Tight tolerance: correctness of the truncation rule and flags.
    const double rtol = 1e-6;
    KernelLowRankOptions options;
    options.method = CompressionMethod::dense_svd;
    const KernelLowRankResult r = kernel_low_rank(K, yy, xx, rtol, options);
    CHECK(!r.used_aca);
    CHECK(r.converged);
    CHECK(!r.hit_max_rank);
    CHECK(( A - r.factors.to_dense() ).norm() / A.norm() <= rtol * ( 1.0 + 1e-9 ));
    CHECK(r.relerr_estimate <= rtol * ( 1.0 + 1e-9 ));
    CHECK(r.factors.U.rows() == 81);
    CHECK(r.factors.V.rows() == 36);

    // Use-representative tolerance: genuine compression (measured rank ~16
    // of 36 at 1e-2; the cushion covers last-ulp toolchain differences).
    const KernelLowRankResult rc = kernel_low_rank(K, yy, xx, 1e-2, options);
    CHECK(rc.converged);
    CHECK(( A - rc.factors.to_dense() ).norm() / A.norm() <= 1e-2 * ( 1.0 + 1e-9 ));
    CHECK(rc.factors.rank() < 24);
}


TEST_CASE("kernel_low_rank: aca pathway agrees with the dense matrix")
{
    const KernelEvaluator K = make_kernel();
    const Eigen::MatrixXd yy = window_grid(9, 0.35, 0.65);
    const Eigen::MatrixXd xx = window_grid(6, 0.40, 0.60);
    const Eigen::MatrixXd A = K.block(yy, xx);

    const double rtol = 1e-6;
    KernelLowRankOptions options;
    options.method = CompressionMethod::aca;
    const KernelLowRankResult r = kernel_low_rank(K, yy, xx, rtol, options);
    CHECK(r.used_aca);
    CHECK(r.converged);
    // The ACA tolerance is a sampled estimate; allow an order of magnitude.
    CHECK(( A - r.factors.to_dense() ).norm() / A.norm() <= 10.0 * rtol);

    // Deterministic for a fixed seed.
    const KernelLowRankResult r2 = kernel_low_rank(K, yy, xx, rtol, options);
    CHECK(( r.factors.U - r2.factors.U ).cwiseAbs().maxCoeff() == 0.0);
    CHECK(( r.factors.V - r2.factors.V ).cwiseAbs().maxCoeff() == 0.0);
}


TEST_CASE("kernel_low_rank: automatic method selection")
{
    const KernelEvaluator K = make_kernel();
    const Eigen::MatrixXd yy = window_grid(7, 0.35, 0.65);
    const Eigen::MatrixXd xx = window_grid(5, 0.40, 0.60);

    // min(49, 25) = 25 <= default dense_min_dim (128): dense.
    CHECK(!kernel_low_rank(K, yy, xx, 1e-4).used_aca);

    KernelLowRankOptions force_aca;
    force_aca.dense_min_dim = 0; // min dim 25 > 0: aca
    CHECK(kernel_low_rank(K, yy, xx, 1e-4, force_aca).used_aca);
}


TEST_CASE("kernel_low_rank: a binding rank cap is reported on both pathways")
{
    const KernelEvaluator K = make_kernel();
    const Eigen::MatrixXd yy = window_grid(7, 0.35, 0.65);
    const Eigen::MatrixXd xx = window_grid(5, 0.40, 0.60);

    for ( CompressionMethod method : { CompressionMethod::dense_svd, CompressionMethod::aca } )
    {
        KernelLowRankOptions options;
        options.method = method;
        options.max_rank = 2;
        const KernelLowRankResult r = kernel_low_rank(K, yy, xx, 1e-12, options);
        CHECK(r.factors.rank() <= 2);
        CHECK(r.hit_max_rank);
        CHECK(!r.converged);
    }
}


TEST_CASE("kernel_low_rank: empty point sets and validation")
{
    const KernelEvaluator K = make_kernel();
    const Eigen::MatrixXd yy = window_grid(4, 0.40, 0.60);
    const Eigen::MatrixXd none(2, 0);

    const KernelLowRankResult r = kernel_low_rank(K, yy, none, 1e-6);
    CHECK(r.converged);
    CHECK(r.factors.rank() == 0);
    CHECK(r.factors.U.rows() == 16);
    CHECK(r.factors.V.rows() == 0);

    const Eigen::MatrixXd bad_dim(3, 4);
    CHECK_THROWS_AS(kernel_low_rank(K, bad_dim, yy, 1e-6), std::invalid_argument);
    CHECK_THROWS_AS(kernel_low_rank(K, yy, bad_dim, 1e-6), std::invalid_argument);
    CHECK_THROWS_AS(kernel_low_rank(K, yy, yy, -1.0), std::invalid_argument);
    KernelLowRankOptions bad_options;
    bad_options.dense_min_dim = -1;
    CHECK_THROWS_AS(kernel_low_rank(K, yy, yy, 1e-6, bad_options), std::invalid_argument);
}
