// SPDX-License-Identifier: MIT
#include "doctest/doctest.h"

#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

#include <Eigen/Dense>

#include "psfi/psfi.hpp"

using namespace psfi;

namespace {

// Structured triangulation of [0, 1]^2 (same as test_partition).
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

void make_interval_mesh( int n, Eigen::MatrixXd& vertices, Eigen::MatrixXi& cells )
{
    vertices.resize(1, n + 1);
    for ( int ii = 0; ii <= n; ++ii )
    {
        vertices(0, ii) = static_cast<double>(ii) / n;
    }
    cells.resize(2, n);
    for ( int ii = 0; ii < n; ++ii )
    {
        cells.col(ii) = Eigen::Vector2i(ii, ii + 1);
    }
}

// Gated field with full moment data (same construction as test_partition).
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

// Deterministic filler in [-1, 1] (mt19937 raw output).
Eigen::MatrixXd pseudo_random( int m, int n, unsigned int seed )
{
    std::mt19937 gen(seed);
    Eigen::MatrixXd A(m, n);
    for ( int jj = 0; jj < n; ++jj )
    {
        for ( int ii = 0; ii < m; ++ii )
        {
            A(ii, jj) = 2.0 * ( static_cast<double>(gen()) / 4294967296.0 ) - 1.0;
        }
    }
    return A;
}

} // end anonymous namespace


TEST_CASE("block_low_rank: rtol = 0 reproduces the kernel matrix exactly")
{
    auto F = make_moment_field();
    const KernelEvaluator K(F, nullptr, gated_config(Frame::whitened_affine));
    Eigen::MatrixXd vertices;
    Eigen::MatrixXi cells;
    make_grid_mesh(8, vertices, cells);
    const Eigen::MatrixXd& yy = vertices;
    const Eigen::MatrixXd xx = square_grid(4, 0.15, 0.85);
    const auto partition = recursive_bisection_partition(xx, 4);

    const BlockLowRankBuildResult r = block_low_rank(K, yy, xx, partition, 0.0);
    CHECK(r.all_converged);
    CHECK(!r.any_hit_max_rank);
    CHECK(r.matrix.num_sources() == 16);
    CHECK(r.matrix.num_targets() == 81);
    CHECK(r.matrix.num_blocks() == static_cast<int>(partition.size()));

    // At rtol = 0 every nonempty block is full rank, the break-even rule
    // picks verbatim-dense storage, and the stored block is the exact
    // kernel block — so the format reproduces block() bit for bit
    // (excluded entries are exactly zero on both sides).
    const Eigen::MatrixXd A = K.block(yy, xx);
    CHECK(( r.matrix.to_dense() - A ).cwiseAbs().maxCoeff() == 0.0);
    for ( const BlockLowRank::Block& blk : r.matrix.blocks() )
    {
        if ( !blk.source_ids.empty() && !blk.target_ids.empty() )
        {
            CHECK(!blk.factored);
        }
    }
}


TEST_CASE("block_low_rank: per-block tolerance bounds the global error")
{
    auto F = make_moment_field();
    const KernelEvaluator K(F, nullptr, gated_config(Frame::whitened_affine));
    Eigen::MatrixXd vertices;
    Eigen::MatrixXi cells;
    make_grid_mesh(8, vertices, cells);
    const Eigen::MatrixXd& yy = vertices;
    const Eigen::MatrixXd xx = square_grid(4, 0.15, 0.85);
    const auto partition = recursive_bisection_partition(xx, 4);
    const Eigen::MatrixXd A = K.block(yy, xx);

    const double rtol = 1e-2;
    const BlockLowRankBuildResult r = block_low_rank(K, yy, xx, partition, rtol);
    CHECK(r.all_converged);
    // Blocks partition the source axis and excluded entries are exactly
    // zero, so per-block relative Frobenius errors compose into the same
    // global bound.
    CHECK(( A - r.matrix.to_dense() ).norm() / A.norm() <= rtol * ( 1.0 + 1e-9 ));

    // The break-even invariant: factored storage only where it is smaller.
    for ( const BlockLowRank::Block& blk : r.matrix.blocks() )
    {
        const long long ns = static_cast<long long>(blk.source_ids.size());
        const long long nt = static_cast<long long>(blk.target_ids.size());
        if ( blk.factored && ns > 0 && nt > 0 )
        {
            CHECK(blk.rank() * ( ns + nt ) < ns * nt);
        }
        CHECK(blk.storage_entries() <= ns * nt);
    }
}


TEST_CASE("block_low_rank: apply and applyT match the dense matrix and each other")
{
    auto F = make_moment_field();
    const KernelEvaluator K(F, nullptr, gated_config(Frame::mean_translation));
    Eigen::MatrixXd vertices;
    Eigen::MatrixXi cells;
    make_grid_mesh(8, vertices, cells);
    const Eigen::MatrixXd& yy = vertices;
    const Eigen::MatrixXd xx = square_grid(4, 0.15, 0.85);
    const auto partition = recursive_bisection_partition(xx, 4);

    const BlockLowRank B = block_low_rank(K, yy, xx, partition, 1e-8).matrix;
    const Eigen::MatrixXd D = B.to_dense();

    const Eigen::MatrixXd U = pseudo_random(16, 3, 1);
    const Eigen::MatrixXd V = pseudo_random(81, 2, 2);
    const Eigen::MatrixXd BU = B.apply(U);
    const Eigen::MatrixXd BtV = B.applyT(V);
    CHECK(BU.rows() == 81);
    CHECK(BU.cols() == 3);
    CHECK(BtV.rows() == 16);
    CHECK(BtV.cols() == 2);
    CHECK(( BU - D * U ).cwiseAbs().maxCoeff() <= 1e-13);
    CHECK(( BtV - D.transpose() * V ).cwiseAbs().maxCoeff() <= 1e-13);

    // Exact adjoint pair: <B u, v> = <u, B^T v> to rounding.
    const double lhs = ( BU.col(0).dot(V.col(0)) );
    const double rhs = ( U.col(0).dot(B.applyT(V.col(0)).col(0)) );
    CHECK(lhs == doctest::Approx(rhs).epsilon(1e-13));

    CHECK_THROWS_AS(B.apply(pseudo_random(15, 1, 3)), std::invalid_argument);
    CHECK_THROWS_AS(B.applyT(pseudo_random(80, 1, 4)), std::invalid_argument);
}


TEST_CASE("block_low_rank: ACA pathway agrees and is deterministic")
{
    auto F = make_moment_field();
    const KernelEvaluator K(F, nullptr, gated_config(Frame::whitened_affine));
    Eigen::MatrixXd vertices;
    Eigen::MatrixXi cells;
    make_grid_mesh(8, vertices, cells);
    const Eigen::MatrixXd& yy = vertices;
    const Eigen::MatrixXd xx = square_grid(4, 0.15, 0.85);
    const auto partition = recursive_bisection_partition(xx, 4);
    const Eigen::MatrixXd A = K.block(yy, xx);

    const double rtol = 1e-4;
    KernelLowRankOptions options;
    options.method = CompressionMethod::aca;
    const BlockLowRankBuildResult r = block_low_rank(K, yy, xx, partition, rtol, options);
    CHECK(r.all_converged);
    for ( size_t pp = 0; pp < partition.size(); ++pp )
    {
        CHECK(r.block_info[pp].used_aca);
    }
    CHECK(( A - r.matrix.to_dense() ).norm() / A.norm() <= 10.0 * rtol);

    const BlockLowRankBuildResult r2 = block_low_rank(K, yy, xx, partition, rtol, options);
    CHECK(( r.matrix.to_dense() - r2.matrix.to_dense() ).cwiseAbs().maxCoeff() == 0.0);
}


TEST_CASE("block_low_rank: rank caps are reported, empty parts and zero columns are fine")
{
    auto F = make_moment_field();
    const KernelEvaluator K(F, nullptr, gated_config(Frame::mean_translation));
    Eigen::MatrixXd vertices;
    Eigen::MatrixXi cells;
    make_grid_mesh(8, vertices, cells);
    const Eigen::MatrixXd& yy = vertices;

    // Sources: one interior part, one part far outside the source mesh
    // (moment fields are needed there, so those columns are identically
    // zero), and one empty part.
    Eigen::MatrixXd xx(2, 6);
    xx.leftCols(4) = square_grid(2, 0.3, 0.6);
    xx.col(4) = Eigen::Vector2d(5.0, 5.0);
    xx.col(5) = Eigen::Vector2d(6.0, 6.0);
    const std::vector<std::vector<int>> partition = { { 0, 1, 2, 3 }, { 4, 5 }, {} };

    const BlockLowRankBuildResult r = block_low_rank(K, yy, xx, partition, 1e-6);
    CHECK(r.all_converged);
    const Eigen::MatrixXd D = r.matrix.to_dense();
    CHECK(D.col(4).cwiseAbs().maxCoeff() == 0.0);
    CHECK(D.col(5).cwiseAbs().maxCoeff() == 0.0);
    CHECK(( D - K.block(yy, xx) ).norm() <= 1e-6 * K.block(yy, xx).norm());

    // A binding rank cap is reported.
    KernelLowRankOptions capped;
    capped.max_rank = 1;
    const BlockLowRankBuildResult rc =
        block_low_rank(K, yy, xx, partition, 1e-12, capped);
    CHECK(rc.any_hit_max_rank);
    CHECK(!rc.all_converged);
}


TEST_CASE("block_low_rank: 1D source domain, 2D target domain")
{
    Eigen::MatrixXd tgt_vertices, src_vertices;
    Eigen::MatrixXi tgt_cells, src_cells;
    make_grid_mesh(8, tgt_vertices, tgt_cells);
    make_interval_mesh(8, src_vertices, src_cells);
    const int nv_tgt = static_cast<int>(tgt_vertices.cols());
    const int nv_src = static_cast<int>(src_vertices.cols());

    auto F = std::make_shared<ImpulseResponseField>(tgt_vertices, tgt_cells,
                                                    src_vertices, src_cells,
                                                    /*batches_normalized=*/false);
    // mu(x) = (x, 0.5 + 0.2 (x - 0.5)); constant diagonal Sigma.
    const Eigen::Matrix2d Sigma_field = Eigen::Vector2d(0.01, 0.02).asDiagonal();
    Eigen::MatrixXd field_mu(2, nv_src);
    Eigen::MatrixXd field_Sigma(4, nv_src);
    for ( int v = 0; v < nv_src; ++v )
    {
        const double x = src_vertices(0, v);
        field_mu.col(v) = Eigen::Vector2d(x, 0.5 + 0.2 * ( x - 0.5 ));
        field_Sigma.col(v) = Eigen::Map<const Eigen::VectorXd>(Sigma_field.data(), 4);
    }
    F->set_moment_fields(Eigen::VectorXd(), field_mu, field_Sigma);

    for ( double x1 : { 0.2, 0.4, 0.6, 0.8 } )
    {
        const Eigen::Vector2d mu1(x1, 0.5 + 0.2 * ( x1 - 0.5 ));
        Eigen::VectorXd psi(nv_tgt);
        for ( int v = 0; v < nv_tgt; ++v )
        {
            psi(v) = std::exp(-( tgt_vertices.col(v) - mu1 ).squaredNorm() / ( 0.1 * 0.1 ));
        }
        Eigen::MatrixXd pts(1, 1);
        pts(0, 0) = x1;
        Eigen::MatrixXd mu_s(2, 1);
        mu_s.col(0) = mu1;
        std::vector<Eigen::MatrixXd> Sigma = { Eigen::Vector2d(0.0025, 0.005).asDiagonal() };
        F->add_batch(pts, psi, Eigen::VectorXd(), mu_s, Sigma);
    }

    const KernelEvaluator K(F, nullptr, gated_config(Frame::mean_translation, 2));
    Eigen::MatrixXd xx(1, 9);
    for ( int ii = 0; ii < 9; ++ii )
    {
        xx(0, ii) = 0.1 + 0.8 * ii / 8.0;
    }
    const auto partition = recursive_bisection_partition(xx, 3);

    const BlockLowRankBuildResult r =
        block_low_rank(K, tgt_vertices, xx, partition, 0.0);
    CHECK(r.all_converged);
    CHECK(r.matrix.num_sources() == 9);
    CHECK(r.matrix.num_targets() == nv_tgt);
    CHECK(( r.matrix.to_dense() - K.block(tgt_vertices, xx) ).cwiseAbs().maxCoeff() == 0.0);

    const Eigen::MatrixXd U = pseudo_random(9, 2, 5);
    CHECK(( r.matrix.apply(U)
            - K.block(tgt_vertices, xx) * U ).cwiseAbs().maxCoeff() <= 1e-13);
}


TEST_CASE("BlockLowRank: container validation")
{
    BlockLowRank::Block ok;
    ok.source_ids = { 0, 1 };
    ok.target_ids = { 0, 2 };
    ok.factored = true;
    ok.target_factor = Eigen::MatrixXd::Zero(2, 1);
    ok.source_factor = Eigen::MatrixXd::Zero(2, 1);
    CHECK_NOTHROW(BlockLowRank(3, 3, { ok }));

    // Source ids must partition (no reuse across blocks).
    BlockLowRank::Block other = ok;
    other.source_ids = { 1, 2 };
    CHECK_THROWS_AS(BlockLowRank(3, 3, { ok, other }), std::invalid_argument);

    // Target ids strictly increasing.
    BlockLowRank::Block bad = ok;
    bad.target_ids = { 2, 0 };
    CHECK_THROWS_AS(BlockLowRank(3, 3, { bad }), std::invalid_argument);

    // Ids in range.
    bad = ok;
    bad.source_ids = { 0, 7 };
    CHECK_THROWS_AS(BlockLowRank(3, 3, { bad }), std::invalid_argument);

    // Factor shapes consistent with the storage mode.
    bad = ok;
    bad.target_factor = Eigen::MatrixXd::Zero(1, 1);
    CHECK_THROWS_AS(BlockLowRank(3, 3, { bad }), std::invalid_argument);
    bad = ok;
    bad.factored = false; // dense needs (|T|, |S|) and empty source_factor
    CHECK_THROWS_AS(BlockLowRank(3, 3, { bad }), std::invalid_argument);
    bad.target_factor = Eigen::MatrixXd::Zero(2, 2);
    bad.source_factor.resize(0, 0);
    CHECK_NOTHROW(BlockLowRank(3, 3, { bad }));
}
