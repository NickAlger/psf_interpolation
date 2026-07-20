// SPDX-License-Identifier: MIT
// The fixed-source fast path: predictions_over_targets against per-pair
// predictions, and block() against entrywise operator(), across
// configurations — including targets whose transported points leave the
// mesh (exclusions), gated far fields, and sources outside the source mesh.
#include "doctest/doctest.h"

#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

#include <Eigen/Dense>

#include "ellipsoid_psf/ellipsoid_psf.hpp"

using namespace ellipsoid_psf;

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

// The configurations exercised: every frame, with and without the gate,
// with and without volume scalings.
std::vector<EvalConfig> sweep_configs()
{
    std::vector<EvalConfig> out;
    for ( Frame frame : { Frame::identity, Frame::translation,
                          Frame::mean_translation, Frame::whitened_affine } )
    {
        for ( Support support : { Support::ellipsoid, Support::none } )
        {
            for ( Scaling scaling : { Scaling::none, Scaling::volume } )
            {
                EvalConfig cfg;
                cfg.frame         = frame;
                cfg.scaling       = scaling;
                cfg.support       = support;
                cfg.tau           = 2.0;
                cfg.num_neighbors = 3;
                out.push_back(cfg);
            }
        }
    }
    return out;
}

} // end anonymous namespace


TEST_CASE("predictions_over_targets: matches per-pair predictions exactly")
{
    auto F = make_moment_field();
    // Targets across the WHOLE square including the boundary (where
    // transported points leave the mesh and samples get excluded); sources
    // include one outside the source mesh.
    const Eigen::MatrixXd yy = square_grid(7, 0.0, 1.0);
    std::vector<Eigen::Vector2d> xs = {{0.3, 0.4}, {0.62, 0.33}, {0.05, 0.95}, {1.4, 1.4}};

    for ( const EvalConfig& cfg : sweep_configs() )
    {
        for ( const Eigen::Vector2d& x : xs )
        {
            const ImpulseResponseField::PredictionSweep sweep =
                F->predictions_over_targets(yy, x, cfg);
            const int k = static_cast<int>(sweep.sample_indices.size());
            for ( int qq = 0; qq < yy.cols(); ++qq )
            {
                const std::vector<Prediction> P = F->predictions(yy.col(qq), x, cfg);
                if ( k == 0 )
                {
                    CHECK(P.empty());
                    continue;
                }
                // The per-pair list is the sweep column with excluded
                // samples dropped, in the same (nearest-first) order, with
                // bitwise-identical values.
                size_t pp = 0;
                for ( int nn = 0; nn < k; ++nn )
                {
                    if ( sweep.excluded(nn, qq) )
                    {
                        continue;
                    }
                    REQUIRE(pp < P.size());
                    CHECK(P[pp].sample_index == sweep.sample_indices[nn]);
                    CHECK(P[pp].value == sweep.values(nn, qq));
                    ++pp;
                }
                CHECK(pp == P.size());
            }
        }
    }
}


TEST_CASE("block: fast path agrees with entrywise evaluation")
{
    auto F = make_moment_field();
    const Eigen::MatrixXd yy = square_grid(7, 0.0, 1.0);   // 49 targets incl. boundary
    Eigen::MatrixXd xx = square_grid(3, 0.2, 0.8);          // 9 sources
    xx.conservativeResize(2, 10);
    xx.col(9) = Eigen::Vector2d(1.5, 1.5);                  // outside the source mesh

    for ( const EvalConfig& cfg : sweep_configs() )
    {
        const KernelEvaluator K(F, nullptr, cfg);
        const Eigen::MatrixXd B = K.block(yy, xx);
        double max_abs = 0.0;
        for ( int jj = 0; jj < xx.cols(); ++jj )
        {
            for ( int ii = 0; ii < yy.cols(); ++ii )
            {
                const double entry = K(yy.col(ii), xx.col(jj));
                max_abs = std::max(max_abs,
                                   std::abs(B(ii, jj) - entry));
                // STRUCTURAL zeros (every prediction gated to zero, or no
                // predictions at all) stay exactly zero on both paths — the
                // sparsity invariants downstream rely on this. Coincidental
                // near-zeros of the solve are only equal up to rounding.
                bool structural_zero = true;
                for ( const Prediction& p : F->predictions(yy.col(ii), xx.col(jj), cfg) )
                {
                    if ( p.value != 0.0 )
                    {
                        structural_zero = false;
                    }
                }
                if ( structural_zero )
                {
                    CHECK(entry == 0.0);
                    CHECK(B(ii, jj) == 0.0);
                }
            }
        }
        CHECK(max_abs <= 1e-11);
    }
}


TEST_CASE("block: symmetric mode and oversized neighbor counts fall back, unchanged")
{
    auto F = make_moment_field();
    const Eigen::MatrixXd yy = square_grid(5, 0.1, 0.9);
    const Eigen::MatrixXd xx = square_grid(3, 0.2, 0.8);

    EvalConfig cfg;
    cfg.frame         = Frame::whitened_affine;
    cfg.scaling       = Scaling::none;
    cfg.support       = Support::ellipsoid;
    cfg.tau           = 2.0;
    cfg.num_neighbors = 3;
    const KernelEvaluator K_sym(F, F, cfg);
    const Eigen::MatrixXd B_sym = K_sym.block(yy, xx);
    for ( int jj = 0; jj < xx.cols(); ++jj )
    {
        for ( int ii = 0; ii < yy.cols(); ++ii )
        {
            CHECK(B_sym(ii, jj) == K_sym(yy.col(ii), xx.col(jj)));
        }
    }

    cfg.num_neighbors = 100; // > 63: the mask-cache guard takes the entry path
    const KernelEvaluator K_wide(F, nullptr, cfg);
    const Eigen::MatrixXd B_wide = K_wide.block(yy, xx);
    for ( int jj = 0; jj < xx.cols(); ++jj )
    {
        for ( int ii = 0; ii < yy.cols(); ++ii )
        {
            CHECK(B_wide(ii, jj) == K_wide(yy.col(ii), xx.col(jj)));
        }
    }
}
