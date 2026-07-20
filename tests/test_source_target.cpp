// SPDX-License-Identifier: MIT
// Rectangular kernels: separate source and target domains, including
// different dimensions. The moment map mu : Omega_source -> Omega_target is
// what makes the cross-domain frames well-typed (paper sec. 4.2).

#include "doctest/doctest.h"

#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

#include <Eigen/Dense>

#include "ellipsoid_psf/ellipsoid_psf.hpp"

using namespace ellipsoid_psf;

namespace {

void make_grid_mesh( int n, Eigen::MatrixXd& vertices, Eigen::MatrixXi& cells )
{
    const int nv = ( n + 1 ) * ( n + 1 );
    vertices.resize(2, nv);
    for ( int jj = 0; jj <= n; ++jj )
    {
        for ( int ii = 0; ii <= n; ++ii )
        {
            vertices.col(jj * ( n + 1 ) + ii)
                = Eigen::Vector2d(ii / double(n), jj / double(n));
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

double affine2(double a, double b, double c, const Eigen::VectorXd& z)
{
    return a + b * z(0) + c * z(1);
}

// The moment map of the 1D -> 2D test kernel: mu(x) = (x, 0.5 + 0.2(x-0.5)).
Eigen::Vector2d mu_map( double x )
{
    return Eigen::Vector2d(x, 0.5 + 0.2 * ( x - 0.5 ));
}

} // end anonymous namespace


TEST_CASE("source/target: 1D source domain, 2D target domain")
{
    // Target: 2D unit-square mesh carrying the batch function; source: 1D
    // interval carrying the moment fields and the sample points.
    Eigen::MatrixXd tgt_vertices, src_vertices;
    Eigen::MatrixXi tgt_cells, src_cells;
    make_grid_mesh(8, tgt_vertices, tgt_cells);
    make_interval_mesh(8, src_vertices, src_cells);
    const int nv_tgt = static_cast<int>(tgt_vertices.cols());
    const int nv_src = static_cast<int>(src_vertices.cols());

    const double a = 2.0, b = 3.0, c = -1.0;
    Eigen::VectorXd psi(nv_tgt);
    for ( int v = 0; v < nv_tgt; ++v )
    {
        psi(v) = affine2(a, b, c, tgt_vertices.col(v));
    }

    ImpulseResponseField F(tgt_vertices, tgt_cells, src_vertices, src_cells);
    CHECK(F.dim_source() == 1);
    CHECK(F.dim_target() == 2);
    CHECK(F.has_separate_source_mesh());
    CHECK(F.num_source_vertices() == nv_src);
    CHECK(F.num_target_vertices() == nv_tgt);

    // Moment fields over the SOURCE mesh with TARGET-valued entries; all
    // affine in x, hence CG1-exact.
    const Eigen::Matrix2d Sigma_field = Eigen::Vector2d(0.01, 0.02).asDiagonal();
    Eigen::VectorXd field_V(nv_src);
    Eigen::MatrixXd field_mu(2, nv_src);
    Eigen::MatrixXd field_Sigma(4, nv_src);
    for ( int v = 0; v < nv_src; ++v )
    {
        const double x = src_vertices(0, v);
        field_V(v)      = 1.0 + 0.5 * x;
        field_mu.col(v) = mu_map(x);
        field_Sigma.col(v) = Eigen::Map<const Eigen::VectorXd>(Sigma_field.data(), 4);
    }
    F.set_moment_fields(field_V, field_mu, field_Sigma);

    // One sample at x1 = 0.4 with target-valued moments.
    const double x1 = 0.4;
    const Eigen::Vector2d mu1 = mu_map(x1);
    const Eigen::Matrix2d Sigma1 = Eigen::Vector2d(0.0025, 0.005).asDiagonal();
    Eigen::MatrixXd pts(1, 1);
    pts(0, 0) = x1;
    Eigen::VectorXd V1(1);
    V1(0) = 1.0 + 0.5 * x1;
    Eigen::MatrixXd mu_s(2, 1);
    mu_s.col(0) = mu1;
    F.add_batch(pts, psi, V1, mu_s, {Sigma1});

    const Eigen::VectorXd x = Eigen::VectorXd::Constant(1, 0.55);
    const Eigen::Vector2d mu_x = mu_map(x(0));
    const double V_x = 1.0 + 0.5 * x(0);
    const Eigen::Vector2d y = mu_x + Eigen::Vector2d(0.04, -0.03);

    EvalConfig cfg;
    cfg.support       = Support::none;
    cfg.num_neighbors = 1;

    // mean_translation + volume: f = V(x) * psi(y - mu(x) + mu1).
    cfg.frame   = Frame::mean_translation;
    cfg.scaling = Scaling::volume;
    {
        std::vector<Prediction> P = F.predictions(y, x, cfg);
        REQUIRE(P.size() == 1);
        CHECK(P[0].point.size() == 1); // sample points are source-dimensional
        const Eigen::Vector2d z = y - mu_x + mu1;
        CHECK(P[0].value == doctest::Approx(V_x * affine2(a, b, c, z)).epsilon(1e-12));
    }

    // whitened_affine + volume_det, diagonal covariances (constant Sigma field).
    cfg.frame   = Frame::whitened_affine;
    cfg.scaling = Scaling::volume_det;
    {
        std::vector<Prediction> P = F.predictions(y, x, cfg);
        REQUIRE(P.size() == 1);
        Eigen::Vector2d z = mu1;
        double det_ratio = 1.0;
        const Eigen::Vector2d sample_diag(0.0025, 0.005), field_diag(0.01, 0.02);
        for ( int ii = 0; ii < 2; ++ii )
        {
            z(ii) += std::sqrt(sample_diag(ii) / field_diag(ii)) * ( y(ii) - mu_x(ii) );
            det_ratio *= sample_diag(ii) / field_diag(ii);
        }
        CHECK(P[0].value
              == doctest::Approx(V_x * std::sqrt(det_ratio) * affine2(a, b, c, z)).epsilon(1e-12));
    }

    // translation is ill-typed across dimensions: validation says so.
    cfg.frame = Frame::translation;
    cfg.scaling = Scaling::none;
    bool threw = false;
    try
    {
        F.validate(cfg);
    }
    catch ( const std::invalid_argument& e )
    {
        threw = true;
        CHECK(std::string(e.what()).find("equal source and target dimensions") != std::string::npos);
    }
    CHECK(threw);

    // Cols-only evaluator works (RBF centers are 1D); symmetric mode refuses.
    auto Fp = std::make_shared<ImpulseResponseField>(std::move(F));
    EvalConfig ecfg;
    ecfg.frame = Frame::mean_translation;
    ecfg.scaling = Scaling::volume;
    ecfg.support = Support::none;
    ecfg.num_neighbors = 1;
    KernelEvaluator K(Fp, nullptr, ecfg);
    CHECK(K.dim_source() == 1);
    CHECK(K.dim_target() == 2);
    {
        const Eigen::Vector2d z = y - mu_x + mu1;
        CHECK(K(y, x) == doctest::Approx(V_x * affine2(a, b, c, z)).epsilon(1e-12));
        Eigen::MatrixXd yy(2, 3), xx(1, 2);
        yy << 0.5, 0.55, 0.6,
              0.5, 0.45, 0.55;
        xx << 0.4, 0.6;
        const Eigen::MatrixXd B = K.block(yy, xx);
        CHECK(B.rows() == 3);
        CHECK(B.cols() == 2);
    }
    CHECK_THROWS_AS(KernelEvaluator(Fp, Fp, ecfg), std::invalid_argument);
}


TEST_CASE("source/target: equal dimensions, different meshes")
{
    // Overlapping-variables situation (e.g. pressure and density meshes):
    // fields live on a coarse source mesh, batches on a fine target mesh;
    // translation is allowed since the dimensions agree.
    Eigen::MatrixXd tgt_vertices, src_vertices;
    Eigen::MatrixXi tgt_cells, src_cells;
    make_grid_mesh(8, tgt_vertices, tgt_cells);
    make_grid_mesh(4, src_vertices, src_cells); // coarser, same unit square
    const int nv_tgt = static_cast<int>(tgt_vertices.cols());
    const int nv_src = static_cast<int>(src_vertices.cols());

    const double a = 1.5, b = -2.0, c = 0.5;
    Eigen::VectorXd psi(nv_tgt);
    for ( int v = 0; v < nv_tgt; ++v )
    {
        psi(v) = affine2(a, b, c, tgt_vertices.col(v));
    }

    ImpulseResponseField F(tgt_vertices, tgt_cells, src_vertices, src_cells,
                           /*batches_normalized=*/false);
    CHECK(F.has_separate_source_mesh());

    const Eigen::Vector2d shift(0.02, -0.03);
    Eigen::VectorXd field_V = Eigen::VectorXd::Ones(nv_src);
    Eigen::MatrixXd field_mu(2, nv_src);
    for ( int v = 0; v < nv_src; ++v )
    {
        field_mu.col(v) = src_vertices.col(v) + shift; // affine: CG1-exact on the coarse mesh
    }
    F.set_moment_fields(field_V, field_mu, Eigen::MatrixXd());

    const Eigen::Vector2d x1(0.35, 0.45);
    Eigen::MatrixXd pts(2, 1);
    pts.col(0) = x1;
    Eigen::MatrixXd mu_s(2, 1);
    mu_s.col(0) = x1 + shift;
    F.add_batch(pts, psi, Eigen::VectorXd(), mu_s, {});

    const Eigen::Vector2d x(0.42, 0.40), y(0.47, 0.44);
    EvalConfig cfg;
    cfg.scaling       = Scaling::none;
    cfg.support       = Support::none;
    cfg.num_neighbors = 1;

    // translation (allowed: equal dimensions) and mean_translation agree here
    // because mu is the identity plus a constant shift on both sides.
    cfg.frame = Frame::translation;
    std::vector<Prediction> P_tr = F.predictions(y, x, cfg);
    cfg.frame = Frame::mean_translation;
    std::vector<Prediction> P_mt = F.predictions(y, x, cfg);
    REQUIRE(P_tr.size() == 1);
    REQUIRE(P_mt.size() == 1);
    const Eigen::Vector2d z = y - x + x1;
    CHECK(P_tr[0].value == doctest::Approx(affine2(a, b, c, z)).epsilon(1e-12));
    CHECK(P_mt[0].value == doctest::Approx(P_tr[0].value).epsilon(1e-12));
}
