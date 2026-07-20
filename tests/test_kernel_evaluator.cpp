// SPDX-License-Identifier: MIT
#include "doctest/doctest.h"

#include <memory>
#include <stdexcept>
#include <vector>

#include <Eigen/Dense>

#include "ellipsoid_psf/ellipsoid_psf.hpp"

using namespace ellipsoid_psf;

namespace {

// Structured triangulation of [0, 1]^2 (same as test_impulse_response_field).
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

Eigen::VectorXd affine_vertex_values( double a, double b, double c, const Eigen::MatrixXd& vertices )
{
    Eigen::VectorXd out(vertices.cols());
    for ( int v = 0; v < vertices.cols(); ++v )
    {
        out(v) = a + b * vertices(0, v) + c * vertices(1, v);
    }
    return out;
}

const Eigen::VectorXd kNoV;
const Eigen::MatrixXd kNoMu;
const std::vector<Eigen::MatrixXd> kNoSigma;

// A field with three single-sample batches carrying affine batch functions,
// raw normalization, no moments: usable with translation/identity frames.
std::shared_ptr<ImpulseResponseField> make_simple_field()
{
    Eigen::MatrixXd vertices;
    Eigen::MatrixXi cells;
    make_grid_mesh(8, vertices, cells);
    auto F = std::make_shared<ImpulseResponseField>(vertices, cells, /*batches_normalized=*/false);
    const std::vector<Eigen::Vector2d> pts = {{0.30, 0.40}, {0.55, 0.50}, {0.45, 0.65},
                                              {0.62, 0.35}, {0.38, 0.58}};
    const std::vector<Eigen::Vector3d> abc = {{2.0, 3.0, -1.0}, {1.0, -1.0, 2.0}, {0.5, 2.0, 1.0},
                                              {-1.0, 1.0, 1.0}, {3.0, -2.0, 0.5}};
    for ( int b = 0; b < 5; ++b )
    {
        Eigen::MatrixXd P(2, 1);
        P.col(0) = pts[b];
        F->add_batch(P, affine_vertex_values(abc[b](0), abc[b](1), abc[b](2), vertices),
                     kNoV, kNoMu, kNoSigma);
    }
    return F;
}

EvalConfig translation_config( int num_neighbors )
{
    EvalConfig cfg;
    cfg.frame         = Frame::translation;
    cfg.scaling       = Scaling::none;
    cfg.support       = Support::none;
    cfg.num_neighbors = num_neighbors;
    return cfg;
}

} // end anonymous namespace


TEST_CASE("KernelEvaluator: single neighbor passes its prediction through")
{
    auto F = make_simple_field();
    const EvalConfig cfg = translation_config(1);
    KernelEvaluator K(F, nullptr, cfg);

    const Eigen::Vector2d y(0.42, 0.44), x(0.33, 0.41);
    const std::vector<Prediction> P = F->predictions(y, x, cfg);
    REQUIRE(P.size() == 1);
    CHECK(K(y, x) == doctest::Approx(P[0].value).epsilon(1e-13));
}


TEST_CASE("KernelEvaluator: constant predictions give the constant")
{
    Eigen::MatrixXd vertices;
    Eigen::MatrixXi cells;
    make_grid_mesh(6, vertices, cells);
    auto F = std::make_shared<ImpulseResponseField>(vertices, cells, /*batches_normalized=*/false);
    Eigen::MatrixXd pts(2, 3);
    pts << 0.3, 0.6, 0.4,
           0.3, 0.5, 0.7;
    F->add_batch(pts, Eigen::VectorXd::Constant(vertices.cols(), 4.5), kNoV, kNoMu, kNoSigma);

    EvalConfig cfg;
    cfg.frame         = Frame::identity;
    cfg.scaling       = Scaling::none;
    cfg.support       = Support::none;
    cfg.num_neighbors = 3;

    for ( RBFKernel kernel : {RBFKernel::gaussian, RBFKernel::thin_plate_spline, RBFKernel::linear} )
    {
        RBFScheme rbf;
        rbf.kernel = kernel;
        KernelEvaluator K(F, nullptr, cfg, rbf);
        CHECK(K(Eigen::Vector2d(0.5, 0.5), Eigen::Vector2d(0.45, 0.5))
              == doctest::Approx(4.5).epsilon(1e-10));
    }
}


TEST_CASE("KernelEvaluator: exact at sample columns without a special case")
{
    // Five neighbors with a degree-1 tail (three coefficients in 2D) keeps
    // k > np: with k <= np the bordered system degenerates to exact
    // polynomial interpolation and smoothing would be a no-op.
    auto F = make_simple_field();
    const EvalConfig cfg = translation_config(5);
    KernelEvaluator K(F, nullptr, cfg); // smoothing = 0

    // x exactly at a sample point: the interpolation centers include the
    // origin, and center reproduction makes the entry match that sample's
    // prediction — no snapping logic involved.
    const Eigen::Vector2d x1(0.30, 0.40);
    const Eigen::Vector2d y(0.38, 0.47);
    const std::vector<Prediction> P = F->predictions(y, x1, cfg);
    REQUIRE(P.size() == 5);
    REQUIRE(P[0].sample_index == 0); // nearest = the sample at x itself
    CHECK(K(y, x1) == doctest::Approx(P[0].value).epsilon(1e-8));

    // With smoothing > 0 the column exactness is (deliberately) given up.
    RBFScheme smoothed;
    smoothed.smoothing = 1e-1;
    KernelEvaluator K_s(F, nullptr, cfg, smoothed);
    CHECK(std::abs(K_s(y, x1) - P[0].value) > 1e-6);
}


TEST_CASE("KernelEvaluator: symmetric mode merges duplicate centers at y = x")
{
    auto F = make_simple_field();
    const EvalConfig cfg = translation_config(3);
    KernelEvaluator K_cols(F, nullptr, cfg);
    KernelEvaluator K_sym(F, F, cfg); // same field as rows and cols

    // At y = x the forward and adjoint prediction sets coincide, so after
    // duplicate merging the symmetric evaluator must agree with cols-only.
    const Eigen::Vector2d p(0.47, 0.52);
    CHECK(K_sym(p, p) == doctest::Approx(K_cols(p, p)).epsilon(1e-12));
    CHECK(K_sym.symmetric());
    CHECK(!K_cols.symmetric());

    // At y != x the pooled set is larger but evaluation still runs.
    const Eigen::Vector2d y(0.52, 0.55), x(0.44, 0.48);
    CHECK(std::isfinite(K_sym(y, x)));
}


TEST_CASE("KernelEvaluator: block matches entrywise evaluation, any thread count")
{
    auto F = make_simple_field();
    KernelEvaluator K(F, F, translation_config(3));

    Eigen::MatrixXd yy(2, 5), xx(2, 4);
    yy << 0.35, 0.45, 0.55, 0.40, 0.60,
          0.40, 0.50, 0.55, 0.62, 0.45;
    xx << 0.32, 0.50, 0.44, 0.58,
          0.42, 0.52, 0.60, 0.48;

    const Eigen::MatrixXd B1 = K.block(yy, xx, 1);
    const Eigen::MatrixXd B4 = K.block(yy, xx, 4);
    REQUIRE(B1.rows() == 5);
    REQUIRE(B1.cols() == 4);
    CHECK(( B1 - B4 ).cwiseAbs().maxCoeff() == 0.0); // same arithmetic, any threading
    for ( int jj = 0; jj < 4; ++jj )
    {
        for ( int ii = 0; ii < 5; ++ii )
        {
            CHECK(B1(ii, jj) == doctest::Approx(K(yy.col(ii), xx.col(jj))).epsilon(1e-14));
        }
    }
}


TEST_CASE("KernelEvaluator: construction validates everything up front")
{
    auto F = make_simple_field();

    CHECK_THROWS_AS(KernelEvaluator(nullptr), std::invalid_argument);

    // Config needing data the field lacks fails at construction, not at eval.
    EvalConfig needs_moments;
    needs_moments.frame = Frame::mean_translation;
    CHECK_THROWS_AS(KernelEvaluator(F, nullptr, needs_moments), std::invalid_argument);

    RBFScheme bad_rbf;
    bad_rbf.degree = 2;
    CHECK_THROWS_AS(KernelEvaluator(F, nullptr, translation_config(3), bad_rbf),
                    std::invalid_argument);

    CHECK_THROWS_AS(KernelEvaluator(F, nullptr, translation_config(3), RBFScheme{}, -1.0),
                    std::invalid_argument);

    // Dimension mismatch between row and col fields.
    Eigen::MatrixXd verts1d(1, 3);
    verts1d << 0.0, 0.5, 1.0;
    Eigen::MatrixXi cells1d(2, 2);
    cells1d << 0, 1,
               1, 2;
    auto F1d = std::make_shared<ImpulseResponseField>(verts1d, cells1d, false);
    CHECK_THROWS_AS(KernelEvaluator(F, F1d, translation_config(3)), std::invalid_argument);
}
