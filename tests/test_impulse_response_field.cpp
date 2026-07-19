// SPDX-License-Identifier: MIT
#include "doctest/doctest.h"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "psfi/psfi.hpp"

using namespace psfi;

namespace {

// Structured triangulation of [0, 1]^2: (n+1)^2 vertices, 2 n^2 cells.
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

// Affine functions are represented exactly by CG1, so evaluating a batch
// whose vertex values sample an affine function gives the affine function.
double affine( double a, double b, double c, const Eigen::VectorXd& z )
{
    return a + b * z(0) + c * z(1);
}

Eigen::VectorXd affine_vertex_values( double a, double b, double c, const Eigen::MatrixXd& vertices )
{
    Eigen::VectorXd out(vertices.cols());
    for ( int v = 0; v < vertices.cols(); ++v )
    {
        out(v) = affine(a, b, c, vertices.col(v));
    }
    return out;
}

void expect_validate_error( const ImpulseResponseField& F, const EvalConfig& cfg,
                            const std::string& substring )
{
    bool threw = false;
    try
    {
        F.validate(cfg);
    }
    catch ( const std::invalid_argument& e )
    {
        threw = true;
        CHECK_MESSAGE(std::string(e.what()).find(substring) != std::string::npos,
                      "message was: ", e.what(), " — expected substring: ", substring);
    }
    CHECK_MESSAGE(threw, "expected validate() to throw for missing: ", substring);
}

const Eigen::VectorXd kNoV;     // size-zero: moment absent
const Eigen::MatrixXd kNoMu;    // 0x0: moment absent
const std::vector<Eigen::MatrixXd> kNoSigma;

} // end anonymous namespace


TEST_CASE("ImpulseResponseField: construction and add_batch validation")
{
    Eigen::MatrixXd vertices;
    Eigen::MatrixXi cells;
    make_grid_mesh(4, vertices, cells);
    const Eigen::VectorXd psi = affine_vertex_values(1.0, 0.0, 0.0, vertices);

    ImpulseResponseField F(vertices, cells);
    CHECK(F.dim() == 2);
    CHECK(F.num_vertices() == 25);
    CHECK(F.num_batches() == 0);
    CHECK(F.num_sample_points() == 0);
    CHECK(F.batches_normalized());

    Eigen::MatrixXd pts(2, 1);
    pts.col(0) = Eigen::Vector2d(0.5, 0.5);

    // Wrong shapes throw.
    Eigen::MatrixXd bad_pts(3, 1);
    bad_pts.setZero();
    CHECK_THROWS_AS(F.add_batch(bad_pts, psi, kNoV, kNoMu, kNoSigma), std::invalid_argument);
    Eigen::VectorXd bad_psi(7);
    bad_psi.setZero();
    CHECK_THROWS_AS(F.add_batch(pts, bad_psi, kNoV, kNoMu, kNoSigma), std::invalid_argument);
    Eigen::VectorXd bad_V(3);
    bad_V.setOnes();
    CHECK_THROWS_AS(F.add_batch(pts, psi, bad_V, kNoMu, kNoSigma), std::invalid_argument);
    Eigen::MatrixXd bad_mu(2, 3);
    bad_mu.setZero();
    CHECK_THROWS_AS(F.add_batch(pts, psi, kNoV, bad_mu, kNoSigma), std::invalid_argument);
    std::vector<Eigen::MatrixXd> bad_Sigma_count(2, Eigen::MatrixXd::Identity(2, 2));
    CHECK_THROWS_AS(F.add_batch(pts, psi, kNoV, kNoMu, bad_Sigma_count), std::invalid_argument);

    // Non-SPD Sigma throws, and a failed add leaves the field unchanged.
    std::vector<Eigen::MatrixXd> indefinite{-Eigen::MatrixXd::Identity(2, 2)};
    CHECK_THROWS_AS(F.add_batch(pts, psi, kNoV, kNoMu, indefinite), std::invalid_argument);
    CHECK(F.num_batches() == 0);
    CHECK(F.num_sample_points() == 0);

    // First batch fixes which per-sample moments are present.
    Eigen::VectorXd V1(1);
    V1(0) = 2.0;
    F.add_batch(pts, psi, V1, kNoMu, kNoSigma);
    CHECK(F.num_batches() == 1);
    CHECK(F.has_sample_V());
    CHECK(!F.has_sample_mu());
    CHECK(!F.has_sample_Sigma());
    CHECK_THROWS_AS(F.add_batch(pts, psi, kNoV, kNoMu, kNoSigma), std::invalid_argument); // V now missing
    CHECK(F.num_batches() == 1);

    // Moment field shape validation.
    CHECK_THROWS_AS(F.set_moment_fields(bad_V, kNoMu, kNoMu), std::invalid_argument);
    Eigen::MatrixXd bad_field_mu(3, F.num_vertices());
    bad_field_mu.setZero();
    CHECK_THROWS_AS(F.set_moment_fields(kNoV, bad_field_mu, kNoMu), std::invalid_argument);
    Eigen::MatrixXd bad_field_Sigma(3, F.num_vertices());
    bad_field_Sigma.setZero();
    CHECK_THROWS_AS(F.set_moment_fields(kNoV, kNoMu, bad_field_Sigma), std::invalid_argument);
}


TEST_CASE("ImpulseResponseField: validate() reports exactly the missing data")
{
    Eigen::MatrixXd vertices;
    Eigen::MatrixXi cells;
    make_grid_mesh(4, vertices, cells);
    const Eigen::VectorXd psi = affine_vertex_values(1.0, 0.0, 0.0, vertices);
    Eigen::MatrixXd pts(2, 1);
    pts.col(0) = Eigen::Vector2d(0.5, 0.5);

    // Field with no moments anywhere, normalized batches.
    ImpulseResponseField F(vertices, cells, /*batches_normalized=*/true);
    F.add_batch(pts, psi, kNoV, kNoMu, kNoSigma);

    EvalConfig minimal;
    minimal.frame   = Frame::translation;
    minimal.scaling = Scaling::none;
    minimal.support = Support::none;

    // Scaling::none on normalized batches needs per-sample V.
    expect_validate_error(F, minimal, "per-sample V");

    {
        EvalConfig cfg = minimal;
        cfg.frame = Frame::mean_translation;
        expect_validate_error(F, cfg, "per-sample mu");
        expect_validate_error(F, cfg, "vertex field mu");
    }
    {
        EvalConfig cfg = minimal;
        cfg.frame = Frame::whitened_affine;
        expect_validate_error(F, cfg, "per-sample Sigma");
        expect_validate_error(F, cfg, "vertex field Sigma");
    }
    {
        EvalConfig cfg = minimal;
        cfg.scaling = Scaling::volume;
        expect_validate_error(F, cfg, "vertex field V");
    }
    {
        EvalConfig cfg = minimal;
        cfg.support = Support::ellipsoid;
        expect_validate_error(F, cfg, "per-sample Sigma");
        cfg.tau = 0.0;
        expect_validate_error(F, cfg, "tau > 0");
    }
    {
        EvalConfig cfg = minimal;
        cfg.num_neighbors = 0;
        expect_validate_error(F, cfg, "num_neighbors");
    }

    // Raw batches + identity frame + no gate + no scaling: nothing required.
    ImpulseResponseField G(vertices, cells, /*batches_normalized=*/false);
    G.add_batch(pts, psi, kNoV, kNoMu, kNoSigma);
    EvalConfig nothing_needed;
    nothing_needed.frame   = Frame::identity;
    nothing_needed.scaling = Scaling::none;
    nothing_needed.support = Support::none;
    CHECK_NOTHROW(G.validate(nothing_needed));

    // volume on raw batches needs per-sample V too.
    {
        EvalConfig cfg = nothing_needed;
        cfg.scaling = Scaling::volume;
        expect_validate_error(G, cfg, "per-sample V");
    }
}


TEST_CASE("ImpulseResponseField: minimal-data mode (raw single-impulse batches)")
{
    // identity frame + no scaling + no gate on raw batches: predictions are
    // just the batch function evaluated at y, from the k nearest samples.
    Eigen::MatrixXd vertices;
    Eigen::MatrixXi cells;
    make_grid_mesh(8, vertices, cells);
    const Eigen::VectorXd psi = affine_vertex_values(2.0, 3.0, -1.0, vertices);

    ImpulseResponseField F(vertices, cells, /*batches_normalized=*/false);
    Eigen::MatrixXd pts(2, 2);
    pts.col(0) = Eigen::Vector2d(0.3, 0.4);
    pts.col(1) = Eigen::Vector2d(0.7, 0.6);
    F.add_batch(pts, psi, kNoV, kNoMu, kNoSigma);

    EvalConfig cfg;
    cfg.frame         = Frame::identity;
    cfg.scaling       = Scaling::none;
    cfg.support       = Support::none;
    cfg.num_neighbors = 2;

    const Eigen::Vector2d y(0.45, 0.55);
    const Eigen::Vector2d x(0.35, 0.40);
    std::vector<Prediction> P = F.predictions(y, x, cfg);
    REQUIRE(P.size() == 2);
    CHECK(P[0].sample_index == 0); // nearest to x first
    CHECK(P[1].sample_index == 1);
    CHECK(P[0].point.isApprox(pts.col(0)));
    CHECK(P[0].value == doctest::Approx(affine(2.0, 3.0, -1.0, y)).epsilon(1e-13));
    CHECK(P[1].value == doctest::Approx(affine(2.0, 3.0, -1.0, y)).epsilon(1e-13));

    // num_neighbors = 1 keeps only the nearest sample.
    cfg.num_neighbors = 1;
    P = F.predictions(y, x, cfg);
    REQUIRE(P.size() == 1);
    CHECK(P[0].sample_index == 0);
}


TEST_CASE("ImpulseResponseField: frame maps and scalings against closed forms")
{
    Eigen::MatrixXd vertices;
    Eigen::MatrixXi cells;
    make_grid_mesh(8, vertices, cells);

    // Batch functions sample affine functions (exact under CG1).
    const double a1 = 2.0, b1 = 3.0, c1 = -1.0;
    const double a2 = 1.0, b2 = -1.0, c2 = 2.0;
    const Eigen::VectorXd psi1 = affine_vertex_values(a1, b1, c1, vertices);
    const Eigen::VectorXd psi2 = affine_vertex_values(a2, b2, c2, vertices);

    // Affine moment fields (exact under CG1): V(z) = 1 + z0,
    // mu(z) = z + (0.05, -0.05), Sigma(z) constant.
    const Eigen::Vector2d mu_shift(0.05, -0.05);
    const Eigen::MatrixXd Sigma_field_const = Eigen::Vector2d(0.04, 0.09).asDiagonal();
    const int nv = static_cast<int>(vertices.cols());
    Eigen::VectorXd field_V(nv);
    Eigen::MatrixXd field_mu(2, nv);
    Eigen::MatrixXd field_Sigma(4, nv);
    for ( int v = 0; v < nv; ++v )
    {
        field_V(v)        = 1.0 + vertices(0, v);
        field_mu.col(v)   = vertices.col(v) + mu_shift;
        field_Sigma.col(v) = Eigen::Map<const Eigen::VectorXd>(Sigma_field_const.data(), 4);
    }

    // Two single-sample batches with full per-sample moments.
    const Eigen::Vector2d x1(0.3, 0.4), mu1(0.35, 0.45);
    const Eigen::Vector2d x2(0.7, 0.6), mu2(0.65, 0.55);
    const double V1 = 2.0, V2 = 1.5;
    const Eigen::MatrixXd Sigma1 = Eigen::Vector2d(0.01, 0.0225).asDiagonal();
    const Eigen::MatrixXd Sigma2 = Eigen::Vector2d(0.01, 0.01).asDiagonal();

    ImpulseResponseField F(vertices, cells, /*batches_normalized=*/true);
    {
        Eigen::MatrixXd pts(2, 1);
        Eigen::VectorXd V(1);
        Eigen::MatrixXd mu(2, 1);
        pts.col(0) = x1; V(0) = V1; mu.col(0) = mu1;
        F.add_batch(pts, psi1, V, mu, {Sigma1});
        pts.col(0) = x2; V(0) = V2; mu.col(0) = mu2;
        F.add_batch(pts, psi2, V, mu, {Sigma2});
    }
    F.set_moment_fields(field_V, field_mu, field_Sigma);
    CHECK(F.num_batches() == 2);
    CHECK(F.num_sample_points() == 2);
    CHECK(F.batch_range(1).first == 1);

    const Eigen::Vector2d y(0.45, 0.42);
    const Eigen::Vector2d x(0.40, 0.40);
    const double V_x = 1.0 + x(0);
    const Eigen::Vector2d mu_x = x + mu_shift;

    EvalConfig cfg;
    cfg.support       = Support::none;
    cfg.num_neighbors = 2;

    auto value_of_sample = []( const std::vector<Prediction>& P, int sample_index ) -> double
    {
        for ( const Prediction& p : P )
        {
            if ( p.sample_index == sample_index )
            {
                return p.value;
            }
        }
        REQUIRE(false);
        return 0.0;
    };

    // frame = translation, scaling = none: f_i = V_i * psi_b(y - x + x_i).
    cfg.frame   = Frame::translation;
    cfg.scaling = Scaling::none;
    {
        std::vector<Prediction> P = F.predictions(y, x, cfg);
        REQUIRE(P.size() == 2);
        const Eigen::Vector2d z1 = y - x + x1;
        const Eigen::Vector2d z2 = y - x + x2;
        CHECK(value_of_sample(P, 0) == doctest::Approx(V1 * affine(a1, b1, c1, z1)).epsilon(1e-13));
        CHECK(value_of_sample(P, 1) == doctest::Approx(V2 * affine(a2, b2, c2, z2)).epsilon(1e-13));
    }

    // frame = mean_translation, scaling = volume: f_i = V(x) * psi_b(y - mu(x) + mu_i).
    cfg.frame   = Frame::mean_translation;
    cfg.scaling = Scaling::volume;
    {
        std::vector<Prediction> P = F.predictions(y, x, cfg);
        REQUIRE(P.size() == 2);
        const Eigen::Vector2d z1 = y - mu_x + mu1;
        const Eigen::Vector2d z2 = y - mu_x + mu2;
        CHECK(value_of_sample(P, 0) == doctest::Approx(V_x * affine(a1, b1, c1, z1)).epsilon(1e-13));
        CHECK(value_of_sample(P, 1) == doctest::Approx(V_x * affine(a2, b2, c2, z2)).epsilon(1e-13));
    }

    // frame = whitened_affine, scaling = volume_det, diagonal covariances:
    // T_i(y) = mu_i + diag(t/s) (y - mu(x)),  s_i factor sqrt(det Sigma_i / det Sigma(x)).
    cfg.frame   = Frame::whitened_affine;
    cfg.scaling = Scaling::volume_det;
    {
        std::vector<Prediction> P = F.predictions(y, x, cfg);
        REQUIRE(P.size() == 2);
        const Eigen::Vector2d w = y - mu_x;
        const Eigen::Vector2d z1 = mu1 + Eigen::Vector2d(std::sqrt(0.01 / 0.04) * w(0),
                                                         std::sqrt(0.0225 / 0.09) * w(1));
        const Eigen::Vector2d z2 = mu2 + Eigen::Vector2d(std::sqrt(0.01 / 0.04) * w(0),
                                                         std::sqrt(0.01 / 0.09) * w(1));
        const double det_ratio1 = std::sqrt(( 0.01 * 0.0225 ) / ( 0.04 * 0.09 ));
        const double det_ratio2 = std::sqrt(( 0.01 * 0.01 ) / ( 0.04 * 0.09 ));
        CHECK(value_of_sample(P, 0)
              == doctest::Approx(V_x * det_ratio1 * affine(a1, b1, c1, z1)).epsilon(1e-12));
        CHECK(value_of_sample(P, 1)
              == doctest::Approx(V_x * det_ratio2 * affine(a2, b2, c2, z2)).epsilon(1e-12));
    }
}


TEST_CASE("ImpulseResponseField: frame-map equivalences")
{
    Eigen::MatrixXd vertices;
    Eigen::MatrixXi cells;
    make_grid_mesh(8, vertices, cells);
    const Eigen::VectorXd psi = affine_vertex_values(1.5, -2.0, 0.5, vertices);
    const int nv = static_cast<int>(vertices.cols());

    // mu_i = x_i + shift and mu(z) = z + shift makes mean_translation reduce
    // to translation; sample Sigma equal to the (constant) Sigma field makes
    // whitened_affine reduce to mean_translation.
    const Eigen::Vector2d shift(0.02, -0.03);
    const Eigen::MatrixXd Sigma_shared = ( Eigen::Matrix2d() << 0.05, 0.01, 0.01, 0.03 ).finished();

    Eigen::VectorXd field_V = Eigen::VectorXd::Ones(nv);
    Eigen::MatrixXd field_mu(2, nv);
    Eigen::MatrixXd field_Sigma(4, nv);
    for ( int v = 0; v < nv; ++v )
    {
        field_mu.col(v)    = vertices.col(v) + shift;
        field_Sigma.col(v) = Eigen::Map<const Eigen::VectorXd>(Sigma_shared.data(), 4);
    }

    ImpulseResponseField F(vertices, cells);
    Eigen::MatrixXd pts(2, 2);
    pts.col(0) = Eigen::Vector2d(0.35, 0.45);
    pts.col(1) = Eigen::Vector2d(0.60, 0.55);
    Eigen::VectorXd V(2);
    V << 1.0, 1.0;
    Eigen::MatrixXd mu = pts.colwise() + shift;
    F.add_batch(pts, psi, V, mu, {Sigma_shared, Sigma_shared});
    F.set_moment_fields(field_V, field_mu, field_Sigma);

    EvalConfig cfg;
    cfg.scaling       = Scaling::volume;
    cfg.support       = Support::ellipsoid;
    cfg.tau           = 10.0; // wide gate: keep everything
    cfg.num_neighbors = 2;

    const std::vector<Eigen::Vector2d> ys = {{0.42, 0.48}, {0.55, 0.50}, {0.38, 0.52}};
    const std::vector<Eigen::Vector2d> xs = {{0.40, 0.45}, {0.52, 0.51}, {0.45, 0.47}};
    for ( size_t tt = 0; tt < ys.size(); ++tt )
    {
        cfg.frame = Frame::translation;
        std::vector<Prediction> P_tr = F.predictions(ys[tt], xs[tt], cfg);
        cfg.frame = Frame::mean_translation;
        std::vector<Prediction> P_mt = F.predictions(ys[tt], xs[tt], cfg);
        cfg.frame = Frame::whitened_affine;
        std::vector<Prediction> P_wa = F.predictions(ys[tt], xs[tt], cfg);

        REQUIRE(P_tr.size() == P_mt.size());
        REQUIRE(P_tr.size() == P_wa.size());
        for ( size_t jj = 0; jj < P_tr.size(); ++jj )
        {
            CHECK(P_mt[jj].value == doctest::Approx(P_tr[jj].value).epsilon(1e-12));
            CHECK(P_wa[jj].value == doctest::Approx(P_mt[jj].value).epsilon(1e-12));
        }
    }
}


TEST_CASE("ImpulseResponseField: support gate and domain exclusion")
{
    Eigen::MatrixXd vertices;
    Eigen::MatrixXi cells;
    make_grid_mesh(8, vertices, cells);
    const Eigen::VectorXd psi = affine_vertex_values(2.0, 3.0, -1.0, vertices);

    const Eigen::Vector2d x1(0.35, 0.45);
    const Eigen::MatrixXd Sigma1 = Eigen::Vector2d(0.01, 0.0225).asDiagonal();

    ImpulseResponseField F(vertices, cells);
    Eigen::MatrixXd pts(2, 1);
    pts.col(0) = x1;
    Eigen::VectorXd V(1);
    V(0) = 2.0;
    Eigen::MatrixXd mu(2, 1);
    mu.col(0) = x1;
    F.add_batch(pts, psi, V, mu, {Sigma1});

    EvalConfig cfg;
    cfg.frame         = Frame::translation;
    cfg.scaling       = Scaling::none;
    cfg.support       = Support::ellipsoid;
    cfg.tau           = 3.0;
    cfg.num_neighbors = 1;

    // Transported point inside mesh and inside the ellipsoid: normal value.
    {
        const Eigen::Vector2d x(0.40, 0.40), y(0.42, 0.44); // z = y - x + x1 = (0.37, 0.49)
        std::vector<Prediction> P = F.predictions(y, x, cfg);
        REQUIRE(P.size() == 1);
        const Eigen::Vector2d z = y - x + x1;
        CHECK(P[0].value == doctest::Approx(2.0 * affine(2.0, 3.0, -1.0, z)).epsilon(1e-13));
    }

    // Inside mesh but outside the ellipsoid: gated to zero, still present.
    {
        const Eigen::Vector2d x(0.40, 0.40), y(0.90, 0.40); // z = (0.85, 0.45): dz = (0.5, 0)
        std::vector<Prediction> P = F.predictions(y, x, cfg);
        REQUIRE(P.size() == 1);
        CHECK(P[0].value == 0.0);
    }

    // Transported point outside the mesh: sample excluded entirely.
    {
        const Eigen::Vector2d x(0.40, 0.40), y(1.10, 0.40); // z = (1.05, 0.45)
        std::vector<Prediction> P = F.predictions(y, x, cfg);
        CHECK(P.empty());
    }

    // Gate off: the same far point evaluates the batch function.
    {
        cfg.support = Support::none;
        const Eigen::Vector2d x(0.40, 0.40), y(0.90, 0.40);
        std::vector<Prediction> P = F.predictions(y, x, cfg);
        REQUIRE(P.size() == 1);
        const Eigen::Vector2d z = y - x + x1;
        CHECK(P[0].value == doctest::Approx(2.0 * affine(2.0, 3.0, -1.0, z)).epsilon(1e-13));
    }
}


TEST_CASE("ImpulseResponseField: x outside the mesh")
{
    Eigen::MatrixXd vertices;
    Eigen::MatrixXi cells;
    make_grid_mesh(4, vertices, cells);
    const Eigen::VectorXd psi = affine_vertex_values(1.0, 1.0, 0.0, vertices);
    const int nv = static_cast<int>(vertices.cols());

    ImpulseResponseField F(vertices, cells);
    Eigen::MatrixXd pts(2, 1);
    pts.col(0) = Eigen::Vector2d(0.5, 0.5);
    Eigen::VectorXd V(1);
    V(0) = 1.0;
    Eigen::MatrixXd mu = pts;
    F.add_batch(pts, psi, V, mu, {Eigen::MatrixXd::Identity(2, 2)});
    F.set_moment_fields(Eigen::VectorXd::Ones(nv), Eigen::MatrixXd(vertices),
                        Eigen::MatrixXd::Identity(2, 2).reshaped(4, 1).replicate(1, nv));

    const Eigen::Vector2d x_out(1.5, 0.5), y(0.6, 0.5);

    // Configurations that evaluate moment fields at x: uninformed => empty.
    EvalConfig cfg;
    cfg.frame   = Frame::mean_translation;
    cfg.scaling = Scaling::volume;
    cfg.support = Support::none;
    CHECK(F.predictions(y, x_out, cfg).empty());

    // Configurations with no field lookups still work for exterior x
    // (translated z = y2 - x_out + x_1 = (0.4, 0.5) is inside the mesh).
    cfg.frame   = Frame::translation;
    cfg.scaling = Scaling::none;
    const Eigen::Vector2d y2(1.4, 0.5);
    std::vector<Prediction> P = F.predictions(y2, x_out, cfg);
    REQUIRE(P.size() == 1);
    CHECK(P[0].value == doctest::Approx(affine(1.0, 1.0, 0.0, Eigen::Vector2d(0.4, 0.5))).epsilon(1e-13));
}


TEST_CASE("ImpulseResponseField: deferred kd-tree rebuild")
{
    Eigen::MatrixXd vertices;
    Eigen::MatrixXi cells;
    make_grid_mesh(4, vertices, cells);
    const Eigen::VectorXd psi = affine_vertex_values(1.0, 0.0, 0.0, vertices);

    ImpulseResponseField F(vertices, cells, /*batches_normalized=*/false);
    Eigen::MatrixXd pts(2, 1);
    pts.col(0) = Eigen::Vector2d(0.5, 0.5);
    F.add_batch(pts, psi, kNoV, kNoMu, kNoSigma, /*rebuild=*/false);

    EvalConfig cfg;
    cfg.frame   = Frame::identity;
    cfg.scaling = Scaling::none;
    cfg.support = Support::none;

    const Eigen::Vector2d y(0.5, 0.5), x(0.5, 0.5);
    CHECK_THROWS_AS(F.predictions(y, x, cfg), std::logic_error);
    F.rebuild_kdtree();
    CHECK(F.predictions(y, x, cfg).size() == 1);
}
