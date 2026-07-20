// SPDX-License-Identifier: MIT
// The library is dimension-generic by construction; these tests make that a
// fact rather than a belief, running the closed-form battery on a 1D
// interval mesh and a 3D Freudenthal (6-tets-per-cube) mesh.

#include "doctest/doctest.h"

#include <cmath>
#include <random>
#include <vector>

#include <Eigen/Dense>

#include "ellipsoid_psf/ellipsoid_psf.hpp"

using namespace ellipsoid_psf;

namespace {

// Interval mesh on [0, 1]: n segments.
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

// Freudenthal (Kuhn) triangulation of [0, 1]^3: n^3 cubes, 6 tets each.
void make_cube_mesh( int n, Eigen::MatrixXd& vertices, Eigen::MatrixXi& cells )
{
    const int m = n + 1;
    vertices.resize(3, m * m * m);
    for ( int kk = 0; kk <= n; ++kk )
    {
        for ( int jj = 0; jj <= n; ++jj )
        {
            for ( int ii = 0; ii <= n; ++ii )
            {
                vertices.col(ii + m * ( jj + m * kk ))
                    = Eigen::Vector3d(ii, jj, kk) / static_cast<double>(n);
            }
        }
    }
    const int perms[6][3] = {{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};
    cells.resize(4, 6 * n * n * n);
    int cc = 0;
    for ( int kk = 0; kk < n; ++kk )
    {
        for ( int jj = 0; jj < n; ++jj )
        {
            for ( int ii = 0; ii < n; ++ii )
            {
                for ( const auto& perm : perms )
                {
                    Eigen::Vector3i p(ii, jj, kk);
                    Eigen::Vector4i tet;
                    tet(0) = p(0) + m * ( p(1) + m * p(2) );
                    for ( int step = 0; step < 3; ++step )
                    {
                        p(perm[step]) += 1;
                        tet(step + 1) = p(0) + m * ( p(1) + m * p(2) );
                    }
                    cells.col(cc++) = tet;
                }
            }
        }
    }
}

double affine( const Eigen::VectorXd& beta, const Eigen::VectorXd& z )
{
    return beta(0) + beta.tail(z.size()).dot(z);
}

// The closed-form battery in dimension d: one sample with full moments,
// affine batch function and affine/constant moment fields (all CG1-exact).
void run_affine_battery( const Eigen::MatrixXd& vertices, const Eigen::MatrixXi& cells )
{
    const int d  = static_cast<int>(vertices.rows());
    const int nv = static_cast<int>(vertices.cols());

    Eigen::VectorXd beta(d + 1); // psi(z) = beta0 + beta_1 z_1 + ...
    for ( int ii = 0; ii <= d; ++ii )
    {
        beta(ii) = 1.0 + 0.5 * ii * ( ii % 2 == 0 ? 1.0 : -1.0 );
    }
    Eigen::VectorXd psi(nv);
    for ( int v = 0; v < nv; ++v )
    {
        psi(v) = affine(beta, vertices.col(v));
    }

    // Sample at a center-ish point in generic position (distinct
    // coordinates: points with equal coordinates lie on shared simplex
    // edges of the Freudenthal mesh, which is a degenerate location).
    Eigen::VectorXd x1(d), mu1(d);
    for ( int ii = 0; ii < d; ++ii )
    {
        x1(ii)  = 0.40 + 0.03 * ii;
        mu1(ii) = x1(ii) + 0.02 + 0.005 * ii;
    }
    const double V1 = 2.0;
    Eigen::VectorXd sample_diag(d), field_diag(d);
    for ( int ii = 0; ii < d; ++ii )
    {
        sample_diag(ii) = 0.01 * ( 1.0 + ii );  // sample Sigma_1
        field_diag(ii)  = 0.04 * ( 1.0 + ii );  // constant Sigma field
    }
    const Eigen::MatrixXd Sigma1 = sample_diag.asDiagonal();
    const Eigen::MatrixXd Sigma_field = field_diag.asDiagonal();

    // Moment fields: V(z) = 1 + z_1 (affine), mu(z) = z + shift, Sigma const.
    Eigen::VectorXd mu_shift(d);
    for ( int ii = 0; ii < d; ++ii )
    {
        mu_shift(ii) = 0.01 + 0.004 * ii;
    }
    Eigen::VectorXd field_V(nv);
    Eigen::MatrixXd field_mu(d, nv);
    Eigen::MatrixXd field_Sigma(d * d, nv);
    for ( int v = 0; v < nv; ++v )
    {
        field_V(v)         = 1.0 + vertices(0, v);
        field_mu.col(v)    = vertices.col(v) + mu_shift;
        field_Sigma.col(v) = Eigen::Map<const Eigen::VectorXd>(Sigma_field.data(), d * d);
    }

    ImpulseResponseField F(vertices, cells);
    Eigen::MatrixXd pts(d, 1);
    pts.col(0) = x1;
    Eigen::VectorXd V(1);
    V(0) = V1;
    Eigen::MatrixXd mu(d, 1);
    mu.col(0) = mu1;
    F.add_batch(pts, psi, V, mu, {Sigma1});
    F.set_moment_fields(field_V, field_mu, field_Sigma);

    Eigen::VectorXd x(d), y(d);
    for ( int ii = 0; ii < d; ++ii )
    {
        x(ii) = 0.45 + 0.02 * ii;
        y(ii) = x(ii) + 0.03 + 0.007 * ii;
    }
    const Eigen::VectorXd mu_x = x + mu_shift;
    const double V_x = 1.0 + x(0);

    EvalConfig cfg;
    cfg.support       = Support::none;
    cfg.num_neighbors = 1;

    // translation + none: f = V1 * psi(y - x + x1).
    cfg.frame   = Frame::translation;
    cfg.scaling = Scaling::none;
    {
        std::vector<Prediction> P = F.predictions(y, x, cfg);
        REQUIRE(P.size() == 1);
        CHECK(P[0].value == doctest::Approx(V1 * affine(beta, y - x + x1)).epsilon(1e-12));
    }

    // mean_translation + volume: f = V(x) * psi(y - mu(x) + mu1).
    cfg.frame   = Frame::mean_translation;
    cfg.scaling = Scaling::volume;
    {
        std::vector<Prediction> P = F.predictions(y, x, cfg);
        REQUIRE(P.size() == 1);
        CHECK(P[0].value == doctest::Approx(V_x * affine(beta, y - mu_x + mu1)).epsilon(1e-12));
    }

    // whitened + volume_det with diagonal covariances (constant Sigma field,
    // so W(x) = Sigma_field^{-1/2} exactly):
    // T(y) = mu1 + diag(sqrt(s1/sf)) (y - mu(x)), s = V(x) sqrt(prod s1/prod sf).
    cfg.frame   = Frame::whitened_affine;
    cfg.scaling = Scaling::volume_det;
    {
        std::vector<Prediction> P = F.predictions(y, x, cfg);
        REQUIRE(P.size() == 1);
        Eigen::VectorXd z = mu1;
        double det_ratio = 1.0;
        for ( int ii = 0; ii < d; ++ii )
        {
            z(ii) += std::sqrt(sample_diag(ii) / field_diag(ii)) * ( y(ii) - mu_x(ii) );
            det_ratio *= sample_diag(ii) / field_diag(ii);
        }
        CHECK(P[0].value
              == doctest::Approx(V_x * std::sqrt(det_ratio) * affine(beta, z)).epsilon(1e-12));
    }

    // Ellipsoid gate: a y far beyond tau standard deviations is kept as zero.
    cfg.frame   = Frame::translation;
    cfg.scaling = Scaling::none;
    cfg.support = Support::ellipsoid;
    cfg.tau     = 3.0;
    {
        Eigen::VectorXd y_far = x;
        y_far(0) += 0.5; // 5 sigma along the first axis
        y_far(d - 1) += 0.03;
        std::vector<Prediction> P = F.predictions(y_far, x, cfg);
        REQUIRE(P.size() == 1);
        CHECK(P[0].value == 0.0);
    }
}

} // end anonymous namespace


TEST_CASE("dimension generality: 1D interval mesh")
{
    Eigen::MatrixXd vertices;
    Eigen::MatrixXi cells;
    make_interval_mesh(16, vertices, cells);
    run_affine_battery(vertices, cells);
}


TEST_CASE("dimension generality: 3D Freudenthal tetrahedral mesh")
{
    Eigen::MatrixXd vertices;
    Eigen::MatrixXi cells;
    make_cube_mesh(4, vertices, cells);
    run_affine_battery(vertices, cells);
}


TEST_CASE("dimension generality: rbf_interpolate in d = 1, 2, 3")
{
    for ( int d : {1, 2, 3} )
    {
        // Deterministic scattered centers in general position (a structured
        // generator can produce exactly coplanar 3D points, for which the
        // degree-1 tail system is singular and affine reproduction is not
        // implied) and affine data: the interpolant reproduces the affine
        // function exactly.
        const int k = 4 + 3 * d;
        std::mt19937 gen(17 + d);
        Eigen::MatrixXd P(d, k);
        for ( int jj = 0; jj < k; ++jj )
        {
            for ( int ii = 0; ii < d; ++ii )
            {
                P(ii, jj) = static_cast<double>(gen()) / 4294967296.0;
            }
        }
        Eigen::VectorXd beta(d + 1);
        for ( int ii = 0; ii <= d; ++ii )
        {
            beta(ii) = 0.7 - 0.3 * ii;
        }
        Eigen::VectorXd f(k);
        for ( int jj = 0; jj < k; ++jj )
        {
            f(jj) = beta(0) + beta.tail(d).dot(P.col(jj));
        }
        Eigen::MatrixXd Q = 0.3 * Eigen::MatrixXd::Ones(d, 3)
            + 0.1 * P.leftCols(3);

        for ( RBFKernel kernel : {RBFKernel::gaussian, RBFKernel::thin_plate_spline} )
        {
            RBFScheme scheme;
            scheme.kernel = kernel;
            const Eigen::VectorXd s = rbf_interpolate(f, P, Q, scheme);
            for ( int qq = 0; qq < 3; ++qq )
            {
                const double expected = beta(0) + beta.tail(d).dot(Q.col(qq));
                CHECK(s(qq) == doctest::Approx(expected).epsilon(1e-9));
            }
            const Eigen::VectorXd at_centers = rbf_interpolate(f, P, P, scheme);
            for ( int jj = 0; jj < k; ++jj )
            {
                CHECK(at_centers(jj) == doctest::Approx(f(jj)).epsilon(1e-8));
            }
        }
    }
}
