// SPDX-License-Identifier: MIT
#include "doctest/doctest.h"

#include <random>
#include <stdexcept>
#include <vector>

#include <Eigen/Dense>

#include "psfi/rbf.hpp"

using namespace psfi;

namespace {

const std::vector<RBFKernel> kAllKernels = {
    RBFKernel::gaussian, RBFKernel::multiquadric, RBFKernel::inverse_multiquadric,
    RBFKernel::linear, RBFKernel::thin_plate_spline, RBFKernel::cubic};

// Deterministic scattered points in [0, 1]^d (no std::*_distribution: those
// are implementation-defined).
Eigen::MatrixXd scattered_points( int d, int k, unsigned seed )
{
    std::mt19937 gen(seed);
    Eigen::MatrixXd P(d, k);
    for ( int jj = 0; jj < k; ++jj )
    {
        for ( int ii = 0; ii < d; ++ii )
        {
            P(ii, jj) = static_cast<double>(gen()) / 4294967296.0;
        }
    }
    return P;
}

Eigen::VectorXd scattered_values( int k, unsigned seed )
{
    std::mt19937 gen(seed);
    Eigen::VectorXd f(k);
    for ( int ii = 0; ii < k; ++ii )
    {
        f(ii) = 2.0 * static_cast<double>(gen()) / 4294967296.0 - 1.0;
    }
    return f;
}

} // end anonymous namespace


TEST_CASE("rbf_interpolate: reproduces its data at the centers (smoothing = 0)")
{
    const Eigen::MatrixXd P = scattered_points(2, 8, 1);
    const Eigen::VectorXd f = scattered_values(8, 2);
    for ( RBFKernel kernel : kAllKernels )
    {
        RBFScheme scheme;
        scheme.kernel = kernel;
        scheme.degree = 1;
        const Eigen::VectorXd at_centers = rbf_interpolate(f, P, P, scheme);
        for ( int ii = 0; ii < f.size(); ++ii )
        {
            CHECK(at_centers(ii) == doctest::Approx(f(ii)).epsilon(1e-8));
        }
    }
}


TEST_CASE("rbf_interpolate: reproduces polynomials of the tail degree exactly")
{
    const Eigen::MatrixXd P = scattered_points(2, 10, 3);
    const Eigen::MatrixXd Q = 0.2 * Eigen::MatrixXd::Ones(2, 4) + scattered_points(2, 4, 4);

    // Affine data with a degree-1 tail: the interpolant IS the affine function.
    const Eigen::Vector2d beta(1.5, -2.0);
    const double alpha = 0.7;
    const Eigen::VectorXd f_affine = ( beta.transpose() * P ).transpose().array() + alpha;
    for ( RBFKernel kernel : kAllKernels )
    {
        RBFScheme scheme;
        scheme.kernel = kernel;
        scheme.degree = 1;
        const Eigen::VectorXd s = rbf_interpolate(f_affine, P, Q, scheme);
        for ( int qq = 0; qq < Q.cols(); ++qq )
        {
            const double expected = alpha + beta.dot(Q.col(qq));
            CHECK(s(qq) == doctest::Approx(expected).epsilon(1e-9));
        }
    }

    // Constant data with a degree-0 tail (where the kernel allows it).
    const Eigen::VectorXd f_const = Eigen::VectorXd::Constant(P.cols(), 3.25);
    for ( RBFKernel kernel : {RBFKernel::gaussian, RBFKernel::multiquadric,
                              RBFKernel::inverse_multiquadric, RBFKernel::linear} )
    {
        RBFScheme scheme;
        scheme.kernel = kernel;
        scheme.degree = 0;
        const Eigen::VectorXd s = rbf_interpolate(f_const, P, Q, scheme);
        for ( int qq = 0; qq < Q.cols(); ++qq )
        {
            CHECK(s(qq) == doctest::Approx(3.25).epsilon(1e-10));
        }
    }
}


TEST_CASE("rbf_interpolate: degenerate data")
{
    RBFScheme scheme; // gaussian, degree 1

    // Single center: constant interpolant.
    Eigen::MatrixXd P1(2, 1);
    P1.col(0) = Eigen::Vector2d(0.3, 0.4);
    Eigen::VectorXd f1(1);
    f1(0) = 5.0;
    const Eigen::MatrixXd Q = scattered_points(2, 3, 7);
    Eigen::VectorXd s = rbf_interpolate(f1, P1, Q, scheme);
    for ( int qq = 0; qq < Q.cols(); ++qq )
    {
        CHECK(s(qq) == doctest::Approx(5.0));
    }

    // All centers coincident: constant at the mean value.
    Eigen::MatrixXd P2 = P1.col(0).replicate(1, 3);
    Eigen::VectorXd f2(3);
    f2 << 1.0, 2.0, 6.0;
    s = rbf_interpolate(f2, P2, Q, scheme);
    CHECK(s(0) == doctest::Approx(3.0));

    // Two centers with a degree-1 request: tail auto-lowers to constant, and
    // the data is still reproduced at the centers — including for TPS, whose
    // minimum degree cannot be satisfied with two points.
    Eigen::MatrixXd P3(2, 2);
    P3.col(0) = Eigen::Vector2d(0.2, 0.2);
    P3.col(1) = Eigen::Vector2d(0.7, 0.5);
    Eigen::VectorXd f3(2);
    f3 << 1.0, -2.0;
    for ( RBFKernel kernel : kAllKernels )
    {
        RBFScheme sc;
        sc.kernel = kernel;
        sc.degree = 1;
        const Eigen::VectorXd at_centers = rbf_interpolate(f3, P3, P3, sc);
        CHECK(at_centers(0) == doctest::Approx(f3(0)).epsilon(1e-9));
        CHECK(at_centers(1) == doctest::Approx(f3(1)).epsilon(1e-9));
    }
}


TEST_CASE("rbf_interpolate: smoothing trades center exactness for regression")
{
    const Eigen::MatrixXd P = scattered_points(2, 12, 11);
    const Eigen::VectorXd f = scattered_values(12, 12);

    RBFScheme scheme;
    scheme.degree = 0;

    // Small smoothing: no longer interpolates exactly.
    scheme.smoothing = 1e-2;
    const Eigen::VectorXd s_small = rbf_interpolate(f, P, P, scheme);
    CHECK(( s_small - f ).norm() > 1e-6);

    // Large smoothing with a constant tail: approaches the least-squares
    // constant fit (kept at 1e6, not larger — the bordered system's condition
    // number grows with lambda, so the extreme limit is numerically absurd).
    scheme.smoothing = 1e6;
    const Eigen::VectorXd s_huge = rbf_interpolate(f, P, P, scheme);
    for ( int ii = 0; ii < f.size(); ++ii )
    {
        CHECK(std::abs(s_huge(ii) - f.mean()) < 1e-4);
    }
}


TEST_CASE("rbf_interpolate: evaluation at a center is finite for phi(0) = 0 kernels")
{
    // u^2 log u and -u are 0/negative at u = 0; the guard must keep values finite.
    const Eigen::MatrixXd P = scattered_points(2, 6, 21);
    const Eigen::VectorXd f = scattered_values(6, 22);
    for ( RBFKernel kernel : {RBFKernel::thin_plate_spline, RBFKernel::linear, RBFKernel::cubic} )
    {
        RBFScheme scheme;
        scheme.kernel = kernel;
        scheme.degree = 1;
        const Eigen::VectorXd s = rbf_interpolate(f, P, P.col(0), scheme);
        CHECK(std::isfinite(s(0)));
        CHECK(s(0) == doctest::Approx(f(0)).epsilon(1e-8));
    }
}


TEST_CASE("rbf_interpolate and validate: argument checking")
{
    const Eigen::MatrixXd P = scattered_points(2, 4, 31);
    const Eigen::VectorXd f = scattered_values(4, 32);

    RBFScheme bad;
    bad.shape = 0.0;
    CHECK_THROWS_AS(validate(bad), std::invalid_argument);
    bad = RBFScheme{};
    bad.smoothing = -1.0;
    CHECK_THROWS_AS(validate(bad), std::invalid_argument);
    bad = RBFScheme{};
    bad.degree = 2;
    CHECK_THROWS_AS(validate(bad), std::invalid_argument);

    // Degree below the kernel minimum is rejected up front.
    bad = RBFScheme{};
    bad.kernel = RBFKernel::thin_plate_spline;
    bad.degree = 0;
    CHECK_THROWS_AS(validate(bad), std::invalid_argument);
    bad.kernel = RBFKernel::linear;
    bad.degree = -1;
    CHECK_THROWS_AS(validate(bad), std::invalid_argument);

    RBFScheme ok;
    CHECK_THROWS_AS(rbf_interpolate(f, Eigen::MatrixXd(2, 0), P, ok), std::invalid_argument);
    Eigen::VectorXd f_bad(3);
    f_bad.setZero();
    CHECK_THROWS_AS(rbf_interpolate(f_bad, P, P, ok), std::invalid_argument);
    CHECK_THROWS_AS(rbf_interpolate(f, P, Eigen::MatrixXd(3, 1), ok), std::invalid_argument);
}
