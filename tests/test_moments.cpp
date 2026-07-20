// SPDX-License-Identifier: MIT
#include "doctest/doctest.h"

#include <stdexcept>
#include <vector>

#include <Eigen/Dense>

#include "ellipsoid_psf/moments.hpp"

using namespace ellipsoid_psf;

namespace {

Eigen::VectorXd flat( const Eigen::MatrixXd& S )
{
    return Eigen::Map<const Eigen::VectorXd>(S.data(), S.size());
}

Eigen::MatrixXd unflat( const Eigen::Ref<const Eigen::VectorXd>& col, int d )
{
    return Eigen::Map<const Eigen::MatrixXd>(col.data(), d, d);
}

} // end anonymous namespace


TEST_CASE("clamp_spd: single matrix")
{
    // Indefinite: the negative eigenvalue is raised to the floor, the
    // positive one untouched.
    const Eigen::MatrixXd S = Eigen::Vector2d(1.0, -0.1).asDiagonal();
    const Eigen::MatrixXd C = clamp_spd(S, 0.01);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(C);
    CHECK(es.eigenvalues()(0) == doctest::Approx(0.01).epsilon(1e-12));
    CHECK(es.eigenvalues()(1) == doctest::Approx(1.0).epsilon(1e-12));

    // Already comfortably SPD: returned bit-for-bit (symmetrized input, no
    // eigen reconstruction).
    const Eigen::MatrixXd G = ( Eigen::Matrix2d() << 0.5, 0.1, 0.1, 0.3 ).finished();
    CHECK(( clamp_spd(G, 0.01) - G ).cwiseAbs().maxCoeff() == 0.0);

    // Asymmetric input is symmetrized first.
    const Eigen::MatrixXd A = ( Eigen::Matrix2d() << 0.5, 0.2, 0.0, 0.3 ).finished();
    const Eigen::MatrixXd CA = clamp_spd(A, 0.01);
    CHECK(( CA - CA.transpose() ).cwiseAbs().maxCoeff() == 0.0);
    CHECK(CA(0, 1) == doctest::Approx(0.1));

    CHECK_THROWS_AS(clamp_spd(S, 0.0), std::invalid_argument);
    CHECK_THROWS_AS(clamp_spd(S, -1.0), std::invalid_argument);
    CHECK_THROWS_AS(clamp_spd(Eigen::MatrixXd(2, 3), 0.01), std::invalid_argument);
}


TEST_CASE("clamp_spd_field: flat covariance stacks")
{
    const int d = 2;
    Eigen::MatrixXd field(d * d, 3);
    const Eigen::MatrixXd good = ( Eigen::Matrix2d() << 0.5, 0.1, 0.1, 0.3 ).finished();
    const Eigen::MatrixXd bad  = Eigen::Vector2d(0.2, -1e-9).asDiagonal(); // numerical-error negative
    const Eigen::MatrixXd tiny = Eigen::Vector2d(0.2, 1e-12).asDiagonal(); // positive but below floor
    field.col(0) = flat(good);
    field.col(1) = flat(bad);
    field.col(2) = flat(tiny);

    const auto result = clamp_spd_field(field, d, 1e-4);
    const Eigen::MatrixXd& cleaned = result.first;
    CHECK(result.second == std::vector<int>{1, 2});
    CHECK(( unflat(cleaned.col(0), d) - good ).cwiseAbs().maxCoeff() == 0.0);
    for ( int v = 0; v < 3; ++v )
    {
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(unflat(cleaned.col(v), d));
        CHECK(es.eigenvalues().minCoeff() >= 1e-4 * ( 1.0 - 1e-12 ));
    }

    // Per-entry floors: a floor vector that only entry 2 violates.
    Eigen::VectorXd floors(3);
    floors << 1e-12, 1e-12, 1e-6;
    const auto result2 = clamp_spd_field(field, d, floors);
    CHECK(result2.second == std::vector<int>{1, 2}); // entry 1's -1e-9 < 1e-12 still

    CHECK_THROWS_AS(clamp_spd_field(field, 3, 1e-4), std::invalid_argument);   // wrong dim
    Eigen::VectorXd short_floors(2);
    short_floors.setOnes();
    CHECK_THROWS_AS(clamp_spd_field(field, d, short_floors), std::invalid_argument);
    floors(1) = 0.0;
    CHECK_THROWS_AS(clamp_spd_field(field, d, floors), std::invalid_argument);
}
