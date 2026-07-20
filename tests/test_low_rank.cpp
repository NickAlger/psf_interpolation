// SPDX-License-Identifier: MIT
#include "doctest/doctest.h"

#include <cmath>
#include <functional>
#include <random>
#include <set>
#include <stdexcept>

#include <Eigen/Dense>

#include "ellipsoid_psf/low_rank.hpp"

using namespace ellipsoid_psf;

namespace {

// Deterministic filler in [-1, 1] (mt19937 raw output; no std distributions).
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

// A matrix with prescribed singular values (orthonormal factors from QR).
Eigen::MatrixXd with_singular_values( int m, int n, const Eigen::VectorXd& sigma,
                                      unsigned int seed )
{
    const int p = static_cast<int>(sigma.size());
    Eigen::HouseholderQR<Eigen::MatrixXd> qu(pseudo_random(m, p, seed));
    Eigen::HouseholderQR<Eigen::MatrixXd> qv(pseudo_random(n, p, seed + 1));
    const Eigen::MatrixXd Qu = qu.householderQ() * Eigen::MatrixXd::Identity(m, p);
    const Eigen::MatrixXd Qv = qv.householderQ() * Eigen::MatrixXd::Identity(n, p);
    return Qu * sigma.asDiagonal() * Qv.transpose();
}

double rel_err( const Eigen::MatrixXd& A, const LowRank& F )
{
    return ( A - F.to_dense() ).norm() / A.norm();
}

std::function<Eigen::VectorXd(int)> row_of( const Eigen::MatrixXd& A )
{
    return [&A]( int ii ) -> Eigen::VectorXd { return A.row(ii); };
}

std::function<Eigen::VectorXd(int)> col_of( const Eigen::MatrixXd& A )
{
    return [&A]( int jj ) -> Eigen::VectorXd { return A.col(jj); };
}

} // end anonymous namespace


TEST_CASE("truncated_svd: exact rank recovery")
{
    const Eigen::MatrixXd A = pseudo_random(12, 3, 1) * pseudo_random(9, 3, 2).transpose();
    const LowRank F = truncated_svd(A, 1e-12);
    CHECK(F.rank() == 3);
    CHECK(F.U.rows() == 12);
    CHECK(F.V.rows() == 9);
    CHECK(rel_err(A, F) <= 1e-12);
    // V orthonormal, scale folded into U.
    CHECK(( F.V.transpose() * F.V - Eigen::MatrixXd::Identity(3, 3) ).norm() <= 1e-12);
}


TEST_CASE("truncated_svd: Frobenius truncation rule")
{
    Eigen::VectorXd sigma(5);
    sigma << 1.0, 0.5, 0.1, 0.05, 0.01;
    const Eigen::MatrixXd A = with_singular_values(11, 8, sigma, 3);

    // Manually: dropping {0.01, 0.05, 0.1} gives tail 0.11225 <= 0.12 * ||A||_F
    // = 0.13485, and also dropping 0.5 overshoots — so rank 2.
    const double rtol = 0.12;
    const LowRank F = truncated_svd(A, rtol);
    CHECK(F.rank() == 2);
    CHECK(rel_err(A, F) <= rtol);

    // The next-larger tolerance boundary: rank 3 at a slightly tighter rtol.
    CHECK(truncated_svd(A, 0.09).rank() == 3);

    // max_rank caps whatever the rule selects.
    CHECK(truncated_svd(A, 0.0, 2).rank() == 2);

    CHECK_THROWS_AS(truncated_svd(A, -1.0), std::invalid_argument);
}


TEST_CASE("truncated_svd: degenerate inputs")
{
    CHECK(truncated_svd(Eigen::MatrixXd::Zero(4, 6), 1e-12).rank() == 0);
    const LowRank E = truncated_svd(Eigen::MatrixXd(5, 0), 0.0);
    CHECK(E.rank() == 0);
    CHECK(E.U.rows() == 5);
    CHECK(E.V.rows() == 0);
}


TEST_CASE("recompress: redundant factors shrink to the true rank")
{
    const Eigen::MatrixXd U0 = pseudo_random(10, 3, 4);
    const Eigen::MatrixXd V0 = pseudo_random(7, 3, 5);
    // Duplicate the factorization: rank-6 factors of a rank-3 matrix.
    Eigen::MatrixXd U1(10, 6);
    U1 << U0, U0;
    Eigen::MatrixXd V1(7, 6);
    V1 << 0.5 * V0, 0.5 * V0;
    const LowRank F = recompress(LowRank{ U1, V1 }, 1e-12);
    CHECK(F.rank() == 3);
    CHECK(( U0 * V0.transpose() - F.to_dense() ).norm() / ( U0 * V0.transpose() ).norm() <= 1e-12);

    // max_rank cap, and rank-0 passthrough.
    CHECK(recompress(LowRank{ U1, V1 }, 0.0, 2).rank() == 2);
    const LowRank Z = recompress(LowRank{ Eigen::MatrixXd(10, 0), Eigen::MatrixXd(7, 0) }, 1e-12);
    CHECK(Z.rank() == 0);
    CHECK(Z.U.rows() == 10);
}


TEST_CASE("aca: exact low-rank matrix is recovered")
{
    const Eigen::MatrixXd A = pseudo_random(30, 4, 6) * pseudo_random(25, 4, 7).transpose();
    const double rtol = 1e-10;
    const ACAResult r = aca(row_of(A), col_of(A), 30, 25, rtol);
    CHECK(r.converged);
    CHECK(!r.hit_max_rank);
    CHECK(rel_err(A, r.factors) <= rtol);
    CHECK(r.factors.rank() == 4); // recompression strips the tiny probe crosses
    CHECK(r.sampled_rank >= 4);
    CHECK(static_cast<int>(r.sampled_rows.size()) == r.sampled_rank);
    CHECK(static_cast<int>(r.sampled_cols.size()) == r.sampled_rank);

    // Sampled rows are distinct and in range; columns in range.
    std::set<int> rows(r.sampled_rows.begin(), r.sampled_rows.end());
    CHECK(static_cast<int>(rows.size()) == r.sampled_rank);
    for ( int ii : r.sampled_rows )
    {
        CHECK(( 0 <= ii && ii < 30 ));
    }
    for ( int jj : r.sampled_cols )
    {
        CHECK(( 0 <= jj && jj < 25 ));
    }
}


TEST_CASE("aca: rtol = 0 runs to exact recovery")
{
    const Eigen::MatrixXd A = pseudo_random(8, 6, 8);
    const ACAResult r = aca(row_of(A), col_of(A), 8, 6, 0.0);
    CHECK(r.converged);
    CHECK(!r.hit_max_rank);
    CHECK(r.sampled_rank == 6);
    CHECK(rel_err(A, r.factors) <= 1e-10);
}


TEST_CASE("aca: random restarts find disconnected support")
{
    // The Pine-Island failure mode: a block-diagonal matrix whose two pieces
    // share no rows or columns. A single partial-pivot chain starting in one
    // piece never produces a pivot in the other; only the random probe rows
    // reach it.
    const int n = 40;
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(n, n);
    A.topLeftCorner(20, 20)     = pseudo_random(20, 2, 9)  * pseudo_random(20, 2, 10).transpose();
    A.bottomRightCorner(20, 20) = pseudo_random(20, 2, 11) * pseudo_random(20, 2, 12).transpose();

    const double rtol = 1e-8;
    const ACAResult r = aca(row_of(A), col_of(A), n, n, rtol);
    CHECK(r.converged);
    CHECK(rel_err(A, r.factors) <= rtol);
    CHECK(r.factors.rank() == 4); // both pieces captured

    // Both halves of the row space were actually sampled.
    bool sampled_first = false;
    bool sampled_second = false;
    for ( int ii : r.sampled_rows )
    {
        ( ii < 20 ? sampled_first : sampled_second ) = true;
    }
    CHECK(sampled_first);
    CHECK(sampled_second);
}


TEST_CASE("aca: a binding rank cap is reported, never silent")
{
    const Eigen::MatrixXd A = pseudo_random(20, 5, 13) * pseudo_random(15, 5, 14).transpose();
    ACAOptions options;
    options.max_rank = 3;
    const ACAResult r = aca(row_of(A), col_of(A), 20, 15, 1e-12, options);
    CHECK(r.hit_max_rank);
    CHECK(!r.converged);
    CHECK(r.sampled_rank == 3);
    CHECK(r.factors.rank() <= 3);
}


TEST_CASE("aca: zero matrix and degenerate shapes")
{
    const Eigen::MatrixXd Z = Eigen::MatrixXd::Zero(7, 5);
    const ACAResult rz = aca(row_of(Z), col_of(Z), 7, 5, 1e-8);
    CHECK(rz.converged);
    CHECK(rz.factors.rank() == 0);
    CHECK(rz.relerr_estimate == 0.0);

    const Eigen::MatrixXd R1 = pseudo_random(1, 6, 15);
    const ACAResult r1 = aca(row_of(R1), col_of(R1), 1, 6, 1e-10);
    CHECK(r1.converged);
    CHECK(rel_err(R1, r1.factors) <= 1e-10);

    const Eigen::MatrixXd C1 = pseudo_random(6, 1, 16);
    const ACAResult c1 = aca(row_of(C1), col_of(C1), 6, 1, 1e-10);
    CHECK(c1.converged);
    CHECK(rel_err(C1, c1.factors) <= 1e-10);

    const ACAResult e = aca(row_of(Z), col_of(Z), 0, 5, 1e-8);
    CHECK(e.converged);
    CHECK(e.factors.rank() == 0);
}


TEST_CASE("aca: input validation")
{
    const Eigen::MatrixXd A = pseudo_random(4, 4, 17);
    CHECK_THROWS_AS(aca(nullptr, col_of(A), 4, 4, 1e-8), std::invalid_argument);
    CHECK_THROWS_AS(aca(row_of(A), nullptr, 4, 4, 1e-8), std::invalid_argument);
    CHECK_THROWS_AS(aca(row_of(A), col_of(A), 4, 4, -1.0), std::invalid_argument);
    CHECK_THROWS_AS(aca(row_of(A), col_of(A), -1, 4, 1e-8), std::invalid_argument);

    ACAOptions bad;
    bad.aca_safety_factor = 0.0;
    CHECK_THROWS_AS(aca(row_of(A), col_of(A), 4, 4, 1e-8, bad), std::invalid_argument);
    bad = ACAOptions{};
    bad.recompress_safety_factor = 1.5;
    CHECK_THROWS_AS(aca(row_of(A), col_of(A), 4, 4, 1e-8, bad), std::invalid_argument);
    bad = ACAOptions{};
    bad.required_consecutive_successes = 0;
    CHECK_THROWS_AS(aca(row_of(A), col_of(A), 4, 4, 1e-8, bad), std::invalid_argument);

    // A slice of the wrong length is caught, not silently mis-shaped.
    auto short_row = []( int ) -> Eigen::VectorXd { return Eigen::VectorXd::Zero(2); };
    CHECK_THROWS_AS(aca(short_row, col_of(A), 4, 4, 1e-8), std::invalid_argument);
}


TEST_CASE("aca: deterministic for a fixed seed")
{
    const Eigen::MatrixXd A = pseudo_random(18, 5, 18) * pseudo_random(14, 5, 19).transpose();
    const ACAResult a = aca(row_of(A), col_of(A), 18, 14, 1e-9);
    const ACAResult b = aca(row_of(A), col_of(A), 18, 14, 1e-9);
    CHECK(a.sampled_rows == b.sampled_rows);
    CHECK(a.sampled_cols == b.sampled_cols);
    CHECK(( a.factors.U - b.factors.U ).cwiseAbs().maxCoeff() == 0.0);
    CHECK(( a.factors.V - b.factors.V ).cwiseAbs().maxCoeff() == 0.0);

    ACAOptions other;
    other.seed = 42;
    const ACAResult c = aca(row_of(A), col_of(A), 18, 14, 1e-9, other);
    CHECK(c.converged);
    CHECK(rel_err(A, c.factors) <= 1e-9);
}


TEST_CASE("randomized_svd: exact low-rank matrix")
{
    const Eigen::MatrixXd A = pseudo_random(25, 3, 20) * pseudo_random(18, 3, 21).transpose();
    auto apply  = [&A]( const Eigen::Ref<const Eigen::MatrixXd>& X ) -> Eigen::MatrixXd
    { return A * X; };
    auto applyT = [&A]( const Eigen::Ref<const Eigen::MatrixXd>& Y ) -> Eigen::MatrixXd
    { return A.transpose() * Y; };

    const LowRank F = randomized_svd(apply, applyT, 25, 18, 3);
    CHECK(F.rank() == 3);
    CHECK(rel_err(A, F) <= 1e-10);
    CHECK(( F.V.transpose() * F.V - Eigen::MatrixXd::Identity(3, 3) ).norm() <= 1e-12);
}


TEST_CASE("randomized_svd: near-best approximation of a decaying spectrum")
{
    Eigen::VectorXd sigma(12);
    for ( int kk = 0; kk < 12; ++kk )
    {
        sigma(kk) = std::pow(2.0, -kk);
    }
    const Eigen::MatrixXd A = with_singular_values(30, 22, sigma, 22);
    auto apply  = [&A]( const Eigen::Ref<const Eigen::MatrixXd>& X ) -> Eigen::MatrixXd
    { return A * X; };
    auto applyT = [&A]( const Eigen::Ref<const Eigen::MatrixXd>& Y ) -> Eigen::MatrixXd
    { return A.transpose() * Y; };

    const int rank = 8;
    RSVDOptions options;
    options.power_iterations = 2;
    const LowRank F = randomized_svd(apply, applyT, 30, 22, rank, options);
    const double best = rel_err(A, truncated_svd(A, 0.0, rank));
    CHECK(rel_err(A, F) <= 10.0 * best);

    // The optional rtol truncation trims below max_rank.
    RSVDOptions loose;
    loose.rtol = 0.3; // tail rule: keep sigma down to ~2^-2
    const LowRank G = randomized_svd(apply, applyT, 30, 22, rank, loose);
    CHECK(G.rank() < rank);
    CHECK(rel_err(A, G) <= 0.3 * 1.5); // sampled, so allow slack over the exact rule

    // Deterministic for a fixed seed.
    const LowRank F2 = randomized_svd(apply, applyT, 30, 22, rank, options);
    CHECK(( F.U - F2.U ).cwiseAbs().maxCoeff() == 0.0);
    CHECK(( F.V - F2.V ).cwiseAbs().maxCoeff() == 0.0);
}


TEST_CASE("randomized_svd: input validation")
{
    auto ok = []( const Eigen::Ref<const Eigen::MatrixXd>& X ) -> Eigen::MatrixXd { return X; };
    CHECK_THROWS_AS(randomized_svd(nullptr, ok, 4, 4, 2), std::invalid_argument);
    CHECK_THROWS_AS(randomized_svd(ok, nullptr, 4, 4, 2), std::invalid_argument);
    CHECK_THROWS_AS(randomized_svd(ok, ok, 4, 4, 0), std::invalid_argument);
    CHECK_THROWS_AS(randomized_svd(ok, ok, -1, 4, 2), std::invalid_argument);
    RSVDOptions bad;
    bad.oversampling = -1;
    CHECK_THROWS_AS(randomized_svd(ok, ok, 4, 4, 2, bad), std::invalid_argument);

    // Mis-shaped functor output is caught.
    auto wrong = []( const Eigen::Ref<const Eigen::MatrixXd>& X ) -> Eigen::MatrixXd
    { return Eigen::MatrixXd::Zero(2, X.cols()); };
    auto square = []( const Eigen::Ref<const Eigen::MatrixXd>& X ) -> Eigen::MatrixXd
    { return Eigen::MatrixXd::Zero(4, X.cols()); };
    CHECK_THROWS_AS(randomized_svd(wrong, square, 4, 4, 2), std::invalid_argument);
}
