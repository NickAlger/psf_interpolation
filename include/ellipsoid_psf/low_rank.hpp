#pragma once
// SPDX-License-Identifier: MIT
// Part of ellipsoid_psf — https://github.com/NickAlger/ellipsoid_psf

/// @file
/// @brief Generic low-rank matrix tools: Frobenius-truncated SVD, adaptive
/// cross approximation (ACA) with random-restart verification, factor
/// recompression, and a randomized SVD driven by matvec functors.
///
/// This layer speaks plain matrix language — rows, columns, entries — and
/// knows nothing about kernels. Kernel-facing code (the compressed formats
/// built from a KernelEvaluator) speaks source/target instead; the two meet
/// in one adapter whose fixed mapping is: target axis = matrix rows, source
/// axis = matrix columns, matching KernelEvaluator::block(yy, xx).
///
/// Every tolerance here is a RELATIVE FROBENIUS tolerance: an approximation
/// B of A is accepted when ||A - B||_F <= rtol * ||A||_F (for ACA, with the
/// norms replaced by their sampled estimates). rtol = 0 requests exact
/// recovery up to rounding.
///
/// All randomness is generated from an explicitly seeded, exactly specified
/// generator (std::mt19937 raw output only — no std distributions, whose
/// results vary across standard libraries), so results are bit-for-bit
/// reproducible across platforms for a given seed.

#include <algorithm>
#include <cmath>
#include <functional>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>

namespace ellipsoid_psf {

/// A rank-r factorization A ~ U V^T with U (num_rows, r) and V (num_cols, r).
/// Scale (e.g. singular values) is folded into U; V has orthonormal columns
/// when produced by truncated_svd / recompress / randomized_svd (not by aca
/// without recompression).
struct LowRank
{
    Eigen::MatrixXd U; ///< left factor, shape (num_rows, rank)
    Eigen::MatrixXd V; ///< right factor, shape (num_cols, rank)

    int rank() const { return static_cast<int>(U.cols()); }
    Eigen::MatrixXd to_dense() const { return U * V.transpose(); }
};

namespace detail {

// Thin Householder QR: A (m, n) = Q (m, k) R (k, n) with k = min(m, n).
inline std::pair<Eigen::MatrixXd, Eigen::MatrixXd> thin_qr( const Eigen::MatrixXd& A )
{
    const int m = static_cast<int>(A.rows());
    const int n = static_cast<int>(A.cols());
    const int k = std::min(m, n);
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(A);
    Eigen::MatrixXd Q = qr.householderQ() * Eigen::MatrixXd::Identity(m, k);
    Eigen::MatrixXd R = qr.matrixQR().topRows(k).triangularView<Eigen::Upper>();
    return std::make_pair(std::move(Q), std::move(R));
}

// Uniform integer in [0, n): mt19937 raw output with modulo. The (negligible)
// modulo bias is a fair price for exactly reproducible cross-platform output.
inline int uniform_index( std::mt19937& gen, int n )
{
    return static_cast<int>(gen() % static_cast<unsigned int>(n));
}

// Standard normal via Box-Muller on mt19937 raw output (no std distributions;
// see the file header note on cross-platform reproducibility).
inline double standard_normal( std::mt19937& gen )
{
    const double scale = 1.0 / 4294967296.0; // 2^-32
    const double u1 = ( static_cast<double>(gen()) + 0.5 ) * scale; // in (0, 1)
    const double u2 = ( static_cast<double>(gen()) + 0.5 ) * scale;
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586477 * u2);
}

} // end namespace detail

/// The best rank-r approximation of A in the Frobenius norm, with r chosen as
/// the smallest rank whose discarded singular-value tail satisfies
/// sqrt(sum_{i>r} sigma_i^2) <= rtol * ||A||_F, then capped at max_rank
/// (max_rank < 0 means no cap). Singular values are folded into U.
inline LowRank truncated_svd( const Eigen::Ref<const Eigen::MatrixXd>& A,
                              double rtol,
                              int    max_rank = -1 )
{
    if ( !( rtol >= 0.0 ) )
    {
        throw std::invalid_argument("ellipsoid_psf::truncated_svd: rtol must be >= 0");
    }
    const int m = static_cast<int>(A.rows());
    const int n = static_cast<int>(A.cols());
    if ( m == 0 || n == 0 )
    {
        return LowRank{ Eigen::MatrixXd(m, 0), Eigen::MatrixXd(n, 0) };
    }
    Eigen::BDCSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    const Eigen::VectorXd& sigma = svd.singularValues();
    const int p = static_cast<int>(sigma.size());

    // Smallest r with tail(r) <= rtol * ||A||_F, via the suffix sums of
    // sigma_i^2 (tail(p) = 0, so r = p always qualifies).
    const double tail_budget_squared = rtol * rtol * sigma.squaredNorm();
    int r = p;
    double tail_squared = 0.0;
    while ( r > 0 )
    {
        const double next = tail_squared + sigma(r - 1) * sigma(r - 1);
        if ( next > tail_budget_squared )
        {
            break;
        }
        tail_squared = next;
        --r;
    }
    if ( max_rank >= 0 )
    {
        r = std::min(r, max_rank);
    }
    return LowRank{ svd.matrixU().leftCols(r) * sigma.head(r).asDiagonal(),
                    svd.matrixV().leftCols(r) };
}

/// Recompresses a (possibly redundant) factorization U V^T to the smallest
/// rank meeting the tolerance: thin QR of each factor, Frobenius-truncated
/// SVD of the small core, factors reassembled. The result satisfies
/// ||U V^T - result||_F <= rtol * ||U V^T||_F.
inline LowRank recompress( const LowRank& F, double rtol, int max_rank = -1 )
{
    if ( F.rank() == 0 || F.U.rows() == 0 || F.V.rows() == 0 )
    {
        return LowRank{ Eigen::MatrixXd(F.U.rows(), 0), Eigen::MatrixXd(F.V.rows(), 0) };
    }
    auto qr_u = detail::thin_qr(F.U);
    auto qr_v = detail::thin_qr(F.V);
    const LowRank core = truncated_svd(qr_u.second * qr_v.second.transpose(), rtol, max_rank);
    return LowRank{ qr_u.first * core.U, qr_v.first * core.V };
}

/// Options for aca(). The stopping tolerance is aca_safety_factor * rtol and
/// the recompression tolerance recompress_safety_factor * rtol, so the two
/// error contributions sum to roughly rtol.
struct ACAOptions
{
    int    max_rank = -1;                    ///< sampled-rank cap; < 0 means min(num_rows, num_cols)
    double aca_safety_factor = 0.25;         ///< fraction of rtol spent on the cross sampling
    double recompress_safety_factor = 0.75;  ///< fraction of rtol spent on recompression
    int    required_consecutive_successes = 10; ///< random probe rows that must pass before stopping
    bool   recompress = true;                ///< recompress the sampled factors before returning
    unsigned int seed = 0;                   ///< seed for the restart row draws
};

/// Result of aca(): the factors plus construction diagnostics. Never let a
/// rank cap bind silently — check hit_max_rank (the GPSF ancestor of this
/// code once lost accuracy for weeks that way).
struct ACAResult
{
    LowRank factors;
    std::vector<int> sampled_rows; ///< row index of each accepted cross, in order
    std::vector<int> sampled_cols; ///< column index of each accepted cross, in order
    int    sampled_rank = 0;       ///< rank before recompression (= sampled_rows.size())
    bool   converged = false;      ///< stopping rule passed, or the matrix was recovered exactly
    bool   hit_max_rank = false;   ///< stopped by the max_rank cap instead
    double relerr_estimate = 0.0;  ///< last cross norm over the running Frobenius estimate
};

/// Low-rank approximation by adaptive cross approximation with partial
/// pivoting (Ballani & Kressner, "Matrices with hierarchical low-rank
/// structures", Algorithm 4) plus random-restart verification — the variant
/// proven out in the GPSF/ymir ice-sheet work, where it is known as ACA+.
///
/// Only sampled rows and columns of the matrix are ever evaluated: get_row(i)
/// must return row i (length num_cols) and get_col(j) column j (length
/// num_rows) of the SAME fixed matrix. Crosses are added at magnitude-largest
/// residual pivots; after each cross small enough to pass the stopping test,
/// the next row is drawn at random and the test must pass at
/// required_consecutive_successes such rows in a row before the sampling
/// stops. The restarts are what protects against support that partial
/// pivoting alone can miss — e.g. a block whose columns live on two barely
/// connected land masses, where a single pivot chain can stay trapped on one
/// piece (observed near Pine Island Bay in the ymir work).
///
/// rtol = 0 disables the stopping rule: sampling runs until the matrix is
/// recovered exactly or max_rank binds. Deterministic for a given seed.
inline ACAResult aca( const std::function<Eigen::VectorXd(int)>& get_row,
                      const std::function<Eigen::VectorXd(int)>& get_col,
                      int num_rows,
                      int num_cols,
                      double rtol,
                      const ACAOptions& options = {} )
{
    if ( !get_row || !get_col )
    {
        throw std::invalid_argument("ellipsoid_psf::aca: get_row and get_col must be callable");
    }
    if ( num_rows < 0 || num_cols < 0 )
    {
        throw std::invalid_argument("ellipsoid_psf::aca: num_rows and num_cols must be >= 0");
    }
    if ( !( rtol >= 0.0 ) )
    {
        throw std::invalid_argument("ellipsoid_psf::aca: rtol must be >= 0");
    }
    if ( !( options.aca_safety_factor > 0.0 && options.aca_safety_factor <= 1.0 )
         || !( options.recompress_safety_factor > 0.0 && options.recompress_safety_factor <= 1.0 ) )
    {
        throw std::invalid_argument("ellipsoid_psf::aca: safety factors must be in (0, 1]");
    }
    if ( options.required_consecutive_successes < 1 )
    {
        throw std::invalid_argument("ellipsoid_psf::aca: required_consecutive_successes must be >= 1");
    }

    ACAResult result;
    result.factors = LowRank{ Eigen::MatrixXd(num_rows, 0), Eigen::MatrixXd(num_cols, 0) };
    if ( num_rows == 0 || num_cols == 0 )
    {
        result.converged = true;
        return result;
    }

    const int full_rank = std::min(num_rows, num_cols);
    const int cap = ( options.max_rank < 0 ) ? full_rank : std::min(options.max_rank, full_rank);
    const double stop_tol = options.aca_safety_factor * rtol;
    const double stop_tol_squared = stop_tol * stop_tol;

    std::mt19937 gen(options.seed);
    std::vector<bool> candidate(num_rows, true);
    int num_candidates = num_rows;
    auto take_random_candidate = [&]() -> int
    {
        int draw = detail::uniform_index(gen, num_candidates);
        for ( int ii = 0; ii < num_rows; ++ii )
        {
            if ( candidate[ii] && draw-- == 0 )
            {
                return ii;
            }
        }
        throw std::logic_error("ellipsoid_psf::aca: internal error, no candidate rows left");
    };

    std::vector<Eigen::VectorXd> uu; // residual columns (crosses' left vectors)
    std::vector<Eigen::VectorXd> vv; // scaled residual rows (right vectors)
    auto fetch_slice = [&]( const std::function<Eigen::VectorXd(int)>& get, int index,
                            int expected, const char* what ) -> Eigen::VectorXd
    {
        Eigen::VectorXd s = get(index);
        if ( s.size() != expected )
        {
            throw std::invalid_argument(std::string("ellipsoid_psf::aca: ") + what + "(" + std::to_string(index)
                                        + ") returned length " + std::to_string(s.size())
                                        + ", expected " + std::to_string(expected));
        }
        return s;
    };

    double norm_Ak_squared = 0.0; // running estimate of ||sum of crosses||_F^2
    int num_successes = 0;
    int ii = take_random_candidate();
    while ( static_cast<int>(uu.size()) < cap )
    {
        // Residual row ii: A(ii, :) minus the crosses so far.
        Eigen::VectorXd R_row = fetch_slice(get_row, ii, num_cols, "get_row");
        for ( size_t ll = 0; ll < uu.size(); ++ll )
        {
            R_row -= uu[ll](ii) * vv[ll];
        }
        candidate[ii] = false;
        --num_candidates;

        int jj = 0;
        const double delta_abs = R_row.cwiseAbs().maxCoeff(&jj); // column pivot by magnitude

        if ( delta_abs == 0.0 )
        {
            // This row's residual is exactly zero: the strongest possible
            // probe success. No cross to add; move to a fresh random row.
            ++num_successes;
            if ( num_successes >= options.required_consecutive_successes || num_candidates == 0 )
            {
                // Either the probes passed, or every row has been visited —
                // and each visited row has zero residual (sampled rows are
                // zeroed by their own cross), so the recovery is exact.
                result.converged = true;
                result.relerr_estimate = 0.0;
                break;
            }
            ii = take_random_candidate();
            continue;
        }

        Eigen::VectorXd R_col = fetch_slice(get_col, jj, num_rows, "get_col");
        for ( size_t ll = 0; ll < uu.size(); ++ll )
        {
            R_col -= vv[ll](jj) * uu[ll];
        }
        Eigen::VectorXd u = std::move(R_col);
        Eigen::VectorXd v = R_row / R_row(jj);

        // ||A_k||_F^2 update: cross terms carry a factor 2 (the ymir/GPSF
        // ancestor omitted it, biasing the estimate low).
        double cross_terms = 0.0;
        for ( size_t ll = 0; ll < uu.size(); ++ll )
        {
            cross_terms += u.dot(uu[ll]) * v.dot(vv[ll]);
        }
        const double uv_norm_squared = u.squaredNorm() * v.squaredNorm();
        norm_Ak_squared = std::max(0.0, norm_Ak_squared + 2.0 * cross_terms + uv_norm_squared);

        result.sampled_rows.push_back(ii);
        result.sampled_cols.push_back(jj);
        uu.push_back(std::move(u));
        vv.push_back(std::move(v));
        result.relerr_estimate = ( norm_Ak_squared > 0.0 )
            ? std::sqrt(uv_norm_squared / norm_Ak_squared) : 0.0;

        if ( static_cast<int>(uu.size()) == cap )
        {
            // Reaching min(num_rows, num_cols) crosses zeroes every row (or
            // every column): exact recovery. A smaller user cap binding is
            // the failure mode worth surfacing.
            result.converged = ( cap == full_rank );
            result.hit_max_rank = !result.converged;
            if ( result.converged )
            {
                result.relerr_estimate = 0.0;
            }
            break;
        }
        if ( num_candidates == 0 )
        {
            result.converged = true; // all rows visited and zeroed: exact
            result.relerr_estimate = 0.0;
            break;
        }

        if ( uv_norm_squared < stop_tol_squared * norm_Ak_squared )
        {
            ++num_successes;
            if ( num_successes >= options.required_consecutive_successes )
            {
                result.converged = true;
                break;
            }
            ii = take_random_candidate(); // random restart: re-probe elsewhere
        }
        else
        {
            num_successes = 0;
            // Next row pivot: magnitude-largest residual entry of the sampled
            // column, among unvisited rows.
            int best = -1;
            double best_abs = -1.0;
            const Eigen::VectorXd& last_u = uu.back();
            for ( int rr = 0; rr < num_rows; ++rr )
            {
                if ( candidate[rr] && std::abs(last_u(rr)) > best_abs )
                {
                    best_abs = std::abs(last_u(rr));
                    best = rr;
                }
            }
            ii = best;
        }
    }

    result.sampled_rank = static_cast<int>(uu.size());
    Eigen::MatrixXd U(num_rows, result.sampled_rank);
    Eigen::MatrixXd V(num_cols, result.sampled_rank);
    for ( int ll = 0; ll < result.sampled_rank; ++ll )
    {
        U.col(ll) = uu[ll];
        V.col(ll) = vv[ll];
    }
    result.factors = LowRank{ std::move(U), std::move(V) };
    if ( options.recompress && result.sampled_rank > 0 )
    {
        result.factors = ellipsoid_psf::recompress(result.factors, options.recompress_safety_factor * rtol);
    }
    return result;
}

/// Options for randomized_svd().
struct RSVDOptions
{
    int    oversampling = 10;     ///< extra sampled directions beyond max_rank
    int    power_iterations = 1;  ///< subspace (power) iterations for sharper spectra
    double rtol = 0.0;            ///< optional Frobenius truncation of the result (0 keeps max_rank)
    unsigned int seed = 0;        ///< seed for the Gaussian test matrix
};

/// Randomized SVD (Halko, Martinsson & Tropp) of a matrix available only
/// through its action: apply(X) = A X on (num_cols, k) blocks and
/// apply_transpose(Y) = A^T Y on (num_rows, k) blocks. Returns roughly the
/// best rank-max_rank approximation; accuracy improves with oversampling and
/// power_iterations. Deterministic for a given seed. This is the piece that
/// turns a block low rank operator into a global low rank one using only its
/// (cheap) applies.
inline LowRank randomized_svd(
    const std::function<Eigen::MatrixXd(const Eigen::Ref<const Eigen::MatrixXd>&)>& apply,
    const std::function<Eigen::MatrixXd(const Eigen::Ref<const Eigen::MatrixXd>&)>& apply_transpose,
    int num_rows,
    int num_cols,
    int max_rank,
    const RSVDOptions& options = {} )
{
    if ( !apply || !apply_transpose )
    {
        throw std::invalid_argument("ellipsoid_psf::randomized_svd: apply and apply_transpose must be callable");
    }
    if ( num_rows < 0 || num_cols < 0 )
    {
        throw std::invalid_argument("ellipsoid_psf::randomized_svd: num_rows and num_cols must be >= 0");
    }
    if ( max_rank < 1 )
    {
        throw std::invalid_argument("ellipsoid_psf::randomized_svd: max_rank must be >= 1");
    }
    if ( options.oversampling < 0 || options.power_iterations < 0 || !( options.rtol >= 0.0 ) )
    {
        throw std::invalid_argument("ellipsoid_psf::randomized_svd: oversampling and power_iterations must be "
                                    ">= 0 and rtol >= 0");
    }
    if ( num_rows == 0 || num_cols == 0 )
    {
        return LowRank{ Eigen::MatrixXd(num_rows, 0), Eigen::MatrixXd(num_cols, 0) };
    }

    const int full_rank = std::min(num_rows, num_cols);
    const int num_samples = std::min(max_rank + options.oversampling, full_rank);

    std::mt19937 gen(options.seed);
    Eigen::MatrixXd Omega(num_cols, num_samples);
    for ( int jj = 0; jj < num_samples; ++jj )
    {
        for ( int rr = 0; rr < num_cols; ++rr )
        {
            Omega(rr, jj) = detail::standard_normal(gen);
        }
    }

    auto checked = [&]( const char* what, Eigen::MatrixXd&& M, int rows, int cols ) -> Eigen::MatrixXd
    {
        if ( M.rows() != rows || M.cols() != cols )
        {
            throw std::invalid_argument(std::string("ellipsoid_psf::randomized_svd: ") + what + " returned shape ("
                                        + std::to_string(M.rows()) + ", " + std::to_string(M.cols())
                                        + "), expected (" + std::to_string(rows) + ", "
                                        + std::to_string(cols) + ")");
        }
        return std::move(M);
    };

    Eigen::MatrixXd Q = detail::thin_qr(
        checked("apply", apply(Omega), num_rows, num_samples)).first;
    for ( int it = 0; it < options.power_iterations; ++it )
    {
        const Eigen::MatrixXd Z = detail::thin_qr(
            checked("apply_transpose", apply_transpose(Q), num_cols, num_samples)).first;
        Q = detail::thin_qr(checked("apply", apply(Z), num_rows, num_samples)).first;
    }

    // A ~ Q (Q^T A) = Q B^T with B = A^T Q; SVD the small B^T.
    const Eigen::MatrixXd B = checked("apply_transpose", apply_transpose(Q), num_cols, num_samples);
    const LowRank core = truncated_svd(B.transpose(), options.rtol, max_rank);
    return LowRank{ Q * core.U, core.V };
}

} // end namespace ellipsoid_psf
