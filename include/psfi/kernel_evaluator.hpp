#pragma once
// SPDX-License-Identifier: MIT
// Part of psfi — https://github.com/NickAlger/psf_interpolation

/// @file
/// @brief KernelEvaluator: the complete kernel approximation
/// Phi(y, x) ~ RBF-combine of per-neighbor predictions, single entries and
/// threaded blocks.
///
/// Cols-only mode uses one ImpulseResponseField (impulse responses are
/// "columns" of the kernel, sampled in x). Symmetric mode additionally takes
/// a row field probed with the transpose operator: since
/// Phi(y, x) = Phi^T(x, y), the row field's predictions at (x, y) predict the
/// same kernel entry from samples near y. Both prediction sets are pooled in
/// displacement coordinates (sample point minus its query point, so the
/// evaluation point is the origin), near-duplicate centers are averaged, and
/// one RBF interpolation combines them — following the paper's construction.
///
/// There is no special case snapping evaluations onto known columns: with
/// smoothing = 0 the RBF interpolant reproduces its centers, so
/// Phi(y, x_i) is exact at sample points automatically, and with
/// smoothing > 0 that exactness is deliberately traded for noise robustness.

#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "etree/detail/parallel_for.hpp"

#include "psfi/config.hpp"
#include "psfi/impulse_response_field.hpp"
#include "psfi/rbf.hpp"

namespace psfi {

/// Evaluates the approximate integral kernel anywhere, by RBF interpolation
/// of the per-neighbor predictions of one (cols-only) or two (symmetric)
/// impulse response fields.
class KernelEvaluator
{
public:
    /// col_field is required; row_field may be null (cols-only) or a field
    /// probed with the transpose operator (symmetric mode; pass the same
    /// field twice for a symmetric operator probed once). Both fields are
    /// validated against `config` here, and the RBF scheme is validated too,
    /// so construction succeeds iff evaluation can run.
    KernelEvaluator( std::shared_ptr<const ImpulseResponseField> col_field,
                     std::shared_ptr<const ImpulseResponseField> row_field = nullptr,
                     EvalConfig config = {},
                     RBFScheme  rbf = {},
                     double     duplicate_tol = 1e-7 )
        : col_field_(std::move(col_field)), row_field_(std::move(row_field)),
          config_(config), rbf_(rbf), duplicate_tol_(duplicate_tol)
    {
        if ( !col_field_ )
        {
            throw std::invalid_argument("psfi::KernelEvaluator: col_field must not be null");
        }
        validate(rbf_);
        col_field_->validate(config_);
        if ( row_field_ )
        {
            if ( row_field_->dim() != col_field_->dim() )
            {
                throw std::invalid_argument("psfi::KernelEvaluator: col_field and row_field must "
                                            "have the same spatial dimension");
            }
            row_field_->validate(config_);
        }
        if ( !( duplicate_tol_ >= 0.0 ) )
        {
            throw std::invalid_argument("psfi::KernelEvaluator: duplicate_tol must be >= 0");
        }
    }

    const std::shared_ptr<const ImpulseResponseField>& col_field() const { return col_field_; }
    const std::shared_ptr<const ImpulseResponseField>& row_field() const { return row_field_; }
    const EvalConfig& config() const  { return config_; }
    const RBFScheme&  rbf() const     { return rbf_; }
    double duplicate_tol() const      { return duplicate_tol_; }
    bool   symmetric() const          { return static_cast<bool>(row_field_); }
    int    dim() const                { return col_field_->dim(); }

    /// The approximate kernel entry Phi(y, x); zero where there are no
    /// predictions (no samples in reach, or x uninformed). Thread-safe.
    double operator()( const Eigen::Ref<const Eigen::VectorXd>& y,
                       const Eigen::Ref<const Eigen::VectorXd>& x ) const
    {
        // Pool predictions in displacement coordinates (evaluation point at
        // the origin): forward centers x_i - x, adjoint centers x_j - y.
        std::vector<Eigen::VectorXd> centers;
        std::vector<double>          values;

        for ( const Prediction& p : col_field_->predictions(y, x, config_) )
        {
            centers.push_back(p.point - x);
            values.push_back(p.value);
        }
        if ( row_field_ )
        {
            const int num_fwd = static_cast<int>(centers.size());
            for ( const Prediction& p : row_field_->predictions(x, y, config_) )
            {
                // Average near-duplicates into their forward partner (the
                // x = y case makes the two sets literally coincide).
                const Eigen::VectorXd c = p.point - y;
                bool merged = false;
                for ( int jj = 0; jj < num_fwd; ++jj )
                {
                    if ( ( centers[jj] - c ).norm() <= duplicate_tol_ )
                    {
                        centers[jj] = 0.5 * ( centers[jj] + c );
                        values[jj]  = 0.5 * ( values[jj] + p.value );
                        merged = true;
                        break;
                    }
                }
                if ( !merged )
                {
                    centers.push_back(c);
                    values.push_back(p.value);
                }
            }
        }

        const int k = static_cast<int>(centers.size());
        if ( k == 0 )
        {
            return 0.0;
        }
        // Far-field short circuit: when every prediction is gated to zero the
        // interpolant is identically zero (also under smoothing) — skip the
        // RBF solve entirely. This is the dominant case for off-diagonal
        // blocks in compressed-matrix assembly.
        bool all_zero = true;
        for ( double v : values )
        {
            if ( v != 0.0 )
            {
                all_zero = false;
                break;
            }
        }
        if ( all_zero )
        {
            return 0.0;
        }
        Eigen::MatrixXd C(dim(), k);
        Eigen::VectorXd f(k);
        for ( int jj = 0; jj < k; ++jj )
        {
            C.col(jj) = centers[jj];
            f(jj) = values[jj];
        }
        return rbf_interpolate(f, C, Eigen::MatrixXd::Zero(dim(), 1), rbf_)(0);
    }

    /// The block of kernel entries [ Phi(yy.col(ii), xx.col(jj)) ]_{ii,jj},
    /// shape (num_y, num_x), evaluated in parallel (num_threads <= 0 uses all
    /// hardware threads).
    Eigen::MatrixXd block( const Eigen::Ref<const Eigen::MatrixXd>& yy, // (dim, num_y)
                           const Eigen::Ref<const Eigen::MatrixXd>& xx, // (dim, num_x)
                           int num_threads = 0 ) const
    {
        if ( yy.rows() != dim() || xx.rows() != dim() )
        {
            throw std::invalid_argument("psfi::KernelEvaluator::block: yy and xx must have dim rows");
        }
        const int ny = static_cast<int>(yy.cols());
        const int nx = static_cast<int>(xx.cols());
        Eigen::MatrixXd out(ny, nx);
        etree::detail::parallel_for(0, static_cast<std::ptrdiff_t>(ny) * nx,
            [&]( std::ptrdiff_t aa, std::ptrdiff_t bb )
            {
                for ( std::ptrdiff_t ind = aa; ind < bb; ++ind )
                {
                    const int ii = static_cast<int>(ind % ny);
                    const int jj = static_cast<int>(ind / ny);
                    out(ii, jj) = ( *this )(yy.col(ii), xx.col(jj));
                }
            }, num_threads);
        return out;
    }

private:
    std::shared_ptr<const ImpulseResponseField> col_field_;
    std::shared_ptr<const ImpulseResponseField> row_field_;
    EvalConfig config_;
    RBFScheme  rbf_;
    double     duplicate_tol_;
};

} // end namespace psfi
