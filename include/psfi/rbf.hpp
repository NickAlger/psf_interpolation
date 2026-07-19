#pragma once
// SPDX-License-Identifier: MIT
// Part of psfi — https://github.com/NickAlger/psf_interpolation

/// @file
/// @brief Radial basis function interpolation of scattered data, with a
/// polynomial tail and optional ridge smoothing — the layer that combines
/// per-neighbor kernel predictions into one value.
///
/// The interpolant through data {(p_i, f_i)} is
///
///     s(p) = sum_i w_i phi(|p - p_i|) + poly(p),
///
/// with the weights and polynomial coefficients solving the standard bordered
/// system [Phi + lambda I, P; P^T, 0] [w; c] = [f; 0]. Distances are scaled
/// locally: phi is evaluated at u = shape * r / r0, where r0 is the diameter
/// (bounding-box diagonal) of the interpolation point set, recomputed per
/// call — so the effective kernel width tracks the local sample spacing and
/// `shape` is the C_RBF parameter of the paper (eq. 5.4). With smoothing = 0
/// the interpolant reproduces its data at the centers; smoothing > 0 trades
/// that exactness for robustness to noise (there is deliberately no special
/// case snapping evaluation points onto centers).

#include <cmath>
#include <stdexcept>
#include <string>

#include <Eigen/Dense>

namespace psfi {

/// Radial kernel phi(u), evaluated at the locally scaled distance
/// u = shape * r / r0. Sign conventions follow scipy.interpolate.RBFInterpolator
/// (the interpolant is invariant to the overall kernel sign).
enum class RBFKernel
{
    gaussian,             ///< exp(-u^2 / 2) (positive definite; the paper's kernel)
    multiquadric,         ///< -sqrt(1 + u^2) (conditionally PD, needs degree >= 0)
    inverse_multiquadric, ///< 1 / sqrt(1 + u^2) (positive definite)
    linear,               ///< -u (conditionally PD, needs degree >= 0)
    thin_plate_spline,    ///< u^2 log u, with phi(0) = 0 (conditionally PD, needs degree >= 1)
    cubic,                ///< u^3 (conditionally PD, needs degree >= 1)
};

/// Configuration of the RBF interpolation.
struct RBFScheme
{
    RBFKernel kernel  = RBFKernel::gaussian;
    double    shape   = 3.0; ///< C_RBF: phi is evaluated at u = shape * r / r0 (local diameter r0)
    int       degree  = 1;   ///< polynomial tail: -1 none, 0 constant, 1 linear
    /// Ridge parameter lambda added to the kernel diagonal. Note: when there
    /// are no more centers than polynomial-tail coefficients (k <= dim + 1
    /// for degree 1) the system degenerates to exact polynomial
    /// interpolation and smoothing has no effect.
    double    smoothing = 0.0;
};

/// Smallest polynomial-tail degree for which interpolation with this kernel
/// is guaranteed solvable (conditional positive definiteness order minus one).
inline int rbf_min_degree( RBFKernel kernel )
{
    switch ( kernel )
    {
        case RBFKernel::gaussian:             return -1;
        case RBFKernel::inverse_multiquadric: return -1;
        case RBFKernel::multiquadric:         return 0;
        case RBFKernel::linear:               return 0;
        case RBFKernel::thin_plate_spline:    return 1;
        case RBFKernel::cubic:                return 1;
    }
    return 1;
}

/// Throws std::invalid_argument if the scheme's parameters are invalid or the
/// requested degree is below the kernel's minimum.
inline void validate( const RBFScheme& scheme )
{
    if ( !( scheme.shape > 0.0 ) )
    {
        throw std::invalid_argument("psfi::validate(RBFScheme): shape must be > 0");
    }
    if ( !( scheme.smoothing >= 0.0 ) )
    {
        throw std::invalid_argument("psfi::validate(RBFScheme): smoothing must be >= 0");
    }
    if ( scheme.degree < -1 || scheme.degree > 1 )
    {
        throw std::invalid_argument("psfi::validate(RBFScheme): degree must be -1, 0, or 1");
    }
    if ( scheme.degree < rbf_min_degree(scheme.kernel) )
    {
        throw std::invalid_argument("psfi::validate(RBFScheme): degree "
                                    + std::to_string(scheme.degree)
                                    + " is below this kernel's minimum degree "
                                    + std::to_string(rbf_min_degree(scheme.kernel)));
    }
}

namespace detail {

inline double rbf_phi( RBFKernel kernel, double u )
{
    switch ( kernel )
    {
        case RBFKernel::gaussian:             return std::exp(-0.5 * u * u);
        case RBFKernel::multiquadric:         return -std::sqrt(1.0 + u * u);
        case RBFKernel::inverse_multiquadric: return 1.0 / std::sqrt(1.0 + u * u);
        case RBFKernel::linear:               return -u;
        case RBFKernel::thin_plate_spline:    return ( u == 0.0 ) ? 0.0 : u * u * std::log(u);
        case RBFKernel::cubic:                return u * u * u;
    }
    return 0.0;
}

} // end namespace detail

/// Evaluates the RBF interpolant of the data {(centers.col(i), values(i))}
/// at each evaluation point (column of eval_points). One linear solve per
/// call, shared by all evaluation points.
///
/// Degenerate data is handled without ceremony: a single center (or all
/// centers coincident, r0 = 0) gives a constant interpolant, and the
/// polynomial-tail degree is lowered automatically when there are fewer
/// centers than tail coefficients (a degree-1 tail needs at least dim + 1
/// centers) — possibly below the kernel's minimum degree, in which case the
/// least-squares solve still returns a usable answer even though the classical
/// solvability guarantee no longer applies.
inline Eigen::VectorXd rbf_interpolate( const Eigen::Ref<const Eigen::VectorXd>& values,      // (k)
                                        const Eigen::Ref<const Eigen::MatrixXd>& centers,     // (dim, k)
                                        const Eigen::Ref<const Eigen::MatrixXd>& eval_points, // (dim, m)
                                        const RBFScheme& scheme )
{
    validate(scheme);
    const int k = static_cast<int>(centers.cols());
    const int d = static_cast<int>(centers.rows());
    const int m = static_cast<int>(eval_points.cols());
    if ( k < 1 )
    {
        throw std::invalid_argument("psfi::rbf_interpolate: need at least one center");
    }
    if ( values.size() != k )
    {
        throw std::invalid_argument("psfi::rbf_interpolate: values must have one entry per center");
    }
    if ( eval_points.rows() != d )
    {
        throw std::invalid_argument("psfi::rbf_interpolate: eval_points and centers must have the "
                                    "same number of rows (spatial dimension)");
    }

    // Local length scale: bounding-box diagonal of the centers.
    const double r0 = ( centers.rowwise().maxCoeff() - centers.rowwise().minCoeff() ).norm();
    if ( k == 1 || r0 == 0.0 )
    {
        // One center (or all coincident): constant interpolant.
        return Eigen::VectorXd::Constant(m, values.mean());
    }
    const double inv_scale = scheme.shape / r0;

    // Polynomial tail, auto-lowered when there are too few centers. Monomials
    // are evaluated on centered/scaled coordinates for conditioning (the
    // interpolant is invariant to this affine change of tail basis).
    int degree = scheme.degree;
    if ( degree == 1 && k < d + 1 )
    {
        degree = 0;
    }
    const int np = ( degree < 0 ) ? 0 : ( ( degree == 0 ) ? 1 : d + 1 );
    const Eigen::VectorXd center_mean = centers.rowwise().mean();
    auto tail = [&]( const Eigen::Ref<const Eigen::VectorXd>& p ) -> Eigen::RowVectorXd
    {
        Eigen::RowVectorXd row(np);
        if ( np >= 1 )
        {
            row(0) = 1.0;
        }
        for ( int jj = 1; jj < np; ++jj )
        {
            row(jj) = ( p(jj - 1) - center_mean(jj - 1) ) * inv_scale;
        }
        return row;
    };

    // Bordered system [Phi + lambda I, P; P^T, 0] [w; c] = [f; 0].
    const int n = k + np;
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(n, n);
    for ( int jj = 0; jj < k; ++jj )
    {
        for ( int ii = 0; ii < k; ++ii )
        {
            const double u = ( centers.col(ii) - centers.col(jj) ).norm() * inv_scale;
            A(ii, jj) = detail::rbf_phi(scheme.kernel, u);
        }
        A(jj, jj) += scheme.smoothing;
    }
    for ( int ii = 0; ii < k; ++ii )
    {
        A.block(ii, k, 1, np) = tail(centers.col(ii));
        A.block(k, ii, np, 1) = A.block(ii, k, 1, np).transpose();
    }
    Eigen::VectorXd b = Eigen::VectorXd::Zero(n);
    b.head(k) = values;

    const Eigen::VectorXd coeffs = A.colPivHouseholderQr().solve(b);

    Eigen::VectorXd out(m);
    for ( int qq = 0; qq < m; ++qq )
    {
        double s = 0.0;
        for ( int ii = 0; ii < k; ++ii )
        {
            const double u = ( centers.col(ii) - eval_points.col(qq) ).norm() * inv_scale;
            s += coeffs(ii) * detail::rbf_phi(scheme.kernel, u);
        }
        if ( np > 0 )
        {
            s += ( tail(eval_points.col(qq)) * coeffs.tail(np) ).value();
        }
        out(qq) = s;
    }
    return out;
}

} // end namespace psfi
