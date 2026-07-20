#pragma once
// SPDX-License-Identifier: MIT
// Part of ellipsoid_psf — https://github.com/NickAlger/ellipsoid_psf

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
///
/// `values` carries one scalar per center today; a multi-right-hand-side
/// overload (one factorization, many value columns) is the planned
/// vector-field extension — see dev/VECTOR_TENSOR_EXTENSION.md.

#include <cmath>
#include <stdexcept>
#include <string>

#include <Eigen/Dense>

namespace ellipsoid_psf {

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
        throw std::invalid_argument("ellipsoid_psf::validate(RBFScheme): shape must be > 0");
    }
    if ( !( scheme.smoothing >= 0.0 ) )
    {
        throw std::invalid_argument("ellipsoid_psf::validate(RBFScheme): smoothing must be >= 0");
    }
    if ( scheme.degree < -1 || scheme.degree > 1 )
    {
        throw std::invalid_argument("ellipsoid_psf::validate(RBFScheme): degree must be -1, 0, or 1");
    }
    if ( scheme.degree < rbf_min_degree(scheme.kernel) )
    {
        throw std::invalid_argument("ellipsoid_psf::validate(RBFScheme): degree "
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

// The bordered system shared by rbf_interpolate and rbf_functional. Both
// must keep identical conventions (length scale, tail lowering, assembly),
// so those live here once.
struct RBFSystem
{
    bool   constant = false; ///< single or fully coincident centers: constant interpolant
    int    k = 0;            ///< number of centers
    int    np = 0;           ///< tail coefficients
    double inv_scale = 1.0;  ///< shape / r0
    Eigen::VectorXd center_mean;
    Eigen::MatrixXd A;       ///< (k+np, k+np) bordered matrix [Phi + lambda I, P; P^T, 0]
};

// Tail-monomial row at p (centered/scaled coordinates for conditioning).
inline Eigen::RowVectorXd rbf_tail_row( const RBFSystem& S,
                                        const Eigen::Ref<const Eigen::VectorXd>& p )
{
    Eigen::RowVectorXd row(S.np);
    if ( S.np >= 1 )
    {
        row(0) = 1.0;
    }
    for ( int jj = 1; jj < S.np; ++jj )
    {
        row(jj) = ( p(jj - 1) - S.center_mean(jj - 1) ) * S.inv_scale;
    }
    return row;
}

// Assumes validate(scheme) has run and centers has at least one column.
inline RBFSystem build_rbf_system( const Eigen::Ref<const Eigen::MatrixXd>& centers,
                                   const RBFScheme& scheme )
{
    RBFSystem S;
    S.k = static_cast<int>(centers.cols());
    const int d = static_cast<int>(centers.rows());

    // Local length scale: bounding-box diagonal of the centers.
    const double r0 = ( centers.rowwise().maxCoeff() - centers.rowwise().minCoeff() ).norm();
    if ( S.k == 1 || r0 == 0.0 )
    {
        S.constant = true;
        return S;
    }
    S.inv_scale = scheme.shape / r0;

    // Polynomial tail, auto-lowered when there are too few centers.
    int degree = scheme.degree;
    if ( degree == 1 && S.k < d + 1 )
    {
        degree = 0;
    }
    S.np = ( degree < 0 ) ? 0 : ( ( degree == 0 ) ? 1 : d + 1 );
    S.center_mean = centers.rowwise().mean();

    const int n = S.k + S.np;
    S.A = Eigen::MatrixXd::Zero(n, n);
    for ( int jj = 0; jj < S.k; ++jj )
    {
        for ( int ii = 0; ii < S.k; ++ii )
        {
            const double u = ( centers.col(ii) - centers.col(jj) ).norm() * S.inv_scale;
            S.A(ii, jj) = rbf_phi(scheme.kernel, u);
        }
        S.A(jj, jj) += scheme.smoothing;
    }
    for ( int ii = 0; ii < S.k; ++ii )
    {
        S.A.block(ii, S.k, 1, S.np) = rbf_tail_row(S, centers.col(ii));
        S.A.block(S.k, ii, S.np, 1) = S.A.block(ii, S.k, 1, S.np).transpose();
    }
    return S;
}

// The basis vector g at p: value of the interpolant is g . [w; c].
inline Eigen::VectorXd rbf_basis_at( const RBFSystem& S,
                                     const Eigen::Ref<const Eigen::MatrixXd>& centers,
                                     const RBFScheme& scheme,
                                     const Eigen::Ref<const Eigen::VectorXd>& p )
{
    Eigen::VectorXd g(S.k + S.np);
    for ( int ii = 0; ii < S.k; ++ii )
    {
        const double u = ( centers.col(ii) - p ).norm() * S.inv_scale;
        g(ii) = rbf_phi(scheme.kernel, u);
    }
    if ( S.np > 0 )
    {
        g.tail(S.np) = rbf_tail_row(S, p).transpose();
    }
    return g;
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
        throw std::invalid_argument("ellipsoid_psf::rbf_interpolate: need at least one center");
    }
    if ( values.size() != k )
    {
        throw std::invalid_argument("ellipsoid_psf::rbf_interpolate: values must have one entry per center");
    }
    if ( eval_points.rows() != d )
    {
        throw std::invalid_argument("ellipsoid_psf::rbf_interpolate: eval_points and centers must have the "
                                    "same number of rows (spatial dimension)");
    }

    const detail::RBFSystem S = detail::build_rbf_system(centers, scheme);
    if ( S.constant )
    {
        // One center (or all coincident): constant interpolant.
        return Eigen::VectorXd::Constant(m, values.mean());
    }

    // Bordered system [Phi + lambda I, P; P^T, 0] [w; c] = [f; 0].
    Eigen::VectorXd b = Eigen::VectorXd::Zero(S.k + S.np);
    b.head(k) = values;
    const Eigen::VectorXd coeffs = S.A.colPivHouseholderQr().solve(b);

    Eigen::VectorXd out(m);
    for ( int qq = 0; qq < m; ++qq )
    {
        out(qq) = detail::rbf_basis_at(S, centers, scheme, eval_points.col(qq)).dot(coeffs);
    }
    return out;
}

/// The evaluation functional of the RBF interpolant at eval_point: the
/// weights lambda with s(eval_point) = lambda . values for EVERY data
/// vector, since interpolation is linear in the data. Same conventions and
/// degeneracy handling as rbf_interpolate (equal results up to rounding —
/// the bordered matrix is symmetric, so the functional is the transposed
/// solve). This is the piece that makes fixed-source kernel evaluation
/// cheap: with the centers fixed, one solve gives lambda, and every further
/// evaluation point costs a dot product.
inline Eigen::VectorXd rbf_functional( const Eigen::Ref<const Eigen::MatrixXd>& centers,     // (dim, k)
                                       const Eigen::Ref<const Eigen::VectorXd>& eval_point,  // (dim)
                                       const RBFScheme& scheme )
{
    validate(scheme);
    const int k = static_cast<int>(centers.cols());
    if ( k < 1 )
    {
        throw std::invalid_argument("ellipsoid_psf::rbf_functional: need at least one center");
    }
    if ( eval_point.size() != centers.rows() )
    {
        throw std::invalid_argument("ellipsoid_psf::rbf_functional: eval_point and centers must have the "
                                    "same spatial dimension");
    }

    const detail::RBFSystem S = detail::build_rbf_system(centers, scheme);
    if ( S.constant )
    {
        return Eigen::VectorXd::Constant(k, 1.0 / k); // the mean, as in rbf_interpolate
    }
    const Eigen::VectorXd g = detail::rbf_basis_at(S, centers, scheme, eval_point);
    // value = g . A^{-1} [f; 0] = (A^{-T} g) . [f; 0]; A is symmetric.
    return S.A.colPivHouseholderQr().solve(g).head(k);
}

} // end namespace ellipsoid_psf
