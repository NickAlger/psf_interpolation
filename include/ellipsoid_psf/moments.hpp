#pragma once
// SPDX-License-Identifier: MIT
// Part of ellipsoid_psf — https://github.com/NickAlger/ellipsoid_psf

/// @file
/// @brief Moment-data hygiene: eigenvalue clamping for covariance data
/// corrupted by numerical error.
///
/// ellipsoid_psf requires strictly positive definite covariances, both per-sample
/// (add_batch) and per-vertex (set_moment_fields). For the vertex field this
/// is sufficient everywhere: the minimum eigenvalue is concave over symmetric
/// matrices, so a CG1 (barycentric) interpolation of SPD vertex matrices
/// satisfies lambda_min(Sigma(x)) >= min over the cell's vertices of
/// lambda_min — validation at the vertices guarantees positive definiteness
/// at every target point.
///
/// Validation alone cannot protect against the *near*-singular case: a
/// covariance with a tiny positive eigenvalue passes the check, but
/// Sigma(x)^{-1/2} (the whitened_affine frame) and det Sigma(x) (the
/// volume_det scaling) amplify enormously there. Clamping with a genuinely
/// meaningful floor is therefore a modelling decision, not just hygiene: a
/// good default is the square of the local mesh spacing (an impulse response
/// cannot be resolved below the mesh scale anyway) — this mirrors the
/// clamping used in the localpsf reference implementation. Regions where the
/// moments are garbage because the operator is uninformative there should be
/// handled through the V field (V ~ 0 drives the kernel to zero continuously
/// under the volume scalings) with Sigma clamped to something harmless.

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>

namespace ellipsoid_psf {

/// Symmetrizes Sigma and clamps its eigenvalues to at least `floor` (> 0
/// required, so the result passes ellipsoid_psf's positive-definiteness validation).
/// If no eigenvalue is below the floor, the symmetrized input is returned
/// unchanged (no eigenvector reconstruction, no rounding).
inline Eigen::MatrixXd clamp_spd( const Eigen::Ref<const Eigen::MatrixXd>& Sigma, double floor )
{
    if ( !( floor > 0.0 ) )
    {
        throw std::invalid_argument("ellipsoid_psf::clamp_spd: floor must be > 0 (clamped covariances must "
                                    "be strictly positive definite)");
    }
    if ( Sigma.rows() != Sigma.cols() )
    {
        throw std::invalid_argument("ellipsoid_psf::clamp_spd: Sigma must be square");
    }
    Eigen::MatrixXd S = 0.5 * ( Sigma + Sigma.transpose() );
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(S);
    if ( es.info() != Eigen::Success )
    {
        throw std::runtime_error("ellipsoid_psf::clamp_spd: eigendecomposition failed");
    }
    if ( es.eigenvalues().minCoeff() >= floor )
    {
        return S;
    }
    const Eigen::VectorXd lam = es.eigenvalues().cwiseMax(floor);
    return es.eigenvectors() * lam.asDiagonal() * es.eigenvectors().transpose();
}

/// clamp_spd applied to a covariance field in the flat layout of
/// ImpulseResponseField::set_moment_fields: Sigma_flat has shape
/// (dim*dim, n) with column v = vec(Sigma_v), and floors gives the per-entry
/// eigenvalue floor (all > 0). Returns the cleaned field plus the indices of
/// the entries that were actually modified (an eigenvalue was below its
/// floor); unmodified columns are only symmetrized.
inline std::pair<Eigen::MatrixXd, std::vector<int>>
clamp_spd_field( const Eigen::Ref<const Eigen::MatrixXd>& Sigma_flat,
                 int dim,
                 const Eigen::Ref<const Eigen::VectorXd>& floors )
{
    const int n = static_cast<int>(Sigma_flat.cols());
    if ( dim < 1 || Sigma_flat.rows() != dim * dim )
    {
        throw std::invalid_argument("ellipsoid_psf::clamp_spd_field: Sigma_flat must have dim*dim rows");
    }
    if ( floors.size() != n )
    {
        throw std::invalid_argument("ellipsoid_psf::clamp_spd_field: floors must have one entry per column");
    }

    Eigen::MatrixXd out(dim * dim, n);
    std::vector<int> modified;
    for ( int v = 0; v < n; ++v )
    {
        if ( !( floors(v) > 0.0 ) )
        {
            throw std::invalid_argument("ellipsoid_psf::clamp_spd_field: floors must be > 0 (entry "
                                        + std::to_string(v) + ")");
        }
        const Eigen::Map<const Eigen::MatrixXd> S(Sigma_flat.col(v).data(), dim, dim);
        const Eigen::MatrixXd S_sym = 0.5 * ( S + S.transpose() );
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(S_sym);
        if ( es.info() != Eigen::Success )
        {
            throw std::runtime_error("ellipsoid_psf::clamp_spd_field: eigendecomposition failed at entry "
                                     + std::to_string(v));
        }
        if ( es.eigenvalues().minCoeff() >= floors(v) )
        {
            out.col(v) = Eigen::Map<const Eigen::VectorXd>(S_sym.data(), dim * dim);
        }
        else
        {
            const Eigen::VectorXd lam = es.eigenvalues().cwiseMax(floors(v));
            const Eigen::MatrixXd C = es.eigenvectors() * lam.asDiagonal() * es.eigenvectors().transpose();
            out.col(v) = Eigen::Map<const Eigen::VectorXd>(C.data(), dim * dim);
            modified.push_back(v);
        }
    }
    return std::make_pair(std::move(out), std::move(modified));
}

/// clamp_spd_field with one floor for every entry.
inline std::pair<Eigen::MatrixXd, std::vector<int>>
clamp_spd_field( const Eigen::Ref<const Eigen::MatrixXd>& Sigma_flat, int dim, double floor )
{
    return clamp_spd_field(Sigma_flat, dim,
                           Eigen::VectorXd::Constant(Sigma_flat.cols(), floor));
}

} // end namespace ellipsoid_psf
