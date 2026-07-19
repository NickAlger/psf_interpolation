#pragma once
// SPDX-License-Identifier: MIT
// Part of psfi — https://github.com/NickAlger/psf_interpolation

/// @file
/// @brief Evaluation configuration: the three independent axes (frame map,
/// scaling, support gate) plus numerical parameters.
///
/// Each nearby sample point x_i predicts the kernel value at (y, x) from its
/// stored impulse response phi_i by
///
///     f_i = s_i * phi_i(T_i(y)),
///
/// where the frame map T_i identifies the neighborhood of x with the
/// neighborhood of x_i, and the scaling s_i is a scalar correction. The
/// predictions {(x_i, f_i)} are then combined by scattered-data interpolation
/// at x (the RBF layer). Equations (4.7)-(4.10) of the paper are the first
/// three frame maps crossed with the first two scalings; `whitened_affine`
/// additionally deforms each impulse response so its support ellipsoid maps
/// onto the ellipsoid at x.

namespace psfi {

/// Frame map T_i: how a stored impulse response at x_i is transported to the
/// query point x. Here mu(x), Sigma(x) are the moment fields evaluated at x
/// (CG1 interpolation of the vertex fields), and mu_i, Sigma_i are the
/// per-sample moments.
enum class Frame
{
    identity,         ///< T_i(y) = y — no transport (paper eq. 4.7)
    translation,      ///< T_i(y) = y - x + x_i (4.8, local translation invariance)
    mean_translation, ///< T_i(y) = y - mu(x) + mu_i (4.9/4.10, local mean displacement invariance)
    /// T_i(y) = mu_i + Sigma_i^{1/2} W(x) (y - mu(x)) — maps the ellipsoid at
    /// x onto the ellipsoid at x_i (symmetric PSD square roots). W is the
    /// inverse-square-root field: the CG1 interpolation of the per-vertex
    /// Sigma_v^{-1/2}, which equals Sigma(x)^{-1/2} at the vertices, is SPD
    /// everywhere by convexity, and needs no per-evaluation eigensolve.
    whitened_affine,
};

/// Scalar correction s_i applied to the transported impulse response value.
/// V(x) is the scaling-factor field at x; V_i is the per-sample value.
enum class Scaling
{
    none,       ///< s_i = 1 (4.9)
    volume,     ///< s_i = V(x)/V_i — preserves peak values (4.10)
    volume_det, ///< s_i = (V(x)/V_i) sqrt(det Sigma_i / det Sigma(x)) — preserves mass under `whitened_affine`
};

/// Support gate: what to do when the transported point T_i(y) lies outside
/// the sample's support ellipsoid E_i = {z : (z-mu_i)^T Sigma_i^{-1} (z-mu_i) <= tau^2}.
/// The gate isolates individual impulse responses within a multi-impulse
/// batch; with single-impulse batches it may be turned off (and then no
/// per-sample ellipsoid data is needed at all).
///
/// The gate is also the far-field short circuit: it is tested BEFORE any
/// mesh lookup (a few flops with the precomputed Sigma_i^{-1}; for
/// whitened_affine one shared test covers all neighbors), so gated
/// predictions cost almost nothing and a gated point is kept with value 0
/// even when it lies outside the mesh — "phi is zero beyond tau standard
/// deviations" is knowledge from the support model, valid regardless of
/// domain membership. Ungated points outside the mesh still exclude the
/// sample (the impulse response was never observed there).
enum class Support
{
    none,      ///< no gate: the batch function is evaluated wherever T_i(y) lands (outside the mesh excludes the sample)
    ellipsoid, ///< f_i = 0 when T_i(y) is outside E_i, tested before (and instead of) any mesh lookup
};

/// Configuration for evaluating kernel predictions from an ImpulseResponseField.
///
/// Different configurations require different data (see
/// ImpulseResponseField::validate): e.g. `identity` + `Scaling::none` +
/// `Support::none` on raw batches needs nothing beyond the batches and sample
/// points, while `whitened_affine` + `volume_det` needs all per-sample moments
/// and all vertex moment fields.
struct EvalConfig
{
    Frame   frame         = Frame::mean_translation;
    Scaling scaling       = Scaling::volume;
    Support support       = Support::ellipsoid;
    double  tau           = 3.0; ///< support ellipsoid scale (standard deviations); used when support == ellipsoid
    int     num_neighbors = 10;  ///< number of nearby sample points contributing predictions
};

} // end namespace psfi
