# SPDX-License-Identifier: MIT
# Slice-2 binding tests: rbf_interpolate cross-checked against
# scipy.interpolate.RBFInterpolator (shape-parameter conversion), and
# KernelEvaluator checked against a numpy reference that composes the slice-1
# prediction reference with the (already scipy-checked) rbf_interpolate.

import numpy as np
import pytest

import ellipsoid_psf as ep

from test_ellipsoid_psf_py import build_field, reference_predictions


# ----------------------------------------------------------------------
#  rbf_interpolate vs scipy
# ----------------------------------------------------------------------

# (ellipsoid_psf kernel, scipy kernel name, epsilon factor): our scaled distance is
# u = shape * r / r0; scipy evaluates phi(eps * r). Gaussian differs by the
# 1/2 in the exponent: exp(-u^2/2) = exp(-(u/sqrt(2))^2), so eps = shape /
# (r0 sqrt(2)); every other kernel matches with eps = shape / r0 (for the
# homogeneous kernels the overall scale factor cancels from the interpolant).
SCIPY_KERNELS = [
    (ep.RBFKernel.gaussian, "gaussian", 1.0 / np.sqrt(2.0)),
    (ep.RBFKernel.multiquadric, "multiquadric", 1.0),
    (ep.RBFKernel.inverse_multiquadric, "inverse_multiquadric", 1.0),
    (ep.RBFKernel.linear, "linear", 1.0),
    (ep.RBFKernel.thin_plate_spline, "thin_plate_spline", 1.0),
    (ep.RBFKernel.cubic, "cubic", 1.0),
]


@pytest.mark.parametrize("smoothing", [0.0, 0.3])
@pytest.mark.parametrize("kernel,scipy_name,eps_factor", SCIPY_KERNELS)
def test_rbf_interpolate_matches_scipy(kernel, scipy_name, eps_factor, smoothing):
    interp_mod = pytest.importorskip("scipy.interpolate")
    rng = np.random.default_rng(7)
    k, d, m = 9, 2, 6
    centers = rng.uniform(0.0, 1.0, (k, d))
    values = rng.standard_normal(k)
    eval_points = rng.uniform(0.2, 0.8, (m, d))

    shape = 2.0
    scheme = ep.RBFScheme(kernel=kernel, shape=shape, degree=1, smoothing=smoothing)
    ours = ep.rbf_interpolate(values, centers, eval_points, scheme)

    r0 = np.linalg.norm(centers.max(axis=0) - centers.min(axis=0))
    eps = shape / r0 * eps_factor
    theirs = interp_mod.RBFInterpolator(centers, values, kernel=scipy_name, epsilon=eps,
                                        degree=1, smoothing=smoothing)(eval_points)
    assert np.allclose(ours, theirs, rtol=1e-8, atol=1e-10)


def test_rbf_interpolate_basics():
    rng = np.random.default_rng(8)
    centers = rng.uniform(0.0, 1.0, (5, 2))
    values = rng.standard_normal(5)

    # Center reproduction at smoothing 0.
    at_centers = ep.rbf_interpolate(values, centers, centers, ep.RBFScheme())
    assert np.allclose(at_centers, values, rtol=1e-8)

    # Single center: constant.
    out = ep.rbf_interpolate(values[:1], centers[:1], centers, ep.RBFScheme())
    assert np.allclose(out, values[0])

    # Scheme validation happens at construction.
    with pytest.raises(ValueError, match="degree"):
        ep.RBFScheme(kernel=ep.RBFKernel.thin_plate_spline, degree=0)
    with pytest.raises(ValueError, match="shape"):
        ep.RBFScheme(shape=0.0)
    assert ep.rbf_min_degree(ep.RBFKernel.thin_plate_spline) == 1

    r = repr(ep.RBFScheme())
    assert "gaussian" in r and "degree=1" in r


# ----------------------------------------------------------------------
#  KernelEvaluator vs numpy reference
# ----------------------------------------------------------------------

def reference_kernel_value(data, y, x, cfg, scheme, symmetric, duplicate_tol=1e-7):
    """Compose the slice-1 prediction reference with rbf_interpolate."""
    _, pF, vF = reference_predictions(data, y, x, cfg)
    centers = [c for c in (pF - x)]
    values = list(vF)
    if symmetric:
        _, pA, vA = reference_predictions(data, x, y, cfg)
        for c, v in zip(pA - y, vA):
            for jj in range(len(vF)):  # only forward entries are merge partners
                if np.linalg.norm(centers[jj] - c) <= duplicate_tol:
                    centers[jj] = 0.5 * (centers[jj] + c)
                    values[jj] = 0.5 * (values[jj] + v)
                    break
            else:
                centers.append(c)
                values.append(v)
    if len(values) == 0:
        return 0.0
    return ep.rbf_interpolate(np.array(values), np.array(centers),
                              np.zeros((1, 2)), scheme)[0]


EVAL_SCHEMES = [
    ep.RBFScheme(),  # gaussian, shape 3, degree 1
    ep.RBFScheme(kernel=ep.RBFKernel.thin_plate_spline, degree=1),
    ep.RBFScheme(smoothing=0.1),
]


@pytest.mark.parametrize("symmetric", [False, True])
@pytest.mark.parametrize("frame", [ep.Frame.translation, ep.Frame.mean_translation,
                                   ep.Frame.whitened_affine])
def test_kernel_evaluator_matches_reference(frame, symmetric):
    F, data = build_field(seed=0)
    cfg = ep.EvalConfig(frame=frame, scaling=ep.Scaling.volume,
                        support=ep.Support.ellipsoid, tau=2.5, num_neighbors=5)
    rng = np.random.default_rng(99)
    for scheme in EVAL_SCHEMES:
        K = ep.KernelEvaluator(F, F if symmetric else None, cfg, scheme)
        assert K.symmetric == symmetric
        for _ in range(6):
            x = rng.uniform(0.15, 0.85, 2)
            y = x + rng.uniform(-0.2, 0.2, 2)
            expected = reference_kernel_value(data, y, x, cfg, scheme, symmetric)
            assert np.isclose(K(y, x), expected, rtol=1e-8, atol=1e-12)


def test_kernel_evaluator_api():
    F, data = build_field(seed=4)
    cfg = ep.EvalConfig(num_neighbors=5)  # defaults: mean_translation, volume, ellipsoid
    K = ep.KernelEvaluator(F, config=cfg)

    # Symmetric evaluator at y = x merges the duplicated prediction sets and
    # agrees with cols-only.
    K_sym = ep.KernelEvaluator(F, F, cfg)
    p = np.array([0.5, 0.45])
    assert np.isclose(K_sym(p, p), K(p, p), rtol=1e-10)

    # block == entrywise __call__, independent of thread count.
    rng = np.random.default_rng(5)
    yy = rng.uniform(0.2, 0.8, (4, 2))
    xx = rng.uniform(0.2, 0.8, (3, 2))
    B = K.block(yy, xx)
    B1 = K.block(yy, xx, num_threads=1)
    assert B.shape == (4, 3)
    assert np.array_equal(B, B1)
    for ii in range(4):
        for jj in range(3):
            assert np.isclose(B[ii, jj], K(yy[ii], xx[jj]), rtol=1e-12)

    # Construction validates config against the field data.
    F_bare, _ = build_field(seed=4, with_moments=False, with_fields=False)
    with pytest.raises(ValueError, match="vertex field"):
        ep.KernelEvaluator(F_bare, config=cfg)

    # Config and scheme are inspectable.
    assert K.config.num_neighbors == 5
    assert K.rbf.degree == 1
    assert K.dim_source == 2 and K.dim_target == 2
    assert not K.symmetric
