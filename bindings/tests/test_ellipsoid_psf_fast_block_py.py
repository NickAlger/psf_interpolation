# SPDX-License-Identifier: MIT
# The fixed-source fast path from Python: rbf_functional vs rbf_interpolate,
# predictions_over_targets vs per-pair predictions, block vs entrywise.

import numpy as np
import pytest

import ellipsoid_psf as ep

from test_ellipsoid_psf_partition_py import make_moment_field, gated_config


def test_rbf_functional_matches_interpolate():
    rng = np.random.default_rng(3)
    centers = rng.uniform(-1, 1, size=(7, 2))
    values = rng.uniform(-1, 1, size=7)
    p = np.array([0.13, -0.07])
    scheme = ep.RBFScheme()
    lam = ep.rbf_functional(centers, p, scheme)
    assert lam.shape == (7,)
    assert np.isclose(lam @ values,
                      ep.rbf_interpolate(values, centers, p[None, :], scheme)[0],
                      rtol=1e-9, atol=1e-12)


def test_predictions_over_targets_matches_per_pair():
    F, vertices = make_moment_field()
    cfg = gated_config(ep.Frame.mean_translation)
    yy = np.array([[x, y] for y in np.linspace(0, 1, 6) for x in np.linspace(0, 1, 6)])
    x = np.array([0.45, 0.55])

    indices, points, values, excluded = F.predictions_over_targets(yy, x, cfg)
    k = len(indices)
    assert values.shape == (k, len(yy)) and excluded.shape == (k, len(yy))
    for jj, y in enumerate(yy):
        pi, pp, pv = F.predictions(y, x, cfg)
        keep = ~excluded[:, jj]
        assert np.array_equal(indices[keep], pi)
        assert np.array_equal(values[keep, jj], pv)  # bitwise


def test_block_matches_entrywise():
    F, vertices = make_moment_field()
    cfg = gated_config(ep.Frame.whitened_affine)
    K = ep.KernelEvaluator(F, config=cfg)
    yy = np.array([[x, y] for y in np.linspace(0, 1, 5) for x in np.linspace(0, 1, 5)])
    xx = np.array([[x, y] for y in np.linspace(0.2, 0.8, 3) for x in np.linspace(0.2, 0.8, 3)])
    B = K.block(yy, xx)
    for jj in range(len(xx)):
        for ii in range(len(yy)):
            assert abs(B[ii, jj] - K(yy[ii], xx[jj])) <= 1e-11
