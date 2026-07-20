# SPDX-License-Identifier: MIT
# recursive_bisection_partition, the support oracles, and block_target_sets.

import numpy as np
import pytest

import ellipsoid_psf as ep

from test_ellipsoid_psf_py import make_grid_mesh


def make_moment_field():
    # Mirrors the C++ test fixture: Gaussian-bump batches with full moment
    # data; the Sigma field varies with position.
    vertices, cells = make_grid_mesh(8)
    nv = vertices.shape[0]
    F = ep.ImpulseResponseField(vertices, cells, batches_normalized=False)
    offset = np.array([0.02, -0.01])
    pts = np.array([[0.25, 0.30], [0.60, 0.35], [0.40, 0.60],
                    [0.70, 0.70], [0.30, 0.75], [0.55, 0.55]])
    for b in range(len(pts)):
        mu = pts[b] + offset
        psi = np.exp(-np.sum((vertices - mu) ** 2, axis=1) / 0.12**2)
        sx = 0.06 + 0.01 * b
        F.add_batch(pts[b:b + 1], psi, np.array([1.0]), mu[None, :],
                    np.diag([sx**2, 0.05**2])[None, :, :])
    field_mu = vertices + offset
    sx = 0.06 + 0.08 * vertices[:, 0]
    sy = 0.05 + 0.04 * vertices[:, 1]
    field_Sigma = np.zeros((nv, 2, 2))
    field_Sigma[:, 0, 0] = sx**2
    field_Sigma[:, 1, 1] = sy**2
    F.set_moment_fields(np.ones(nv), field_mu, field_Sigma)
    return F, vertices


def gated_config(frame, num_neighbors=3, tau=2.0):
    return ep.EvalConfig(frame=frame, scaling=ep.Scaling.none,
                         support=ep.Support.ellipsoid, tau=tau,
                         num_neighbors=num_neighbors)


def test_recursive_bisection_partition():
    rng = np.random.default_rng(0)
    pts = rng.uniform(size=(100, 2))
    parts = ep.recursive_bisection_partition(pts, 30)
    assert len(parts) == 4  # 100 -> 50 -> 25
    flat = sorted(i for part in parts for i in part)
    assert flat == list(range(100))
    for part in parts:
        assert len(part) <= 30
        assert part == sorted(part)
    assert ep.recursive_bisection_partition(pts, 30) == parts
    assert len(ep.recursive_bisection_partition(pts, 100)) == 1
    with pytest.raises(ValueError):
        ep.recursive_bisection_partition(pts, 0)


def test_support_ellipsoids_invariant():
    F, _ = make_moment_field()
    cfg = gated_config(ep.Frame.mean_translation)
    ys = np.array([[x, y] for y in np.linspace(0, 1, 9) for x in np.linspace(0, 1, 9)])

    outside_checked = 0
    for x in [np.array([0.3, 0.4]), np.array([0.6, 0.6]), np.array([0.45, 0.7])]:
        centers, Sigmas = F.support_ellipsoids(x, cfg)
        assert centers.shape == (3, 2) and Sigmas.shape == (3, 2, 2)
        for y in ys:
            d = y[None, :] - centers
            inside = any(d[k] @ np.linalg.solve(Sigmas[k], d[k]) <= 1.0
                         for k in range(len(centers)))
            if not inside:
                outside_checked += 1
                _, _, values = F.predictions(y, x, cfg)
                assert np.all(values == 0.0)
    assert outside_checked > 0

    # Support.none has no compact support.
    with pytest.raises(ValueError):
        F.support_ellipsoids(np.array([0.5, 0.5]),
                             ep.EvalConfig(frame=ep.Frame.translation,
                                           scaling=ep.Scaling.none,
                                           support=ep.Support.none))


def test_block_target_sets_invariant():
    F, vertices = make_moment_field()
    cfg = gated_config(ep.Frame.whitened_affine)
    K = ep.KernelEvaluator(F, config=cfg)

    xs = np.linspace(0.15, 0.85, 4)
    xx = np.array([[a, b] for b in xs for a in xs])
    partition = ep.recursive_bisection_partition(xx, 4)
    sets = ep.block_target_sets(K, vertices, xx, partition)
    assert len(sets) == len(partition)

    B = K.block(vertices, xx)
    for part, tset in zip(partition, sets):
        excluded = np.setdiff1d(np.arange(len(vertices)), tset)
        assert len(excluded) > 0  # genuine sparsity
        assert np.all(B[np.ix_(excluded, part)] == 0.0)

    # target_support at a query point matches the whitened field ellipsoid
    # count (one shared gate).
    centers, Sigmas = K.target_support(np.array([0.5, 0.5]))
    assert centers.shape == (1, 2) and Sigmas.shape == (1, 2, 2)

    # source_support requires symmetric mode.
    with pytest.raises(Exception):
        K.source_support(np.array([0.5, 0.5]))
    K_sym = ep.KernelEvaluator(F, F, config=cfg)
    centers_s, _ = K_sym.source_support(np.array([0.5, 0.5]))
    assert centers_s.shape[0] == 1
