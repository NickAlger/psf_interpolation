# SPDX-License-Identifier: MIT
# kernel_low_rank: the kernel-facing low-rank adapter (target axis = matrix
# rows, source axis = matrix columns), cross-checked against KernelEvaluator
# .block + numpy.

import numpy as np
import pytest

import ellipsoid_psf as ep

from test_ellipsoid_psf_py import make_grid_mesh


def make_kernel():
    # Gaussian-bump batches on an 8x8 grid mesh, translation frame, all
    # samples as neighbors (constant neighbor set => smooth kernel; see the
    # C++ test for why that matters for compressibility).
    vertices, cells = make_grid_mesh(8)
    F = ep.ImpulseResponseField(vertices, cells, batches_normalized=False)
    pts = np.array([[0.30, 0.40], [0.55, 0.50], [0.45, 0.65],
                    [0.62, 0.35], [0.38, 0.58]])
    for b in range(5):
        w = 0.15 + 0.03 * b
        vals = np.exp(-((vertices[:, 0] - pts[b, 0]) ** 2
                        + (vertices[:, 1] - pts[b, 1]) ** 2) / w**2)
        F.add_batch(pts[b:b + 1], vals)
    cfg = ep.EvalConfig(frame=ep.Frame.translation, scaling=ep.Scaling.none,
                        support=ep.Support.none, num_neighbors=5)
    return ep.KernelEvaluator(F, config=cfg)


def window_grid(m, lo, hi):
    # Windows keep every transported point y - x + x_i inside the mesh (an
    # exit would exclude the sample and put jumps in the kernel).
    xs = np.linspace(lo, hi, m)
    return np.array([[x, y] for y in xs for x in xs])


def test_dense_svd_pathway_matches_numpy():
    K = make_kernel()
    yy = window_grid(9, 0.35, 0.65)
    xx = window_grid(6, 0.40, 0.60)
    A = K.block(yy, xx)

    rtol = 1e-2
    options = ep.KernelLowRankOptions(method=ep.CompressionMethod.dense_svd)
    r = ep.kernel_low_rank(K, yy, xx, rtol, options)
    assert not r.used_aca
    assert r.converged and not r.hit_max_rank
    B = r.factors.to_dense()
    assert np.linalg.norm(A - B) <= rtol * np.linalg.norm(A) * (1 + 1e-9)

    # The rank matches the Frobenius truncation rule computed with numpy.
    s = np.linalg.svd(A, compute_uv=False)
    tails = np.sqrt(np.cumsum((s**2)[::-1]))[::-1]
    expected_rank = next(r for r in range(len(s) + 1)
                         if (tails[r] if r < len(s) else 0.0) <= rtol * np.linalg.norm(s))
    assert r.factors.rank == expected_rank


def test_aca_pathway_and_automatic_selection():
    K = make_kernel()
    yy = window_grid(9, 0.35, 0.65)
    xx = window_grid(6, 0.40, 0.60)
    A = K.block(yy, xx)

    rtol = 1e-4
    r = ep.kernel_low_rank(K, yy, xx, rtol,
                           ep.KernelLowRankOptions(method=ep.CompressionMethod.aca))
    assert r.used_aca and r.converged
    assert np.linalg.norm(A - r.factors.to_dense()) <= 10 * rtol * np.linalg.norm(A)

    # automatic: min dim 36 <= 128 -> dense; dense_min_dim = 0 -> aca.
    assert not ep.kernel_low_rank(K, yy, xx, rtol).used_aca
    assert ep.kernel_low_rank(K, yy, xx, rtol,
                              ep.KernelLowRankOptions(dense_min_dim=0)).used_aca


def test_rank_cap_and_validation():
    K = make_kernel()
    yy = window_grid(5, 0.35, 0.65)
    xx = window_grid(4, 0.40, 0.60)

    r = ep.kernel_low_rank(K, yy, xx, 1e-12,
                           ep.KernelLowRankOptions(max_rank=2))
    assert r.hit_max_rank and not r.converged
    assert r.factors.rank <= 2

    with pytest.raises(ValueError):
        ep.kernel_low_rank(K, yy, xx, -1.0)
    with pytest.raises(ValueError):
        ep.kernel_low_rank(K, np.zeros((4, 3)), xx, 1e-6)  # wrong dim
