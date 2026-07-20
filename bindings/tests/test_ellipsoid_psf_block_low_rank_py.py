# SPDX-License-Identifier: MIT
# BlockLowRank: the BRLR format — builder, applies, container round-trip.

import numpy as np
import pytest

import ellipsoid_psf as ep

from test_ellipsoid_psf_partition_py import make_moment_field, gated_config


def build_problem(frame=ep.Frame.whitened_affine):
    F, vertices = make_moment_field()
    K = ep.KernelEvaluator(F, config=gated_config(frame))
    xs = np.linspace(0.15, 0.85, 4)
    xx = np.array([[a, b] for b in xs for a in xs])
    partition = ep.recursive_bisection_partition(xx, 4)
    return K, vertices, xx, partition


def test_exact_at_rtol_zero_and_applies():
    K, yy, xx, partition = build_problem()
    r = ep.block_low_rank(K, yy, xx, partition, 0.0)
    assert r.all_converged and not r.any_hit_max_rank

    A = K.block(yy, xx)
    B = r.matrix
    assert B.num_sources == 16 and B.num_targets == 81
    assert np.array_equal(B.to_dense(), A)  # bit-for-bit at rtol = 0

    rng = np.random.default_rng(0)
    U = rng.uniform(size=(16, 3))
    V = rng.uniform(size=(81, 2))
    assert np.allclose(B.apply(U), A @ U, atol=1e-13)
    assert np.allclose(B.applyT(V), A.T @ V, atol=1e-13)
    # Exact adjoint pair.
    assert np.isclose((B.apply(U)[:, 0] @ V[:, 0]),
                      (U[:, 0] @ B.applyT(V)[:, 0]), rtol=1e-13)


def test_tolerance_and_diagnostics():
    K, yy, xx, partition = build_problem()
    A = K.block(yy, xx)

    rtol = 1e-2
    r = ep.block_low_rank(K, yy, xx, partition, rtol)
    assert r.all_converged
    assert np.linalg.norm(A - r.matrix.to_dense()) <= rtol * np.linalg.norm(A) * (1 + 1e-9)
    assert r.matrix.storage_entries <= A.size
    assert len(r.block_info) == len(partition)

    capped = ep.block_low_rank(K, yy, xx, partition, 1e-12,
                               ep.KernelLowRankOptions(max_rank=1))
    assert capped.any_hit_max_rank and not capped.all_converged


def test_brlr_to_global_low_rank():
    K, yy, xx, partition = build_problem()
    B = ep.block_low_rank(K, yy, xx, partition, 1e-10).matrix
    D = B.to_dense()

    G = ep.randomized_svd(B, 16)  # full-width sampling: exact to rounding
    assert G.U.shape == (81, 16) and G.V.shape == (16, 16)
    assert np.linalg.norm(D - G.to_dense()) <= 1e-10 * np.linalg.norm(D)

    G6 = ep.randomized_svd(B, 6, ep.RSVDOptions(power_iterations=2))
    best = np.linalg.norm(D - ep.truncated_svd(D, 0.0, max_rank=6).to_dense())
    assert np.linalg.norm(D - G6.to_dense()) <= 10 * best + 1e-14


def test_container_round_trip():
    K, yy, xx, partition = build_problem()
    B = ep.block_low_rank(K, yy, xx, partition, 1e-6).matrix

    # Rebuild the container from its own blocks (the serialization seam).
    rebuilt = ep.BlockLowRank(B.num_sources, B.num_targets, list(B.blocks))
    assert np.array_equal(rebuilt.to_dense(), B.to_dense())

    # Validation: reused source ids across blocks throw.
    blk = ep.BlockLowRank.Block([0, 1], [0, 2], True,
                                np.zeros((2, 1)), np.zeros((2, 1)))
    dup = ep.BlockLowRank.Block([1, 2], [0, 2], True,
                                np.zeros((2, 1)), np.zeros((2, 1)))
    with pytest.raises(ValueError):
        ep.BlockLowRank(3, 3, [blk, dup])
