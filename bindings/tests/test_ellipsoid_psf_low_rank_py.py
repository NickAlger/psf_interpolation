# SPDX-License-Identifier: MIT
# Low-rank tools: truncated SVD, ACA with random restarts, recompression,
# functor-driven randomized SVD — cross-checked against numpy.

import numpy as np
import pytest

import ellipsoid_psf as ep


def pseudo_random(m, n, seed):
    return np.random.default_rng(seed).uniform(-1.0, 1.0, size=(m, n))


def with_singular_values(m, n, sigma, seed):
    p = len(sigma)
    qu, _ = np.linalg.qr(pseudo_random(m, p, seed))
    qv, _ = np.linalg.qr(pseudo_random(n, p, seed + 1))
    return qu @ np.diag(sigma) @ qv.T


def rel_err(A, F):
    return np.linalg.norm(A - F.to_dense()) / np.linalg.norm(A)


def test_low_rank_struct():
    U = pseudo_random(6, 2, 0)
    V = pseudo_random(4, 2, 1)
    F = ep.LowRank(U, V)
    assert F.rank == 2
    assert np.allclose(F.to_dense(), U @ V.T)
    assert "rank=2" in repr(F)
    with pytest.raises(ValueError):
        ep.LowRank(U, pseudo_random(4, 3, 2))


def test_truncated_svd_against_numpy():
    sigma = np.array([1.0, 0.5, 0.1, 0.05, 0.01])
    A = with_singular_values(11, 8, sigma, 3)

    # The Frobenius truncation rule, computed independently with numpy.
    s = np.linalg.svd(A, compute_uv=False)
    rtol = 0.12
    tails = np.sqrt(np.cumsum((s**2)[::-1]))[::-1]  # tails[r] = ||sigma_{r+1:}||
    expected_rank = int(np.argmax(np.concatenate([tails[1:], [0.0]])
                                  <= rtol * np.linalg.norm(s))) + 1
    F = ep.truncated_svd(A, rtol)
    assert F.rank == expected_rank == 2
    assert rel_err(A, F) <= rtol

    # Exact-rank recovery and the max_rank cap.
    B = pseudo_random(12, 3, 4) @ pseudo_random(9, 3, 5).T
    assert ep.truncated_svd(B, 1e-12).rank == 3
    assert rel_err(B, ep.truncated_svd(B, 1e-12)) <= 1e-12
    assert ep.truncated_svd(B, 0.0, max_rank=2).rank == 2
    with pytest.raises(ValueError):
        ep.truncated_svd(B, -1.0)


def test_recompress():
    U0 = pseudo_random(10, 3, 6)
    V0 = pseudo_random(7, 3, 7)
    inflated = ep.LowRank(np.hstack([U0, U0]), np.hstack([0.5 * V0, 0.5 * V0]))
    F = ep.recompress(inflated, 1e-12)
    assert F.rank == 3
    assert np.allclose(F.to_dense(), U0 @ V0.T)


def test_aca_recovers_and_reports():
    A = pseudo_random(30, 4, 8) @ pseudo_random(25, 4, 9).T
    rtol = 1e-10
    result = ep.aca(lambda i: A[i, :], lambda j: A[:, j], 30, 25, rtol)
    assert result.converged
    assert not result.hit_max_rank
    assert result.factors.rank == 4
    assert rel_err(A, result.factors) <= rtol
    assert len(result.sampled_rows) == result.sampled_rank
    assert all(0 <= i < 30 for i in result.sampled_rows)
    assert all(0 <= j < 25 for j in result.sampled_cols)

    # Only sampled rows/columns are ever requested (the point of ACA).
    assert result.sampled_rank < 30

    # Deterministic for a fixed seed.
    again = ep.aca(lambda i: A[i, :], lambda j: A[:, j], 30, 25, rtol)
    assert again.sampled_rows == result.sampled_rows
    assert np.array_equal(again.factors.U, result.factors.U)


def test_aca_rank_cap_is_reported():
    A = pseudo_random(20, 5, 10) @ pseudo_random(15, 5, 11).T
    result = ep.aca(lambda i: A[i, :], lambda j: A[:, j], 20, 15, 1e-12,
                    ep.ACAOptions(max_rank=3))
    assert result.hit_max_rank
    assert not result.converged
    assert result.sampled_rank == 3


def test_aca_disconnected_support():
    # Block-diagonal: partial pivoting alone can stay trapped in one block;
    # the random restarts must reach the other.
    A = np.zeros((40, 40))
    A[:20, :20] = pseudo_random(20, 2, 12) @ pseudo_random(20, 2, 13).T
    A[20:, 20:] = pseudo_random(20, 2, 14) @ pseudo_random(20, 2, 15).T
    result = ep.aca(lambda i: A[i, :], lambda j: A[:, j], 40, 40, 1e-8)
    assert result.converged
    assert result.factors.rank == 4
    assert rel_err(A, result.factors) <= 1e-8


def test_aca_validation():
    A = pseudo_random(4, 4, 16)
    with pytest.raises(ValueError):
        ep.aca(lambda i: A[i, :], lambda j: A[:, j], 4, 4, -1.0)
    with pytest.raises(ValueError):
        ep.aca(lambda i: A[i, :], lambda j: A[:, j], 4, 4, 1e-8,
               ep.ACAOptions(required_consecutive_successes=0))
    with pytest.raises(ValueError):
        ep.aca(lambda i: np.zeros(2), lambda j: A[:, j], 4, 4, 1e-8)


def test_randomized_svd():
    A = pseudo_random(25, 3, 17) @ pseudo_random(18, 3, 18).T
    F = ep.randomized_svd(lambda X: A @ X, lambda Y: A.T @ Y, 25, 18, 3)
    assert F.rank == 3
    assert rel_err(A, F) <= 1e-10

    # Near-best on a decaying spectrum, and deterministic for a fixed seed.
    sigma = 2.0 ** -np.arange(12)
    B = with_singular_values(30, 22, sigma, 19)
    options = ep.RSVDOptions(power_iterations=2)
    G = ep.randomized_svd(lambda X: B @ X, lambda Y: B.T @ Y, 30, 22, 8, options)
    best = rel_err(B, ep.truncated_svd(B, 0.0, max_rank=8))
    assert rel_err(B, G) <= 10.0 * best
    G2 = ep.randomized_svd(lambda X: B @ X, lambda Y: B.T @ Y, 30, 22, 8, options)
    assert np.array_equal(G.U, G2.U)

    with pytest.raises(ValueError):
        ep.randomized_svd(lambda X: A @ X, lambda Y: A.T @ Y, 25, 18, 0)
