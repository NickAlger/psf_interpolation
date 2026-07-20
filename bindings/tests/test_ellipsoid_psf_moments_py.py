# SPDX-License-Identifier: MIT
# clamp_spd and the Sigma-field positive-definiteness validation.

import numpy as np
import pytest

import ellipsoid_psf as ep

from test_ellipsoid_psf_py import make_grid_mesh, build_field


def test_clamp_spd_stack():
    good = np.array([[0.5, 0.1], [0.1, 0.3]])
    bad = np.diag([0.2, -1e-9])       # numerical-error negative
    tiny = np.diag([0.2, 1e-12])      # positive but below the floor
    stack = np.stack([good, bad, tiny])

    cleaned, modified = ep.clamp_spd(stack, 1e-4)
    assert np.array_equal(modified, [1, 2])
    assert np.array_equal(cleaned[0], good)  # untouched entries pass through exactly
    for S in cleaned:
        assert np.linalg.eigvalsh(S).min() >= 1e-4 * (1 - 1e-12)

    # Per-entry floors.
    cleaned2, modified2 = ep.clamp_spd(stack, np.array([1e-12, 1e-12, 1e-6]))
    assert np.array_equal(modified2, [1, 2])

    # Asymmetric input is symmetrized.
    asym = np.array([[[0.5, 0.2], [0.0, 0.3]]])
    cleaned3, _ = ep.clamp_spd(asym, 1e-6)
    assert np.allclose(cleaned3[0], cleaned3[0].T)
    assert np.isclose(cleaned3[0][0, 1], 0.1)

    with pytest.raises(ValueError, match="floor"):
        ep.clamp_spd(stack, 0.0)
    with pytest.raises(ValueError, match="floor"):
        ep.clamp_spd(stack, np.array([1e-6, 0.0, 1e-6]))


def test_set_moment_fields_validates_sigma():
    vertices, cells = make_grid_mesh(4)
    nv = vertices.shape[0]
    F = ep.ImpulseResponseField(vertices, cells)

    field_Sigma = np.tile(0.01 * np.eye(2), (nv, 1, 1))
    field_Sigma[7] = np.diag([0.01, -1e-9])

    with pytest.raises(ValueError, match="not positive definite") as excinfo:
        F.set_moment_fields(Sigma=field_Sigma)
    assert "v=7" in str(excinfo.value)
    assert "clamp_spd" in str(excinfo.value)
    assert not F.has_field_Sigma  # a failed set leaves the fields unchanged

    cleaned, modified = ep.clamp_spd(field_Sigma, 1e-6)
    assert np.array_equal(modified, [7])
    F.set_moment_fields(Sigma=cleaned)
    assert F.has_field_Sigma


def test_add_batch_error_mentions_clamp_spd():
    vertices, cells = make_grid_mesh(4)
    F = ep.ImpulseResponseField(vertices, cells)
    with pytest.raises(ValueError, match="clamp_spd"):
        F.add_batch(np.array([[0.5, 0.5]]), np.zeros(vertices.shape[0]),
                    Sigma=-np.eye(2)[None, :, :])


def test_validated_field_still_evaluates():
    # End-to-end: a field whose Sigma passed validation runs the
    # whitened_affine + volume_det path as before.
    F, _ = build_field(seed=0)
    cfg = ep.EvalConfig(frame=ep.Frame.whitened_affine, scaling=ep.Scaling.volume_det,
                        num_neighbors=5)
    K = ep.KernelEvaluator(F, config=cfg)
    v = K(np.array([0.5, 0.52]), np.array([0.48, 0.5]))
    assert np.isfinite(v)
