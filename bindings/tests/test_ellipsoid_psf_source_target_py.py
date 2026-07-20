# SPDX-License-Identifier: MIT
# Rectangular kernels through the bindings: separate source/target meshes
# with different dimensions (1D source -> 2D target).

import numpy as np
import pytest

import ellipsoid_psf as ep

from test_ellipsoid_psf_py import make_grid_mesh, make_interval_mesh


def build_cross_field():
    tgt_v, tgt_c = make_grid_mesh(8)
    src_v, src_c = make_interval_mesh(8)
    F = ep.ImpulseResponseField(tgt_v, tgt_c,
                                source_vertices=src_v, source_cells=src_c)

    a, b, c = 2.0, 3.0, -1.0
    psi = a + b * tgt_v[:, 0] + c * tgt_v[:, 1]  # affine: CG1-exact

    mu_map = lambda x: np.array([x, 0.5 + 0.2 * (x - 0.5)])
    xs = src_v[:, 0]
    field_V = 1.0 + 0.5 * xs
    field_mu = np.stack([mu_map(x) for x in xs])
    field_Sigma = np.tile(np.diag([0.01, 0.02]), (len(xs), 1, 1))
    F.set_moment_fields(field_V, field_mu, field_Sigma)

    x1 = 0.4
    F.add_batch(np.array([[x1]]), psi, V=np.array([1.0 + 0.5 * x1]),
                mu=mu_map(x1).reshape(1, 2),
                Sigma=np.diag([0.0025, 0.005])[None, :, :])
    return F, (a, b, c), mu_map


def test_cross_domain_shapes_and_closed_form():
    F, (a, b, c), mu_map = build_cross_field()
    assert F.dim_source == 1
    assert F.dim_target == 2
    assert F.has_separate_source_mesh
    assert F.source_mesh_vertices.shape == (9, 1)
    assert F.target_mesh_vertices.shape == (81, 2)
    assert F.sample_points.shape == (1, 1)
    assert F.sample_mu.shape == (1, 2)
    assert F.sample_Sigma.shape == (1, 2, 2)

    cfg = ep.EvalConfig(frame=ep.Frame.mean_translation, scaling=ep.Scaling.volume,
                        support=ep.Support.none, num_neighbors=1)
    x = np.array([0.55])
    mu_x = mu_map(x[0])
    y = mu_x + np.array([0.04, -0.03])
    idx, pts, vals = F.predictions(y, x, cfg)
    assert pts.shape == (1, 1)
    z = y - mu_x + mu_map(0.4)
    expected = (1.0 + 0.5 * x[0]) * (a + b * z[0] + c * z[1])
    assert np.isclose(vals[0], expected, rtol=1e-12)

    K = ep.KernelEvaluator(F, config=cfg)
    assert K.dim_source == 1 and K.dim_target == 2
    assert np.isclose(K(y, x), expected, rtol=1e-12)
    B = K.block(np.array([[0.5, 0.5], [0.55, 0.45]]), np.array([[0.4], [0.6]]))
    assert B.shape == (2, 2)


def test_cross_domain_restrictions():
    F, _, _ = build_cross_field()
    with pytest.raises(ValueError, match="equal source and target dimensions"):
        F.validate(ep.EvalConfig(frame=ep.Frame.translation, scaling=ep.Scaling.none,
                                 support=ep.Support.none))
    cfg = ep.EvalConfig(frame=ep.Frame.mean_translation, scaling=ep.Scaling.volume,
                        support=ep.Support.none)
    with pytest.raises(ValueError, match="symmetric mode"):
        ep.KernelEvaluator(F, F, cfg)
    with pytest.raises(ValueError, match="both source_vertices and source_cells"):
        tgt_v, tgt_c = make_grid_mesh(4)
        ep.ImpulseResponseField(tgt_v, tgt_c, source_vertices=np.zeros((3, 1)))
