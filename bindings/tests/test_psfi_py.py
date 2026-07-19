# SPDX-License-Identifier: MIT
# psfi binding tests: a deliberately slow, obviously-correct pure-numpy
# reference implementation of the prediction pipeline, compared against the
# C++ implementation over every configuration axis, plus API behavior tests.

import numpy as np
import pytest

import psfi


# ----------------------------------------------------------------------
#  Mesh helper: structured triangulation of [0, 1]^2
# ----------------------------------------------------------------------

def make_grid_mesh(n):
    xs = np.linspace(0.0, 1.0, n + 1)
    X, Y = np.meshgrid(xs, xs, indexing="xy")
    vertices = np.column_stack([X.ravel(), Y.ravel()])  # row-major by rows of constant y
    cells = []
    for jj in range(n):
        for ii in range(n):
            v00 = jj * (n + 1) + ii
            v10 = v00 + 1
            v01 = v00 + (n + 1)
            v11 = v01 + 1
            cells.append([v00, v10, v11])
            cells.append([v00, v11, v01])
    return vertices, np.array(cells, dtype=np.int32)


# ----------------------------------------------------------------------
#  Pure-numpy reference implementation
# ----------------------------------------------------------------------

def locate(vertices, cells, p, tol=1e-12):
    """Brute-force point location: containing cell and barycentric coords."""
    for c in range(cells.shape[0]):
        V = vertices[cells[c]]  # (3, 2)
        A = np.vstack([np.ones(3), V.T])
        b = np.concatenate([[1.0], p])
        alpha = np.linalg.solve(A, b)
        if np.all(alpha >= -tol):
            return c, alpha
    return -1, None


def sym_sqrt(S):
    lam, Q = np.linalg.eigh(S)
    return Q @ np.diag(np.sqrt(lam)) @ Q.T


def sym_inv_sqrt(S):
    lam, Q = np.linalg.eigh(S)
    return Q @ np.diag(1.0 / np.sqrt(lam)) @ Q.T


def reference_predictions(data, y, x, cfg):
    """Reference for ImpulseResponseField.predictions; data is a plain dict."""
    Frame, Scaling, Support = psfi.Frame, psfi.Scaling, psfi.Support
    m = data["sample_points"].shape[0]
    if m == 0:
        return np.array([], int), np.zeros((0, 2)), np.array([])

    need_mu_x = cfg.frame in (Frame.mean_translation, Frame.whitened_affine)
    need_Sig_x = (cfg.frame == Frame.whitened_affine) or (cfg.scaling == Scaling.volume_det)
    need_V_x = cfg.scaling in (Scaling.volume, Scaling.volume_det)

    V_x = mu_x = W_x = det_Sig_x = None
    if need_mu_x or need_Sig_x or need_V_x:
        c, alpha = locate(data["vertices"], data["cells"], x)
        if c < 0:
            return np.array([], int), np.zeros((0, 2)), np.array([])
        vids = data["cells"][c]
        if need_V_x:
            V_x = alpha @ data["field_V"][vids]
        if need_mu_x:
            mu_x = alpha @ data["field_mu"][vids]
        if need_Sig_x:
            # The W field: CG1 interpolation of the vertex inverse square
            # roots (exact at vertices, SPD by convexity); det Sigma(x) is
            # defined through it.
            W_vert = np.stack([sym_inv_sqrt(data["field_Sigma"][v]) for v in vids])
            W_x = np.einsum("k,kij->ij", alpha, W_vert)
            det_Sig_x = np.linalg.det(W_x) ** -2.0

    # For whitened_affine the whitened offset (and hence the gate) is shared
    # by every neighbor.
    if cfg.frame == Frame.whitened_affine:
        wvec = W_x @ (y - mu_x)
        shared_inside = wvec @ wvec <= cfg.tau**2

    d2 = np.sum((data["sample_points"] - x) ** 2, axis=1)
    order = np.argsort(d2, kind="stable")[: min(cfg.num_neighbors, m)]

    idx, pts, vals = [], [], []
    for i in order:
        xi = data["sample_points"][i]
        if cfg.frame == Frame.identity:
            z = np.array(y, dtype=float)
        elif cfg.frame == Frame.translation:
            z = y - x + xi
        elif cfg.frame == Frame.mean_translation:
            z = y - mu_x + data["sample_mu"][i]
        else:
            z = data["sample_mu"][i] + sym_sqrt(data["sample_Sigma"][i]) @ wvec

        # Support gate first (far-field short circuit): gated points are kept
        # with value 0 without a mesh lookup, even outside the mesh.
        if cfg.support == Support.ellipsoid:
            if cfg.frame == Frame.whitened_affine:
                inside = shared_inside
            else:
                dz = z - data["sample_mu"][i]
                inside = dz @ np.linalg.solve(data["sample_Sigma"][i], dz) <= cfg.tau**2
            if not inside:
                idx.append(i)
                pts.append(xi)
                vals.append(0.0)
                continue

        c, alpha = locate(data["vertices"], data["cells"], z)
        if c < 0:
            continue  # ungated point outside the domain: sample excluded

        psi = data["batch_psi"][data["point2batch"][i]]
        raw = alpha @ psi[data["cells"][c]]
        normalized = data["batches_normalized"]
        if cfg.scaling == Scaling.none:
            s = data["sample_V"][i] if normalized else 1.0
        elif cfg.scaling == Scaling.volume:
            s = V_x if normalized else V_x / data["sample_V"][i]
        else:
            det_i = np.linalg.det(data["sample_Sigma"][i])
            s = (V_x if normalized else V_x / data["sample_V"][i]) * np.sqrt(det_i / det_Sig_x)
        idx.append(i)
        pts.append(xi)
        vals.append(s * raw)
    return np.array(idx, int), np.array(pts).reshape(len(idx), 2), np.array(vals)


# ----------------------------------------------------------------------
#  Test data
# ----------------------------------------------------------------------

def random_spd_stack(rng, n, scale=0.05, floor=0.01):
    W = rng.standard_normal((n, 2, 2)) * scale
    return np.einsum("nij,nkj->nik", W, W) + floor * np.eye(2)


def build_field(seed=0, normalized=True, with_moments=True, with_fields=True, n=6):
    rng = np.random.default_rng(seed)
    vertices, cells = make_grid_mesh(n)
    nv = vertices.shape[0]

    F = psfi.ImpulseResponseField(vertices, cells, batches_normalized=normalized)
    data = {
        "vertices": vertices,
        "cells": cells,
        "batches_normalized": normalized,
        "sample_points": np.zeros((0, 2)),
        "sample_V": np.array([]),
        "sample_mu": np.zeros((0, 2)),
        "sample_Sigma": np.zeros((0, 2, 2)),
        "point2batch": np.array([], int),
        "batch_psi": [],
        "field_V": None,
        "field_mu": None,
        "field_Sigma": None,
    }

    if with_fields:
        field_V = 1.0 + rng.uniform(0.5, 1.5, nv)
        field_mu = vertices + rng.uniform(-0.03, 0.03, (nv, 2))
        field_Sigma = random_spd_stack(rng, nv)
        F.set_moment_fields(field_V, field_mu, field_Sigma)
        data.update(field_V=field_V, field_mu=field_mu, field_Sigma=field_Sigma)

    for b in range(3):
        nb = int(rng.integers(2, 5))
        pts = rng.uniform(0.15, 0.85, (nb, 2))
        psi = rng.standard_normal(nv)
        if with_moments:
            V = rng.uniform(0.5, 2.0, nb)
            mu = pts + rng.uniform(-0.02, 0.02, (nb, 2))
            Sigma = random_spd_stack(rng, nb)
            F.add_batch(pts, psi, V, mu, Sigma)
        else:
            V = np.full(nb, np.nan)
            mu = np.full((nb, 2), np.nan)
            Sigma = np.full((nb, 2, 2), np.nan)
            F.add_batch(pts, psi)
        data["sample_points"] = np.vstack([data["sample_points"], pts])
        data["sample_V"] = np.concatenate([data["sample_V"], V])
        data["sample_mu"] = np.vstack([data["sample_mu"], mu])
        data["sample_Sigma"] = np.concatenate([data["sample_Sigma"], Sigma])
        data["point2batch"] = np.concatenate([data["point2batch"], np.full(nb, b, int)])
        data["batch_psi"].append(psi)
    return F, data


# ----------------------------------------------------------------------
#  C++ vs reference over every configuration axis
# ----------------------------------------------------------------------

ALL_FRAMES = [psfi.Frame.identity, psfi.Frame.translation,
              psfi.Frame.mean_translation, psfi.Frame.whitened_affine]
ALL_SCALINGS = [psfi.Scaling.none, psfi.Scaling.volume, psfi.Scaling.volume_det]
ALL_SUPPORTS = [psfi.Support.none, psfi.Support.ellipsoid]


@pytest.mark.parametrize("normalized", [True, False])
@pytest.mark.parametrize("support", ALL_SUPPORTS)
@pytest.mark.parametrize("scaling", ALL_SCALINGS)
@pytest.mark.parametrize("frame", ALL_FRAMES)
def test_predictions_match_reference(frame, scaling, support, normalized):
    F, data = build_field(seed=0, normalized=normalized)
    cfg = psfi.EvalConfig(frame=frame, scaling=scaling, support=support,
                          tau=2.5, num_neighbors=5)
    rng = np.random.default_rng(12345)
    num_nonempty = 0
    num_gated = 0
    # Tight y offsets stay within the support ellipsoids; wide offsets push
    # transported points past the gate (and sometimes out of the mesh), so
    # every code path is exercised under every frame.
    for offset in [0.15, 0.4]:
        for _ in range(8):
            x = rng.uniform(0.1, 0.9, 2)
            y = x + rng.uniform(-offset, offset, 2)
            got_i, got_p, got_v = F.predictions(y, x, cfg)
            ref_i, ref_p, ref_v = reference_predictions(data, y, x, cfg)
            assert np.array_equal(got_i, ref_i)
            assert np.allclose(got_p, ref_p, atol=1e-14)
            assert np.allclose(got_v, ref_v, rtol=1e-9, atol=1e-12)
            num_nonempty += int(len(got_i) > 0)
            num_gated += int(np.any(got_v == 0.0))
    assert num_nonempty >= 8  # the comparison actually exercised predictions
    if support == psfi.Support.ellipsoid:
        assert num_gated >= 1  # ... including the gate


def test_num_neighbors_truncation():
    F, data = build_field(seed=0)
    x = np.array([0.5, 0.5])
    y = np.array([0.52, 0.48])
    for k in [1, 3, 100]:
        cfg = psfi.EvalConfig(frame=psfi.Frame.translation, scaling=psfi.Scaling.none,
                              support=psfi.Support.none, num_neighbors=k)
        got_i, _, _ = F.predictions(y, x, cfg)
        ref_i, _, _ = reference_predictions(data, y, x, cfg)
        assert np.array_equal(got_i, ref_i)
        assert len(got_i) <= min(k, F.num_sample_points)


# ----------------------------------------------------------------------
#  API behavior
# ----------------------------------------------------------------------

def test_validate_reports_missing_data():
    F, _ = build_field(seed=1, with_moments=False, with_fields=False)

    with pytest.raises(ValueError, match="per-sample V"):
        F.validate(psfi.EvalConfig(frame=psfi.Frame.translation, scaling=psfi.Scaling.none,
                                   support=psfi.Support.none))
    with pytest.raises(ValueError, match="vertex field mu"):
        F.validate(psfi.EvalConfig(frame=psfi.Frame.mean_translation, scaling=psfi.Scaling.none,
                                   support=psfi.Support.none))
    with pytest.raises(ValueError, match="per-sample Sigma"):
        F.validate(psfi.EvalConfig(frame=psfi.Frame.whitened_affine, scaling=psfi.Scaling.none,
                                   support=psfi.Support.none))
    with pytest.raises(ValueError, match="vertex field V"):
        F.validate(psfi.EvalConfig(frame=psfi.Frame.translation, scaling=psfi.Scaling.volume,
                                   support=psfi.Support.none))
    with pytest.raises(ValueError, match="per-sample mu"):
        F.validate(psfi.EvalConfig(frame=psfi.Frame.translation, scaling=psfi.Scaling.none,
                                   support=psfi.Support.ellipsoid))
    with pytest.raises(ValueError, match="tau > 0"):
        F.validate(psfi.EvalConfig(support=psfi.Support.ellipsoid, tau=0.0))
    with pytest.raises(ValueError, match="num_neighbors"):
        F.validate(psfi.EvalConfig(num_neighbors=0))


def test_minimal_data_mode():
    # Raw single-impulse batches with no moments anywhere: identity frame,
    # no scaling, no gate. Predictions are the batch function at y.
    vertices, cells = make_grid_mesh(6)
    nv = vertices.shape[0]
    F = psfi.ImpulseResponseField(vertices, cells, batches_normalized=False)
    psi = 2.0 + 3.0 * vertices[:, 0] - vertices[:, 1]  # affine: CG1-exact
    pts = np.array([[0.3, 0.4], [0.7, 0.6]])
    F.add_batch(pts, psi)

    cfg = psfi.EvalConfig(frame=psfi.Frame.identity, scaling=psfi.Scaling.none,
                          support=psfi.Support.none, num_neighbors=2)
    F.validate(cfg)
    y = np.array([0.45, 0.55])
    x = np.array([0.35, 0.40])
    got_i, got_p, got_v = F.predictions(y, x, cfg)
    expected = 2.0 + 3.0 * y[0] - y[1]
    assert np.array_equal(got_i, [0, 1])
    assert np.allclose(got_v, expected, rtol=1e-12)

    # But volume scaling correctly complains.
    with pytest.raises(ValueError, match="per-sample V"):
        F.validate(psfi.EvalConfig(frame=psfi.Frame.identity, scaling=psfi.Scaling.volume,
                                   support=psfi.Support.none))


def test_batch_moment_consistency_enforced():
    vertices, cells = make_grid_mesh(4)
    nv = vertices.shape[0]
    F = psfi.ImpulseResponseField(vertices, cells)
    pts = np.array([[0.5, 0.5]])
    psi = np.zeros(nv)
    F.add_batch(pts, psi, V=np.array([1.0]))
    with pytest.raises(ValueError, match="same set of per-sample moments"):
        F.add_batch(pts, psi)  # V now missing
    assert F.num_batches == 1


def test_non_spd_sigma_rejected():
    vertices, cells = make_grid_mesh(4)
    F = psfi.ImpulseResponseField(vertices, cells)
    pts = np.array([[0.5, 0.5]])
    psi = np.zeros(vertices.shape[0])
    bad = -np.eye(2)[None, :, :]
    with pytest.raises(ValueError, match="not positive definite"):
        F.add_batch(pts, psi, Sigma=bad)
    assert F.num_batches == 0


def test_stale_kdtree():
    vertices, cells = make_grid_mesh(4)
    F = psfi.ImpulseResponseField(vertices, cells, batches_normalized=False)
    pts = np.array([[0.5, 0.5]])
    F.add_batch(pts, np.zeros(vertices.shape[0]), rebuild=False)
    cfg = psfi.EvalConfig(frame=psfi.Frame.identity, scaling=psfi.Scaling.none,
                          support=psfi.Support.none)
    p = np.array([0.5, 0.5])
    with pytest.raises(RuntimeError, match="kd-tree is stale"):
        F.predictions(p, p, cfg)
    F.rebuild_kdtree()
    got_i, _, _ = F.predictions(p, p, cfg)
    assert len(got_i) == 1


def test_x_outside_mesh():
    F, data = build_field(seed=2)
    y = np.array([0.6, 0.5])
    x_out = np.array([1.5, 0.5])

    # Field lookups needed at x: uninformed, empty result.
    cfg = psfi.EvalConfig(frame=psfi.Frame.mean_translation, scaling=psfi.Scaling.volume,
                          support=psfi.Support.none)
    got_i, _, _ = F.predictions(y, x_out, cfg)
    assert len(got_i) == 0

    # No field lookups: exterior x fine (matches the reference).
    cfg = psfi.EvalConfig(frame=psfi.Frame.identity, scaling=psfi.Scaling.none,
                          support=psfi.Support.none)
    got_i, _, got_v = F.predictions(y, x_out, cfg)
    ref_i, _, ref_v = reference_predictions(data, y, x_out, cfg)
    assert np.array_equal(got_i, ref_i)
    assert np.allclose(got_v, ref_v, rtol=1e-9, atol=1e-12)


def test_introspection_roundtrip():
    F, data = build_field(seed=3)
    assert F.dim == 2
    assert F.num_batches == 3
    assert F.num_sample_points == data["sample_points"].shape[0]
    assert F.batches_normalized
    assert F.has_sample_V and F.has_sample_mu and F.has_sample_Sigma
    assert F.has_field_V and F.has_field_mu and F.has_field_Sigma

    assert np.allclose(F.sample_points, data["sample_points"])
    assert np.allclose(F.sample_V, data["sample_V"])
    assert np.allclose(F.sample_mu, data["sample_mu"])
    assert np.allclose(F.sample_Sigma, data["sample_Sigma"])  # SPD input: symmetrization is a no-op
    assert np.array_equal(F.point2batch, data["point2batch"])
    for b in range(3):
        start, stop = F.batch_range(b)
        assert np.all(data["point2batch"][start:stop] == b)
        assert np.allclose(F.batch_values(b), data["batch_psi"][b])

    assert np.allclose(F.mesh_vertices, data["vertices"])
    assert np.array_equal(F.mesh_cells, data["cells"])

    assert isinstance(psfi.__version__, str)
    r = repr(psfi.EvalConfig())
    assert "mean_translation" in r and "volume" in r
