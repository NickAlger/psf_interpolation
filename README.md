# psfi

**Point spread function interpolation.** Evaluate an integral kernel
Φ(y, x) at arbitrary points by interpolating *transported* impulse responses
sampled at scattered locations — the kernel-approximation core of the
PSF method of [Alger, Hartland, Petra, Ghattas, SISC 2024](https://doi.org/10.1137/23M1584745).
Header-only C++17 with Python bindings; depends on Eigen and
[etree](https://github.com/NickAlger/ellipsoid_tree).

> **Status: work in progress.** The full evaluation pipeline is implemented
> and tested: the impulse-response container, the per-neighbor prediction
> machinery (all frame maps and scalings, with data-requirement validation),
> the RBF layer (six kernels, polynomial tails, ridge smoothing;
> cross-checked against scipy's `RBFInterpolator`), and the threaded
> `KernelEvaluator` (cols-only and symmetric), with an end-to-end worked
> example — plus the downstream compressed matrix formats: global low rank
> (truncated SVD / ACA / randomized SVD) and the source-partitioned block
> low rank format with its `apply`/`applyT` matvecs, with a second worked
> example. API docs and the PyPI release are still to come.

<p align="center">
<img src="docs/img/frog_kernel__01_batch_1.png" width="480">
</p>

*One impulse response batch of the "frog" kernel with its support ellipsoids
and two evaluation targets (red), from the
[end-to-end example](docs/examples/frog_kernel.md): support ellipsoids from
a-priori moments → non-overlapping batch picking (etree) → impulse response
batches → kernel evaluation anywhere. With 10 batches the median relative
column error is 4.6% for the paper's configuration and 3.6% for
`whitened_affine` + `volume_det`, which deforms each impulse to the local
ellipsoid shape — see the example page for the full figure sequence,
regenerated from the code by `docs/generate_examples.py` and freshness-checked
in CI.*

## The idea

An operator A with a locally supported integral kernel is probed by applying
it to Dirac combs, giving *batches* of impulse responses φ_i — functions on a
simplicial mesh, each localized inside a support ellipsoid (μ_i, Σ_i). To
evaluate the kernel at an arbitrary pair (y, x), each nearby sample point x_i
predicts

```
f_i = s_i · φ_i(T_i(y)),
```

where the **frame map** T_i identifies the neighborhood of x with the
neighborhood of x_i and the **scaling** s_i corrects the amplitude; the
predictions {(x_i, f_i)} are then combined by radial basis function
interpolation at x. The two axes are independent:

| frame map `T_i(y)` | paper eq. |
| --- | --- |
| `identity`: y | (4.7) |
| `translation`: y − x + x_i | (4.8) |
| `mean_translation`: y − μ(x) + μ_i | (4.9) |
| `whitened_affine`: μ_i + Σ_i^{1/2} Σ(x)^{−1/2} (y − μ(x)) | new |

| scaling `s_i` | meaning |
| --- | --- |
| `none`: 1 | raw values (4.9) |
| `volume`: V(x)/V_i | preserves peak values (4.10) |
| `volume_det`: (V(x)/V_i)·√(det Σ_i / det Σ(x)) | preserves mass under `whitened_affine` |

Here V, μ, Σ are the moment fields of the impulse responses (mass, mean,
covariance), evaluated at sample points and — when a configuration needs
them — at target points via CG1 interpolation of vertex fields. A **support
gate** (`Support.ellipsoid`) zeroes predictions whose transported point falls
outside the sample's ellipsoid at scale τ, which is what isolates individual
impulses within a multi-impulse batch; transported points outside the mesh
exclude the sample entirely (the method never evaluates impulse responses
outside their domain of definition).

**Data requirements are minimal per configuration** and validated up front:
`identity`/`translation` with no gate and no scaling needs nothing but the
batches and sample points (classical scattered-PSF interpolation of
single-impulse batches), while `whitened_affine` + `volume_det` needs all
per-sample moments and all vertex moment fields.
`ImpulseResponseField.validate(config)` lists exactly what is missing.

**Different source and target domains are supported**, including different
dimensions: the field optionally takes a second mesh, the *source* mesh,
carrying the moment fields and locating the query point x, while the *target*
mesh carries the impulse responses. The moment map μ : Ω_src → Ω_tgt sends
each source point to where its impulse response lands, which is what makes
`mean_translation` and `whitened_affine` well-typed across domains (paper
§4.2); `translation` requires equal dimensions, and symmetric evaluation
requires the single-domain case. By default source = target and nothing
distinguishes the two roles.

**Compressed matrix formats** turn the evaluated kernel into operators.
The kernel matrix over given target/source points can be assembled dense
(`KernelEvaluator.block`), compressed globally
(`kernel_low_rank`: truncated SVD or adaptive cross approximation), or —
the production format — compressed as **block low rank** (`block_low_rank`):
the source axis is partitioned into spatially coherent subdomains, each
block couples its sources to the target set where its kernel columns can be
nonzero (computed *exactly* from the support-ellipsoid gate, so the block
sparsity is lossless and all error is per-block truncation), and each block
stores factors or the verbatim dense block, whichever is smaller.
`BlockLowRank.apply` integrates against the source axis and `applyT` is the
transpose action; if the impulse batches came from forward applies of an
operator A, `apply` approximates A in the nodal kernel sense and `applyT`
approximates Aᵀ (mass matrices stay with the consumer, A = M Φ M — paper
eq. 3.5). A global low rank can then be recovered from the block applies
alone (`randomized_svd`). See the
[compression example](docs/examples/frog_compression.md) for the whole
pipeline on the frog kernel, and `docs/notes/` for boundary effects and the
distributed-format design.

**Covariances must be strictly positive definite**, and this is enforced at
data entry (`add_batch` for per-sample Σ_i, `set_moment_fields` for the
vertex Σ field). Vertex-level validation is enough for the whole domain:
λ_min is concave, so every CG1-interpolated Σ(x) inherits positive
definiteness from the cell's vertices. Fields corrupted by numerical error
are repaired with `psfi.clamp_spd` (eigenvalue flooring; scalar or
per-vertex floors). Choosing the floor is a modelling decision, not just
hygiene — a *near*-singular Σ passes validation but amplifies through
Σ(x)^{−1/2} and det Σ(x); the square of the local mesh spacing is a
reasonable default (an impulse response can't be resolved below the mesh
scale anyway), and genuinely uninformative regions belong to the V field,
which drives the kernel to zero continuously under the `volume` scalings.

## Quick look (Python)

```python
import numpy as np, psfi

F = psfi.ImpulseResponseField(vertices, cells)   # (nv, d), (nc, d+1); points are rows
F.set_moment_fields(V, mu, Sigma)                # (nv,), (nv, d), (nv, d, d)
F.add_batch(points, psi, V_i, mu_i, Sigma_i)     # one Dirac-comb response per batch

cfg = psfi.EvalConfig(frame=psfi.Frame.whitened_affine,
                      scaling=psfi.Scaling.volume_det,
                      tau=3.0, num_neighbors=10)
indices, points, values = F.predictions(y, x, cfg)   # per-neighbor predictions at (y, x)

K = psfi.KernelEvaluator(F, config=cfg,
                         rbf=psfi.RBFScheme(kernel=psfi.RBFKernel.gaussian, shape=3.0))
value = K(y, x)         # one kernel entry
A = K.block(yy, xx)     # (num_y, num_x) block, threaded

parts = psfi.recursive_bisection_partition(xx, 200)          # source subdomains
B = psfi.block_low_rank(K, yy, xx, parts, rtol=1e-3).matrix  # BRLR format
v = B.apply(u[:, None])[:, 0]                                # ~ operator apply
G = psfi.randomized_svd(B, max_rank=200)                     # global low rank from B
```

## Building and testing

```sh
cmake -S . -B build && cmake --build build -j 4 && ctest --test-dir build
```

(`-j 4`, not `-j $(nproc)` — see the compile-time note below.)

etree is found via `find_package(etree)`, with a pinned FetchContent download
as fallback (for local development against a checkout:
`-DFETCHCONTENT_SOURCE_DIR_ETREE=/path/to/ellipsoid_tree`). Python module:
`pip install .`, or `cmake -B build -DPSFI_BUILD_PYTHON=ON` and use
`build/bindings`. Binding tests (`bindings/tests`) check the C++ against a
pure-numpy reference implementation over every configuration axis.

## Compile time and memory

psfi is header-only and includes Eigen (via etree), so every translation
unit that includes a psfi header pays the full template-instantiation cost —
and psfi is substantially heavier than a bare Eigen include, because the
library's concrete inline function bodies (among them the SVD/QR
factorizations of the low-rank layer) are compiled in every such TU.
Measured on an ordinary 8-core laptop with g++: a file including the
`psfi/psfi.hpp` umbrella costs roughly **50 s and 1.5 GB of RAM** to
compile (`-O0` is no cheaper — the cost is front-end instantiation, not
optimization). Including only the headers you use roughly halves that
(`psfi/kernel_evaluator.hpp` alone: ~23 s, ~1.0 GB).

The practical consequence: **on a memory-limited machine, do not
over-parallelize the build.** Concurrent compile jobs each hold their peak
simultaneously, and a machine pushed into swap by parallel Eigen
instantiation can freeze outright. Keep at least ~2 GB of RAM per job —
`cmake --build build -j N` with N no larger than your RAM in GB divided
by 2 — or set up a precompiled header on your side.

## Locality (a note for distributed use)

The mesh is *a* mesh, not necessarily the whole domain: in a domain-decomposed
setting each rank holds a submesh plus halo and the samples whose ellipsoids
reach it. All queries are answered from local data; "outside the mesh" is
treated as "outside the domain". Callers must provide enough halo that this
identification is correct for the queries they issue — the membership test is
isolated in one place so a global-domain indicator can plug in later.

## References

- N. Alger, T. Hartland, N. Petra, O. Ghattas, *Point spread function
  approximation of high-rank Hessians with locally supported non-negative
  integral kernels*, SIAM Journal on Scientific Computing 46(3), 2024,
  A1658–A1689 — the method this library extracts and extends.
- [etree](https://github.com/NickAlger/ellipsoid_tree) — geometry layer
  (simplicial meshes, kd-trees, ellipsoid intersections, batch picking).

MIT license.
