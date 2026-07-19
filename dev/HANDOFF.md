# psfi — development notes / handoff

Working notes for development. Committed during early development on purpose;
will be cleaned up (or moved out of the repo) before release. Last updated:
2026-07-19 (second session).

## What this is

Standalone extraction of the **impulse-response interpolation** component of
the localpsf paper (Alger, Hartland, Petra, Ghattas, *Point spread function
approximation of high-rank Hessians...*, SISC 46(3) 2024, DOI
10.1137/23M1584745, §5.3), generalized and cleaned. Scope: given impulse
response batches on a CG1 simplex mesh + per-sample/field moments + target
points, evaluate the approximate integral kernel anywhere. Out of scope by
design: applying the operator (moments/Dirac combs stay in consuming
projects) and downstream compressed formats (H-matrix, BRLR, block row
dense). The original research code lives in
`hlibpro_python_wrapper/src/product_convolution_kernel.h` +
`rbf_interpolation.h` (maintainer machine); this repo supersedes it.

Depends on [etree](https://github.com/NickAlger/ellipsoid_tree) (same
maintainer; geometry layer: SimplexMesh, KDTree, ellipsoids, batch picking)
+ Eigen. Infrastructure deliberately mirrors etree: header-only C++17,
pybind11 bindings via scikit-build-core, doctest, show-don't-tell docs
pipeline, version single-sourced in `include/psfi/psfi.hpp`.

## Design decisions (agreed with Nick, 2026-07-18/19)

- **Two independent axes** replace the paper's fixed method: `Frame`
  {identity (4.7), translation (4.8), mean_translation (4.9),
  whitened_affine (new: `T_i(y) = mu_i + Sigma_i^{1/2} Sigma(x)^{-1/2}
  (y - mu(x))`, symmetric PSD roots)} × `Scaling` {none, volume = V(x)/V_i
  (4.10), volume_det = volume·sqrt(det Sigma_i/det Sigma(x)),
  mass-preserving}. Support gate optional (`Support::none` enables classical
  single-impulse-batch PSF interpolation with no moment data at all).
- **Minimal data per configuration**, validated up front
  (`ImpulseResponseField::validate` lists exactly what is missing; see the
  requirement table in its doc comment). `batches_normalized` flag: true =
  paper convention (batches store sum phi_i/V_i).
- **Sample points are arbitrary coordinates**, not mesh vertices; `add_batch`
  takes per-sample moments directly.
- **All covariances strictly SPD at data entry** (add_batch and
  set_moment_fields; lambda_min concavity makes vertex validation cover every
  interpolated Sigma(x)). `clamp_spd` / `clamp_spd_field` repair noisy data;
  floor ≈ local mesh spacing squared recommended; genuinely uninformative
  regions belong to the V field (continuous → kernel → 0), not the SPD cliff.
- **No exact-column snapping** in the evaluator: with smoothing = 0, RBF
  center reproduction makes entries at sample columns exact automatically;
  smoothing > 0 deliberately trades that away. (Note: smoothing is a no-op
  when #centers <= #tail coefficients — exact polynomial regime.)
- **RBF conventions**: u = shape·r/r0 with r0 = neighbor-set bounding-box
  diagonal per call, so `shape` is the paper's C_RBF and scaling stays local.
  The original code had a precedence bug (`d/s*s`) that silently pinned
  C_RBF = 1 (still locally scaled); fixed here, default C_RBF = 3 honored.
  Kernel signs and parameters mirror scipy RBFInterpolator (tests convert:
  gaussian eps = shape/(r0·sqrt2), others eps = shape/r0).
- **MPI-forward commitments** (no MPI code yet): everything instance-local,
  mesh may be a submesh (caller provides halos), "outside mesh = outside
  domain" isolated in one `locate()` hook, flat-array constructors, no
  globals, const thread-safe evaluation, no vertex-index = global-DOF
  assumptions.
- **Symmetric mode** pools forward (col field at (y,x)) and adjoint (row
  field at (x,y)) predictions in displacement coordinates, averaging
  near-duplicate centers (`duplicate_tol`, default 1e-7), then one RBF fit —
  same construction as the paper's FWD/ADJ merge.

## State (all pushed to main, CI fully green)

- `include/psfi/`: config.hpp, impulse_response_field.hpp, moments.hpp
  (clamp_spd), rbf.hpp, kernel_evaluator.hpp; umbrella psfi.hpp with version
  macros (0.1.0).
- Tests: 30 doctest cases / ~420 assertions; 84 pytest tests including a
  pure-numpy reference of the full prediction pipeline over all 48
  frame×scaling×support×normalization combos, scipy RBFInterpolator
  cross-checks, and an evaluator reference (prediction reference + merge +
  scipy-checked RBF).
- Bindings: module `psfi` (dist name `psf-interpolation`, free on PyPI, NOT
  yet published); points-are-rows convention like etree; field holder is
  shared_ptr (evaluators keep fields alive).
- Example + docs pipeline: `examples/frog_kernel.cpp` →
  `docs/examples/frog_kernel.md` via `docs/generate_examples.py`
  (etree-style; CI freshness-checks the markdown, figures exempt).
  Headline result: on the rotating frog kernel, whitened_affine+volume_det
  beats the paper config at every stage (median col err 0.267/0.061/0.036 vs
  0.384/0.114/0.046 at 1/5/10 batches, k=10).
- CI: build+test (g++/clang++), pip wheel (exercises FetchContent fallbacks),
  bindings, ASan/UBSan + TSan, version consistency, docs freshness.

## The stage-5 error hot spot (diagnosed 2026-07-19)

The 5-batch k=10 error maps show a blocky hot spot near (0.2, 0.75). NOT a
coverage hole (nearest-sample distances normal) and NOT ill-conditioning
(Lebesgue constant of the interpolation functional ≈ 1.5–2.4 there).
Mechanism: with ~30 samples, the k=10 neighbors include impulses misrotated
by up to ~45° (theta changes along (1,1); the frog impulse is exactly
translation-invariant along theta-level lines), each with 20–60% individual
column error following a clean error-vs-Δtheta law. The evaluator is a
signed linear functional (sum |w| ≈ 2) of those predictions: usually the
individual errors cancel (k=10 beats k=1 nearly everywhere), but in that
region the neighbor set's misrotation pattern aligns with the weight signs
and errors add, up to the Λ·max bound. k=1 maps show no hot spot; 10 batches
shrink the neighbor radius and it disappears. Diagnostic scripts were
session-scratch (not committed); the example page explains the effect and
shows k=1 vs k=10 maps.

## Recent changes (2026-07-19, second session)

- **Far-field short circuits**: support gate tested before any mesh lookup
  (gated points kept with value 0 even outside the mesh — semantic change vs
  the paper's exclusion rule, documented in config.hpp); whitened gate is one
  shared test per entry; evaluator returns 0 without an RBF solve when all
  predictions are zero. Frog example: 39.5 s -> 21.2 s.
- **Interpolated inverse-sqrt field W**: whitened_affine and volume_det now
  interpolate per-vertex Sigma_v^{-1/2} (stored by set_moment_fields as a
  byproduct of validation) instead of eigendecomposing interpolated Sigma per
  evaluation; det Sigma(x) := det W(x)^{-2}; eval-time SPD backstop removed.
- **Dimension generality is now tested**: 1D + 3D closed-form batteries,
  d = 1..3 RBF checks, numpy reference generalized with d = 1, 3 comparisons.
- **Two-domain generalization LANDED** (2026-07-19, same day as the
  proposal): optional source mesh via constructor overload (default: source
  = target, bit-for-bit unchanged); moment fields live on the source mesh
  with target-valued entries (mu : Omega_src -> Omega_tgt);
  dim_source()/dim_target() replace dim() everywhere incl. bindings
  (num_*_vertices, *_mesh_vertices likewise); frame translation requires
  equal dims (allowed across different same-dimension meshes — Nick's
  overlapping-variables use case); symmetric evaluator requires all-equal
  dims. Tests: 1D-source -> 2D-target closed-form battery + coarse/fine
  same-dim case, C++ and bindings.
- **etree upstream issue found**: SimplexMesh point location misses points
  exactly on the interior cube-diagonal edge shared by all 6 Freudenthal
  tets in 3D (returns -1; violates the closed-simplex convention).
  Vertex-coincident and face-diagonal queries work. Measure-zero for generic
  queries, so not blocking; fix belongs in etree's intersection tolerance
  with its own tests.

## Remaining work (rough order)

1. **API docs**: Doxygen + GitHub Pages, mirroring etree
   (`docs/Doxyfile`, doxygen-awesome theme, deploy workflow). Public API
   prose already lives as `///` comments.
2. **Release wiring**: wheels.yml (cibuildwheel + PyPI Trusted Publishing —
   register project `psf-interpolation`), CITATION.cff, CHANGELOG.md,
   version-consistency extended to CITATION.cff. Copy etree's setup; see
   ellipsoid_tree/dev/HANDOFF.md for the wiring details and the
   repo-name-vs-dist-name gotcha.
3. **Python quickstart notebook** (etree has one; execute-checked in CI).
4. **Port consumers**: localpsf-style driver (the old
   ProductConvolutionKernel/ImpulseResponseBatches Python classes) and the
   GPSF/ymir-adjacent BRLR/GLR projects consume this instead of
   hlibpro_python_wrapper.
5. **Performance, column-major eval**: `predictions()` recomputes target-side
   quantities (locate x, moment interpolation, Sigma(x)^{-1/2} eigh) for
   every (y,x) pair though they depend only on x. A column/batched API
   (fix x, many y) would amortize kNN + target-side work — likely 2–3× for
   the BRLR/H-matrix access pattern. ~10 us/eval today (Release build).
6. **MPI**: the locate() hook → optional global-domain indicator; halo
   documentation; nothing else should need to change.

## Parked ideas

- **Per-neighbor ridge ("trust weights")** — from the hot-spot diagnosis:
  the bordered system currently uses one uniform smoothing lambda; making it
  per-point (`diag(lambda_i)` instead of `lambda I`) lets far/deformed
  neighbors inform the fit without being interpolated through exactly.
  Natural lambda_i: increasing in frame distance |x_i - x| (or a
  shape-change metric like the whitening mismatch). Would damp the
  constructive-interference failure mode while keeping near-neighbor
  exactness. Straightforward in rbf.hpp (values-side diagonal); needs a
  principled default schedule + a study on the frog example.
- **Adaptive k**: k=10 out of ~30 samples reaches ~0.4 away (Δtheta ~ 40°);
  scale k with sample count or drop neighbors much farther than the nearest.
- **Vector/tensor-valued kernels** — full parked design in
  [`dev/VECTOR_TENSOR_EXTENSION.md`](VECTOR_TENSOR_EXTENSION.md): matrix
  kernels, componentwise target with shared moments (one RBF factorization,
  q right-hand sides), source components as labeled samples (restriction,
  not interpolation; mixed-component Dirac combs), equivariant component
  frames as the whitened_affine analogue, signed cross-block moment risk.
  Additive-extension audit done: no hooks needed; hold the geometry/value
  separation invariant.
- **Symmetric-merge tolerance**: `duplicate_tol` is absolute (1e-7),
  inherited from the paper code; a scale-relative tolerance may be better.

## Gotchas (hard-won)

- **ALWAYS configure with `-DCMAKE_BUILD_TYPE=Release`** (or RelWithDebInfo).
  An empty build type cost a factor ~100 in the example (~700 us/eval vs
  ~10 us) and blew a 10-minute timeout before being caught.
- Local dev against the etree checkout:
  `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DFETCHCONTENT_SOURCE_DIR_ETREE=$HOME/repos/ellipsoid_tree`.
  Installed-etree route also works (`find_package`); PSFI_INSTALL requires it
  (exporting against vendored etree is a hard error by design).
- Python on Nick's machine: use the **t3toolbox** env
  (`-DPython_EXECUTABLE=$HOME/miniconda3/envs/t3toolbox/bin/python`,
  `PYTHONPATH=build/bindings pytest bindings/tests`); the fenics env lacks
  pytest. For `pip wheel .` prepend `env CC=gcc CXX=g++` (stray env vars
  point at a broken compiler). etree's own python module (for cross-checking
  / the batch picker) is at
  `~/repos/ellipsoid_tree/build/bindings` (also cpython-311).
- Docs regen: `python3 docs/generate_examples.py --build-dir build` and
  commit; CI diffs the .md only (figure bytes exempt). The frog example runs
  ~40 s wall on 8 threads (~5 CPU-min). Example stdout must stay
  deterministic across toolchains — order statistics printed at %.3f have
  proven stable; avoid printing raw FP sums.
- If example figure *names* change, delete the stale files in `docs/img/`
  by hand — the generator copies but never removes.
- Claude session context for this project lives in the auto-memory of the
  `nicks_research_experiments` project
  (`~/.claude/projects/-home-nick-repos-nicks-research-experiments/memory/psf-interpolation-package.md`),
  since sessions have been run from that repo.

## Quick commands

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -DFETCHCONTENT_SOURCE_DIR_ETREE=$HOME/repos/ellipsoid_tree \
      -DPSFI_BUILD_PYTHON=ON -DPython_EXECUTABLE=$HOME/miniconda3/envs/t3toolbox/bin/python
cmake --build build -j $(nproc) && ctest --test-dir build
PYTHONPATH=build/bindings ~/miniconda3/envs/t3toolbox/bin/python -m pytest bindings/tests -q
python3 docs/generate_examples.py --build-dir build   # regenerate example docs
```
