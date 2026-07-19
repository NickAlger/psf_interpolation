# Vector/tensor-valued kernels — parked design notes

Status: **parked research idea** (discussed Nick + Claude, 2026-07-19; deliberately
deferred in favor of completing the scalar package and the scalar ice-sheet
Hessian work). This document exists so the design is not lost and so that
neither this library nor downstream code drifts in a way that blocks it.

## Motivating applications

1. Ice-sheet basal friction Hessian with **anisotropic** friction (the
   inverted-for field is an SPD tensor field: p = q = 3 components in 2D).
2. Seismic inverse Hessian with **velocity + pressure** parameters (p = q = 2
   coupled fields with genuinely different response footprints).
3. Convection-diffusion transport of **vector/tensor fields** (operator maps
   vector fields to vector fields; rotationally equivariant).

## The mathematical object

For u : Omega_src -> R^p and outputs valued in R^q, the kernel is
matrix-valued, Phi(y, x) in R^{q x p}. The probe-able unit is the impulse
response *at x in source-component direction j*:
phi_{x,j} = Phi(., x) e_j — a q-vector-valued function on the target mesh.
Samples therefore live on Omega_src x {1..p} (a discrete label joins the
location), and each sample's stored data is a q-component CG1 field.

## Target components (q > 1)

Componentwise evaluation is correct, and a native version beats q separate
scalar problems for a structural reason: **all expensive per-evaluation work
is component-independent** (kNN, frame map, gate, point location, and the RBF
factorization — which is applied to q right-hand sides). Cost per entry is
nearly flat in q.

The design question is the moments of a vector-valued response:

- **Shared moments** (one V, mu, Sigma per sample; one transport for all
  components): cheap, and moves all components with the SAME T_i, preserving
  inter-component structure (polarization, tensor coupling patterns). The
  intended native default.
- **Per-component moments**: needed when components have different footprints
  (seismic: velocity vs pressure channels spread at different wave speeds).
  Deliberate scope cut: this is *exactly equivalent* to q scalar fields
  sharing sample locations, which the current library already expresses —
  per-component transport is NEVER to become a native feature. Only the
  shared-moments vector target earns native support (amortization + coherence).

## Source components (p > 1)

**The component index is not an interpolation axis; it is a partition of the
samples.** There is no generic geometric relation between components j=1 and
j=2 at the same point; predicting column j of Phi(y, x) uses only neighbors
labeled j (restriction, not interpolation, over the discrete factor).

The one place a native mechanism pays: **Dirac combs can mix components.**
The packing constraint is only support-disjointness, so one operator
application can recover responses for different components at different
points. Batch picking becomes packing over (point, component) pairs — etree's
picker needs nothing (it packs ellipsoids), and the evaluation library needs
exactly one mechanism: a per-sample component tag plus tag-filtered neighbor
selection. Cost of p components then trades against spatial sampling density
instead of multiplying operator applications.

**Equivariance exception** (research-grade, potentially publishable): when
the operator commutes with rotations (application 3), components at nearby x
are related by conjugation with the local rotation, and the right transport
is the componentwise frame map PLUS a component-space rotation — the tensor
analogue of whitened_affine. Given how decisively whitened_affine won on the
rotating frog (median col err 0.267/0.061/0.036 vs 0.384/0.114/0.046), an
"equivariant frame" should win similarly on rotating vector kernels. A
"vector frog" (component frame rotating with the same theta(x) as the blob)
is the natural test kernel.

## Main math risk: signed cross-blocks

Moment estimation needs non-negativity. Diagonal blocks Phi_jj of
multi-parameter Hessians are plausibly non-negative; cross-blocks
(d^2/dv dp, off-diagonal tensor couplings) are generically signed —
Assumption-3 territory (paper Fig. 7). Mitigations to test empirically:

- **Ellipsoid borrowing**: take a cross-block's support from (the union of)
  the corresponding diagonal blocks' ellipsoids — the coupling cannot outrun
  either channel's footprint.
- Note the probing constraint: matrix-free moment probing can only produce
  per-component *signed* moments (moments of |phi| are not matvec-probeable);
  robust pooled moments are only available a priori by formula.

## Phasing (when unparked)

0. **Experiment first, in the research repo, zero library code**: vector frog
   kernel run as a q x p array of current scalar fields. Answers empirically:
   shared vs per-component moments, cross-block borrowing quality, whether
   mixed-component packing saves what it should.
1. **Vector target, shared moments**: psi becomes (num_target_vertices, q);
   Prediction gains vector values; one RBF factorization, q right-hand sides.
2. **Labeled samples**: optional per-sample source-component tag +
   tag-filtered kNN (enables mixed-component combs). Only if phase 0 shows
   the packing trade matters.
3. Parked with a star: equivariant component frames.

## Planned API shapes (all additive)

```cpp
// Phase 1 (target components):
F.add_batch(points, psi /* (nv_tgt, q) matrix overload */, V, mu, Sigma);
struct Prediction { int sample_index; VectorXd point; VectorXd values; };  // or a parallel type
VectorXd KernelEvaluator::eval_vector(y, x);       // scalar operator() stays
rbf_interpolate(values /* (k, m_rhs) overload */, centers, eval_points, scheme);

// Phase 2 (source components):
F.add_batch(..., source_component /* optional (nb,) int, default all-0 */);
K.block(yy, xx, target_component, source_component);  // or matrix-valued entry API
```

## Additive-extension audit (why no hooks are needed now)

Checked 2026-07-19, per public surface:

- `add_batch(psi VectorXd)`: matrix overload later — additive.
- `Prediction.value : double`: vector variant is a new member/type — additive
  (scalar path untouched).
- `rbf_interpolate(values VectorXd)`: multi-RHS is a new overload — additive.
- `KernelEvaluator::operator() -> double`, `block() -> MatrixXd`: vector
  variants are new methods (`eval_vector`, 3-axis block) — additive.
- `set_moment_fields` / sample moments: SHARED across components by design
  (see scope cut above) — unchanged.
- Validation matrix, batches_normalized, gate, short circuits, source/target
  split, bindings conventions: component-blind — unchanged.
- Batch picking (etree): packs ellipsoids, label-agnostic — unchanged.

Conclusion: nothing in the current API blocks the extension; every extension
point is additive. The protection worth having is the invariants below, not
speculative machinery.

## Drift guards (invariants to hold while the idea is parked)

1. **Geometry/value separation.** Everything geometric — neighbor selection,
   frame maps, gating, mesh location, RBF weights — must depend only on
   geometry, never on prediction VALUES. This separation is what makes the
   vector extension nearly free (one factorization, q right-hand sides).
   Named consequence: the parked per-neighbor-ridge idea must key lambda_i on
   frame distance or shape change (geometric — fine), NOT on value
   magnitudes (would entangle the axes and break vector amortization).
2. **Moments are per-sample and component-shared.** Do not introduce
   anything that ties a moment to "the scalar value" of a sample.
3. **Batch storage stays a CG1 vertex table** (a vector per batch today, a
   (nv, q) matrix later) — no flattened/interleaved formats.
4. **Downstream guidance** (ice-sheet driver and friends): treat kernel
   evaluation output as "a value" that may become a small vector; keep
   component indexing out of file formats and inner assembly ABIs. Scalar
   work needs nothing special — it is the q = p = 1 case of every planned
   signature.
