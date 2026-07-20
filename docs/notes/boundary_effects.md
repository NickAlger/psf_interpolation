# Boundary effects: sample exclusion, kernel discontinuities, and compressibility

Status: recorded 2026-07-19 while building the low-rank compression tests.
Boundary artifacts have been a recurring source of error across this line of
work; this note pins down one specific mechanism now observable in ellipsoid_psf, so
it can be thought through properly later.

## The mechanism

When a prediction transports an evaluation point with the frame map,
`T_i(y)` can land outside the target mesh. ellipsoid_psf then applies two different
rules (see `config.hpp` and `ImpulseResponseField::predictions`):

- **Gated** (`Support::ellipsoid`, `T_i(y)` outside the ellipsoid `E_i`):
  the sample is *kept* with prediction value 0, wherever `T_i(y)` lies.
  "phi is zero beyond tau standard deviations" is knowledge from the
  support model, valid regardless of domain membership. This is continuous.
- **Ungated but off the mesh** (`Support::none`, or inside `E_i` but
  outside the mesh): the sample is *excluded* from the neighbor set — the
  impulse response was never observed there, so the prediction cannot be
  formed. Exclusion changes the RBF fit discontinuously: the kernel jumps
  across the surface in `(y, x)` space where `T_i(y)` crosses the mesh
  boundary.

So the kernel is smooth away from these exclusion surfaces, and:

- with `Support::none`, exclusion surfaces run wherever transported points
  exit the mesh — for translation-type frames, a band of width comparable
  to the sample-to-query distance along the entire boundary;
- with `Support::ellipsoid`, exclusions are confined to where support
  ellipsoids overhang the domain boundary (interior ellipsoids never
  trigger them).

## The observed effect on compressibility

Found while constructing the `kernel_low_rank` tests (translation frame,
Gaussian-bump batches on an 8x8 grid mesh of the unit square, 81 targets x
36 sources, `Support::none`):

- Evaluation windows spanning the domain (transported points frequently
  exit the mesh): singular values barely decay — relative Frobenius tail
  still 2.7e-2 at rank 30 of 36. The matrix is effectively **full rank at
  any useful tolerance**.
- Windows chosen so every transported point stays inside the mesh: normal
  algebraic decay (limited by the CG1 kinks of the mesh interpolation),
  tail 1e-2 by rank ~16 of 36.

Jump discontinuities put slowly-decaying components into the singular
spectrum, on top of whatever pointwise error the exclusions represent. For
the block-low-rank format this means: blocks whose targets or transported
points interact with the boundary in ungated configurations will carry
silently inflated ranks. The `hit_max_rank` diagnostics on the compression
routines are the guard rail that makes this visible.

## Related boundary artifacts (context)

This is one member of a family. Two others recur in applications of this
method and are handled upstream of ellipsoid_psf: impulse responses truncated by the
domain boundary corrupt moment estimates (means pulled inward, covariances
shrunk along the boundary normal), and near-boundary batch functions carry
mass the Dirac-comb model attributes incorrectly. ellipsoid_psf assumes clean
moments at data entry; those corrections stay in consuming projects.

## To think through later

- Is exclusion the right rule? Keeping the sample at zero would be
  continuous but biased (the true impulse response is generally nonzero
  just outside the observed mesh); excluding is less biased pointwise but
  discontinuous. A third option is a per-neighbor down-weighting (the
  parked "trust weights" ridge idea in `dev/HANDOFF.md`) that fades a
  sample out as `T_i(y)` approaches the boundary instead of dropping it.
- Meshes slightly larger than the region of interest (a halo of observed
  impulse-response data) eliminate the exclusions for interior evaluation;
  this interacts with the planned MPI/halo semantics.
- Whether the support oracles / block target sets should intersect their
  ellipsoids with the domain, and whether that changes the lossless-
  sparsity invariant near the boundary.

Until then, practical guidance: prefer `Support::ellipsoid` with tau chosen
so ellipsoids do not badly overhang the boundary; expect degraded
compression (not wrong answers) where they do; and watch `hit_max_rank`.
