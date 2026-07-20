# Distributed block low rank: the design ellipsoid_psf is shaped for

Status: design note — ellipsoid_psf contains no MPI code, and by design will contain
at most a thin optional layer. But the `BlockLowRank` format was shaped so
that distributing it is wiring, not redesign. This note records the
mapping.

## Blocks are the unit of distribution

The format already speaks in global ids: each block holds its `source_ids`
(disjoint across blocks — the owned degrees of freedom of one subdomain),
its `target_ids` (where that subdomain's kernel columns reach — its
"extension", overlapping between blocks), and its factors. A rank owns a
small number of blocks; nothing in a block refers to anything outside it
except through those global ids. A block serializes as two int arrays, one
or two matrices, and a flag (the Python-constructible `Block` mirrors this
exactly), so shipping blocks between ranks is trivial.

## The two applies become two scatters

- `apply` (source-values in, target-values out): each rank gathers its
  blocks' inputs at `source_ids` — locally owned, no communication —
  runs its local GEMMs, and accumulates into the global output at
  `target_ids`. Targets overlap between ranks, so the output assembly is
  a **reverse scatter with add semantics**, and the output vector must be
  zeroed first (a lesson paid for in this format's ancestry: an unzeroed
  add-scatter survives every fresh-vector test and corrupts the first
  reused one).
- `applyT` (target-values in, source-values out): a **forward gather with
  insert semantics** of the input at `target_ids` (duplication across
  ranks is fine — everyone reads), local GEMMs, then a plain write to the
  owned `source_ids` — disjoint, so no communication on the output side.

In a PETSc realization this is one `MatShell` holding two `VecScatter`s;
symmetric consumers (Hessians) register `(apply + applyT)/2` as both the
multiply and the transpose-multiply, since it is its own transpose.

## Construction is rank-local, given halos

Building a block needs kernel evaluations only at (its target set) x (its
sources), and the support oracles need moment data only near its sources.
With the locality semantics of the library (each rank holds a submesh plus
enough halo — see the README's locality note), the whole construction runs
rank-locally: local partition of owned sources, local `EllipsoidTree`
queries for the target sets (the candidate target points must include the
halo, since extensions reach beyond the owned subdomain), local per-block
compression. There is deliberately no gather-to-root step anywhere — the
serial-construction bottleneck was the main scalability limit of this
format's research ancestor.

## What remains genuinely global

Only two things: agreeing on the global source partition (each rank
partitioning its own dofs suffices), and the two scatter patterns, which
are read off the blocks' id arrays once after construction. Both are
metadata-sized. Everything expensive — kernel evaluation, compression,
factor storage, GEMMs — is per-block and stays where the block lives.
