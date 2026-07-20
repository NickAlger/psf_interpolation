# MPI strategy for the block low rank format — design discussion record

Discussed with Nick 2026-07-19 (end of the downstream-matrix session);
**no decisions yet** — the open questions at the bottom are awaiting his
answers. Nothing here is implementation. Companion user-facing note:
`docs/notes/distributed_design.md` (the mechanics of blocks-as-the-unit
and the two scatters); this note is about *what the library should own*.

## The four layers

1. **Local numerics** — kernel evaluation, target sets, per-block
   compression. Done, and already the entire per-rank computation:
   `block_low_rank` on a rank's own sources with a local evaluator needs
   zero communication given the right local data.
2. **Index/ownership bookkeeping** — deriving, from block id arrays plus a
   dof-ownership map, exactly who sends what to whom for apply (reverse
   add-scatter) and applyT (forward gather). Pure integer logic — and the
   place every historical bug lived (the ancestor's MPI_Send-count
   corruption, the unzeroed ADD_VALUES output, the three-orderings
   confusion).
3. **Transport** — moving the bytes: raw MPI, PETSc VecScatter, mpi4py.
   Trivial once layer 2 is right, and consumer-specific (ymir will always
   want its own MatShell over PETSc Vecs).
4. **Data staging & solver integration** — mesh distribution, halo
   assembly (batches and moments come from the consumer's operator applies
   on the consumer's dof layout), and everything downstream of the matvec.
   Cannot live in psfi without psfi becoming a framework.

## Options considered

- **A. Fully MPI-free library + recipes** (status quo). No new
  dependency, no mpirun CI; but every consumer rewrites ~200 lines of
  scatter-derivation glue — which is layer 2, the proven bug generator.
- **B. Optional psfi/mpi.hpp with a turnkey DistributedBlockLowRank.**
  Turn-key for frameworkless consumers, but psfi takes on communicator
  lifecycle + MPI CI, must parameterize the distributed vector layout
  (the consumer's dof ordering) anyway, and the consumers we actually
  have would bypass its matvec for their own MatShell. High cost, low
  uptake.
- **C. Transport-agnostic ScatterPlan: psfi owns layer 2, MPI-free.**
  Computed from plain data (block id arrays + ownership map): per-peer
  send/receive index lists for both directions, insert-vs-accumulate
  semantics, the zero-before-accumulate invariant, and the local
  gather/GEMM/scatter schedule. Consumers execute it on their transport
  (the lists are literally what a VecScatter is built from, or a dozen
  lines of Isend/Irecv, or numpy + mpi4py). Killer feature: testable
  WITHOUT MPI — N simulated ranks in one process, memcpy transport,
  assert distributed apply == serial apply bitwise; the whole class of
  historical scatter bugs becomes an ordinary unit test in ordinary CI.
- **D. psfi orchestrates distributed construction end to end.** Rejected:
  halo assembly requires knowing where mesh/batches/fields come from —
  the consumer's solver stack.

## Recommendation (Claude's, pending Nick)

**C layered on A; B kept open but not promised; D never.** Principle:
psfi owns everything that is deterministic integer/linear-algebra logic
and nothing that touches a communicator. The future distributed-enablement
phase, all MPI-free and serial-CI-testable:

1. **Global-id affordance in the builder** (tiny): let blocks carry global
   dof ids — an optional relabeling argument or a post-hoc remap helper
   (blocks are plain data).
2. **ScatterPlan** built from (my blocks' global ids, ownership map). v1
   contract: build from replicated metadata (id arrays are metadata-sized;
   the consumer Allgathers them once), with a documented handshake
   alternative for very large rank counts.
3. **Simulated-rank test harness**: split the frog problem across k fake
   ranks, in-process transport, equality with the serial BlockLowRank —
   including deliberate regression tests for the unzeroed-accumulate and
   duplicated-target cases.
4. **Recipes remain the interface for layers 3/4**:
   docs/notes/distributed_design.md grows a PETSc-realization section
   (plan fields -> VecScatter construction; symmetrization as
   (apply+applyT)/2) and an mpi4py sketch. The ymir port then DELETES the
   hand-rolled index derivation instead of reimplementing it.

Distributed construction needs nothing beyond the global-id affordance
(embarrassingly parallel given consumer-staged halos — the library's
long-standing locality contract). Distributed GLR/eigensolves stay
downstream. If a frameworkless consumer materializes, psfi/mpi.hpp
becomes ~150 lines of plan execution over already-tested logic; the
mpi4py route may serve that niche better anyway (plan arrays are already
Python-visible through the bindings layer).

## Open questions (Nick to decide; no implementation until then)

1. **Ownership-map shape**: plain `owner[global_target_id]` array
   (simplest, replicated) vs contiguous per-rank ranges (smaller, but
   imposes an ordering convention on consumers).
2. **Replicated-metadata plan construction as the v1 contract** — is it
   acceptable for the rank counts foreseen (hundreds vs thousands)?
3. **Consumer-first design forcing function**: design the plan API by
   writing ymir's VecScatter construction against it on paper first
   (Claude advocates yes — same method that produced the source/target
   dictionary).
