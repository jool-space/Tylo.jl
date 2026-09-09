# Next batch: reductions, broadcasts, and boundary tiles

Scoped 2026-09-10. This is an execution plan; the work below is not yet implemented.

## Outcome

Make Tylo useful for normalization and softmax as well as matrix multiplication.
Deliver reusable row reductions and broadcasts, complete standalone kernels, and
an actual Megakernels consumer. Then extend the existing warp GEMM to boundary
tiles. All required runtime acceptance can happen on the local GB10.

Work in the order below. Finish and validate each milestone before expanding its
API. If the primary work finishes early, continue to the next milestone; the
optional migration is the last item, not a competing redesign.

## 1. Row operations with explicit ownership

Introduce the smallest row-result representation and reduction/broadcast API
needed by the consumers below. Names should follow the existing Julia interface
once the first kernels make the contracts concrete.

Support three specific distributions:

- `RowFragment`: one logical row's values are local to a lane.
- Warp MMA accumulators: a row is distributed across the atom's lane group;
  support repeated atoms within a warp as needed by the example.
- A row striped over a warp: required for Megakernels' existing normalization,
  where each lane accumulates a subset of the features.

The third case matters: normalization currently combines eight warp partials
through caller-owned shared scratch. Supporting only lane-local rows and MMA
fragments would not establish reuse in that consumer.

Provide FP32 row sum/max and application of a row result to its corresponding
values, sufficient for subtraction, scaling, and normalization. Represent which
logical rows a result belongs to and where it is replicated. A broadcast must
preserve ownership or explicitly perform communication. Support known mappings;
an arbitrary-layout reduction compiler is not required.

Keep local arithmetic, shuffle communication, and shared-memory communication
identifiable in the implementation and documentation. A metadata view cannot
redistribute values. Specify collective participation independently of data
validity; invalid values contribute identities while participating lanes still
execute the collective. Reject unsupported distributions clearly.

Acceptance:

- CPU coordinate oracles cover every lane/value and resulting logical row for
  the supported mappings, including atom repetition and row-result replication.
- GPU reductions and broadcasts agree with independent references on asymmetric
  data that exposes lane or row mixups, cancellation, and non-power-of-two widths.
- Small representative probes show the intended shuffle groups and no accidental
  local-memory materialization from dynamic tuple indexing.
- Host-only loading continues to work without CUDA.

## 2. Complete softmax examples and Megakernels normalization

Build standalone row-softmax examples using lane-local and distributed rows,
including a warp-MMA accumulator consumer. At least one example must actually
produce the accumulator with MMA before applying softmax. This is a tile-local
softmax example; do not imply it implements full attention or normalizes across
independently scheduled output tiles.

Use max subtraction and FP32 accumulation. Support finite valid inputs and
explicitly masked entries; masked entries produce zero and a fully masked row
produces zeros. Define this behavior before implementation. Document any other
non-finite-input limits instead of silently inheriting incidental shuffle/max
behavior. BF16/FP16 storage should use their Julia element types.

In Megakernels, migrate the shared `RMSNorm`/`ResidualNorm` implementation and
`AddRMSNorm` to the same row-reduction vocabulary. Preserve the scheduler's
scratch ownership, dependency waits, worker barrier, and eight-warp combination.
Keep the cross-warp step explicit; a new CTA synchronization framework is not
needed to replace the warp-local reduction.

Preserve the operations' distinct numerical contracts:

- `ResidualNorm` rounds the updated residual to its storage type before computing
  the norm and exposes that rounded residual to later operations.
- `AddRMSNorm` reduces the unrounded FP32 sum already kept in shared scratch.

A different reduction order may change rounding; validate against the correct
operation reference with justified tolerances, not bitwise equality by default.
Do not migrate unrelated attention or projection reductions as incidental cleanup.

Acceptance:

- Softmax checks include shifted logits, extreme finite ranges, masks, entirely
  masked rows, irregular widths, and correct normalization of valid rows.
- Normalization checks include widths below a warp, irregular tails, multiple
  batches, residual rounding cases, and replay with changed input values.
- Existing Megakernels executor and dependency tests remain green, with focused
  integration coverage for the changed operations.
- Run matching-toolkit memcheck, racecheck, and synccheck on representative new
  kernels and the normalization integration. Preserve caller synchronization.
- Benchmark against simple standalone baselines and the pre-change Megakernels
  implementation on fixed inputs and launch settings. Report latency and
  resource differences even if there is no speedup.

This is the primary deliverable: two packages using the same row operations,
with readable examples and measured evidence.

## 3. Predicated copies and ragged warp GEMM

Extend the existing BF16/FP16 warp GEMM example to arbitrary positive M, N, and K
within its existing datatype and layout scope. Derive validity from logical
coordinates using the same partitioning as the data. Preserve the distinction
between a tile's static capacity and its runtime valid extent.

Keep the aligned interior copy path. Handle partial vectors with an explicit,
correct boundary path: use supported zero-fill copy forms where applicable and
a scalar fallback where alignment or a partial element vector requires it.
Check the PTX contract before choosing the instruction path. Zero every invalid
shared-memory element consumed by MMA, predicate output stores, and retain full
collective participation. Test nonzero windows without losing swizzle phase.

Acceptance:

- Cover independent M/N/K tails, dimensions smaller than a tile, one- and
  two-stage pipelines, plain/swizzled shared layouts, and padded leading strides.
- Compare with an independent numerical reference and use output sentinels plus
  sanitizer runs to detect invalid reads/writes and stale shared-memory tails.
- Recheck representative aligned code generation and timing; explain any
  regression. Measure boundary overhead separately from aligned throughput.
- Document the coordinate/mask contract with one worked edge-tile example.

## 4. Optional: migrate the legacy warp projection

After milestones 1–3 are complete, assess `AsyncProjection` in Megakernels as a
second consumer of Tylo's warp MMA and copy primitives. Migrate it only if the
existing APIs express its ownership and pipeline without inventing a new general
framework. Preserve GEMM/gate-up behavior and the scheduler's synchronization.

Require runtime comparisons, resource inspection, and before/after timings for
this migration too. Remove private tile helpers only when all their real users
have moved. If it exposes a larger missing abstraction, write the concrete gap
and leave this migration for the following batch.

## Execution and evidence

Use an isolated test environment with the last validated PTX revision
`32e36c122bc1c7af5f171cf478324b628b06af3a`, or deliberately update that pin with
fresh validation if a needed capability requires it. Do not consume unrelated
in-progress PTX checkout changes as a moving dependency.

Run focused checks while iterating, then the relevant complete Tylo and
Megakernels suites once the batch settles. Existing Hopper/TMEM assembly tests
remain regression checks; their execution status stays hardware-pending.
Keep validation proportional to changed behavior rather than maximizing counts.

Produce a dated report with exact source/dependency revisions or hashes, commands,
correctness and sanitizer results, compiler/resource evidence, and reproducible
warm benchmark results. Separate compilation, packing, and transfer costs from
steady-state kernel timing; distinguish individual operations from whole-program
measurements. End with a short explanation of which abstraction became reusable,
where specialization remains, and what the measurements justify doing next.

The main work needs no Hopper rental. Keep the existing TMA/WGMMA runtime suite
ready for H100/H200, and TMEM/tcgen05 execution for datacenter Blackwell. Further
WGMMA pipeline tuning, full attention, automatic pipelines, generic register
redistribution, new numerical formats, and extracting Laythe are outside this
batch. Recover old layout-algebra ideas only when a concrete operation needs them.
