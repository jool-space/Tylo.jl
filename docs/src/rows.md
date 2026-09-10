```@meta
CurrentModule = Tylo
DocTestSetup = :(using Tylo)
```

# Row reductions and broadcasts

A row operation combines register values according to **logical ownership**.
Its communication cost depends on how the row is distributed, independently
of the memory layout used to load the values.

| Fragment | Row ownership | Reduction communication | Result per thread |
|:--|:--|:--|:--|
| `RowFragment` | One complete row per lane | Local register arithmetic | One row result |
| `WarpRowFragment` | One row striped across 32 lanes | Five butterfly shuffle rounds | One result, replicated across the warp |
| MMA accumulator | Two rows per four-lane group per M repetition | Two shuffle rounds per row after local reduction | Two results per M repetition, replicated across four lanes |

`row_sum` and `row_max` currently operate on FP32 values. Convert a lower
precision fragment explicitly with `map(Float32, fragment)` where supported.
The reduction tree is balanced locally; its rounding need not match a
sequential sum. Distributed reductions use `PTX.Warps.warp_reduce`.

The returned `RowValues` carries row ownership. `row_map(op, fragment, values)`
applies `op(element, row_result)` without changing which thread holds an
element. It selects an already replicated result; it does not shuffle values
into a new distribution. A result for a warp-striped row cannot be used to
broadcast onto lane-local rows.

```jldoctest
julia> f = RowFragment((1f0, 2f0, 3f0));

julia> only(row_sum(f))
6.0f0

julia> row_map(-, f, row_max(f)).data
(-2.0f0, -1.0f0, 0.0f0)
```

`row_coordinate(result, thread, Val(e))` returns the zero-based logical row
for result slot `e`. `Layouts.coordinate(Layouts.layout(acc), thread, Val(e))`
returns the row and column of a tiled MMA accumulator's flattened register
slot. The latter mapping also drives masks and epilogues.

## Participation and scope

All 32 lanes of each warp execute distributed reductions, including the
four-lane MMA reductions. Bounds masks select data and neutral values; they
must not make participation conditional. Use zero for a sum and `-Inf32` for
a maximum over padded entries. There is no implicit CTA barrier.

Tiled MMA supports repetitions in M and N, but row reduction currently
requires **one warp along N**. Multiple warps along M own independent rows.
With multiple N warps a complete row needs communication between warps;
Tylo rejects that reduction instead of returning an unlabeled partial result.

Megakernels' normalization uses `WarpRowFragment((partial_sum,))` to combine
the contributions within each of its eight consumer warps. Its shared scratch,
worker barrier, and combination of eight partial sums remain in Megakernels.
The residual storage and rounding contracts also remain operation-specific.

## Worked softmax

The [softmax example](https://github.com/jool-space/Tylo.jl/tree/main/examples/softmax)
contains lane-local and warp-striped kernels and a fused epilogue over actual
MMA output. All use the same arithmetic:

```julia
shifted = row_map((x, m) -> m == -Inf32 ? -Inf32 : x - m, f, row_max(f))
weights = map(exp, shifted)
result = row_map((x, s) -> s == 0f0 ? 0f0 : x / s, weights, row_sum(weights))
```

Valid logits must be finite. Explicit masks supply `-Inf32`; masked output
entries are zero and an entirely masked row produces zeros. Other NaN or
infinite valid logits are outside this example's contract. The MMA example
normalizes the complete row of one output tile, not rows spanning independently
scheduled tiles, and does not implement an attention kernel.

The standalone kernels use physical `(columns, rows)` matrices. Warp-striped
ownership gives adjacent lanes adjacent addresses in this storage. Lane-local
ownership illustrates another distribution but does not guarantee competitive
memory traffic or register usage for wide rows. `examples/softmax/run.jl`
compares both against a simple scalar warp kernel with the same storage and mask.
