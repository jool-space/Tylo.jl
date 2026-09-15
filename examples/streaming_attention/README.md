# Streaming forward attention

A complete multi-head BF16 attention kernel for head dimension 64. Q/K/V use
BF16 storage; row statistics and the weighted output accumulate in FP32.
Sequence lengths and head counts are runtime integers. An optional Boolean
mask, causal masking, query/key tails, empty key sets and fully masked rows are
supported. Inputs must keep dot products and weighted sums in FP32 range;
valid scores must be finite.

Each CTA owns 128 query rows of one head as two groups of four warps; every
warp owns 16 rows with all 64 score columns of a key tile and all 64 output
columns. Q is loaded once by TMA and held in registers as A operands. K and V
tiles of 64 keys arrive by TMA through four shared stages, one thread issuing
three tiles ahead while its group is off the tensor pipe, and each thread waits
on the stage's mbarrier. The two groups alternate on the tensor pipe through
two named barriers: block b of a group issues PV of tile b-1 and QK of tile b,
then hands the pipe to the other group and runs its softmax. Tails are
zero-filled by TMA; nothing is copied by scalar loops. No global score or
probability workspace exists; a supplied Boolean mask occupies M×N bytes.

The block loop is visible in [kernel.jl](kernel.jl):

```julia
(b>0 || g==1) && ptx"bar.sync"(1+g,256)            # my turn on the tensor pipe
b>0 && (out=weighted_values(config.output,w,v_stage,out,lane))
wait_tile(full,b)
s=scores(config.scores,qf,k_stage,lane)
ptx"bar.arrive"(2-g,256)                           # the other group's turn
s=plain ? s .* 0.125f0 : mask_scores(s,mask,wtid,row,key,m,n,Val(Causal))
update=softmax_update(state,s)
out=out .* update.rescale
w=packed_weights(atom,update.weights)
state=update.state
```

`packed_weights` converts adjacent FP32 score-result atoms into BF16 A
operands with `pack_operand_a`, so the weights of a tile occupy 16 registers
until its PV block. B operands of both products load in pairs with one
`ldmatrix.x4`, derived from a two-atom ownership.

Physical Julia arrays are Q `(64,M,H)`, K `(64,N,H)`, V `(ldv,64,H)`, mask
`(N,M)` or `nothing`, and output `(64,M,H)`; matrices are one head. V's leading
dimension is a multiple of eight (a TMA stride rule) and may exceed the logical
N; padding is outside the logical key range. All arrays are contiguous
column-major. `prepare` uploads the TMA descriptors once; keep its result alive
for the launches and graph replays that use it:

```julia
include("examples/streaming_attention/kernel.jl")
bindings = StreamingAttention.prepare(q, k, StreamingAttention.pad_values(v))
StreamingAttention.launch!(output, bindings, mask; causal=false)
StreamingAttention.launch!(output, q, k, v)   # prepares per call, no mask
```

An entirely masked row produces zero. With `causal=true`, key index j is valid
only if j≤query index i (top-left alignment for rectangular shapes); the
schedule skips key tiles entirely above the diagonal.

## Rounding

The running denominator sums FP32 exponentials. The PV numerator uses those
unnormalized weights rounded to BF16, separately for each 64-key tile. After
streaming, divide the numerator by the final denominator. This is different
from rounding already normalized full-row probabilities to BF16. Exponentials
are `exp2` of a scaled argument, within two ulp.

[reference.jl](reference.jl) contains an independent Float64 attention reference
and a diagnostic reference with the same per-tile BF16 boundary. Tests use
absolute error bounds as well as relative criteria, including cancellation
near zero. No bitwise equivalence to materialized softmax is claimed.

## Run and measure

From Tylo's root with Julia 1.12 and compatible PTX checked out beside Tylo:

```sh
julia --project=examples/streaming_attention -e 'using Pkg; Pkg.instantiate()'
julia --project=examples/streaming_attention examples/streaming_attention/run.jl /tmp/tylo-attention-results
```

Use a fresh output directory; add `quick` for a subset of cases. The dated
validation report records the measurements. The GPU test suite includes this
kernel; `test/tools/sanitize.jl` includes its copy/compute/reuse and replay
workload.

The benchmark compares against materialized cuBLAS QK and PV (strided batched
over heads) with a scalar warp softmax between them. It fixes
`CUBLAS_COMPUTE_32F` and `DEFAULT_MATH`, BF16 inputs, FP32 GEMM outputs, and a
BF16 normalized-probability buffer. Scalar device references are prepared
before graph capture and retained throughout replay. Both paths use the same
inputs, effective mask, scale, and FP32 output. It is a comparison with this
baseline, not a tuned FlashAttention library; TFLOPS count four flops per valid
query/key pair and head dimension, so causal cases count half the pairs.

Results retain all interleaved timing samples, numerical errors, registers,
local memory, and explicit workspace bytes. Compilation/first execution,
preparation and transfers are outside warm graph timings. The baseline's
explicit score/probability storage is 6MNH bytes; library-internal workspace
and common inputs/output/mask are excluded from that number.

On the GB10 the SM runs at about 0.9 GHz under this load and an `mma.sync`
micro-benchmark reaches 45 TFLOPS; this kernel reaches 23 TFLOPS on
1024×1024×16 heads unmasked and 26 on 4096×4096×4 heads. Single-head cases below 2048 queries occupy a
fraction of the 48 SMs. It does not implement backward, dropout, other head
dimensions, or decode scheduling.
