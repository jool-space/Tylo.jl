"""
    WGMMAAtom(T, n, k=64, partials=1)

Hopper shared/shared warpgroup MMA: 128 threads compute a 64×n×k product,
accumulating in FP32. BF16 or FP16 inputs take 16 of K per instruction, FP8
(`Float8_E4M3FN`, `Float8_E5M2`) 32: each instruction covers 32 bytes of K,
and `k` covers one, two or four instructions (16, 32 or 64 for 16-bit
inputs; 32, 64 or 128 for FP8). n=8:8:256. Each instruction updates one of
`partials` independent accumulators cyclically. Partials must divide the
instruction count; total registers per thread must not exceed 128.
Shared operands are K-major, or, for 16-bit inputs, MN-major (transposed
by the hardware). A 16-bit A may also come from registers: a `Fragment` in
`operand_layout(atom, OperandA())`, the accumulator's layout for 64×k, so
an FP32 result converted to BF16 is the next product's A without moving
values between threads.
`partials=4` at n=8,k=64 preserves Cohere's four independent K chains.
The sizes must be constants. With one partial, the ready accumulator
(from `zero_accumulator` and `wait_mma`) is a `Fragment` in
`operand_layout(atom, Accumulator())`, to compute with, rescale or store like
any other; with several, `finish_mma` sums them into one. No allocation or
CTA barrier.
"""
struct WGMMAAtom{T,N,K,P} <: Atom
    function WGMMAAtom{T,N,K,P}() where {T,N,K,P}
        T in (BFloat16,Float16,Float8_E4M3FN,Float8_E5M2) || throw(ArgumentError("BF16, FP16 or FP8 (E4M3, E5M2) required"))
        N isa Int && 8 <= N <= 256 && N % 8 == 0 && K isa Int && K*sizeof(T) in (32,64,128) &&
            P isa Int && P > 0 && (K÷_instruction_k(T)) % P == 0 && N÷2*P <= 128 ||
            throw(ArgumentError("invalid WGMMA shape or partial accumulator count"))
        new{T,N,K,P}()
    end
end
Base.@constprop :aggressive @inline WGMMAAtom(::Type{T},n::Integer,k::Integer=64,partials::Integer=1) where T =
    WGMMAAtom{T,Int(n),Int(k),Int(partials)}()
Base.size(::WGMMAAtom{T,N,K}) where {T,N,K} = (64,N,K)
threads(::WGMMAAtom) = 128
Base.eltype(::WGMMAAtom{T},::Union{OperandA,OperandB}) where T = T
Base.eltype(::WGMMAAtom,::Accumulator) = Float32
_wg_registers(::WGMMAAtom{T,N,K,P}) where {T,N,K,P} = N÷2*P
# One instruction covers 32 bytes of K.
_instruction_k(::Type{T}) where T = 32 ÷ sizeof(T)
struct WGMMAAccumulator{P,R}
    data::NTuple{R,Float32}
    function WGMMAAccumulator(p::P,data::NTuple{R,Float32}) where {P<:WGMMAAtom,R}
        R == _wg_registers(p) || throw(ArgumentError("wrong WGMMA register count"))
        new{P,R}(data)
    end
end
struct PendingWGMMA{P,R}
    data::NTuple{R,Float32}
end
# A ready accumulator: with one partial, a Fragment in the accumulator
# layout, like any other register tile; with several, their registers, which
# `finish_mma` sums.
@inline _ready(p::WGMMAAtom{T,N,K,1},data) where {T,N,K} = Fragment(data,operand_layout(p,Accumulator()))
@inline _ready(p::WGMMAAtom,data) = WGMMAAccumulator(p,data)
@inline zero_accumulator(p::WGMMAAtom) = _ready(p,ntuple(_ -> 0f0,Val(_wg_registers(p))))

# PTX ISA wgmma accumulator figure: lane groups t%4 select column pairs,
# (t%32)÷4 the row, warps 16-row bands; slots alternate column, row half,
# 8-column block.
function _operand_layout(::WGMMAAtom{T,N},::Accumulator) where {T,N}
    L, s = Layouts.Layout, Layouts.static
    Layouts.Ownership((64,N),L(((s(4),s(8),s(4)),(s(2),s(2),s(N÷8))),
                               ((s(128),s(1),s(16)),(s(64),s(8),s(512)))))
end

# A 16-bit A in registers holds 64×K in the accumulator's layout: for each
# instruction's 16 columns, the pairs a0a1 (row r), a2a3 (row r+8), a4a5 and
# a6a7 (the next 8 columns), as in the PTX ISA's register fragment figure.
function _operand_layout(::WGMMAAtom{T,N,K},::OperandA) where {T,N,K}
    sizeof(T) == 2 || throw(ArgumentError("only 16-bit A operands come from registers"))
    L, s = Layouts.Layout, Layouts.static
    Layouts.Ownership((64,K),L(((s(4),s(8),s(4)),(s(2),s(2),s(K÷8))),
                               ((s(128),s(1),s(16)),(s(64),s(8),s(512)))))
end

# The role, the plan, the major order and the byte step between the plan's
# instructions travel with the descriptor, preventing A/B swaps or reuse
# with a different instruction shape in the typed issue method.
struct WGMMAOperand{P,Role,Major,Step}
    descriptor::UInt64
end


"""
    mma_async(atom, a, b, accumulator)
    mma_async(atom, a, b)

Warpgroup collective (128 threads, converged). Issue the atom's WGMMA
instructions on the shared operands `a` and `b` (from [`operand`](@ref))
and commit them as one group, after a `wgmma.fence`; the result is pending
until [`wait_mma`](@ref). The accumulator is ready (from `zero_accumulator`
or a completed wait) or still pending from the previous issue: a chain of
issues on one accumulator needs no wait in between, which keeps the tensor
cores busy while a pipeline waits for its next stage. Without an
accumulator the product starts from zero, without zeroing registers first
(the first instructions ignore their accumulator).
Operands must be visible to the async proxy and unchanged until their
group completes.
"""
function mma_async end
"""
    wait_mma(pending)
    wait_mma(pending, n)
    wait_mma(pending; newer=n)

Wait for this warpgroup's WGMMA groups. Without `n`, or with `n = 0`, wait
for all of them and return the ready accumulator. With a constant `n > 0`,
wait until at most the `n` most recent groups are in flight, so the shared
operands of all older groups may be reused, and return `pending`, still
pending.

With `newer = n`, `pending`'s own group is older than the `n` most recent,
which stay in flight: wait until only they are, and return `pending` ready.
Two products overlap this way, as attention's S and O:

```julia
s = mma_async(qk, q, k(j+1), zero_accumulator(qk))    # the next scores
...                                                   # softmax of the current ones
o = wait_mma(o; newer=1)                              # o's group is older than s's
o = mma_async(pv, p, v(j), rescaled(o))
s = wait_mma(s; newer=1)                              # o's newest group runs on
```

The count is of groups committed after `pending`'s, one per `mma_async`.

```julia
wait(full(0), 0)
c = mma_async(atom, a(0), b(0))
for i in 1:stages-1
    wait(full(i), phase(i))
    c = mma_async(atom, a(i), b(i), c)      # chained: no wait
    c = wait_mma(c, 1)                      # stage i-1's group is complete
    arrive!(empty(i-1))                     # its buffers may be refilled
end
f = wait_mma(c)                             # a Fragment
```
"""
function wait_mma end

@inline finish_mma(f::Fragment{Float32}) = f
@generated function finish_mma(c::WGMMAAccumulator{WGMMAAtom{T,N,K,P}}) where {T,N,K,P}
    values = [foldl((a,b) -> :($a + $b), [:(c.data[$(i+j*(N÷2))]) for j in 0:P-1]) for i in 1:N÷2]
    quote
        Base.@inline
        Fragment(($(values...),),operand_layout(WGMMAAtom{T,N,K,P}(),Accumulator()))
    end
end

"Check canonical operand geometry on the host; pointer alignment is checked at binding."
function validate_wgmma(p::WGMMAAtom{T,N,K},role::Role,l::Layouts.AbstractLayout,origin=(0,0)) where {T,N,K,Role<:Union{OperandA,OperandB}}
    A = Role === OperandA ? 2 : 1
    s = swizzled_structure(typeof(l),T,A)
    s === nothing && throw(ArgumentError("operand storage is not a canonical swizzled encoding"))
    s.major === :K || sizeof(T) == 2 || throw(ArgumentError("only 16-bit WGMMA operands may be MN-major"))
    row = s.row_elements
    k,r = origin[A],origin[3-A]
    outer = Role === OperandA ? 64 : N
    if s.major === :K
        K <= row || throw(ArgumentError("the plan's K does not fit one swizzle row"))
        k >= 0 && k % _instruction_k(T) == 0 && k % row + K <= row && k+K <= size(l)[A] && r >= 0 && r % 8 == 0 && r+outer <= size(l)[3-A] ||
            throw(ArgumentError("WGMMA origin or extent is not descriptor-compatible"))
    else
        # MN-major: K runs over rows of the swizzle pattern, the outer extent
        # over groups of one row's elements.
        k >= 0 && k % 8 == 0 && k+K <= size(l)[A] && r >= 0 && r % row == 0 && r+outer <= size(l)[3-A] &&
            r+outer <= row*s.groups || throw(ArgumentError("WGMMA origin or extent is not descriptor-compatible"))
    end
    nothing
end

