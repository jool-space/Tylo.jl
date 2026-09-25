"""
    WGMMAAtom(T, n, k=64, partials=1)

Hopper shared/shared warpgroup MMA: 128 threads compute a 64×n×k product.
BF16/FP16 inputs, FP32 accumulation; n=8:8:256, k in (16,32,64). Each K=16
instruction updates one of `partials` independent accumulators cyclically.
Partials must divide k/16; total registers per thread must not exceed 128.
`partials=4` at n=8,k=64 preserves Cohere's four independent K chains.
Use `finish_mma` to sum partials after waiting. No allocation or CTA barrier.
The sizes must be constants. The finished accumulator is a `Fragment` in
`operand_layout(atom, Accumulator())`; A and B are shared-memory descriptors
from [`operand`](@ref), not registers.
"""
struct WGMMAAtom{T,N,K,P} <: Atom
    function WGMMAAtom{T,N,K,P}() where {T,N,K,P}
        T === BFloat16 || T === Float16 || throw(ArgumentError("BF16 or FP16 required"))
        N isa Int && 8 <= N <= 256 && N % 8 == 0 && K isa Int && K in (16,32,64) &&
            P isa Int && P > 0 && (K÷16) % P == 0 && N÷2*P <= 128 ||
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
@inline zero_accumulator(p::WGMMAAtom) =
    WGMMAAccumulator(p,ntuple(_ -> 0f0,Val(_wg_registers(p))))

# PTX ISA wgmma accumulator figure: lane groups t%4 select column pairs,
# (t%32)÷4 the row, warps 16-row bands; slots alternate column, row half,
# 8-column block.
function _operand_layout(::WGMMAAtom{T,N},::Accumulator) where {T,N}
    L, s = Layouts.Layout, Layouts.static
    Layouts.Ownership((64,N),L(((s(4),s(8),s(4)),(s(2),s(2),s(N÷8))),
                               ((s(128),s(1),s(16)),(s(64),s(8),s(512)))))
end

# The role and complete plan travel with the descriptor, preventing A/B swaps
# or reuse with a different instruction shape in the typed issue method.
struct WGMMAOperand{P,Role}
    descriptor::UInt64
end


"""
    mma_async(atom, a, b, accumulator)

Warpgroup collective (128 threads, converged). Issue the atom's WGMMA
instructions on the shared operands `a` and `b` (from [`operand`](@ref))
and commit them as one group, after a `wgmma.fence`; the result is pending
until [`wait_mma`](@ref). The accumulator is ready (from `zero_accumulator`
or a completed wait) or still pending from the previous issue: a chain of
issues on one accumulator needs no wait in between, which keeps the tensor
cores busy while a pipeline waits for its next stage.
Operands must be visible to the async proxy and unchanged until their
group completes.
"""
function mma_async end
"""
    wait_mma(pending)
    wait_mma(pending, n)

Wait for this warpgroup's WGMMA groups. Without `n`, or with `n = 0`, wait
for all of them and return the ready accumulator. With a constant `n > 0`,
wait until at most the `n` most recent groups are in flight, so the shared
operands of all older groups may be reused, and return `pending`, still
pending.

```julia
wait(full(0), 0)
c = mma_async(atom, a(0), b(0), zero_accumulator(atom))
for i in 1:stages-1
    wait(full(i), phase(i))
    c = mma_async(atom, a(i), b(i), c)      # chained: no wait
    c = wait_mma(c, 1)                      # stage i-1's group is complete
    arrive!(empty(i-1))                     # its buffers may be refilled
end
f = finish_mma(wait_mma(c))
```
"""
function wait_mma end

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
    s.major === :K || throw(ArgumentError("shared/shared WGMMA operands are K-major"))
    row = s.row_elements
    K <= row || throw(ArgumentError("the plan's K does not fit one swizzle row"))
    k,r = origin[A],origin[3-A]
    outer = Role === OperandA ? 64 : N
    k >= 0 && k % 16 == 0 && k % row + K <= row && k+K <= size(l)[A] && r >= 0 && r % 8 == 0 && r+outer <= size(l)[3-A] ||
        throw(ArgumentError("WGMMA origin or extent is not descriptor-compatible"))
    nothing
end

