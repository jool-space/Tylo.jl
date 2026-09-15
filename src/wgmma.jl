"""
    WGMMA64(T, Val(n), Val(k)=Val(64), Val(partials)=Val(1))

Hopper shared/shared warpgroup MMA: 128 threads compute a 64×n×k product.
BF16/FP16 inputs, FP32 accumulation; n=8:8:256, k in (16,32,64). Each K=16
instruction updates one of `partials` independent accumulators cyclically.
Partials must divide k/16; total registers per thread must not exceed 128.
`partials=4` at n=8,k=64 preserves Cohere's four independent K chains.
Use `finish_mma` to sum partials after waiting. No allocation or CTA barrier.
"""
struct WGMMA64{T,N,K,P}
    function WGMMA64(::Type{T},::Val{N},::Val{K}=Val(64),::Val{P}=Val(1)) where {T,N,K,P}
        T === BFloat16 || T === Float16 || throw(ArgumentError("BF16 or FP16 required"))
        N isa Int && 8 <= N <= 256 && N % 8 == 0 && K isa Int && K in (16,32,64) &&
            P isa Int && P > 0 && (K÷16) % P == 0 && N÷2*P <= 128 ||
            throw(ArgumentError("invalid WGMMA shape or partial accumulator count"))
        new{T,N,K,P}()
    end
end
Base.size(::WGMMA64{T,N,K}) where {T,N,K} = (64,N,K)
threads(::WGMMA64) = 128
_wg_registers(::WGMMA64{T,N,K,P}) where {T,N,K,P} = N÷2*P
struct WGMMAAccumulator{P,R}
    data::NTuple{R,Float32}
    function WGMMAAccumulator(p::P,data::NTuple{R,Float32}) where {P<:WGMMA64,R}
        R == _wg_registers(p) || throw(ArgumentError("wrong WGMMA register count"))
        new{P,R}(data)
    end
end
struct PendingWGMMA{P,R}
    data::NTuple{R,Float32}
end
@inline zero_accumulator(p::WGMMA64) =
    WGMMAAccumulator(p,ntuple(_ -> 0f0,Val(_wg_registers(p))))

struct WGMMAOwnership{N} end
Base.size(::WGMMAOwnership{N}) where N = (64,N)
operand_layout(::WGMMA64{T,N},::Accumulator) where {T,N} = WGMMAOwnership{N}()
@inline function Layouts.coordinate(::WGMMAOwnership{N},t::Integer,::Val{E}) where {N,E}
    E isa Int && 0 <= E < N÷2 || throw(BoundsError())
    lane = t % oftype(t,32)
    row = (t ÷ oftype(t,32))*oftype(t,16) + lane÷oftype(t,4) + oftype(t,8*((E%4)÷2))
    col = (lane % oftype(t,4))*oftype(t,2) + oftype(t,E%2+8*(E÷4))
    (row,col)
end

# The role and complete plan travel with the descriptor, preventing A/B swaps
# or reuse with a different instruction shape in the typed issue method.
struct WGMMAOperand{P,Role}
    descriptor::UInt64
end

"""
    wgmma_operand(plan, role, tile, origin=(Int32(0),Int32(0)))

Borrow an operand from canonical K-major swizzled shared storage (see
[`swizzled_structure`](@ref)). Origins are LOGICAL; A is (M,K), B is (K,N).
Non-K origins must be multiples of eight, K origins multiples of 16; the
plan's K fits one swizzle row and the whole plan fits the tile. Root
storage is aligned to eight swizzle rows. Bounds/alignment checks may be
elided with `@inbounds` once proven.
"""
function wgmma_operand end

"""
    mma_async(plan, a, b, accumulator)

128-thread collective. Fence accumulator registers, issue the plan's WGMMA
instructions and commit ONE group. Returns pending registers; only `wait_mma`
accepts them. Operands must already be visible to the async proxy and remain
unchanged until the wait completes. All four warps execute in convergence.
"""
function mma_async end
"Wait for ALL this warpgroup's committed WGMMA groups; return ready registers."
function wait_mma end

@generated function finish_mma(c::WGMMAAccumulator{WGMMA64{T,N,K,P}}) where {T,N,K,P}
    values = [foldl((a,b) -> :($a + $b), [:(c.data[$(i+j*(N÷2))]) for j in 0:P-1]) for i in 1:N÷2]
    quote
        Base.@inline
        Fragment(($(values...),),WGMMAOwnership{$N}())
    end
end

"Check canonical operand geometry on the host; pointer alignment is checked at binding."
function validate_wgmma(p::WGMMA64{T,N,K},role::Role,l::Layouts.AbstractLayout,origin=(0,0)) where {T,N,K,Role<:Union{OperandA,OperandB}}
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

_register_count(::WGMMAOwnership{N}) where N = N÷2
