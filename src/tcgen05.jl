"""
    Tcgen05MMA{(m,n,k),TA,TB,TC}()
    Tcgen05MMA((m,n,k), TA, TC=Float32)

A single-thread `tcgen05.mma.cta_group::1` instruction described by its
shape and element types: `.kind::f16` for BF16/FP16, `.kind::tf32` for
Float32, `.kind::f8f6f4` for FP8 and `.kind::i8` for INT8 inputs. `k` is
the instruction's fixed K (256 bits of each operand row), `m` is 128 and
`n` a multiple of 16 up to 256.

Its operands are not thread-owned. A and B are shared-memory encodings, a
canonical swizzled core-matrix layout (`Swizzle{B,M,3}` composed with a
layout of packed 32-, 64- or 128-byte rows, which is Tylo's TMA storage;
see [`swizzled_structure`](@ref)), or a `TmemTile` for A; the
accumulator is a `TmemTile` whose layout is [`operand_layout`](@ref) with
`Accumulator`. Descriptors and the instruction descriptor derive
from those encodings ([`tcgen05_operand`](@ref)); `mma` steps the operands
along K through their layouts, so K-major and MN-major storage share one
path. Completion is asynchronous: [`commit_mma`](@ref) arrives on an
mbarrier once every prior instruction of this thread has finished.
"""
struct Tcgen05MMA{S,TA,TB,TC}
    function Tcgen05MMA{S,TA,TB,TC}() where {S,TA,TB,TC}
        _check_tcgen05(S,TA,TB,TC)
        new{S,TA,TB,TC}()
    end
end
Base.@constprop :aggressive @inline Tcgen05MMA(shape::NTuple{3,Int},::Type{TA},::Type{TC}=_default_accumulator(TA)) where {TA,TC} =
    Tcgen05MMA{shape,TA,TA,TC}()
Base.@constprop :aggressive @inline Tcgen05MMA(shape::NTuple{3,Int},::Type{TA},::Type{TB},::Type{TC}) where {TA,TB,TC} =
    Tcgen05MMA{shape,TA,TB,TC}()
Base.size(::Tcgen05MMA{S}) where S = S
threads(::Tcgen05MMA) = 1
Base.eltype(::Tcgen05MMA{S,TA,TB,TC},::OperandA) where {S,TA,TB,TC} = TA
Base.eltype(::Tcgen05MMA{S,TA,TB,TC},::OperandB) where {S,TA,TB,TC} = TB
Base.eltype(::Tcgen05MMA{S,TA,TB,TC},::Accumulator) where {S,TA,TB,TC} = TC
_tcgen05_kind(::Type{T}) where T = (f = _input_family(T); f === :fp16 ? :f16 : f === :tf32 ? :tf32 : f === :fp8 ? :f8f6f4 : f === :int8 ? :i8 : :none)
@inline function _check_tcgen05(S,TA,TB,TC)
    S isa NTuple{3,Int} || throw(ArgumentError("MMA shape must be an (m,n,k) tuple"))
    family = _input_family(TA)
    family !== :none && family === _input_family(TB) ||
        throw(ArgumentError("A and B must share an input family"))
    family === :fp16 && TA !== TB && throw(ArgumentError("16-bit A and B must share a type"))
    m,n,k = S
    k == 256 ÷ _element_bits(TA) || throw(ArgumentError("tcgen05 K is 256 bits of each operand row"))
    m == 128 || throw(ArgumentError("tcgen05 atoms currently describe M=128; the M=64 accumulator layout is not described"))
    n % 16 == 0 && 16 <= n <= 256 || throw(ArgumentError("tcgen05 N is a multiple of 16 up to 256"))
    TC === (family === :int8 ? Int32 : Float32) ||
        throw(ArgumentError("unsupported accumulator type for these inputs"))
    nothing
end

# The accumulator: TMEM lane m holds row m, column n holds column n.
_accumulator_layout(m,n) = Layouts.Layout((Layouts.static(m),Layouts.static(n)),(Layouts.static(1),Layouts.static(128)))
@generated operand_layout(::A,::Accumulator) where {A<:Tcgen05MMA} = :($(_accumulator_layout(size(A())[1],size(A())[2])))
"""
    accumulator(atom, address::UInt32)

The atom's accumulator as a `TmemTile` at a TMEM address: lane `m` holds
row `m`, column `n` holds column `n`. The caller owns the allocation; the
first `mma` with `accumulate=false` initializes it.
"""
@inline accumulator(a::Tcgen05MMA,address::UInt32) = TmemTile(eltype(a,Accumulator()),address,operand_layout(a,Accumulator()))

_tcgen05_dtype(::Type{BFloat16}) = :bf16
_tcgen05_dtype(::Type{Float16}) = :f16
_tcgen05_dtype(::Type{Float32}) = :tf32
_tcgen05_dtype(::Type{Float8E4M3}) = :e4m3
_tcgen05_dtype(::Type{Float8E5M2}) = :e5m2
_tcgen05_dtype(::Type{Int8}) = :s8
_tcgen05_dtype(::Type{UInt8}) = :u8
"""
    instruction_descriptor(atom; a_major=:K, b_major=:K) -> UInt32

The `tcgen05.mma` instruction descriptor of an atom for the given operand
majorness: a pure function of the atom's shape and element types, as
`mma` derives it from its operands' encodings.
"""
function instruction_descriptor(a::Tcgen05MMA{S,TA,TB,TC};a_major::Symbol=:K,b_major::Symbol=:K) where {S,TA,TB,TC}
    m,n,k = S
    kind = _tcgen05_kind(TA)
    if kind === :f16 || kind === :tf32
        PTX.tcgen05_instr_desc_f16bf16_f32(;m,n,ab_dtype=_tcgen05_dtype(TA),a_major,b_major)
    elseif kind === :i8
        PTX.tcgen05_instr_desc_i8(;m,n,a_dtype=_tcgen05_dtype(TA),b_dtype=_tcgen05_dtype(TB),a_major,b_major)
    else
        PTX.tcgen05_instr_desc_f8f6f4(;m,n,a_dtype=_tcgen05_dtype(TA),b_dtype=_tcgen05_dtype(TB),d_dtype=:f32,a_major,b_major)
    end
end
# A TMEM tile in the accumulator convention: lanes along axis 1, one typed
# position per column along axis 2, possibly a window of a larger tile.
_tmem_canonical(::Type,rows) = false
_tmem_canonical(::Type{Layouts.Layout{S,D}},rows) where {S,D} =
    S <: Tuple{Layouts.StaticInt{rows},Layouts.StaticInt} && D === Tuple{Layouts.StaticInt{1},Layouts.StaticInt{128}}
_tmem_canonical(::Type{Layouts.Window{S,L,O}},rows) where {S,L,O} =
    S[1] == rows && L <: Layouts.Layout && L.parameters[2] === Tuple{Layouts.StaticInt{1},Layouts.StaticInt{128}}
_k_axis(::OperandA) = 2
_k_axis(::OperandB) = 1
_k_axis(::Type{OperandA}) = 2
_k_axis(::Type{OperandB}) = 1

"""
    Tcgen05Operand

A shared-memory A or B operand of a [`Tcgen05MMA`](@ref): the descriptor of
the tile's base, the tile's layout and the logical origin. `mma` derives each
K step's descriptor from the layout, so the same operand serves K-major and
MN-major storage. Construct with [`tcgen05_operand`](@ref).
"""
struct Tcgen05Operand{P,Role,Major,T,L}
    base::UInt64              # descriptor of the tile's base
    at::UInt64                # descriptor at the origin
    layout::L                 # the tile's layout without its swizzle
    origin::Tuple{Int32,Int32}
end
Base.size(o::Tcgen05Operand) = size(o.layout)
"""
    tcgen05_operand(atom, role, tile::SharedTile, origin=(Int32(0),Int32(0)))

Borrow an A (M,K) or B (K,N) operand from a canonical swizzled shared
tile. The tile's static layout is recognized structurally (see
[`swizzled_structure`](@ref)); the origin's coordinates are multiples of
eight, and the non-K extent from the origin covers the atom's M or N. The
root storage is aligned to eight swizzle rows (1024 bytes for 128-byte
rows). Bounds checks may be elided with `@inbounds`.
"""
function tcgen05_operand end
"""
    commit_mma(barrier)

Arrive on a shared mbarrier once all prior `tcgen05.mma` instructions issued
by this thread have completed. Single thread; no wait is implied.
"""
function commit_mma end
"""
    tmem_allocate!(slot, Val(columns))
    tmem_deallocate!(address, Val(columns))
    tmem_relinquish_permit()

Warp-collective TMEM management. `tmem_allocate!` writes the base address of
`columns` (a power of two from 32 to 512) to a shared slot; the CTA reads it
after a barrier. `tmem_deallocate!` returns the columns and
`tmem_relinquish_permit` lets other CTAs allocate. One warp executes each.
"""
function tmem_allocate! end
@doc (@doc tmem_allocate!) tmem_deallocate!
function tmem_deallocate! end
@doc (@doc tmem_allocate!) tmem_relinquish_permit
function tmem_relinquish_permit end
