"""
    TMATile(T, (m,k), 2)        # logical A(M,K), 128-byte rows
    TMATile(T, (k,n), 1, 64)    # logical B(K,N), 64-byte rows
    TMATile(T, (m,128), 2)      # BF16: two 128-byte stripes along K

A 2D tile that TMA moves between global and shared memory. Elements are
one, two or four bytes wide. Axis `A` is the inner axis: a whole number of
swizzle rows of `W` bytes (32, 64 or 128, default 128), up to eight, each a *stripe*
stored after the previous one; the other extent is a positive multiple of
eight, at most 256. Shared storage is [`shared_layout`](@ref)`(plan)`,
allocated at a multiple of `8W` bytes, and is the canonical swizzled
encoding `ldmatrix`, WGMMA and tcgen05 read. One [`tma_load!`](@ref) or
[`tma_store!`](@ref) moves the whole tile, one copy per stripe.
`transfer_bytes` counts every element, including zero-filled out-of-bounds
elements of a load. The plan owns no memory, descriptor or barrier;
[`prepare_tma`](@ref) binds it to a device array. The shape, axis and width
must be constants.
"""
struct TMATile{T,S,A,W}
    @inline function TMATile{T,S,A,W}() where {T,S,A,W}
        isbitstype(T) && sizeof(T) in (1,2,4) || throw(ArgumentError("TMA elements are one, two or four bytes wide"))
        W isa Int && W in (32,64,128) || throw(ArgumentError("swizzle rows are 32, 64 or 128 bytes"))
        S isa Tuple && length(S) == 2 && all(x -> x isa Int && x > 0,S) && A isa Int && A in (1,2) &&
            (S[A]*sizeof(T)) % W == 0 && S[A]*sizeof(T) <= 8W && S[3-A] % 8 == 0 && S[3-A] <= 256 ||
            throw(ArgumentError("the inner extent is a whole number of swizzle rows and the outer extent a multiple of eight, at most 256"))
        new{T,S,A,W}()
    end
end
Base.@constprop :aggressive @inline TMATile(::Type{T},shape::Tuple{Integer,Integer},axis::Integer,width::Integer=128) where T =
    TMATile{T,map(Int,shape),Int(axis),Int(width)}()
Base.size(::TMATile{T,S}) where {T,S} = S
"""
    transfer_bytes(plan_or_binding)

The bytes one [`tma_load!`](@ref) of the tile delivers, every element
counted (out-of-bounds ones arrive as zeros): what its barrier's phase
must expect.
"""
transfer_bytes(::TMATile{T,S}) where {T,S} = sizeof(T)*prod(S)
swizzle_bytes(::TMATile{T,S,A,W}) where {T,S,A,W} = W
# Swizzle rows side by side along the inner axis, each one TMA box.
_stripes(::TMATile{T,S,A,W}) where {T,S,A,W} = S[A]*sizeof(T) ÷ W
_box(::TMATile{T,S,A,W}) where {T,S,A,W} = (W ÷ sizeof(T),S[3-A])
# The hardware swizzle family in elements: 16-byte chunks (the low M bits
# stay), 2^B chunks per row, the chunk index XORed with row bits 1..3 of a
# 128-byte address period (shift 3). B=3,2,1 are the 128/64/32-byte modes.
@inline _swizzle(::Type{T},W) where T = Layouts.Swizzle{trailing_zeros(W ÷ 16),trailing_zeros(16 ÷ sizeof(T)),3}()
"""
    shared_layout(plan)

The canonical storage a TMA tile occupies: rows of the swizzle width along
the inner axis, packed, with the matching hardware swizzle, as
`Swizzle{B,M,3}` composed with the row layout. Offsets are elements. The
same encoding is what `ldmatrix` loads, `operand` describes (see [`swizzled_structure`](@ref)).
"""
@inline function shared_layout(p::TMATile{T,S,A,W}) where {T,S,A,W}
    st = Layouts.static
    row, outer, stripes = W ÷ sizeof(T), S[3-A], _stripes(p)
    # One stripe: rows of the swizzle width. Several: stripe after stripe.
    inner = stripes == 1 ? (st(row),st(1)) : ((st(row),st(stripes)),(st(1),st(row*outer)))
    l = A == 2 ? Layouts.Layout((st(outer),inner[1]),(st(row),inner[2])) :
                 Layouts.Layout((inner[1],st(outer)),(inner[2],st(row)))
    Layouts.compose(_swizzle(T,W),l)
end
@inline shared_tile(p::TMATile{T},ptr::Core.LLVMPtr{U,3}) where {T,U} =
    SharedTile(reinterpret(Core.LLVMPtr{T,3},ptr),shared_layout(p))

# Host bindings retain BOTH the source tensor and the descriptor allocation.
# Adaptation creates the isbits device carrier without address-space casts in
# device IR. No lifetime claim is attached to the borrowed device value. `R`
# is the array rank the descriptor was encoded for: 2 for one matrix, 3 for
# a batch of matrices along the third axis (one tile of one batch per copy).
struct HostTMA{P,A,D,B,R}
    plan::P
    source::A
    descriptor::D
    storage::B
    HostTMA(plan::P,source::A,descriptor::D,storage::B,::Val{R}) where {P,A,D,B,R} = new{P,A,D,B,R}(plan,source,descriptor,storage)
end
struct TMABinding{P,D,R}
    plan::P
    descriptor::D
    TMABinding(plan::P,descriptor::D,::Val{R}) where {P,D,R} = new{P,D,R}(plan,descriptor)
end
"`TMABinding(plan, descriptor)`: a device binding of one matrix to a descriptor already on the device."
TMABinding(plan,descriptor) = TMABinding(plan,descriptor,Val(2))
shared_layout(b::Union{HostTMA,TMABinding}) = shared_layout(b.plan)
transfer_bytes(b::Union{HostTMA,TMABinding}) = transfer_bytes(b.plan)
swizzle_bytes(b::Union{HostTMA,TMABinding}) = swizzle_bytes(b.plan)
@inline shared_tile(b::Union{HostTMA,TMABinding},ptr) = shared_tile(b.plan,ptr)

"""
    prepare_tma(plan, array; bounds=size(array))

Encode and upload a descriptor on the host. `array` is a dense device matrix
stored as (inner, outer), irrespective of the plan's logical axis order, or a
three-dimensional array of such matrices along its third axis, in which case
origins take a third coordinate selecting the matrix and every matrix is
bounded separately. It must have a 16-byte-aligned pointer and column
stride. `bounds` names the logical extents when the array is padded: copies
zero-fill loads and skip stores beyond them. The returned binding retains
array and descriptor storage; keep it
alive through all launches and graph replays. One binding serves loads and
stores. Requires CUDACore and PTX. Source data may change; address and shape
may not.
"""
function prepare_tma end

"""
    tma_load!(dst, binding, origin, barrier::Barrier)

Issue one TMA copy of the binding's box at a zero-based logical `origin` (a
pair of integers, or a triple whose third coordinate selects the matrix of
a three-dimensional binding) into the canonical shared tile `dst`. One
thread issues it. The copy completes on `barrier`, whose current phase
must expect its [`transfer_bytes`](@ref) (`arrive!(barrier, bytes)`);
[`wait`](@ref Base.wait(::Barrier,::Integer)) for that phase before reading
`dst`. Elements outside the tensor's bounds arrive as zeros. The inner
origin starts a 16-byte chunk; the hardware rejects other origins.
Nothing else is inserted: initialize and publish the barrier first, and
release the consumers of `dst` before reusing it.
"""
function tma_load! end

"""
    tma_store!(binding, src, origin)

Issue one copy from the canonical shared tile to a zero-based LOGICAL
global origin, from one elected thread. Origins are non-negative and the
inner origin starts a 16-byte chunk; elements past the array's far edges
are not written. The copy belongs to the issuing thread's current bulk
group: [`commit_tma_stores`](@ref) closes the group, and
[`wait_tma_reads`](@ref) or [`wait_tma_stores`](@ref) observes it. Shared
data written through ordinary stores becomes visible to the copy only after
[`fence_async_shared`](@ref) and a rendezvous of the writing threads with
the issuing thread; Tylo inserts neither.
"""
function tma_store! end

"""
    commit_tma_stores()

Close the issuing thread's bulk group of TMA stores.
"""
function commit_tma_stores end

"""
    wait_tma_reads(n)

Wait until at most `n` (a constant) of the issuing thread's committed bulk
groups are still reading shared memory; the source tiles of the others may be reused.
"""
function wait_tma_reads end

"""
    wait_tma_stores(n)

Wait until at most `n` (a constant) of the issuing thread's committed bulk
groups are still in flight; the global writes of the others are complete and visible
to the issuing thread.
"""
function wait_tma_stores end

# Structural analysis of a static shared-memory layout type as a canonical
# swizzled encoding. Each logical axis flattens to (extent, stride) pairs in
# elements, innermost first. The contiguous axis starts with a (row,1) pair
# (one swizzle row) and the other axis with an (8k, row) pair (eight
# contiguous rows form a core-matrix group); every further stride is a
# multiple of eight rows so the swizzle phase repeats.
_axis_pairs(s::Layouts.StaticInt,d::Layouts.StaticInt) = [(Int(s),Int(d))]
_axis_pairs(s::Tuple,d::Tuple) = reduce(vcat,map(_axis_pairs,s,d);init=Tuple{Int,Int}[])
_axis_pairs(s,d) = nothing
_swizzled_pairs(::Type,::Type) = nothing
function _swizzled_pairs(::Type{Layouts.Composition{Layouts.Swizzle{B,M,3},L}},::Type{T}) where {B,M,L,T}
    1 <= B <= 3 && M == trailing_zeros(16 ÷ sizeof(T)) || return nothing
    l = _static_instance(L)
    l isa Layouts.Layout && length(Layouts.shape(l)) == 2 || return nothing
    pairs = map(_axis_pairs,Layouts.shape(l),strides(l))
    any(isnothing,pairs) ? nothing : (16 << B,pairs)
end
_swizzled_pairs(::Type{Layouts.Window{S,L,O}},::Type{T}) where {S,L,O,T} = _swizzled_pairs(L,T)
"""
    swizzled_structure(layout_type, T, kaxis) -> (; major, swizzle_bytes, leading_bytes, stride_bytes, row_elements, groups) or nothing

Recognize a static shared layout type as a canonical swizzled encoding of
`T` elements (`Swizzle{B,M,3}` over packed rows of `16 << B` bytes, `M`
the chunk bits of 16 bytes) and read the descriptor fields for an operand
whose logical K is axis `kaxis`: `major` is `:K` when K is the contiguous
axis and `:MN` otherwise, `swizzle_bytes` the row width, `leading_bytes`
the distance between row-width groups along the contiguous axis,
`stride_bytes` the distance between eight-row core-matrix groups,
`row_elements` one row and `groups` the number of rows placed side by
side along the contiguous axis.
"""
function swizzled_structure(::Type{L},::Type{T},kaxis::Int) where {L,T}
    found = _swizzled_pairs(L,T)
    found === nothing && return nothing
    bytes, pairs = found
    row = bytes ÷ sizeof(T)          # elements per swizzle row
    cycle = 8row                     # elements per eight-row swizzle cycle
    contiguous = findall(p -> !isempty(p) && p[1] == (row,1),pairs)
    length(contiguous) == 1 || return nothing
    c = contiguous[1]
    inner, outer = pairs[c], pairs[3-c]
    !isempty(outer) && outer[1][2] == row && outer[1][1] % 8 == 0 || return nothing
    all(p -> p[2] % cycle == 0,inner[2:end]) && all(p -> p[2] % cycle == 0,outer[2:end]) || return nothing
    major = c == kaxis ? :K : :MN
    # Hardware ignores the leading offset of K-major swizzled operands; the
    # MN-major offset is the distance between consecutive row-width groups.
    leading = major === :MN && length(inner) > 1 ? inner[2][2]*sizeof(T) : 16
    stride = outer[1][1] > 8 ? 8bytes : length(outer) > 1 ? outer[2][2]*sizeof(T) : 8bytes
    (; major, swizzle_bytes = bytes, leading_bytes = leading, stride_bytes = stride, row_elements = row,
       groups = length(inner) > 1 ? inner[2][1] : 1)
end
# Descriptor layout codes of a row width, for the two instruction families.
_tcgen05_swizzle(bytes) = bytes == 128 ? PTX.BlackwellLayout.B128 : bytes == 64 ? PTX.BlackwellLayout.B64 : PTX.BlackwellLayout.B32
_wgmma_swizzle(bytes) = bytes == 128 ? PTX.WgmmaSwizzle.B128 : bytes == 64 ? PTX.WgmmaSwizzle.B64 : PTX.WgmmaSwizzle.B32
