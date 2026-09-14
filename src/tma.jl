"""
    TMALoad(T, Val((m,k)), Val(2))   # logical A(M,K)
    TMALoad(T, Val((k,n)), Val(1))   # logical B(K,N)

A 2D global-to-shared TMA load with BF16/FP16 elements, K=64 and 128-byte
swizzling. The non-K extent is a positive multiple of eight, at most 256.
The shared allocation must be 1024-byte aligned; `transfer_bytes` includes
all zero-filled out-of-bounds elements. This plan owns no memory or barriers.
"""
struct TMALoad{T,S,A}
    @inline function TMALoad(::Type{T},::Val{S},::Val{A}) where {T,S,A}
        T === BFloat16 || T === Float16 || throw(ArgumentError("BF16 or FP16 required"))
        S isa Tuple && length(S) == 2 && all(x -> x isa Int && x > 0,S) &&
            A isa Int && A in (1,2) && S[A] == 64 && S[3-A] % 8 == 0 && S[3-A] <= 256 ||
            throw(ArgumentError("TMA requires K=64 and a multiple-of-eight outer extent <=256"))
        new{T,S,A}()
    end
end
Base.size(::TMALoad{T,S}) where {T,S} = S
transfer_bytes(p::TMALoad) = 2prod(size(p))

# A descriptor-compatible subset of the general layout algebra. Keeping the
# canonical form explicit prevents arbitrary layouts from becoming descriptors
# that silently describe different storage. Offsets are ELEMENTS, as elsewhere.
struct TMASharedLayout{S,A} <: Layouts.AbstractLayout end
Layouts.shape(::TMASharedLayout{S}) where S = map(Layouts.static,S)
Layouts.cosize(l::TMASharedLayout) = prod(size(l))
@inline function (::TMASharedLayout{S,A})(c::Tuple) where {S,A}
    k,r = c[A],c[3-A]
    offset = r*oftype(r,64)+k
    xor(offset,(offset >> 3) & oftype(offset,56))
end
shared_layout(::TMALoad{T,S,A}) where {T,S,A} = TMASharedLayout{S,A}()
@inline shared_tile(p::TMALoad{T},ptr::Core.LLVMPtr{U,3}) where {T,U} =
    SharedTile(reinterpret(Core.LLVMPtr{T,3},ptr),shared_layout(p))

# Host bindings retain BOTH the source tensor and the descriptor allocation.
# Adaptation creates the isbits device carrier without address-space casts in
# device IR. No lifetime claim is attached to the borrowed device value.
struct PreparedTMA{P,A,D,B}
    plan::P
    source::A
    descriptor::D
    storage::B
end
struct DeviceTMA{P,D}
    plan::P
    descriptor::D
end
shared_layout(b::Union{PreparedTMA,DeviceTMA}) = shared_layout(b.plan)
transfer_bytes(b::Union{PreparedTMA,DeviceTMA}) = transfer_bytes(b.plan)
@inline shared_tile(b::Union{PreparedTMA,DeviceTMA},ptr) = shared_tile(b.plan,ptr)

"""
    prepare_tma(plan, array)

Encode and upload a descriptor on the host. `array` is a dense device matrix
stored as (K, non-K), irrespective of the plan's logical axis order. It must
have a 16-byte-aligned pointer and column stride. The returned binding retains
array and descriptor storage; keep it alive through all launches/graph replays.
Requires CUDACore and PTX. Source data may change; address/shape may not.
"""
function prepare_tma end

"""
    tma_load!(dst, binding, origin, barrier)

Issue one copy from a zero-based LOGICAL global origin (a pair of Int32 values) to the canonical shared
tile. One elected thread executes. The caller initializes the 8-byte-aligned
shared mbarrier, publishes its initialization to the async proxy, and supplies
an expected transaction count including `transfer_bytes(binding)`. Wait for
that barrier phase before reading; release consumers before reusing storage.
No implicit arrival, wait, thread rendezvous, or proxy fence is inserted.
"""
function tma_load! end
