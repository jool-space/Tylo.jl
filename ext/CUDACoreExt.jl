module CUDACoreExt
using Tylo, PTX, CUDACore, Adapt
using Tylo: @rtuple, Fragment, PackedFragment, BFloat16
using CUDACore: @device_override
using PTX: @ptx_str

# Device-only implementations of host-callable generics. The host keeps the
# generic method (or raises); kernels compiled through CUDACore's method
# table use these.
@device_override @inline function Tylo.pack(::Type{T}, f::Fragment{Float32,N}) where {T<:Union{BFloat16,Float16},N}
    iseven(N) || throw(ArgumentError("packing requires complete pairs"))
    words = @rtuple(i -> Tylo._pack_mma_pair(T,f.data[2i-1],f.data[2i]), 1:N÷2)
    PackedFragment(T,words,Tylo.Layouts.layout(f))
end
@device_override @inline Tylo._warp_reduce(op::F,x,::Val{W}) where {F,W} = PTX.Warps.warp_reduce(op,x,Val(W))
@device_override @inline function Tylo._shuffle_xor(op::F,x::T,::Val{Offset}) where {F,T,Offset}
    partner = ptx"shfl.sync.bfly.b32"(reinterpret(UInt32,x),UInt32(Offset),UInt32(0x1f),0xffffffff % UInt32)
    op(x,reinterpret(T,partner))
end

function Tylo.prepare_tma(p::TMALoad{T,S,A},array::CuArray{T,2}) where {T,S,A}
    size(array,1) > 0 && size(array,2) > 0 || throw(ArgumentError("empty TMA source"))
    UInt(pointer(array)) % 16 == 0 && (size(array,1)*sizeof(T)) % 16 == 0 ||
        throw(ArgumentError("TMA source pointer and column stride must be 16-byte aligned"))
    dtype = T === Tylo.BFloat16 ? :bf16 : :f16
    map = GC.@preserve array PTX.tensor_map_encode_tiled(dtype,UInt(pointer(array)),
        (size(array,1),size(array,2)),(sizeof(T)*size(array,1),),
        (64,S[3-A]);swizzle=:B128)
    uploaded = PTX.upload_tma_descriptor(map)
    Tylo.PreparedTMA(p,array,uploaded.ptr,uploaded.blob)
end
function Adapt.adapt_structure(to::CUDACore.KernelAdaptor,b::Tylo.PreparedTMA)
    # Register both hidden allocations with CUDA's stream/lifetime tracking.
    # Their device views need not become kernel parameters, but adaptation's
    # host-side bookkeeping is still required for cross-stream launches.
    Adapt.adapt(to,b.source)
    Adapt.adapt(to,b.storage)
    Tylo.DeviceTMA(b.plan,b.descriptor)
end
end
