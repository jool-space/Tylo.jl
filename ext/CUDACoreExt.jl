module CUDACoreExt
using Tylo, PTX, CUDACore, Adapt
using Tylo: @rtuple, Fragment, PackedFragment, BFloat16, TMATile
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
# stmatrix exists from sm_90; the compile target decides, folded at compile time.
@device_override @inline Tylo._stmatrix_available() = CUDACore.compute_capability().major >= 9
@device_override @inline function Tylo._shuffle_xor(op::F,x::T,::Val{Offset}) where {F,T,Offset}
    partner = ptx"shfl.sync.bfly.b32"(reinterpret(UInt32,x),UInt32(Offset),UInt32(0x1f),0xffffffff % UInt32)
    op(x,reinterpret(T,partner))
end

# Descriptor element types name the bit width; FP8, INT8 and other unlisted
# types move as unsigned words of their size.
_tma_dtype(::Type{T}) where T = T === Tylo.BFloat16 ? :bf16 : T === Float16 ? :f16 : T === Float32 ? :f32 :
    sizeof(T) == 1 ? :u8 : sizeof(T) == 2 ? :u16 : :u32
function Tylo.prepare_tma(p::TMATile{T,S,A,W},array::CuArray{T,N};bounds=size(array)) where {T,S,A,W,N}
    N in (2,3) || throw(ArgumentError("TMA binds a matrix or a batch of matrices"))
    length(bounds) == N && all(map((b,s) -> 0 < b <= s,bounds,size(array))) || throw(ArgumentError("bounds exceed the array or are empty"))
    UInt(pointer(array)) % 16 == 0 && (size(array,1)*sizeof(T)) % 16 == 0 ||
        throw(ArgumentError("TMA source pointer and column stride must be 16-byte aligned"))
    strides = N == 2 ? (sizeof(T)*size(array,1),) : (sizeof(T)*size(array,1),sizeof(T)*size(array,1)*size(array,2))
    box = N == 2 ? (S[A],S[3-A]) : (S[A],S[3-A],1)
    tmap = GC.@preserve array PTX.tensor_map_encode_tiled(_tma_dtype(T),UInt(pointer(array)),
        map(Int,bounds),strides,box;swizzle=Symbol(:B,W))
    uploaded = PTX.upload_tma_descriptor(tmap)
    Tylo.HostTMA(p,array,uploaded.ptr,uploaded.blob,Val(N))
end
function Adapt.adapt_structure(to::CUDACore.KernelAdaptor,b::Tylo.HostTMA{P,A,D,B,R}) where {P,A,D,B,R}
    # Register both hidden allocations with CUDA's stream/lifetime tracking.
    # Their device views need not become kernel parameters, but adaptation's
    # host-side bookkeeping is still required for cross-stream launches.
    Adapt.adapt(to,b.source)
    Adapt.adapt(to,b.storage)
    Tylo.TMABinding(b.plan,b.descriptor,Val(R))
end
end
