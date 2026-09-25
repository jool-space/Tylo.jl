module CUDACoreExt
using Tylo, PTX, CUDACore, Adapt
using Tylo: @rtuple, Fragment, BFloat16, TMATile
using CUDACore: @device_override
using PTX: @ptx_str

# Device-only implementations of host-callable generics. The host keeps the
# generic method (or raises); kernels compiled through CUDACore's method
# table use these.
# FP32 to BF16/FP16 converts pairs with one `cvt` each.
@device_override @inline function Tylo._convert_fragment(::Type{T}, f::Fragment{Float32,N}) where {T<:Union{BFloat16,Float16},N}
    iseven(N) || return Tylo.map(Base.Fix1(Tylo.convert_element,T),f)
    words = @rtuple(i -> Tylo._pack_mma_pair(T,f.data[2i-1],f.data[2i]), 1:N÷2)
    Tylo._from_words(T,words,Tylo.Layouts.layout(f))
end
# FP32 to FP8 converts pairs with one `cvt` each where the target has it
# (sm_89 on); a pair's 16 bits are half of a fragment word, so the words
# form without splitting pairs into bytes.
@inline _fp8_pair(::Type{Tylo.Float8_E4M3FN},lo,hi) = reinterpret(UInt16,ptx"cvt.rn.satfinite.e4m3x2.f32"(hi,lo))
@inline _fp8_pair(::Type{Tylo.Float8_E5M2},lo,hi) = reinterpret(UInt16,ptx"cvt.rn.satfinite.e5m2x2.f32"(hi,lo))
@inline _has_fp8() = (c = CUDACore.compute_capability(); c.major > 8 || (c.major == 8 && c.minor >= 9))
@device_override @inline function Tylo._convert_fragment(::Type{T}, f::Fragment{Float32,N}) where {T<:Union{Tylo.Float8_E4M3FN,Tylo.Float8_E5M2},N}
    (N % 4 == 0 && _has_fp8()) ||
        return Fragment(Tuple(Tylo.Microfloats.cvt(Tylo.Microfloats.SVector{N,T},f.data,RoundNearest,Tylo.SAT)),Tylo.Layouts.layout(f))
    pairs = @rtuple(i -> _fp8_pair(T,f.data[2i-1],f.data[2i]), 1:N÷2)
    words = @rtuple(i -> UInt32(pairs[2i-1]) | (UInt32(pairs[2i]) << 16), 1:N÷4)
    Tylo._from_words(T,words,Tylo.Layouts.layout(f))
end
@device_override @inline Tylo._smem_address(p::Core.LLVMPtr{T,3}) where T = PTX.smem_addr_u32(p)
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
    box = N == 2 ? Tylo._box(p) : (Tylo._box(p)...,1)     # one stripe per copy
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
