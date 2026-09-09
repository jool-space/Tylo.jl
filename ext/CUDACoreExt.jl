module CUDACoreExt
using Tylo, PTX, CUDACore, Adapt

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
