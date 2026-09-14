Base.@propagate_inbounds function tma_load!(dst::SharedTile{T,TMASharedLayout{S,A}},
        b::DeviceTMA{TMALoad{T,S,A}},origin::Tuple{Int32,Int32},barrier::Core.LLVMPtr{UInt64,3}) where {T,S,A}
    @boundscheck PTX.smem_addr_u32(dst.ptr) % UInt32(1024) == 0 || throw(ArgumentError("TMA storage alignment"))
    ptx"cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"(
        reinterpret(Core.LLVMPtr{UInt8,3},dst.ptr),b.descriptor,origin[A],origin[3-A],barrier)
    nothing
end
