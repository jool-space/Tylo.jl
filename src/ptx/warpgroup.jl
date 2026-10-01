@inline function release_registers(::Val{N}) where N
    _register_limit(Val(N))
    ptx"setmaxnreg.dec.sync.aligned.u32"(Val(N))
end
@inline function acquire_registers(::Val{N}) where N
    _register_limit(Val(N))
    ptx"setmaxnreg.inc.sync.aligned.u32"(Val(N))
end
# A constant limit, as setmaxnreg's immediate.
Base.@constprop :aggressive @inline release_registers(n::Integer) = release_registers(Val(Int(n)))
Base.@constprop :aggressive @inline acquire_registers(n::Integer) = acquire_registers(Val(Int(n)))
@inline _register_limit(::Val{N}) where N =
    N isa Int && 24 <= N <= 256 && N % 8 == 0 || throw(ArgumentError("a register limit is a multiple of 8 from 24 to 256"))
@inline warp_uniform(x::T) where T<:Union{Int32,UInt32,Float32} =
    reinterpret(T,ptx"shfl.sync.idx.b32"(reinterpret(UInt32,x),UInt32(0),UInt32(0x1f),0xffffffff % UInt32))
@inline warp_uniform(x::Integer) = oftype(x,warp_uniform(x % Int32))
@inline elect_one() = ptx"elect.sync"(0xffffffff % UInt32)[2]
