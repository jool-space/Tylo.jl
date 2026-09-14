# Warp communication for the derived reduction recipes.
@inline Tylo._warp_reduce(op::F,x,::Val{W}) where {F,W} = PTX.Warps.warp_reduce(op,x,Val(W))
@inline function Tylo._shuffle_xor(op::F,x::T,::Val{Offset}) where {F,T,Offset}
    partner = ptx"shfl.sync.bfly.b32"(reinterpret(UInt32,x),UInt32(Offset),UInt32(0x1f),0xffffffff % UInt32)
    op(x,reinterpret(T,partner))
end
