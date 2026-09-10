@inline function Tylo._reduce_values(op,data,::Tylo.WarpRowLayout)
    (PTX.Warps.warp_reduce(op,Tylo._local_reduce(op,data)),)
end
# The public entry point first validates this is the supported MMA ownership.
@inline function Tylo._reduce_values(op::F,data,::Tylo.Layouts.Ownership) where F
    lo = PTX.Warps.warp_reduce(op,op(data[1],data[2]),Val(4))
    hi = PTX.Warps.warp_reduce(op,op(data[3],data[4]),Val(4))
    (lo,hi)
end
@generated function Tylo._reduce_values(op::F,data,
        ::Tylo.TiledMMAOwnership{TiledMMA{A,W,R,K}}) where {F,A,W,R,K}
    W[2] == 1 || return :(throw(ArgumentError("reduction requires one warp along N")))
    rows = Expr[]
    for i in 1:R[1],half in 0:1
        values = [:(data[$(4*(i-1+(j-1)*R[1])+2half+e)]) for j in 1:R[2] for e in 1:2]
        push!(rows,:(PTX.Warps.warp_reduce(op,Tylo._local_reduce(op,($(values...),)),Val(4))))
    end
    quote
        Base.@inline
        ($(rows...),)
    end
end
