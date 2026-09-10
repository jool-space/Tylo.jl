@inline function Tylo._row_reduce(op,f::WarpRowFragment)
    value = PTX.Warps.warp_reduce(op,Tylo._local_reduce(op,f.data))
    RowValues(row_ownership(f),(value,))
end
@inline function Tylo._row_reduce(op::F,f::Tylo.MMAFragment{Float32,Accumulator}) where F
    lo = PTX.Warps.warp_reduce(op,op(f.data[1],f.data[2]),Val(4))
    hi = PTX.Warps.warp_reduce(op,op(f.data[3],f.data[4]),Val(4))
    RowValues(row_ownership(f),(lo,hi))
end
@generated function Tylo._row_reduce(op::F,a::Tylo.MMAAccumulator{TiledMMA{A,W,R,K}}) where {F,A,W,R,K}
    W[2] == 1 || return :(throw(ArgumentError("row reduction requires one warp along N")))
    rows = Expr[]
    for i in 1:R[1],half in 0:1
        values = [:(a.data[$(i+(j-1)*R[1])].data[$(2half+e)]) for j in 1:R[2] for e in 1:2]
        push!(rows,:(PTX.Warps.warp_reduce(op,Tylo._local_reduce(op,($(values...),)),Val(4))))
    end
    quote
        Base.@inline
        RowValues(row_ownership(a),($(rows...),))
    end
end
