Base.@propagate_inbounds function Tylo.load_a(a::MMA16x8x16{T},t::SharedTile{T},lane::Integer) where T
    size(t) == (16,16) || throw(DimensionMismatch("A operand must be 16×16"))
    @boundscheck 0 <= lane < 32 || throw(BoundsError())
    c = (lane & oftype(lane,15),(lane >> 4)*oftype(lane,8))
    @boundscheck _check_vector(t.layout,c,2,Val(8),T)
    data = ptx"ldmatrix.sync.aligned.m8n8.x4.shared.b16"(pointer(t,c))
    Tylo.MMAFragment(T,OperandA(),data)
end
Base.@propagate_inbounds function Tylo.load_b(a::MMA16x8x16{T},t::SharedTile{T},lane::Integer) where T
    size(t) == (16,8) || throw(DimensionMismatch("B operand must be 16×8"))
    @boundscheck 0 <= lane < 32 || throw(BoundsError())
    # Lanes 16:31 also provide valid addresses, duplicating lanes 0:15.
    c = (((lane >> 3) & one(lane))*oftype(lane,8),lane & oftype(lane,7))
    @boundscheck _check_vector(t.layout,c,1,Val(8),T)
    data = ptx"ldmatrix.sync.aligned.m8n8.x2.shared.b16"(pointer(t,c))
    Tylo.MMAFragment(T,OperandB(),data)
end
for (T,name) in ((BFloat16,"bf16"),(Float16,"f16"))
    instruction = Expr(:macrocall,Symbol("@ptx_str"),LineNumberNode(0),
        "mma.sync.aligned.m16n8k16.row.col.f32.$name.$name.f32")
    @eval @inline function Tylo.mma(::MMA16x8x16{$T},
            a::Tylo.MMAFragment{$T,OperandA,4,UInt32},
            b::Tylo.MMAFragment{$T,OperandB,2,UInt32},
            c::Tylo.MMAFragment{Float32,Accumulator,4,Float32})
        Tylo.MMAFragment(Float32,Accumulator(),$instruction(a.data,b.data,c.data))
    end
end
@generated function Tylo.store!(dst::GlobalTile{Float32},
        f::Tylo.MMAFragment{Float32,Accumulator,4,Float32},lane::Integer)
    stores = [:(ptx"st.global.f32"(pointer(dst,Tylo.Layouts.coordinate(
        Tylo.Layouts.layout(f),lane,Val($i))),f.data[$(i+1)])) for i in 0:3]
    quote
        Base.@inline
        size(dst) == (16,8) || throw(DimensionMismatch("accumulator output must be 16×8"))
        @boundscheck 0 <= lane < 32 || throw(BoundsError())
        $(stores...)
        nothing
    end
end

@generated function Tylo.mma(p::TiledMMA{MMA16x8x16{T},W,R,K},
        a::SharedTile{T},b::SharedTile{T},
        accum::Tylo.MMAAccumulator{TiledMMA{MMA16x8x16{T},W,R,K}},tid::Integer) where {T,W,R,K}
    cs = [Symbol(:c_,i) for i in 1:prod(R)]
    statements = [:( $(cs[i]) = accum.data[$i] ) for i in eachindex(cs)]
    for k in 0:16:K-16
        av = [Symbol(:a_,i) for i in 1:R[1]]
        bv = [Symbol(:b_,j) for j in 1:R[2]]
        for i in 1:R[1]
            push!(statements,:($(av[i]) = load_a(p.atom,
                window(a,(wm*oftype(tid,$(16R[1]))+oftype(tid,$(16(i-1))),oftype(tid,$k)),Val((16,16))),lane)))
        end
        for j in 1:R[2]
            push!(statements,:($(bv[j]) = load_b(p.atom,
                window(b,(oftype(tid,$k),wn*oftype(tid,$(8R[2]))+oftype(tid,$(8(j-1)))),Val((16,8))),lane)))
        end
        for j in 1:R[2],i in 1:R[1]
            c = cs[i+(j-1)*R[1]]
            push!(statements,:($c = mma(p.atom,$(av[i]),$(bv[j]),$c)))
        end
    end
    m,n = 16W[1]*R[1],8W[2]*R[2]
    quote
        Base.@inline
        size(a) == ($m,$K) && size(b) == ($K,$n) || throw(DimensionMismatch("MMA plan/view shapes differ"))
        @boundscheck 0 <= tid < $(32prod(W)) || throw(BoundsError())
        lane = tid & oftype(tid,31)
        warp = tid >> 5
        wm,wn = warp % oftype(tid,$(W[1])),warp ÷ oftype(tid,$(W[1]))
        @inbounds begin
            $(statements...)
        end
        Tylo.MMAAccumulator(p,($(cs...),))
    end
end

@generated function Tylo.store!(p::TiledMMA{A,W,R,K},dst::GlobalTile{Float32},
        accum::Tylo.MMAAccumulator{TiledMMA{A,W,R,K}},tid::Integer) where {A,W,R,K}
    stores = [:(store!(window(dst,
        (wm*oftype(tid,$(16R[1]))+oftype(tid,$(16(i-1))),
         wn*oftype(tid,$(8R[2]))+oftype(tid,$(8(j-1)))),Val((16,8))),
        accum.data[$(i+(j-1)*R[1])],lane)) for j in 1:R[2] for i in 1:R[1]]
    quote
        Base.@inline
        size(dst) == ($(16W[1]*R[1]),$(8W[2]*R[2])) || throw(DimensionMismatch("output plan/view shapes differ"))
        @boundscheck 0 <= tid < $(32prod(W)) || throw(BoundsError())
        lane = tid & oftype(tid,31)
        warp = tid >> 5
        wm,wn = warp % oftype(tid,$(W[1])),warp ÷ oftype(tid,$(W[1]))
        @inbounds begin
            $(stores...)
        end
        nothing
    end
end
