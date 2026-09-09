# Tied SSA operands are necessary: memory clobbers alone do not stop arithmetic
# on an async result from moving above the wait (or initialization below fence).
@generated function _wgmma_register_barrier(::Val{Kind},d::NTuple{R,Float32}) where {Kind,R}
    Kind in (:fence,:wait) || error("unknown WGMMA barrier")
    asm = Kind === :fence ? "wgmma.fence.sync.aligned;" : "wgmma.wait_group.sync.aligned 0;"
    constraints = join(vcat(fill("=f",R),string.(0:R-1),["~{memory}"]),",")
    ir = PTX.convergent_asm_ir(asm,constraints,NTuple{R,Float32},fill(Float32,R))
    args = [:(d[$i]) for i in 1:R]
    quote
        Base.@inline
        Base.llvmcall(($ir,"entry"),NTuple{$R,Float32},NTuple{$R,Float32},$(args...))
    end
end

Base.@propagate_inbounds function Tylo.wgmma_operand(p::Tylo.WGMMA64{T,N,K},role::Role,
        t::SharedTile{T,Tylo.TMASharedLayout{S,A}},origin::Tuple{Int32,Int32}=(Int32(0),Int32(0))) where {T,N,K,Role<:Union{OperandA,OperandB},S,A}
    # A/B compatibility is structural, even when bounds checks are elided.
    A == (Role === OperandA ? 2 : 1) || throw(ArgumentError("operand K axis does not match role"))
    @boundscheck begin
        PTX.smem_addr_u32(t.ptr) % UInt32(1024) == 0 || throw(ArgumentError("WGMMA storage alignment"))
        Tylo.validate_wgmma(p,role,t.layout,origin)
    end
    k,r = origin[A],origin[3-A]
    # Non-K origin is aligned to a full eight-row swizzle cycle. Start address
    # is unswizzled; hardware applies the descriptor's swizzle to each access.
    start = PTX.smem_addr_u32(t.ptr) + (r*Int32(128)+k*Int32(2)) % UInt32
    desc = PTX.wgmma_descriptor(start;leading_byte_offset=16,stride_byte_offset=1024,
        swizzle=PTX.WgmmaSwizzle.B128)
    Tylo.WGMMAOperand{typeof(p),Role}(desc)
end

@generated function Tylo.mma_async(p::Tylo.WGMMA64{T,N,K,P},
        a::Tylo.WGMMAOperand{Tylo.WGMMA64{T,N,K,P},OperandA},
        b::Tylo.WGMMAOperand{Tylo.WGMMA64{T,N,K,P},OperandB},
        c::Tylo.WGMMAAccumulator{Tylo.WGMMA64{T,N,K,P},R}) where {T,N,K,P,R}
    dtype = T === Tylo.BFloat16 ? "bf16" : "f16"
    instruction = Expr(:macrocall,Symbol("@ptx_str"),LineNumberNode(0),
        "wgmma.mma_async.sync.aligned.m64n$(N)k16.f32.$dtype.$dtype")
    init = [:( $(Symbol(:part,j)) = ($( [:(d[$(i+j*(N÷2))]) for i in 1:N÷2]... ),)) for j in 0:P-1]
    ops = [begin
        name = Symbol(:part,j%P)
        :($name = $instruction($name,PTX.step_desc(a.descriptor,$(32j)),PTX.step_desc(b.descriptor,$(32j)),Val(true)))
    end for j in 0:K÷16-1]
    vals = [:($(Symbol(:part,j))[$i]) for j in 0:P-1 for i in 1:N÷2]
    quote
        Base.@inline
        d = _wgmma_register_barrier(Val(:fence),c.data)
        $(init...)
        $(ops...)
        ptx"wgmma.commit_group.sync.aligned"()
        Tylo.PendingWGMMA{typeof(p),$R}(($(vals...),))
    end
end
@inline function Tylo.wait_mma(c::Tylo.PendingWGMMA{P}) where P
    Tylo.WGMMAAccumulator(Tylo._plan(P),_wgmma_register_barrier(Val(:wait),c.data))
end
# Reconstitute an isbits static plan, including validated constructor semantics.
Tylo._plan(::Type{Tylo.WGMMA64{T,N,K,P}}) where {T,N,K,P} = Tylo.WGMMA64(T,Val(N),Val(K),Val(P))

@generated function Tylo.store!(dst::GlobalTile,c::Tylo.WGMMAFragment{N},thread::Int32) where N
    stores = [:(unsafe_store!(pointer(dst,Tylo.Layouts.coordinate(Tylo.Layouts.layout(c),thread,Val($i))),c.data[$(i+1)])) for i in 0:N÷2-1]
    quote
        Base.@inline
        $(stores...)
        nothing
    end
end
