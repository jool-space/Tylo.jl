# Tied SSA operands are necessary: memory clobbers alone do not stop arithmetic
# on an async result from moving above the wait (or initialization below fence).
@generated function _wgmma_register_barrier(::Val{Kind},d::NTuple{R,Float32}) where {Kind,R}
    asm = Kind === :fence ? "wgmma.fence.sync.aligned;" :
          Kind isa Tuple{Symbol,Int} && Kind[1] === :wait && Kind[2] >= 0 ? "wgmma.wait_group.sync.aligned $(Kind[2]);" :
          error("unknown WGMMA barrier")
    constraints = join(vcat(fill("=f",R),string.(0:R-1),["~{memory}"]),",")
    ir = PTX.convergent_asm_ir(asm,constraints,NTuple{R,Float32},fill(Float32,R))
    args = [:(d[$i]) for i in 1:R]
    quote
        Base.@inline
        Base.llvmcall(($ir,"entry"),NTuple{$R,Float32},NTuple{$R,Float32},$(args...))
    end
end

Base.@propagate_inbounds function operand(p::WGMMAAtom{T,N,K},role::Role,
        t::SharedTile{T,L},origin::Tuple{Int32,Int32}=(Int32(0),Int32(0))) where {T,N,K,Role<:Union{OperandA,OperandB},L}
    # A/B compatibility is structural, even when bounds checks are elided.
    s = _operand_structure(L,T,Role)
    s.major === :K || throw(ArgumentError("shared/shared WGMMA operands are K-major"))
    K <= s.row_elements || throw(ArgumentError("the plan's K does not fit one swizzle row"))
    @boundscheck begin
        _smem_address(t.ptr) % UInt32(8s.swizzle_bytes) == 0 || throw(ArgumentError("WGMMA storage alignment"))
        validate_wgmma(p,role,t.layout,origin)
    end
    # Non-K origin is aligned to a full eight-row swizzle cycle. Start address
    # is unswizzled; hardware applies the descriptor's swizzle to each access.
    start = _smem_address(t.ptr) + (_unswizzled(t.layout)(origin)*Int32(sizeof(T))) % UInt32
    desc = PTX.wgmma_descriptor(start;leading_byte_offset=16,stride_byte_offset=s.stride_bytes,
        swizzle=_wgmma_swizzle(s.swizzle_bytes))
    WGMMAOperand{typeof(p),Role}(desc)
end

# Issue a plan's K/16 instructions and commit them as one group. A ready
# accumulator's registers were last written by ordinary instructions, so a
# wgmma.fence orders them first; a pending one's were last written by a
# wgmma of the same shape, which the hardware orders by itself (PTX ISA
# §9.7.17.7.1), so a chain of issues needs no fence and no wait.
function _mma_async_expr(::Type{WGMMAAtom{T,N,K,P}},R,fence::Bool) where {T,N,K,P}
    dtype = T === BFloat16 ? "bf16" : "f16"
    instruction = ptx"wgmma.mma_async.sync.aligned.m64n$(N)k16.f32.$dtype.$dtype"
    init = [:( $(Symbol(:part,j)) = ($( [:(d[$(i+j*(N÷2))]) for i in 1:N÷2]... ),)) for j in 0:P-1]
    ops = [begin
        name = Symbol(:part,j%P)
        :($name = $instruction($name,_k_step(a,Val($j)),_k_step(b,Val($j)),Val(true)))
    end for j in 0:K÷16-1]
    vals = [:($(Symbol(:part,j))[$i]) for j in 0:P-1 for i in 1:N÷2]
    quote
        Base.@inline
        d = $(fence ? :(_wgmma_register_barrier(Val(:fence),c.data)) : :(c.data))
        $(init...)
        $(ops...)
        ptx"wgmma.commit_group.sync.aligned"()
        PendingWGMMA{typeof(p),$R}(($(vals...),))
    end
end
@generated mma_async(p::A,a::WGMMAOperand{A,OperandA},b::WGMMAOperand{A,OperandB},
                     c::WGMMAAccumulator{A,R}) where {A<:WGMMAAtom,R} = _mma_async_expr(A,R,true)
@generated mma_async(p::A,a::WGMMAOperand{A,OperandA},b::WGMMAOperand{A,OperandB},
                     c::PendingWGMMA{A,R}) where {A<:WGMMAAtom,R} = _mma_async_expr(A,R,false)
# The descriptor of the `j`-th K=16 instruction: 32 bytes further along K.
@inline _k_step(o::WGMMAOperand,::Val{J}) where J = PTX.step_desc(o.descriptor,32J)
@inline wait_mma(c::PendingWGMMA) = wait_mma(c,0)
Base.@constprop :aggressive @inline wait_mma(c::PendingWGMMA,n::Integer) = _wait_mma(c,Val(Int(n)))
@inline function _wait_mma(c::PendingWGMMA{P},::Val{0}) where P
    WGMMAAccumulator(_plan(P),_wgmma_register_barrier(Val((:wait,0)),c.data))
end
# Older groups complete; this one may still be in flight, so it stays pending.
@inline function _wait_mma(c::PendingWGMMA{P,R},::Val{N}) where {P,R,N}
    N isa Int && N > 0 || throw(ArgumentError("wait for at most a nonnegative constant number of groups"))
    PendingWGMMA{P,R}(_wgmma_register_barrier(Val((:wait,N)),c.data))
end
# Reconstitute an isbits static plan, including validated constructor semantics.
_plan(::Type{WGMMAAtom{T,N,K,P}}) where {T,N,K,P} = WGMMAAtom{T,N,K,P}()
Base.@propagate_inbounds operand(p::WGMMAAtom,role::Union{OperandA,OperandB},t::SharedTile,origin::Tuple{Integer,Integer}) =
    operand(p,role,t,map(x -> x % Int32,origin))
