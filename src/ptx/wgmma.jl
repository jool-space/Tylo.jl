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
    s.major === :K || sizeof(T) == 2 || throw(ArgumentError("only 16-bit WGMMA operands may be MN-major"))
    s.major === :MN || K <= s.row_elements || throw(ArgumentError("the plan's K does not fit one swizzle row"))
    @boundscheck begin
        _smem_address(t.ptr) % UInt32(8s.swizzle_bytes) == 0 || throw(ArgumentError("WGMMA storage alignment"))
        validate_wgmma(p,role,t.layout,origin)
    end
    # The origin is aligned to a full eight-row swizzle cycle. Start address
    # is unswizzled; hardware applies the descriptor's swizzle to each access.
    layout = _unswizzled(t.layout)
    start = _smem_address(t.ptr) + (layout(origin)*Int32(sizeof(T))) % UInt32
    desc = PTX.wgmma_descriptor(start;leading_byte_offset=s.leading_bytes,stride_byte_offset=s.stride_bytes,
        swizzle=_wgmma_swizzle(s.swizzle_bytes))
    WGMMAOperand{typeof(p),Role,s.major,_instruction_step(typeof(layout),T,Role,s.major)}(desc)
end
# Bytes between the instructions' starts: 32 along a K-major row; for an
# MN-major operand, one instruction's K rows through the layout.
@generated function _instruction_step(::Type{L},::Type{T},::Type{Role},::Val{Major}=Val(:K)) where {L,T,Role,Major}
    Major === :K && return 32
    l = _static_instance(L <: Layouts.Window ? L.parameters[2] : L)
    kaxis = _k_axis(Role)
    unit = kaxis == 1 ? (1,0) : (0,1)
    rows = 32 ÷ sizeof(T)
    :($(Int(l(map(x -> Int32(rows*x),unit))-l((Int32(0),Int32(0))))*sizeof(T)))
end
_instruction_step(::Type{L},::Type{T},::Type{Role},major::Symbol) where {L,T,Role} = _instruction_step(L,T,Role,Val(major))

# Issue a plan's instructions, 32 bytes of K each, and commit them as one
# group. Tylo spells the instructions itself, so that shared operands may be
# transposed (MN-major) and A may come from registers. Every issue fences, a
# chained one too: the hardware orders wgmma after wgmma on the same
# accumulator (PTX ISA §9.7.17.7.1), but ptxas then inserts a warpgroup
# arrive of its own, and where that lands on a path it cannot prove
# convergent (after an mbarrier wait loop, say) it serializes every wgmma in
# the kernel. The explicit fence is the same single instruction.
_wgmma_dtype(::Type{T}) where T = T === BFloat16 ? "bf16" : T === Float16 ? "f16" : T === Float8_E4M3FN ? "e4m3" : "e5m2"
# With `d === nothing`, a fresh product: scale-d 0 ignores the accumulator,
# so its registers are outputs only and need no zeroing.
@generated function _wgmma(::Val{N},::Type{T},::Val{Transposes},d,a,b::UInt64) where {N,T,Transposes}
    fresh = d === Nothing
    R = fresh ? N ÷ 2 : length(d.parameters)
    dt = _wgmma_dtype(T)
    registers = a <: NTuple{4,UInt32}
    # Transpose bits exist for 16-bit inputs only, and for A only in shared memory.
    tail = sizeof(T) != 2 ? "" : registers ? ", $(Int(Transposes[2]))" : ", $(Int(Transposes[1])), $(Int(Transposes[2]))"
    outputs = "{" * join(("\$$i" for i in 0:R-1),", ") * "}"
    aslot = registers ? "{" * join(("\$$(R+i)" for i in 0:3),", ") * "}" : "\$$R"
    bslot = registers ? "\$$(R+4)" : "\$$(R+1)"
    asm = "wgmma.mma_async.sync.aligned.m64n$(N)k$(_instruction_k(T)).f32.$dt.$dt $outputs, $aslot, $bslot, $(fresh ? 0 : 1), 1, 1$tail;"
    inputs = registers ? ["r","r","r","r","l"] : ["l","l"]
    ties = fresh ? String[] : string.(0:R-1)
    constraints = join(vcat(fill("=f",R),inputs,ties,["~{memory}"]),",")
    accumulator = fresh ? DataType[] : fill(Float32,R)
    argtypes = registers ? [UInt32,UInt32,UInt32,UInt32,UInt64,accumulator...] : [UInt64,UInt64,accumulator...]
    ir = PTX.convergent_asm_ir(asm,constraints,NTuple{R,Float32},argtypes)
    dargs = fresh ? Expr[] : [:(d[$i]) for i in 1:R]
    args = registers ? [[:(a[$i]) for i in 1:4];:b;dargs] : [:a;:b;dargs]
    quote
        Base.@inline
        Base.llvmcall(($ir,"entry"),NTuple{$R,Float32},Tuple{$(argtypes...)},$(args...))
    end
end
function _mma_async_expr(::Type{WGMMAAtom{T,N,K,P}},R,A,B;fresh::Bool=false) where {T,N,K,P}
    ik = _instruction_k(T)
    registers = A <: Fragment
    transposes = (!registers && A.parameters[3] === :MN, B.parameters[3] === :MN)
    # A fresh product starts each partial accumulator with scale-d 0.
    init = fresh ? [:($(Symbol(:part,j)) = nothing) for j in 0:P-1] :
        [:( $(Symbol(:part,j)) = ($( [:(d[$(i+j*(N÷2))]) for i in 1:N÷2]... ),)) for j in 0:P-1]
    ops = [begin
        name = Symbol(:part,j%P)
        aop = registers ? :((words[$(4j+1)],words[$(4j+2)],words[$(4j+3)],words[$(4j+4)])) : :(_k_step(a,Val($j)))
        :($name = _wgmma(Val($N),$T,Val($transposes),$name,$aop,_k_step(b,Val($j))))
    end for j in 0:K÷ik-1]
    vals = [:($(Symbol(:part,j))[$i]) for j in 0:P-1 for i in 1:N÷2]
    quote
        Base.@inline
        $(registers ? :(words = _pack_words(a)) : nothing)
        $(fresh ? :(ptx"wgmma.fence.sync.aligned"()) : :(d = _wgmma_register_barrier(Val(:fence),c.data)))
        $(init...)
        $(ops...)
        ptx"wgmma.commit_group.sync.aligned"()
        PendingWGMMA{typeof(p),$R}(($(vals...),))
    end
end
@generated function mma_async(p::A,a::WGMMAOperand{A,OperandA},b::WGMMAOperand{A,OperandB},
                              c::Union{WGMMAAccumulator{A,R},PendingWGMMA{A,R}}) where {A<:WGMMAAtom,R}
    _mma_async_expr(A,R,a,b)
end
# A fresh product: no accumulator, none zeroed.
@generated function mma_async(p::A,a::Union{WGMMAOperand{A,OperandA},Fragment},b::WGMMAOperand{A,OperandB}) where {A<:WGMMAAtom}
    if a <: Fragment
        a.parameters[3] === typeof(_operand_layout(A(),OperandA())) ||
            return :(throw(ArgumentError("the A fragment is not in the atom's operand layout")))
    end
    _mma_async_expr(A,_wg_registers(A()),a,b;fresh=true)
end
# A in registers: a fragment in the operand layout, the accumulator's for 64×K.
@generated function mma_async(p::A,a::Fragment{T,NA,L},b::WGMMAOperand{A,OperandB},
                              c::Union{WGMMAAccumulator{A,R},PendingWGMMA{A,R}}) where {T,N,K,P,A<:WGMMAAtom{T,N,K,P},NA,L,R}
    L === typeof(_operand_layout(A(),OperandA())) ||
        return :(throw(ArgumentError("the A fragment is not in the atom's operand layout")))
    _mma_async_expr(A,R,a,b)
end
# The descriptor of the `j`-th instruction, a constant step further along K.
@inline _k_step(o::WGMMAOperand{P,Role,Major,Step},::Val{J}) where {P,Role,Major,Step,J} = PTX.step_desc(o.descriptor,Step*J)
# With `newer`, `c`'s group is older than the `newer` most recent: waiting
# until only those are in flight completes it.
Base.@constprop :aggressive @inline function wait_mma(c::PendingWGMMA{P};newer::Integer=0) where P
    WGMMAAccumulator(_plan(P),_wgmma_register_barrier(Val((:wait,Int(newer))),c.data))
end
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
