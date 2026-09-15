using PTX: @ptx_str
import PTX

# The result registers must depend on the wait at LLVM SSA level. A memory
# clobber alone cannot prevent arithmetic on already-returned SSA values
# from moving above a no-argument wait. Tied operands carry that dependency.
# Pair words as the NVPTX wide-load lowering does. Preserving this packing
# also preserves backend unrolling/register allocation in the full kernel;
# the paired attention test compares the resulting machine-code bytes.
@inline _load_tuple(x::UInt32) = (x,)
@inline _load_tuple(x::Tuple) = x
@generated function _wait_words(words::NTuple{1,UInt32})
    ir = PTX.convergent_asm_ir("tcgen05.wait::ld.sync.aligned;", "=r,0,~{memory}", UInt32, [UInt32])
    :( (Base.llvmcall(($ir,"entry"), UInt32, Tuple{UInt32}, words[1]),) )
end

@generated function _wait_words(words::NTuple{N,UInt32}) where N
    iseven(N) && N > 0 || error("a load wait requires complete register pairs")
    P = N÷2
    constraints = join(vcat(fill("=l",P),string.(0:P-1),["~{memory}"]),",")
    ir = PTX.convergent_asm_ir("tcgen05.wait::ld.sync.aligned;",
        constraints,NTuple{P,UInt64},fill(UInt64,P))
    args = [:(UInt64(words[$(2i-1)]) | (UInt64(words[$(2i)]) << 32)) for i in 1:P]
    values = [isodd(i) ? :(pairs[$((i+1)÷2)] % UInt32) :
                        :((pairs[$(i÷2)] >> 32) % UInt32) for i in 1:N]
    quote
        Base.@inline
        pairs = Base.llvmcall(($ir,"entry"),NTuple{$P,UInt64},NTuple{$P,UInt64},$(args...))
        ($(values...),)
    end
end

@inline function wait_load(p::PendingLoad{Float32,N}) where N
    words = _wait_words(p.words)
    Fragment(@rtuple(i -> reinterpret(Float32,words[i]), 1:N),p.ownership)
end

@inline function wait_load(p::PendingLoad{T,N}) where {T<:Union{BFloat16,Float16},N}
    PackedFragment(T, _wait_words(p.words), p.ownership)
end

for N in (1,2,4,8,16,32,64,128)
    ld = ptx"tcgen05.ld.sync.aligned.32x32b.x$N.b32"
    st = ptx"tcgen05.st.sync.aligned.32x32b.x$N.b32"
    @eval begin
        @inline load_async(t::TmemPartition{Float32,$N}) = PendingLoad(Float32, _load_tuple($ld(t.address)),Layouts.layout(t))
        @inline load_async(t::TmemPartition{T,$N}) where {T<:Union{BFloat16,Float16}} =
            PendingLoad(T,_load_tuple($ld(t.address)),Layouts.layout(t))
        @inline function store_async!(t::TmemPartition{Float32,$N}, f::Fragment{Float32,$N})
            _check_tmem_store(t,f)
            $st(t.address,@rtuple(i -> reinterpret(UInt32,f.data[i]), 1:$N))
            nothing
        end
        @inline function store_async!(t::TmemPartition{T,$N}, f::PackedFragment{T,$N}) where {T<:Union{BFloat16,Float16}}
            _check_tmem_store(t,f)
            $st(t.address,f.data)
            nothing
        end
    end
end


@generated function store!(ptr::Core.LLVMPtr{U,PTX.AS.Global},
                                    f::PackedFragment{T,W}) where {U,T,W}
    U in (T,UInt16) || return :(throw(ArgumentError("packed store element type differs from pointer")))
    stores = [:(ptx"st.global.v4.b32"(ptr + $(16i),
                   ($( [:(f.data[$j]) for j in 4i+1:4i+4]... ),))) for i in 0:W÷4-1]
    offset = 4*(W÷4)
    if W % 4 >= 2
        push!(stores,:(ptx"st.global.v2.b32"(ptr+$(4offset),(f.data[$(offset+1)],f.data[$(offset+2)]))))
        offset += 2
    end
    if isodd(W)
        push!(stores,:(ptx"st.global.b32"(ptr+$(4offset),f.data[$(offset+1)])))
    end
    quote
        Base.@inline
        $(stores...)
        nothing
    end
end

@inline wait_stores() = ptx"tcgen05.wait::st.sync.aligned"()
@inline fence_after_thread_sync() = ptx"tcgen05.fence::after_thread_sync"()
@inline fence_before_thread_sync() = ptx"tcgen05.fence::before_thread_sync"()

include("copy.jl")
include("mma.jl")
include("rows.jl")
include("tma.jl")
include("wgmma.jl")
include("tcgen05.jl")
