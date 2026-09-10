module PTXExt

using Tylo
using Tylo: BFloat16, PendingLoad
using PTX: @ptx_str
import PTX

# The result registers must depend on the wait at LLVM SSA level. A memory
# clobber alone cannot prevent arithmetic on already-returned SSA values
# from moving above a no-argument wait. Tied operands carry that dependency.
# Pair words as the NVPTX wide-load lowering does. Preserving this packing
# also preserves backend unrolling/register allocation in the full kernel;
# the paired attention test compares the resulting machine-code bytes.
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

@inline function Tylo.wait_load(p::PendingLoad{N}) where N
    words = _wait_words(p.words)
    Fragment(ntuple(i -> reinterpret(Float32,words[i]),Val(N)),p.ownership)
end

for N in (16,32,64)
    ld = ptx"tcgen05.ld.sync.aligned.32x32b.x$N.b32"
    st = ptx"tcgen05.st.sync.aligned.32x32b.x$N.b32"
    @eval begin
        @inline Tylo.load_async(t::Tylo.TmemPartition{Float32,$N}) = PendingLoad($ld(t.address),Tylo.Layouts.layout(t))
        @inline function Tylo.store_async!(t::Tylo.TmemPartition{Float32,$N}, f::Fragment{Float32,$N})
            Tylo._check_tmem_store(t,f)
            $st(t.address,ntuple(i -> reinterpret(UInt32,f.data[i]),Val($N)))
            nothing
        end
        @inline function Tylo.store_async!(t::Tylo.TmemPartition{BFloat16,$N}, f::PackedBF16{$N})
            Tylo._check_tmem_store(t,f)
            $st(t.address,f.data)
            nothing
        end
    end
end

@generated function Tylo.pack_bf16(f::Fragment{Float32,N}) where N
    iseven(N) || error("packing BF16 requires an even number of values")
    words = [:(PTX.bf16x2_pack(f.data[$(2i-1)],f.data[$(2i)])) for i in 1:N÷2]
    quote
        Base.@inline
        PackedBF16(($(words...),),Tylo.Layouts.layout(f))
    end
end

@generated function Tylo.store!(ptr::Core.LLVMPtr{UInt16,PTX.AS.Global},
                                    f::PackedBF16{W}) where W
    W % 4 == 0 || error("vector stores require a multiple of eight BF16 values")
    stores = [:(ptx"st.global.v4.b32"(ptr + $(16i),
                   ($( [:(f.data[$j]) for j in 4i+1:4i+4]... ),))) for i in 0:W÷4-1]
    quote
        Base.@inline
        $(stores...)
        nothing
    end
end

@inline Tylo.wait_stores() = ptx"tcgen05.wait::st.sync.aligned"()
@inline Tylo.fence_after_thread_sync() = ptx"tcgen05.fence::after_thread_sync"()
@inline Tylo.fence_before_thread_sync() = ptx"tcgen05.fence::before_thread_sync"()

include("copy.jl")
include("mma.jl")
include("rows.jl")
include("tma.jl")
include("wgmma.jl")

end
