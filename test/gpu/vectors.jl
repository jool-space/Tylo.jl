# TEST_TARGET: cc>=8.0
using Tylo.Layouts: @Layout, LocalOwnership

# Aligned tiles vectorize fragment loads and stores; tiles with natural
# alignment take the scalar path. Both must agree exactly.
function vector_kernel!(out,inp,ownership,::Val{A}) where A
    T = eltype(inp)
    lane = Int32(threadIdx().x)-Int32(1)
    m,n = size(ownership)
    # The launch guarantees 256-byte pointers and rows of whole 16-byte vectors.
    src = @inbounds GlobalTile(pointer(inp),@Layout((m,n),(n,1)),Val(A))
    dst = @inbounds GlobalTile(pointer(out),@Layout((m,n),(n,1)),Val(A))
    f = load_fragment(ownership,src,lane)
    store!(dst,f,lane)
    nothing
end

cases() = ((operand_layout(MMAAtom((16,8,16),BFloat16),Accumulator()),Float32),
           (LocalOwnership{4,2}(),Float32),
           (operand_layout(MMAAtom((16,8,16),BFloat16),OperandA()),BFloat16),
           (operand_layout(MMAAtom((16,8,32),Float8E4M3),OperandA()),Float8E4M3))

@testset "Vector access assembly" begin
    for (ownership,T) in cases(), A in (sizeof(T),16)
        tt = Tuple{CuDeviceMatrix{T,1},CuDeviceMatrix{T,1},typeof(ownership),Val{A}}
        code = compile_kernel(vector_kernel!,tt;arch=CUDACore.SMVersion(12,1,:arch),threads=32)
        save_code("vectors-$(nameof(typeof(ownership)))-$T-$A",code)
        body = entry_body(code.ptx)
        @test !occursin(".local .",body)
        @test !occursin(r"\bcall",body)
        # One access per vector when a plan derives, one per element otherwise;
        # the scalar path declares element alignment, so nothing merges.
        plan = Tylo.vector_plan(ownership,T,2,A)
        expected = plan === nothing ? Tylo._register_count(ownership) : length(plan.groups)
        @test count("ld.global.",body) == expected
        @test count("st.global.",body) == expected
    end
end

if runtime_supported(@__FILE__)
    @testset "Vector and scalar paths agree" begin
        rng = MersenneTwister(8)
        for (ownership,T) in cases()
            m,n = size(ownership)
            values = T <: AbstractFloat ? T.(rand(rng,-8:8,m,n) ./ 4) : rand(rng,T,m,n)
            inp = CuArray(permutedims(values))    # row-major storage of the m×n tile
            for A in (sizeof(T),16)
                out = CUDACore.zeros(T,n,m)
                @cuda threads=32 vector_kernel!(out,inp,ownership,Val(A))
                @test permutedims(Array(out)) == values
            end
        end
    end
end
