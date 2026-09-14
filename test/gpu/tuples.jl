# TEST_TARGET: cc>=8.0
using Tylo: @rtuple
using Tylo.Layouts: coordinate

@generated function rtuple_values(ownership,lane,data::NTuple{N},::Val{:literal}) where N
    ex = Expr(:tuple, [:(let c=coordinate(ownership,lane,Val($(i-1)))
        data[$i] + Float32(c[1]) + Float32(c[2])
    end) for i in 1:N]...)
    quote
        Base.@inline
        $ex
    end
end
@inline function rtuple_values(ownership,lane,data::NTuple{N},::Val{:rtuple}) where N
    @rtuple(0:N-1) do e
        c = coordinate(ownership,lane,Val(e))
        data[e+1] + Float32(c[1]) + Float32(c[2])
    end
end

function rtuple_kernel!(output,input,ownership,::Val{N},::Val{Tagged}) where {N,Tagged}
    lane = Int32(threadIdx().x)-Int32(1)
    data = ntuple(i -> (@inbounds input[Int(lane)*N+i]),Val(N))
    values = rtuple_values(ownership,lane,data,Val(Tagged))
    ntuple(Val(N)) do i
        @inbounds output[Int(lane)*N+i] = values[i]
    end
    nothing
end

@testset "Static range tuple code generation" begin
    for (ownership,n) in ((operand_layout(MMAAtom((16,8,16),BFloat16),Accumulator()),4),
                           (Tylo.TmemTransfer{(32,64),2}(),64))
        codes = map((:literal,:rtuple)) do mode
            tt = Tuple{CuDeviceVector{Float32,1},CuDeviceVector{Float32,1},typeof(ownership),Val{n},Val{mode}}
            code = compile_kernel(rtuple_kernel!,tt;arch=CUDACore.SMVersion(12,1,:arch),threads=32)
            body = entry_body(code.ptx)
            @test !occursin(r"\bcall",body)
            @test !occursin(".local .",body)
            save_code("rtuple-$n-$mode",code)
            code
        end
        @test kernel_text(codes[1].image,".text.") == kernel_text(codes[2].image,".text.")
    end
end

if runtime_supported(@__FILE__)
@testset "Static range tuple coordinates on device" begin
    for (ownership,n) in ((operand_layout(MMAAtom((16,8,16),BFloat16),Accumulator()),4),
                           (Tylo.TmemTransfer{(32,64),2}(),64))
        input = CuArray(collect(Float32,1:32n))
        output = similar(input)
        expected = Float32[]
        for lane in 0:31, e in 0:n-1
            c = coordinate(ownership,Int32(lane),Val(e))
            push!(expected,Float32(lane*n+e+1)+Float32(c[1])+Float32(c[2]))
        end
        @cuda threads=32 rtuple_kernel!(output,input,ownership,Val(n),Val(:rtuple))
        @test Array(output) == expected
    end
end
end
