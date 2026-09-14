# TEST_TARGET: cc>=8.0
using Tylo.Layouts: @Layout
row_fragment(o::Tylo.Layouts.Ownership,data) = Fragment(data,o)
function array_pointwise!(output,input,kind,::Val{N},::Val{Permute}) where {N,Permute}
    t=Int32(threadIdx().x)-Int32(1)
    f=Fragment(row_fragment(kind,ntuple(i -> @inbounds(input[N*t+i]),Val(N))))
    g=Permute ? permutedims(f) : f
    mask=g .> 0f0 # Materialized Bool values retain the same ownership.
    rounded=Float32.(Float16.(g))
    result=ifelse.(mask,(rounded .+ 2f0).^2,-1f0)
    ntuple(Val(N)) do i
        @inbounds output[N*t+i]=result.data[i]
    end
    nothing
end
function array_softmax!(output,minima,input,kind,::Val{N},::Val{Permute}) where {N,Permute}
    t=Int32(threadIdx().x)-Int32(1)
    f=Fragment(row_fragment(kind,ntuple(i -> @inbounds(input[N*t+i]),Val(N))))
    g=Permute ? permutedims(f) : f
    dims=Permute ? 1 : 2
    m=maximum(g;dims)
    weights=exp.(((x,m) -> m == -Inf32 ? -Inf32 : x-m).(g,m))
    result=((x,s) -> s == 0f0 ? 0f0 : x/s).(weights,sum(weights;dims))
    lo=minimum(g;dims)
    ntuple(Val(N)) do i
        @inbounds output[N*t+i]=result.data[i]
    end
    ntuple(Val(length(lo.data))) do i
        @inbounds minima[length(lo.data)*t+i]=lo.data[i]
    end
    nothing
end

begin # assembly checks
@testset "Fragment broadcast and permutation assembly" begin
    patch=Tylo.Layouts.Ownership(Val((16,8)),@Layout(((8,4),(2,2)),((2,32),(1,16))))
    for permute in (false,true)
        tt=Tuple{CuDeviceVector{Float32,1},CuDeviceVector{Float32,1},typeof(patch),Val{4},Val{permute}}
        code=compile_kernel(array_pointwise!,tt;arch=CUDACore.SMVersion(12,1,:arch),threads=32)
        body=entry_body(code.ptx)
        @test !occursin(".local .",body)
        @test !occursin(r"\bcall",body)
        @test !occursin("shfl.sync",body)
    end
    for (kind,n,shuffles) in ((Val(:local),3,0),(Val(:warp),3,15),(MMAAtom((16,8,16),BFloat16),4,12))
        codes=String[]
        for permute in (false,true)
            tt=Tuple{CuDeviceVector{Float32,1},CuDeviceVector{Float32,1},CuDeviceVector{Float32,1},typeof(kind),Val{n},Val{permute}}
            code=compile_kernel(array_softmax!,tt;arch=CUDACore.SMVersion(12,1,:arch),threads=32)
            save_code("array-softmax-$(kind isa Val ? typeof(kind).parameters[1] : nameof(typeof(kind)))-$permute",code)
            body=entry_body(code.ptx)
            @test !occursin(".local .",body)
            @test !occursin(r"\bcall",body)
            @test count("shfl.sync.bfly",body) == shuffles
            push!(codes,body)
        end
        # Entry symbols encode the different Julia Val types; instruction
        # counts should still match when only logical coordinates are changed.
        @test count(';',codes[1]) == count(';',codes[2])
    end
end
end

if runtime_supported(@__FILE__)
@testset "Fragment arithmetic and reductions on either logical axis" begin
    atom=MMAAtom((16,8,16),BFloat16)
    cases=Any[(Val(:local),3,32),(Val(:warp),3,64),(atom,4,32)]
    for wm in (1,2),rm in (1,2),rn in (1,3)
        push!(cases,(TiledMMA(atom,Val((wm,1)),Val((rm,rn)),Val(16)),4rm*rn,32wm))
    end
    for (kind,n,threads) in cases
        values=randn(MersenneTwister(17n+threads),Float32,n,threads)
        # Entirely masked logical row, including every participating lane.
        for t in 0:threads-1,e in 0:n-1
            reference_row(kind,n,t,e) == 0 && (values[e+1,t+1]=-Inf32)
        end
        byrow=Dict{Int,Vector{Float32}}()
        for t in 0:threads-1,e in 0:n-1
            push!(get!(byrow,reference_row(kind,n,t,e),Float32[]),values[e+1,t+1])
        end
        f=row_fragment(kind,ntuple(_ -> 0f0,Val(n)))
        nr=Tylo._register_count(Tylo._reduced_ownership(Tylo.Layouts.layout(f),Val(2)))
        input=CuArray(vec(values)); output=similar(input); minima=CuArray{Float32}(undef,nr*threads)
        expected=similar(values); expected_min=zeros(Float32,nr,threads)
        for t in 0:threads-1,e in 0:n-1
            vals=Float64.(byrow[reference_row(kind,n,t,e)])
            m=maximum(vals)
            expected[e+1,t+1]=m == -Inf ? 0f0 : exp(values[e+1,t+1]-m)/sum(exp.(vals .- m))
        end
        for t in 0:threads-1,i in 0:nr-1
            e=kind isa Union{Val{:local},Val{:warp}} ? 0 : 4*(i÷2)+2*(i%2)
            expected_min[i+1,t+1]=minimum(byrow[reference_row(kind,n,t,e)])
        end
        results=Vector{Float32}[]
        for permute in (false,true)
            @cuda threads=threads array_pointwise!(output,input,kind,Val(n),Val(permute))
            @test Array(output) == vec(ifelse.(values .> 0f0,(Float32.(Float16.(values)) .+ 2f0).^2,-1f0))
            @cuda threads=threads array_softmax!(output,minima,input,kind,Val(n),Val(permute))
            got=Array(output)
            @test got ≈ vec(expected) atol=2e-6 rtol=2e-5
            @test reshape(Array(minima),nr,threads) == expected_min
            push!(results,got)
        end
        @test results[1] == results[2]
    end
    patch=Tylo.Layouts.Ownership(Val((16,8)),@Layout(((8,4),(2,2)),((2,32),(1,16))))
    values=randn(MersenneTwister(7),Float32,128)
    input=CuArray(values);output=similar(input)
    for permute in (false,true)
        @cuda threads=32 array_pointwise!(output,input,patch,Val(4),Val(permute))
        @test Array(output) == ifelse.(values .> 0f0,(Float32.(Float16.(values)) .+ 2f0).^2,-1f0)
    end
end
end
