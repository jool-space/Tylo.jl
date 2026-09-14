# TEST_TARGET: cc>=8.0
function row_probe!(sums,maxima,shifted,input,kind,::Val{N}) where N
    t=Int32(threadIdx().x)-Int32(1)
    f=row_fragment(kind,ntuple(i -> @inbounds(input[N*t+i]),Val(N)))
    s,m=sum(f; dims=2),maximum(f; dims=2)
    y=row_words(f .- m)
    ntuple(Val(length(s.data))) do i
        @inbounds sums[length(s.data)*t+i]=s.data[i]
        @inbounds maxima[length(s.data)*t+i]=m.data[i]
    end
    ntuple(Val(N)) do i
        @inbounds shifted[N*t+i]=y[i]
    end
    nothing
end


begin # assembly checks
@testset "Row collective assembly" begin
    for (kind,n,nshfl) in ((Val(:local),17,0),(Val(:warp),3,10),(MMAAtom((16,8,16),BFloat16),4,8)),
        arch in (CUDACore.SMVersion(8,0),CUDACore.SMVersion(12,1,:arch))
        tt=Tuple{CuDeviceVector{Float32,1},CuDeviceVector{Float32,1},CuDeviceVector{Float32,1},CuDeviceVector{Float32,1},typeof(kind),Val{n}}
        code=compile_kernel(row_probe!,tt;arch,threads=32)
        save_code("rows-$(nameof(typeof(kind)))-$n-$arch",code)
        body=entry_body(code.ptx)
        @test !occursin(".local .",body)
        @test !occursin(r"\bcall",body)
        @test count("shfl.sync.bfly",body) == nshfl
    end
end
end
if runtime_supported(@__FILE__)
@testset "Row reductions, result replication and broadcasts" begin
    cases=Any[(Val(:local),n,32) for n in (1,3,17)]
    append!(cases,[(Val(:warp),n,64) for n in (1,3,17)])
    push!(cases,(MMAAtom((16,8,16),BFloat16),4,32))
    for wm in (1,2),rm in (1,2),rn in (1,3)
        push!(cases,(TiledMMA(MMAAtom((16,8,16),BFloat16),Val((wm,1)),Val((rm,rn)),Val(16)),4rm*rn,32wm))
    end
    for (kind,n,threads) in cases
        values=randn(MersenneTwister(4n+threads),Float32,n,threads)
        # Large cancelling values mixed with small asymmetrical values.
        values[1,1]=4096f0; values[1,2]=-4096f0
        byrow=Dict{Int,Vector{Float32}}()
        for t in 0:threads-1,e in 0:n-1
            push!(get!(byrow,reference_row(kind,n,t,e),Float32[]),values[e+1,t+1])
        end
        f=row_fragment(kind,ntuple(_ -> 0f0,Val(n)))
        nr=Tylo._register_count(Tylo._reduced_ownership(Tylo.Layouts.layout(f),Val(2)))
        input=CuArray(vec(values)); sums=CuArray{Float32}(undef,nr*threads)
        maxima=similar(sums); shifted=similar(input)
        @cuda threads=threads row_probe!(sums,maxima,shifted,input,kind,Val(n))
        ss,mm,yy=reshape(Array(sums),nr,threads),reshape(Array(maxima),nr,threads),reshape(Array(shifted),n,threads)
        expected_shift=similar(values); expected_sum=zeros(Float64,nr,threads); expected_max=similar(expected_sum); bound=similar(expected_sum)
        for t in 0:threads-1,i in 0:nr-1
            # MMA's row slots enumerate the two rows of each M atom repeat.
            e=kind isa Union{Val{:local},Val{:warp}} ? 0 : 4*(i÷2)+2*(i%2)
            vals=byrow[reference_row(kind,n,t,e)]
            expected_sum[i+1,t+1]=sum(Float64,vals); expected_max[i+1,t+1]=maximum(vals)
            bound[i+1,t+1]=2e-6*sum(abs,Float64.(vals))+1e-6
        end
        for t in 0:threads-1,e in 0:n-1
            expected_shift[e+1,t+1]=values[e+1,t+1]-maximum(byrow[reference_row(kind,n,t,e)])
        end
        @test all(abs.(ss .- expected_sum) .<= bound)
        @test mm == expected_max
        @test yy == expected_shift
    end
end
end
