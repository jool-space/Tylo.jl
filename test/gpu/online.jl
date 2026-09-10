include("../../examples/softmax/streaming.jl")

function online_probe!(maxima,sums,numerator,input,values,kind,::Val{N}) where N
    tid=Int32(threadIdx().x)-Int32(1)
    prototype=row_fragment(kind,ntuple(_ -> 0f0,Val(N)))
    state=SoftmaxState(prototype); out=row_sum(prototype)
    for chunk in Int32(1):size(input,3)%Int32
        f=row_fragment(kind,ntuple(e -> @inbounds(input[e,tid+Int32(1),chunk]),Val(N)))
        v=row_fragment(kind,ntuple(e -> @inbounds(values[e,tid+Int32(1),chunk]),Val(N)))
        u=softmax_update(state,f)
        p=row_words(u.weights); vv=row_words(v)
        weighted=row_fragment(kind,ntuple(e -> p[e]*vv[e],Val(N)))
        out=Tylo._row_zip(muladd,u.rescale,out,row_sum(weighted))
        state=u.state
    end
    let state=state,out=out
    ntuple(Val(length(out.data))) do e
        @inbounds maxima[e,tid+Int32(1)]=state.maximum.data[e]
        @inbounds sums[e,tid+Int32(1)]=state.sum.data[e]
        @inbounds numerator[e,tid+Int32(1)]=out.data[e]
    end
    end
    nothing
end
if !("--runtime-only" in ARGS)
@testset "Fixed-capacity streaming softmax assembly" begin
    tt=Tuple{CuDeviceMatrix{Float32,1},CuDeviceMatrix{Float32,1},CuDeviceMatrix{Bool,1},Val{4}}
    for arch in (CUDACore.SMVersion(8,0),CUDACore.SMVersion(12,1,:arch))
        code=compile_kernel(streaming_softmax_kernel!,tt;arch,threads=128)
        save_code("softmax-streaming-$arch",code)
        @test !occursin(".local .",entry_body(code.ptx))
        @test !occursin(r"\bcall",entry_body(code.ptx))
    end
end
end
if CUDACore.functional()
@testset "Online state distributed rows" begin
    for kind in (Val(:local),Val(:warp),MMA16x8x16(BFloat16),
                 TiledMMA(MMA16x8x16(BFloat16),Val((2,1)),Val((2,3)),Val(16)))
        n=kind isa Val ? 3 : kind isa MMA16x8x16 ? 4 : 24
        nt=kind isa TiledMMA ? 64 : 32; chunks=7
        rng=MersenneTwister(n+nt); x=randn(rng,Float32,n,nt,chunks)
        v=randn(rng,Float32,n,nt,chunks) .* 10f0
        x[:,:,1].=-Inf32; x[:,:,4].=-Inf32; x[:,:,7].=-Inf32
        x[:,:,3].+=10f0; x[:,:,5].+=13f0
        nr=Tylo._row_count(row_ownership(row_fragment(kind,ntuple(_ -> 0f0,n))))
        # One entire logical row stays empty across all chunks.
        for t in 0:nt-1,e in 0:n-1
            reference_row(kind,n,t,e)==0 && (x[e+1,t+1,:].=-Inf32)
        end
        dx,dv=CuArray(x),CuArray(v)
        dm=CuArray{Float32}(undef,nr,nt); ds=similar(dm); dn=similar(dm)
        @cuda threads=nt online_probe!(dm,ds,dn,dx,dv,kind,Val(n))
        mm,ss,nn=Array(dm),Array(ds),Array(dn)
        for t in 0:nt-1,i in 0:nr-1
            e=kind isa Val ? 0 : 4*(i÷2)+2*(i%2)
            row=reference_row(kind,n,t,e)
            xx=Float64[]; vv=Float64[]
            for c in 1:chunks,t2 in 0:nt-1,e2 in 0:n-1
                if reference_row(kind,n,t2,e2)==row
                    push!(xx,x[e2+1,t2+1,c]);push!(vv,v[e2+1,t2+1,c])
                end
            end
            m=maximum(xx);w=m == -Inf ? zeros(length(xx)) : exp.(xx.-m)
            @test mm[i+1,t+1] == m
            @test ss[i+1,t+1] ≈ sum(w) rtol=5e-6 atol=1e-7
            @test abs(nn[i+1,t+1]-sum(w.*vv)) <= 1e-5*sum(abs,w.*vv)+1e-6
        end
    end
end
@testset "Streaming two-pass masked softmax" begin
    for T in (Float32,BFloat16,Float16),width in (1,31,129,257,4099)
        rows=9;rng=MersenneTwister(width)
        x=T.(randn(rng,Float32,width,rows).*3f0)
        mask=rand(rng,Float32,width,rows).>0.2f0;mask[:,2].=false
        width>128 && (mask[1:128,3].=false)
        x[:,4].+=T(256)
        dx,dm=CuArray(x),CuArray(mask); output=similar(dx)
        @cuda threads=128 blocks=cld(rows,4) streaming_softmax_kernel!(output,dx,dm,Val(4))
        y=Float64.(Array(output)); ref=Float64.(T.(softmax_reference(x,mask)))
        @test y ≈ ref rtol=(T==BFloat16 ? 0.008 : T==Float16 ? 0.001 : 3e-5) atol=3e-7
        @test all(iszero,y[.!mask])
        @test all(isfinite,y)
    end
end
end
