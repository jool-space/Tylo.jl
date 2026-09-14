# TEST_TARGET: cc>=8.0
if !isdefined(@__MODULE__,:tiled_gemm_kernel!)
    include("../../examples/gemm/kernel.jl")
end

function boundary_copy_kernel!(out,input,dstlayout,srclayout,plan,origin,::Val{Span}) where Span
    T=eltype(input); t=Int32(threadIdx().x)-Int32(1)
    smem=@inbounds CuDynamicSharedArray(T,Span)
    for i in t+Int32(1):Int32(blockDim().x):Int32(Span)
        @inbounds smem[i]=T(99)
    end
    sync_threads()
    dst=SharedTile(pointer(smem),dstlayout)
    src=GlobalTile(pointer(input),srclayout)
    @inbounds copy_async!(plan,dst,src,origin,t)
    commit_copies();wait_copies(Val(0));sync_threads()
    for i in t+Int32(1):Int32(blockDim().x):Int32(Span)
        @inbounds out[i]=smem[i]
    end
    nothing
end

function check_boundary_copy(T,axis,step,origin)
    s=(19,23);ld=(s[axis]+3)*step
    strides=axis==1 ? (step,ld) : (ld,step)
    src=@Layout $s $strides
    host=fill(T(NaN32),Int(cosize(src)))
    logical=T===Float32 ? reshape(Float32.(1:prod(s)),s) :
        reshape(reinterpret(T,UInt16.(0x3800:0x3800+prod(s)-1)),s)
    for r in 0:s[1]-1,c in 0:s[2]-1
        host[r*strides[1]+c*strides[2]+1]=logical[r+1,c+1]
    end
    parent=axis==1 ? @Layout((64, 32), (1, 64)) :
                     @Layout((32, 64), (64, 1))
    start=axis==1 ? (8,3) : (3,8)
    dst=Tylo.Layouts.window(compose(Swizzle{3,3,3}(),parent),map(Int32,start),Val((16,16)))
    p=CopyPlan{(16,16),32,axis}()
    # Validate destination contiguity independently from the bounded source.
    validate_copy(p,T,dst,dst)
    span=Int(cosize(parent));expected=fill(T(99),span)
    for r in 0:15,c in 0:15
        i,j=r+origin[1],c+origin[2]
        x=0<=i<s[1] && 0<=j<s[2] ? logical[i+1,j+1] : zero(T)
        raw=axis==1 ? (r+8)+64*(c+3) : 64*(r+3)+(c+8)
        offset=xor(raw,(raw>>3)&56)
        expected[offset+1]=x
    end
    input=CuArray(host);out=CuArray{T}(undef,span)
    @cuda threads=32 shmem=span*sizeof(T) boundary_copy_kernel!(out,input,dst,src,p,map(Int32,origin),Val(span))
    @test Array(out) == expected
end

function check_ragged_gemm(T,cfg,m,n,k;pad=0,outputtype=Float32)
    @assert value(cfg.bounds)
    bm,bn,bk=size(cfg.plan);lda,ldb,ldc=k+pad,k+2pad,m+5
    rng=MersenneTwister(13m+7n+k)
    a=T.(0.2f0 .* randn(rng,Float32,m,k));b=T.(0.2f0 .* randn(rng,Float32,k,n))
    ha,hb=fill(T(NaN32),lda,m),fill(T(NaN32),ldb,n)
    ha[1:k,:] .= permutedims(a);hb[1:k,:] .= b
    sentinel=outputtype(-1234)
    hd=fill(sentinel,ldc,n+2);hd[1:m,1:n] .= outputtype(NaN32)
    da,db,dd=CuArray(ha),CuArray(hb),CuArray(hd)
    @cuda threads=Tylo.threads(cfg.plan) blocks=(cld(m,bm),cld(n,bn)) shmem=shared_bytes(cfg,T) tiled_gemm_kernel!(
        dd,da,db,Int32(m),Int32(n),Int32(k),Int32(lda),Int32(ldb),Int32(ldc),cfg,-0.75f0,Val(true))
    actual=Array(dd);expected=max.(-0.75 .* (Float64.(a)*Float64.(b)),0.0)
    tolerance=outputtype===Float32 ? 2e-4 : 0.01
    @test all(abs.(Float64.(actual[1:m,1:n]) .- expected) .<= tolerance .+ tolerance .* abs.(expected))
    @test all(==(sentinel),actual[m+1:end,:]) && all(==(sentinel),actual[:,n+1:end])
end

begin # assembly checks
@testset "Bounded copy and GEMM assembly" begin
    cfg=gemm_config(BFloat16;bounds=true)
    tt=Tuple{CuDeviceVector{Float32,1},CuDeviceVector{BFloat16,1},CuDeviceVector{BFloat16,1},
             Int32,Int32,Int32,Int32,Int32,Int32,typeof(cfg),Float32,Val{true}}
    for arch in (CUDACore.SMVersion(8,0),CUDACore.SMVersion(12,1,:arch))
        code=compile_kernel(tiled_gemm_kernel!,tt;arch,threads=128)
        save_code("ragged-gemm-$arch",code);body=entry_body(code.ptx)
        @test !occursin(".local .",body)
        @test !occursin(r"\bcall",body)
        @test occursin("cp.async.cg.shared.global",body)
        @test occursin("mma.sync.aligned.m16n8k16",body)
    end
end
end
if runtime_supported(@__FILE__)
@testset "Predication and nonzero swizzle origins" begin
    for T in (BFloat16,Float16,Float32),axis in (1,2),step in (1,2),origin in ((0,0),(-3,5),(7,-1),(5,13),(40,40))
        check_boundary_copy(T,axis,step,origin)
    end
end
@testset "Ragged GEMM, zero fill and masked stores" begin
    configs=((false,16,1),(true,32,2),(true,64,1),(false,32,2))
    shapes=((7,3,1,0),(65,64,32,0),(64,67,32,0),(64,64,33,0),(65,97,73,3),(129,67,227,8))
    for T in (BFloat16,Float16),(swizzled,bk,stages) in configs
        cfg=gemm_config(T;block=(64,64,bk),swizzled,stages,bounds=true)
        for (m,n,k,pad) in shapes
            check_ragged_gemm(T,cfg,m,n,k;pad)
        end
    end
    for T in (BFloat16,Float16)
        cfg=gemm_config(T;block=(16,8,16),warps=(1,1),bounds=true)
        check_ragged_gemm(T,cfg,17,9,49;pad=1,outputtype=T)
    end
end
end
