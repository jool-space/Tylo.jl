# TEST_TARGET: cc>=8.0
if !isdefined(@__MODULE__,:tiled_gemm_kernel!)
    include("../../examples/gemm/kernel.jl")
end

function boundary_copy_kernel!(out,input,dstlayout,srclayout,ownership,origin,::Val{Span},::Val{Align}) where {Span,Align}
    T=eltype(input); t=Int32(threadIdx().x)-Int32(1)
    smem=@inbounds CuDynamicSharedArray(T,Span)
    for i in t+Int32(1):Int32(blockDim().x):Int32(Span)
        @inbounds smem[i]=T(99)
    end
    sync_threads()
    dst=SharedTile(pointer(smem),dstlayout,16)
    src=bound(GlobalTile(pointer(input),srclayout,Align),size(srclayout))
    copy_async!(ownership,dst,window(src,origin,(16,16)),t)
    commit_copies();wait_copies(0);sync_threads()
    for i in t+Int32(1):Int32(blockDim().x):Int32(Span)
        @inbounds out[i]=smem[i]
    end
    nothing
end

# A 19×23 source with a runtime leading dimension `ld`: contiguous along
# `axis` with a static unit stride (cp.async vectors, as wide as `ld`
# allows, cut where the bound falls), or every `step`-th element (scalars).
# The destination is a window of a swizzled tile: vectors still derive.
@testset "Copies into a swizzled window assembly" begin
    parent=@Layout((64, 32), (1, 64))
    dst=Tylo.Layouts.window(compose(Swizzle{3,3,3}(),parent),(Int32(8),Int32(3)),(16,16))
    src=@Layout(($(Int32(19)),$(Int32(23))),(1,$(Int32(32))))
    tt=Tuple{CuDeviceVector{BFloat16,1},CuDeviceVector{BFloat16,1},typeof(dst),typeof(src),
             typeof(distribute((16,16),32;vector=8,axis=1)),Tuple{Int32,Int32},Val{2048},Val{16}}
    body=entry_body(compile_kernel(boundary_copy_kernel!,tt;arch=CUDACore.SMVersion(12,1,:arch),threads=32).ptx)
    @test count("cp.async.cg.shared.global",body) == 1
end

function check_boundary_copy(T,axis,step,ld,origin)
    s=(19,23)
    strides=axis==1 ? (step,ld) : (ld,step)
    src=step==1 ? (axis==1 ? @Layout(($(s[1]),$(s[2])),(1,$ld)) : @Layout(($(s[1]),$(s[2])),($ld,1))) : @Layout $s $strides
    align=step==1 ? min(16,1 << trailing_zeros(ld*sizeof(T))) : sizeof(T)
    host=fill(T(NaN32),Int(cosize(Layout(s,strides))))
    logical=T===Float32 ? reshape(Float32.(1:prod(s)),s) :
        reshape(reinterpret(T,UInt16.(0x3800:0x3800+prod(s)-1)),s)
    for r in 0:s[1]-1,c in 0:s[2]-1
        host[r*strides[1]+c*strides[2]+1]=logical[r+1,c+1]
    end
    parent=axis==1 ? @Layout((64, 32), (1, 64)) :
                     @Layout((32, 64), (64, 1))
    start=axis==1 ? (8,3) : (3,8)
    dst=Tylo.Layouts.window(compose(Swizzle{3,3,3}(),parent),map(Int32,start),(16,16))
    o=distribute((16,16),32;vector=16÷sizeof(T),axis)
    span=Int(cosize(parent));expected=fill(T(99),span)
    for r in 0:15,c in 0:15
        i,j=r+origin[1],c+origin[2]
        x=i<s[1] && j<s[2] ? logical[i+1,j+1] : zero(T)
        raw=axis==1 ? (r+8)+64*(c+3) : 64*(r+3)+(c+8)
        offset=xor(raw,(raw>>3)&56)
        expected[offset+1]=x
    end
    input=CuArray(host);out=CuArray{T}(undef,span)
    @cuda threads=32 shmem=span*sizeof(T) boundary_copy_kernel!(out,input,dst,src,o,map(Int32,origin),Val(span),Val(align))
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
        @test isempty(audit(code))
        @test occursin("cp.async.cg.shared.global",body)
        @test occursin("mma.sync.aligned.m16n8k16",body)
    end
end
end
if runtime_supported(@__FILE__)
@testset "Bounded copies and nonzero swizzle origins" begin
    # Origins along the contiguous axis keep vectors aligned; the bounds at
    # 19 and 23 fall inside vectors, which copy a prefix and zero the rest.
    for T in (BFloat16,Float16,Float32),axis in (1,2),(step,ld) in ((1,32),(1,26),(1,25),(2,50)),
            origin in (axis == 1 ? ((0,0),(8,13),(16,1),(40,40)) : ((0,0),(13,8),(1,16),(40,40)))
        check_boundary_copy(T,axis,step,ld,origin)
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
