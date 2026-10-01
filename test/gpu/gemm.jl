# TEST_TARGET: cc>=8.0
include("../../examples/gemm/kernel.jl")

function gemm_signature(config,T)
    Tuple{CuDeviceVector{Float32,1},CuDeviceVector{T,1},CuDeviceVector{T,1},
          Int32,Int32,Int32,Int32,Int32,Int32,typeof(config),Float32}
end

begin # assembly checks
@testset "Complete GEMM assembly" begin
    for T in (BFloat16,Float16),swizzled in (false,true)
        cfg = gemm_config(T;swizzled)
        for arch in (CUDACore.SMVersion(8,0),CUDACore.SMVersion(9,0,:arch),
                     CUDACore.SMVersion(10,0,:arch),CUDACore.SMVersion(12,1,:arch))
            code = compile_kernel(tiled_gemm_kernel!,gemm_signature(cfg,T);arch)
            save_code("gemm-$(T)-$(swizzled)-$(arch)",code)
            body = entry_body(code.ptx)
            @test !isempty(code.image)
            @test isempty(audit(code))
            @test occursin("cp.async.cg.shared.global",body)
            @test occursin("cp.async.commit_group",body)
            @test occursin("cp.async.wait_group 1",body)
            @test occursin("cp.async.wait_group 0",body)
            @test occursin("ldmatrix.sync.aligned.m8n8.x4",body)
            @test occursin("ldmatrix.sync.aligned.m8n8.x2",body)
            @test occursin("mma.sync.aligned.m16n8k16",body)
        end
    end
end

end

# Separate load test: decode raw register words and compare against logical
# matrix coordinates computed independently of Tylo's ownership mapping.
function operand_load_kernel!(outa,outb,ina,inb,config)
    T = eltype(ina)
    tid = Int32(threadIdx().x)-Int32(1)
    smem = @inbounds CuDynamicSharedArray(T,2048)
    sa = SharedTile(pointer(smem),config.sa,16)
    sb = SharedTile(pointer(smem)+2048,config.sb,16)
    a = GlobalTile(pointer(ina),@Layout((32, 32), (32, 1)),16)
    b = GlobalTile(pointer(inb),@Layout((32, 32), (1, 32)),16)
    @inbounds begin
        copy_async!(distribute((32,32),32;vector=8,axis=2),sa,a,tid)
        copy_async!(distribute((32,32),32;vector=8,axis=1),sb,b,tid)
    end
    commit_copies()
    wait_copies(0)
    sync_threads()
    atom = MMAAtom((16,8,16),T)
    av = @inbounds load_fragment(atom,OperandA(),window(sa,(Int32(3),Int32(8)),(16,16)),tid)
    bv = @inbounds load_fragment(atom,OperandB(),window(sb,(Int32(8),Int32(3)),(16,8)),tid)
    wa, wb = Tylo._pack_words(av), Tylo._pack_words(bv)
    PTX.Utils.@unroll for j in 1:4
        @inbounds outa[4tid+j] = wa[j]
    end
    PTX.Utils.@unroll for j in 1:2
        @inbounds outb[2tid+j] = wb[j]
    end
    nothing
end

function check_operand_load(T,swizzled)
    la = @Layout((32, 32), (32, 1))
    lb = @Layout((32, 32), (1, 32))
    config = (sa=swizzled ? compose(Swizzle{2,3,2}(),la) : la,
              sb=swizzled ? compose(Swizzle{2,3,2}(),lb) : lb)
    # All entries distinguishable as BF16 too; no arithmetic involved.
    bits = reshape(UInt16.(0x3800:0x3bff),32,32)
    a = collect(reinterpret(T,bits))
    b = permutedims(a)
    da,db = CuArray(permutedims(a)),CuArray(b)
    oa,ob = CuArray{UInt32}(undef,128),CuArray{UInt32}(undef,64)
    @cuda threads=32 shmem=4096 operand_load_kernel!(oa,ob,da,db,config)
    va = reshape(collect(reinterpret(UInt16,Array(oa))),8,32)
    vb = reshape(collect(reinterpret(UInt16,Array(ob))),4,32)
    expected_a = [reinterpret(UInt16,a[3+t÷4+8*((e÷2)%2)+1,8+2*(t%4)+(e%2)+8*(e÷4)+1])
                  for e in 0:7,t in 0:31]
    expected_b = [reinterpret(UInt16,b[8+2*(t%4)+(e%2)+8*(e÷2)+1,3+t÷4+1])
                  for e in 0:3,t in 0:31]
    @test va == expected_a
    @test vb == expected_b
end

function check_gemm(T,config,m,n,k;pad=0,relu=false)
    bm,bn,bk = size(config.plan)
    m % bm == n % bn == k % bk == 0 || throw(ArgumentError("full tiles required"))
    lda,ldb,ldc = k+pad,k+2pad,m+5
    rng = MersenneTwister(m*17+n*5+k)
    a = T.(0.2f0 .* randn(rng,Float32,m,k))
    b = T.(0.2f0 .* randn(rng,Float32,k,n))
    ha,hb = fill(T(NaN32),lda,m),fill(T(NaN32),ldb,n)
    ha[1:k,:] .= permutedims(a)
    hb[1:k,:] .= b
    hd = fill(-12345f0,ldc,n)
    hd[1:m,:] .= NaN32
    da,db,dd = CuArray(ha),CuArray(hb),CuArray(hd)
    # Full-tile launches promise 16-byte rows.
    @assert lda % 8 == ldb % 8 == 0
    alpha = relu ? -0.75f0 : 1f0
    config = merge(config,(;relu))
    @cuda threads=Tylo.threads(config.plan) blocks=(m÷bm,n÷bn) shmem=shared_bytes(gemm_shared(config,T)) tiled_gemm_kernel!(
        dd,da,db,Int32(m),Int32(n),Int32(k),Int32(lda),Int32(ldb),Int32(ldc),config,alpha)
    actual = Array(dd)
    expected = Float64(alpha) .* (Float64.(a)*Float64.(b))
    relu && (expected = max.(expected,0.0))
    @test all(abs.(actual[1:m,:] .- expected) .<= 2e-4 .+ 2e-4 .* abs.(expected))
    @test all(==(-12345f0),actual[m+1:end,:])
end

if runtime_supported(@__FILE__)
    @testset "Shared matrix loads preserve coordinates" begin
        for T in (BFloat16,Float16),swizzled in (false,true)
            check_operand_load(T,swizzled)
        end
    end
    @testset "Tiled GEMM runtime, layouts and buffer reuse" begin
        configs = ((false,16,1,(2,2)),(true,16,2,(2,2)),(true,32,2,(2,2)),
                   (false,32,2,(4,1)),(true,64,2,(2,2)))
        for T in (BFloat16,Float16),(swizzled,bk,stages,warps) in configs
            cfg = gemm_config(T;block=(64,64,bk),swizzled,stages,warps)
            # One iteration, pipeline fill/drain, repeated reuse and padded
            # runtime leading dimensions. Rectangular grids expose role bugs.
            check_gemm(T,cfg,64,64,bk)
            check_gemm(T,cfg,64,128,2bk;pad=8)
            check_gemm(T,cfg,128,64,3bk;pad=8)
            check_gemm(T,cfg,128,128,7bk;pad=8,relu=true)
        end
        # More threads than vectors: some producers commit empty groups.
        cfg = gemm_config(BFloat16;block=(16,8,16),warps=(1,1),stages=2)
        check_gemm(BFloat16,cfg,32,24,80;pad=8,relu=true)
    end
else
    @test_skip false # warp MMA runtime requires CC >= 8.0
end
