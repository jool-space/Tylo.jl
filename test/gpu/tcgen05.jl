# TEST_TARGET: cc==10|cc==11
using Tylo.Layouts: @Layout, Swizzle, compose
using PTX.MBarriers: barrier_init, barrier_wait

# One CTA of four warps computes a 128×128 product over K=64 with one atom:
# shared operands filled through their canonical layouts, four instructions
# issued by one thread, completion observed on an mbarrier, and the TMEM
# accumulator read back through transfer partitions. Variants store B
# K-major (`:k`) or MN-major (`:mn`).
b_layout(::Val{:k}) = compose(Swizzle{3,3,3}(),@Layout((64,128),(1,64)))
b_layout(::Val{:mn}) = compose(Swizzle{3,3,3}(),@Layout((64,(64,2)),(64,(1,4096))))
function tcgen05_gemm_kernel!(out,a_data,b_data,atom,variant::Val)
    tid = Int32(threadIdx().x)-Int32(1)
    warp,lane = tid >> 5,tid & Int32(31)
    smem = @inbounds CuDynamicSharedArray(UInt8,32768)
    slot = @inbounds CuStaticSharedArray(UInt32,1)
    mbar = @inbounds CuStaticSharedArray(UInt64,1)
    sa = SharedTile(reinterpret(Core.LLVMPtr{BFloat16,3},pointer(smem)),compose(Swizzle{3,3,3}(),@Layout((128,64),(64,1))))
    sb = SharedTile(reinterpret(Core.LLVMPtr{BFloat16,3},pointer(smem)+16384),b_layout(variant))
    a = GlobalTile(pointer(a_data),@Layout((128,64),(1,128)))
    b = GlobalTile(pointer(b_data),@Layout((64,128),(1,64)))
    for i in tid:Int32(128):Int32(128*64-1)
        r,c = i % Int32(128),i ÷ Int32(128)
        unsafe_store!(pointer(sa,(r,c)),unsafe_load(pointer(a,(r,c))))
        unsafe_store!(pointer(sb,(c,r)),unsafe_load(pointer(b,(c,r))))
    end
    if tid == Int32(0)
        barrier_init(pointer(mbar),1)
    end
    ptx"fence.proxy.async"()
    warp == Int32(0) && tmem_allocate!(pointer(slot),Val(128))
    sync_threads()
    base = @inbounds slot[1]
    d = accumulator(atom,base)
    if tid == Int32(0)
        oa = @inbounds tcgen05_operand(atom,OperandA(),sa)
        ob = @inbounds tcgen05_operand(atom,OperandB(),sb)
        @inbounds mma(atom,d,oa,ob,Val(64),false)
        commit_mma(pointer(mbar))
    end
    barrier_wait(pointer(mbar),UInt32(0))
    fence_after_thread_sync()
    band = @inbounds partition(TmemTransfer{(32,128),2}(),window(d,(Int32(32)*warp,Int32(0)),Val((32,128))))
    values = wait_load(load_async(band))
    c = GlobalTile(pointer(out),@Layout((128,128),(1,128)))
    @inbounds store!(window(c,(Int32(32)*warp,Int32(0)),Val((32,128))),values,lane)
    sync_threads()
    if warp == Int32(0)
        tmem_deallocate!(base,Val(128))
        tmem_relinquish_permit()
    end
    nothing
end

@testset "tcgen05 GEMM assembly" begin
    atom = Tcgen05MMA((128,128,16),BFloat16)
    for variant in (:k,:mn)
        tt = Tuple{CuDeviceMatrix{Float32,1},CuDeviceMatrix{BFloat16,1},CuDeviceMatrix{BFloat16,1},typeof(atom),Val{variant}}
        code = compile_kernel(tcgen05_gemm_kernel!,tt;arch=CUDACore.SMVersion(10,0,:arch),threads=128)
        save_code("tcgen05-gemm-$variant",code)
        body = entry_body(code.ptx)
        @test !occursin(".local .",body)
        @test !occursin(r"\bcall",body)
        @test count("tcgen05.mma.cta_group::1.kind::f16",body) == 4
        @test count("tcgen05.commit",body) == 1
        @test count("tcgen05.alloc",body) == 1 && count("tcgen05.dealloc",body) == 1
        @test count("tcgen05.ld.sync.aligned.32x32b.x128",body) == 1
    end
end

if runtime_supported(@__FILE__)
    @testset "tcgen05 GEMM execution" begin
        atom = Tcgen05MMA((128,128,16),BFloat16)
        rng = MersenneTwister(2026)
        a = BFloat16.(rand(rng,-4:4,128,64) ./ 4); b = BFloat16.(rand(rng,-4:4,64,128) ./ 4)
        expected = Float32.(Float64.(a)*Float64.(b))
        for variant in (:k,:mn)
            out = CUDACore.zeros(Float32,128,128)
            @cuda threads=128 shmem=32768 tcgen05_gemm_kernel!(out,CuArray(a),CuArray(b),atom,Val(variant))
            @test Array(out) == expected
        end
    end
end
