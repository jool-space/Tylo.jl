# TEST_TARGET: cc>=9.0
# mbarriers: arrivals counted per phase, and cp.async copies completing on a
# barrier instead of through wait_copies.
const BAR_COPIES = distribute((32,64),128;vector=8,axis=2)

function barrier_phases!(out,rounds::Int32)
    smem = @inbounds CuDynamicSharedArray(UInt64,2)
    bar = Barrier(pointer(smem))
    counter = reinterpret(Core.LLVMPtr{UInt32,3},pointer(smem)+8)
    tid = Int32(threadIdx().x)-Int32(1)
    if tid == 0
        initialize!(bar,128)           # every thread arrives each phase
        unsafe_store!(counter,UInt32(0))
        fence_barrier_init()
    end
    sync_threads()
    for phase in Int32(0):rounds-Int32(1)
        tid == Int32(5) && unsafe_store!(counter,unsafe_load(counter)+UInt32(1))
        arrive!(bar)
        wait(bar,phase)
        # Every thread sees the write made before the phase completed.
        @inbounds out[phase*Int32(128)+tid+Int32(1)] = unsafe_load(counter)
        sync_threads()
    end
    tid == 0 && invalidate!(bar)
    nothing
end

function barrier_copies!(out,src)
    smem = @inbounds CuDynamicSharedArray(BFloat16,32*64+8)
    bar = Barrier(pointer(smem))
    tile = @inbounds SharedTile(pointer(smem)+16,@Layout((32,64),(64,1)),16)
    tid = Int32(threadIdx().x)-Int32(1)
    if tid == 0
        initialize!(bar,128)           # each thread's copies are one arrival
        fence_barrier_init()
    end
    sync_threads()
    @inbounds copy_async!(BAR_COPIES,tile,GlobalTile(pointer(src),@Layout((32,64),(64,1)),16),tid)
    commit_copies(bar)
    wait(bar,0)
    for i in tid:Int32(128):Int32(32*64-1)
        r,c = i ÷ Int32(64),i % Int32(64)
        @inbounds out[i+Int32(1)] = unsafe_load(pointer(tile,(r,c)))
    end
    nothing
end

@testset "Barrier assembly" begin
    body = entry_body(compile_kernel(barrier_copies!,Tuple{CuDeviceVector{BFloat16,1},CuDeviceVector{BFloat16,1}};
                                     arch=CUDACore.SMVersion(12,1,:arch)).ptx)
    @test occursin("mbarrier.init.shared.b64",body)
    @test occursin("fence.mbarrier_init.release.cluster",body)
    @test occursin("cp.async.mbarrier.arrive.noinc.shared.b64",body)
    @test occursin("mbarrier.try_wait.parity.shared.b64",body)
    @test count("cp.async.cg.shared.global",body) == 2
    @test !occursin(r"\bcall",body)
end

if runtime_supported(@__FILE__)
    @testset "Barrier phases and copies" begin
        out = CUDACore.zeros(UInt32,128*5)
        @cuda threads=128 shmem=16 barrier_phases!(out,Int32(5))
        @test reshape(Array(out),128,5) == repeat(UInt32.(1:5)',128)
        src = BFloat16.(reshape(Float32.(1:2048),64,32))    # row-major 32×64 as a 64×32 array
        out = CuArray{BFloat16}(undef,2048)
        @cuda threads=128 shmem=2*(32*64+8) barrier_copies!(out,CuArray(src))
        @test Array(out) == vec(src)
    end
end
