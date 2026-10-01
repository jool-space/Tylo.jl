# TEST_TARGET: cc>=9.0
# A persistent, warp-specialized pipeline over a SharedLayout and Rings: a TMA
# producer thread fills a ring of stages, four consumer warps sum each stage
# and free it, and per item hand their sums to an epilogue warp through a
# two-slot ring of results. Stage counts that are and are not powers of two,
# more items than CTAs, so that every ring turns many times.
@inline pipeline_layout(stages,bytes) = SharedLayout((;
    full = barriers(stages;arrivals=1),                    # TMA bytes
    empty = barriers(stages;arrivals=4),                   # one lane per consumer warp
    done = barriers(2;arrivals=128),                       # every consumer thread's sum
    drained = barriers(2;arrivals=1),                      # the epilogue's lane 0
    sums = scalars(Float32,2*128),
    stage = buffers(bytes,stages;align=1024)))

function pipeline_kernel!(out,binding,items::Int32,blocks::Int32,cfg::Config)
    bytes = transfer_bytes(binding)
    layout = pipeline_layout(cfg.stages,bytes)
    smem = bind(layout,pointer(@inbounds CuDynamicSharedArray(UInt8,shared_bytes(layout))))
    ring = Ring(smem.full,smem.empty)
    results = Ring(smem.done,smem.drained)
    tid = Int32(threadIdx().x)-Int32(1)
    warp, lane = tid ÷ Int32(32), tid % Int32(32)
    if tid == Int32(0)
        initialize!(smem)
        fence_barrier_init()
    end
    sync_threads()
    first, stride = Int32(blockIdx().x)-Int32(1), Int32(gridDim().x)
    if warp == Int32(4)
        if lane == Int32(0)
            c = Int32(0)
            for item in first:stride:items-Int32(1), j in Int32(0):blocks-Int32(1)
                s = ring[c]
                acquire(s)
                arrive!(s.full,bytes)
                @inbounds tma_load!(shared_tile(binding,smem.stage[s.slot]),binding,(Int32(0),(item*blocks+j)*Int32(64)),s.full)
                c += Int32(1)
            end
        end
    elseif warp < Int32(4)
        c = Int32(0); n = Int32(0)
        for item in first:stride:items-Int32(1)
            total = 0f0
            for j in Int32(0):blocks-Int32(1)
                s = ring[c]
                wait(s)
                tile = shared_tile(binding,smem.stage[s.slot])
                for e in tid:Int32(128):Int32(64*64-1)
                    total += Float32(@inbounds tile[(e % Int32(64),e ÷ Int32(64))])
                end
                sync_warp()
                lane == Int32(0) && arrive!(s.empty)
                c += Int32(1)
            end
            u = results[n]
            acquire(u)
            smem.sums[u.slot*Int32(128)+tid] = total
            arrive!(u.full)
            n += Int32(1)
        end
    elseif warp == Int32(5)
        n = Int32(0)
        for item in first:stride:items-Int32(1)
            u = results[n]
            wait(u)
            for r in lane:Int32(32):Int32(127)
                @inbounds out[r+Int32(1),item+Int32(1)] = smem.sums[u.slot*Int32(128)+r]
            end
            sync_warp()
            lane == Int32(0) && arrive!(u.empty)
            n += Int32(1)
        end
    end
    nothing
end

const PIPELINE_TILE = TMATile(BFloat16,(64,64),1)

@testset "Pipeline assembly" begin
    binding = Tylo.TMABinding{typeof(PIPELINE_TILE),PTX.TMADescriptorPtr,2}
    for stages in (3,4)
        tt = Tuple{CuDeviceMatrix{Float32,1},binding,Int32,Int32,typeof(Config(;stages))}
        code = compile_kernel(pipeline_kernel!,tt;arch=CUDACore.SMVersion(12,1,:arch),threads=192)
        save_code("pipeline-s$stages",code)
        @test isempty(audit(code))
        body = entry_body(code.ptx)
        # Every group's barriers initialized, no runtime division for slots.
        @test count("mbarrier.init.shared.b64",body) >= 4
        @test !occursin("div.s32",body) && !occursin("rem.s32",body)
    end
end

if runtime_supported(@__FILE__)
@testset "Pipeline over rings" begin
    items, blocks = 7, 5
    host = BFloat16.(Float32.(reshape(mod.(0:64*64*items*blocks-1,7),64,64*items*blocks)))
    binding = prepare_tma(PIPELINE_TILE,CuArray(host))
    expected = [sum(Float32.(host[:,64*blocks*i+1:64*blocks*(i+1)])) for i in 0:items-1]
    for stages in (1,2,3,4), grid in (1,2,3)
        cfg = Config(;stages)
        shmem = shared_bytes(pipeline_layout(stages,transfer_bytes(PIPELINE_TILE)))
        out = CUDACore.zeros(Float32,128,items)
        GC.@preserve binding begin
            @cuda threads=192 blocks=grid shmem pipeline_kernel!(out,binding,Int32(items),Int32(blocks),cfg)
            synchronize()
        end
        @test vec(sum(Array(out);dims=1)) == expected
    end
end
end
