# Hardware-only paired correctness and optional event-timed measurements.
function attention_case(B,H,S; input_scale=0.5f0, blocks=nothing, splitp=true)
    mod = ReferenceAttention
    rows = B*H*S
    rng = MersenneTwister(B*7919+H*131+S)
    Q,K = (randn(rng,Float32,rows,128).*input_scale for _ in 1:2)
    V = randn(rng,Float32,rows,128)
    device_inputs = map(x -> CuArray(mod.fab_pack(x)),(Q,K,V))
    maps = map(device_inputs) do x
        desc = PTX.tensor_map_tile_2d(:bf16,pointer(x),rows,128,128,64;swizzle=:B128)
        CuArray(collect(desc.data))
    end
    # Descriptor address conversion happens on the host; retain maps while
    # either compiled kernel or a benchmark execution can still use them.
    GC.@preserve maps device_inputs begin
        descriptors = map(x -> reinterpret(PTX.TMADescriptorPtr,UInt(pointer(x))),maps)
        output = CuArray{UInt16}(undef,rows*128)
        debug = CUDACore.zeros(UInt32,512)
        mblocks = S÷256
        total_work = mblocks*B*H
        sm_count = CUDACore.attribute(CUDACore.device(),
                                     CUDACore.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
        grid = isnothing(blocks) ? min(total_work,sm_count) : blocks
        cfg = (;mod.FAB_CFG_DEFAULT...,splitp)
        args = (output,descriptors...,UInt32(S),UInt32(trailing_zeros(mblocks)),
                UInt32(mblocks-1),UInt32(S÷128),UInt32(total_work),
                inv(sqrt(128f0))*Float32(mod.FAB_LOG2E),debug,Val(cfg))
        kernels = map((ReferenceAttention,TiledAttention)) do m
            kernel = @cuda launch=false minthreads=512 m.fab_kernel!(args...)
            CUDACore.attributes(kernel.fun)[CUDACore.FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES] =
                mod.FAB_SMEM_BYTES
            kernel
        end
        launch = k -> k(args...;blocks=grid,threads=512,shmem=mod.FAB_SMEM_BYTES)
        results = map(kernels) do k
            fill!(output,UInt16(0x7fc0)) # NaN sentinel exposes missing output writes
            launch(k)
            CUDACore.synchronize()
            copy(Array(output))
        end
        @test results[1] == results[2]
        @test all(iszero,Array(debug))
        actual = reshape(results[2],128,rows)
        for head in 1:B*H
            r = (head-1)*S+1:head*S
            reference = mod.fab_cpu_ref(Q[r,:],K[r,:],V[r,:],inv(sqrt(128f0)))
            decoded = permutedims(mod.bf16_to_f32.(actual[:,r]))
            @test maximum(abs.(decoded-reference)) < 5f-2
        end
        if "--bench" in ARGS
            # Each graph contains 32 launches; divide event time by 32. Pair order
            # alternates to reduce drift. Allocation/compilation/reference are out.
            graphs = map(kernels) do kernel
                CUDACore.instantiate(CUDACore.capture() do
                    for _ in 1:32
                        launch(kernel)
                    end
                end)
            end
            timings = (Float64[],Float64[])
            for round in 1:23
                for i in (isodd(round) ? (1,2) : (2,1))
                    elapsed = CUDACore.@elapsed CUDACore.launch(graphs[i])
                    round > 2 && push!(timings[i],elapsed*1e6/32)
                end
            end
            medians = map(x -> sort(x)[11],timings)
            println((;B,H,S,input_scale,grid,splitp,reference_us=medians[1],
                     tylo_us=medians[2],ratio=medians[2]/medians[1]))
        end
    end
    nothing
end
function run_attention_cases()
@testset "Paired FlashAttention execution" begin
    for splitp in (true,false)
        attention_case(1,1,256;splitp)
        attention_case(1,2,512;splitp,input_scale=2f0)
        attention_case(2,4,1024;splitp)
        # Force repeated work items through each CTA's TMEM/barrier allocation.
        attention_case(1,2,512;splitp,input_scale=2f0,blocks=1)
    end
end

end
