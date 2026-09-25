# TEST_TARGET: cc==9.0
# WGMMA operand forms against the shared/shared K-major product: B transposed
# in shared memory (MN-major, as attention's V), and A from registers (as
# attention's probabilities). Small integers make every product exact.
function wgmma_forms_probe!(out,p::WGMMAAtom{T,N,K},::Val{Form}) where {T,N,K,Form}
    workspace = @inbounds CuDynamicSharedArray(UInt8,40960+1024)   # A, B, Bᵀ: 8, 16, 16 KiB at most
    root = pointer(workspace)
    root += (UInt32(0)-PTX.smem_addr_u32(root)) & UInt32(1023)
    a = shared_tile(TMATile(T,(64,64),2),root)                 # 64×K, K contiguous
    b = shared_tile(TMATile(T,(64,N),1),root+8192)             # K×N, K contiguous
    bt = shared_tile(TMATile(T,(64,N),2),root+8192+64*N*2)     # the same B, N contiguous
    tid = Int32(threadIdx().x)-Int32(1)
    for i in tid:Int32(128):Int32(64*64-1)
        r,k = i÷Int32(64),i%Int32(64)
        @inbounds a[(r,k)] = T(Float32((r+Int32(3)*k)%Int32(7)-Int32(3)))
    end
    for i in tid:Int32(128):Int32(64*N-1)
        k,n = i%Int32(64),i÷Int32(64)
        v = T(Float32((Int32(2)*n+k)%Int32(5)-Int32(2)))
        @inbounds b[(k,n)] = v
        @inbounds bt[(k,n)] = v
    end
    fence_async_shared()
    sync_threads()
    bop = Form === :transposed ? @inbounds(operand(p,OperandB(),bt)) : @inbounds(operand(p,OperandB(),b))
    aop = Form === :registers ? @inbounds(load_fragment(operand_layout(p,OperandA()),window(a,(0,0),(64,K)),tid)) :
                                @inbounds(operand(p,OperandA(),a))
    c = Form === :fresh ? finish_mma(wait_mma(mma_async(p,aop,bop))) :
                          finish_mma(wait_mma(mma_async(p,aop,bop,zero_accumulator(p))))
    @inbounds store!(GlobalTile(pointer(out),Layouts.Layout((64,N),(1,64))),c,tid)
    nothing
end

@testset "WGMMA operand forms assemble" begin
    for T in (BFloat16,Float16), (n,k) in ((64,16),(64,64),(128,64)), form in (:shared,:transposed,:registers,:fresh)
        p = WGMMAAtom(T,n,k)
        code = compile_kernel(wgmma_forms_probe!,Tuple{CuDeviceMatrix{Float32,1},typeof(p),Val{form}};arch=CUDACore.SMVersion(9,0,:arch))
        save_code("wgmma-form-$T-n$n-k$k-$form",code)
        @test isempty(audit(code))
        body = entry_body(code.ptx)
        instruction = filter(l -> occursin("wgmma.mma_async",l),split(body,"\n"))
        @test length(instruction) == k÷16
        tail = form in (:shared,:fresh) ? "1, 1, 1, 0, 0;" : form === :transposed ? "1, 1, 1, 0, 1;" : "1, 1, 1, 0;"
        if form === :fresh                                   # the first ignores its accumulator
            @test endswith(strip(first(instruction)),"0, 1, 1, 0, 0;") && all(l -> endswith(strip(l),tail),instruction[2:end])
        else
            @test all(l -> endswith(strip(l),tail),instruction)
        end
        @test all(l -> occursin(r"\}, \{%r",l),instruction) == (form === :registers)
    end
end

if capability_is(v"9.0")
@testset "WGMMA operand forms agree on Hopper" begin
    for T in (BFloat16,Float16), (n,k) in ((64,16),(64,64),(128,64))
        p = WGMMAAtom(T,n,k)
        results = map((:shared,:transposed,:registers,:fresh)) do form
            out = CuArray{Float32}(undef,64,n)
            @cuda threads=128 shmem=40960+1024 arch=CUDACore.SMVersion(9,0,:arch) wgmma_forms_probe!(out,p,Val(form))
            Array(out)
        end
        ha = [Float64((r+3kk)%7-3) for r in 0:63, kk in 0:k-1]
        hb = [Float64((2nn+kk)%5-2) for kk in 0:k-1, nn in 0:n-1]
        @test results[1] == ha*hb
        @test results[2] == results[1] && results[3] == results[1] && results[4] == results[1]
    end
end
end
