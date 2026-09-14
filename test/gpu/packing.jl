# TEST_TARGET: cc>=8.0
# Both the conversion path and the bit-preserving path run on ordinary NVIDIA
# GPUs. Typed destinations deliberately retain BF16/FP16 as their element type.
function packing_kernel!(out,back,input,::Type{T},::Val{W}) where {T,W}
    t = Int32(threadIdx().x)-Int32(1)
    f = Fragment(ntuple(i -> @inbounds(input[2W*t+i]),Val(2W)),Tylo.Layouts.LocalOwnership{2W,2}())
    packed = pack(T,f)
    # Every thread receives a 16-byte aligned destination, including short tails.
    store!(pointer(out)+Int(t)*cld(4W,16)*16,packed)
    decoded = unpack(packed)
    Tylo.@rtuple(1:2W) do i
        @inbounds back[2W*t+i] = Float32(decoded.data[i])
    end
    nothing
end
function packed_bits_kernel!(out,input,::Type{T}) where T
    t = Int32(threadIdx().x)-Int32(1)
    f = Fragment(ntuple(i -> @inbounds(input[8t+i]),Val(8)),Tylo.Layouts.LocalOwnership{8,2}())
    g = unpack(pack(f))
    Tylo.@rtuple(1:8) do i
        @inbounds out[8t+i] = g.data[i]
    end
    nothing
end
@testset "BF16 and FP16 register packing" begin
    for T in (BFloat16,Float16), w in (1,2,3,4,16)
        begin # assembly checks
            code = compile_kernel(packing_kernel!,Tuple{CuDeviceVector{T,1},CuDeviceVector{Float32,1},CuDeviceVector{Float32,1},Type{T},Val{w}};
                                  arch=CUDACore.SMVersion(12,1,:arch),threads=32)
            save_code("packing-$(T)-w$w",code)
            body = entry_body(code.ptx)
            @test occursin(T === BFloat16 ? "cvt.rn.bf16x2.f32" : "cvt.rn.f16x2.f32",body)
            @test !occursin(".local .",body)
            @test !occursin(r"\bcall",body)
        end
        if runtime_supported(@__FILE__)
            values = randn(MersenneTwister(47),Float32,32*2w)
            values[1:8] .= Float32[0,-0.0,Inf,-Inf,1.00390625,1.01171875,1.00048828125,1f-40]
            input = CuArray(values)
            stride = cld(4w,16)*8
            output = CuArray(fill(T(-7),stride*32))
            back = similar(input)
            @cuda threads=32 packing_kernel!(output,back,input,T,Val(w))
            expected = T.(values)
            actual = reshape(Array(output),stride,32)
            @test reinterpret.(UInt16,vec(actual[1:2w,:])) == reinterpret.(UInt16,expected)
            @test all(==(T(-7)),actual[2w+1:end,:])
            @test isequal(Array(back),Float32.(expected))
        end
    end
    if runtime_supported(@__FILE__)
        for T in (BFloat16,Float16)
            bits = UInt16[0,0x8000,0x7fff,0xffff,0x7f81,0x7c01,1,0x3c00]
            values = repeat(reinterpret.(T,bits),32)
            input = CuArray(values); out = similar(input)
            @cuda threads=32 packed_bits_kernel!(out,input,T)
            @test reinterpret.(UInt16,Array(out)) == reinterpret.(UInt16,values)
        end
    end
end

# A complete allocation/load/store/readback/deallocation path for every exposed
# .32x32b width and logical element type. All addresses are warp-uniform.
function typed_tmem_kernel!(out,input,::Val{W},::Val{A}) where {W,A}
    T = eltype(input)
    tid = UInt32(threadIdx().x)-UInt32(1)
    warp = tid >> UInt32(5)
    slot = @inbounds CuStaticSharedArray(UInt32,1)
    if warp == UInt32(0)
        ptx"tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32"(PTX.smem_addr_u32(pointer(slot)),UInt32(128))
    end
    sync_threads()
    base = @inbounds slot[1]
    n = W*(4÷sizeof(T))
    shape = A == 2 ? (32,n) : (n,32)
    tile = TmemTile(T,base,@Layout((128,n),(1,128)))
    band = @inbounds window(tile,(UInt32(32)*warp,UInt32(0)),Val((32,n)))
    band = A == 2 ? band : permutedims(band)
    plan = TmemTransfer{shape,A}()
    access = @inbounds partition(plan,band)
    f = Fragment(ntuple(i -> @inbounds(input[Int(tid)*n+i]),Val(n)),plan)
    store_async!(access,T === Float32 ? f : pack(f))
    wait_stores()
    pending = load_async(access)
    ready = wait_load(pending)
    values = T === Float32 ? ready : unpack(ready)
    Tylo.@rtuple(1:n) do i
        @inbounds out[Int(tid)*n+i] = values.data[i]
    end
    sync_threads()
    if warp == UInt32(0)
        ptx"tcgen05.dealloc.cta_group::1.sync.aligned.b32"(base,UInt32(128))
        ptx"tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned"()
    end
    nothing
end
@testset "Typed TMEM transfers, x1 through x128" begin
    for T in (Float32,BFloat16,Float16), w in (1,2,4,8,16,32,64,128), axis in (1,2)
        begin # assembly checks
            code = compile_kernel(typed_tmem_kernel!,Tuple{CuDeviceVector{T,1},CuDeviceVector{T,1},Val{w},Val{axis}})
            save_code("typed-tmem-$(T)-w$w-a$axis",code)
            body = entry_body(code.ptx)
            @test occursin("tcgen05.ld.sync.aligned.32x32b.x$w.b32",body)
            @test occursin("tcgen05.st.sync.aligned.32x32b.x$w.b32",body)
            @test occursin("tcgen05.wait::ld.sync.aligned",body)
            @test !occursin(r"\bcall",body)
            @test !occursin(".local .",body)
        end
        if capability_major(10)
            values = T.(randn(MersenneTwister(623),Float32,128*w*(4÷sizeof(T))))
            input = CuArray(values); out = similar(input)
            @cuda threads=128 typed_tmem_kernel!(out,input,Val(w),Val(axis))
            @test Array(out) == values
        end
    end
end
