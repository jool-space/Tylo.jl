using Tylo.Layouts: @Layout

function conversion_probe!(out,input,atom)
    lane=Int32(threadIdx().x)-Int32(1)
    left=Tylo.MMAFragment(Float32,Accumulator(),ntuple(i -> @inbounds(input[i,lane+1]),Val(4)))
    right=Tylo.MMAFragment(Float32,Accumulator(),ntuple(i -> @inbounds(input[i+4,lane+1]),Val(4)))
    a=pack_operand_a(atom,left,right)
    ntuple(i -> (@inbounds out[i,lane+1]=a.data[i]),Val(4))
    nothing
end
function conversion_shared_probe!(out,input,atom::MMA16x8x16{T}) where T
    lane=Int32(threadIdx().x)-Int32(1)
    smem=CuStaticSharedArray(T,256)
    # Independent source coordinate oracle; scalar stores establish the
    # existing shared-load route without invoking the conversion under test.
    ntuple(Val(8)) do i
        e=i-1
        r=lane÷Int32(4)+Int32(8*((e%4)÷2))
        c=Int32(2)*(lane%Int32(4))+Int32(e%2+8*(e÷4))
        @inbounds smem[16r+c+1]=T(input[i,lane+1])
    end
    sync_threads()
    tile=SharedTile(pointer(smem),@Layout((16, 16), (16, 1)))
    a=@inbounds load_a(atom,tile,lane)
    ntuple(i -> (@inbounds out[i,lane+1]=a.data[i]),Val(4))
    nothing
end
function chained_mma_kernel!(out,a_data,b_data,v_data,atom::MMA16x8x16{T}) where T
    s=CuStaticSharedArray(T,640);tid=Int32(threadIdx().x)-Int32(1)
    sa=SharedTile(pointer(s),@Layout((16, 16), (16, 1)))
    sb=SharedTile(pointer(s)+512,@Layout((16, 16), (1, 16)))
    sv=SharedTile(pointer(s)+1024,@Layout((16, 8), (1, 16)))
    @inbounds begin
        copy_async!(CopyPlan{(16,16),32,2}(),sa,GlobalTile(pointer(a_data),sa.layout),tid)
        copy_async!(CopyPlan{(16,16),32,1}(),sb,GlobalTile(pointer(b_data),sb.layout),tid)
        copy_async!(CopyPlan{(16,8),32,1}(),sv,GlobalTile(pointer(v_data),sv.layout),tid)
    end
    commit_copies();wait_copies(Val(0));sync_threads()
    @inbounds begin
        aa=load_a(atom,sa,tid)
        b0=load_b(atom,window(sb,(Int32(0),Int32(0)),Val((16,8))),tid)
        b1=load_b(atom,window(sb,(Int32(0),Int32(8)),Val((16,8))),tid)
        c0=mma(atom,aa,b0,zero_accumulator(atom))
        c1=mma(atom,aa,b1,zero_accumulator(atom))
        c=mma(atom,pack_operand_a(atom,c0,c1),load_b(atom,sv,tid),zero_accumulator(atom))
        store!(GlobalTile(pointer(out),@Layout((16, 8), (1, 16))),c,tid)
    end
    nothing
end
if !("--runtime-only" in ARGS)
@testset "Same-lane operand conversion assembly" begin
    for T in (BFloat16,Float16),arch in (CUDACore.SMVersion(8,0),CUDACore.SMVersion(12,1,:arch))
        atom=MMA16x8x16(T)
        tt=Tuple{CuDeviceMatrix{UInt32,1},CuDeviceMatrix{Float32,1},typeof(atom)}
        code=compile_kernel(conversion_probe!,tt;arch,threads=32)
        save_code("operand-a-$T-$arch",code);body=entry_body(code.ptx)
        @test occursin(T==BFloat16 ? "cvt.rn.bf16x2.f32" : "cvt.rn.f16x2.f32",body)
        @test !occursin(r"shfl|ldmatrix|\.shared|\.local|\bcall",body)
    end
end
end
if CUDACore.functional()
@testset "Accumulator conversion bits and shared-load oracle" begin
    for T in (BFloat16,Float16)
        atom=MMA16x8x16(T)
        delta=T==BFloat16 ? 2f0^-7 : 2f0^-10
        cases=Float32[0,-0.0,1,1+delta/2,1+3delta/2,-1-delta/2,
            nextfloat(0f0),-nextfloat(0f0),Float32(nextfloat(zero(T))),
            -Float32(nextfloat(zero(T))),Float32(nextfloat(zero(T)))/2,Float32(floatmin(T)),Float32(floatmax(T)),
            floatmax(Float32),-floatmax(Float32),Inf,-Inf,NaN]
        x=randn(MersenneTwister(89),Float32,8,32).*4f0
        for i in eachindex(cases);x[i]=cases[i];end
        dx=CuArray(x); direct=CuArray{UInt32}(undef,4,32); shared=similar(direct)
        @cuda threads=32 conversion_probe!(direct,dx,atom)
        @cuda threads=32 conversion_shared_probe!(shared,dx,atom)
        dw,sw=Array(direct),Array(shared)
        for t in 1:32,e in 1:8
            shift=16*((e-1)%2);word=(e-1)÷2+1
            db=UInt16((dw[word,t]>>shift)&0xffff);sb=UInt16((sw[word,t]>>shift)&0xffff)
            if isnan(x[e,t])
                @test isnan(reinterpret(T,db)) && isnan(reinterpret(T,sb))
            else
                @test db == sb == reinterpret(UInt16,T(x[e,t]))
            end
        end
    end
end
@testset "Chained warp MMA with explicit rounding boundary" begin
    for T in (BFloat16,Float16)
        rng=MersenneTwister(213)
        a=T.(randn(rng,Float32,16,16).*0.3f0); b=T.(randn(rng,Float32,16,16).*0.3f0)
        v=T.(randn(rng,Float32,16,8))
        out=CuArray{Float32}(undef,16,8)
        @cuda threads=32 chained_mma_kernel!(out,CuArray(permutedims(a)),CuArray(b),CuArray(v),MMA16x8x16(T))
        reference=Float64.(T.(Float64.(a)*Float64.(b)))*Float64.(v)
        @test Array(out) ≈ reference rtol=5e-5 atol=3e-6
    end
end
end
