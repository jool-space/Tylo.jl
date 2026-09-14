using Tylo.Layouts: @Layout

# The generic oracle: load A and B through scalar loads at the ownership
# coordinates, multiply, store C the same way, and compare with a host
# matmul. A wrong ownership in any operand fails it. Every atom with an
# instruction binding is registered here.
function atom_oracle_kernel!(out,a_data,b_data,atom)
    lane=Int32(threadIdx().x)-Int32(1)
    m,n,k=size(atom)
    a=GlobalTile(pointer(a_data),@Layout((m,k),(1,m)))
    b=GlobalTile(pointer(b_data),@Layout((k,n),(1,k)))
    c=GlobalTile(pointer(out),@Layout((m,n),(1,m)))
    fa=load_fragment(operand_layout(atom,OperandA()),a,lane)
    fb=load_fragment(operand_layout(atom,OperandB()),b,lane)
    store!(c,mma(atom,fa,fb,zero_accumulator(atom)),lane)
    nothing
end

# ldmatrix loads must agree word for word with the generic loads.
function atom_shared_loads_kernel!(words_a,words_b,a_data,b_data,atom)
    T=eltype(atom,OperandA())
    lane=Int32(threadIdx().x)-Int32(1)
    m,n,k=size(atom)
    smem=@inbounds CuDynamicSharedArray(T,m*k+k*n)
    sa=SharedTile(pointer(smem),@Layout((m,k),(k,1)))
    sb=SharedTile(pointer(smem)+m*k*sizeof(T),@Layout((k,n),(1,k)))
    a=GlobalTile(pointer(a_data),@Layout((m,k),(1,m)))
    b=GlobalTile(pointer(b_data),@Layout((k,n),(1,k)))
    for i in lane:Int32(32):Int32(m*k-1)
        r,c=i%Int32(m),i÷Int32(m)
        unsafe_store!(pointer(sa,(r,c)),unsafe_load(pointer(a,(r,c))))
    end
    for i in lane:Int32(32):Int32(k*n-1)
        r,c=i%Int32(k),i÷Int32(k)
        unsafe_store!(pointer(sb,(r,c)),unsafe_load(pointer(b,(r,c))))
    end
    sync_threads()
    fa=@inbounds load_a(atom,sa,lane)
    fb=@inbounds load_b(atom,sb,lane)
    ga=load_fragment(operand_layout(atom,OperandA()),a,lane)
    gb=load_fragment(operand_layout(atom,OperandB()),b,lane)
    ntuple(i -> (@inbounds words_a[i,lane+1]=fa.data[i]; @inbounds words_a[i+4,lane+1]=ga.data[i]),Val(4))
    ntuple(i -> (@inbounds words_b[i,lane+1]=fb.data[i]; @inbounds words_b[i+2,lane+1]=gb.data[i]),Val(2))
    nothing
end

oracle_atoms() = [MMAAtom((16,8,16),T) for T in (BFloat16,Float16)]

if !("--runtime-only" in ARGS)
@testset "Atom oracle assembly" begin
    for atom in oracle_atoms(), arch in (CUDACore.SMVersion(8,0),CUDACore.SMVersion(12,1,:arch))
        T=eltype(atom,OperandA())
        tt=Tuple{CuDeviceMatrix{Float32,1},CuDeviceMatrix{T,1},CuDeviceMatrix{T,1},typeof(atom)}
        code=compile_kernel(atom_oracle_kernel!,tt;arch,threads=32)
        save_code("atom-oracle-$(T)-$(join(size(atom),'x'))-$arch",code)
        body=entry_body(code.ptx)
        @test !occursin(".local .",body)
        @test !occursin(r"\bcall",body)
        @test occursin("mma.sync.aligned.m16n8k16",body)
    end
end
end

if CUDACore.functional()
@testset "Atom oracle: generic loads, MMA and stores match a host matmul" begin
    for atom in oracle_atoms()
        T=eltype(atom,OperandA())
        m,n,k=size(atom)
        rng=MersenneTwister(m+n+k+sizeof(T))
        a=T.(randn(rng,Float32,m,k)); b=T.(randn(rng,Float32,k,n))
        expected=Float32.(a)*Float32.(b)
        out=CuArray(fill(NaN32,m,n))
        @cuda threads=32 atom_oracle_kernel!(out,CuArray(a),CuArray(b),atom)
        @test isapprox(Array(out),expected;atol=1e-3,rtol=1e-3)
        words_a=CUDACore.zeros(UInt32,8,32); words_b=CUDACore.zeros(UInt32,4,32)
        @cuda threads=32 shmem=(m*k+k*n)*sizeof(T) atom_shared_loads_kernel!(words_a,words_b,CuArray(a),CuArray(b),atom)
        wa,wb=Array(words_a),Array(words_b)
        @test wa[1:4,:] == wa[5:8,:]
        @test wb[1:2,:] == wb[3:4,:]
    end
end
end
