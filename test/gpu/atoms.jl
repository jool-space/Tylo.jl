# TEST_TARGET: cc>=8.0
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

# Host inputs whose products and sums are exact in the atom's arithmetic,
# so the comparison exposes ownership mistakes rather than rounding.
oracle_inputs(rng,::Type{T},dims) where T<:Union{BFloat16,Float16} = T.(rand(rng,-4:4,dims...) ./ 4)
oracle_inputs(rng,::Type{Float32},dims) = Float32.(rand(rng,-8:8,dims...) ./ 8)  # exact in TF32
oracle_inputs(rng,::Type{T},dims) where T<:Union{Float8E4M3,Float8E5M2} = T.(rand(rng,-4:4,dims...) ./ 2)
oracle_inputs(rng,::Type{Int8},dims) = rand(rng,Int8(-8):Int8(7),dims...)
oracle_inputs(rng,::Type{UInt8},dims) = rand(rng,UInt8(0):UInt8(15),dims...)
oracle_expected(a,b,::Type{TC}) where TC = TC.(Float64.(a)*Float64.(b))
oracle_expected(a,b,::Type{Int32}) = Int32.(Int64.(a)*Int64.(b))
oracle_archs(atom) = eltype(atom,OperandA()) in (Float8E4M3,Float8E5M2) ?
    (CUDACore.SMVersion(8,9),CUDACore.SMVersion(12,1,:arch)) :
    (CUDACore.SMVersion(8,0),CUDACore.SMVersion(12,1,:arch))
oracle_name(atom) = join((string(eltype(atom,OperandA())),string(eltype(atom,OperandB())),
                          string(eltype(atom,Accumulator())),join(size(atom),'x')),'-')

begin # assembly checks
@testset "Atom oracle assembly" begin
    @test length(instruction_atoms()) == 24
    for atom in instruction_atoms(), arch in oracle_archs(atom)
        TA,TB,TC=eltype(atom,OperandA()),eltype(atom,OperandB()),eltype(atom,Accumulator())
        tt=Tuple{CuDeviceMatrix{TC,1},CuDeviceMatrix{TA,1},CuDeviceMatrix{TB,1},typeof(atom)}
        code=compile_kernel(atom_oracle_kernel!,tt;arch,threads=32)
        save_code("atom-oracle-$(oracle_name(atom))-$arch",code)
        body=entry_body(code.ptx)
        @test !occursin(".local .",body)
        @test !occursin(r"\bcall",body)
        @test occursin("mma.sync.aligned.m16n8k$(size(atom)[3])",body)
    end
end
end

if runtime_supported(@__FILE__)
@testset "Atom oracle: generic loads, MMA and stores match a host matmul" begin
    for atom in instruction_atoms()
        TA,TB,TC=eltype(atom,OperandA()),eltype(atom,OperandB()),eltype(atom,Accumulator())
        m,n,k=size(atom)
        rng=MersenneTwister(hash((m,n,k,TA,TB,TC)) % 100000)
        a=oracle_inputs(rng,TA,(m,k)); b=oracle_inputs(rng,TB,(k,n))
        expected=oracle_expected(a,b,TC)
        out=CuArray(fill(TC(0),m,n))
        @cuda threads=32 atom_oracle_kernel!(out,CuArray(a),CuArray(b),atom)
        @test Array(out) == expected
        if size(atom) == (16,8,16) && Tylo._element_bits(TA) == 16
            words_a=CUDACore.zeros(UInt32,8,32); words_b=CUDACore.zeros(UInt32,4,32)
            @cuda threads=32 shmem=(m*k+k*n)*sizeof(TA) atom_shared_loads_kernel!(words_a,words_b,CuArray(a),CuArray(b),atom)
            wa,wb=Array(words_a),Array(words_b)
            @test wa[1:4,:] == wa[5:8,:]
            @test wb[1:2,:] == wb[3:4,:]
        end
    end
end
end
