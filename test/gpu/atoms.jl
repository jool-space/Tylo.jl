# TEST_TARGET: cc>=8.0
using Tylo.Layouts: @Layout, Layout, Swizzle, compose, static

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

# Shared-memory copies derived from the operand ownerships. Registers reach
# shared memory through `store!` (stmatrix from sm_90, scalar stores
# otherwise), the raw tile is exported for the host, and the tile is loaded
# back through `load_fragment` (ldmatrix wherever a plan derives).
@generated function export_words!(out,f::F,g::F,lane) where F
    W = F.parameters[2]
    stores = [:(@inbounds out[$i,lane+1] = f.data[$i]; @inbounds out[$(i+W),lane+1] = g.data[$i]) for i in 1:W]
    quote
        Base.@inline
        $(stores...)
        nothing
    end
end
function atom_copy_kernel!(raw_a,raw_b,words_a,words_b,a_data,b_data,atom,config)
    TA,TB=eltype(atom,OperandA()),eltype(atom,OperandB())
    lane=Int32(threadIdx().x)-Int32(1)
    m,n,k=size(atom)
    bytes_a=m*k*sizeof(TA)
    smem=@inbounds CuDynamicSharedArray(UInt8,bytes_a+k*n*sizeof(TB))
    sa=SharedTile(reinterpret(Core.LLVMPtr{TA,3},pointer(smem)),config.sa)
    sb=SharedTile(reinterpret(Core.LLVMPtr{TB,3},pointer(smem)+bytes_a),config.sb)
    a=GlobalTile(pointer(a_data),@Layout((m,k),(1,m)))
    b=GlobalTile(pointer(b_data),@Layout((k,n),(1,k)))
    ga=load_fragment(operand_layout(atom,OperandA()),a,lane)
    gb=load_fragment(operand_layout(atom,OperandB()),b,lane)
    @inbounds store!(sa,ga,lane)
    @inbounds store!(sb,gb,lane)
    sync_threads()
    for i in lane:Int32(32):Int32(m*k-1)
        r,c=i%Int32(m),i÷Int32(m)
        @inbounds raw_a[r+1,c+1]=unsafe_load(pointer(sa,(r,c)))
    end
    for i in lane:Int32(32):Int32(k*n-1)
        r,c=i%Int32(k),i÷Int32(k)
        @inbounds raw_b[r+1,c+1]=unsafe_load(pointer(sb,(r,c)))
    end
    fa=@inbounds load_fragment(operand_layout(atom,OperandA()),sa,lane)
    fb=@inbounds load_fragment(operand_layout(atom,OperandB()),sb,lane)
    export_words!(words_a,fa,ga,lane)
    export_words!(words_b,fb,gb,lane)
    nothing
end

# Static shared layouts with the contiguous axis chosen, optionally
# swizzled at 16-byte granularity when a row holds at least two vectors.
function copy_layout(T,rows,cols,axis,swizzled)
    l = axis == 2 ? Layout((static(rows),static(cols)),(static(cols),static(1))) :
                    Layout((static(rows),static(cols)),(static(1),static(rows)))
    swizzled || return l
    groups = (axis == 2 ? cols : rows)*sizeof(T) ÷ 16
    groups >= 2 || return nothing
    bits = trailing_zeros(groups)
    compose(Swizzle{bits,trailing_zeros(16 ÷ sizeof(T)),bits}(),l)
end
# (A axis, B axis, swizzled): K contiguous for both operands, then MN contiguous.
copy_cases(atom) = size(atom) == (16,8,16) && Tylo._element_bits(eltype(atom,OperandA())) == 16 ?
    ((2,1,false),(2,1,true),(1,2,false),(1,2,true)) : ((2,1,false),(2,1,true))
function copy_config(atom,case)
    TA,TB=eltype(atom,OperandA()),eltype(atom,OperandB())
    m,n,k=size(atom)
    sa=copy_layout(TA,m,k,case[1],case[3])
    sb=copy_layout(TB,k,n,case[2],case[3])
    sa === nothing || sb === nothing ? nothing : (;sa,sb)
end
copy_plans(atom,case) = (Tylo.matrix_copy_plan(operand_layout(atom,OperandA()),eltype(atom,OperandA()),case[1]),
                         Tylo.matrix_copy_plan(operand_layout(atom,OperandB()),eltype(atom,OperandB()),case[2]))
words_type(atom) = Tylo._element_bits(eltype(atom,OperandA())) == 32 ? Float32 : UInt32

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
# The instructions in the assembly follow the derived plans exactly: one
# ldmatrix per block group with the planned width and transposition, and
# stmatrix only when compiling for sm_90 or later.
@testset "Derived matrix copy assembly" begin
    for atom in instruction_atoms(), case in copy_cases(atom), arch in oracle_archs(atom)
        config=copy_config(atom,case)
        config === nothing && continue
        TA,TB=eltype(atom,OperandA()),eltype(atom,OperandB())
        WT=words_type(atom)
        tt=Tuple{CuDeviceMatrix{TA,1},CuDeviceMatrix{TB,1},CuDeviceMatrix{WT,1},CuDeviceMatrix{WT,1},
                 CuDeviceMatrix{TA,1},CuDeviceMatrix{TB,1},typeof(atom),typeof(config)}
        code=compile_kernel(atom_copy_kernel!,tt;arch,threads=32)
        save_code("atom-copy-$(oracle_name(atom))-$(case[1])$(case[2])$(case[3] ? "s" : "")-$arch",code)
        body=entry_body(code.ptx)
        @test !occursin(".local .",body)
        @test !occursin(r"\bcall",body)
        expected=String[]
        for plan in copy_plans(atom,case)
            plan === nothing && continue
            for group in Tylo._matrix_groups(length(plan.words))
                push!(expected,"ldmatrix.sync.aligned.m8n8.x$(length(group))$(plan.trans ? ".trans" : "").shared.b16")
            end
        end
        @test count("ldmatrix",body) == length(expected)
        for instruction in unique(expected)
            @test count(instruction,body) == count(==(instruction),expected)
        end
        stores=arch.major >= 9 ? length(expected) : 0
        @test count("stmatrix",body) == stores
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
    end
end
@testset "Derived matrix copies round-trip through shared memory" begin
    for atom in instruction_atoms(), case in copy_cases(atom)
        config=copy_config(atom,case)
        config === nothing && continue
        TA,TB=eltype(atom,OperandA()),eltype(atom,OperandB())
        m,n,k=size(atom)
        rng=MersenneTwister(hash((size(atom),TA,TB,case)) % 100000)
        a=oracle_inputs(rng,TA,(m,k)); b=oracle_inputs(rng,TB,(k,n))
        wa=Tylo._register_count(operand_layout(atom,OperandA()))*Tylo._element_bits(TA)÷32
        wb=Tylo._register_count(operand_layout(atom,OperandB()))*Tylo._element_bits(TB)÷32
        WT=words_type(atom)
        raw_a=CUDACore.zeros(TA,m,k); raw_b=CUDACore.zeros(TB,k,n)
        words_a=CUDACore.zeros(WT,2wa,32); words_b=CUDACore.zeros(WT,2wb,32)
        @cuda threads=32 shmem=m*k*sizeof(TA)+k*n*sizeof(TB) atom_copy_kernel!(
            raw_a,raw_b,words_a,words_b,CuArray(a),CuArray(b),atom,config)
        @test Array(raw_a) == a
        @test Array(raw_b) == b
        ha,hb=Array(words_a),Array(words_b)
        @test ha[1:wa,:] == ha[wa+1:2wa,:]
        @test hb[1:wb,:] == hb[wb+1:2wb,:]
    end
end
end
