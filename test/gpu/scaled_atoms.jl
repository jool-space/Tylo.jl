# TEST_TARGET: cc==12
scaled_atoms() = filter(a -> a isa ScaledMMAAtom,instruction_atoms())
using Tylo.Layouts: @Layout

# A block-scaled product over NB microscaling blocks of 32 along K: operands
# and scales load through the atom's ownerships, a scale register serves four
# blocks, and the accumulator carries the sum. No value is dequantized.
function scaled_atom_kernel!(out,a_data,b_data,sa_data,sb_data,atom,::Val{NB}) where NB
    lane=Int32(threadIdx().x)-Int32(1)
    m,n,k=size(atom)
    a=GlobalTile(pointer(a_data),@Layout((m,k*NB),(1,m)))
    b=GlobalTile(pointer(b_data),@Layout((k*NB,n),(1,k*NB)))
    sa=GlobalTile(pointer(sa_data),@Layout((m,NB),(1,m)))
    sb=GlobalTile(pointer(sb_data),@Layout((NB,n),(1,NB)))
    c=GlobalTile(pointer(out),@Layout((m,n),(1,m)))
    acc=zero_accumulator(atom)
    # The launch supplies NB complete blocks; block offsets are in bounds by construction.
    @inbounds for word in Int32(0):Int32(NB÷4-1)
        fsa=load_fragment(operand_layout(atom,ScaleA()),window(sa,(Int32(0),4word),(m,4)),lane)
        fsb=load_fragment(operand_layout(atom,ScaleB()),window(sb,(4word,Int32(0)),(4,n)),lane)
        for block in Int32(0):Int32(3)
            s=4word+block
            fa=load_fragment(operand_layout(atom,OperandA()),window(a,(Int32(0),Int32(k)*s),(m,k)),lane)
            fb=load_fragment(operand_layout(atom,OperandB()),window(b,(Int32(k)*s,Int32(0)),(k,n)),lane)
            acc=mma(atom,fa,fb,acc,fsa,fsb,block)
        end
    end
    store!(c,acc,lane)
    nothing
end
scaled_name(atom) = join((string(eltype(atom,OperandA())),string(eltype(atom,OperandB()))),'-')

@testset "Block-scaled atom assembly" begin
    @test length(scaled_atoms()) == 4
    for atom in scaled_atoms(), arch in (CUDACore.SMVersion(12,1,:arch),CUDACore.SMVersion(12,0,:family)), nb in (4,8)
        TA,TB=eltype(atom,OperandA()),eltype(atom,OperandB())
        tt=Tuple{CuDeviceMatrix{Float32,1},CuDeviceMatrix{TA,1},CuDeviceMatrix{TB,1},
                 CuDeviceMatrix{Float8_E8M0FNU,1},CuDeviceMatrix{Float8_E8M0FNU,1},typeof(atom),Val{nb}}
        code=compile_kernel(scaled_atom_kernel!,tt;arch,threads=32)
        save_code("scaled-atom-$(scaled_name(atom))-b$nb-$arch",code)
        body=entry_body(code.ptx)
        @test !occursin(".local .",body)
        @test !occursin(r"\bcall",body)
        @test occursin("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale",body)
    end
end

if runtime_supported(@__FILE__)
    @testset "Block-scaled atom oracle" begin
        rng=MersenneTwister(29)
        for atom in scaled_atoms(), nb in (4,8)
            TA,TB=eltype(atom,OperandA()),eltype(atom,OperandB())
            m,n,k=size(atom)
            # Products and sums are exact in FP32 for these values and scale
            # exponents, so the comparison exposes ownership mistakes, not rounding.
            a=TA.(rand(rng,-4:4,m,k*nb) ./ 2); b=TB.(rand(rng,-4:4,k*nb,n) ./ 2)
            sa=reinterpret.(Float8_E8M0FNU,UInt8.(rand(rng,124:130,m,nb)))
            sb=reinterpret.(Float8_E8M0FNU,UInt8.(rand(rng,124:130,nb,n)))
            out=CUDACore.zeros(Float32,m,n)
            @cuda threads=32 scaled_atom_kernel!(out,CuArray(a),CuArray(b),CuArray(sa),CuArray(sb),atom,Val(nb))
            expected=zeros(Float64,m,n)
            for s in 1:nb
                ks=k*(s-1)+1:k*s
                expected .+= (Float64.(a[:,ks]) .* Float64.(sa[:,s])) * (Float64.(b[ks,:]) .* Float64.(sb[s:s,:]))
            end
            @test Array(out) == Float32.(expected)
            # Every row and column scale matters: a product with unit scales differs.
            @test Array(out) != Float32.(Float64.(a)*Float64.(b))
        end
    end
end
