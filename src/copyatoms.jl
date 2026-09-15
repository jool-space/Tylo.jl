"""
    CopyAtom{:load|:store,Trans}()
    CopyAtom(op, trans=false)

A warp-collective copy of one 8×8 matrix of 16-bit units between shared
memory and registers: `ldmatrix` for `:load`, `stmatrix` for `:store`, with
the `.trans` modifier when `Trans` is true. Like an [`MMAAtom`](@ref), the
atom contributes only ownerships. They use memory coordinates: axis 1
indexes the eight 16-byte rows, axis 2 the eight units of a row.
[`operand_layout`](@ref) with [`Addresses`](@ref) gives the row each lane
addresses, and with [`Registers`](@ref) the two units each lane holds in
one register word.

Copies of larger ownerships derive from these tables. `matrix_copy_plan`
covers a target ownership with 8×8 blocks along the axis that memory stores
contiguously, chooses the transposed variant when the register pattern
requires it, groups blocks into `.x1`, `.x2` and `.x4` instructions and
permutes the result words into the target's slot order. Adjacent pairs of
8-bit elements form one unit, so 8-bit operands qualify as well.
`load_fragment` and `store!` use the derived copy for shared tiles.
"""
struct CopyAtom{Op,Trans}
    function CopyAtom{Op,Trans}() where {Op,Trans}
        Op in (:load,:store) && Trans isa Bool ||
            throw(ArgumentError("copy atoms are :load or :store, transposed or not"))
        new{Op,Trans}()
    end
end
CopyAtom(op::Symbol,trans::Bool=false) = CopyAtom{op,trans}()
Base.size(::CopyAtom) = (8,8)
threads(::CopyAtom) = 32
"The register operand of a copy atom: two units per lane in one word."
struct Registers <: OperandRole end
"The address operand of a copy atom: the 16-byte row each lane supplies."
struct Addresses <: OperandRole end

# PTX ISA ldmatrix/stmatrix figures. Lane t holds units (t÷4, 2(t%4)+e) of
# a matrix, or their transpose with `.trans`; lanes 8i:8i+7 address the
# eight rows of matrix i, so lane t supplies row t%8 of its matrix.
_registers_layout(trans) = trans ? _ownership((8,8),((4,2),(8,8)),((2,1),)) :
                                   _ownership((8,8),((4,16),(8,1)),((2,8),))
_addresses_layout() = _ownership((8,8),((8,1),(4,0)),((8,8),))
@generated operand_layout(::CopyAtom{Op,Trans},::Registers) where {Op,Trans} = :($(_registers_layout(Trans)))
@generated operand_layout(::CopyAtom{Op,Trans},::Addresses) where {Op,Trans} = :($(_addresses_layout()))
"The four matrix copy atoms: loads and stores, plain and transposed."
copy_atoms() = [CopyAtom{op,trans}() for trans in (false,true) for op in (:load,:store)]
