# The clock read is volatile asm with a memory clobber: it stays among the
# memory operations and barriers around it, and repeated reads stay apart.
@inline function recorder(t::Timeline,r::Integer)
    records = t.records
    capacity, recorders, blocks = size(records)
    block = ptx"mov.u32"(PTX.@sreg_str("%ctaid.x")) % Int
    inside = 0 <= r < recorders && block < blocks
    Recorder(pointer(records)+8capacity*(Int(r)+recorders*block),inside ? capacity % Int32 : Int32(0))
end
@inline recorder(::Nothing,r::Integer) = nothing

@inline function record!(r::Recorder,slot::Integer,event::Integer)
    if slot % UInt32 < r.capacity % UInt32
        clock = ptx"mov.u64"(PTX.@sreg_str("%clock64"))
        unsafe_store!(r.records,clock | (UInt64(event % UInt8) << 56),Int(slot)+1,Val(8))
    end
    nothing
end
@inline record!(::Nothing,slot::Integer,event::Integer) = nothing
@inline record!(t::Timeline,r::Integer,slot::Integer,event::Integer) = record!(recorder(t,r),slot,event)
