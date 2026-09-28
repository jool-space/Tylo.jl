# The clock read is volatile asm with a memory clobber: it stays among the
# memory operations and barriers around it, and repeated reads stay apart.
@inline function record!(t::Timeline,recorder::Integer,slot::Integer,event::Integer)
    records = t.records
    if 0 <= slot < size(records,1)
        clock = ptx"mov.u64"(PTX.@sreg_str("%clock64"))
        block = ptx"mov.u32"(PTX.@sreg_str("%ctaid.x")) % Int
        @inbounds records[Int(slot)+1,Int(recorder)+1,block+1] = (clock << 8) | (event % UInt64 & 0xff)
    end
    nothing
end
