using CUDACore.GPUCompiler: CompilerJob, methodinstance

function compile_kernel(f, tt; arch=CUDACore.SMVersion(10,0,:arch), threads=128)
    config = CUDACore.compiler_config(nothing;kernel=true,arch,minthreads=threads)
    job = CompilerJob(methodinstance(typeof(f),tt),config)
    image = CUDACore.invoke_frozen(CUDACore.compile,job).image
    io = IOBuffer()
    CUDACore.invoke_frozen(CUDACore.GPUCompiler.code_native,io,job;dump_module=true)
    (;image,ptx=String(take!(io)))
end

function save_code(name, code)
    haskey(ENV,"TYLO_EVIDENCE") || return
    dir = ENV["TYLO_EVIDENCE"]
    mkpath(dir)
    write(joinpath(dir,name*".ptx"),code.ptx)
    write(joinpath(dir,name*".cubin"),code.image)
end

function entry_body(ptx)
    split(split(ptx,".visible .entry";limit=2)[2],"// -- End function";limit=2)[1]
end

# Extract a named entry's machine-code bytes from a CUDA ELF64 little-endian
# cubin. Debug/source metadata are deliberately outside this comparison.
function kernel_text(image, prefix=".text._Z11fab_kernel_")
    @assert image[1:6] == UInt8[0x7f,0x45,0x4c,0x46,2,1]
    u16(off) = Int(ltoh(reinterpret(UInt16,image[off+1:off+2])[1]))
    u32(off) = Int(ltoh(reinterpret(UInt32,image[off+1:off+4])[1]))
    u64(off) = Int(ltoh(reinterpret(UInt64,image[off+1:off+8])[1]))
    table = u64(0x28)
    stride, count, names_idx = u16(0x3a), u16(0x3c), u16(0x3e)
    names = u64(table+names_idx*stride+0x18)
    for idx in 0:count-1
        header = table+idx*stride
        start = names+u32(header)+1
        stop = findnext(iszero,image,start)
        name = String(image[start:stop-1])
        if startswith(name,prefix)
            offset, bytes = u64(header+0x18), u64(header+0x20)
            return image[offset+1:offset+bytes]
        end
    end
    error("kernel code section not found: $prefix")
end
