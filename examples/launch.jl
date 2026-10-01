using Tylo, CUDACore

# A kernel launch prepared on the host: the kernel, its arguments (host values,
# converted as `@cuda` converts them), and its compile and launch options.
# Nothing compiles until `compile`, so the machine code of a launch prepared
# for another device (say, `sms = 148` for a B200) can be inspected anywhere:
# `machine_code(launch; arch)`.
struct Launch{F,A,O,R}
    kernel::F
    args::A
    options::O      # (; threads, blocks, shmem, clustersize, minthreads, maxregs)
    result::R       # what the launch computes, the first argument unless given
end
Launch(kernel,args;result=first(args),threads,blocks,shmem=0,clustersize=1,minthreads=nothing,maxregs=nothing) =
    Launch(kernel,args,(;threads,blocks,shmem,clustersize,minthreads,maxregs),result)

argument_types(l::Launch) = Tuple{map(a -> Core.Typeof(CUDACore.cudaconvert(a)),l.args)...}

"The compiled kernel, its dynamic shared memory limit raised to the launch's."
function compile(l::Launch)
    (; minthreads, maxregs, shmem) = l.options
    kernel = cufunction(l.kernel,argument_types(l);minthreads,maxregs)
    shmem > 0 && (CUDACore.attributes(kernel.fun)[CUDACore.FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES] = shmem)
    kernel
end
"Launch `l` with its compiled `kernel`; the host arguments live until it is issued."
function (l::Launch)(kernel=compile(l))
    (; threads, blocks, shmem, clustersize) = l.options
    GC.@preserve l kernel(l.args...;threads,blocks,shmem,clustersize)
end

# The machine code the launch would run, compiled for `arch` without a device.
function Tylo.machine_code(l::Launch;arch,kwargs...)
    (; minthreads, maxregs) = l.options
    options = (; arch, (minthreads === nothing ? (;) : (;threads=minthreads))..., (maxregs === nothing ? (;) : (;maxregs))..., kwargs...)
    Tylo.machine_code(l.kernel,argument_types(l);options...)
end

"The device's multiprocessor count, which persistent launches size their grid by."
multiprocessors() = CUDACore.attribute(CUDACore.device(),CUDACore.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
