"""
    Config(; name = value, ...)

A kernel's constants as one argument: `cfg.name` in the kernel is the value,
known while the kernel compiles, so that branches on it vanish and
arithmetic with it folds. Integers come out as `Int32`, the kernels' index
type; other values (Booleans, tuples, symbols, functions without fields) as
they are. Each distinct configuration compiles its own kernel.

```julia
cfg = Config(stages=4, ctas=2, causal=false)
@cuda kernel!(out, cfg)               # in the kernel: Ring stages, `cfg.causal && ...`
```
"""
struct Config{P}
    function Config{P}() where P
        P isa NamedTuple || throw(ArgumentError("a configuration is a NamedTuple of constants"))
        new{P}()
    end
end
Config(; kwargs...) = Config{(; kwargs...)}()
Config(p::NamedTuple) = Config{p}()

@inline _constant(x::Integer) = x isa Bool ? x : x % Int32
@inline _constant(x) = x
@inline Base.getproperty(::Config{P},name::Symbol) where P = _constant(getfield(P,name))
Base.propertynames(::Config{P}) where P = keys(P)
Base.hasproperty(::Config{P},name::Symbol) where P = haskey(P,name)
"The configuration's constants as a NamedTuple, as given."
Base.NamedTuple(::Config{P}) where P = P
"`cfg` with some constants replaced or added."
Base.merge(::Config{P},kwargs::NamedTuple) where P = Config(merge(P,kwargs))
Base.show(io::IO,::Config{P}) where P = (print(io,"Config"); show(io,P))
