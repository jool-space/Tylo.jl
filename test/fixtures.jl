# Explicit ownership patterns reused across host and device test fixtures.
@inline local_fragment(data::NTuple{N}) where N = Fragment(data,Tylo.Layouts.LocalOwnership{N,2}())
@inline striped_fragment(data::NTuple{N}) where N = Fragment(data,Tylo.Layouts.StripedOwnership{N,2}())
