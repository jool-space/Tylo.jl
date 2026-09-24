# Two register distributions of a row. Lane t owns row t, `n` values of it;
# or the 32 lanes of a warp interleave along one row of `32n` values.
@inline row_per_lane(n) = Ownership((32,n),Layout((Tylo.Layouts.static(32),Tylo.Layouts.static(n)),
                                                  (Tylo.Layouts.static(1),Tylo.Layouts.static(32))))
@inline row_per_warp(n) = Ownership((1,32n),Layout((Tylo.Layouts.static(32),Tylo.Layouts.static(n)),
                                                   (Tylo.Layouts.static(1),Tylo.Layouts.static(32))))
