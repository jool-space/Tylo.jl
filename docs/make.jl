using Tylo
using Documenter

DocMeta.setdocmeta!(Tylo, :DocTestSetup, :(using Tylo); recursive=true)

makedocs(;
    modules=[Tylo],
    authors="AntonOresten <antonoresten@proton.me> and contributors",
    sitename="Tylo.jl",
    format=Documenter.HTML(;
        canonical="https://jool-space.github.io/Tylo.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/jool-space/Tylo.jl",
    devbranch="main",
)
