using Tylo
using Documenter

DocMeta.setdocmeta!(Tylo, :DocTestSetup, :(using Tylo); recursive=true)

makedocs(;
    modules=[Tylo],
    authors="AntonOresten <antonoresten@proton.me> and contributors",
    sitename="Tylo.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://docs.jool.space/Tylo.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Start here" => "index.md",
        "Design and boundaries" => "design.md",
        "Layouts" => "layouts.md",
        "Fragments and scalar arithmetic" => "rows.md",
        "Walk through GEMM" => "gemm.md",
        "TMA and WGMMA" => "hopper.md",
        "TMEM tiles and transfers" => "tmem.md",
        "Streaming attention" => "streaming.md",
        "Boundary tiles" => "boundaries.md",
        "Read the implementation" => "codebase.md",
        "Current status and validation" => "validation.md",
        "API reference" => "api.md",
        "Validation history" => "validation-history.md",
    ],
)

deploydocs(;
    repo="github.com/jool-space/Tylo.jl",
    deploy_repo="github.com/jool-space/docs",
    devbranch="main",
    dirname="Tylo.jl",
)
