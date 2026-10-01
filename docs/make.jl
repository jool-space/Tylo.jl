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
        size_threshold=400_000,
        size_threshold_warn=250_000,
    ),
    pages=[
        "Tylo" => "index.md",
        "Layouts and ownership" => "layouts.md",
        "Memory tiles" => "tiles.md",
        "Fragments" => "fragments.md",
        "Hardware operations" => "operations.md",
        "Pipelines" => "pipelines.md",
        "A complete kernel" => "kernel.md",
        "Machine code, tuning and timelines" => "machine.md",
        "Design" => "design.md",
        "Status" => "status.md",
        "API reference" => "api.md",
    ],
)

deploydocs(;
    repo="github.com/jool-space/Tylo.jl",
    deploy_repo="github.com/jool-space/docs",
    devbranch="main",
    dirname="Tylo.jl",
)
