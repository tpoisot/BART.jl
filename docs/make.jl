using Documenter
using BART

makedocs(
    sitename = "BART",
    format = Documenter.HTML(),
    modules = [BART],
    pages = [
        "BART.jl" => "index.md",
        "Trees" => "trees.md",
        "Datasets" => "datasets.md",
    ]
)

deploydocs(
    repo = "github.com/tpoisot/BART.jl.git",
    push_preview = true,
)