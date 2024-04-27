using Documenter
using BART

makedocs(
    sitename = "BART",
    format = Documenter.HTML(),
    modules = [BART]
)

deploydocs(
    repo = "github.com/tpoisot/BART.jl.git",
    push_preview = true,
)