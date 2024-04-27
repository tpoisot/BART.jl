module BART

using Distributions
using Statistics

using TestItems

include("tree.jl")
export Tree
export DecisionNode

include("moves.jl")
export grow!
export prune!
export change!
export swap!

include("utilities.jl")

include("core.jl")

# Datasets
include("data/bigfoot.jl")
include("data/friedman.jl")

end
