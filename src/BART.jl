module BART

using Distributions
using Statistics

using TestItems

include("tree.jl")
export Tree
export DecisionNode

include("utilities.jl")

# Datasets
include("data/bigfoot.jl")
include("data/friedman.jl")

end
