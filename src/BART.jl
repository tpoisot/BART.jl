module BART

using Distributions
using Statistics

using TestItems

include("tree.jl")
export Tree
export DecisionNode

include("utilities.jl")

# Datasets
include("bigfoot.jl")
include("friedman.jl")

end
