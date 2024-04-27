module BART

using Distributions
using Statistics
using StatsBase
using TestItems

include("parameters.jl")
export HyperParameters
export StateParameters
export NodeParameters

include("tree.jl")
export Tree
export DecisionNode

include("ensemble.jl")
export BART

include("moves.jl")
export grow!
export prune!
export change!
export swap!

include("utilities.jl")

include("core.jl")
include("priors.jl")
include("updates.jl")

# Datasets
include("data/bigfoot.jl")
include("data/friedman.jl")

end
