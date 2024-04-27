module BART

using Distributions
using Statistics
using TestItems

Base.@kwdef struct HyperParameters
    k::Float64 = 2.0
    m::Integer = 200
end
export HyperParameters

mutable struct StateParameters
    σ::Float64
    σᵤ::Float64
    ν::Float64
    λ::Float64
end
export StateParameters

function StateParameters(y, HP::HyperParameters; ν=3.0, q=0.90)
    σ = std(y)
    σᵤ = 0.5/(HP.k*sqrt(HP.m))
    λ = σ^2 * quantile(Chisq(ν),1-q)/ν
    return StateParameters(σ, σᵤ, ν, λ)
end

Base.@kwdef mutable struct NodeParameters
    μ::Float64 = NaN
    σ::Float64 = NaN
    v::Float64 = NaN
end
export NodeParameters

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
include("parameters.jl")
include("priors.jl")

# Datasets
include("data/bigfoot.jl")
include("data/friedman.jl")

end
