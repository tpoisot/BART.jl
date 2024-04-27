Base.@kwdef struct HyperParameters
    k::Float64 = 2.0
    m::Integer = 200
    α::Float64 = 0.95
    β::Float64 = 2.0
end

mutable struct StateParameters
    σ::Float64
    σᵤ::Float64
    ν::Float64
    λ::Float64
end

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