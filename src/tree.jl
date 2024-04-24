mutable struct BARTTree
    root::BARTNode
end

mutable struct BARTDecision
    value
end

mutable struct BARTNode
    v::Float64
    μ::Float64
    σ::Float64
    left
    right
    i::Vector{Int}
end
