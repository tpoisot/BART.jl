"""
    BART

The main type
"""
mutable struct BARTModel
    y
    X
    encoder
    decoder
    trees::Vector{Tree}
    state::StateParameters
    hyperparameters::HyperParameters
end

function BARTModel(y, X; ν=30, q=0.90, k=2.0, m=200, α=0.95, β=2.0)
    enc, dec = BART.transformer(y)
    y₀ = enc(y)
    HP = HyperParameters(m=m, k=k, α=α, β=β)
    SP = StateParameters(y₀, HP; ν=ν, q=q)
    R₀ = y₀ .- mean(y₀) / HP.m
    NP = NodeParameters(mean(R₀), std(R₀), 0.0)
    root = DecisionNode(collect(axes(y, 1)), missing, missing, nothing, nothing, 0, deepcopy(NP))
    ensemble = [Tree(y₀, X, deepcopy(root)) for _ in Base.OneTo(HP.m)]
    BART.updateleaf!(ensemble[1].root, SP)
    for i in 2:HP.m
        ensemble[i].y = BART.R(ensemble[i-1])
        ensemble[i].root.parameters.μ = mean(ensemble[i].y)
        ensemble[i].root.parameters.σ = std(ensemble[i].y)
        BART.updateleaf!(ensemble[i].root, SP)
    end
    return BARTModel(y, X, enc, dec, ensemble, SP, HP)
end