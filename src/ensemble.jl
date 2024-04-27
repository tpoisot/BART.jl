"""
    BART

The main type
"""
mutable struct BART
    tree::Vector{Tree}
    state::StateParameters
    hyperparameters::HyperParameters
end