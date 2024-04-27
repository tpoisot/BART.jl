"""
    BART

The main type
"""
mutable struct BARTModel
    trees::Vector{Tree}
    state::StateParameters
    hyperparameters::HyperParameters
end