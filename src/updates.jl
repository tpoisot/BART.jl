function updateleaf!(node::DecisionNode, SP::StateParameters)
    n = length(node.pool)
    M = n*node.parameters.μ*SP.σᵤ^2/(n*SP.σᵤ^2+SP.σ^2)
    S = sqrt(SP.σ^2*SP.σᵤ^2/(n*SP.σᵤ^2+SP.σ^2))
    node.parameters.v = M + S * randn()
    return node
end