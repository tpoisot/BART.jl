"""
    probability_nonterminal(node::DecisionNode, HP::HyperParameters)

The probability that a node is non-terminal is ``α(1+d)^{-β}``, where ``d`` is
the depth of the node - the root has depth of 0.
"""
function probability_nonterminal(node::DecisionNode, HP::HyperParameters)
    return HP.α*(BART.depth(node)+1)^(-HP.β)
end

"""
    prior_for_node(node::DecisionNode, HP::HyperParameters)

This is the *log* of the prior probabilit for a given tree, which accounts for
the probability that the node is non-terminal, as well as the number of
instances currently retained in this node.

See also [`BART.probability_nonterminal`](@ref)
"""
function prior_for_node(node::DecisionNode, HP::HyperParameters)
    ntp = probability_nonterminal(node, HP)
    if BART.isterminal(node)
        return log(1-ntp)
    end
    P = log(ntp/length(node.pool))
    P += prior_for_node(node.left, HP)
    P += prior_for_node(node.right, HP) 
    return P
end

"""
    node_likelihood(node::DecisionNode, SP::StateParameters)

This the *log* of the likelihood of a given node, measured iteratively.

See also [`BART.prior_for_node`](@ref)
"""
function node_likelihood(node::DecisionNode, SP::StateParameters)
    if isempty(node.pool)
        return Inf
    end
    n = length(node.pool)
    if !isterminal(node)
        return node_likelihood(node.left, SP) + node_likelihood(node.right, SP)
    else
        ℒ = 0.5 * (log(SP.σ^2)-log(SP.σ^2+n*SP.σᵤ^2))
        ℒ -= 0.5 * n * node.parameters.σ^2/SP.σ^2
        ℒ -= 0.5 * n * node.parameters.μ^2/(n*SP.σᵤ^2+SP.σ^2)
        return ℒ
    end
end