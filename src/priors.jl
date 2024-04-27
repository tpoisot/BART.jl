"""
    Pt

Log of the prior for the tree
"""
function Pt(node::DecisionNode, HP::HyperParameters)
    P = log(HP.α*(BART.depth(node)+1)^(-HP.β))
    if BART.isterminal(node)
        return 1-P
    else
        P -= log(length(node.pool))
        P += Pt(node.left, HP)
        P += Pt(node.right, HP) 
    end
    return P
end
Pt(tree::Tree, HP::HyperParameters) = Pt(tree.root, HP)

"""
    logL

Log of the node likelihood
"""
function logL(node::DecisionNode, SP::StateParameters)
    n = length(node.pool)
    if iszero(n)
        return Inf
    end
    if !isterminal(node)
        return logL(node.left, SP) + logL(node.right, SP)
    else
        ℒ = 0.5 * (log(SP.σ^2)-log(SP.σ^2+n*SP.σᵤ^2))
        ℒ -= 0.5 * n * node.parameters.σ^2/SP.σ^2
        ℒ -= 0.5 * n * node.parameters.μ^2/(n*SP.σᵤ^2+SP.σ^2)
        return ℒ
    end
end