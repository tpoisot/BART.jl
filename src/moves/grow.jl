function propose_grow!(tree::Tree, SP::StateParameters, HP::HyperParameters)
    # We pick a node at random among the terminal nodes
    node = rand(leaves(tree))
    
    # And we update its output value
    BART.updateleaf!(node, SP)
    
    # Number of terminal nodes
    n_term = length(leaves(tree))

    # Probability that the node is not terminal and log-likelihood
    p_node_nt = probability_nonterminal(node, HP)
    L_node = node_likelihood(node, SP)
    
    # New rule for the node
    node.feature = rand(axes(tree.X, 2))
    node.value = rand(tree.X[node.pool, node.feature])

    # Implementation of the new rule
    perform_grow!(node, tree, SP)

    # Likelihood for the new nodes
    L_left, L_right = node_likelihood(node.left, SP), node_likelihood(node.right, SP)
    p_left_nt, p_right_nt = probability_nonterminal(node.left, HP), probability_nonterminal(node.right, HP)

    # How many nodes that can be pruned now?
    n_prun = length(prunables(tree))

    # L + ratio
    initial_p_growth = iszero(node.depth) ? 1.0 : 0.5
    a = (1/2)*(1-p_left_nt)*(1-p_right_nt)*n_term*p_node_nt
    a /= initial_p_growth*(1-p_node_nt)*n_prun
    a *= exp(L_left + L_right - L_node)
    
    # Probability of accepting the move
    Pc = exp(a)
    if rand() > Pc
        node.feature = missing
        node.value = missing
        node.left = nothing
        node.right = nothing
    end
    return node
end

function perform_grow!(node::DecisionNode, tree::Tree, SP::StateParameters)
    R₀ = BART.R(tree)
    if length(node.pool) <= 2
        return node
    end
    idx_left = node.pool[findall(tree.X[node.pool, node.feature] .< node.value)]
    idx_right = node.pool[findall(tree.X[node.pool, node.feature] .>= node.value)]
    if isempty(idx_left) || isempty(idx_right)
        node.feature = missing
        node.value = missing
        return node
    end
    node.left = DecisionNode(idx_left, missing, missing, nothing, nothing, node.depth + 1, NodeParameters())
    node.right = DecisionNode(idx_right, missing, missing, nothing, nothing, node.depth + 1, NodeParameters())
    node.left.parameters.μ = mean(R₀[node.left.pool])
    node.left.parameters.σ = sqrt(mean((R₀[node.left.pool] .- mean(R₀[node.left.pool])) .^ 2))
    node.right.parameters.μ = mean(R₀[node.right.pool])
    node.right.parameters.σ = sqrt(mean((R₀[node.right.pool] .- mean(R₀[node.right.pool])) .^ 2))
    BART.updateleaf!(node.left, SP)
    BART.updateleaf!(node.right, SP)
    return node
end