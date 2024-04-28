function propose_prune!(tree::Tree, SP::StateParameters, HP::HyperParameters)
    # We pick a node that can be pruned randomly among all possible candidates
    node = rand(prunables(tree))

    # Probability that we grow the terminal node is one only if the root is terminal
    p_grow = iszero(node.depth) ? 1.0 : 1/2
    
    # Probability of the nodes being non-terminal initially
    p_node_nt = probability_nonterminal(node, HP)
    p_left_nt = probability_nonterminal(node.left, HP)
    p_right_nt = probability_nonterminal(node.right, HP)

    # Likelihood of the node under consideration
    L_node = node_likelihood(node, SP)

    # Number of nodes that can be pruned / that are terminal
    n_prun, n_term = length(prunables(tree)), length(leaves(tree))

    # Update the node
    BART.updateleaf!(node, SP)

    # Prune the node
    oldfeature, oldvalue = node.feature, node.value
    oldleft, oldright = node.left, node.right
    node.feature, node.value = missing, missing
    node.left, node.right = nothing, nothing

    # Get the new likelihood
    L_proposal = node_likelihood(node, SP)

    # Likelihood ratio (using logs!)
    a = (1-p_node_nt)*n_prun*p_grow
    a /= (1-p_left_nt)*(1-p_right_nt)*p_node_nt*(n_term-1)*(1-p_grow)
    a *= exp(L_proposal - L_node)

    # Probability of accepting the move
    Pc = exp(a)
    if rand() > Pc
        node.feature = oldfeature
        node.value = oldvalue
        node.left = oldleft
        node.right = oldright
    end
    return node
end