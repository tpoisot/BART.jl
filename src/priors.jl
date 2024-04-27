P_node_nonterminal(node::DecisionNode; α=0.95, β=2.0) = α*(1+node.depth)^(-β)
