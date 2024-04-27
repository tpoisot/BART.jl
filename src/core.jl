function g(x::Vector, node::DecisionNode)
    if BART.isterminal(node)
        return node.decision
    end
    return x[node.feature] .< node.value ? g(x, node.left) : g(x, node.right)
end
g(x::Vector, tree::Tree) = g(x, tree.root)
g(X::Matrix, tree::Tree) = g(X, tree.root)
g(X::Matrix, node::DecisionNode) = vec(mapslices(x -> g(x, node), X, dims=2))