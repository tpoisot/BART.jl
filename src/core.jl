function ℊ(x::Vector, node::DecisionNode)
    if BART.isterminal(node)
        return node.μ
    end
    return x[node.feature] .< node.value ? ℊ(x, node.left) : ℊ(x, node.right)
end
ℊ(x::Vector, tree::Tree) = ℊ(x, tree.root)
ℊ(X::Matrix, tree::Tree) = ℊ(X, tree.root)
ℊ(X::Matrix, node::DecisionNode) = vec(mapslices(x -> ℊ(x, node), X, dims=2))

