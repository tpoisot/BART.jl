function transformer(y)
    y₋, y₊ = extrema(y)
    _encoder(u) = @. (u - y₋) / (y₊ - y₋) - 1 / 2
    _decoder(u) = @. (u + 1 / 2) * (y₊ - y₋) + y₋
    return _encoder, _decoder
end

function g(x::Vector, node::DecisionNode)
    if BART.isterminal(node)
        return node.parameters.v
    end
    return x[node.feature] .< node.value ? g(x, node.left) : g(x, node.right)
end
g(x::Vector, tree::Tree) = g(x, tree.root)
g(X::Matrix, tree::Tree) = g(X, tree.root)
g(tree::Tree) = g(tree.X, tree.root)
g(X::Matrix, node::DecisionNode) = vec(mapslices(x -> g(x, node), X, dims=2))

function R(y, X, tree::Tree)
    ŷ = BART.g(X, tree)
    return y .- ŷ
end
R(tree::Tree) = R(tree.y, tree.X, tree)
R(node::DecisionNode, tree::Tree) = tree.y[node.pool] .- BART.g(tree.X[node.pool, :], tree)