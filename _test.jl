using Revise
using BART

y, X = BART.friedman()

# Do the initial tree
tree = Tree(y, X)

# Let's make some bad splits
while depth(tree) < 4
    spl = rand(leaves(tree))
    while length(spl.pool) < 5
        spl = rand(leaves(tree))
    end
    split!(spl, tree)
end

function g(x::Vector, node::DecisionNode)
    if BART.isterminal(node)
        return node.decision
    end
    return x[node.feature] .<= node.value ? g(x, node.left) : g(x, node.right)
end
g(x::Vector, tree::Tree) = g(x, tree.root)
g(X::Matrix, node::DecisionNode) = vec(mapslices(x -> g(x, node), X, dims=2))
g(X::Matrix, tree::Tree) = g(X, tree.root)

g(X, tree)