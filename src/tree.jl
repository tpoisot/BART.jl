decider(y::Matrix) = vec(Statistics.mean(y, dims=1))
decider(y::Vector) = Statistics.mean(y)
decider(y::Matrix, idx) = decider(y[idx, :])
decider(y::Vector, idx) = decider(y[idx])

mutable struct Tree
    y
    X
    root
end

function Tree(y, X::Matrix)
    p₀ = decider(y)
    return Tree(y, X, DecisionNode(collect(axes(X, 1)), missing, missing, p₀, nothing, nothing, 0))
end
export Tree

"""
    DecisionNode{T}

A decision node that works for both classification and regression.
"""
mutable struct DecisionNode{T}
    pool
    feature
    value
    decision::T
    left
    right
    depth
end
export DecisionNode

depth(node::DecisionNode) = node.depth
depth(tree::Tree) = maximum(depth.(leaves(tree)))
export depth

isterminal(::Nothing) = false
isdecision(node::DecisionNode) = (!isnothing(node.left)) & (!isnothing(node.right))
isterminal(node::DecisionNode) = isnothing(node.left) & isnothing(node.right)
isswappable(node::DecisionNode) = !isterminal(node) & (!isterminal(node.left) | !isterminal(node.right))

leaves(::Nothing) = nothing
leaves(tree::Tree) = leaves(tree.root)
leaves(node::DecisionNode) = isterminal(node) ? [node] : vcat(leaves(node.left), leaves(node.right))
export leaves

nonterminals(::Nothing) = nothing
nonterminals(tree::Tree) = nonterminals(tree.root)
nonterminals(node::DecisionNode) = isterminal(node) ? [nothing] : filter(!isnothing, vcat(node, nonterminals(node.left), nonterminals(node.right)))
export nonterminals

prunables(::Nothing) = nothing
prunables(tree::Tree) = prunables(tree.root)
prunables(node::DecisionNode) = isprunable(node) ? [node] : filter(!isnothing, vcat(prunables(node.left), prunables(node.right)))
export prunables

swappables(::Nothing) = nothing
swappables(tree::Tree) = swappables(tree.root)
swappables(node::DecisionNode) = !isswappable(node) ? [nothing] : filter(!isnothing, vcat(node, swappables(node.left), swappables(node.right)))
export swappables

function createrule!(node::DecisionNode, tree::Tree)
    node.feature = rand(axes(tree.X, 2))
    node.value = rand(tree.X[node.pool, node.feature])
    update!(node, tree)
    return node
end

function update!(node::DecisionNode, tree::Tree)
    if ismissing(node.feature)
        return node
    end
    idx_left = node.pool[findall(tree.X[node.pool, node.feature] .<= node.value)]
    idx_right = node.pool[findall(tree.X[node.pool, node.feature] .> node.value)]
    if isempty(idx_right) || isempty(idx_left)
        collapse!(node, tree)
        return node
    else
        d_left = BART.decider(tree.y, idx_left)
        d_right = BART.decider(tree.y, idx_right)
        if isnothing(node.left)
            node.left = DecisionNode(idx_left, missing, missing, d_left, nothing, nothing, node.depth + 1)
        else
            node.left.pool = idx_left
            update!(node.left, tree)
        end
        if isnothing(node.right)
            node.right = DecisionNode(idx_right, missing, missing, d_right, nothing, nothing, node.depth + 1)
        else
            node.right.pool = idx_right
            update!(node.right, tree)
        end
        return node
    end
end

function collapse!(node::DecisionNode, tree::Tree)
    node.value = missing
    node.feature = missing
    if !isnothing(node.left)
        collapse!(node.left, tree)
        node.left = nothing
    end
    if !isnothing(node.right)
        collapse!(node.right, tree)
        node.right = nothing
    end
    return node
end

function split!(node::DecisionNode, tree::Tree)
    @assert isterminal(node)
    BART.createrule!(node, tree)
    return node
end
export split!

function prune!(node::DecisionNode, tree::Tree)
    @assert isprunable(node)
    collapse!(node, tree)
    return node
end
export prune!

function changerule!(node::DecisionNode, tree::Tree)
    @assert !isdecision(node)
    BART.createrule!(node, tree)
    return node
end
export changerule!

function swaprule!(node::DecisionNode, tree::Tree)
    @assert isswappable(node)
    swapwith = rand([node.left, node.right])
    swapwith.value, node.value = node.value, swapwith.value
    swapwith.feature, node.feature = node.feature, swapwith.feature
    update!(node, tree)
    return node
end
export swaprule!