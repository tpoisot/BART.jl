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
    return Tree(y, X, DecisionNode{typeof(p₀)}(collect(axes(X, 1)), missing, missing, p₀, nothing, nothing))
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
end
export DecisionNode

isclassifier(::DecisionNode{T}) where {T} = T <: Vector
isclassifier(tree::Tree) = isclassifier(tree.root)
isregressor(a) = !isclassifier(a)
export isregressor
export isclassifier

isdecision(node::DecisionNode) = (!isnothing(node.left)) & (!isnothing(node.right))
isconterminal(node::DecisionNode) = xor(isnothing(node.left), isnothing(node.right))
isterminal(node::DecisionNode) = !(isdecision(node) | isconterminal(node))

leaves(::Nothing) = nothing
leaves(tree::Tree) = leaves(tree.root)
leaves(node::DecisionNode) = isterminal(node) ? [node] : vcat(leaves(node.left), leaves(node.right))
export leaves

function split!(node::DecisionNode, tree::Tree)
    @assert isterminal(node)
    idx_left, idx_right = _generate_valid_split!(node, tree)
    d_left = BART.decider(tree.y, idx_left)
    d_right = BART.decider(tree.y, idx_right)
    node.left = DecisionNode(idx_left, missing, missing, d_left, nothing, nothing)
    node.right = DecisionNode(idx_right, missing, missing, d_right, nothing, nothing)
    return node
end

function _generate_valid_split!(node, tree)
    invalid = true
    global idx_left, idx_right
    while invalid
        node.feature = rand(axes(tree.X, 2))
        node.value = rand(tree.X[node.pool, node.feature])
        idx_left = node.pool[findall(tree.X[node.pool, node.feature] .<= node.value)]
        idx_right = node.pool[findall(tree.X[node.pool, node.feature] .> node.value)]
        invalid = isempty(idx_left) || isempty(idx_right)
    end
    return idx_left, idx_right
end

export split!