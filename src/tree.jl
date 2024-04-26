abstract type AbstractNode end
abstract type AbstractLeaf <: AbstractNode end

mutable struct ClassificationNode <: AbstractNode
    feature
    value
    decision
    left
    right
end

export ClassificationNode

isdecision(node::AbstractNode) = (!isnothing(node.left)) & (!isnothing(node.right))
isconterminal(node::AbstractNode) = xor(isnothing(node.left), isnothing(node.right))
isterminal(node::AbstractNode) = !(isdecision(node) | isconterminal(node))

leaves(::Nothing) = nothing
leaves(node::AbstractNode) = isterminal(node) ? [node] : vcat(leaves(node.left), leaves(node.right))

export leaves

function split!(node::ClassificationNode, y, X)
    @assert isterminal(node)
    node.feature = rand(axes(X, 2))
    node.value = rand(X[:,node.feature])
    idx_left = findall(X[:,node.feature] .<= node.value)
    idx_right = findall(X[:,node.feature] .> node.value)
    d_left = vec(Statistics.mean(y[idx_left,:], dims=1))
    d_right = vec(Statistics.mean(y[idx_right,:], dims=1))
    node.left = ClassificationNode(nothing, nothing, d_left, nothing, nothing)
    node.right = ClassificationNode(nothing, nothing, d_right, nothing, nothing)
    return node
end

export split!