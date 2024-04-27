"""
    Tree

A tree is a simple datastructure that binds a response `y`, a matrix of features
`X`, and the tree itself stored in `root`. When the type is constructed without
a root, the default behavior is to create a root that is a terminal node of
depth 0, with a response set to the average of `y`.
"""
mutable struct Tree
    y
    X
    root
end

"""
    DecisionNode{T}

A decision node that works for both classification and regression.
"""
mutable struct DecisionNode
    pool
    feature
    value
    left
    right
    depth
    parameters
end

"""
    Tree(y, X::Matrix)

Creates a tree of depth 0 with the response `y` and features `X`, where the root
is a terminal node with a response equal to the average of the response.
"""
function Tree(y, X::Matrix)
    root = DecisionNode(collect(axes(X, 1)), missing, missing, nothing, nothing, 0, NodeParameters())
    tree = Tree(y, X, root)
    updateleaf!(tree.root, tree)
    return tree
end

"""
    depth(node::DecisionNode)

Returns the depth of a node, which is stored in the node metadata.
"""
depth(node::DecisionNode) = node.depth

"""
    depth(tree::Tree)

Returns the depth of a tree, which is measured as the lower depth of its terminal nodes.
"""
depth(tree::Tree) = maximum(depth.(leaves(tree)))

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

"""
    BART.createrule!(node::DecisionNode, tree::Tree)

Creates a new rule for a tree, by assigning a random variable and value for the
decision at this node. Calling this function will automatically update every
node downstream of the new rule.
"""
function createrule!(node::DecisionNode, tree::Tree)
    node.feature = rand(axes(tree.X, 2))
    node.value = rand(tree.X[node.pool, node.feature])
    update!(node, tree)
    return node
end

"""
    BART.update!(node::DecisionNode, tree::Tree)

Propagates changes to a tree by assigning the correct instances to each node.
This method is called when the rules are changed, or when nodes are
created/merged.
"""
function update!(node::DecisionNode, tree::Tree, SP::StateParameters)
    if ismissing(node.feature)
        return node
    end
    idx_left = node.pool[findall(tree.X[node.pool, node.feature] .< node.value)]
    idx_right = node.pool[findall(tree.X[node.pool, node.feature] .>= node.value)]
    if isempty(idx_right) || isempty(idx_left)
        collapse!(node, tree)
        return node
    else
        if isnothing(node.left)
            node.left = DecisionNode(idx_left, missing, missing, nothing, nothing, node.depth + 1, NodeParameters())
            updateleaf!(node.left, SP)
        else
            node.left.pool = idx_left
            update!(node.left, tree, SP)
        end
        if isnothing(node.right)
            node.right = DecisionNode(idx_right, missing, missing, nothing, nothing, node.depth + 1, NodeParameters())
            updateleaf!(node.right, SP)
        else
            node.right.pool = idx_right
            update!(node.right, tree, SP)
        end
        return node
    end
end

"""
    BART.collapse!(node::DecisionNode, tree::Tree)

Collapses a decision tree by removing its decision rule, and destroying its left
and right descendants.
"""
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