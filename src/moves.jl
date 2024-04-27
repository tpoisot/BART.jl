"""
    grow!(node::DecisionNode, tree::Tree)

Grows the tree by splitting a terminal node. Note that the value of the variable
used for the split is drawn from the values present within the instances
including in the terminal node being split, in order to maximize the chances of
finding a valid split.
"""
function grow!(node::DecisionNode, tree::Tree)
    @assert isterminal(node)
    BART.createrule!(node, tree)
    return node
end

"""
    BART.grow(node::DecisionNode, tree::Treee)

Creates a copy of the tree, calls `grow!` on it, and returns the modified copy
of the tree.
"""
function grow(node::DecisionNode, tree::Treee)
    ct = deepcopy(tree)
    grow!(node, ct)
    return ct
end

function prune!(node::DecisionNode, tree::Tree)
    @assert isprunable(node)
    collapse!(node, tree)
    return node
end

function change!(node::DecisionNode, tree::Tree)
    @assert !isdecision(node)
    BART.createrule!(node, tree)
    return node
end

function swap!(node::DecisionNode, tree::Tree)
    @assert isswappable(node)
    swapwith = rand([node.left, node.right])
    swapwith.value, node.value = node.value, swapwith.value
    swapwith.feature, node.feature = node.feature, swapwith.feature
    update!(node, tree)
    return node
end