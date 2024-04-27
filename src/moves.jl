"""
    grow!(node::DecisionNode, tree::Tree)

Grows the tree by splitting a terminal node. Note that the value of the variable
used for the split is drawn from the values present within the instances
that are included in the terminal node currently being split.
"""
function grow!(node::DecisionNode, tree::Tree)
    @assert isterminal(node)
    BART.createrule!(node, tree)
    return node
end

"""
    prune!(node::DecisionNode, tree::Tree)

Prunes the tree by collapsing a node whose children are both terminal.
"""
function prune!(node::DecisionNode, tree::Tree)
    @assert isprunable(node)
    collapse!(node, tree)
    return node
end

"""
    change!(node::DecisionNode, tree::Tree)

Changes a decision node (regardless of the status of the children of this node)
so that the feature *and* the value are re-drawn uniformly. This also updates
the allocation of instances to all the downstream nodes in the tree.
"""
function change!(node::DecisionNode, tree::Tree)
    @assert !isdecision(node)
    BART.createrule!(node, tree)
    return node
end

"""
    swap!(node::DecisionNode, tree::Tree)

Changes a decision node and one of its children by swapping their decision
rules. This also updates the allocation of instances to all the downstream nodes
in the tree.
"""
function swap!(node::DecisionNode, tree::Tree)
    @assert isswappable(node)
    swapwith = rand([node.left, node.right])
    swapwith.value, node.value = node.value, swapwith.value
    swapwith.feature, node.feature = node.feature, swapwith.feature
    update!(node, tree)
    return node
end