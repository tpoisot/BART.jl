"""
    BART.findterminalnodes(node::DecisionNode)

Returns an array of nodes that are terminal, *i.e.* nodes that return a decision, starting the search from the node under consideration.
"""
function findterminalnodes(node::DecisionNode)
    if isnothing(node.left) & isnothing(node.right)
        return node
    end
    return vcat(findterminalnodes(node.left), findterminalnodes(node.right))
end

findgrowablenodes(x) = findterminalnodes(x)

"""
    BART.findprunablenodes(node::DecisionNode)

Returns an array of nodes that can be pruned, *i.e.* nodes for which the two descendants are terminal nodes.
"""
function findprunablenodes(node::DecisionNode)
    if isterminal(node.left) & isterminal(node.right)
        return node
    end
    return vcat(findprunablenodes(node.left), findprunablenodes(node.right))
end

findterminalnodes(tree::Tree) = findterminalnodes(tree.root)
findprunablenodes(tree::Tree) = findprunablenodes(tree.root)
findgrowablenodes(tree::Tree) = findgrowablenodes(tree.root)