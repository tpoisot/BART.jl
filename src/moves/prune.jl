function propose_prune!(tree::Tree, SP::StateParameters, HP::HyperParameters)
    node = rand(prunables(tree))
    nprun, nterm = length(prunables(tree)), length(leaves(tree))
    BART.updateleaf!(node, SP)
    baseline = BART.logL(node, SP)
    oldfeature, oldvalue = node.feature, node.value
    oldleft, oldright = node.left, node.right
    node.feature, node.value = missing, missing
    node.left, node.right = nothing, nothing
    proposal = BART.logL(node, SP)
    Pc = (1-Pnonterm(node, HP))*nprun*1/2
    Pc /= (1-Pnonterm(oldleft, HP))*(1-Pnonterm(oldright, HP))*Pnonterm(node, HP)*(nterm-1)*1/2
    Pc *= exp(proposal - baseline)
    if rand() > Pc
        node.feature = oldfeature
        node.value = oldvalue
        node.left = oldleft
        node.right = oldright
    end
    return node
end