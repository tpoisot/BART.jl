function propose_grow!(tree::Tree, SP::StateParameters, HP::HyperParameters)
    node = rand(leaves(tree))
    BART.updateleaf!(node, SP)
    baseline = BART.Pt(node, HP) + BART.logL(node, SP)
    node.feature = rand(axes(tree.X, 2))
    node.value = rand(tree.X[node.pool, node.feature])
    perform_grow!(node, tree, SP)
    proposal = BART.Pt(node, HP) + BART.logL(node, SP)
    Pc = min(exp(proposal - baseline), 1.0)
    if rand() > Pc
        node.feature = missing
        node.value = missing
        node.left = nothing
        node.right = nothing
    end
    return node
end

function perform_grow!(node::DecisionNode, tree::Tree, SP::StateParameters)
    R₀ = BART.R(tree)
    if length(node.pool) <= 2
        return node
    end
    idx_left = node.pool[findall(tree.X[node.pool, node.feature] .< node.value)]
    idx_right = node.pool[findall(tree.X[node.pool, node.feature] .>= node.value)]
    if isempty(idx_left) || isempty(idx_right)
        node.feature = missing
        node.value = missing
        return node
    end
    node.left = DecisionNode(idx_left, missing, missing, nothing, nothing, node.depth + 1, NodeParameters())
    node.right = DecisionNode(idx_right, missing, missing, nothing, nothing, node.depth + 1, NodeParameters())
    node.left.parameters.μ = mean(R₀[node.left.pool])
    node.left.parameters.σ = sqrt(mean((R₀[node.left.pool] .- mean(R₀[node.left.pool])) .^ 2))
    node.right.parameters.μ = mean(R₀[node.right.pool])
    node.right.parameters.σ = sqrt(mean((R₀[node.right.pool] .- mean(R₀[node.right.pool])) .^ 2))
    BART.updateleaf!(node.left, SP)
    BART.updateleaf!(node.right, SP)
    return node
end