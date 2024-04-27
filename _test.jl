using Revise
using BART

using Statistics
using Distributions

y, X = BART.friedman(n=100)
enc, dec = BART.transformer(y)
y = enc(y)

HP = HyperParameters(m=10)
SP = StateParameters(y, HP)

# Do the initial tree
R₀ = y .- mean(y)/HP.m # (HP.m-1)*
NP = NodeParameters(mean(R₀), std(R₀), 0.0)
root = DecisionNode(collect(axes(y,1)), missing, missing, nothing, nothing, 0, deepcopy(NP))

ensemble = [Tree(y, X, deepcopy(root)) for _ in Base.OneTo(HP.m)]
BART.updateleaf!(ensemble[1].root, SP)
for i in 2:HP.m
    ensemble[i].y = BART.R(ensemble[i-1])
    ensemble[i].root.parameters.μ = mean(ensemble[i].y)
    ensemble[i].root.parameters.σ = std(ensemble[i].y)
    BART.updateleaf!(ensemble[i].root, SP)
end

tree = Tree(y, X, deepcopy(root))

node = rand(leaves(tree))
R₀ = BART.R(tree)
BART.updateleaf!(node, SP)
baseline = BART.Pt(node, HP) + BART.logL(node, SP)
node.feature = rand(axes(tree.X,2))
node.value = rand(tree.X[node.pool,node.feature])
BART.update!(node, tree, SP)
if !isnothing(node.left)
    node.left.parameters.μ = mean(R₀[node.left.pool])
    node.right.parameters.μ = mean(R₀[node.right.pool])
    node.left.parameters.σ = sqrt(mean((R₀[node.left.pool].-mean(R₀[node.left.pool])).^2))
    node.right.parameters.σ = sqrt(mean((R₀[node.right.pool].-mean(R₀[node.right.pool])).^2))
    BART.updateleaf!(node.left, SP)
    BART.updateleaf!(node.right, SP)
    proposal = BART.Pt(node, HP) + BART.logL(node, SP)
    Pc = min(exp(proposal - baseline), 1.0)
    if rand() > Pc
        node.feature = missing
        node.value = missing
        node.left = nothing
        node.right = nothing
    end
end
@info BART.logL(tree.root, SP)


# Generate a proposal for growth
nt = deepcopy(ensemble[1])
node = rand(leaves(ensemble[1]))
baseline = BART.Pt(node, HP) + BART.logL(node, SP)
grow!(node, ensemble[1])
proposal = BART.Pt(node, HP) + BART.logL(node, SP)
Pc = exp(proposal - baseline)
if rand() > Pc
    ensemble[1] = nt
end
for leaf in leaves(ensemble[1])
    BART.updateleaf!(leaf, ensemble[1])
end
