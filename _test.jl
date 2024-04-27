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
BART.updateleaf!(ensemble[1].root, ensemble[1])
for i in 2:HP.m
    ensemble[i].y = BART.R(ensemble[i-1])
    ensemble[i].root.parameters.μ = mean(ensemble[i].y)
    ensemble[i].root.parameters.σ = std(ensemble[i].y)
    BART.updateleaf!(ensemble[i].root, SP)
end

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
