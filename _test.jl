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
R₀ = y .- (HP.m-1)*mean(y)/HP.m
NP = NodeParameters(mean(R₀), std(R₀), 0.0)
root = DecisionNode(collect(axes(y,1)), missing, missing, nothing, nothing, 0, deepcopy(NP))
tree = Tree(y, X, root)

#BART.updateleaf!(tree.root, tree)
BART.R(tree)
ensemble = [Tree(y, X, deepcopy(root)) for _ in Base.OneTo(HP.m)]
for tree in ensemble
    BART.updateleaf!(tree.root, tree)
end
BART.R(ensemble[3])

# Generate a proposal for growth
nt = deepcopy(tree)
node = rand(leaves(tree))
baseline = Pt(node, HP) + logL(node, SP)
grow!(node, tree)
proposal = Pt(node, HP) + logL(node, SP)
Pc = exp(proposal - baseline)
if rand() > Pc
    tree = nt
end

"""
    Pt

Log of the prior for the tree
"""
function Pt(node::DecisionNode, HP::HyperParameters)
    Pt = log(HP.α*(BART.depth(node)+1)^(-β))
    if BART.isterminal(node)
        return Pt
    else
        Pt -= log(length(node.pool))
        Pt += loglik(node.left)
        Pt += loglik(node.right) 
    end
end
Pt(tree::Tree) = Pt(tree.root)

"""
    logL

Log of the node likelihood
"""
function logL(node::DecisionNode, SP::StateParameters)
    n = length(node.pool)
    if iszero(n)
        return Inf
    end
    ℒ = 0.5 * (log(SP.σ^2)-log(SP.σ^2+n*SP.σᵤ^2))
    ℒ -= 0.5 * n * node.parameters.σ^2/SP.σ^2
    ℒ -= 0.5 * n * node.parameters.μ^2/(n*SP.σᵤ^2+SP.σ^2)
    return ℒ
end



# Let's make some bad splits
while BART.depth(tree) < 4
    grow!(rand(leaves(tree)), tree)
end

for leaf in leaves(tree)
    BART.updateleaf!(leaf, tree)
end

BART.g(X, tree)
BART.R(tree)

[BART.R(leaf, tree) for leaf in leaves(tree)]
