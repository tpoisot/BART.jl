using Revise
using BART

using Statistics
using Distributions

y, X = BART.friedman(n=100)
enc, dec = BART.transformer(y)
y = enc(y)

HP = HyperParameters()
SP = StateParameters(y, HP)

# Do the initial tree
tree = Tree(y, X)

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
