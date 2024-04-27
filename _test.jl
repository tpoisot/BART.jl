using Revise
using BART

y, X = BART.friedman(n=100)

# Do the initial tree
tree = Tree(y, X)

# Let's make some bad splits
while depth(tree) < 4
    split!(rand(leaves(tree)), tree)
end

BART.g(X, tree)