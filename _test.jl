using Revise
using BART

y, X = BART.bigfoot()

# Do the initial tree
tree = Tree(y, X)

# We split a node with
while depth(tree) < 6
    split!(rand(leaves(tree)), tree)
end

# Probability that a node is non terminal
α = 0.5
d = depth.(leaves(tree))
α.^d