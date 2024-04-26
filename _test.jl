using Revise
using BART

y, X = BART.bigfoot()

# Do the initial tree
tree = Tree(y, X)

# We split a node with
split!(rand(leaves(tree)), tree)
leaves(tree)
