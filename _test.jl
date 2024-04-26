using Revise
using BART
import Statistics

y, X = BART.bigfoot()

init = vec(Statistics.mean(y, dims=1))

# Do the initial tree
tree = ClassificationNode(nothing, nothing, init, nothing, nothing)
