using Revise
using BART

using Statistics
using Distributions

y, X = BART.friedman(n=100)
mod = BARTModel(y, X)

BART.findprunablenodes(mod.trees[1])

BART.propose_grow!(mod.trees[1], mod.state, mod.hyperparameters)
@info BART.depth(mod.trees[1])

BART.propose_prune!(mod.trees[1], mod.state, mod.hyperparameters)
@info BART.depth(mod.trees[1])