var documenterSearchIndex = {"docs":
[{"location":"trees/#Trees","page":"Trees","title":"Trees","text":"","category":"section"},{"location":"trees/#Overview-of-the-types","page":"Trees","title":"Overview of the types","text":"","category":"section"},{"location":"trees/","page":"Trees","title":"Trees","text":"Tree","category":"page"},{"location":"trees/#BART.Tree","page":"Trees","title":"BART.Tree","text":"Tree\n\nA tree is a simple datastructure that binds a response y, a matrix of features X, and the tree itself stored in root. When the type is constructed without a root, the default behavior is to create a root that is a terminal node of depth 0, with a response set to the average of y.\n\n\n\n\n\n","category":"type"},{"location":"trees/","page":"Trees","title":"Trees","text":"DecisionNode","category":"page"},{"location":"trees/#BART.DecisionNode","page":"Trees","title":"BART.DecisionNode","text":"DecisionNode{T}\n\nA decision node that works for both classification and regression.\n\n\n\n\n\n","category":"type"},{"location":"trees/#Tree-utilities","page":"Trees","title":"Tree utilities","text":"","category":"section"},{"location":"trees/","page":"Trees","title":"Trees","text":"BART.depth","category":"page"},{"location":"trees/#BART.depth","page":"Trees","title":"BART.depth","text":"depth(node::DecisionNode)\n\nReturns the depth of a node, which is stored in the node metadata.\n\n\n\n\n\ndepth(tree::Tree)\n\nReturns the depth of a tree, which is measured as the lower depth of its terminal nodes.\n\n\n\n\n\n","category":"function"},{"location":"#BART.jl","page":"BART.jl","title":"BART.jl","text":"","category":"section"},{"location":"","page":"BART.jl","title":"BART.jl","text":"Documentation for BART.jl","category":"page"},{"location":"datasets/#Datasets","page":"Datasets","title":"Datasets","text":"","category":"section"}]
}