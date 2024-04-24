abstract type AbstractNode end
abstract type AbstractLeaf <: AbstractNode end

struct Regression{T} <: AbstractLeaf
    value::T
end

@testitem "We can create a regression node" begin
    Regression(0.2)
end

struct Classification{T} <: AbstractLeaf
    probabilities::Vector{T}
end

@testitem "We can create a classification node" begin
    Classification(rand(3))
end

mutable struct Node
    feature::Integer
    value
    left
    right
end

@testitem "We can create a terminal node" begin
    Node(1, 0.3, Regression(0.2), Regression(0.5))
end

@testitem "We can create a con-terminal node" begin
    n1 = Node(1, 0.3, Regression(0.2), Regression(0.2))
    Node(2, 0.5, n1, Regression(1.5))
end

@testitem "We can create a non-terminal node" begin
    n1 = Node(1, 0.3, Regression(0.1), Regression(0.2))
    n2 = Node(2, 0.4, Regression(0.4), Regression(0.2))
    Node(3, 0.5, n1, n2)
end

@testitem "We can make a terminal node con-terminal" begin
    n1 = Node(1, 0.3, Regression(0.1), Regression(0.2))
    n2 = Node(2, 0.4, Regression(0.4), Regression(0.2))
    n1.left = n2
end

@testitem "We can make a con-terminal node terminal" begin
    n1 = Node(1, 0.3, Regression(0.1), Regression(0.2))
    n2 = Node(2, 0.4, Regression(0.4), Regression(0.2))
    n1.left = n2
    n1.left = Regression(0.6)
end

mutable struct Tree
    root
end

@testitem "We can create a tree with a single node" begin
    Tree(Regression(0.2))
end

@testitem "We can split the root of a decision tree" begin
    t = Tree(Regression(0.2))
    t.root = Node(1, 0.3, Regression(0.1), Regression(0.2))
end