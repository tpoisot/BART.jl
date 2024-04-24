Base.@kwdef mutable struct BARTMoveProbabilities{T<:AbstractFloat}
    node::T = 0.5
    change::T = 0.3
    swap::T = 0.2
    function BARTMoveProbabilities(node, change, swap)
        p = [node, change, swap]
        if sum(p) != one(eltype(p))
            p ./= sum(p)
        end
        return new{eltype(p)}(p...)
    end
end

@testitem "We can create a vector of move probabilities" begin
    bmp = BARTMoveProbabilities(1/3, 1/3, 1/3)
    @test bmp.node ≈ 1/3
    @test bmp.change ≈ 1/3
    @test bmp.swap ≈ 1/3
end

@testitem "We can create a vector of move probabilities using the default values" begin
    bmp = BARTMoveProbabilities()
    @test bmp.node ≈ 0.5
    @test bmp.change ≈ 0.3
    @test bmp.swap ≈ 0.2
end

@testitem "We can create a vector of move probabilities when the input does not sum to one" begin
    bmp = BARTMoveProbabilities(1.0, 2.0, 3.0)
    @test bmp.node ≈ 1/6
    @test bmp.change ≈ 2/6
    @test bmp.swap ≈ 3/6
end

Base.@kwdef mutable struct BARTHyperParameters
    m::Integer = 200 # Number of trees
    p::Integer = 1000 # Number of draws
    L::Integer = 100 # Length of the burn-in
    i::Integer = 20 # Sample every ith tree in the chain
    α::AbstractFloat = 0.95
    β::AbstractFloat = 2.0
    k::Integer = 2
    P₀::BARTMoveProbabilities = BARTMoveProbabilities() # Prior on the probability of moves
end