"""
    BART.friedman(x::Vector{Float64})

Generates one output of the Friedman 5-parameters function with no error term.
"""
function friedman(x::Vector{Float64})
    return 10 * sin(π * x[1] * x[2]) + 20 * (x[3] - 0.5)^2 + 10 * x[4] + 5 * x[5]
end

"""
    BART.friedman(; n=20, p=3)

Generates `n` data points for the Friedman 5-parameters function, with an extra
set of `p` predictors that have no effect on the value of y. The error term is
in 𝒩(0,1).
"""
function friedman(; n=20, p=3)
    ε = rand(Normal(), n)
    X = rand(Uniform(), (n, 5+p))
    y = vec(mapslices(BART.friedman, X, dims=2)) .+ ε
    return y, X
end