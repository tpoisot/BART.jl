function friedman(x::Vector{Float64})
    return 10 * sin(π * x[1] * x[2]) + 20 * (x[3] - 0.5)^2 + 10 * x[4] + 5 * x[5]
end

function friedman(; n=20, p=3)
    ε = rand(Normal(), n)
    X = rand(Uniform(), (n, 5+p))
    y = vec(mapslices(BART.friedman, X, dims=2)) .+ ε
    return y, X
end