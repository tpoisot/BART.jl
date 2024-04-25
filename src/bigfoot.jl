function bigfoot()
    dpath = joinpath(dirname(pathof(@__MODULE__)), "..", "data")
    Z = vcat(permutedims.(split.(readlines(joinpath(dpath, "bigfoot.csv"))[2:end], ","))...)
    p = parse.(Bool, Z[:, 3])
    y = zeros(Bool, (length(p), 2))
    y[findall(p), 1] .= true
    y[findall(!, p), 2] .= true
    X = parse.(Float64, Z[:, 4:end])
    return y, X
end

@testitem "We can load the bigfoot data" begin
    y, X = BART.bigfoot()
    @test y isa Matrix{Bool}
    @test X isa Matrix{Float64}
end