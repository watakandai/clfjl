function StabilityExample(;x=[1., 1.],
                          termLB::Union{Vector{<:Real}, <:Real}=-0.01,
                          termUB::Union{Vector{<:Real}, <:Real}=0.01,
                          boundLB::Union{Vector{<:Real}, <:Real}=-1.1,
                          boundUB::Union{Vector{<:Real}, <:Real}=1.1)::Nothing
    N = 2
    @assert length(x) == N

    termLBs = isa(termLB, Vector{<:Real}) ? termLB : termLB.*ones(N)
    termUBs = isa(termUB, Vector{<:Real}) ? termUB : termUB.*ones(N)
    boundLBs = isa(boundLB, Vector{<:Real}) ? boundLB : boundLB.*ones(N)
    boundUBs = isa(boundUB, Vector{<:Real}) ? boundUB : boundUB.*ones(N)
    termSet = clfjl.HyperRectangle(termLBs, termUBs)
    bound = clfjl.HyperRectangle(boundLBs, boundUBs)

    scatter([0], [0], markershape=:circle)
    numTraj = 100
    for i in 1:numTraj
        x = rand(N) * 2 .- 1
        status, X, U = simulateStabilityExample(x, termSet, bound)
        Xs = [x[1] for x in X]
        Ys = [x[2] for x in X]
        plot!(Xs, Ys, aspect_ratio=:equal)
        # scatter!(Xs, Ys, aspect_ratio=:equal, markershape=:circle)
    end
    savefig(joinpath(@__DIR__, "StabilityExample.png"))
end
# StabilityExample(x=[-1., 1.], boundLB=-1.1, boundUB=1.1)

