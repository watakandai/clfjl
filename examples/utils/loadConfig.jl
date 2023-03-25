using clfjl


"Read the config and initialize clfjl.HybridSystem f(x)=A_ix+b"
function getHybridSystem(params::clfjl.Parameters)::clfjl.HybridSystem

    config = params.config
    N = params.optDim

    numMode = length(config["dynamics"])
    numDim = size(config["dynamics"][1]["A"], 1)

    @assert numDim >= N

    dynamics = Dict()
    for i in 1:numMode
        A = config["dynamics"][i]["A"]
        A = map(e -> Float64(e), reduce(vcat,transpose.(A))) # vec<vec> to matrix
        A = A[1:N, 1:N]
        b = config["dynamics"][i]["b"]
        b = map(e -> Float64(e), b)
        b = b[1:N]
        dynamics[i] = clfjl.Dynamics(A, b, numDim)
    end
    return clfjl.HybridSystem(dynamics, numMode, numDim)
end


"Read the config and initialize clfjl.Env"
function getEnv(params::clfjl.Parameters)::clfjl.Env

    config = params.config
    N = params.optDim

    @assert length(config["start"]) >= N
    lb = config["start"][1:N] .- config["startThreshold"]
    ub = config["start"][1:N] .+ config["startThreshold"]
    initSet = clfjl.HyperRectangle(lb, ub)

    @assert length(config["goal"]) >= N
    lb = config["goal"][1:N] .- config["goalThreshold"]
    ub = config["goal"][1:N] .+ config["goalThreshold"]
    termSet = clfjl.HyperRectangle(lb, ub)

    @assert length(config["lowerBound"]) >= N
    @assert length(config["upperBound"]) >= N
    lb = config["lowerBound"][1:N]
    ub = config["upperBound"][1:N]
    workspace = clfjl.HyperRectangle(lb, ub)

    obstacles::Vector{clfjl.HyperRectangle} = []
    if !isnothing(config["obstacles"])
        for o in config["obstacles"]
            # TODO, NOTE: For now, we only consider 2D obstacle objects, especially rectangles
            lb = [o["x"]-o["l"]/2, o["y"]-o["l"]/2]
            ub = [o["x"]+o["l"]/2, o["y"]+o["l"]/2]
            push!(obstacles, clfjl.HyperRectangle(lb, ub))
        end
    end

    hybridSystem::clfjl.HybridSystem = getHybridSystem(params)

    return clfjl.Env(numStateDim=config["numStateDim"],
                     numSpaceDim=config["numSpaceDim"],
                     initSet=initSet,
                     termSet=termSet,
                     workspace=workspace,
                     obstacles=obstacles,
                     hybridSystem=hybridSystem)
end

