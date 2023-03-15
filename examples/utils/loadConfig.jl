using clfjl


"Read the config and initialize clfjl.HybridSystem f(x)=A_ix+b"
function getHybridSystemFromConfig(config::Dict{Any,Any})::clfjl.HybridSystem
    numMode = length(config["dynamics"])
    numDim = size(config["dynamics"][1]["A"], 1)

    dynamics = Dict()
    for i in 1:numMode
        A = config["dynamics"][i]["A"]
        A = map(e -> Float64(e), reduce(vcat,transpose.(A))) # vec<vec> to matrix
        b = config["dynamics"][i]["b"]
        b = map(e -> Float64(e), b)
        dynamics[i] = clfjl.Dynamics(A, b, numDim)
    end
    return clfjl.HybridSystem(dynamics, numMode, numDim)
end


"Read the config and initialize clfjl.Env"
function getEnvFromConfig(config::Dict{Any,Any})::clfjl.Env
    # TODO: Change config["goalThreshold"] -> config["startThreshold"]
    # initSetLB = [-0.25, -0.25]
    # initSetUB = [0.25, 0.25]
    lb = [-0.75, -0.75]
    ub = [-0.55, -0.55]
    # lb = config["initSetLB"]
    # ub = config["initSetUB"]
    # initSet = [clfjl.HyperRectangle(lb, ub)]
    initSet = clfjl.HyperRectangle(lb, ub)

    lb = config["goal"] .- config["goalThreshold"]
    ub = config["goal"] .+ config["goalThreshold"]
    # termSet = [clfjl.HyperRectangle(lb, ub)]
    termSet = clfjl.HyperRectangle(lb, ub)

    lb = [config["xBound"][1], config["yBound"][1]]
    ub = [config["xBound"][2], config["yBound"][2]]
    workspace = clfjl.HyperRectangle(lb, ub)

    obstacles::Vector{clfjl.HyperRectangle} = []
    if !isnothing(config["obstacles"])
        for o in config["obstacles"]
            lb = [o["x"]-o["l"]/2, o["y"]-o["l"]/2]
            ub = [o["x"]+o["l"]/2, o["y"]+o["l"]/2]
            push!(obstacles, clfjl.HyperRectangle(lb, ub))
        end
    end

    hybridSystem::clfjl.HybridSystem = getHybridSystemFromConfig(config)

    return clfjl.Env(initSet, termSet, workspace, obstacles, hybridSystem)
end

