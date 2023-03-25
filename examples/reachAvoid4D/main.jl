module ExampleReachAvoid

using LinearAlgebra
using JuMP
using Gurobi
using Plots; gr()
import YAML
using Suppressor

using clfjl


const GUROBI_ENV = Gurobi.Env()
EXECPATH = "/Users/kandai/Documents/projects/research/clf/build/clfPlanner4D"
# CONFIGPATH = joinpath(@__DIR__, "config.yaml")
CONFIGPATH = joinpath(@__DIR__, "config2.yaml")


"Read the config and initialize clfjl.HybridSystem f(x)=A_ix+b"
function getHybridSystem(params::clfjl.Parameters)::clfjl.HybridSystem

    config = params.config

    numMode = length(config["dynamics"])
    k = collect(keys(config["dynamics"]))[1]
    numDim = size(config["dynamics"][k]["A"], 1)

    dynamics = Dict()
    for (k, v) in config["dynamics"]
        A = v["A"]
        A = map(e -> Float64(e), reduce(vcat,transpose.(A))) # vec<vec> to matrix
        b = v["b"]
        b = map(e -> Float64(e), b)
        dynamics[k] = clfjl.Dynamics(A, b, numDim)
    end
    return clfjl.HybridSystem(dynamics, numMode, numDim)
end


"Read the config and initialize clfjl.Env"
function getEnv(params::clfjl.Parameters)::clfjl.Env
    lbCirc = -1
    ubCirc = 1

    config = params.config
    st = config["startThreshold"]
    # lb = config["start"] - [st, st, 0, 0]
    # ub = config["start"] + [st, st, 0, 0]
    lb = config["start"] - [st, st, 0, 0.4]
    ub = config["start"] + [st, st, 0, 0.4]
    initSet = clfjl.HyperRectangle(lb, ub)

    gt = config["goalThreshold"]
    lb = config["goal"] - [gt, gt, 0, 0]
    ub = config["goal"] + [gt, gt, 0, 0]
    lb[3:4] = [lbCirc, lbCirc]
    ub[3:4] = [ubCirc, ubCirc]
    termSet = clfjl.HyperRectangle(lb, ub)

    lb = config["lowerBound"]
    ub = config["upperBound"]
    workspace = clfjl.HyperRectangle(lb, ub)

    obstacles::Vector{clfjl.HyperRectangle} = []
    if !isnothing(config["obstacles"])
        for o in config["obstacles"]
            # TODO, NOTE: For now, we only consider 2D obstacle objects, especially rectangles
            lb = [o["x"]-o["l"]/2, o["y"]-o["l"]/2, lbCirc, lbCirc]
            ub = [o["x"]+o["l"]/2, o["y"]+o["l"]/2, ubCirc, ubCirc]
            # lb = [o["x"]-o["l"]/2, o["y"]-o["l"]/2]
            # ub = [o["x"]+o["l"]/2, o["y"]+o["l"]/2]
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


function main(optDim, execPath, configPath)
    config::Dict{Any, Any} = YAML.load(open(configPath))

    params = clfjl.Parameters(
        optDim=optDim,
        config=config,
        execPath=execPath,
        pathFilePath=joinpath(pwd(), "path.txt"),
        imgFileDir=joinpath(@__DIR__, "output"),
        startPoint=config["start"],
        maxIteration=100,
        maxLyapunovGapForGenerator=10,
        maxLyapunovGapForVerifier=10,
        thresholdLyapunovGapForGenerator=1e-3,
        thresholdLyapunovGapForVerifier=1e-1,
        print=true,
        padding=true
    )

    env::clfjl.Env = getEnv(params)

    "Setup Gurobi"
    solver() = Model(optimizer_with_attributes(() -> Gurobi.Optimizer(GUROBI_ENV),
                                            "OutputFlag"=>false))

    "Synthesize Control Lyapunov functions for the given env"
    clfjl.synthesizeCLF(params, env, solver, clfjl.sampleTrajectory3D,
                                             clfjl.plot4DCLF)
end


# ------------------------------ Main ------------------------------ #
@suppress_err begin # Plotting gives warnings, so I added the supress command.
    optDim = 4
    main(optDim, EXECPATH, CONFIGPATH)
end
# ------------------------------------------------------------------ #


end # module
