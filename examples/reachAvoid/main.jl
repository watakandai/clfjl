module ExampleReachAvoid

using LinearAlgebra
using JuMP
using Gurobi
using Plots; gr()
import YAML
include("plotFunc.jl")

using clfjl
using Debugger

const GUROBI_ENV = Gurobi.Env()
EXECPATH = "/Users/kandai/Documents/projects/research/clf/build/main"
CONFIGPATH = joinpath(@__DIR__, "config.yaml")
# CONFIGPATH = joinpath(@__DIR__, "configWithObstacles.yaml")


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


function main(execPath, configPath)
    config::Dict{Any, Any} = YAML.load(open(configPath))

    params = clfjl.Parameters(
        config,
        execPath,
        joinpath(pwd(), "path.txt"),
        joinpath(@__DIR__, "output"),
        config["start"],
        15,
        100,
        10,
        10,
        1e-5,
        1e-5,
        true
    )

    env::clfjl.Env = getEnvFromConfig(config)

    "Setup Gurobi"
    solver() = Model(optimizer_with_attributes(() -> Gurobi.Optimizer(GUROBI_ENV),
                                            "OutputFlag"=>false))

    "Synthesize Control Lyapunov functions for the given env"
    # t = @elapsed clfjl.synthesizeCLF(params, env, solver)
    # println("Total Time: ", t)
    clfjl.synthesizeCLF(params, env, solver, plotCLF)
end

# ------------------------------ Main ------------------------------ #
main(EXECPATH, CONFIGPATH)
# ------------------------------------------------------------------ #


end # module
