module ExampleReachAvoid

using LinearAlgebra
using JuMP
using Gurobi
using Plots; gr()
import YAML
include("plotFunc.jl")

using clfjl
EXECPATH = "/Users/kandai/Documents/projects/research/clf/build/main"
CONFIGPATH = "examples/reachAvoid/config.yaml"


function getHybridSystemFromConfig(config)
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


function getWorkspaceFromConfig(config)
    # lb = [config["xBound"][1], config["yBound"][1], -3.14]
    # ub = [config["xBound"][2], config["yBound"][2], -3.14]
    lb = [config["xBound"][1], config["yBound"][1]]
    ub = [config["xBound"][2], config["yBound"][2]]

    obstacles = []
    if !isnothing(config["obstacles"])
        for o in config["obstacles"]
            lb = [o["x"]-o["l"]/2, o["y"]-o["l"]/2]
            ub = [o["x"]+o["l"]/2, o["y"]+o["l"]/2]
            push!(obstacles, clfjl.Obstacle(lb, ub))
        end
    end
    return clfjl.Workspace(lb, ub, obstacles)
end


config = YAML.load(open(CONFIGPATH))
params = clfjl.Parameters(
    EXECPATH,
    joinpath(pwd(), "path.txt"),
    config["start"],
    15,
    100,
    1,
    1,
    1e-5,
    1e-5,
    true
)

hybridSystem = getHybridSystemFromConfig(config)
workspace = getWorkspaceFromConfig(config)
const GUROBI_ENV = Gurobi.Env()
solver() = Model(optimizer_with_attributes(() -> Gurobi.Optimizer(GUROBI_ENV), "OutputFlag"=>false))
filedir = joinpath(@__DIR__, "output")
# clfjl.synthesizeCLF(config, params, hybridSystem, workspace, solver, filedir, plotCLF, plotSamples)
clfjl.synthesizeCLF(config, params, hybridSystem, workspace, solver, filedir, plotCLF)

end # module
