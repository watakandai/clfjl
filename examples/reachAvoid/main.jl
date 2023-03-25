module ExampleReachAvoid

using LinearAlgebra
using JuMP
using Gurobi
using Plots; gr()
import YAML
using Suppressor

using clfjl

include("../utils/loadConfig.jl")

const GUROBI_ENV = Gurobi.Env()
EXECPATH = "/Users/kandai/Documents/projects/research/clf/build/clfPlanner2D"
# CONFIGPATH = joinpath(@__DIR__, "config.yaml")
CONFIGPATH = joinpath(@__DIR__, "configWithObstacles.yaml")


function main(optDim, execPath, configPath)
    config::Dict{Any, Any} = YAML.load(open(configPath))

    params = clfjl.Parameters(
        optDim=optDim,
        config=config,
        execPath=execPath,
        pathFilePath=joinpath(pwd(), "path.txt"),
        imgFileDir=joinpath(@__DIR__, "output"),
        startPoint=config["start"],
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
    clfjl.synthesizeCLF(params, env, solver, clfjl.sampleTrajectory2D,
                                             clfjl.plot2DCLF)
end


# ------------------------------ Main ------------------------------ #
@suppress_err begin # Plotting gives warnings, so I added the supress command.
    optDim = 2
    main(optDim, EXECPATH, CONFIGPATH)
end
# ------------------------------------------------------------------ #


end # module
