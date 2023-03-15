module ExampleUnreachableRegion

using LinearAlgebra
using JuMP
using Gurobi
using Plots; gr()
import YAML
using Suppressor

using clfjl
using Debugger

include("../utils/loadConfig.jl")

const GUROBI_ENV = Gurobi.Env()
EXECPATH = "/Users/kandai/Documents/projects/research/clf/build/main"
CONFIGPATH = joinpath(@__DIR__, "config.yaml")


function main(execPath, configPath)
    config::Dict{Any, Any} = YAML.load(open(configPath))

    params = clfjl.Parameters(
        config=config,
        execPath=execPath,
        pathFilePath=joinpath(pwd(), "path.txt"),
        imgFileDir=joinpath(@__DIR__, "output"),
        startPoint=config["start"],
        maxXNorm=15,
        maxIteration=100,
        maxLyapunovGapForGenerator=10,
        maxLyapunovGapForVerifier=10,
        thresholdLyapunovGapForGenerator=1e-5,
        thresholdLyapunovGapForVerifier=1e-5,
        print=true,
        padding=true
    )

    env::clfjl.Env = getEnvFromConfig(config)

    "Setup Gurobi"
    solver() = Model(optimizer_with_attributes(() -> Gurobi.Optimizer(GUROBI_ENV),
                                            "OutputFlag"=>false))

    "Synthesize Control Lyapunov functions for the given env"
    # t = @elapsed clfjl.synthesizeCLF(params, env, solver)
    # println("Total Time: ", t)
    regions::Vector{clfjl.LyapunovFunctions} = clfjl.getUnreachableRegions(params, env, solver)

    for lfs in regions
        A = map(lf->round.(lf.a, digits=2), lfs) #vec{vec}
        A = reduce(hcat, A)' # matrix
        b = map(lf->round(lf.b, digits=2), lfs)
        convexObstacleDict = Dict("type" => "Convex", "A" => A, "b" => b)
        push!(config["obstacles"], convexObstacleDict)
    end
    clfjl.callOracle(params.execPath, config)
    clfjl.plot_env(env)
    clfjl.plotUnreachableRegion(regions, params, env)
    filepath = joinpath(params.imgFileDir, "UnreachableRegion.png")
    savefig(filepath)
end


# ------------------------------ Main ------------------------------ #
@suppress_err begin # Plotting gives warnings, so I added the supress command.
    main(EXECPATH, CONFIGPATH)
end
# ------------------------------------------------------------------ #


end # module
