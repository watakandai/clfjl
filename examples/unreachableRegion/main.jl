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
EXECPATH = "/Users/kandai/Documents/projects/research/clf/build/clfPlanner2D"
CONFIGPATH = joinpath(@__DIR__, "config.yaml")


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

    env::clfjl.Env = getEnvg(params)

    "Setup Gurobi"
    solver() = Model(optimizer_with_attributes(() -> Gurobi.Optimizer(GUROBI_ENV),
                                            "OutputFlag"=>false))

    "Synthesize Control Lyapunov functions for the given env"
    # t = @elapsed clfjl.synthesizeCLF(params, env, solver)
    # println("Total Time: ", t)
    regions::Vector{clfjl.LyapunovFunctions} = clfjl.getUnreachableRegions(params, env, solver)
    counterExamples::Vector{clfjl.CounterExample} = []
    clfjl.sampleTrajectory(counterExamples, params.startPoint, params, env)

    for lfs in regions
        A = map(lf->round.(lf.a, digits=2), lfs) #vec{vec}
        A = reduce(hcat, A)' # matrix
        b = map(lf->round(lf.b, digits=2), lfs)
        convexObstacleDict = Dict("type" => "Convex", "A" => A, "b" => b)
        push!(config["obstacles"], convexObstacleDict)
    end
    clfjl.callOracle(params.execPath, config)
    clfjl.plotEnv(env)
    clfjl.plotUnreachableRegion(regions, params, env)
    x = counterExamples[1].x
    clfjl.plotProjectionToConvexSet([-0.5, -0.6], regions[1])
    clfjl.plotProjectionToConvexSet([-0.75, -0.75], regions[1])
    filepath = joinpath(params.imgFileDir, "UnreachableRegion.png")
    savefig(filepath)
end


# ------------------------------ Main ------------------------------ #
@suppress_err begin # Plotting gives warnings, so I added the supress command.
    optDim = 2
    main(optDim, EXECPATH, CONFIGPATH)
end
# ------------------------------------------------------------------ #


end # module
