using YAML
using Test
using JuMP
using Gurobi
using clfjl

include("../examples/utils/loadConfig.jl")

OPTDIM = 2
EXECPATH = "/Users/kandai/Documents/projects/research/clf/build/clfPlanner2D"
CONFIGPATH = joinpath(dirname(@__DIR__), "examples/reachAvoid/configWithObstacles.yaml")
const GUROBI_ENV = Gurobi.Env()
solver() = Model(optimizer_with_attributes(() -> Gurobi.Optimizer(GUROBI_ENV),
                                        "OutputFlag"=>false))

config = YAML.load(open(CONFIGPATH))
params = clfjl.Parameters(
    optDim=OPTDIM,
    config=config,
    execPath=EXECPATH,
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
regions = clfjl.getUnreachableRegions(parameters, env, solver)
for lfs in regions
    A = map(lf->round.(lf.a, digits=2), lfs) #vec{vec}
    A = reduce(hcat, A)' # matrix
    b = map(lf->round(lf.b, digits=2), lfs)
    convexObstacleDict = Dict("type" => "Convex", "A" => A, "b" => b)
    push!(parameters.config["obstacles"], convexObstacleDict)
end

@testset "Generate Without any erorr" begin
    numDim = 2
    data = [-0.75 -0.75 1.57 0 0
            -0.7499 -0.625 1.57 0 1
            -0.749801 -0.5 1.57 0 1
            -0.749701 -0.375 1.57 0 1
            -0.749602 -0.25 1.57 0 1
            -0.624602 -0.2501 -0.000796327 -1 1
            -0.499602 -0.250199 -0.000796327 0 1
            -0.374602 -0.250299 -0.000796327 0 1
            -0.249602 -0.250398 -0.000796327 0 1
            -0.124602 -0.250498 -0.000796327 0 1
            -0.124502 -0.125498 1.57 1 1
            -0.124403 -0.000497942 1.57 0 1]
    data = [-0.75 -0.75 1.57 0 0
            -0.7499 -0.625 1.57 0 1
            -0.749801 -0.5 1.57 0 1
            -0.749701 -0.375 1.57 0 1
            -0.624701 -0.3751 -0.000796327 -1 1
            -0.499701 -0.375199 -0.000796327 0 1
            -0.374701 -0.375299 -0.000796327 0 1
            -0.249702 -0.375398 -0.000796327 0 1
            -0.124702 -0.375498 -0.000796327 0 1
            -0.124602 -0.250498 1.57 1 1
            -0.124502 -0.125498 1.57 0 1
            -0.124403 -0.000497942 1.57 0 1]
    counterExamples::Vector{clfjl.CounterExample} = []
    clfjl.dataToCounterExamples(counterExamples, data, env, config, 3)

    @test length(counterExamples) != 0

    (lfs::Vector,
     genLyapunovGap::Real) = clfjl.generateCandidateCLF(counterExamples,
                                                        parameters,
                                                        env,
                                                        solver,
                                                        regions)

    println(genLyapunovGap)
    @test genLyapunovGap >= parameters.thresholdLyapunovGapForGenerator
end
