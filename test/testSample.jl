using YAML
using Test
using Distributions
using clfjl

include("../examples/utils/loadConfig.jl")

OPTDIM = 2
EXECPATH = "/Users/kandai/Documents/projects/research/clf/build/clfPlanner2D"
CONFIGPATH = joinpath(dirname(@__DIR__), "examples/reachAvoid/configWithObstacles.yaml")

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
regions = clfjl.getUnreachableRegions(params, env, solver)
for lfs in regions
    A = map(lf->round.(lf.a, digits=2), lfs) #vec{vec}
    A = reduce(hcat, A)' # matrix
    b = map(lf->round(lf.b, digits=2), lfs)
    convexObstacleDict = Dict("type" => "Convex", "A" => A, "b" => b)
    push!(params.config["obstacles"], convexObstacleDict)
end

function generateSample(lb, ub, numDim)
    @assert length(lb) == length(ub) == numDim
    return [rand(Uniform(lb[i], ub[i])) for i in 1:numDim]
end

@testset "Avoid Random Sampling to Fail" begin
    numDim = 2
    counterExamples::Vector{clfjl.CounterExample} = []
    samplePoint = vcat(generateSample(env.workspace.lb, env.workspace.ub, numDim), 0)
    clfjl.sampleTrajectory(counterExamples, samplePoint, params, env)
    @test !isnothing(counterExamples)
end

# @testset "loops" begin
#     numDim = 2
#     numSample = 100
#     counterExamples::Vector{clfjl.CounterExample} = []
#     samplePoints = [generateSample(env.workspace.lb, env.workspace.ub, numDim) for _ in 1:numSample]
#     samplePoints = map(s -> vcat(s, 0.0), samplePoints)
#     for samplePoint in samplePoints
#         @test clfjl.sampleTrajectory(counterExamples, samplePoint, params, env)
#     end
# end

# @testset "throws an error" begin
    # @test_throws BoundsError [1, 2, 3][4]
# end

@testset "Add an sample in the obstacle regions as a counter example with isUnsafe=true" begin
    numDim = 2
    counterExamples::Vector{clfjl.CounterExample} = []
    for o in env.obstacles
        samplePoint = vcat(generateSample(o.lb, o.ub, numDim), 0)
        clfjl.sampleTrajectory(counterExamples, samplePoint, params, env)
        @test counterExamples[end].isUnsafe
    end
end
