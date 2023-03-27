module ExampleReachAvoid

using LinearAlgebra
using JuMP
using Gurobi
using Plots; gr()
import YAML
using Suppressor

using clfjl

const GUROBI_ENV = Gurobi.Env()

# X = [x, y, v], U = [α, θ]
function main()
    N = 3
    execPath = "/home/kandai/Documents/projects/research/ControlLyapunovFunctionPlanners//build/clfPlanner3D"
    configPath = joinpath(@__DIR__, "config.yaml")

    config::Dict{Any, Any} = YAML.load(open(configPath))

    # X = [x, y, v], U = [α, θ]
    x0 = config["start"]
    lb = [-0.1, 0.9, 0.0]
    ub = [ 0.1, 1.1, 0.0]
    initSet = clfjl.HyperRectangle(lb, ub)
    @assert all(lb .<= x0) && all(x0 .<= ub)    # x0 ∈ I

    xyThreshold = config["goalThreshold"]
    velThreshold = 0.1
    lb = config["goal"] - [xyThreshold, xyThreshold, velThreshold]
    ub = config["goal"] + [xyThreshold, xyThreshold, velThreshold]
    termSet = clfjl.HyperRectangle(lb, ub)
    @assert !(all(lb .<= x0) && all(x0 .<= ub)) # x0 ∉ T

    lb = config["lowerBound"]
    ub = config["upperBound"]
    workspace = clfjl.HyperRectangle(lb, ub)
    @assert all(lb .<= x0) && all(x0 .<= ub)    # x0 ∈ I

    lb = config["controlLowerBound"]
    ub = config["controlUpperBound"]
    inputSet = clfjl.HyperRectangle(lb, ub)

    params = clfjl.Parameters(
        optDim=N,
        imgFileDir=joinpath(@__DIR__, "output3"),
        lfsFileDir=@__DIR__,
        maxIteration=1000,
        maxLyapunovGapForGenerator=10,
        maxLyapunovGapForVerifier=10,
        thresholdLyapunovGapForGenerator=1e-12,
        thresholdLyapunovGapForVerifier=0,
        print=true,
        padding=true,
        omplConfig=config,
    )

    pathFilePath = joinpath(pwd(), "path.txt")
    dt = config["propagationStepSize"]

    function getDynamicsf(α, θ)::clfjl.Dynamics
        A = [1 0 cos(θ)*dt;
             0 1 sin(θ)*dt;
             0 0 1]
        b = [0, 0, α*dt]
        # f(x) = A * x + b
        return clfjl.Dynamics(A, b, 3)
    end

    env = clfjl.Env(numStateDim=config["numStateDim"],
                    numSpaceDim=config["numSpaceDim"],
                    initSet=initSet,
                    termSet=termSet,
                    workspace=workspace,
                    obstacles=[])

    "Setup Gurobi"
    solver() = Model(optimizer_with_attributes(() -> Gurobi.Optimizer(GUROBI_ENV),
                                            "OutputFlag"=>false))

    function sampleOMPLDubinsCar(counterExamples::Vector{clfjl.CounterExample},
                                 x0_::Vector{<:Real},
                                 env::clfjl.Env)
        return clfjl.sampleOMPLDubin(counterExamples, x0_, env,
                                     N, execPath, pathFilePath,
                                     config, inputSet, getDynamicsf)
    end

    "Synthesize Control Lyapunov functions for the given env"
    clfjl.synthesizeCLF(x0, params, env, solver, sampleOMPLDubinsCar, clfjl.plot3DCLF)
end


# ------------------------------ Main ------------------------------ #
# X = [x, y, v, theta] in OMPL but we only use [x, y, v] for synthesizing CLFs
@suppress_err begin # Plotting gives warnings, so I added the supress command.
    main()
end
# main()
# ------------------------------------------------------------------ #


end # module
