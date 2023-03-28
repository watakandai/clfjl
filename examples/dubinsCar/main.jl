module ExampleReachAvoid

using LinearAlgebra
using JuMP
using Gurobi
using Plots; gr()
import YAML
using Suppressor

# using clfjl
include("../../src/clfjl.jl")

const GUROBI_ENV = Gurobi.Env()

# Undert the hood, the following dynamics is used:
#   X = [x, y, θ], U = [ω]
# We are going to assume that the model is
#   X = [x, y], U = [θ]
function main()
    execPath = "/Users/kandai/Documents/projects/research/clf/build/DubinsCar"
    # execPath = "/home/kandai/Documents/projects/research/ControlLyapunovFunctionPlanners/build/DubinsCar"
    configPath = joinpath(@__DIR__, "config.yaml")

    config::Dict{Any, Any} = YAML.load(open(configPath))

    N = 2
    x0 = config["start"][1:N]

    # Under the hood, X = [x, y, θ], U = [ω]
    # In the optimization, we assume X = [x, y], U = [θ]
    lb = [-0.1, 0.9]
    ub = [ 0.1, 1.1]
    initSet = clfjl.HyperRectangle(lb, ub)
    @assert all(lb .<= x0) && all(x0 .<= ub)    # x0 ∈ I

    lb = config["goal"][1:N] .- config["goalThreshold"]
    ub = config["goal"][1:N] .+ config["goalThreshold"]
    termSet = clfjl.HyperRectangle(lb, ub)
    @assert !(all(lb .<= x0) && all(x0 .<= ub)) # x0 ∉ T

    lb = config["lowerBound"][1:N]
    ub = config["upperBound"][1:N]
    workspace = clfjl.HyperRectangle(lb, ub)
    @assert all(lb .<= x0) && all(x0 .<= ub)    # x0 ∈ I

    # Must translate to [ω] -> [θ]
    lb = [config["lowerBound"][N+1]]
    ub = [config["upperBound"][N+1]]
    inputSet = clfjl.HyperRectangle(lb, ub)

    params = clfjl.Parameters(
        optDim=N,
        imgFileDir=joinpath(@__DIR__, "output"),
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

    # TODO: Translate data X and U to θ
    function getDynamicsf(θ)::clfjl.Dynamics
        velocity = 0.1
        A = [1 0;
             0 1]
        b = [velocity*cos(θ)*dt, velocity*sin(θ)*dt]
        # f(x) = A * x + b
        return clfjl.Dynamics(A, b, N)
    end
    filterStateFunc(x, u) = x[1:N]
    filterInputFunc(x, u) = [x[N+1]]

    function setOmplConfigFunc(omplConfig, x0, xT)
        omplConfig_ = deepcopy(omplConfig)
        θ = atan(xT[2] - x0[2], xT[1] - x0[1])
        omplConfig_["start"][1:N] = x0
        omplConfig_["start"][N+1] = θ
        omplConfig_["goal"][N+1] = θ
        return omplConfig_
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
                                     config, inputSet,
                                     getDynamicsf,
                                     filterStateFunc,
                                     filterInputFunc,
                                     setOmplConfigFunc)
    end

    "Synthesize Control Lyapunov functions for the given env"
    clfjl.synthesizeCLF(x0, params, env, solver, sampleOMPLDubinsCar, clfjl.plot2DCLF)
end


# ------------------------------ Main ------------------------------ #
# X = [x, y, v, theta] in OMPL but we only use [x, y, v] for synthesizing CLFs
@suppress_err begin # Plotting gives warnings, so I added the supress command.
    main()
end
# main()
# ------------------------------------------------------------------ #


end # module
